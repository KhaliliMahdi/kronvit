"""Compare the wall time and GPU memory of small networks:

1. Dense:                ordinary nn.Linear layers, no regularizer.
2. Dense + group lasso:  nn.Linear layers with a group-lasso penalty on block x block blocks of W
                         (the usual way to get block-wise sparsity in a dense model).
3. Kron (plain):         each weight is W = sum_i (S * A_i) kron B_i, with an L1 penalty on S.
3b. Kron (par):          same W, but the sum over r is one matrix multiply (KronLinearParallel, not in the sweep).
4. Kron (vec):          same layer, computed with x @ kron(A, B) = vec(A^T X B) (W is never built).
5. Kron (vec-par):      vec trick with all r terms in two matrix multiplies (no loop over r).
Each one is also run with torch.compile (default mode).
The main sweep goes over rank x block size at one batch size (run_rank_block_sweep);
run_batch_sweep sweeps the batch size at the fixed `rank` and `block` below.

Run:  python kron_simple_benchmark.py
"""
import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_sizes = [64, 256, 1024, 4096, 16384]  # batch sizes for run_batch_sweep
repeats = 1          # run each measurement this many times and average (variance was low, 1 is enough)
ranks = [1, 2, 4]    # ranks for run_rank_block_sweep
blocks = [2, 4, 8]   # block sizes for run_rank_block_sweep
sweep_batch_size = 20000  # batch size for run_rank_block_sweep
in_dim = 1024        # input features
hidden_dim = 1024    # hidden features
num_classes = 10
block = 4            # B_i is block x block, so W is made of block x block blocks
rank = 4             # number of Kronecker terms r
l1_weight = 1e-4     # strength of the L1 penalty on S (Kron networks)
group_lasso_weight = 1e-4  # strength of the group-lasso penalty (dense + group lasso)
steps = 100          # timed training steps
warmup = 10          # untimed steps first (CUDA start-up, torch.compile compiling)


# ----------------------------------------------------------------------------
# The Kronecker layer
# ----------------------------------------------------------------------------
class KronLinear(nn.Module):
    """y = x @ W + bias, with W = sum_i (S * A_i) kron B_i.

    W has shape (in_features, out_features).
    A_i and S have shape (in_features / block, out_features / block).
    B_i has shape (block, block).
    If an entry of S is 0, the matching block x block block of W is all zeros.
    """

    def __init__(self, in_features, out_features, block, rank):
        super().__init__()
        p = in_features // block   # number of block rows
        q = out_features // block  # number of block columns
        self.S = nn.Parameter(torch.ones(p, q))                   # sparsity mask (gets the L1 penalty)
        self.A = nn.Parameter(torch.randn(rank, p, q) / p ** 0.5)  # r "coarse" factors
        self.B = nn.Parameter(torch.randn(rank, block, block) / block ** 0.5)  # r small blocks
        self.bias = nn.Parameter(torch.zeros(out_features))

    def weight(self):
        # Build the full weight: add up r Kronecker products.
        W = 0
        for i in range(self.A.shape[0]):
            W = W + torch.kron(self.S * self.A[i], self.B[i])
        return W

    def forward(self, x):
        return x @ self.weight() + self.bias


class KronLinearParallel(KronLinear):
    """Same layer and parameters, but builds W with all r terms at once (no Python loop).

    kron(A, B)[i*m + k, j*n + l] = A[i, j] * B[k, l], so
    W[i*m + k, j*n + l] = sum_r (S * A_r)[i, j] * B_r[k, l].
    The sum over r is one matrix multiply: (p*q, r) @ (r, m*n).
    """

    def weight(self):
        r, p, q = self.A.shape
        m, n = self.B.shape[1:]
        SA = (self.S * self.A).reshape(r, p * q)      # row r = flattened S * A_r
        Bf = self.B.reshape(r, m * n)                 # row r = flattened B_r
        W4 = (SA.T @ Bf).view(p, q, m, n)             # W4[i, j, k, l] = sum_r SA_r[i, j] * B_r[k, l]
        return W4.permute(0, 2, 1, 3).reshape(p * m, q * n)  # reorder to rows (i, k), columns (j, l)


class KronLinearVec(KronLinear):
    """Same layer and parameters, but never builds W.

    Uses the identity  x @ kron(A, B) = vec(A^T X B), where X is x reshaped
    (row-major) to shape (p, m). This is the row-vector form of
    kron(A, B) vec(X) = vec(B X A^T).
    """

    def forward(self, x):
        N = x.shape[0]
        p, q = self.S.shape        # A_i is p x q
        m, n = self.B.shape[1:]    # B_i is m x n
        X = x.view(N, p, m)        # each input row becomes a p x m matrix
        Y = 0
        for i in range(self.A.shape[0]):
            A = self.S * self.A[i]                 # (p, q), with sparsity mask
            Y = Y + A.T @ (X @ self.B[i])          # (q,p) @ (N,p,n) -> (N, q, n)
        return Y.reshape(N, q * n) + self.bias     # back to a vector of length q*n


class KronLinearVecParallel(KronLinear):
    """Vec trick with all r terms at once: two matrix multiplies, no Python loop.

    Y = sum_r (S*A_r)^T X B_r is computed as
      1. Z = X @ [B_1 | ... | B_r]                  (all r right-multiplications in one matmul)
      2. Y = Z' @ [S*A_1; ... ; S*A_r]              (left-multiplications + the sum over r in one matmul,
                                                     the sum over r and p is the matmul's inner dimension)
    """

    def forward(self, x):
        N = x.shape[0]
        r, p, q = self.A.shape
        m, n = self.B.shape[1:]
        X = x.view(N, p, m)
        B_all = self.B.permute(1, 0, 2).reshape(m, r * n)          # columns: [B_1 | B_2 | ... | B_r]
        Z = X @ B_all                                               # (N, p, r*n): Z[:, :, r-th chunk] = X @ B_r
        Z = Z.view(N, p, r, n).permute(0, 3, 2, 1).reshape(N * n, r * p)  # rows (sample, l), cols (r, i)
        A_stack = (self.S * self.A).reshape(r * p, q)               # rows: S*A_1 stacked on ... on S*A_r
        Y = (Z @ A_stack).view(N, n, q)                             # Y[s, l, j] = sum_r sum_i Z * A
        return Y.transpose(1, 2).reshape(N, q * n) + self.bias      # back to (N, q*n), index j*n + l


# ----------------------------------------------------------------------------
# The two networks: same shape, different hidden layers
# ----------------------------------------------------------------------------
# Each network's forward returns (logits, penalty), so torch.compile also optimizes the penalty.

def group_lasso(weight, block):
    """Sum over all block x block blocks of W of the block's Frobenius norm.

    weight has shape (out, in), as in nn.Linear. A block whose norm reaches 0 is all zeros.
    """
    out_f, in_f = weight.shape
    blocks = weight.view(out_f // block, block, in_f // block, block)  # [block row, i, block col, j]
    norms = (blocks.pow(2).sum(dim=(1, 3)) + 1e-12).sqrt()  # one norm per block (+eps: finite gradient at 0)
    return norms.sum()


class DenseNet(nn.Module):
    def __init__(self, use_group_lasso=False):
        super().__init__()
        self.use_group_lasso = use_group_lasso
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.head(x)
        if self.use_group_lasso:
            # Same layers as the Kron network penalizes; the 10-class head is not penalized.
            penalty = group_lasso_weight * (group_lasso(self.fc1.weight, block) + group_lasso(self.fc2.weight, block))
        else:
            penalty = torch.zeros((), device=x.device)
        return logits, penalty


class KronNet(nn.Module):
    def __init__(self, layer=KronLinear):
        super().__init__()
        self.fc1 = layer(in_dim, hidden_dim, block, rank)
        self.fc2 = layer(hidden_dim, hidden_dim, block, rank)
        self.head = nn.Linear(hidden_dim, num_classes)  # 10 classes do not split into 4x4 blocks: keep dense

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.head(x)
        penalty = l1_weight * (self.fc1.S.abs().sum() + self.fc2.S.abs().sum())  # L1 on the masks S
        return logits, penalty


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
def sync():
    # GPU work is asynchronous: wait for it to finish before reading the clock.
    if device == "cuda":
        torch.cuda.synchronize()


MB = 1024 ** 2


def benchmark(name, model, batch_size):
    """Time `steps` training steps, split into forward / backward / optimizer, and measure GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()  # start each model from a clean GPU memory state

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(batch_size, in_dim, device=device)          # random inputs
    y = torch.randint(0, num_classes, (batch_size,), device=device)  # random labels

    # CUDA events are timestamps recorded on the GPU: e[0] start, e[1] after forward,
    # e[2] after backward, e[3] after optimizer step.
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

    def train_step():
        events[0].record()
        logits, penalty = model(x)
        loss = F.cross_entropy(logits, y) + penalty  # forward includes the loss and the penalty
        events[1].record()
        optimizer.zero_grad()
        loss.backward()
        events[2].record()
        optimizer.step()
        events[3].record()

    # Warm-up: not timed in detail (includes torch.compile compiling).
    start = time.perf_counter()
    for _ in range(warmup):
        train_step()
    sync()
    warmup_time = time.perf_counter() - start

    # Memory between steps: parameters + gradients + Adam state (+ the input batch).
    steady = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()  # from here on, track the highest point

    # Timed steps: per-phase GPU time from the events, and wall time from the CPU clock.
    fwd, bwd, opt = [], [], []
    start = time.perf_counter()
    for _ in range(steps):
        train_step()
        events[3].synchronize()  # wait for this step so its events can be read
        fwd.append(events[0].elapsed_time(events[1]))  # milliseconds
        bwd.append(events[1].elapsed_time(events[2]))
        opt.append(events[2].elapsed_time(events[3]))
    sync()
    wall = (time.perf_counter() - start) / steps * 1000  # ms per step

    peak = torch.cuda.max_memory_allocated()          # highest memory during a step
    reserved = torch.cuda.max_memory_reserved()       # memory PyTorch held from the GPU (closer to nvidia-smi)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_params = sum(p.numel() for p in model.parameters())

    mean = lambda v: sum(v) / len(v)
    return dict(name=name, params=n_params / 1e6, warmup=warmup_time,
                fwd=mean(fwd), bwd=mean(bwd), opt=mean(opt),
                step=mean(fwd) + mean(bwd) + mean(opt), wall=wall,
                param_mb=param_bytes / MB, steady_mb=steady / MB, peak_mb=peak / MB,
                extra_mb=(peak - steady) / MB, reserved_mb=reserved / MB)


def benchmark_repeated(name, make_model, batch_size):
    """Run `benchmark` `repeats` times on one model and return the mean, std and variance of each number."""
    model = make_model()
    runs = [benchmark(name, model, batch_size) for _ in range(repeats)]
    out = dict(name=name)
    for key in ["fwd", "bwd", "opt", "step", "wall", "peak_mb"]:
        values = torch.tensor([r[key] for r in runs])
        out[key] = values.mean().item()
        # sample std / variance (divide by repeats - 1); undefined for a single run, report 0
        out[key + "_std"] = values.std().item() if len(values) > 1 else 0.0
        out[key + "_var"] = values.var().item() if len(values) > 1 else 0.0
    return out


def print_batch_table(batch_size, rows):
    print(f"\n=== batch size {batch_size}  (mean of {repeats} runs x {steps} steps) ===")
    print(f"{'model':<22}{'forward':>9}{'backward':>10}{'optim':>8}{'wall (mean +- std)':>22}{'wall var':>12}{'peak':>9}")
    print(f"{'':<22}{'(ms)':>9}{'(ms)':>10}{'(ms)':>8}{'(ms/step)':>22}{'(ms^2)':>12}{'(MB)':>9}")
    for r in rows:
        wall = f"{r['wall']:.3f} +- {r['wall_std']:.3f}"
        print(f"{r['name']:<22}{r['fwd']:9.3f}{r['bwd']:10.3f}{r['opt']:8.3f}{wall:>22}"
              f"{r['wall_var']:12.2e}{r['peak_mb']:9.1f}")


def print_summary(results):
    # results[batch_size] = list of rows, one per model (same order for every batch size)
    names = [r["name"] for r in results[batch_sizes[0]]]

    print("\n=== wall time per step (ms) vs batch size ===")
    print(f"{'model':<22}" + "".join(f"{b:>18}" for b in batch_sizes))
    for i, name in enumerate(names):
        cells = [f"{results[b][i]['wall']:.2f} +- {results[b][i]['wall_std']:.2f}" for b in batch_sizes]
        print(f"{name:<22}" + "".join(f"{c:>18}" for c in cells))

    print("\n=== variance of wall time per step (ms^2) across repeats vs batch size ===")
    print(f"{'model':<22}" + "".join(f"{b:>12}" for b in batch_sizes))
    for i, name in enumerate(names):
        print(f"{name:<22}" + "".join(f"{results[b][i]['wall_var']:12.2e}" for b in batch_sizes))

    print("\n=== wall time per sample (microseconds) = wall time / batch size ===")
    print(f"{'model':<22}" + "".join(f"{b:>10}" for b in batch_sizes))
    for i, name in enumerate(names):
        print(f"{name:<22}" + "".join(f"{results[b][i]['wall'] / b * 1000:10.3f}" for b in batch_sizes))

    print("\n=== peak GPU memory (MB) vs batch size ===")
    print(f"{'model':<22}" + "".join(f"{b:>10}" for b in batch_sizes))
    for i, name in enumerate(names):
        print(f"{name:<22}" + "".join(f"{results[b][i]['peak_mb']:10.1f}" for b in batch_sizes))


def model_list(names):
    """(name, function that builds a fresh model) for each name. Uses the current global `rank` and `block`.

    dynamic=False: compile for the exact batch size.
    """
    comp = lambda m: torch.compile(m, dynamic=False)
    all_models = {
        "Dense":                lambda: DenseNet().to(device),
        "Dense (compiled)":     lambda: comp(DenseNet().to(device)),
        "Dense+GL":             lambda: DenseNet(use_group_lasso=True).to(device),
        "Dense+GL (compiled)":  lambda: comp(DenseNet(use_group_lasso=True).to(device)),
        "Kron (plain)":         lambda: KronNet().to(device),
        "Kron (compiled)":      lambda: comp(KronNet().to(device)),
        "Kron (vec)":           lambda: KronNet(KronLinearVec).to(device),
        "Kron (vec, compiled)": lambda: comp(KronNet(KronLinearVec).to(device)),
        "Kron (vec-par)":       lambda: KronNet(KronLinearVecParallel).to(device),
        "Kron (vec-par, comp)": lambda: comp(KronNet(KronLinearVecParallel).to(device)),
    }
    return [(n, all_models[n]) for n in names]


DENSE = ["Dense", "Dense (compiled)"]                      # do not depend on rank or block
DENSE_GL = ["Dense+GL", "Dense+GL (compiled)"]             # depend on block only
KRON = ["Kron (plain)", "Kron (compiled)", "Kron (vec)", "Kron (vec, compiled)",
        "Kron (vec-par)", "Kron (vec-par, comp)"]          # depend on rank and block


def check_correctness():
    # Same parameters, same input -> all three Kron layers must give the same output.
    plain = KronLinear(in_dim, hidden_dim, block, rank).to(device)
    with torch.no_grad():
        plain.S.uniform_()             # non-trivial mask so S is actually tested
        x = torch.randn(8, in_dim, device=device)
        ref = plain(x)
        diffs = []
        for cls in [KronLinearVec, KronLinearVecParallel]:
            layer = cls(in_dim, hidden_dim, block, rank).to(device)
            layer.load_state_dict(plain.state_dict())
            diffs.append((layer(x) - ref).abs().max().item())
    return max(diffs)


def run_batch_sweep():
    """Every model at every batch size in `batch_sizes`, at the fixed `rank` and `block`."""
    print(f"device={device}  block={block}x{block}  rank={rank}  batch sizes={batch_sizes}  repeats={repeats}")
    print(f"max |vec trick - plain| = {check_correctness():.2e}")
    models = model_list(DENSE + DENSE_GL + KRON)
    results = {}
    for b in batch_sizes:
        results[b] = [benchmark_repeated(name, make, b) for name, make in models]
        print_batch_table(b, results[b])
    print_summary(results)


def run_rank_block_sweep():
    """Every model for every (rank, block) in `ranks` x `blocks`, at batch size `sweep_batch_size`."""
    global rank, block   # KronNet, DenseNet and group_lasso read these
    b = sweep_batch_size
    print(f"device={device}  batch={b}  ranks={ranks}  blocks={blocks}  repeats={repeats}")

    dense = [benchmark_repeated(n, m, b) for n, m in model_list(DENSE)]   # same for every rank/block
    gl, kron = {}, {}
    for block in blocks:
        gl[block] = [benchmark_repeated(n, m, b) for n, m in model_list(DENSE_GL)]
        for rank in ranks:
            diff = check_correctness()
            kron[rank, block] = [benchmark_repeated(n, m, b) for n, m in model_list(KRON)]
            print(f"done rank={rank} block={block}  (max |vec trick - plain| = {diff:.1e})")

    for key, unit, title in [("wall", "ms/step", "wall time"), ("peak_mb", "MB", "peak GPU memory")]:
        print(f"\n=== {title} ({unit}), batch {b} ===")
        print(f"{'':<22}" + "".join(f"{n:>11}" for n in ["Dense", "Dense comp"]))
        print(f"{'(any rank/block)':<22}" + "".join(f"{r[key]:11.2f}" for r in dense))
        print(f"\n{'block':<22}" + "".join(f"{n:>11}" for n in ["Dense+GL", "GL comp"]))
        for block in blocks:
            print(f"{f'{block}x{block}':<22}" + "".join(f"{r[key]:11.2f}" for r in gl[block]))
        short = ["plain", "comp", "vec", "vec comp", "vecpar", "vecpar comp"]
        print(f"\n{'Kron: rank, block':<22}{'FLOPs/dense':>12}" + "".join(f"{n:>12}" for n in short))
        for block in blocks:
            for rank in ranks:
                ratio = rank / block + rank * block / in_dim   # vec trick multiply-adds / dense, per layer
                print(f"{f'r={rank}, {block}x{block}':<22}{ratio:12.3f}"
                      + "".join(f"{r[key]:12.2f}" for r in kron[rank, block]))


if __name__ == "__main__":
    torch.manual_seed(0)
    torch._dynamo.config.cache_size_limit = 64  # allow torch.compile to keep one version per shape
    run_rank_block_sweep()
