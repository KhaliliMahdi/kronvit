# An Efficient Training Algorithm for Models with Block-wise Sparsity

**Ding Zhu, Zhiqun Zuo, Mahdi Khalili**

*Transactions on Machine Learning Research (TMLR), 2025* · [Paper (arXiv)](https://arxiv.org/abs/2503.21928) · [Project page](https://khalilimahdi.github.io/publication/blockwise-sparsity)

This repository trains vision transformers whose weight matrices are **block-wise sparse** without ever training a dense model. Each linear layer is parameterized as a sum of Kronecker products,

W = Σ<sub>i=1..r</sub> (S ⊙ A<sub>i</sub>) ⊗ B<sub>i</sub>,

with an ℓ1 penalty on S. When S is sparse, W is block-wise sparse, and the block size equals the size of B<sub>i</sub>. Only the small factors S, A<sub>i</sub>, B<sub>i</sub> are trained.

![Kronecker parameterization](https://khalilimahdi.github.io/images/publications/blockwise-sparsity-kronecker.png)

The code builds on [DeiT](https://github.com/facebookresearch/deit) (Apache 2.0). The Kronecker layer is `KronLinear` in [`local_models/KronLinear.py`](local_models/KronLinear.py). Linear layers are converted by `kron_decompose_model` in [`local_utils/model_utils.py`](local_utils/model_utils.py).

## Installation

```bash
git clone https://github.com/KhaliliMahdi/kronvit.git
cd kronvit
pip install -r requirements.txt   # torch 1.13.1, torchvision 0.14.1, timm 0.4.12
```

Check the installation. This builds a dense and two Kronecker ViT-tiny models and runs one training step on random data. It runs on CPU in under a minute:

```bash
PYTHONPATH=. python tests/smoke_test.py
```

```
deit_tiny_patch16_224 {'kron_rank': 0, 'block_size': 0}: 5.544M trainable params, fc1=Linear, forward/backward ok (2, 100)
kron_deit_tiny_patch16_224 {'kron_rank': 4, 'block_size': 4}: 1.884M trainable params, fc1=KronLinear a=(4, 48, 192) b=(4, 4, 4), forward/backward ok (2, 100)
kron_deit_tiny_patch16_224 {'kron_rank': 4, 'block_size': 8}: 0.662M trainable params, fc1=KronLinear a=(4, 24, 96) b=(4, 8, 8), forward/backward ok (2, 100)
```

## Training on CIFAR-100

CIFAR-100 downloads automatically to `--data-path`. Training uses `torchrun` and one or more CUDA GPUs.

**Block-wise sparse ViT (ours).** Here `--kron_rank` is the rank r and `--block_size` is the block size, for 4 × 4 blocks:

```bash
torchrun --nproc_per_node=2 main.py --model kron_deit_tiny_patch16_224 \
    --kron_rank 4 --block_size 4 --epochs 300 --batch-size 256 --lr 1e-4 \
    --data-set CIFAR --data-path ./data/cifar100 --output_dir ./output/kron_tiny_r4_b4
```

Use `kron_deit_small_patch16_224` or `kron_deit_base_patch16_224` for larger models. Linear layers whose dimensions are not divisible by the block size stay dense; with 8 × 8 blocks this is the 100-class head.

**Baselines.** These are dense training, group LASSO, and elastic group LASSO on the same backbone:

```bash
torchrun --nproc_per_node=2 main.py --model deit_tiny_patch16_224 --batch-size 128 \
    --data-set CIFAR --data-path ./data/cifar100 --output_dir ./output/dense_tiny            # dense
# add --group_lasso or --elastic_group_lasso for the sparse baselines
```

The shell scripts in [`script/`](script/) are the authors' experiment launchers. They read `DATA_PATH`, `OUTPUT_ROOT`, `GPU_NUM`, `CUDA_VISIBLE_DEVICES`, and optionally `FINETUNE` (a checkpoint to start from):

```bash
DATA_PATH=./data/cifar100 OUTPUT_ROOT=./output GPU_NUM=2 bash script/train_cifar_kron_rank4patch4x4.sh
```

Checkpoints (`checkpoint.pth`, `best_checkpoint.pth`) and `log.txt` are written to `--output_dir`. To evaluate a checkpoint, add `--eval --resume <output_dir>/best_checkpoint.pth`.

## Citation

```bibtex
@article{zhu2025blockwise,
  title   = {An Efficient Training Algorithm for Models with Block-wise Sparsity},
  author  = {Zhu, Ding and Zuo, Zhiqun and Khalili, Mohammad Mahdi},
  journal = {Transactions on Machine Learning Research},
  year    = {2025}
}
```
