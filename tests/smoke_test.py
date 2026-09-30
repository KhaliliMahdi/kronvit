"""Build dense and Kronecker ViT-tiny models and run one forward/backward pass on random data (CPU is fine).

Run from the repository root:  PYTHONPATH=. python tests/smoke_test.py
"""
import torch, timm
from timm.models import create_model
import models  # registers the kron_* models with timm
print("torch", torch.__version__, "timm", timm.__version__)
common = dict(pretrained=False, num_classes=100, drop_rate=0.0, drop_path_rate=0.1, drop_block_rate=None,
              img_size=224, shape_bias=0, freeze_A=False, freeze_B=False, structured_sparse=True)
def count(m): return sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
for name, kw in [("deit_tiny_patch16_224", dict(kron_rank=0, block_size=0)),
                 ("kron_deit_tiny_patch16_224", dict(kron_rank=4, block_size=4)),
                 ("kron_deit_tiny_patch16_224", dict(kron_rank=4, block_size=8))]:
    m = create_model(name, **common, **kw)
    blk = m.blocks[0].mlp.fc1
    y = m(torch.randn(2, 3, 224, 224)); y.sum().backward()
    print(f"{name} {kw}: {count(m):.3f}M trainable params, fc1={type(blk).__name__}"
          + (f" a={tuple(blk.a.shape)} b={tuple(blk.b.shape)}" if hasattr(blk, 'a') else "")
          + f", forward/backward ok {tuple(y.shape)}")
