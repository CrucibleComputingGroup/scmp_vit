"""Sanity-check unipolar SC matmul on fc2-shaped inputs."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sc"))
sys.path.insert(0, str(HERE))

import torch
from sc_triton import sc_matmul_grouped_enable_triton
from config_helpers import make_sobol_simple_config

torch.manual_seed(0)
device = "cuda"

# fc2 shape: in=4096, out=1024. MLP fc2 input = GELU(...) → asymmetric.
N_tok, D_in, D_out = 256, 4096, 1024      # 256 tokens (like a patch sequence)
# Simulate post-GELU activation: mostly positive, small negative shelf.
x_raw = torch.randn(N_tok, D_in, device=device)
x = torch.nn.functional.gelu(x_raw * 2.0)   # heavy-tailed asymmetric

# Weights: zero-mean, symmetric (typical post-init / trained Linear weights).
w = torch.randn(D_out, D_in, device=device) * 0.02

y_fp = x @ w.T     # (N_tok, D_out)

print(f"x min/max/mean: {x.min().item():.3f} {x.max().item():.3f} {x.mean().item():.3f}")
print(f"w min/max/mean: {w.min().item():.3f} {w.max().item():.3f} {w.mean().item():.3f}")
print(f"y_fp min/max/mean/std: {y_fp.min().item():.3f} {y_fp.max().item():.3f} "
      f"{y_fp.mean().item():.3f} {y_fp.std().item():.4f}")

for sc_prec in (8, 10):
    cfg = make_sobol_simple_config(D_in, D_in, sc_prec)
    for mode in ("bipolar", "unipolar"):
        y_sc = sc_matmul_grouped_enable_triton(
            x, w, group_a=1, group_b=1, mode=mode,
            sc_prec=sc_prec, config=cfg,
        )
        err = (y_sc - y_fp)
        rel_l2 = err.pow(2).sum().sqrt() / y_fp.pow(2).sum().sqrt()
        cos = torch.nn.functional.cosine_similarity(
            y_sc.flatten().unsqueeze(0), y_fp.flatten().unsqueeze(0)).item()
        print(f"[prec={sc_prec} mode={mode:8s}] "
              f"y_sc min/max/mean/std: {y_sc.min().item():.3f} {y_sc.max().item():.3f} "
              f"{y_sc.mean().item():.3f} {y_sc.std().item():.4f}  | "
              f"rel_l2={rel_l2.item():.4f} cos={cos:.4f}")
