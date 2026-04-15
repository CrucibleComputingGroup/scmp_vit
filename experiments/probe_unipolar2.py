"""Isolate the unipolar bias: test 4 regimes (nonneg / signed, on each operand)."""
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
N_tok, D_in, D_out = 64, 1024, 512       # smaller to avoid OOM on 16GB
sc_prec = 8
cfg = make_sobol_simple_config(D_in, D_in, sc_prec)

def run(label, x, w):
    y_fp = x @ w.T
    for mode in ("bipolar", "unipolar"):
        y_sc = sc_matmul_grouped_enable_triton(
            x, w, group_a=1, group_b=1, mode=mode, sc_prec=sc_prec, config=cfg,
        )
        rel_l2 = (y_sc - y_fp).pow(2).sum().sqrt() / y_fp.pow(2).sum().sqrt()
        bias = (y_sc - y_fp).mean().item()
        print(f"[{label:28s} {mode:8s}] "
              f"y_fp μ={y_fp.mean().item():+.3f} σ={y_fp.std().item():.3f}  "
              f"y_sc μ={y_sc.mean().item():+.3f} σ={y_sc.std().item():.3f}  "
              f"bias={bias:+.4f} rel_l2={rel_l2:.4f}")
    print()

x_pos = torch.rand(N_tok, D_in, device=device) * 2.0                  # nonneg [0,2]
x_sym = torch.randn(N_tok, D_in, device=device)                       # zero-mean
w_pos = torch.rand(D_out, D_in, device=device) * 0.1                  # nonneg
w_sym = torch.randn(D_out, D_in, device=device) * 0.02                # zero-mean

run("x:nonneg  w:nonneg",  x_pos, w_pos)
run("x:nonneg  w:signed",  x_pos, w_sym)
run("x:signed  w:nonneg",  x_sym, w_pos)
run("x:signed  w:signed",  x_sym, w_sym)
