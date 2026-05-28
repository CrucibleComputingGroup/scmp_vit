"""Sanity probe: compare SCLinear forward output between
generic grouped kernel and the new chunk_d MLP-specialized kernel
on a synthetic fc-shaped layer. Quick check before running full det eval.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_DET = _HERE.parent
_REPO = _DET.parent
for p in (_DET, _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sc_integration.sc_linear import SCLinear


def probe(name: str, M: int, D_in: int, D_out: int, sc_prec: int,
          chunk_d: int, mode: str = "bipolar"):
    torch.manual_seed(0)
    lin = nn.Linear(D_in, D_out, bias=True).cuda().eval()
    x = torch.randn(M, D_in, device="cuda") * 0.5

    # Reference: FP
    y_fp = lin(x)

    # Path A: generic grouped (no chunk_d)
    sc_a = SCLinear(lin, sc_prec=sc_prec, mode=mode, chunk_d=0).cuda().eval()
    with torch.no_grad():
        y_a = sc_a(x)

    # Path B: chunk_d MLP-specialized
    sc_b = SCLinear(lin, sc_prec=sc_prec, mode=mode, chunk_d=chunk_d).cuda().eval()
    with torch.no_grad():
        y_b = sc_b(x)

    def stats(y, ref):
        d = (y - ref).abs()
        rel = d.mean() / ref.abs().mean().clamp(min=1e-6)
        return d.mean().item(), d.max().item(), rel.item()

    am, amax, arel = stats(y_a, y_fp)
    bm, bmax, brel = stats(y_b, y_fp)
    ab_m, ab_max, _ = stats(y_a, y_b)
    print(f"[{name:>10s}] M={M} D={D_in}->{D_out} sc_prec={sc_prec} "
          f"chunk_d={chunk_d}")
    print(f"   generic  vs FP: mae={am:.4e}  max={amax:.4e}  rel={arel:.4f}")
    print(f"   chunked  vs FP: mae={bm:.4e}  max={bmax:.4e}  rel={brel:.4f}")
    print(f"   generic vs chunked: mae={ab_m:.4e}  max={ab_max:.4e}")


if __name__ == "__main__":
    # det fc1 / fc2 shapes at sc_prec=6 (matching the baseline run)
    # M trimmed so the probe finishes quickly.
    probe("fc1", M=512, D_in=1408, D_out=6144, sc_prec=6, chunk_d=64)
    probe("fc2", M=512, D_in=6144, D_out=1408, sc_prec=6, chunk_d=64)
    # Also at sc_prec=8 (compensator side)
    probe("fc1_p8", M=512, D_in=1408, D_out=6144, sc_prec=8, chunk_d=64)
