"""Microbench: where does time go in chunked vs generic at p6/p8?

Measures full-call latency for:
  - sc_matmul_grouped_enable_triton          (generic)
  - sc_matmul_enable_triton_mlp + chunk_d    (chunked)

at the actual det fc1 (M=6400, D=1408, N=6144) and fc2 (M=6400, D=6144, N=1408)
shapes used by ViTDet @ 1280, for sc_prec ∈ {6, 8} and chunk_d ∈ {0, 64, 256}.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_DET = _HERE.parent
_REPO = _DET.parent
for p in (_DET, _REPO, _REPO / "sc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scmp_kernels.sc import sc_matmul, det_kernel_tuning
from scmp_kernels.sc.config_helpers import make_sobol_simple_config


def bench(fn, name: str, n_warmup=2, n_iter=5):
    # Warmup (autotune + cache fill).
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) / n_iter * 1000
    print(f"  {name:>40s}: {dt_ms:7.2f} ms")
    return dt_ms


def run_layer(label: str, M: int, D: int, N: int, sc_prec: int):
    print(f"\n=== {label}  M={M} D={D} N={N}  sc_prec={sc_prec} ===")
    a = torch.randn(M, D, device="cuda") * 0.5
    b = torch.randn(N, D, device="cuda") * 0.05  # weight-like

    cfg_full = make_sobol_simple_config(D, D, sc_prec)

    def gen():
        return sc_matmul(
            a, b, granularity="per_row", group_a=1, group_b=1,
            mode="bipolar", sc_prec=sc_prec, config=cfg_full)

    bench(gen, "generic (chunk_d=0)")

    for cd in (64, 128, 256):
        cfg_chunk = make_sobol_simple_config(cd, cd, sc_prec)

        def chunked(cd=cd, cfg=cfg_chunk):
            return sc_matmul(
                a, b, granularity="per_row",
                mode="bipolar", sc_prec=sc_prec, config=cfg,
                group_a=1, group_b=1, chunk_d=cd)

        bench(chunked, f"chunked chunk_d={cd}")


if __name__ == "__main__":
    # det shapes at 1280 resolution: M=6400 tokens, D_in/D_out as below.
    # Wrap in det_kernel_tuning so both generic and chunked use the
    # SCLinear-tuned tile heuristic (matches what sc_eval.py does at runtime).
    with det_kernel_tuning():
        for sc_prec in (6, 8):
            run_layer("fc1", M=6400, D=1408, N=6144, sc_prec=sc_prec)
            run_layer("fc2", M=6400, D=6144, N=1408, sc_prec=sc_prec)
