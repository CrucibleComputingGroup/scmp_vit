"""Probe SC matmul output for cross-arch reproducibility.

Run on A100 and gl1807 (or any two machines), compare outputs:

    python probe_sc_matmul.py

Reports md5 / sum / amax / max-abs-diff-vs-fp for each shape.
Determines whether SC kernel produces bit-identical / numerically-close
output across architectures for the shapes used in det @1280 vs @1024.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent  # vit_sc/
for p in (_REPO, _REPO / "sc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scmp_kernels.sc import sc_matmul  # noqa: E402  unified dispatcher (enable path)
from scmp_kernels.sc.config_helpers import make_sobol_simple_config  # noqa: E402


def probe(M: int, label: str, D: int = 88, sc_prec: int = 7,
          BH: int = 16, stoc_len: int = 128, mode: str = "bipolar",
          seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    A = torch.randn(BH, M, D, device="cuda")
    B = torch.randn(BH, M, D, device="cuda")

    cfg = make_sobol_simple_config(D, D, sc_prec)
    out = sc_matmul(
        A, B, granularity="per_row", group_a=1, group_b=1, mode=mode,
        sc_prec=sc_prec, config=cfg, stoc_len=stoc_len,
    )
    out_cpu = out.detach().cpu().contiguous()

    fp = torch.einsum("bmd,bnd->bmn", A.float(), B.float()).cpu()
    diff = (out_cpu - fp).abs()

    h = hashlib.md5(out_cpu.numpy().tobytes()).hexdigest()[:16]
    s = float(out_cpu.sum())
    a = float(out_cpu.abs().max())
    fp_amax = float(fp.abs().max())
    d_max = float(diff.max())
    d_mean = float(diff.mean())

    return {
        "label": label,
        "shape": tuple(out_cpu.shape),
        "md5": h,
        "sum": s,
        "amax": a,
        "fp_amax": fp_amax,
        "diff_max": d_max,
        "diff_mean": d_mean,
    }


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch={torch.__version__}  triton={__import__('triton').__version__}  "
          f"cuda={torch.version.cuda}")
    print(f"tf32_matmul={torch.backends.cuda.matmul.allow_tf32}  "
          f"tf32_cudnn={torch.backends.cudnn.allow_tf32}")
    print()

    rows = []
    for M, label in [(4096, "1024_shape"), (6400, "1280_shape")]:
        rows.append(probe(M, label))

    # column-aligned report
    print(f"{'label':<14s}  {'shape':<22s}  {'md5':<18s}  "
          f"{'sum':>14s}  {'amax':>10s}  {'diff_max':>10s}  {'diff_mean':>11s}")
    for r in rows:
        print(f"{r['label']:<14s}  {str(r['shape']):<22s}  {r['md5']:<18s}  "
              f"{r['sum']:>14.6e}  {r['amax']:>10.4e}  "
              f"{r['diff_max']:>10.4e}  {r['diff_mean']:>11.4e}")


if __name__ == "__main__":
    main()
