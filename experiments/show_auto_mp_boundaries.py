"""Pretty-print per-(block, op) boundaries + realized-fraction table from
auto_calibrate_mp's output JSON (the `calib.per_block[*].boundaries_per_op`
field). Also estimates the row-level distribution each boundary implies
assuming the metric is approximately uniform on [0, 1] — useful for
sanity-checking that oracle search found non-degenerate allocations.

Usage:
    python experiments/show_auto_mp_boundaries.py <path/to/auto_mp_*.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fracs_from_boundaries(b: list[float]) -> list[float]:
    """Assume uniform metric in [0, 1]. Top fraction = 1 - b[0]; middle
    fractions = b[k-1] - b[k]; last fraction = b[-1]."""
    if not b:
        return [1.0]
    fracs = [1.0 - b[0]]
    for k in range(len(b) - 1):
        fracs.append(b[k] - b[k + 1])
    fracs.append(b[-1])
    return fracs


def main():
    if len(sys.argv) < 2:
        print("usage: show_auto_mp_boundaries.py <results.json>")
        sys.exit(1)
    path = Path(sys.argv[1])
    j = json.loads(path.read_text())
    per_block = j["calib"]["per_block"]
    cfg_lvls = None
    # infer level list from first entry's boundary count (+1)

    print(f"\n=== {path.name} ===")
    fp = j["results"].get("fp", {}).get("top1", float("nan"))
    sc = j["results"].get("sc_comp", {}).get("top1", float("nan"))
    print(f"  fp.top1={fp:.4f}  sc_comp.top1={sc:.4f}\n")

    # Header (ops vary per run; collect union)
    all_ops = set()
    for e in per_block:
        all_ops.update(e.get("boundaries_per_op", {}).keys())
    ops = sorted(all_ops)

    for op in ops:
        print(f"  --- op = {op} ---")
        print(f"  {'blk':>3s}  {'rmse_before':>11s}  {'rmse_after':>11s}  "
              f"{'r2':>6s}  boundaries -> approx. level fractions (uniform prior)")
        for e in per_block:
            bp = e.get("boundaries_per_op", {}).get(op)
            if bp is None:
                continue
            b = bp["boundaries"]
            fr = fracs_from_boundaries(b)
            bstr = ", ".join(f"{x:.3f}" for x in b)
            fstr = ", ".join(f"{x:.2f}" for x in fr)
            print(f"  {e['block']:3d}  {e['rmse_before']:11.4e}  "
                  f"{e['rmse_after']:11.4e}  {e['r2']:+6.3f}  "
                  f"[{bstr}] -> [{fstr}]")
        print()

    # Aggregate: mean boundaries per op across blocks
    print(f"  --- mean boundary per op (across enabled blocks) ---")
    for op in ops:
        acc = None
        n = 0
        for e in per_block:
            if not e.get("enabled", True):
                continue
            bp = e.get("boundaries_per_op", {}).get(op)
            if bp is None:
                continue
            b = bp["boundaries"]
            if acc is None:
                acc = [0.0] * len(b)
            for i, v in enumerate(b):
                acc[i] += v
            n += 1
        if acc and n:
            mean = [v / n for v in acc]
            fr = fracs_from_boundaries(mean)
            print(f"  {op:<10s} mean b = [{', '.join(f'{x:.3f}' for x in mean)}]"
                  f"  -> [{', '.join(f'{x:.2f}' for x in fr)}]")


if __name__ == "__main__":
    main()
