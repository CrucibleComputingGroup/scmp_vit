#!/usr/bin/env python3
"""MAC-weighted fraction a hybrid mask removes from the SC budget.

`1/(1-f)` is the re-budgeting rule that makes a hybrid cell iso-compute with
its unmasked reference (ISO_REPORT.md §1), and `f` is NOT the group fraction:
attention groups are cheap, so the 14 groups of `mask10_t32` are 7.9% of MACs,
not 10%. Getting this wrong is what made every pre-2026-07-29 hybrid arm look
like a loss when it was a win on the energy axis.

True MACs come from a UNIFORM cell's trace, which issues exactly one kernel
call per (op, block) and therefore is not inflated by the `_padded_level_gather`
attention over-count (ISO_REPORT.md §3).

    python cls/mask_mac_frac.py results/hybrid/mask10_t96.json \
        --ref results/fig1k/uni_96_trace.json --budget 96
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def true_macs(trace: Path) -> dict:
    out: dict = {}
    for g in json.loads(trace.read_text()).get("groups", []):
        key = (g.get("op"), g.get("block"))
        out[key] = out.get(key, 0) + g.get("macs", 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mask", type=Path)
    ap.add_argument("--ref", type=Path, required=True,
                    help="precision trace of a UNIFORM cell (unpadded MACs)")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="if given, also print the 1/(1-f) raised target")
    args = ap.parse_args()

    macs = true_macs(args.ref)
    total = sum(macs.values())
    sched = json.loads(args.mask.read_text())["schedule"]

    masked = 0
    n_int = 0
    per_op: dict = {}
    for op, blocks in sched.items():
        for blk, kind in enumerate(blocks):
            if kind != "int":
                continue
            m = macs.get((op, blk), 0)
            if m == 0:
                print(f"[warn] masked group ({op},{blk}) absent from the reference trace")
            masked += m
            n_int += 1
            per_op[op] = per_op.get(op, 0) + m

    f = masked / total if total else 0.0
    print(f"[frac] mask   : {args.mask.name}")
    print(f"[frac] ref    : {args.ref.name}  total true MACs = {total/1e12:.3f}T")
    print(f"[frac] groups : {n_int} INT")
    for op in sorted(per_op, key=lambda o: -per_op[o]):
        print(f"[frac]   {op:9s} {per_op[op]/total*100:5.2f}% of MACs")
    print(f"[frac] f      = {f:.4f}   (group fraction would be {n_int/sum(len(v) for v in sched.values()):.4f})")
    if args.budget:
        raised = args.budget / (1.0 - f)
        print(f"[frac] target = {args.budget:g} / (1-f) = {raised:.2f} -> {round(raised)}")
    # machine-readable last line for shell consumption
    print(f"F={f:.6f}")


if __name__ == "__main__":
    main()
