#!/usr/bin/env python3
"""Re-emit a hybrid mask at a different INT width, without re-probing.

`build_hybrid_mask.py` takes `--int_bits`, but that flag only lands in the
emitted JSON (argparse -> the `int_bits` field). It does NOT enter the
sensitivity probe, which ranks (op, block) groups by marginal knock-down dLoss
at a stream length. So the SET of masked groups is independent of the INT width
and the 145-forward probe -- 40-55 minutes -- does not need to be repeated.

Verified on the pair that already exists: `mask10_t32.json` (int7) and
`mask10_t32_int6.json` (int6) cite the same `sens_t32.json` and have BYTE-EQUAL
schedules.

Why this matters for the formal sweep: holding INT7 across all budgets makes the
masked groups' resolution relative to the SC groups drift from 2x at eff-64
(SC 64 levels vs INT7's 128) to 0.5x at eff-256 (SC 256 vs 128), so "10% hybrid"
would not mean the same thing at the two ends of the curve. Matching the INT
width to the SC level count holds that ratio at 1.

    python cls/remask_bits.py results/hybrid/mask10_t128.json --int-bits 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mask", type=Path)
    ap.add_argument("--int-bits", type=int, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <stem>_int<bits>.json next to the source")
    args = ap.parse_args()

    src = json.loads(args.mask.read_text())
    old = src.get("int_bits")
    if old == args.int_bits:
        print(f"[remask] {args.mask.name} is already int{old}; nothing to do")
        return

    out = args.out or args.mask.with_name(f"{args.mask.stem}_int{args.int_bits}.json")
    src["int_bits"] = int(args.int_bits)
    src["derived_from"] = args.mask.name
    out.write_text(json.dumps(src, indent=2))

    n_int = sum(1 for blocks in src["schedule"].values() for k in blocks if k == "int")
    print(f"[remask] {args.mask.name} (int{old}) -> {out.name} (int{args.int_bits})")
    print(f"[remask] schedule unchanged: {n_int} groups -> INT, "
          f"sensitivity {src.get('source_sensitivity')}")


if __name__ == "__main__":
    main()
