#!/usr/bin/env python3
"""Judge the three orderings from a results directory, and say which are violated.

    python cls/check_orderings.py results/final10k
    python cls/check_orderings.py results/final10k --json results/final10k/verdict.json
    python cls/check_orderings.py results/final10k --violated 48   # exit 0 if violated

Orderings, at each budget T (effective levels = 2*T):
    (1) MP > uniform            mp_T        vs uni_T
    (2) hybrid MP > hybrid uni  hyb_mp_T[2] vs hyb_uni_T
    (3) hybrid MP > MP          hyb_mp_T[2] vs mp_T

WHAT COUNTS AS A VIOLATION, decided in advance so the threshold is not chosen to
suit the answer:

  * a NEGATIVE point estimate at a budget that still has headroom.

Not a violation: a positive-but-insignificant delta (that is a power question,
not a config question), or ANY result at a saturated budget. Saturation is the
important guard -- at eff-192 uniform SC is within 0.4 pt of FP and ordering
(2)'s comparator sits AT FP, so no allocator can win and a config search there
would be fitting noise. Budgets with headroom below --min-headroom are reported
as SATURATED and are never called violations.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BUDGETS = [32, 48, 64, 128]


def top1(d: Path, tag: str):
    p = d / f"{tag}.json"
    return json.loads(p.read_text())["top1"] if p.exists() else None


def paired(d: Path, base: str, cell: str):
    """(delta_pt, p) from cls/paired_stats.py, or (None, None)."""
    if not (d / f"{base}_hits.npy").exists() or not (d / f"{cell}_hits.npy").exists():
        return None, None
    try:
        out = subprocess.run(
            [sys.executable, "cls/paired_stats.py", str(d), "--base", base,
             "--against", cell],
            capture_output=True, text=True, timeout=600).stdout
    except Exception:
        return None, None
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 6 and f[0] == cell:
            try:
                return float(f[2]), float(f[5].rstrip("*"))
            except ValueError:
                return None, None
    return None, None


def hyb_tag(d: Path, t: int) -> str:
    """The hybrid-MP cell that landed iso -- mp_hybrid_iso may have iterated."""
    for cand in (f"hyb_mp_{t}2", f"hyb_mp_{t}"):
        if (d / f"{cand}.json").exists():
            return cand
    return f"hyb_mp_{t}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--violated", type=int, default=None,
                    help="exit 0 if THIS budget violates an ordering, else 1")
    ap.add_argument("--min-headroom", type=float, default=1.5,
                    help="pt of FP-minus-uniform below which a budget is called "
                         "saturated and is never judged violated")
    args = ap.parse_args()
    d = args.dir

    fp = top1(d, "fp")
    verdict = {"fp": fp, "budgets": {}}

    for t in BUDGETS:
        uni, mp = top1(d, f"uni_{t}"), top1(d, f"mp_{t}")
        if uni is None:
            continue
        h = hyb_tag(d, t)
        head = 100 * (fp - uni) if fp is not None else None
        sat = head is not None and head < args.min_headroom
        row = {"eff": 2 * t, "uniform": uni, "mp": mp,
               "hyb_uni": top1(d, f"hyb_uni_{t}"), "hyb_mp": top1(d, h),
               "headroom_pt": None if head is None else round(head, 2),
               "saturated": sat, "orderings": {}, "violated": []}

        for key, base, cell in (("1_mp_gt_uni", f"uni_{t}", f"mp_{t}"),
                                ("2_hybmp_gt_hybuni", f"hyb_uni_{t}", h),
                                ("3_hybmp_gt_mp", f"mp_{t}", h)):
            delta, p = paired(d, base, cell)
            row["orderings"][key] = {"delta_pt": delta, "p": p,
                                     "base": base, "cell": cell}
            if delta is not None and delta < 0 and not sat:
                row["violated"].append(key)
        verdict["budgets"][str(t)] = row

    # ---- human-readable
    print(f"{'budget':>8} {'headroom':>9}  {'(1) MP>uni':>14} {'(2) hM>hU':>14} {'(3) hM>MP':>14}   state")
    for t, row in verdict["budgets"].items():
        def cell(k):
            o = row["orderings"][k]
            if o["delta_pt"] is None:
                return f"{'--':>14}"
            star = "*" if (o["p"] is not None and o["p"] < 0.05) else " "
            return f"{o['delta_pt']:+9.2f}{star}{'':>4}"
        state = "SATURATED" if row["saturated"] else (
            "VIOLATED:" + ",".join(row["violated"]) if row["violated"] else "ok")
        hr = "--" if row["headroom_pt"] is None else f"{row['headroom_pt']:+.2f}"
        print(f"eff-{row['eff']:<4d} {hr:>9}  {cell('1_mp_gt_uni')}{cell('2_hybmp_gt_hybuni')}{cell('3_hybmp_gt_mp')}   {state}")

    if args.json:
        args.json.write_text(json.dumps(verdict, indent=2))
        print(f"\n[verdict] wrote {args.json}")

    if args.violated is not None:
        row = verdict["budgets"].get(str(args.violated))
        sys.exit(0 if (row and row["violated"]) else 1)


if __name__ == "__main__":
    main()
