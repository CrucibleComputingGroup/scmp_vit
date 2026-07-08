"""Build skip_worst_K schedules from a sensitivity JSON.

Ranks (op, block) pairs by the chosen metric (default: l2; alt: bbox_ap_drop)
descending, marks the top K as FP (0) and everything else SC (1). Missing
pairs (e.g. incomplete sweep) default to SC=1 — verify before using.

Usage::

    python sensitivity/build_skip.py \\
        sensitivity/sensitivity_int6_n100.json \\
        --ks 10 20 30

    # → sensitivity/skip/skip_worst_{10,20,30}_int6.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _tag(cfg: dict) -> str:
    sl = int(cfg["stoc_len"])
    sp = int(cfg["sc_prec"])
    return f"int{sp}" if sl == (1 << sp) else f"len{sl}"


def build_skip_worst_k(data: dict, k: int, rank_by: str = "l2") -> dict:
    n_blocks = data["config"]["n_blocks"]
    op_names = list(data["config"]["op_names"])
    ranked = sorted(data["grid"], key=lambda r: r[rank_by], reverse=True)
    if k > len(ranked):
        print(f"  [warn] k={k} > grid size {len(ranked)}; capping to "
              f"{len(ranked)}")
        k = len(ranked)
    drop_set = {(r["op"], int(r["block"])) for r in ranked[:k]}
    sched = {op: [1] * n_blocks for op in op_names}
    for op, bi in drop_set:
        sched[op][bi] = 0
    total_on = sum(sum(v) for v in sched.values())
    return {
        "schedule": sched,
        "dropped": sorted(drop_set),
        "total_sc_ops": total_on,
        "total_ops": len(op_names) * n_blocks,
        "k": k,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sensitivity_json", type=str,
                   help="Input sensitivity JSON (from sensitivity_sweep.py).")
    p.add_argument("--ks", nargs="+", type=int, default=[10, 20, 30],
                   help="K values for skip_worst_K schedules.")
    p.add_argument("--rank_by", choices=["l2", "bbox_ap_drop"], default="l2",
                   help="Ranking metric (default: l2).")
    p.add_argument("--out_dir", type=str, default="",
                   help="Default: <parent-of-input>/skip/")
    args = p.parse_args()

    in_path = Path(args.sensitivity_json).resolve()
    with open(in_path) as f:
        data = json.load(f)

    n_blocks = data["config"]["n_blocks"]
    op_names = data["config"]["op_names"]
    n_grid = len(data["grid"])
    n_expected = len(op_names) * n_blocks
    tag = _tag(data["config"])

    print(f"[input] {in_path}")
    print(f"        tag={tag}  grid={n_grid}/{n_expected} "
          f"({100*n_grid/n_expected:.0f}%)")
    if n_grid < n_expected:
        done_pairs = {(r["op"], r["block"]) for r in data["grid"]}
        missing_by_op = {}
        for op in op_names:
            miss = [b for b in range(n_blocks) if (op, b) not in done_pairs]
            if miss:
                missing_by_op[op] = miss
        for op, blocks in missing_by_op.items():
            print(f"        missing: {op} blocks {blocks}  "
                  f"(will default to SC=1)")

    # Top-20 ranking
    ranked = sorted(data["grid"], key=lambda r: r[args.rank_by], reverse=True)
    print(f"\nTop-20 most sensitive (op, block) by {args.rank_by}:")
    print(f"{'rank':>4s}  {'op':>10s}  {'blk':>3s}  {'bbox_drop':>9s}  "
          f"{'segm_drop':>9s}  {'l2':>9s}")
    for i, r in enumerate(ranked[:20]):
        print(f"{i+1:>4d}  {r['op']:>10s}  {r['block']:>3d}  "
              f"{r['bbox_ap_drop']:>+9.2f}  {r['segm_ap_drop']:>+9.2f}  "
              f"{r.get('l2', float('nan')):>9.3f}")

    # Build schedules
    out_dir = Path(args.out_dir) if args.out_dir else (in_path.parent / "skip")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[output] {out_dir}/")

    for k in sorted(set(args.ks)):
        sk = build_skip_worst_k(data, k, rank_by=args.rank_by)
        fname = out_dir / f"skip_worst_{k}_{tag}.json"
        with open(fname, "w") as f:
            json.dump(sk["schedule"], f, indent=2)
        per_op = {op: int(sum(sk["schedule"][op])) for op in op_names}
        print(f"  k={k:>2d}  SC {sk['total_sc_ops']:>3d}/{sk['total_ops']}  "
              f"per-op: {per_op}  → {fname.name}")


if __name__ == "__main__":
    main()
