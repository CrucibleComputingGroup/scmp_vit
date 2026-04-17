"""Merge per-shard sensitivity JSONs and build skip_worst_K schedule JSONs.

Usage:
    python merge_and_build_skip_worst.py \
        --shards results/sensitivity/shard_*.json \
        --merged results/sensitivity/sensitivity_all_ops.json \
        --skip_ks 20 30 40 50 \
        --schedule_dir results/sensitivity/schedules/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_shards(shard_paths: list[str]) -> dict:
    """Merge multiple shard JSONs into one. De-duplicates by (op, block)."""
    seen = {}
    config = None
    fp_baseline = None
    for sp in shard_paths:
        with open(sp) as f:
            data = json.load(f)
        if config is None:
            config = data.get("config")
            fp_baseline = data.get("fp_baseline")
        for row in data.get("grid", []):
            key = (row["op"], int(row["block"]))
            seen[key] = row
    grid = sorted(seen.values(), key=lambda r: (r["op"], r["block"]))
    return {"config": config, "fp_baseline": fp_baseline, "grid": grid}


def build_skip_worst_k(merged: dict, k: int) -> dict:
    """Build a per-(op, block) schedule: SC everywhere EXCEPT the top-K worst
    (op, block) pairs by bbox_ap_drop.

    Returns a dict suitable for --sc_ops_per_block_json.
    """
    n_blocks = merged["config"]["n_blocks"]
    op_names = merged["config"]["op_names"]
    ranked = sorted(merged["grid"], key=lambda r: r["bbox_ap_drop"], reverse=True)
    drop_set = {(r["op"], int(r["block"])) for r in ranked[:k]}

    sched = {op: [1] * n_blocks for op in op_names}
    for (op, bi) in drop_set:
        sched[op][bi] = 0

    total_on = sum(sum(v) for v in sched.values())
    total = len(op_names) * n_blocks
    return {
        "skip_worst_k": k,
        "total_sc_ops": total_on,
        "total_ops": total,
        "dropped": sorted(drop_set),
        "schedule": sched,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", nargs="+", required=True,
                   help="Paths to shard JSONs to merge.")
    p.add_argument("--merged",
                   default="results/sensitivity/sensitivity_all_ops.json",
                   help="Output path for merged sensitivity JSON.")
    p.add_argument("--skip_ks", nargs="+", type=int, default=[20, 30, 40, 50],
                   help="Values of K for skip_worst_K schedules.")
    p.add_argument("--schedule_dir",
                   default="results/sensitivity/schedules/",
                   help="Dir for output schedule JSONs.")
    args = p.parse_args()

    # --- Merge ---
    print(f"Merging {len(args.shards)} shards...")
    merged = merge_shards(args.shards)
    n_entries = len(merged["grid"])
    expected = len(merged["config"]["op_names"]) * merged["config"]["n_blocks"]
    print(f"  {n_entries}/{expected} (op, block) entries")
    if n_entries < expected:
        missing = expected - n_entries
        print(f"  WARNING: {missing} entries missing — sweep incomplete?")

    merged_path = Path(args.merged)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"  wrote {merged_path}")

    # --- Top-20 ranking ---
    ranked = sorted(merged["grid"], key=lambda r: r["bbox_ap_drop"], reverse=True)
    print(f"\nTop-20 most sensitive (op, block):")
    print(f"{'rank':>4s}  {'op':>10s}  {'blk':>3s}  {'bbox_drop':>9s}")
    for i, r in enumerate(ranked[:20]):
        print(f"{i+1:>4d}  {r['op']:>10s}  {r['block']:>3d}  "
              f"{r['bbox_ap_drop']:>+9.2f}")

    # --- Build schedules ---
    sched_dir = Path(args.schedule_dir)
    sched_dir.mkdir(parents=True, exist_ok=True)
    for k in args.skip_ks:
        sk = build_skip_worst_k(merged, k)
        fname = sched_dir / f"skip_worst_{k}.json"
        with open(fname, "w") as f:
            json.dump(sk["schedule"], f, indent=2)
        print(f"  skip_worst_{k}: {sk['total_sc_ops']}/{sk['total_ops']} SC ops → {fname}")

    print("\nDone. Run sc_eval.py with:")
    for k in args.skip_ks:
        print(f"  python sc_eval.py --sc_ops_per_block_json "
              f"{sched_dir}/skip_worst_{k}.json --out-tag skip_worst_{k}")


if __name__ == "__main__":
    main()
