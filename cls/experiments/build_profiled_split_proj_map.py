#!/usr/bin/env python
"""Build a split-proj heuristic sl_map from FP amax profiling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "cls"))

_SEQ = 257
_MACS = {
    "qk": 16 * _SEQ * 64 * _SEQ,
    "av": 16 * _SEQ * _SEQ * 64,
    "qkv_proj": _SEQ * 1024 * 3072,
    "out_proj": _SEQ * 1024 * 1024,
    "mlp_fc1": _SEQ * 1024 * 4096,
    "mlp_fc2": _SEQ * 4096 * 1024,
}
_ALL_EVAL_OPS = ("qk", "av", "qkv_proj", "out_proj", "mlp_fc1", "mlp_fc2")
_SC_OP_NAMES = ("mlp_fc1", "mlp_fc2", "qk", "av", "proj")
_FINE_KS = {"skip_worst50": 50, "skip_worst40": 40, "skip_worst30": 30, "skip_worst20": 20, "all_ops": 0}

_QKV_HIGH = [128] * 11 + [96] * 2 + [64] * 7      # avg 102.4
_QKV_MID = [128] * 10 + [96] * 2 + [64] * 8       # avg 99.2
_QKV_MID_LITE = [128] * 10 + [96] * 1 + [64] * 9  # avg 97.6
_QKV_LOW_STRONG = [128] * 7 + [96] * 5 + [64] * 8 # avg 94.4
_QKV_LOW_WEAK = [128] * 7 + [96] * 4 + [64] * 9   # avg 92.8


def build_skip_worst_k_schedule(k: int, n_blocks: int = 24) -> dict[str, list[int]]:
    sens_path = HERE / "results" / "sensitivity_all_ops.json"
    data = json.loads(sens_path.read_text())
    rows = sorted(data["grid"], key=lambda r: r["l2"], reverse=True)
    drop = {(r["op"], int(r["block"])) for r in rows[:k]}
    spec = {name: [1] * n_blocks for name in _SC_OP_NAMES}
    for op, bi in drop:
        spec[op][bi] = 0
    return spec


def build_active_ops(sc_config: str, n_blocks: int = 24) -> set[tuple[str, int]]:
    if sc_config in _FINE_KS:
        sched = build_skip_worst_k_schedule(_FINE_KS[sc_config], n_blocks)
        return {
            (op, bi)
            for op in _SC_OP_NAMES
            for bi in range(n_blocks)
            if sched[op][bi]
        }
    raise ValueError(f"Unsupported sc_config for this builder: {sc_config!r}")


def _entry_bins(entry) -> list[int]:
    if isinstance(entry, list):
        return [int(x) for x in entry]
    return [int(entry)]


def _entry_avg_sl(entry) -> float:
    bins = _entry_bins(entry)
    return float(sum(bins)) / float(len(bins))


def _entry_levels_fractions(entry) -> tuple[list[int], list[float]]:
    bins = _entry_bins(entry)
    counts = {}
    for sl in bins:
        counts[int(sl)] = counts.get(int(sl), 0) + 1
    total = float(len(bins))
    levels = sorted(counts.keys(), reverse=True)
    fracs = [counts[sl] / total for sl in levels]
    return levels, fracs


def _summarize_per_op(sl_map: dict[str, list], active_ops: set[tuple[str, int]]) -> dict[str, dict[str, float]]:
    by_op = {}
    totals = {}
    for op, bi in active_ops:
        by_op.setdefault(op, {})
        totals[op] = totals.get(op, 0.0) + 1.0
        entry = sl_map[op][bi]
        levels, fracs = _entry_levels_fractions(entry)
        for sl, frac in zip(levels, fracs):
            by_op[op][int(sl)] = by_op[op].get(int(sl), 0.0) + float(frac)
    out = {}
    for op in sorted(by_op):
        out[op] = {str(sl): round(by_op[op][sl] / totals[op], 4) for sl in sorted(by_op[op], reverse=True)}
    return out


def _summarize_allocation(sl_map: dict[str, list], active_ops: set[tuple[str, int]]) -> dict[str, float]:
    level_macs = {}
    total = 0.0
    for op, bi in active_ops:
        levels, fracs = _entry_levels_fractions(sl_map[op][bi])
        for sl, frac in zip(levels, fracs):
            level_macs[int(sl)] = level_macs.get(int(sl), 0.0) + _MACS[op] * float(frac)
            total += _MACS[op] * float(frac)
    if total == 0:
        return {}
    return {str(sl): round(100.0 * macs / total, 1) for sl, macs in sorted(level_macs.items(), reverse=True)}


def _compute_main_sl(sl_map: dict[str, list], active_ops: set[tuple[str, int]]) -> float:
    num = 0.0
    den = 0.0
    for op, bi in active_ops:
        num += _MACS[op] * _entry_avg_sl(sl_map[op][bi])
        den += _MACS[op]
    return num / den if den > 0 else 0.0


def _build_qkv_entry(q50: float, tail95: float, block_idx: int) -> tuple[list[int], str]:
    if tail95 >= 1.80:
        return list(_QKV_HIGH), "high_tail"
    if tail95 >= 1.45:
        if block_idx == 11:
            return list(_QKV_MID_LITE), "mid_tail_lite"
        return list(_QKV_MID), "mid_tail"
    if q50 >= 10.0:
        return list(_QKV_LOW_STRONG), "low_tail_strong_scale"
    return list(_QKV_LOW_WEAK), "low_tail_weak_scale"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile_json", default="results/e2e/amax_row_block_table_fp_500.json")
    ap.add_argument("--base_map_json", default="results/best_skip_worst30_mainsl128_alloc_seed1_i4.json")
    ap.add_argument("--sc_config", default="skip_worst30")
    ap.add_argument("--out_json", default="results/profiled_split_proj_map_main128.json")
    args = ap.parse_args()

    profile = json.loads(Path(args.profile_json).read_text())
    base = json.loads(Path(args.base_map_json).read_text())
    base_sl_map = base["sl_map"] if isinstance(base, dict) and "sl_map" in base else base

    active_merged = build_active_ops(args.sc_config)
    n_blocks = 1 + max(bi for _op, bi in active_merged)

    sl_map = {op: [0] * n_blocks for op in _ALL_EVAL_OPS}
    for op in ("qk", "av", "mlp_fc1", "mlp_fc2"):
        if op in base_sl_map:
            sl_map[op] = list(base_sl_map[op])

    proj_active = sorted(bi for op, bi in active_merged if op == "proj")
    qkv_meta = {}
    for bi in proj_active:
        qkv_stats = profile["ops"]["qkv_proj"][str(bi)]
        q50 = float(qkv_stats["q50"])
        tail95 = float(qkv_stats["q95"]) / max(q50, 1e-12)
        entry, bucket = _build_qkv_entry(q50, tail95, bi)
        sl_map["qkv_proj"][bi] = entry
        sl_map["out_proj"][bi] = 128
        qkv_meta[str(bi)] = {
            "q50": q50,
            "q95": float(qkv_stats["q95"]),
            "tail95": tail95,
            "bucket": bucket,
            "avg_sl": _entry_avg_sl(entry),
        }

    active_expanded = set()
    for op, bi in active_merged:
        if op == "proj":
            active_expanded.add(("qkv_proj", bi))
            active_expanded.add(("out_proj", bi))
        else:
            active_expanded.add((op, bi))

    out = {
        "meta": {
            "source_profile_json": args.profile_json,
            "source_base_map_json": args.base_map_json,
            "sc_config": args.sc_config,
            "design": "split proj into qkv_proj/out_proj; keep qk/av/fc1/fc2 from base map; set out_proj=128 uniform on active blocks; assign qkv_proj fractions by FP q95/q50 tail bucket.",
        },
        "main_sl": _compute_main_sl(sl_map, active_expanded),
        "allocation": _summarize_allocation(sl_map, active_expanded),
        "per_op_allocation": _summarize_per_op(sl_map, active_expanded),
        "qkv_proj_profile_buckets": qkv_meta,
        "sl_map": sl_map,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {out_path}")
    print(f"[main_sl] {out['main_sl']:.4f}")
    print(f"[allocation] {out['allocation']}")
    print(f"[per_op] {out['per_op_allocation']}")


if __name__ == "__main__":
    main()
