#!/usr/bin/env python
"""Evaluate a custom stoc_len map with FP QwT compensation."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parents[2]   # vit_sc/
CLS = HERE / "cls"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CLS))

QWT_SC_LIB = HERE / "third_party" / "QwT-SC" / "QwT-vit-sc"
if (QWT_SC_LIB / "qwt_sc").exists():
    sys.path.insert(0, str(QWT_SC_LIB))

from eval import load_model, seed_all
from qwt_sc import calibrate_qwt
from qwt_sc_compensation import build_dataset, evaluate, get_blocks
from sc_attention_patch import SC_OP_NAMES, SCLinear, make_sc_attention_forward, set_noise_model
from sc_integration.mp_linear import MPConfig
from mp_budget_swap_search import (
    build_active_ops,
)
# Reuse the comp factories from qwt_sc_overnight so the QwT compensation block
# matches the uniform-path behavior bit-for-bit (SC-kernel comp with the same
# Sobol seed bank, scales, and optional head-aligned per-head routing).
from qwt_sc_overnight import (
    make_sc_factory,
    make_fp_factory,
    make_head_aligned_factory,
    build_seed_bank,
)

_SEQ = 257
_MACS = {
    "qk": 16 * _SEQ * 64 * _SEQ,
    "av": 16 * _SEQ * _SEQ * 64,
    "qkv_proj": _SEQ * 1024 * 3072,
    "out_proj": _SEQ * 1024 * 1024,
    "mlp_fc1": _SEQ * 1024 * 4096,
    "mlp_fc2": _SEQ * 4096 * 1024,
}
_EVAL_OPS = ("qk", "av", "qkv_proj", "out_proj", "mlp_fc1", "mlp_fc2")


def make_uniform_sl_map(active_ops: set[tuple[str, int]],
                        n_blocks: int,
                        uniform_sl: int) -> dict[str, list[int]]:
    sl_map = {op: [0] * n_blocks for op in SC_OP_NAMES}
    for op, bi in active_ops:
        sl_map[op][bi] = int(uniform_sl)
    return sl_map


def load_sl_map(path: str) -> dict[str, list]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "sl_map" in data:
        data = data["sl_map"]
    return dict(data)


def _entry_bins(entry) -> list[int]:
    if isinstance(entry, list):
        return [int(x) for x in entry]
    return [int(entry)]


def _safe_get_entry(sl_map: dict[str, list], op: str, bi: int):
    vals = sl_map.get(op)
    if vals is None or bi >= len(vals):
        return 0
    return vals[bi]


def _get_eval_entry(sl_map: dict[str, list], op: str, bi: int):
    if op in ("qkv_proj", "out_proj"):
        split_entry = _safe_get_entry(sl_map, op, bi)
        if split_entry != 0:
            return split_entry
        return _safe_get_entry(sl_map, "proj", bi)
    return _safe_get_entry(sl_map, op, bi)


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


def _entry_to_mpconfig(entry):
    levels, fracs = _entry_levels_fractions(entry)
    kept = [(int(sl), float(frac)) for sl, frac in zip(levels, fracs) if int(sl) > 0]
    if not kept:
        return None
    k_levels = [sl for sl, _ in kept]
    k_fracs = [frac for _, frac in kept]
    s = sum(k_fracs)
    k_fracs = [x / s for x in k_fracs]
    return MPConfig(k_levels, k_fracs)


def expand_active_ops(active_ops: set[tuple[str, int]], sl_map: dict[str, list]) -> set[tuple[str, int]]:
    has_split_proj = ("qkv_proj" in sl_map) or ("out_proj" in sl_map)
    expanded = set()
    for op, bi in active_ops:
        if op == "proj" and has_split_proj:
            expanded.add(("qkv_proj", bi))
            expanded.add(("out_proj", bi))
        elif op == "proj":
            expanded.add(("qkv_proj", bi))
            expanded.add(("out_proj", bi))
        else:
            expanded.add((op, bi))
    return expanded


def compute_main_sl_custom(sl_map: dict[str, list], active_ops: set[tuple[str, int]]) -> float:
    num = 0.0
    den = 0.0
    for op, bi in active_ops:
        entry = _get_eval_entry(sl_map, op, bi)
        avg = _entry_avg_sl(entry)
        num += _MACS[op] * avg
        den += _MACS[op]
    return num / den if den > 0 else 0.0


def summarize_allocation_custom(sl_map: dict[str, list], active_ops: set[tuple[str, int]]) -> dict[str, float]:
    level_macs = {}
    total = 0.0
    for op, bi in active_ops:
        levels, fracs = _entry_levels_fractions(_get_eval_entry(sl_map, op, bi))
        for sl, frac in zip(levels, fracs):
            level_macs[int(sl)] = level_macs.get(int(sl), 0.0) + _MACS[op] * float(frac)
            total += _MACS[op] * float(frac)
    if total == 0:
        return {}
    return {str(sl): round(100.0 * macs / total, 1) for sl, macs in sorted(level_macs.items(), reverse=True)}


def summarize_per_op_custom(sl_map: dict[str, list], active_ops: set[tuple[str, int]]) -> dict[str, dict[str, float]]:
    by_op = {}
    totals = {}
    for op, bi in active_ops:
        by_op.setdefault(op, {})
        totals[op] = totals.get(op, 0.0) + 1.0
        levels, fracs = _entry_levels_fractions(_get_eval_entry(sl_map, op, bi))
        for sl, frac in zip(levels, fracs):
            by_op[op][int(sl)] = by_op[op].get(int(sl), 0.0) + float(frac)
    out = {}
    for op in sorted(by_op):
        out[op] = {str(sl): round(by_op[op][sl] / totals[op], 4) for sl in sorted(by_op[op], reverse=True)}
    return out


def patch_model_with_custom_sl_map(model, sc_prec: int, sl_map: dict[str, list]):
    blocks = list(get_blocks(model))
    nb = len(blocks)
    patch_stats = {}
    for i, blk in enumerate(blocks):
        attn_mod = None
        for m in blk.modules():
            if hasattr(m, "qkv") and hasattr(m, "proj") and hasattr(m, "num_heads"):
                attn_mod = m
                break
        qk_mp = _entry_to_mpconfig(sl_map.get("qk", [0] * nb)[i] if "qk" in sl_map else 0)
        av_mp = _entry_to_mpconfig(sl_map.get("av", [0] * nb)[i] if "av" in sl_map else 0)
        if attn_mod is not None and (qk_mp is not None or av_mp is not None):
            fwd = make_sc_attention_forward(
                sc_prec=sc_prec,
                sc_av=av_mp is not None,
                sc_qk=qk_mp is not None,
                qk_mp_cfg=qk_mp,
                av_mp_cfg=av_mp,
            )
            attn_mod.forward = fwd.__get__(attn_mod, type(attn_mod))
            patch_stats["attn"] = patch_stats.get("attn", 0) + 1

        qkv_entry = sl_map.get("qkv_proj", sl_map.get("proj", [0] * nb))[i] if ("qkv_proj" in sl_map or "proj" in sl_map) else 0
        out_entry = sl_map.get("out_proj", sl_map.get("proj", [0] * nb))[i] if ("out_proj" in sl_map or "proj" in sl_map) else 0
        qkv_mp = _entry_to_mpconfig(qkv_entry)
        out_mp = _entry_to_mpconfig(out_entry)
        if attn_mod is not None and qkv_mp is not None and isinstance(attn_mod.qkv, torch.nn.Linear):
            attn_mod.qkv = SCLinear(attn_mod.qkv, sc_prec, mode="bipolar", mp_cfg=qkv_mp)
            patch_stats["qkv_proj"] = patch_stats.get("qkv_proj", 0) + 1
        if attn_mod is not None and out_mp is not None and isinstance(attn_mod.proj, torch.nn.Linear):
            attn_mod.proj = SCLinear(attn_mod.proj, sc_prec, mode="bipolar", mp_cfg=out_mp)
            patch_stats["out_proj"] = patch_stats.get("out_proj", 0) + 1

        mlp = getattr(blk, "mlp", None)
        if mlp is not None:
            for fc_name, attr in (("mlp_fc1", "fc1"), ("mlp_fc2", "fc2")):
                entry = sl_map.get(fc_name, [0] * nb)[i] if fc_name in sl_map else 0
                mp = _entry_to_mpconfig(entry)
                linear = getattr(mlp, attr, None)
                if mp is not None and isinstance(linear, torch.nn.Linear):
                    setattr(mlp, attr, SCLinear(linear, sc_prec, mode="bipolar", mp_cfg=mp))
                    patch_stats[fc_name] = patch_stats.get(fc_name, 0) + 1
    return patch_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data")
    ap.add_argument("--sc_config", default="skip_worst30")
    ap.add_argument("--uniform_sl", type=int, default=0,
                    help="Use a uniform sl on all active SC ops. Mutually exclusive with --sl_map_json.")
    ap.add_argument("--sl_map_json", default="",
                    help="Path to a JSON containing either {sl_map: ...} or the sl_map itself.")
    ap.add_argument("--sc_prec", type=int, default=8)
    ap.add_argument("--n_calib", type=int, default=128)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--calib_seed", type=int, default=1)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--fwd_chunk", type=int, default=32)
    ap.add_argument("--avg_sc_draws", type=int, default=1)
    ap.add_argument("--start_block", type=int, default=0)
    ap.add_argument("--r2_threshold", type=float, default=0.0,
                    help="Skip QwT compensation in a block if its fit r2 is "
                         "<= this threshold. Matches qwt_sc_overnight semantics.")
    ap.add_argument("--max_block_for_qwt", type=int, default=10**9,
                    help="Hard cap: never compensate blocks with index >= this. "
                         "Matches qwt_sc_overnight semantics.")
    ap.add_argument("--last_block_r2_threshold", type=float, default=-1.0,
                    help="Stricter r2 gate for the LAST block only (negative = "
                         "use --r2_threshold). Matches qwt_sc_overnight semantics.")
    ap.add_argument("--lookahead_veto", action="store_true",
                    help="Per-block 1-step binary lookahead. For each block that "
                         "passes the r2 gate, compare downstream FP propagation "
                         "with vs without comp and veto apply if skip is better. "
                         "Matches qwt_sc_overnight semantics.")
    ap.add_argument("--rmse_margin", type=float, default=0.0,
                    help="Require raw_rmse/after_rmse > this margin as an extra "
                         "gate on top of --r2_threshold. 0 = disabled. Matches "
                         "qwt_sc_overnight semantics.")
    ap.add_argument("--skip_qwt", action="store_true",
                    help="Skip QwT calibration; evaluate raw SC model only (sc_raw).")
    # Comp-block knobs (mirror qwt_sc_overnight.py). Default is SC comp so the
    # QwT residual runs through the same Sobol/XNOR kernels as the block ops,
    # matching uniform-path behavior. Pass --comp_mode fp to restore the
    # historical FP-comp behavior implied by this script's filename.
    ap.add_argument("--comp_mode", choices=["fp", "sc"], default="sc")
    ap.add_argument("--comp_sc_prec", type=int, default=8)
    ap.add_argument("--comp_sc_mode", choices=["bipolar", "unipolar"], default="bipolar")
    ap.add_argument("--head_aligned", action="store_true",
                    help="Use HeadAlignedSCLinear (per-head D=64, reuses block QK pool) "
                         "for SC comp. No effect when --comp_mode fp.")
    ap.add_argument("--n_heads", type=int, default=16,
                    help="Head count for HeadAlignedSCLinear; must divide D=1024.")
    ap.add_argument("--out_json", default="results/eval_custom_sl_map_fpcomp.json")
    args = ap.parse_args()

    if bool(args.uniform_sl > 0) == bool(args.sl_map_json):
        raise ValueError("Specify exactly one of --uniform_sl or --sl_map_json")

    os.environ.setdefault("XFORMERS_DISABLED", "1")
    seed_all(args.eval_seed)
    set_noise_model(False)
    device = torch.device("cuda")

    active_ops_merged = build_active_ops(args.sc_config)
    n_blocks = 1 + max(bi for _op, bi in active_ops_merged)
    if args.uniform_sl > 0:
        sl_map = make_uniform_sl_map(active_ops_merged, n_blocks, args.uniform_sl)
    else:
        sl_map = load_sl_map(args.sl_map_json)
    active_ops = expand_active_ops(active_ops_merged, sl_map)
    main_sl = compute_main_sl_custom(sl_map, active_ops)
    print(f"[config] sc_config={args.sc_config} main_sl={main_sl:.2f}", flush=True)
    print(f"[config] alloc={summarize_allocation_custom(sl_map, active_ops)}", flush=True)
    print(f"[config] per_op={summarize_per_op_custom(sl_map, active_ops)}", flush=True)

    calib_ds = build_dataset(args.data_root, args.calib_seed, args.n_calib)
    eval_ds = build_dataset(args.data_root, args.eval_seed, args.n_eval)
    calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model_fp = load_model(device).eval()
    model_sc = load_model(device).eval()
    patch_stats = patch_model_with_custom_sl_map(model_sc, args.sc_prec, sl_map)

    results_block: dict = {}
    calib_report = None
    calib_s = 0.0

    if args.skip_qwt:
        print("[eval] SC raw (QwT skipped)", flush=True)
        seed_all(args.eval_seed)
        res_sc_raw = evaluate(model_sc, eval_loader, device)
        print("[result:sc_raw]", res_sc_raw, flush=True)
        results_block["sc_raw"] = res_sc_raw
    else:
        # Build comp factory: single-variant SC (default) or FP fallback. Mirrors
        # qwt_sc_overnight's single-variant fast path — scale=1.0, polarity=+1.
        D_resid = 1024
        if args.comp_mode == "sc":
            if args.head_aligned:
                comp_factory = make_head_aligned_factory(
                    args.comp_sc_prec, args.comp_sc_mode,
                    n_heads=args.n_heads, cfg_override=None,
                    scale=1.0, polarity=1,
                )
                comp_desc = f"sc/head_aligned/h{args.n_heads}/prec{args.comp_sc_prec}"
            else:
                seed_bank = build_seed_bank(D_resid, args.comp_sc_prec, 1)
                cfg0 = seed_bank[0][1]
                comp_factory = make_sc_factory(
                    args.comp_sc_prec, args.comp_sc_mode,
                    cfg_override=cfg0, scale=1.0, polarity=1,
                )
                comp_desc = f"sc/default/prec{args.comp_sc_prec}/{args.comp_sc_mode}"
        else:
            comp_factory = make_fp_factory(scale=1.0, polarity=1)
            comp_desc = "fp"
        print(f"[comp] factory={comp_desc}", flush=True)

        t0 = time.time()
        seed_all(args.calib_seed)
        calib_report = calibrate_qwt(
            model_fp=model_fp, model_sc=model_sc,
            blocks_fp=list(get_blocks(model_fp)),
            blocks_sc_container=get_blocks(model_sc),
            calib_loader=calib_loader,
            device=device,
            n_calib=args.n_calib,
            ridge=args.ridge,
            start_block=args.start_block,
            fwd_chunk=args.fwd_chunk,
            avg_sc_draws=args.avg_sc_draws,
            comp_factory=comp_factory,
            comp_factory_variants=None,
            comp_refit_iters=0,
            r2_threshold=args.r2_threshold,
            max_block_for_qwt=args.max_block_for_qwt,
            last_block_r2_threshold=(args.last_block_r2_threshold
                                     if args.last_block_r2_threshold >= 0.0
                                     else None),
            rmse_margin=args.rmse_margin,
            lookahead_veto=args.lookahead_veto,
        )
        calib_s = time.time() - t0
        print(f"[calib] done in {calib_s:.1f}s", flush=True)

        print("[eval] SC + FP-comp", flush=True)
        seed_all(args.eval_seed)
        res_sc_comp = evaluate(model_sc, eval_loader, device)
        print("[result:sc_comp]", res_sc_comp, flush=True)
        results_block["sc_comp"] = res_sc_comp

    print("[eval] FP reference", flush=True)
    seed_all(args.eval_seed)
    res_fp = evaluate(model_fp, eval_loader, device)
    print("[result:fp]", res_fp, flush=True)
    results_block["fp"] = res_fp

    out = {
        "config": {
            "sc_config": args.sc_config,
            "uniform_sl": args.uniform_sl,
            "sl_map_json": args.sl_map_json,
            "sc_prec": args.sc_prec,
            "n_calib": args.n_calib,
            "n_eval": args.n_eval,
            "calib_seed": args.calib_seed,
            "eval_seed": args.eval_seed,
            "batch_size": args.batch_size,
            "ridge": args.ridge,
            "start_block": args.start_block,
            "avg_sc_draws": args.avg_sc_draws,
            "r2_threshold": args.r2_threshold,
            "max_block_for_qwt": args.max_block_for_qwt,
            "last_block_r2_threshold": args.last_block_r2_threshold,
            "rmse_margin": args.rmse_margin,
            "lookahead_veto": bool(args.lookahead_veto),
            "skip_qwt": bool(args.skip_qwt),
            "comp_mode": args.comp_mode,
            "comp_sc_prec": args.comp_sc_prec,
            "comp_sc_mode": args.comp_sc_mode,
            "head_aligned": bool(args.head_aligned),
            "n_heads": args.n_heads if args.head_aligned else None,
        },
        "main_sl": main_sl,
        "allocation": summarize_allocation_custom(sl_map, active_ops),
        "per_op_allocation": summarize_per_op_custom(sl_map, active_ops),
        "patch_stats": patch_stats,
        "calib": None if args.skip_qwt else {
            "elapsed_s": calib_s,
            "per_block": calib_report,
        },
        "results": results_block,
        "sl_map": sl_map,
    }
    if args.skip_qwt:
        # Mirror raw SC top-1/top-5 at top level so the sweep summarizer
        # (which treats eval.py-style outputs as "top-level top1") picks it up.
        out["top1"] = results_block["sc_raw"].get("top1")
        out["top5"] = results_block["sc_raw"].get("top5")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
