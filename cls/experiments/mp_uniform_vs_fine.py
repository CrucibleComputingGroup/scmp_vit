#!/usr/bin/env python
"""Compare uniform vs sensitivity-aware per-(op,block) stoc_len allocation.

At the same FLOP-weighted average stoc_len (main_sl), does sensitivity-aware
allocation achieve higher accuracy than uniform?

Strategies:
  - uniform:     All active SC ops at the same stoc_len
  - sensitivity: Most sensitive ops get highest stoc_len, least sensitive lowest
  - inverse:     Control — least sensitive get highest (should be worst)
  - random:      Random assignment with matched average (avg over 3 seeds)

Quick start (one comparison):
    python experiments/mp_uniform_vs_fine.py \\
        --sc_config skip_worst40 --target_main_sl 128 \\
        --strategy sensitivity --n_eval 500

Sweep all strategies + targets for one sc_config:
    python experiments/mp_uniform_vs_fine.py \\
        --sc_config skip_worst40 --sweep --n_eval 500

Full overnight sweep across multiple sc_configs:
    python experiments/mp_uniform_vs_fine.py \\
        --sweep_all --n_eval 500
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]   # vit_sc/
CLS = HERE / "cls"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CLS))
sys.path.insert(0, str(HERE / "sc"))
QWT_SC_LIB = HERE / "third_party" / "QwT-SC" / "QwT-vit-sc"
if (QWT_SC_LIB / "qwt_sc").exists():
    sys.path.insert(0, str(QWT_SC_LIB))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from eval import build_transform, load_model, seed_all
from imagenet_parquet import ImageNetParquetVal
from sc_attention_patch import (
    make_sc_attention_forward, SCLinear, SC_OP_NAMES, set_noise_model,
)
from sc_integration.mp_linear import MPConfig, sc_prec_for_stoc_len

SENSITIVITY_JSON = HERE / "results" / "sensitivity_all_ops.json"

# Per-op MACs per block (DINOv2 ViT-L/14: D=1024, 16 heads, head_dim=64, seq=257)
_SEQ = 257
_OP_MACS = {
    "qk":      16 * _SEQ * 64 * _SEQ,
    "av":      16 * _SEQ * _SEQ * 64,
    "proj":    _SEQ * 1024 * 3072 + _SEQ * 1024 * 1024,
    "mlp_fc1": _SEQ * 1024 * 4096,
    "mlp_fc2": _SEQ * 4096 * 1024,
}
_COMP_MACS = _SEQ * 1024 * 1024

# ─── schedule builders (reuse overnight infra) ───────────────────────────

_SC_PRESETS = {
    "full_attn": dict(sc_qk=True, sc_av=True, sc_qkv_proj=True,
                      sc_out_proj=True, sc_mlp=False),
    "qk_only":   dict(sc_qk=True, sc_av=False, sc_qkv_proj=False,
                       sc_out_proj=False, sc_mlp=False),
    "qk_av":     dict(sc_qk=True, sc_av=True, sc_qkv_proj=False,
                       sc_out_proj=False, sc_mlp=False),
}
_FINE_KS = {
    "skip_worst50": 50, "skip_worst40": 40, "skip_worst30": 30,
    "skip_worst20": 20, "all_ops": 0,
}


def _build_skip_worst_k(k, n_blocks=24):
    with open(SENSITIVITY_JSON) as f:
        data = json.load(f)
    rows = sorted(data["grid"], key=lambda r: r["l2"], reverse=True)
    drop = {(r["op"], int(r["block"])) for r in rows[:k]}
    spec = {name: [1] * n_blocks for name in SC_OP_NAMES}
    for op, bi in drop:
        spec[op][bi] = 0
    return spec


def build_active_ops(sc_config, n_blocks=24):
    """Return set of active (op, block) pairs."""
    if sc_config in _FINE_KS:
        sched = _build_skip_worst_k(_FINE_KS[sc_config], n_blocks)
        return {(op, bi)
                for op in SC_OP_NAMES
                for bi in range(n_blocks)
                if sched[op][bi]}
    if sc_config in _SC_PRESETS:
        p = _SC_PRESETS[sc_config]
        active = set()
        for bi in range(n_blocks):
            if p.get("sc_qk"):
                active.add(("qk", bi))
            if p.get("sc_av"):
                active.add(("av", bi))
            if p.get("sc_qkv_proj") or p.get("sc_out_proj"):
                active.add(("proj", bi))
            if p.get("sc_mlp"):
                active.add(("mlp_fc1", bi))
                active.add(("mlp_fc2", bi))
        return active
    raise ValueError(f"Unknown sc_config: {sc_config}")


# ─── sensitivity ──────────────────────────────────────────────────────────

def load_sensitivity():
    with open(SENSITIVITY_JSON) as f:
        data = json.load(f)
    return {(r["op"], int(r["block"])): r["l2"] for r in data["grid"]}


# ─── stoc_len computation ────────────────────────────────────────────────

def compute_main_sl(sl_map, active_ops):
    total_macs = total_bitops = 0
    for op, bi in active_ops:
        sl = sl_map[op][bi]
        m = _OP_MACS[op]
        total_macs += m
        total_bitops += m * sl
    return total_bitops / total_macs if total_macs else 0


def compute_eff_sl(main_sl, active_ops, comp_sl=256, n_blocks=24):
    main_macs = sum(_OP_MACS[op] for op, _ in active_ops)
    comp_macs = n_blocks * _COMP_MACS
    return (main_sl * main_macs + comp_sl * comp_macs) / (main_macs + comp_macs)


# ─── allocation strategies ───────────────────────────────────────────────

def _empty_sl_map(n_blocks=24):
    return {op: [0] * n_blocks for op in SC_OP_NAMES}


def alloc_uniform(active_ops, stoc_len, n_blocks=24):
    """All active ops at the same stoc_len."""
    m = _empty_sl_map(n_blocks)
    for op, bi in active_ops:
        m[op][bi] = stoc_len
    return m


def _ranked_active_ops(active_ops, sens_map, reverse=False):
    """Sort active ops by sensitivity. reverse=True → ascending (least sensitive first)."""
    return sorted(active_ops,
                  key=lambda x: sens_map.get(x, 0),
                  reverse=not reverse)


def alloc_sensitivity(active_ops, target_main_sl, sens_map,
                      levels=(256, 128, 64), n_blocks=24):
    """Greedy: most sensitive ops get highest level, least sensitive get lowest.
    Two cutoffs are searched to match target_main_sl."""
    ranked = _ranked_active_ops(active_ops, sens_map, reverse=False)
    return _greedy_alloc(ranked, target_main_sl, levels, n_blocks)


def alloc_inverse(active_ops, target_main_sl, sens_map,
                  levels=(256, 128, 64), n_blocks=24):
    """Control: LEAST sensitive get highest level (should be worst)."""
    ranked = _ranked_active_ops(active_ops, sens_map, reverse=True)
    return _greedy_alloc(ranked, target_main_sl, levels, n_blocks)


def alloc_random(active_ops, target_main_sl, levels=(256, 128, 64),
                 seed=42, n_blocks=24):
    """Random assignment with matched average (shuffled sensitivity order)."""
    ranked = list(active_ops)
    random.Random(seed).shuffle(ranked)
    return _greedy_alloc(ranked, target_main_sl, levels, n_blocks)


def _greedy_alloc(ranked_ops, target, levels, n_blocks=24):
    """Given ops in priority order and available levels (descending),
    search for the best two cutoffs (k_high, k_mid) to match target main_sl.

    ranked_ops[0..k_high) → levels[0]  (highest)
    ranked_ops[k_high..k_mid) → levels[1]  (middle, if 3 levels)
    ranked_ops[k_mid..) → levels[-1]  (lowest)
    """
    n = len(ranked_ops)
    mac_list = [_OP_MACS[op] for op, _ in ranked_ops]
    total_macs = sum(mac_list)

    # Prefix sums for fast range queries
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + mac_list[i]

    best_map = None
    best_diff = float("inf")

    if len(levels) == 1:
        # Trivial: all at one level
        m = _empty_sl_map(n_blocks)
        for op, bi in ranked_ops:
            m[op][bi] = levels[0]
        return m

    if len(levels) == 2:
        hi, lo = levels
        for k in range(n + 1):
            macs_hi = prefix[k]
            avg = (hi * macs_hi + lo * (total_macs - macs_hi)) / total_macs
            d = abs(avg - target)
            if d < best_diff:
                best_diff = d
                best_k = k
        m = _empty_sl_map(n_blocks)
        for i, (op, bi) in enumerate(ranked_ops):
            m[op][bi] = hi if i < best_k else lo
        return m

    # 3 levels: search two cutoffs
    hi, mid, lo = levels[0], levels[1], levels[2]
    best_kh = best_km = 0
    for kh in range(n + 1):
        macs_hi = prefix[kh]
        for km in range(kh, n + 1):
            macs_mid = prefix[km] - prefix[kh]
            macs_lo = total_macs - prefix[km]
            avg = (hi * macs_hi + mid * macs_mid + lo * macs_lo) / total_macs
            d = abs(avg - target)
            if d < best_diff:
                best_diff = d
                best_kh, best_km = kh, km

    m = _empty_sl_map(n_blocks)
    for i, (op, bi) in enumerate(ranked_ops):
        if i < best_kh:
            m[op][bi] = hi
        elif i < best_km:
            m[op][bi] = mid
        else:
            m[op][bi] = lo
    return m


# ─── model patching ──────────────────────────────────────────────────────

def _discover_blocks(model):
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        return list(model.backbone.blocks)
    if hasattr(model, "blocks"):
        return list(model.blocks)
    raise RuntimeError("Cannot find transformer blocks")


def patch_model_with_sl_map(model, sc_prec, sl_map):
    """Patch model in-place with per-(op,block) stoc_len values.

    For each (op, block), if sl_map[op][block] > 0 the op is replaced by an
    SC variant running at that stoc_len.  We use MPConfig with a single level
    to set arbitrary stoc_len through the existing MP infrastructure.
    """
    blocks = _discover_blocks(model)
    nb = len(blocks)
    stats = Counter()

    for i, blk in enumerate(blocks):
        attn_mod = None
        for m in blk.modules():
            if hasattr(m, "qkv") and hasattr(m, "proj") and hasattr(m, "num_heads"):
                attn_mod = m
                break

        qk_sl = sl_map.get("qk", [0] * nb)[i]
        av_sl = sl_map.get("av", [0] * nb)[i]

        if attn_mod and (qk_sl > 0 or av_sl > 0):
            qk_mp = MPConfig([qk_sl], [1.0]) if qk_sl > 0 else None
            av_mp = MPConfig([av_sl], [1.0]) if av_sl > 0 else None
            fwd = make_sc_attention_forward(
                sc_prec, sc_qk=qk_sl > 0, sc_av=av_sl > 0,
                qk_mp_cfg=qk_mp, av_mp_cfg=av_mp,
            )
            attn_mod.forward = fwd.__get__(attn_mod, type(attn_mod))
            stats["attn"] += 1

        proj_sl = sl_map.get("proj", [0] * nb)[i]
        if attn_mod and proj_sl > 0:
            mp = MPConfig([proj_sl], [1.0])
            if isinstance(attn_mod.qkv, nn.Linear):
                attn_mod.qkv = SCLinear(attn_mod.qkv, sc_prec, mode="bipolar",
                                        mp_cfg=mp)
                stats["qkv_proj"] += 1
            if isinstance(attn_mod.proj, nn.Linear):
                attn_mod.proj = SCLinear(attn_mod.proj, sc_prec, mode="bipolar",
                                         mp_cfg=mp)
                stats["out_proj"] += 1

        mlp = getattr(blk, "mlp", None)
        if mlp:
            for fc_name in ("mlp_fc1", "mlp_fc2"):
                fc_sl = sl_map.get(fc_name, [0] * nb)[i]
                attr = fc_name.replace("mlp_", "")  # fc1 / fc2
                linear = getattr(mlp, attr, None)
                if fc_sl > 0 and isinstance(linear, nn.Linear):
                    mp = MPConfig([fc_sl], [1.0])
                    setattr(mlp, attr,
                            SCLinear(linear, sc_prec, mode="bipolar", mp_cfg=mp))
                    stats[fc_name] += 1

    return dict(stats)


# ─── evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, log_every=20):
    model.eval()
    n = top1 = top5 = 0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        _, p5 = logits.topk(5, 1)
        c = p5.eq(y.unsqueeze(1))
        top1 += c[:, 0].sum().item()
        top5 += c.any(1).sum().item()
        n += y.numel()
        if (i + 1) % log_every == 0:
            dt = time.time() - t0
            print(f"  [{i+1}] top1={top1/n:.4f}  top5={top5/n:.4f}  "
                  f"({n} imgs, {dt:.1f}s)", flush=True)
    dt = time.time() - t0
    return {"n": n, "top1": top1 / n, "top5": top5 / n, "elapsed_s": round(dt, 1)}


def build_eval_loader(data_root, n_eval, seed, batch_size, workers=4):
    ds = ImageNetParquetVal(data_root, transform=build_transform(224))
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if 0 < n_eval < len(ds):
        ds = Subset(ds, idx[:n_eval])
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True)


# ─── comp support (optional) ─────────────────────────────────────────────

def calibrate_and_attach_comp(model_fp, model_sc, calib_loader_a, calib_loader_b, device,
                              n_calib=256, ridge=1e-4,
                              cos_threshold=0.5, norm_floor=0.0,
                              last_block_cos_threshold=0.8,
                              lookahead_veto=False):
    """Calibrate the QwT cross-seed cosine-gated compensator and attach to
    ``model_sc`` in-place. Requires two disjoint calib loaders (the
    algorithm's core — see ``qwt_sc.compensation``)."""
    from qwt_sc import calibrate_qwt
    from experiments.qwt_sc_overnight import get_blocks

    blocks_fp = list(get_blocks(model_fp))
    blocks_sc_container = get_blocks(model_sc)
    last_tau = last_block_cos_threshold if last_block_cos_threshold > cos_threshold else None

    report = calibrate_qwt(
        model_fp=model_fp, model_sc=model_sc,
        blocks_fp=blocks_fp, blocks_sc_container=blocks_sc_container,
        calib_loader_a=calib_loader_a, calib_loader_b=calib_loader_b,
        device=device, n_calib=n_calib, ridge=ridge,
        start_block=0, fwd_chunk=32,
        cos_threshold=cos_threshold, norm_floor=norm_floor,
        last_block_cos_threshold=last_tau,
        lookahead_veto=lookahead_veto,
    )
    return report


# ─── single run ───────────────────────────────────────────────────────────

def run_one(sl_map, active_ops, sc_prec, eval_loader, device, seed,
            with_comp=False, calib_loader=None, comp_kwargs=None):
    """Load a fresh model, patch, optionally add comp, evaluate."""
    # Clear Sobol config cache from previous runs to free GPU memory
    from sc_attention_patch import _CFG_CACHE
    _CFG_CACHE.clear()
    torch.cuda.empty_cache()

    model = load_model(device).eval()
    stats = patch_model_with_sl_map(model, sc_prec, sl_map)

    raw_res = None
    comp_res = None

    if not with_comp:
        seed_all(seed)
        raw_res = evaluate(model, eval_loader, device)
    else:
        # Raw SC
        seed_all(seed)
        raw_res = evaluate(model, eval_loader, device)

        # Comp
        model_fp = load_model(device).eval()
        kw = comp_kwargs or {}
        seed_all(kw.get("calib_seed", 1))
        calibrate_and_attach_comp(
            model_fp, model, calib_loader, device,
            n_calib=kw.get("n_calib", 256),
            ridge=kw.get("ridge", 1e-4),
            sc_prec=sc_prec,
        )
        seed_all(seed)
        comp_res = evaluate(model, eval_loader, device)
        del model_fp

    del model
    torch.cuda.empty_cache()
    return {"raw": raw_res, "comp": comp_res, "patch_stats": stats}


# ─── allocation summary ──────────────────────────────────────────────────

def summarize_allocation(sl_map, active_ops):
    level_macs = Counter()
    for op, bi in active_ops:
        level_macs[sl_map[op][bi]] += _OP_MACS[op]
    total = sum(level_macs.values())
    return {str(k): round(100 * v / total, 1) for k, v in sorted(level_macs.items(), reverse=True)}


# ─── main / sweep ────────────────────────────────────────────────────────

STRATEGIES = ("uniform", "sensitivity", "inverse", "random", "proportional")
# Focus on targets where mixing is non-trivial.
# With levels=(256,128): useful range is [128, 256].
# With levels=(256,64): useful range is [64, 256].
DEFAULT_TARGETS = (256, 224, 192, 160, 128)

# For mixed strategies, use 2-level sets {hi, lo} that FORCE mixing.
# The "uniform" strategy always uses a single stoc_len = target.
MIX_LEVELS = (256, 64)  # default for mixed strategies — forces mixing at target=128


def alloc_proportional(active_ops, target_main_sl, sens_map,
                       min_sl=64, max_sl=256, n_blocks=24):
    """Assign stoc_len continuously proportional to sensitivity.

    Most sensitive op → max_sl, least sensitive → min_sl.
    All values scaled to hit target_main_sl as the FLOP-weighted average.
    Each op gets its own integer stoc_len ∈ [min_sl, max_sl].
    """
    sens_vals = {k: sens_map.get(k, 0) for k in active_ops}
    s_min = min(sens_vals.values())
    s_max = max(sens_vals.values())

    if s_max - s_min < 1e-8:
        return alloc_uniform(active_ops, target_main_sl, n_blocks)

    # Map: highest sensitivity → 1.0, lowest → 0.0
    fracs = {k: (v - s_min) / (s_max - s_min) for k, v in sens_vals.items()}

    # Initial linear mapping to [min_sl, max_sl]
    raw = {k: min_sl + f * (max_sl - min_sl) for k, f in fracs.items()}

    # Scale to hit target average (FLOP-weighted)
    total_macs = sum(_OP_MACS[op] for op, _ in active_ops)
    raw_avg = sum(_OP_MACS[op] * raw[(op, bi)] for op, bi in active_ops) / total_macs
    if abs(raw_avg) < 1e-8:
        return alloc_uniform(active_ops, target_main_sl, n_blocks)

    scale = target_main_sl / raw_avg
    scaled = {k: max(min_sl, min(max_sl, v * scale)) for k, v in raw.items()}

    # Quantize to a small set of levels to avoid OOM from too many
    # unique stoc_len values (each needs a Sobol config on GPU).
    # Round to nearest in {64, 96, 128, 160, 192, 224, 256}.
    QUANT_LEVELS = [64, 96, 128, 160, 192, 224, 256]
    m = _empty_sl_map(n_blocks)
    for (op, bi), sl in scaled.items():
        # Pick the closest quantization level
        best = min(QUANT_LEVELS, key=lambda q: abs(q - sl))
        m[op][bi] = best
    return m


def gen_sl_map(strategy, active_ops, target, sens_map, levels, seed=42):
    if strategy == "uniform":
        return alloc_uniform(active_ops, target)
    elif strategy == "sensitivity":
        return alloc_sensitivity(active_ops, target, sens_map, levels)
    elif strategy == "inverse":
        return alloc_inverse(active_ops, target, sens_map, levels)
    elif strategy == "random":
        return alloc_random(active_ops, target, levels, seed=seed)
    elif strategy == "proportional":
        lo, hi = min(levels), max(levels)
        return alloc_proportional(active_ops, target, sens_map,
                                  min_sl=lo, max_sl=hi)
    raise ValueError(strategy)


def run_sweep(sc_config, targets, strategies, args):
    """Run all (strategy, target_main_sl) combinations for one sc_config."""
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    set_noise_model(False)
    device = torch.device("cuda")

    sens_map = load_sensitivity()
    active_ops = build_active_ops(sc_config)
    mix_levels = tuple(int(x) for x in args.levels.split(","))

    print(f"\n{'='*72}")
    print(f"  sc_config={sc_config}  active_ops={len(active_ops)}  "
          f"mix_levels={mix_levels}  with_comp={args.with_comp}")
    print(f"{'='*72}\n")

    eval_loader = build_eval_loader(
        args.data_root, args.n_eval, args.eval_seed, args.batch_size,
        workers=args.workers,
    )
    calib_loader = None
    if args.with_comp:
        calib_loader = build_eval_loader(
            args.data_root, args.n_calib, args.calib_seed, args.batch_size,
            workers=args.workers,
        )

    # FP baseline (once)
    print("[fp] evaluating baseline", flush=True)
    model_fp = load_model(device).eval()
    seed_all(args.eval_seed)
    fp_res = evaluate(model_fp, eval_loader, device)
    del model_fp
    torch.cuda.empty_cache()
    print(f"[fp] top1={fp_res['top1']:.4f}\n", flush=True)

    all_results = {"fp": fp_res, "runs": []}

    for target in targets:
        for strategy in strategies:
            # For "uniform": all ops at the same stoc_len = target
            # Works for any stoc_len (not just powers of 2) via MPConfig path

            n_random = 1  # single seed for random; add more if needed
            for rseed_idx in range(n_random):
                rseed = args.eval_seed + rseed_idx * 1000
                tag = f"{strategy}"
                if strategy == "random":
                    tag += f"_s{rseed}"

                # Uniform uses single level; mixed strategies use mix_levels
                use_levels = (target,) if strategy == "uniform" else mix_levels
                sl_map = gen_sl_map(strategy, active_ops, target, sens_map,
                                    use_levels, seed=rseed)
                actual_msl = compute_main_sl(sl_map, active_ops)
                actual_esl = compute_eff_sl(actual_msl, active_ops)
                alloc = summarize_allocation(sl_map, active_ops)

                print(f"--- {sc_config} | target_msl={target} | {tag} ---",
                      flush=True)
                print(f"  actual main_sl={actual_msl:.1f}  eff_sl={actual_esl:.1f}"
                      f"  alloc={alloc}", flush=True)

                comp_kw = {
                    "calib_seed": args.calib_seed,
                    "n_calib": args.n_calib,
                    "ridge": args.ridge,
                }
                res = run_one(
                    sl_map, active_ops, args.sc_prec, eval_loader, device,
                    seed=args.eval_seed, with_comp=args.with_comp,
                    calib_loader=calib_loader, comp_kwargs=comp_kw,
                )

                raw_t1 = res["raw"]["top1"] if res["raw"] else None
                comp_t1 = res["comp"]["top1"] if res["comp"] else None
                print(f"  => raw_top1={raw_t1}  comp_top1={comp_t1}\n",
                      flush=True)

                entry = {
                    "sc_config": sc_config,
                    "target_main_sl": target,
                    "strategy": tag,
                    "actual_main_sl": round(actual_msl, 1),
                    "actual_eff_sl": round(actual_esl, 1),
                    "allocation": alloc,
                    "raw": res["raw"],
                    "comp": res["comp"],
                    "sl_map": {op: sl_map[op] for op in SC_OP_NAMES},
                }
                all_results["runs"].append(entry)

                # Incremental save
                out_dir = Path(args.out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{sc_config}_sweep.json"
                with open(out_path, "w") as f:
                    json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*72}")
    print(f"  SUMMARY: {sc_config}  (FP baseline: {fp_res['top1']:.4f})")
    print(f"{'='*72}")
    hdr = f"  {'target':>6} {'strategy':<16} {'main_sl':>8} {'eff_sl':>7} {'raw_t1':>7}"
    if args.with_comp:
        hdr += f" {'comp_t1':>8}"
    hdr += f"  {'alloc'}"
    print(hdr)
    print(f"  {'-'*70}")
    for r in all_results["runs"]:
        line = (f"  {r['target_main_sl']:>6} {r['strategy']:<16} "
                f"{r['actual_main_sl']:>8.1f} {r['actual_eff_sl']:>7.1f} "
                f"{r['raw']['top1']:>7.4f}")
        if r["comp"]:
            line += f" {r['comp']['top1']:>8.4f}"
        line += f"  {r['allocation']}"
        print(line)
    print()

    return all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc_config", default="skip_worst40")
    ap.add_argument("--target_main_sl", type=int, default=0,
                    help="Single target main_sl (0 = use sweep defaults)")
    ap.add_argument("--strategy", default="",
                    help="Single strategy (empty = all)")
    ap.add_argument("--levels", default="256,128,64")
    ap.add_argument("--sc_prec", type=int, default=8)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--n_calib", type=int, default=256)
    ap.add_argument("--calib_seed", type=int, default=1)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data_root", default="/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data")
    ap.add_argument("--out_dir", default="results/mp_comparison")
    ap.add_argument("--with_comp", action="store_true",
                    help="Include QwT compensation (slower, ~8min/run)")
    # Sweep modes
    ap.add_argument("--sweep", action="store_true",
                    help="Sweep all strategies + targets for --sc_config")
    ap.add_argument("--sweep_all", action="store_true",
                    help="Sweep multiple sc_configs")
    args = ap.parse_args()

    if args.sweep_all:
        configs = ["skip_worst40", "skip_worst30", "full_attn", "skip_worst50"]
        for cfg in configs:
            args.sc_config = cfg
            run_sweep(cfg, DEFAULT_TARGETS, STRATEGIES, args)
    elif args.sweep:
        run_sweep(args.sc_config, DEFAULT_TARGETS, STRATEGIES, args)
    else:
        targets = (args.target_main_sl,) if args.target_main_sl else DEFAULT_TARGETS
        strategies = (args.strategy,) if args.strategy else STRATEGIES
        run_sweep(args.sc_config, targets, strategies, args)


if __name__ == "__main__":
    main()
