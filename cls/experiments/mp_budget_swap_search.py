#!/usr/bin/env python
"""Budget-matched iterative MP search using only calibration-time local proxy.

This script searches a per-(op, block) stoc_len map under a fixed main-budget
constraint, without using end-to-end accuracy during search.

Design choices for the first implementation:

* Search space is block-level only (no per-row/per-head bins yet).
* Search uses only local block proxy on a small calibration set:
  - ``raw_mse``: mean block output MSE to FP
  - ``comp_residual``: residual MSE after closed-form linear compensation
* The search operates on zero-budget pairwise swaps:
  - one searchable slot upgrades by one discrete step
  - another searchable slot downgrades by the same step
  - total main budget is preserved exactly
* ``qk`` / ``av`` can be fixed to low stoc_len and kept out of the search.

Typical usage:

    python cls/experiments/mp_budget_swap_search.py \
        --sc_config skip_worst30 \
        --target_main_sl 128 \
        --search_ops proj,mlp_fc1,mlp_fc2 \
        --fixed_ops qk=64,av=64 \
        --op_min_levels proj=96,mlp_fc1=96,mlp_fc2=192 \
        --levels 64,96,128,192,256 \
        --n_search 32 \
        --proxy comp_residual
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parents[2]   # vit_sc/
CLS = HERE / "cls"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CLS))
sys.path.insert(0, str(HERE / "sc"))
QWT_SC_LIB = HERE / "third_party" / "QwT-SC" / "QwT-vit-sc"
if (QWT_SC_LIB / "qwt_sc").exists():
    sys.path.insert(0, str(QWT_SC_LIB))

from eval import build_transform, load_model, seed_all
from imagenet_parquet import ImageNetParquetVal
from sc_attention_patch import (
    SC_OP_NAMES,
    SCLinear,
    make_sc_attention_forward,
    set_noise_model,
)
from sc_integration.mp_linear import MPConfig
from mp_auto_calibrator import RidgeFitter


SENSITIVITY_JSON = HERE / "results" / "sensitivity_all_ops.json"

# DINOv2 ViT-L/14 per-block MACs for the main SC ops.
_SEQ = 257
_OP_MACS = {
    "qk":      16 * _SEQ * 64 * _SEQ,
    "av":      16 * _SEQ * _SEQ * 64,
    "proj":    _SEQ * 1024 * 3072 + _SEQ * 1024 * 1024,
    "mlp_fc1": _SEQ * 1024 * 4096,
    "mlp_fc2": _SEQ * 4096 * 1024,
}
_COMP_MACS = _SEQ * 1024 * 1024

_SC_PRESETS = {
    "full_attn": dict(sc_qk=True, sc_av=True, sc_qkv_proj=True,
                      sc_out_proj=True, sc_mlp=False),
    "qk_only":   dict(sc_qk=True, sc_av=False, sc_qkv_proj=False,
                      sc_out_proj=False, sc_mlp=False),
    "qk_av":     dict(sc_qk=True, sc_av=True, sc_qkv_proj=False,
                      sc_out_proj=False, sc_mlp=False),
    "full_everything": dict(sc_qk=True, sc_av=True, sc_qkv_proj=True,
                            sc_out_proj=True, sc_mlp=True),
    "mlp_only": dict(sc_qk=False, sc_av=False, sc_qkv_proj=False,
                     sc_out_proj=False, sc_mlp=True),
    "av_only":  dict(sc_qk=False, sc_av=True, sc_qkv_proj=False,
                     sc_out_proj=False, sc_mlp=False),
}
_FINE_KS = {
    "skip_worst50": 50,
    "skip_worst40": 40,
    "skip_worst30": 30,
    "skip_worst20": 20,
    "all_ops": 0,
}


def _build_skip_worst_k(k: int, n_blocks: int = 24) -> dict[str, list[int]]:
    with open(SENSITIVITY_JSON) as f:
        data = json.load(f)
    rows = sorted(data["grid"], key=lambda r: r["l2"], reverse=True)
    drop = {(r["op"], int(r["block"])) for r in rows[:k]}
    spec = {name: [1] * n_blocks for name in SC_OP_NAMES}
    for op, bi in drop:
        spec[op][bi] = 0
    return spec


def build_active_ops(sc_config: str, n_blocks: int = 24) -> set[tuple[str, int]]:
    if sc_config in _FINE_KS:
        sched = _build_skip_worst_k(_FINE_KS[sc_config], n_blocks)
        return {
            (op, bi)
            for op in SC_OP_NAMES
            for bi in range(n_blocks)
            if sched[op][bi]
        }
    if sc_config in _SC_PRESETS:
        p = _SC_PRESETS[sc_config]
        active: set[tuple[str, int]] = set()
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


def load_sensitivity() -> dict[tuple[str, int], float]:
    with open(SENSITIVITY_JSON) as f:
        data = json.load(f)
    return {(r["op"], int(r["block"])): float(r["l2"]) for r in data["grid"]}


def compute_main_sl(sl_map: dict[str, list[int]],
                    active_ops: set[tuple[str, int]]) -> float:
    total_macs = 0.0
    total_bitops = 0.0
    for op, bi in active_ops:
        sl = sl_map[op][bi]
        m = float(_OP_MACS[op])
        total_macs += m
        total_bitops += m * float(sl)
    return total_bitops / total_macs if total_macs > 0 else 0.0


def compute_eff_sl(main_sl: float, active_ops: set[tuple[str, int]],
                   comp_sl: int = 256, n_blocks: int = 24) -> float:
    main_macs = sum(_OP_MACS[op] for op, _ in active_ops)
    comp_macs = n_blocks * _COMP_MACS
    return (main_sl * main_macs + comp_sl * comp_macs) / (main_macs + comp_macs)


def summarize_allocation(sl_map: dict[str, list[int]],
                         active_ops: set[tuple[str, int]]) -> dict[str, float]:
    level_macs = Counter()
    for op, bi in active_ops:
        level_macs[sl_map[op][bi]] += _OP_MACS[op]
    total = sum(level_macs.values())
    if total == 0:
        return {}
    return {
        str(sl): round(100.0 * macs / total, 1)
        for sl, macs in sorted(level_macs.items(), reverse=True)
    }


def summarize_per_op(sl_map: dict[str, list[int]],
                     active_ops: set[tuple[str, int]]) -> dict[str, dict[str, float]]:
    by_op: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for op, bi in active_ops:
        by_op[op][sl_map[op][bi]] += 1
        totals[op] += 1
    out: dict[str, dict[str, float]] = {}
    for op in sorted(by_op):
        n = totals[op]
        out[op] = {
            str(sl): round(by_op[op][sl] / n, 4)
            for sl in sorted(by_op[op], reverse=True)
        }
    return out


def _empty_sl_map(n_blocks: int) -> dict[str, list[int]]:
    return {op: [0] * n_blocks for op in SC_OP_NAMES}


def _parse_int_map(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    if not text:
        return out
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"expected key=value entry, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def _parse_levels(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("levels must be non-empty")
    return sorted(set(vals))


def _nearest_level(target: int, levels: list[int]) -> int:
    return min(levels, key=lambda x: abs(x - target))


def build_level_spaces(
    *,
    search_slots: list[tuple[str, int]],
    global_levels: list[int],
    op_min_levels: dict[str, int],
    op_max_levels: dict[str, int],
) -> dict[tuple[str, int], list[int]]:
    spaces: dict[tuple[str, int], list[int]] = {}
    for op, bi in search_slots:
        lo = op_min_levels.get(op, global_levels[0])
        hi = op_max_levels.get(op, global_levels[-1])
        allowed = [sl for sl in global_levels if lo <= sl <= hi]
        if not allowed:
            raise ValueError(f"no valid levels for {(op, bi)} in [{lo}, {hi}]")
        spaces[(op, bi)] = allowed
    return spaces


def choose_initial_levels_exact(
    *,
    search_slots: list[tuple[str, int]],
    spaces: dict[tuple[str, int], list[int]],
    target_sum_sl: int,
    sens_map: dict[tuple[str, int], float],
) -> tuple[dict[tuple[str, int], int], int, float]:
    """Exact/closest DP over searchable slots.

    Search slots are assumed to share the same MAC, so matching main-budget
    reduces to matching the sum of assigned stoc_len values.
    """
    states: dict[int, tuple[float, Optional[tuple[int, int]]]] = {0: (0.0, None)}
    history: list[dict[int, tuple[int, int]]] = []

    for idx, slot in enumerate(search_slots):
        next_states: dict[int, tuple[float, tuple[int, int]]] = {}
        choices: dict[int, tuple[int, int]] = {}
        sens = sens_map.get(slot, 0.0)
        for cur_sum, (cur_util, _prev) in states.items():
            for level in spaces[slot]:
                new_sum = cur_sum + level
                new_util = cur_util + sens * float(level)
                prev_best = next_states.get(new_sum)
                if prev_best is None or new_util > prev_best[0]:
                    next_states[new_sum] = (new_util, (cur_sum, level))
                    choices[new_sum] = (cur_sum, level)
        states = next_states
        history.append(choices)
        if not states:
            raise RuntimeError("DP state collapsed to empty")

    best_sum = None
    best_key = None
    best_util = None
    for sum_sl, (util, _prev) in states.items():
        key = (abs(sum_sl - target_sum_sl), -util, sum_sl)
        if best_key is None or key < best_key:
            best_key = key
            best_sum = sum_sl
            best_util = util
    assert best_sum is not None and best_util is not None

    assignment: dict[tuple[str, int], int] = {}
    cur_sum = best_sum
    for idx in range(len(search_slots) - 1, -1, -1):
        prev_sum, level = history[idx][cur_sum]
        assignment[search_slots[idx]] = level
        cur_sum = prev_sum
    return assignment, int(best_sum), float(best_util)


def choose_initial_levels_from_base(
    *,
    search_slots: list[tuple[str, int]],
    spaces: dict[tuple[str, int], list[int]],
    target_sum_sl: int,
    base_levels: dict[tuple[str, int], int],
    sens_map: dict[tuple[str, int], float],
) -> tuple[dict[tuple[str, int], int], int, float]:
    """DP around a user-provided baseline assignment.

    The objective is to stay close to ``base_levels`` while matching the
    budget exactly / as closely as possible. Deviation from the base is
    penalized more on high-sensitivity slots.
    """
    states: dict[int, tuple[float, Optional[tuple[int, int]]]] = {0: (0.0, None)}
    history: list[dict[int, tuple[int, int]]] = []

    for slot in search_slots:
        next_states: dict[int, tuple[float, tuple[int, int]]] = {}
        choices: dict[int, tuple[int, int]] = {}
        sens = sens_map.get(slot, 0.0)
        base = int(base_levels[slot])
        for cur_sum, (cur_score, _prev) in states.items():
            for level in spaces[slot]:
                new_sum = cur_sum + level
                delta = abs(level - base)
                # Keep the solution close to the uniform baseline. When a
                # change is necessary, prefer changing lower-sensitivity slots.
                penalty = (1.0 + sens) * float(delta)
                new_score = cur_score - penalty
                prev_best = next_states.get(new_sum)
                if prev_best is None or new_score > prev_best[0]:
                    next_states[new_sum] = (new_score, (cur_sum, level))
                    choices[new_sum] = (cur_sum, level)
        states = next_states
        history.append(choices)
        if not states:
            raise RuntimeError("baseline-repair DP state collapsed to empty")

    best_sum = None
    best_key = None
    best_score = None
    for sum_sl, (score, _prev) in states.items():
        key = (abs(sum_sl - target_sum_sl), -score, sum_sl)
        if best_key is None or key < best_key:
            best_key = key
            best_sum = sum_sl
            best_score = score
    assert best_sum is not None and best_score is not None

    assignment: dict[tuple[str, int], int] = {}
    cur_sum = best_sum
    for idx in range(len(search_slots) - 1, -1, -1):
        prev_sum, level = history[idx][cur_sum]
        assignment[search_slots[idx]] = level
        cur_sum = prev_sum
    return assignment, int(best_sum), float(best_score)


def build_initial_sl_map(
    *,
    active_ops: set[tuple[str, int]],
    search_ops: set[str],
    fixed_ops: dict[str, int],
    target_main_sl: int,
    level_spaces: dict[tuple[str, int], list[int]],
    sens_map: dict[tuple[str, int], float],
    n_blocks: int,
    init_mode: str = "sensitivity_dp",
    init_level: Optional[int] = None,
) -> tuple[dict[str, list[int]], dict[str, object]]:
    sl_map = _empty_sl_map(n_blocks)
    search_slots = sorted((slot for slot in active_ops if slot[0] in search_ops),
                          key=lambda x: (x[1], x[0]))
    fixed_total = 0.0
    total_macs = 0.0
    search_macs = {_OP_MACS[op] for op, _ in search_slots}
    if len(search_macs) != 1:
        raise ValueError(
            "first implementation requires search_ops to share the same MAC; "
            f"got {sorted(search_macs)}"
        )
    search_mac = float(next(iter(search_macs))) if search_macs else 0.0

    for op, bi in active_ops:
        total_macs += float(_OP_MACS[op])
        if op in search_ops:
            continue
        sl = fixed_ops.get(op, target_main_sl)
        sl_map[op][bi] = int(sl)
        fixed_total += float(_OP_MACS[op]) * float(sl)

    target_total = float(target_main_sl) * total_macs
    if search_slots:
        remaining = target_total - fixed_total
        target_sum_sl = int(round(remaining / search_mac))
        min_sum = sum(min(level_spaces[slot]) for slot in search_slots)
        max_sum = sum(max(level_spaces[slot]) for slot in search_slots)
        clipped_target = min(max(target_sum_sl, min_sum), max_sum)
        if init_mode == "uniform_repair":
            base_target = target_main_sl if init_level is None else int(init_level)
            base_levels: dict[tuple[str, int], int] = {}
            for slot in search_slots:
                allowed = level_spaces[slot]
                if base_target <= min(allowed):
                    base_levels[slot] = min(allowed)
                elif base_target >= max(allowed):
                    base_levels[slot] = max(allowed)
                elif base_target in allowed:
                    base_levels[slot] = base_target
                else:
                    base_levels[slot] = _nearest_level(base_target, allowed)
            assignment, chosen_sum, util = choose_initial_levels_from_base(
                search_slots=search_slots,
                spaces=level_spaces,
                target_sum_sl=clipped_target,
                base_levels=base_levels,
                sens_map=sens_map,
            )
        elif init_mode == "sensitivity_dp":
            assignment, chosen_sum, util = choose_initial_levels_exact(
                search_slots=search_slots,
                spaces=level_spaces,
                target_sum_sl=clipped_target,
                sens_map=sens_map,
            )
        else:
            raise ValueError(f"unknown init_mode: {init_mode}")
        for (op, bi), sl in assignment.items():
            sl_map[op][bi] = int(sl)
    else:
        clipped_target = target_main_sl
        chosen_sum = 0
        util = 0.0

    actual_main = compute_main_sl(sl_map, active_ops)
    meta = {
        "init_mode": init_mode,
        "init_level": init_level,
        "search_slots": len(search_slots),
        "fixed_bitops": fixed_total,
        "target_total_bitops": target_total,
        "target_search_sum_sl": None if not search_slots else target_sum_sl,
        "clipped_target_search_sum_sl": None if not search_slots else clipped_target,
        "chosen_search_sum_sl": chosen_sum,
        "utility": util,
        "actual_main_sl": actual_main,
    }
    return sl_map, meta


def build_loader(data_root: str, n_images: int, seed: int,
                 batch_size: int, workers: int) -> DataLoader:
    ds = ImageNetParquetVal(data_root, transform=build_transform(224))
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if 0 < n_images < len(ds):
        ds = Subset(ds, idx[:n_images])
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True)


def get_blocks(model: nn.Module):
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        return list(model.backbone.blocks)
    if hasattr(model, "blocks"):
        return list(model.blocks)
    raise RuntimeError("Cannot find transformer blocks")


def patch_model_with_sl_map(model: nn.Module, sc_prec: int,
                            sl_map: dict[str, list[int]]) -> dict[str, int]:
    blocks = get_blocks(model)
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
                sc_prec=sc_prec,
                sc_av=av_sl > 0,
                sc_qk=qk_sl > 0,
                qk_mp_cfg=qk_mp,
                av_mp_cfg=av_mp,
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
        if mlp is not None:
            for fc_name in ("mlp_fc1", "mlp_fc2"):
                fc_sl = sl_map.get(fc_name, [0] * nb)[i]
                attr = fc_name.replace("mlp_", "")
                linear = getattr(mlp, attr, None)
                if fc_sl > 0 and isinstance(linear, nn.Linear):
                    mp = MPConfig([fc_sl], [1.0])
                    setattr(mlp, attr,
                            SCLinear(linear, sc_prec, mode="bipolar", mp_cfg=mp))
                    stats[fc_name] += 1

    return dict(stats)


def patch_block_with_spec(block: nn.Module, sc_prec: int,
                          block_spec: dict[str, int]) -> nn.Module:
    blk = copy.deepcopy(block)
    attn_mod = None
    for m in blk.modules():
        if hasattr(m, "qkv") and hasattr(m, "proj") and hasattr(m, "num_heads"):
            attn_mod = m
            break

    qk_sl = int(block_spec.get("qk", 0))
    av_sl = int(block_spec.get("av", 0))
    if attn_mod and (qk_sl > 0 or av_sl > 0):
        qk_mp = MPConfig([qk_sl], [1.0]) if qk_sl > 0 else None
        av_mp = MPConfig([av_sl], [1.0]) if av_sl > 0 else None
        fwd = make_sc_attention_forward(
            sc_prec=sc_prec,
            sc_av=av_sl > 0,
            sc_qk=qk_sl > 0,
            qk_mp_cfg=qk_mp,
            av_mp_cfg=av_mp,
        )
        attn_mod.forward = fwd.__get__(attn_mod, type(attn_mod))

    proj_sl = int(block_spec.get("proj", 0))
    if attn_mod and proj_sl > 0:
        mp = MPConfig([proj_sl], [1.0])
        if isinstance(attn_mod.qkv, nn.Linear):
            attn_mod.qkv = SCLinear(attn_mod.qkv, sc_prec, mode="bipolar",
                                    mp_cfg=mp)
        if isinstance(attn_mod.proj, nn.Linear):
            attn_mod.proj = SCLinear(attn_mod.proj, sc_prec, mode="bipolar",
                                     mp_cfg=mp)

    mlp = getattr(blk, "mlp", None)
    if mlp is not None:
        for fc_name in ("mlp_fc1", "mlp_fc2"):
            fc_sl = int(block_spec.get(fc_name, 0))
            attr = fc_name.replace("mlp_", "")
            linear = getattr(mlp, attr, None)
            if fc_sl > 0 and isinstance(linear, nn.Linear):
                mp = MPConfig([fc_sl], [1.0])
                setattr(mlp, attr,
                        SCLinear(linear, sc_prec, mode="bipolar", mp_cfg=mp))

    return blk


def capture_block_inputs(model: nn.Module, loader: DataLoader, device: torch.device,
                         n_search: int, cache_dtype: torch.dtype) -> list[torch.Tensor]:
    blocks = get_blocks(model)
    captured: list[list[torch.Tensor]] = [[] for _ in blocks]

    handles = []
    for idx, blk in enumerate(blocks):
        def _make_hook(block_idx: int):
            def _hook(_m, args):
                x = args[0].detach().to(device="cpu", dtype=cache_dtype)
                captured[block_idx].append(x)
            return _hook
        handles.append(blk.register_forward_pre_hook(_make_hook(idx)))

    seen = 0
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
                imgs = imgs.to(device, non_blocking=True)
                model(imgs)
                seen += imgs.size(0)
                if seen >= n_search:
                    break
    finally:
        for handle in handles:
            handle.remove()

    return [torch.cat(xs, dim=0)[:n_search].contiguous() for xs in captured]


def forward_block_batched(block: nn.Module, x_cpu: torch.Tensor,
                          device: torch.device, chunk: int) -> torch.Tensor:
    outs = []
    for start in range(0, x_cpu.size(0), chunk):
        xb = x_cpu[start:start + chunk].to(device=device, dtype=torch.float32,
                                           non_blocking=True)
        y = block(xb)
        outs.append(y.detach().float().cpu())
    return torch.cat(outs, dim=0)


def compute_proxy_score(
    *,
    proxy: str,
    x_cpu: torch.Tensor,
    y_fp_cpu: torch.Tensor,
    y_sc_cpu: torch.Tensor,
    device: torch.device,
    ridge: float,
) -> float:
    if proxy == "raw_mse":
        return float((y_fp_cpu - y_sc_cpu).pow(2).mean().item())
    if proxy != "comp_residual":
        raise ValueError(f"unknown proxy: {proxy}")
    x_flat = x_cpu.reshape(-1, x_cpu.size(-1)).to(device=device, dtype=torch.float32)
    y_fp_flat = y_fp_cpu.reshape(-1, y_fp_cpu.size(-1)).to(device=device, dtype=torch.float32)
    y_sc_flat = y_sc_cpu.reshape(-1, y_sc_cpu.size(-1)).to(device=device, dtype=torch.float32)
    fitter = RidgeFitter(x_flat, ridge=ridge)
    score = fitter.residual_norm_sq(y_fp_flat - y_sc_flat) / float(y_fp_flat.numel())
    del x_flat, y_fp_flat, y_sc_flat, fitter
    return float(score)


def evaluate_block_moves(
    *,
    block_idx: int,
    fp_block: nn.Module,
    x_cpu: torch.Tensor,
    current_spec: dict[str, int],
    search_ops_in_block: list[str],
    spaces: dict[tuple[str, int], list[int]],
    sc_prec: int,
    proxy: str,
    ridge: float,
    device: torch.device,
    fwd_chunk: int,
    block_weight: float,
) -> dict[str, object]:
    fp_block_cpu = copy.deepcopy(fp_block).cpu().eval()
    fp_block_dev = copy.deepcopy(fp_block_cpu).to(device).eval()
    y_fp_cpu = forward_block_batched(fp_block_dev, x_cpu, device, fwd_chunk)
    del fp_block_dev
    torch.cuda.empty_cache()

    current_blk = patch_block_with_spec(fp_block_cpu, sc_prec, current_spec).to(device).eval()
    y_cur_cpu = forward_block_batched(current_blk, x_cpu, device, fwd_chunk)
    current_score = compute_proxy_score(
        proxy=proxy, x_cpu=x_cpu, y_fp_cpu=y_fp_cpu, y_sc_cpu=y_cur_cpu,
        device=device, ridge=ridge,
    )
    del current_blk, y_cur_cpu
    torch.cuda.empty_cache()

    candidates = []
    for op in search_ops_in_block:
        allowed = spaces[(op, block_idx)]
        cur_sl = int(current_spec[op])
        pos = allowed.index(cur_sl)
        for next_pos in (pos - 1, pos + 1):
            if not (0 <= next_pos < len(allowed)):
                continue
            new_sl = int(allowed[next_pos])
            cand_spec = dict(current_spec)
            cand_spec[op] = new_sl
            cand_blk = patch_block_with_spec(fp_block_cpu, sc_prec, cand_spec).to(device).eval()
            y_new_cpu = forward_block_batched(cand_blk, x_cpu, device, fwd_chunk)
            score = compute_proxy_score(
                proxy=proxy, x_cpu=x_cpu, y_fp_cpu=y_fp_cpu, y_sc_cpu=y_new_cpu,
                device=device, ridge=ridge,
            )
            candidates.append({
                "slot": (op, block_idx),
                "op": op,
                "block": block_idx,
                "old_sl": cur_sl,
                "new_sl": new_sl,
                "delta_sl": new_sl - cur_sl,
                "delta_score": block_weight * (score - current_score),
                "score": score,
            })
            del cand_blk, y_new_cpu
            torch.cuda.empty_cache()

    del fp_block_cpu, y_fp_cpu
    torch.cuda.empty_cache()
    return {
        "block": block_idx,
        "weight": block_weight,
        "current_score": current_score,
        "candidates": candidates,
    }


def build_block_weight(active_ops: set[tuple[str, int]],
                       sens_map: dict[tuple[str, int], float]) -> dict[int, float]:
    weights: dict[int, float] = defaultdict(float)
    for op, bi in active_ops:
        weights[bi] += sens_map.get((op, bi), 0.0)
    return dict(weights)


def build_current_spec(sl_map: dict[str, list[int]], block_idx: int) -> dict[str, int]:
    return {op: int(sl_map[op][block_idx]) for op in SC_OP_NAMES}


def iterative_swap_search(
    *,
    active_ops: set[tuple[str, int]],
    search_ops: set[str],
    sl_map: dict[str, list[int]],
    spaces: dict[tuple[str, int], list[int]],
    loader: DataLoader,
    device: torch.device,
    sc_prec: int,
    proxy: str,
    ridge: float,
    max_iters: int,
    fwd_chunk: int,
    cache_dtype: torch.dtype,
    log_fn=print,
) -> tuple[dict[str, list[int]], list[dict[str, object]], dict[str, object]]:
    sens_map = load_sensitivity()
    block_weight = build_block_weight(active_ops, sens_map)
    search_slots = {slot for slot in active_ops if slot[0] in search_ops}
    history: list[dict[str, object]] = []
    iteration_meta: dict[str, object] = {}

    for it in range(max_iters):
        t0 = time.time()
        model_sc = load_model(device).eval()
        patch_stats = patch_model_with_sl_map(model_sc, sc_prec, sl_map)
        x_by_block = capture_block_inputs(model_sc, loader, device,
                                          loader.dataset.__len__(), cache_dtype)
        del model_sc
        torch.cuda.empty_cache()

        model_fp = load_model(torch.device("cpu")).eval()
        fp_blocks = get_blocks(model_fp)

        upgrades_by_step: dict[int, list[dict[str, object]]] = defaultdict(list)
        downgrades_by_step: dict[int, list[dict[str, object]]] = defaultdict(list)
        block_reports = []
        search_blocks = sorted({bi for _op, bi in search_slots})
        for bi in search_blocks:
            ops_here = sorted(op for op, bj in search_slots if bj == bi)
            if not ops_here:
                continue
            report = evaluate_block_moves(
                block_idx=bi,
                fp_block=fp_blocks[bi],
                x_cpu=x_by_block[bi],
                current_spec=build_current_spec(sl_map, bi),
                search_ops_in_block=ops_here,
                spaces=spaces,
                sc_prec=sc_prec,
                proxy=proxy,
                ridge=ridge,
                device=device,
                fwd_chunk=fwd_chunk,
                block_weight=block_weight.get(bi, 1.0),
            )
            block_reports.append({
                "block": bi,
                "weight": report["weight"],
                "current_score": report["current_score"],
            })
            for cand in report["candidates"]:
                step = abs(int(cand["delta_sl"]))
                if cand["delta_sl"] > 0:
                    upgrades_by_step[step].append(cand)
                elif cand["delta_sl"] < 0:
                    downgrades_by_step[step].append(cand)

        del model_fp, fp_blocks, x_by_block
        torch.cuda.empty_cache()

        best_pair = None
        best_delta = 0.0
        for step, upgrades in upgrades_by_step.items():
            downgrades = downgrades_by_step.get(step, [])
            if not downgrades:
                continue
            for up in upgrades:
                for down in downgrades:
                    if up["slot"] == down["slot"]:
                        continue
                    pair_delta = float(up["delta_score"]) + float(down["delta_score"])
                    if best_pair is None or pair_delta < best_delta:
                        best_delta = pair_delta
                        best_pair = {
                            "step": step,
                            "upgrade": up,
                            "downgrade": down,
                            "delta_score": pair_delta,
                        }

        actual_main = compute_main_sl(sl_map, active_ops)
        iteration_meta = {
            "iter": it,
            "actual_main_sl": actual_main,
            "patch_stats": patch_stats,
            "block_reports": block_reports,
            "elapsed_s": round(time.time() - t0, 2),
        }
        log_fn(f"[swap_search] iter={it} main_sl={actual_main:.2f} "
               f"best_delta={best_delta:.4e} "
               f"elapsed={iteration_meta['elapsed_s']:.1f}s")
        if best_pair is None or best_delta >= -1e-12:
            iteration_meta["stopped"] = True
            break

        up = best_pair["upgrade"]
        down = best_pair["downgrade"]
        sl_map[up["op"]][up["block"]] = int(up["new_sl"])
        sl_map[down["op"]][down["block"]] = int(down["new_sl"])
        history.append({
            "iter": it,
            "delta_score": float(best_pair["delta_score"]),
            "step": int(best_pair["step"]),
            "upgrade": {
                "slot": list(up["slot"]),
                "old_sl": int(up["old_sl"]),
                "new_sl": int(up["new_sl"]),
                "delta_score": float(up["delta_score"]),
            },
            "downgrade": {
                "slot": list(down["slot"]),
                "old_sl": int(down["old_sl"]),
                "new_sl": int(down["new_sl"]),
                "delta_score": float(down["delta_score"]),
            },
            "main_sl_after": compute_main_sl(sl_map, active_ops),
        })

    return sl_map, history, iteration_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc_config", default="skip_worst30",
                    choices=list(_SC_PRESETS) + list(_FINE_KS))
    ap.add_argument("--target_main_sl", type=int, default=128)
    ap.add_argument("--levels", default="64,96,128,192,256")
    ap.add_argument("--search_ops", default="proj,mlp_fc1,mlp_fc2")
    ap.add_argument("--fixed_ops", default="qk=64,av=64")
    ap.add_argument("--op_min_levels", default="proj=96,mlp_fc1=96,mlp_fc2=192")
    ap.add_argument("--op_max_levels", default="")
    ap.add_argument("--proxy", default="comp_residual",
                    choices=["raw_mse", "comp_residual"])
    ap.add_argument("--init_mode", default="sensitivity_dp",
                    choices=["sensitivity_dp", "uniform_repair"],
                    help="Initial allocation strategy before iterative swaps. "
                         "'uniform_repair' starts from an all-init_level map "
                         "and only changes what is needed to satisfy the "
                         "budget and hard per-op floors.")
    ap.add_argument("--init_level", type=int, default=128,
                    help="Target per-slot stoc_len used by uniform_repair.")
    ap.add_argument("--n_search", type=int, default=32)
    ap.add_argument("--search_seed", type=int, default=1)
    ap.add_argument("--max_iters", type=int, default=8)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--fwd_chunk", type=int, default=16)
    ap.add_argument("--cache_dtype", default="float16",
                    choices=["float16", "float32"])
    ap.add_argument("--sc_prec", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data_root",
                    default="/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data")
    ap.add_argument("--out_json", default="results/mp_budget_swap_search.json")
    args = ap.parse_args()

    os.environ.setdefault("XFORMERS_DISABLED", "1")
    seed_all(args.search_seed)
    set_noise_model(False)
    device = torch.device("cuda")

    active_ops = build_active_ops(args.sc_config)
    levels = _parse_levels(args.levels)
    search_ops = {x.strip() for x in args.search_ops.split(",") if x.strip()}
    fixed_ops = _parse_int_map(args.fixed_ops)
    op_min_levels = _parse_int_map(args.op_min_levels)
    op_max_levels = _parse_int_map(args.op_max_levels)
    sens_map = load_sensitivity()

    n_blocks = 1 + max(bi for _op, bi in active_ops)
    search_slots = sorted(slot for slot in active_ops if slot[0] in search_ops)
    spaces = build_level_spaces(
        search_slots=search_slots,
        global_levels=levels,
        op_min_levels=op_min_levels,
        op_max_levels=op_max_levels,
    )

    sl_map, init_meta = build_initial_sl_map(
        active_ops=active_ops,
        search_ops=search_ops,
        fixed_ops=fixed_ops,
        target_main_sl=args.target_main_sl,
        level_spaces=spaces,
        sens_map=sens_map,
        n_blocks=n_blocks,
        init_mode=args.init_mode,
        init_level=args.init_level,
    )

    cache_dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32
    loader = build_loader(
        data_root=args.data_root,
        n_images=args.n_search,
        seed=args.search_seed,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    print("[swap_search] initial main_sl="
          f"{compute_main_sl(sl_map, active_ops):.2f} "
          f"eff_sl={compute_eff_sl(compute_main_sl(sl_map, active_ops), active_ops):.2f}")
    print(f"[swap_search] initial alloc={summarize_allocation(sl_map, active_ops)}")
    print(f"[swap_search] initial per_op={summarize_per_op(sl_map, active_ops)}")

    final_sl_map, history, last_iter = iterative_swap_search(
        active_ops=active_ops,
        search_ops=search_ops,
        sl_map=sl_map,
        spaces=spaces,
        loader=loader,
        device=device,
        sc_prec=args.sc_prec,
        proxy=args.proxy,
        ridge=args.ridge,
        max_iters=args.max_iters,
        fwd_chunk=args.fwd_chunk,
        cache_dtype=cache_dtype,
    )

    final_main = compute_main_sl(final_sl_map, active_ops)
    final_eff = compute_eff_sl(final_main, active_ops)
    result = {
        "sc_config": args.sc_config,
        "target_main_sl": args.target_main_sl,
        "proxy": args.proxy,
        "levels": levels,
        "search_ops": sorted(search_ops),
        "fixed_ops": fixed_ops,
        "op_min_levels": op_min_levels,
        "op_max_levels": op_max_levels,
        "n_search": args.n_search,
        "search_seed": args.search_seed,
        "init_meta": init_meta,
        "history": history,
        "last_iter": last_iter,
        "final_main_sl": final_main,
        "final_eff_sl": final_eff,
        "allocation": summarize_allocation(final_sl_map, active_ops),
        "per_op_allocation": summarize_per_op(final_sl_map, active_ops),
        "sl_map": {op: final_sl_map[op] for op in SC_OP_NAMES},
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("[swap_search] final main_sl="
          f"{final_main:.2f} eff_sl={final_eff:.2f}")
    print(f"[swap_search] final alloc={result['allocation']}")
    print(f"[swap_search] wrote {out_path}")


if __name__ == "__main__":
    main()
