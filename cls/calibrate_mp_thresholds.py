"""Calibrate per-operator, per-block SC mixed-precision thresholds for a ViT.

ViT port of ``scmp_llm/benchmark/ppl/calibrate_mp_thresholds.py``, by way of
scmp_cnn's port of the same file. From scmp_llm: the transformer operator
taxonomy and the per-block layer bucketing (a ViT block is a decoder layer).
From scmp_cnn: the decision to ship ONLY the ``act_global`` importance signal —
the LLM sweep showed the cheap reconstruction-error signal is about as good as
the expensive loss-aware variants once the budget is truly iso-compute, so the
grad / measured / fisher *probes* are not ported.

What IS ported from scmp_llm (2026-07-25, after MP_REPORT.md) are the two knobs
that shape the allocation objective itself, because act_global's default pair —
``W_g = 1`` with a MAC-weighted cost — is what pins the cheap ``out_proj`` at
the ladder ceiling while starving ``mlp_fc1``:

* ``--objective {sigma,sigma2,delta_sigma2}`` (``mp_objectives.py``, copied
  verbatim from scmp_llm) — the currency the per-row error is priced in.
* ``--cross-layer-weight {uniform,macs}`` — the per-group importance weight
  ``W_g``. ``macs`` puts benefit and cost in the SAME currency, which collapses
  the per-row argmin to one global cycles-per-sigma exchange rate shared by
  every operator (see ``_weight_for_key``).

Mechanism (unchanged from both):

  1. FP teacher forward over a few calibration batches.
  2. Per SC module call, record each input row's metric (``|row|.amax``, the
     exact quantity the runtime classifier sees) and its per-level SC
     reconstruction error sigma (relative L2 vs the FP output, one sc_matmul
     per level).
  3. Solve ONE global Lagrangian: a single shared lambda across all
     (operator, layer-bucket) groups, with costs MAC-weighted by default so
     iso-budget == iso-compute.
  4. Per-level counts -> thresholds on the sorted normalized metric -> JSON,
     consumed unchanged by ``scmp_kernels.mp.AdaptiveMPConfig`` and dispatched
     by ``model/sc_vit.py``'s SCLinear.

Two ViT-specific notes:

* **No trace file.** The LLM needs ``--mac-weights-trace`` because attention
  dims grow per decode step. A ViT has a fixed token count, so MACs/row is
  static per module (``in_features * out_features``) and is read straight off
  the layer.
* **qk / av are NOT calibrated** (v1). scmp_llm dispatches attention MP per
  query row; ``model/sc_vit.py`` dispatches qk per HEAD, because its uniform
  path quantizes QK with ``granularity="per_head"``. Calibrating one and
  deploying the other is exactly the mismatch scmp_llm's qk branch warns
  about, so attention is left uniform here and its MACs are simply charged to
  the budget. On DINOv2 ViT-L/14 attention is ~4% of SC MACs (mlp 64%,
  proj 32%), so the headroom given up is small. See ``--attn_stoc_len``.

Defaults differ deliberately from scmp_llm's: ``--budget-weight macs`` and
``--budget-scope global`` are the DEFAULTS here. In scmp_llm they are opt-in
(``rows`` / ``per_bucket``) even though every shipped recipe passes the other
two, and forgetting them silently reproduces the row-weighted budget bug that
repo spent a release diagnosing.

Levels/budget are in HALVED space when ``--halve 1`` (cap ``2**(sc_prec-1)``),
same convention as scmp_llm and scmp_cnn: a level value IS the cycle count,
and effective levels = 2x it.

Usage:

  python cls/calibrate_mp_thresholds.py \
    --mp_levels 128,96,64,48 --budget_ratio 0.75 \
    --n_calib_images 256 \
    --output_json results/mp_calib/vitl__avg96.json

  # then wrap it for deployment and evaluate
  python cls/eval.py --mode sc --adaptive_mp 1 \
    --adaptive_mp_table results/mp_calib/vitl__avg96_wrapper.json ...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

_HERE = Path(__file__).resolve().parent          # vit_sc/cls
_ROOT = _HERE.parent                             # vit_sc
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from imagenet_parquet import ImageNetParquetVal  # noqa: E402
from mp_objectives import objective_errors as _objective_errors  # noqa: E402
from eval import build_transform, load_model, seed_all  # noqa: E402
from model.sc_vit import (  # noqa: E402
    SCLinear, patch_model, set_sc_defaults, set_sc_enabled, sc_enabled,
    set_group_stoclen, set_qk_granularity, total_blocks,
)
from scmp_kernels import sc_matmul  # noqa: E402

DEFAULT_ROOT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data"

# The MP-dispatchable ops. qk/av are deliberately absent — see the module
# docstring: their runtime dispatch granularity (per-head) does not match the
# per-row calibration this file performs.
DEFAULT_OPERATORS = "qkv_proj,out_proj,mlp_fc1,mlp_fc2"


# =====================================================================
# ViT adapters: build the SC model + a calibration forward iterator.
# Everything below this section is task-agnostic (it only ever sees SC
# modules), so ViT-specific code stays confined to these two functions.
# =====================================================================


def _build_sc_model(args, device):
    """DINOv2 ViT-L/14 + linear head with every MP-eligible op on SC.

    The calibrator hooks SCLinear modules, so proj/mlp must be SC-patched even
    though the teacher output it compares against is FP: the hook recomputes
    both sides itself from the layer's own input. Attention is patched at
    ``--attn_stoc_len`` because it still consumes budget, but it carries no
    MP table (see the module docstring).
    """
    set_sc_defaults(halve_bipolar_stoc_len=bool(args.halve),
                    chunk_d=args.chunk_d)
    set_qk_granularity(args.qk_granularity)
    model = load_model(device)
    ops = "mlp_fc1,mlp_fc2,qk,av,proj" if args.sc_attention else "mlp_fc1,mlp_fc2,proj"
    stats = patch_model(
        model, sc_prec=args.sc_prec,
        sc_ops=ops,
        stoc_len=args.attn_stoc_len or None,
        chunk_d=args.chunk_d,
    )
    print(f"[calib] patched {stats}", flush=True)
    return model


def _build_calib_iter(args, device):
    """Calibration batches from ImageNet val, EXPLICITLY disjoint from the
    images accuracy is later reported on.

    Two independent shuffles are not disjoint: drawing 256 calibration images
    with one seed and 2048 eval images with another overlaps in ~13 images by
    chance. That is far below the sampling noise on top-1, but MP thresholds
    ARE fitted on this data, so the eval indices are removed from the pool
    outright rather than argued away. ``--eval_seed`` / ``--eval_exclude_n``
    must match the eval invocation for the exclusion to be the right set.
    """
    import random as _random

    ds = ImageNetParquetVal(args.data_root, transform=build_transform(224))
    n = len(ds)

    excluded: set = set()
    if args.eval_exclude_n > 0:
        e = list(range(n))
        _random.Random(args.eval_seed).shuffle(e)
        excluded = set(e[: args.eval_exclude_n])

    idx = [i for i in range(n) if i not in excluded]
    _random.Random(args.calib_seed).shuffle(idx)
    lo = args.calib_offset
    sel = idx[lo: lo + args.n_calib_images]
    if len(sel) < args.n_calib_images:
        raise ValueError(
            f"calibration needs {args.n_calib_images} images from offset "
            f"{lo}, but only {len(idx)} remain after excluding "
            f"{len(excluded)} eval images from {n}")
    print(f"[calib] calib pool {len(idx)} imgs "
          f"({len(excluded)} eval images excluded), using {len(sel)}",
          flush=True)
    loader = DataLoader(
        Subset(ds, sel), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    def run_forward(model, batch):
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        model(imgs.to(device, non_blocking=True))

    # The exact dataset indices thresholds were fitted on. Recorded in the
    # table so a FULL-val evaluation can subtract them and stay honestly
    # disjoint: --eval_exclude_n only works when the eval subset is the first
    # N of a known shuffle, which is false once eval covers all 50k.
    return loader, run_forward, sorted(int(i) for i in sel)


# =====================================================================
# Metric / error utilities (verbatim from the LLM calibrator)
# =====================================================================


def _normalize_metric(metric: torch.Tensor) -> torch.Tensor:
    metric = metric.float()
    if metric.numel() == 0:
        return metric
    m_min = metric.min()
    m_max = metric.max()
    if (m_max - m_min).item() < 1e-8:
        return torch.ones_like(metric, dtype=torch.float32)
    return (metric - m_min) / (m_max - m_min)


def _metric_candidates(x_rows: torch.Tensor) -> dict:
    """act_global_v2 candidate dispatch metrics for one call, per-call
    NORMALIZED like the primary amax metric. ``x_rows`` is the SAME row-source
    tensor amax was computed from, flattened to (rows, D). Formulas mirror
    ``scmp_kernels.mp.compute_row_metric`` exactly (runtime parity):
    l2 = ‖row‖₂, crest = ‖row‖_inf / ‖row‖₂ (eps 1e-12)."""
    xf = x_rows.float().reshape(-1, x_rows.shape[-1])
    l2 = xf.norm(dim=-1)
    amax = xf.abs().amax(dim=-1)
    crest = amax / (l2 + 1e-12)
    return {"l2": _normalize_metric(l2), "crest": _normalize_metric(crest)}


def _relative_l2_rows(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = pred.float().reshape(pred.shape[0], -1)
    target_f = target.float().reshape(target.shape[0], -1)
    denom = target_f.norm(dim=-1).clamp_min(1e-8)
    return (pred_f - target_f).norm(dim=-1) / denom


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    d = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def _bucket_index(value: int, total: int, num_buckets: int) -> int:
    if num_buckets <= 1 or total <= 1:
        return 0
    ratio = value / max(total - 1, 1)
    return min(num_buckets - 1, int(ratio * num_buckets))


# =====================================================================
# Budget allocation (Lagrangian binary search) — verbatim from the LLM
# =====================================================================


def _cost_assignments(
    errors: np.ndarray,
    costs: np.ndarray,
    budget_total: float,
) -> np.ndarray:
    """Per-group budget: pick a level per row minimizing sum(error) under
    sum(cost) <= budget (used only for --budget-scope per_bucket)."""
    n_units = errors.shape[0]
    min_cost = float(n_units * costs[-1])
    max_cost = float(n_units * costs[0])
    if budget_total <= min_cost:
        return np.full(n_units, len(costs) - 1, dtype=np.int64)
    if budget_total >= max_cost:
        return np.zeros(n_units, dtype=np.int64)

    def solve_for_lambda(lmbd: float):
        objective = errors + lmbd * costs[None, :]
        assignment = objective.argmin(axis=1)
        total_cost = float(costs[assignment].sum())
        return assignment, total_cost

    lo, hi = 0.0, 1.0
    _, cost_hi = solve_for_lambda(hi)
    while cost_hi > budget_total and hi < 1e6:
        hi *= 2.0
        _, cost_hi = solve_for_lambda(hi)

    best = np.zeros(n_units, dtype=np.int64)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        cand, cost_mid = solve_for_lambda(mid)
        best = cand
        if cost_mid > budget_total:
            lo = mid
        else:
            hi = mid
    return best


def _global_lambda(
    weighted_errors: list,
    costs: np.ndarray,
    budget_ratio: float,
    ref: float,
    row_counts: list,
    prices: Optional[list] = None,
) -> float:
    """ONE shared Lagrange multiplier across all groups for a global,
    R_g-weighted budget — the cross-layer allocation core. ``row_counts[g]``
    is group g's TRUE per-forward weight (rows, or MACs under
    --budget-weight macs, in which case iso-budget = iso-compute).
    ``prices[g]`` is the per-stored-row cycle PRICE in the argmin. Default (None)
    is ``rep_g = R_g / n_g`` — each stored (subsampled) row stands for rep_g true
    rows. ``--argmin-pricing macs`` passes ``mac_per_row[op]`` instead: the
    KKT-consistent price where the subsample ratio cancels between benefit and
    cost (the extra true/stored factor in rep_g otherwise over-prices heavily
    subsampled ops and starves them — the LLM's attention/dense mispricing)."""
    total_rows = float(sum(row_counts))
    if total_rows <= 0:
        return 0.0
    if prices is None:
        prices = [float(R) / max(int(e.shape[0]), 1)
                  for e, R in zip(weighted_errors, row_counts)]
    reps = prices
    global_budget = budget_ratio * ref * total_rows
    min_cost = float(total_rows * costs[-1])
    max_cost = float(total_rows * costs[0])
    if global_budget >= max_cost:
        return 0.0
    if global_budget <= min_cost:
        return 1e12

    def total_cost(lmbd: float) -> float:
        t = 0.0
        for e, R, rep in zip(weighted_errors, row_counts, reps):
            assign = (e + lmbd * rep * costs[None, :]).argmin(axis=1)
            t += float(R) * float(costs[assign].mean())
        return t

    lo, hi = 0.0, 1.0
    while total_cost(hi) > global_budget and hi < 1e12:
        hi *= 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if total_cost(mid) > global_budget:
            lo = mid
        else:
            hi = mid
    return hi


def _global_lambda_multi(
    weighted_errors: list,
    group_costs: list,
    budget_ratio: float,
    ref: float,
    row_counts: list,
    prices: list,
) -> float:
    """``_global_lambda`` with a DIFFERENT cost ladder per group.

    Needed once per-group ladders exist: the shared-``costs`` version prices
    every group against the global ladder, so a group that was re-fitted onto
    its own rungs would be solved at one ladder and deployed at another. The
    budget semantics are unchanged — one shared λ, cost measured in the
    ``budget_weight`` currency (MACs under ``--budget-weight macs``).
    """
    total_rows = float(sum(row_counts))
    if total_rows <= 0:
        return 0.0
    global_budget = budget_ratio * ref * total_rows
    min_cost = sum(float(R) * float(c[-1]) for R, c in zip(row_counts, group_costs))
    max_cost = sum(float(R) * float(c[0]) for R, c in zip(row_counts, group_costs))
    if global_budget >= max_cost:
        return 0.0
    if global_budget <= min_cost:
        return 1e12

    def total_cost(lmbd: float) -> float:
        t = 0.0
        for e, R, rep, costs in zip(weighted_errors, row_counts, prices,
                                    group_costs):
            assign = (e + lmbd * rep * costs[None, :]).argmin(axis=1)
            t += float(R) * float(costs[assign].mean())
        return t

    lo, hi = 0.0, 1.0
    while total_cost(hi) > global_budget and hi < 1e12:
        hi *= 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if total_cost(mid) > global_budget:
            lo = mid
        else:
            hi = mid
    return hi


def _select_group_ladder(err_obj: np.ndarray, costs: np.ndarray,
                         lam: float, rep: float, weight: float,
                         k: int) -> np.ndarray:
    """Pick the ``k`` rungs of ``costs`` this group actually uses.

    A calibrated table historically carried ONE ladder shared by every
    (op x layer-bucket) bucket, with only the thresholds varying. Measured
    occupancy shows that wastes roughly half the rungs: on ViT-L/14 the
    ``out_proj`` buckets pin at the ceiling and put ZERO mass on the bottom
    rungs while ``mlp_fc1`` lives entirely at the bottom — the two populations
    sit at opposite ends of a shared ladder, so each gets about half the
    available resolution.

    The fix is to calibrate sigma on a DENSE candidate ladder and let each
    group keep only the rungs its own argmin puts mass on. Returns INDICES into
    ``costs``, ascending, so the caller can slice both costs and errors.

    Selection keeps the two ENDPOINTS of the group's occupied range and fills
    the rest by occupancy. Ranking purely by mass is wrong here and was measured
    to be wrong: the first version dropped the high rungs (few rows, by
    definition — the importance distribution is heavy-tailed) and capped whole
    `mlp_fc1` buckets at 24 cycles, so no row in them could reach high precision
    however fragile it was. That cost 1.66 pt vs the shared ladder at eff 64
    (`s32_glk` 0.7407 vs `s32_base` 0.7573). The top rung is exactly the one a
    mass criterion discards and exactly the one outlier rows need.
    """
    assign = ((weight * err_obj) + lam * rep * costs[None, :]).argmin(axis=1)
    mass = np.bincount(assign, minlength=costs.size).astype(np.float64)
    k = max(2, min(int(k), int(costs.size)))
    occupied = np.flatnonzero(mass > 0)
    if occupied.size <= k:
        keep = occupied
    else:
        # Endpoints of the occupied range are non-negotiable: the top rung is
        # the headroom, the bottom rung is the cheap end the budget is paid
        # from. Middle slots go by occupancy, ties toward the longer stream.
        forced = {int(occupied[0]), int(occupied[-1])}
        order = np.lexsort((np.arange(costs.size), -mass))
        for idx in order:
            if len(forced) >= k:
                break
            if mass[idx] > 0:
                forced.add(int(idx))
        keep = np.sort(np.fromiter(forced, dtype=np.int64))
    if keep.size < 2:
        # Degenerate: everything on one rung. Keep a neighbour so the group
        # still has something to trade against — a 1-rung "ladder" is uniform,
        # and AdaptiveMPConfig rejects it anyway.
        only = int(keep[0]) if keep.size else 0
        nb = only + 1 if only + 1 < costs.size else only - 1
        keep = np.sort(np.array([only, max(nb, 0)], dtype=np.int64))
    return keep


def _normalize_group_weights(raw: dict, floor_frac: float = 0.1) -> dict:
    """Floor ΔL at a small positive fraction of the mean, then scale to mean 1.

    The floor keeps measurement noise (or a genuinely negative ΔL) from starving
    a group outright; the mean-1 scaling keeps the global budget interpretation
    (realized avg ≈ budget_ratio·ref) unchanged, since a uniform scale on W_g is
    absorbed by λ. Ported from scmp_llm's ``_normalize_group_weights``.
    """
    vals = np.asarray(list(raw.values()), dtype=np.float64)
    if vals.size == 0:
        return {}
    pos = np.clip(vals, 0.0, None)
    mean_pos = float(pos.mean()) if pos.mean() > 0 else 1.0
    floor = floor_frac * mean_pos
    floored = {k: max(float(v), floor) for k, v in raw.items()}
    m = float(np.mean(list(floored.values()))) or 1.0
    return {k: v / m for k, v in floored.items()}


@torch.no_grad()
def _measure_group_sensitivity(model, probe_batches, operators, total_blocks,
                               measure_buckets, baseline_sl, probe_sl):
    """Phase-2 measured-ΔLoss knock-down probe (`measured_marg` semantics).

    Runs the patched model in real SC inference, all groups uniform at
    ``baseline_sl``, to get a reference loss L0. Then, for each
    (operator, probe-bucket) group, re-runs with ONLY that group knocked down to
    ``probe_sl`` and records ΔL = L_g − L0: the group's marginal loss
    sensitivity to its own precision budget. This is the ground-truth
    cross-layer signal the σ reconstruction error only proxies for.

    ``probe_sl`` is a SMALL step below baseline, not the ladder floor — scmp_llm
    found the floor knock-down saturates past the collapse cliff, so ΔL stops
    being a derivative and the ranking degrades at exactly the tight budgets
    where it matters (its `measured` vs `measured_marg` split).

    Cost is ``|ops| × measure_buckets + 1`` forwards over ``probe_batches``, all
    forward-only (no autograd). Returns ({(op, probe_bucket): ΔL}, L0).
    """
    ops = sorted(operators)
    loss_fn = nn.CrossEntropyLoss()

    def mean_loss(gmap):
        set_group_stoclen(gmap)
        tot, n = 0.0, 0
        for imgs, labels in probe_batches:
            logits = model(imgs)
            tot += float(loss_fn(logits, labels)) * labels.numel()
            n += labels.numel()
        return tot / max(n, 1)

    was_enabled = sc_enabled()
    set_sc_enabled(True)          # the probe is meaningless on an FP trajectory
    try:
        # Empty map => every module falls back to its own stoc_len, which
        # _build_sc_model already pinned to the deployment average.
        L0 = mean_loss({})
        print(f"[measure] baseline L0={L0:.5f} (uniform sl={baseline_sl}, "
              f"probe sl={probe_sl}, {len(probe_batches)} batches)", flush=True)
        weights = {}
        for op in ops:
            for lb in range(measure_buckets):
                blocks = [b for b in range(total_blocks)
                          if _bucket_index(b, total_blocks, measure_buckets) == lb]
                if not blocks:
                    continue
                Lg = mean_loss({(op, b): probe_sl for b in blocks})
                weights[(op, lb)] = Lg - L0
                print(f"[measure] {op:<10} bucket{lb}: dL={Lg - L0:+.5f} "
                      f"(blocks={len(blocks)})", flush=True)
    finally:
        set_group_stoclen(None)
        set_sc_enabled(was_enabled)
    return weights, L0


@torch.no_grad()
def _collect_channel_act_scales(model, batches, run_forward) -> dict:
    """Per-input-channel activation amax for every SCLinear, over a few batches.

    This is the ViT stand-in for scmp_llm's cached SmoothQuant ``act_scales``:
    the same statistic (per-channel |x| amax), measured directly instead of
    loaded, because vit_sc has no SmoothQuant front-end. SC is bypassed for the
    pass so the statistic describes the FP activation distribution rather than
    one polluted by SC noise.
    """
    scales: dict = {}

    def hook(module, inputs, _output):
        x = inputs[0]
        a = x.reshape(-1, x.shape[-1]).abs().amax(dim=0).float()
        key = (module.operator, module.block_idx)
        prev = scales.get(key)
        scales[key] = a.cpu() if prev is None else torch.maximum(prev, a.cpu())

    handles = [m.register_forward_hook(hook)
               for m in model.modules() if isinstance(m, SCLinear)]
    set_sc_enabled(False)
    try:
        for batch in batches:
            run_forward(model, batch)
    finally:
        set_sc_enabled(True)
        for h in handles:
            h.remove()
    return scales


def _select_protected_channels(model, act_scales, frac, metric):
    """Top-``frac`` salient input channels per SCLinear -> {(op, block): idx}.

    ``act_weight`` scores a channel by ``a_c^2 * ||W[:, c]||^2`` — its
    contribution to the output energy, the SmoothQuant/AWQ salience proxy.
    ``act`` scores by the activation scale alone, which is the SpQR/LLM.int8()
    view: what matters is not that the channel carries energy but that it SETS
    the shared per-row quantization scale, so splitting it off shrinks the step
    for every OTHER channel in the row. On ViT that is the ``mlp_fc2`` story —
    post-GELU rows carry sparse large outliers over a near-zero bulk.

    Returns ``({}, {})`` when ``frac <= 0``.
    """
    if frac <= 0.0:
        return {}, {}
    if not 0.0 < frac < 1.0:
        raise SystemExit(f"[calib] --protect-channel-frac must be in (0, 1), got {frac}")
    indices, dims = {}, {}
    for m in model.modules():
        if not isinstance(m, SCLinear) or m.operator is None or m.block_idx is None:
            continue
        key = (m.operator, m.block_idx)
        w = m.weight.detach().float()
        D = int(w.shape[1])
        n_keep = int(round(frac * D))
        if n_keep <= 0:
            continue
        a = act_scales.get(key)
        if a is None:
            raise SystemExit(f"[calib] no activation scale collected for {key}")
        a = a.to(w.device).float()
        if metric == "act":
            score = a
        elif metric == "act_weight":
            score = a.pow(2) * w.pow(2).sum(dim=0)
        elif metric == "weight":
            score = w.pow(2).sum(dim=0)
        else:
            raise SystemExit(f"[calib] unknown --protect-channel-metric {metric!r}")
        top = torch.topk(score, n_keep).indices.sort().values
        indices[key] = [int(i) for i in top.cpu().tolist()]
        dims[key] = D
    return indices, dims


def _thresholds_from_counts(metrics: np.ndarray, counts: np.ndarray) -> list:
    if metrics.size == 0:
        return []
    sorted_metrics = np.sort(metrics)[::-1]
    thresholds: list = []
    offset = 0
    for count in counts[:-1]:
        offset += int(count)
        if offset <= 0:
            thresholds.append(1.0)
        elif offset >= len(sorted_metrics):
            thresholds.append(0.0)
        else:
            thresholds.append(
                float(0.5 * (sorted_metrics[offset - 1] + sorted_metrics[offset]))
            )
    return thresholds


# =====================================================================
# Calibrator (slim act_global port; W_g from --cross-layer-weight)
# =====================================================================


class ThresholdCalibrator:
    def __init__(
        self,
        levels: list,
        operators: Iterable[str],
        total_blocks: int,
        layer_buckets: int,
        budget_ratio: float,
        budget_ref_stoc_len: Optional[int],
        max_units_per_call: int,
        min_bucket_units: int,
        rng_seed: int,
        budget_scope: str = "global",
        budget_weight: str = "macs",
        metric_select: str = "amax",
        metric_select_margin: float = 0.05,
        argmin_pricing: str = "rep",
        objective: str = "sigma",
        cross_layer_weight: str = "uniform",
        group_ladder_k: int = 0,
        escape_gate_k: Optional[float] = None,
        escape_stoc_len: int = 0,
    ):
        # Escape gate. Deployed by the WRAPPER, but the calibrator has to know
        # about it: the gate hands rows to a high rung AFTER the Lagrangian has
        # allocated, so an ungated solve overshoots the budget by whatever the
        # gate takes. Measured on ViT that is 13-23% of rows, i.e. not a
        # rounding error. See _gated_mac_avg.
        self.escape_gate_k = (float(escape_gate_k)
                              if escape_gate_k is not None else None)
        self.escape_stoc_len = int(escape_stoc_len or max(levels))
        # Per-group ladders: 0 = off (one shared ladder, the historical
        # behaviour); k>0 = calibrate sigma on the dense --mp_levels ladder but
        # deploy only the k rungs each (op x layer-bucket) actually uses.
        self.group_ladder_k = int(group_ladder_k)
        # Protected input channels: {(op, block): [channel indices]} run at a
        # fixed high stream length for every row, split off along the
        # contraction dim. Empty unless --protect-channel-frac > 0.
        self.protected_channel_indices: dict = {}
        self.protect_channel_frac = 0.0
        self.protect_channel_stoc_len = 0
        self.protect_channel_metric = "none"
        self.protect_compensate_budget = False
        self.protected_channel_dims: dict = {}
        # Allocation objective currency (mp_objectives.objective_errors) and the
        # per-group importance weight W_g. Defaults reproduce act_global exactly.
        self.objective = objective
        self.cross_layer_weight = cross_layer_weight
        self._w_norm: Optional[float] = None
        # Phase-2 measured sensitivity, normalized to mean 1 and expanded to
        # per-BLOCK so it can be averaged onto any table bucketing.
        # {(operator, block_idx): W} — empty unless a measured probe ran.
        self.block_weights: dict = {}
        self.measure_meta: dict = {}
        # act_global_v2 dispatch-metric selection: "amax" (legacy, byte-
        # identical) or "auto" (capture l2/crest candidates, pick per operator
        # the signed candidate with best |Spearman ρ| vs σ-benefit).
        self.metric_select = metric_select
        self.metric_select_margin = metric_select_margin
        self.dispatch_metric_selection: dict = {}
        # act_global_v2 argmin cost pricing: "rep" (legacy rep_g = R_g/n_g,
        # includes the subsample ratio) or "macs" (KKT-consistent mac_per_row
        # only; use with --budget-weight macs + global scope).
        self.argmin_pricing = argmin_pricing
        self.levels = levels
        self.operators = set(operators)
        self.total_blocks = total_blocks
        self.timestep_buckets = 1
        self.layer_buckets = layer_buckets
        self.budget_ratio = budget_ratio
        self.budget_ref_stoc_len = (int(budget_ref_stoc_len)
                                    if budget_ref_stoc_len else max(levels))
        self.max_units_per_call = max_units_per_call
        self.min_bucket_units = min_bucket_units
        self.budget_scope = budget_scope
        self.budget_weight = budget_weight
        self.costs = np.asarray(levels, dtype=np.float64)
        self.records: dict = defaultdict(
            lambda: {"metrics": [], "errors": [], "alts": defaultdict(list)})
        # True (un-subsampled) per-forward row / MAC counts per group. MACs
        # accumulate rows*macs_per_row per call, so groups that pool blocks
        # with different channel widths are priced by their true mean.
        self.true_counts: dict = defaultdict(int)
        self.true_macs: dict = defaultdict(float)
        self._rng = np.random.default_rng(rng_seed)

    def use_operator(self, op: str) -> bool:
        return op in self.operators

    def add(
        self,
        operator: str,
        block_idx: int,
        metric_norm: torch.Tensor,
        errors_by_level: list,
        mac_per_row: float,
        alts: Optional[dict] = None,
    ):
        if not self.use_operator(operator):
            return
        metrics = metric_norm.detach().float().reshape(-1).cpu().numpy()
        err = torch.stack(
            [e.detach().float().reshape(-1) for e in errors_by_level], dim=-1
        )
        errors = err.cpu().numpy()
        if metrics.size == 0:
            return
        # act_global_v2 candidate metrics (per-call NORMALIZED, aligned with the
        # amax rows/errors) captured only under --metric-select auto.
        alt_np = ({k: v.detach().float().reshape(-1).cpu().numpy()
                   for k, v in alts.items()}
                  if (self.metric_select == "auto" and alts) else {})
        true_size = int(metrics.size)
        if self.max_units_per_call > 0 and metrics.size > self.max_units_per_call:
            keep = self._rng.choice(metrics.size, self.max_units_per_call,
                                    replace=False)
            metrics = metrics[keep]
            errors = errors[keep]
            alt_np = {k: (v[keep] if v.size == true_size else v)
                      for k, v in alt_np.items()}
        l_bucket = _bucket_index(block_idx, self.total_blocks, self.layer_buckets)
        key = (operator, 0, l_bucket)
        self.records[key]["metrics"].append(metrics)
        self.records[key]["errors"].append(errors)
        for k, v in alt_np.items():
            self.records[key]["alts"][k].append(v)
        self.true_counts[key] += true_size
        self.true_macs[key] += true_size * float(mac_per_row)

    # -- budget weights ---------------------------------------------------
    def _weight_count(self, key, n_fallback=None) -> float:
        """Per-group budget weight R_g: rows or MACs per self.budget_weight."""
        n = (n_fallback if n_fallback is not None
             else sum(int(m.size) for m in self.records[key]["metrics"]))
        if self.budget_weight == "macs":
            return float(self.true_macs.get(key, float(n)))
        return float(self.true_counts.get(key, n))

    def _rep_for_key(self, key) -> float:
        n = sum(int(m.size) for m in self.records[key]["metrics"])
        return self._weight_count(key) / max(n, 1)

    def _rep_for_op(self, operator) -> float:
        n = R = 0.0
        for key, rec in self.records.items():
            if key[0] != operator:
                continue
            ng = sum(int(m.size) for m in rec["metrics"])
            n += ng
            R += self._weight_count(key, ng)
        return (R / max(n, 1)) if n else 1.0

    def _mac_per_row(self, key) -> float:
        """True MACs per row for a group = true_macs / true_rows (subsample-
        ratio-free, unlike rep_g = true_macs / n_stored)."""
        tc = float(self.true_counts.get(key, 0))
        return (float(self.true_macs.get(key, 0.0)) / tc) if tc > 0 else 1.0

    def _mac_per_row_op(self, operator) -> float:
        tm = tc = 0.0
        for key in self.records:
            if key[0] == operator:
                tm += float(self.true_macs.get(key, 0.0))
                tc += float(self.true_counts.get(key, 0))
        return (tm / tc) if tc > 0 else 1.0

    def _price_for_key(self, key) -> float:
        """Per-row cycle PRICE for the argmin. 'macs' = mac_per_row (KKT); 'rep'
        = rep_g = R_g/n_g (legacy, includes the subsample ratio)."""
        if self.argmin_pricing == "macs":
            return self._mac_per_row(key)
        return self._rep_for_key(key)

    def _price_for_op(self, operator) -> float:
        if self.argmin_pricing == "macs":
            return self._mac_per_row_op(operator)
        return self._rep_for_op(operator)

    # -- cross-layer importance weight W_g ---------------------------------
    def _weight_norm(self) -> float:
        """Scale so max(W_g) == 1. A GLOBAL scale on W_g is absorbed by lambda
        (the bisection just lands elsewhere), so this only keeps the exported
        global_lambda in a readable range — it changes no assignment."""
        if self._w_norm is None:
            vals = [self._mac_per_row(k) for k in self.records]
            self._w_norm = max(max(vals), 1e-12) if vals else 1.0
        return self._w_norm

    def _weight_for_key(self, key) -> float:
        """Per-group importance weight W_g for the global cross-layer solve.

        ``uniform`` (act_global, and scmp_llm's default) is W_g = 1: the
        objective is then Sigma over STORED rows of sigma while the cost side is
        MAC-weighted, so an operator that is cheap per row buys error reduction
        cheaply and the Lagrangian pins it at the ladder ceiling. Measured on
        DINOv2 ViT-L/14 (MP_REPORT.md): ``out_proj`` — 1.05 M MACs/row, 8 % of
        SC MACs — sits at 60 of a possible 64 cycles while ``mlp_fc1`` (4.19 M
        MACs/row, 32 % of MACs) is starved to 22, below its own quality cliff,
        and MP goes NEGATIVE against uniform SC at that budget.

        ``macs`` sets W_g = MACs/row, putting benefit and cost in one currency:
        minimize MAC-weighted Sigma sigma subject to MAC-weighted cycles <= B.
        Because ``rep_g`` already carries the same MACs/row factor, the per-row
        argmin then reduces to ``sigma + lambda' * cycles`` for EVERY operator —
        a single global cycles-per-sigma exchange rate, with no cost-side
        favouritism toward narrow layers. Rows still compete on their own sigma
        curves, so a genuinely fragile operator can still win budget.

        ``measured`` / ``macs_measured`` fill the same slot from the phase-2
        knock-down probe (see ``_measure_group_sensitivity``). Note that
        ``measured`` ALONE re-introduces the cost-side favouritism ``macs``
        removes: ``rep_g`` already carries a MACs/row factor, so the per-row
        argmin is ``(W_g / mac_g)·σ + λ′·cycles``, and only ``W_g ∝ mac_g``
        cancels it. ``macs_measured`` = ``mac_g × ΔL̂_g`` therefore keeps the
        currency aligned AND adds the measured tilt; plain ``measured`` is kept
        for ablation only.
        """
        clw = self.cross_layer_weight
        if clw == "macs":
            return self._mac_per_row(key) / self._weight_norm()
        if clw in ("measured", "macs_measured"):
            w = self._measured_for_key(key)
            if clw == "measured":
                return w
            return (self._mac_per_row(key) / self._weight_norm()) * w
        return 1.0

    def _measured_for_key(self, key) -> float:
        """Mean measured W over the blocks this table bucket covers.

        The probe runs on its own (coarser) bucketing, so it is stored per block
        and averaged onto whatever bucketing the threshold table uses. Falls back
        to 1.0 when the probe never covered this operator.
        """
        operator, _t, l_bucket = key
        vals = [w for (op, b), w in self.block_weights.items()
                if op == operator
                and _bucket_index(b, self.total_blocks, self.layer_buckets) == l_bucket]
        return float(np.mean(vals)) if vals else 1.0

    def _op_weight(self, operator) -> float:
        """Mean W_g over an operator's buckets (for the operator_default fit)."""
        clw = self.cross_layer_weight
        if clw == "uniform":
            return 1.0
        mac = (self._mac_per_row_op(operator) / self._weight_norm()
               if clw in ("macs", "macs_measured") else 1.0)
        if clw in ("measured", "macs_measured"):
            vals = [w for (op, _b), w in self.block_weights.items() if op == operator]
            mac *= float(np.mean(vals)) if vals else 1.0
        return mac

    # -- diagnostics --------------------------------------------------------
    def metric_fidelity_rho(self) -> dict:
        """Spearman rho(metric, sigma-benefit) per operator — how well the
        dispatch metric ranks rows by their true need for cycles. Low/negative
        rho => a better metric (not more levels) is the lever."""
        per_op = defaultdict(lambda: {"m": [], "b": []})
        for key, rec in self.records.items():
            if not rec["metrics"]:
                continue
            m = np.concatenate(rec["metrics"], axis=0)
            eo = _objective_errors(
                np.concatenate(rec["errors"], axis=0), self.objective)
            benefit = eo[:, -1] - eo[:, 0]
            per_op[key[0]]["m"].append(m)
            per_op[key[0]]["b"].append(benefit)
        out = {}
        for op, d in per_op.items():
            m = np.concatenate(d["m"]); b = np.concatenate(d["b"])
            out[op] = {"rho": _spearman(m, b), "n": int(m.size),
                       "benefit_mean": float(b.mean())}
        return out

    def select_dispatch_metrics(self) -> dict:
        """act_global_v2 (--metric-select auto): per-operator SIGNED dispatch-
        metric selection. For each operator, pool all buckets and compute
        Spearman ρ(candidate, objective-benefit) for amax + every captured alt;
        pick the candidate with the largest |ρ| (negative ρ ⇒ deploy with
        sign −1, which inverts the ranking). Switch away from raw amax only when
        the winner clears the incumbent by ``metric_select_margin`` (noise
        guard). The benefit is measured in the SAME currency the allocator uses
        (``--objective``), so metric selection and allocation never disagree
        about what a row gains from more cycles.

        On a switch the stored per-bucket metric arrays are REWRITTEN to the
        selected candidate (sign −1 ⇒ 1−m, exactly what the runtime's negated-
        metric min–max normalization produces), so thresholds / summaries / the
        post-selection ρ all reflect the deployed metric. The choice is exported
        via payload["dispatch_metrics"]. Assignments/budget are untouched — the
        metric only maps counts to thresholds."""
        if self.metric_select != "auto":
            return {}
        per_op: dict = defaultdict(lambda: defaultdict(lambda: {"m": [], "b": []}))
        for key, rec in self.records.items():
            if not rec["metrics"]:
                continue
            e = _objective_errors(
                np.concatenate(rec["errors"], axis=0), self.objective)
            b = e[:, -1] - e[:, 0]                 # obj(min L) − obj(max L) ≥ 0
            cands = {"amax": np.concatenate(rec["metrics"], axis=0)}
            for name, lst in rec["alts"].items():
                if lst:
                    cands[name] = np.concatenate(lst, axis=0)
            for name, m in cands.items():
                if m.shape[0] != b.shape[0]:
                    # Partial capture — never kill a long calibration over a
                    # diagnostic candidate; amax always stays available.
                    print(f"[calib] WARN metric-select: candidate '{name}' "
                          f"misaligned for group {key} "
                          f"({m.shape[0]} vs {b.shape[0]} rows) — skipped.")
                    continue
                per_op[key[0]][name]["m"].append(m)
                per_op[key[0]][name]["b"].append(b)
        chosen: dict = {}
        for op, cands in per_op.items():
            rhos = {name: _spearman(np.concatenate(d["m"]),
                                    np.concatenate(d["b"]))
                    for name, d in cands.items()}
            base = float(rhos.get("amax", 0.0))
            best_name = max(rhos, key=lambda n: abs(rhos[n]))
            best_rho = float(rhos[best_name])
            sign = 1.0 if best_rho >= 0.0 else -1.0
            if best_name == "amax" and sign > 0.0:
                continue                            # incumbent stands
            if abs(best_rho) < max(base, 0.0) + self.metric_select_margin:
                continue                            # not clearly better
            if best_name != "amax":
                # A switched candidate must cover EVERY stored row of the op.
                covered = all(
                    sum(int(a.size) for a in rec["alts"][best_name])
                    == sum(int(m.size) for m in rec["metrics"])
                    for k, rec in self.records.items() if k[0] == op)
                if not covered:
                    print(f"[calib] WARN metric-select: '{best_name}' wins on "
                          f"{op} but has partial capture — keeping amax.")
                    continue
            for k, rec in self.records.items():
                if k[0] != op:
                    continue
                src = (rec["metrics"] if best_name == "amax"
                       else rec["alts"][best_name])
                rec["metrics"] = [(1.0 - m) if sign < 0.0 else m for m in src]
            chosen[op] = {"metric": best_name, "sign": sign,
                          "rho_signed": abs(best_rho), "rho_amax": base}
        self.dispatch_metric_selection = chosen
        return chosen

    def _gated_mac_avg(self, lam: float, ladder_keep: dict) -> float:
        """MAC-weighted mean stream length INCLUDING escaped rows.

        Reproduces exactly what the runtime does: per group, take the
        Lagrangian argmin on that group's own ladder, then override every row
        whose normalized metric exceeds ``mu_g + k*sigma_g`` to
        ``escape_stoc_len``. mu/sigma are the same per-bucket statistics
        ``_fit_group`` exports and ``AdaptiveMPConfig`` reads back, so the
        threshold here IS the deployed threshold.
        """
        num = den = 0.0
        for key in self.records:
            metrics = np.concatenate(self.records[key]["metrics"], axis=0)
            errors = np.concatenate(self.records[key]["errors"], axis=0)
            if metrics.size == 0:
                continue
            err_obj = _objective_errors(errors, self.objective)
            costs = self.costs
            keep = ladder_keep.get(key)
            if keep is not None:
                costs = self.costs[keep]
                err_obj = err_obj[:, keep]
            assign = ((self._weight_for_key(key) * err_obj)
                      + lam * self._price_for_key(key) * costs[None, :]
                      ).argmin(axis=1)
            cost_row = costs[assign].astype(np.float64)
            t_esc = float(metrics.mean()) + self.escape_gate_k * float(metrics.std())
            if t_esc < 1.0:
                cost_row = np.where(metrics > t_esc,
                                    float(self.escape_stoc_len), cost_row)
            R = self._weight_count(key, metrics.size)
            num += R * float(cost_row.mean())
            den += R
        return (num / den) if den else 0.0

    def _global_protected_fraction(self) -> float:
        """MAC-weighted share of the model that runs on protected channels.

        Protected channels are a slice of the CONTRACTION dim, so a module with
        ``n_prot`` of ``D`` input channels spends ``n_prot/D`` of its MACs at
        the protected stream length. Weighting by each group's true MACs turns
        the per-module fractions into the one number the budget compensation
        needs.
        """
        num = den = 0.0
        for key, macs in self.true_macs.items():
            op, _t, l_bucket = key
            fracs = [len(v) / max(self.protected_channel_dims.get((o, b), 1), 1)
                     for (o, b), v in self.protected_channel_indices.items()
                     if o == op
                     and _bucket_index(b, self.total_blocks,
                                       self.layer_buckets) == l_bucket]
            if not fracs:
                den += float(macs)
                continue
            num += float(macs) * (sum(fracs) / len(fracs))
            den += float(macs)
        return (num / den) if den > 0 else 0.0

    # -- solve + export ------------------------------------------------------
    def _fit_group(self, metrics: np.ndarray, errors: np.ndarray,
                   lam: Optional[float] = None, rep: float = 1.0,
                   weight: float = 1.0,
                   keep: Optional[np.ndarray] = None) -> dict:
        # The ASSIGNMENT is solved in the objective currency; every reported
        # error stays raw relative-L2 so summaries stay comparable across
        # --objective settings.
        err_obj = _objective_errors(errors, self.objective)
        # ``keep`` restricts this group to its OWN ladder (per-group ladders).
        # Slicing costs AND errors together keeps rung r's sigma attached to
        # rung r's cycle count; the exported "stoc_len_levels" then tells the
        # runtime which rungs these thresholds were fitted against.
        costs = self.costs
        levels = self.levels
        if keep is not None:
            costs = self.costs[keep]
            levels = [self.levels[int(i)] for i in keep]
            err_obj = err_obj[:, keep]
            errors = errors[:, keep]
        if lam is None:
            budget_total = (self.budget_ratio * self.budget_ref_stoc_len
                            * metrics.size)
            assignment = _cost_assignments(err_obj, costs, budget_total)
        else:
            objective = (weight * err_obj) + lam * rep * costs[None, :]
            assignment = objective.argmin(axis=1)
        counts = np.bincount(assignment, minlength=len(levels))
        thresholds = _thresholds_from_counts(metrics, counts)
        avg_cost = float(costs[assignment].mean()) if assignment.size else 0.0
        avg_error = (float(errors[np.arange(errors.shape[0]), assignment].mean())
                     if assignment.size else 0.0)
        out = {
            "num_units": int(metrics.size),
            "counts": counts.tolist(),
            "fractions": (counts / max(metrics.size, 1)).tolist(),
            "thresholds": thresholds,
            "avg_stoc_len": avg_cost,
            "avg_error": avg_error,
            "level_mean_error": [float(errors[:, i].mean())
                                 for i in range(errors.shape[1])],
            "metric_mean": float(metrics.mean()) if metrics.size else 0.0,
            "metric_std": float(metrics.std()) if metrics.size else 0.0,
        }
        if keep is not None:
            out["stoc_len_levels"] = [int(x) for x in levels]
        return out

    def export(self):
        summary_rows: list = []
        payload = {
            "stoc_len_levels": self.levels,
            "budget_ratio": self.budget_ratio,
            "budget_ref_stoc_len": self.budget_ref_stoc_len,
            "timestep_buckets": self.timestep_buckets,
            "layer_buckets": self.layer_buckets,
            "budget_scope": self.budget_scope,
            "budget_weight": self.budget_weight,
            "argmin_pricing": self.argmin_pricing,
            "objective": self.objective,
            "cross_layer_weight": self.cross_layer_weight,
            "operator_defaults": {},
            "buckets": {},
        }
        if self.protected_channel_indices:
            # Protected channels are charged at protect_stoc_len for EVERY row,
            # so they raise the realized average above the nominal target
            # unless the residual budget is lowered to pay for them. p is the
            # MAC-weighted protected fraction; the residual target solves
            #   p*protect_sl + (1-p)*rest = target.
            p = self._global_protected_fraction()
            target = float(self.budget_ratio) * float(self.budget_ref_stoc_len)
            protected = {
                "stoc_len": int(self.protect_channel_stoc_len),
                "frac": float(self.protect_channel_frac),
                "metric": self.protect_channel_metric,
                "compensate_budget": bool(self.protect_compensate_budget),
                "global_mac_weighted_frac": float(p),
                "indices": {f"{op}:b{b}": [int(i) for i in v]
                            for (op, b), v in
                            sorted(self.protected_channel_indices.items())},
                "dims": {f"{op}:b{b}": int(v) for (op, b), v in
                         sorted(self.protected_channel_dims.items())},
            }
            if self.protect_compensate_budget and p > 0.0:
                rest = (target - p * float(self.protect_channel_stoc_len)) / max(
                    1.0 - p, 1e-12)
                if rest <= 0.0:
                    raise SystemExit(
                        f"[calib] protected-channel compensation produced a "
                        f"non-positive residual target {rest:.4f}: the "
                        f"protected fraction ({p:.4f}) at stoc_len "
                        f"{self.protect_channel_stoc_len} already exceeds the "
                        f"whole budget {target:.2f}. Lower "
                        f"--protect-channel-frac or --protect-channel-stoc-len.")
                protected["original_budget_ratio"] = float(self.budget_ratio)
                protected["residual_target_avg_stoc_len"] = float(rest)
                self.budget_ratio = float(rest) / float(self.budget_ref_stoc_len)
                protected["residual_budget_ratio"] = float(self.budget_ratio)
                payload["budget_ratio"] = self.budget_ratio
            payload["protected_channels"] = protected

        global_lam = None
        ladder_keep: dict = {}
        if self.budget_scope == "global":
            keys = list(self.records.keys())
            werrs, row_counts, prices = [], [], []
            for key in keys:
                rec = self.records[key]
                e = _objective_errors(
                    np.concatenate(rec["errors"], axis=0), self.objective)
                werrs.append(self._weight_for_key(key) * e)
                row_counts.append(self._weight_count(key, e.shape[0]))
                prices.append(self._price_for_key(key))
            global_lam = _global_lambda(
                werrs, self.costs, self.budget_ratio,
                float(self.budget_ref_stoc_len), row_counts, prices=prices)

            # ---- per-group ladders: pass 1 selects, pass 2 re-prices --------
            # Pass 1's lambda is solved against the DENSE ladder. Restricting a
            # group to k rungs removes options from its argmin, so its realized
            # cost moves and the budget no longer holds at that lambda. Pass 2
            # re-solves ONE shared lambda against the per-group ladders that
            # will actually be deployed, which is what puts the realized
            # MAC-weighted average back on target.
            if 0 < self.group_ladder_k < len(self.levels):
                group_costs = []
                for key, e in zip(keys, werrs):
                    raw = _objective_errors(
                        np.concatenate(self.records[key]["errors"], axis=0),
                        self.objective)
                    keep = _select_group_ladder(
                        raw, self.costs, global_lam,
                        self._price_for_key(key), self._weight_for_key(key),
                        self.group_ladder_k)
                    ladder_keep[key] = keep
                    group_costs.append(self.costs[keep])
                werrs2 = [w[:, ladder_keep[k]] for w, k in zip(werrs, keys)]
                global_lam = _global_lambda_multi(
                    werrs2, group_costs, self.budget_ratio,
                    float(self.budget_ref_stoc_len), row_counts, prices)
                payload["group_ladder_k"] = int(self.group_ladder_k)
                payload["group_ladders"] = {
                    f"{op}:l{l}": [int(self.levels[int(i)])
                                   for i in ladder_keep[(op, t, l)]]
                    for (op, t, l) in sorted(ladder_keep)}
            # ---- escape-gate budget compensation ---------------------------
            # The gate fires at RUNTIME, after the allocator has spent the
            # budget, so an ungated solve overshoots by whatever the gate
            # takes (measured: 13-23% of rows on ViT, worth ~+2.6% realized
            # cycles at k=2). Shrink the target until the GATED average lands
            # on the nominal budget, so a gated cell is iso-compute with an
            # ungated one instead of quietly buying accuracy with cycles.
            if self.escape_gate_k is not None:
                target = float(self.budget_ratio) * float(self.budget_ref_stoc_len)
                ratio = float(self.budget_ratio)
                gated = self._gated_mac_avg(global_lam, ladder_keep)
                payload["ungated_gated_avg_stoc_len"] = float(gated)
                for _ in range(8):
                    if gated <= 0 or abs(gated - target) <= 0.005 * target:
                        break
                    ratio *= target / gated
                    lam_try = _global_lambda(
                        werrs, self.costs, ratio,
                        float(self.budget_ref_stoc_len), row_counts,
                        prices=prices)
                    if ladder_keep:
                        lam_try = _global_lambda_multi(
                            [w[:, ladder_keep[k]] for w, k in zip(werrs, keys)],
                            [self.costs[ladder_keep[k]] for k in keys],
                            ratio, float(self.budget_ref_stoc_len),
                            row_counts, prices)
                    global_lam = lam_try
                    gated = self._gated_mac_avg(global_lam, ladder_keep)
                self.budget_ratio = ratio
                payload["budget_ratio"] = ratio
                payload["escape_gate_k"] = self.escape_gate_k
                payload["escape_stoc_len"] = int(self.escape_stoc_len)
                payload["escape_compensated_budget_ratio"] = float(ratio)
                payload["expected_gated_avg_stoc_len"] = float(gated)
                print(f"[calib] escape gate k={self.escape_gate_k}: "
                      f"compensated budget_ratio -> {ratio:.4f}, "
                      f"gated MAC-avg {gated:.2f} (target {target:.2f})",
                      flush=True)

            payload["global_lambda"] = float(global_lam)
            if self.cross_layer_weight != "uniform":
                payload["group_weights"] = {
                    f"{op}:l{l}": self._weight_for_key((op, t, l))
                    for (op, t, l) in sorted(self.records)}

        # Operator-level fallback (concat all layer buckets per op).
        for operator in sorted(self.operators):
            op_metrics, op_errors = [], []
            for (op, _, _), rec in self.records.items():
                if op != operator:
                    continue
                op_metrics.extend(rec["metrics"])
                op_errors.extend(rec["errors"])
            if op_metrics:
                op_err = np.concatenate(op_errors, axis=0)
                op_keep = None
                if ladder_keep:
                    # The operator default is the fallback for any bucket the
                    # table has no entry for, so it gets its own pooled ladder
                    # rather than inheriting the dense one — otherwise a bucket
                    # miss would silently deploy k dense rungs where every real
                    # bucket deploys k selected ones.
                    op_keep = _select_group_ladder(
                        _objective_errors(op_err, self.objective), self.costs,
                        global_lam, self._price_for_op(operator),
                        self._op_weight(operator), self.group_ladder_k)
                fitted = self._fit_group(
                    np.concatenate(op_metrics, axis=0), op_err,
                    lam=global_lam, rep=self._price_for_op(operator),
                    weight=self._op_weight(operator), keep=op_keep,
                )
                payload["operator_defaults"][operator] = fitted
                summary_rows.append({
                    "scope": "operator_default", "operator": operator,
                    "t_bucket": -1, "l_bucket": -1,
                    **_flatten_summary(fitted),
                })

        # Per-bucket entries.
        for key in sorted(self.records.keys()):
            metrics = np.concatenate(self.records[key]["metrics"], axis=0)
            errors = np.concatenate(self.records[key]["errors"], axis=0)
            if metrics.size < self.min_bucket_units:
                continue
            operator, t_bucket, l_bucket = key
            fitted = self._fit_group(metrics, errors, lam=global_lam,
                                     rep=self._price_for_key(key),
                                     weight=self._weight_for_key(key),
                                     keep=ladder_keep.get(key))
            bucket_key = f"{operator}:t{t_bucket}:l{l_bucket}"
            payload["buckets"][bucket_key] = fitted
            summary_rows.append({
                "scope": "bucket", "operator": operator,
                "t_bucket": t_bucket, "l_bucket": l_bucket,
                **_flatten_summary(fitted),
            })

        # Predicted realized averages. Row-weighted (diagnostic) + MAC-weighted
        # (the true iso-compute check — must land near budget_ratio*ref).
        num = den = fnum = fden = 0.0
        for bkey, fitted in payload["buckets"].items():
            operator, _t, lpart = bkey.split(":")
            l = int(lpart[1:])
            key = (operator, 0, l)
            R = float(self.true_counts.get(key, fitted["num_units"]))
            Rm = float(self.true_macs.get(key, R))
            avg = float(fitted["avg_stoc_len"])
            num += R * avg
            den += R
            fnum += Rm * avg
            fden += Rm
        payload["expected_avg_stoc_len"] = (num / den) if den else 0.0
        payload["expected_flop_avg_stoc_len"] = (fnum / fden) if fden else 0.0
        # Informational: mean MACs/row per bucket (static for CNNs).
        payload["mac_per_row"] = {
            f"{op}:l{l}": self.true_macs[(op, 0, l)] / max(self.true_counts[(op, 0, l)], 1)
            for (op, _t, l) in self.records
        }
        return payload, summary_rows


def _flatten_summary(fitted: dict) -> dict:
    row = {
        "num_units": fitted["num_units"],
        "avg_stoc_len": fitted["avg_stoc_len"],
        "avg_error": fitted["avg_error"],
        "metric_mean": fitted["metric_mean"],
        "metric_std": fitted["metric_std"],
    }
    row["counts"] = ",".join(str(int(x)) for x in fitted["counts"])
    row["fractions"] = ",".join(f"{float(x):.6f}" for x in fitted["fractions"])
    row["thresholds"] = ",".join(f"{float(x):.6f}" for x in fitted["thresholds"])
    row["level_mean_error"] = ",".join(
        f"{float(x):.6f}" for x in fitted["level_mean_error"])
    return row


def _write_summary_csv(path: str, rows: list):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =====================================================================
# Hooks
# =====================================================================




def make_calib_attention_forward(calibrator, levels, sc_prec, halve,
                                 deploy_stoc_len, block_idx,
                                 qk_granularity="per_head"):
    """Attention forward that records qk / av calibration data as it runs.

    Granularity mirrors ``model/sc_vit.py``'s RUNTIME dispatch exactly, which is
    the whole point — scmp_llm's qk branch warns that calibrating at one
    granularity and dispatching at another mis-prices the operator in the
    budget by ~N. Here:

    * **qk unit = one attention head**, pooled over the batch, because
      ``_classify_heads`` computes ``|q|.amax(dim=(0,2,3))`` and buckets HEADS,
      and the kernel quantizes QK at ``granularity="per_head"``. MACs per unit
      = ``N*N*D`` per image.
    * **av unit = one attention row**, pooled GLOBALLY over ``B*H*N`` to match
      ``_sc_av``'s global normalization. MACs per unit = ``N*D`` per image.

    Both conventions agree with the linear path (units per image x MACs per
    unit reproduces the measured per-block MACs), so the MAC-weighted budget
    stays consistent across all six operators.

    The forward RETURNS the SC result at ``deploy_stoc_len`` so downstream
    blocks see the deployed noisy trajectory, matching ``--calib-on-sc``.
    """
    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        H = self.num_heads
        D = C // H
        q, k, v = torch.unbind(self.qkv(x).reshape(B, N, 3, H, D), 2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        qf = q.float().reshape(B * H, N, D).contiguous()
        kf = k.float().reshape(B * H, N, D).contiguous()

        with torch.no_grad():
            teacher_qk = torch.matmul(q.float(), k.float().transpose(-2, -1))

            def sc_qk(sl):
                return sc_matmul(qf, kf, granularity=qk_granularity,
                                 mode="bipolar", group_a=1, group_b=1,
                                 sc_prec=sc_prec, stoc_len=int(sl),
                                 halve_bipolar_stoc_len=halve
                                 ).reshape(B, H, N, N)

            if calibrator.use_operator("qk") and qk_granularity == "per_row":
                # Per-QUERY-ROW unit, pooled globally over B*H*N — matches
                # _sc_qk's per_row dispatch. MACs per unit = N*D per image.
                q2 = qf.reshape(B * H * N, D)
                metric = _normalize_metric(q2.abs().amax(dim=-1))
                alts = (_metric_candidates(q2)
                        if calibrator.metric_select == "auto" else None)
                t2 = teacher_qk.reshape(B * H * N, N)
                tnorm = t2.norm(dim=-1).clamp_min(1e-8)
                errs = []
                for sl in levels:
                    s2 = (sc_qk(sl).reshape(B * H * N, N) if sl > 0
                          else torch.zeros_like(t2))
                    errs.append((s2 - t2).norm(dim=-1) / tnorm)
                calibrator.add("qk", block_idx, metric, errs,
                               mac_per_row=float(N) * D, alts=alts)
            elif calibrator.use_operator("qk"):
                # Per-head metric, pooled over batch — the runtime metric.
                metric = _normalize_metric(q.float().abs().amax(dim=(0, 2, 3)))
                errs = []
                tnorm = teacher_qk.permute(1, 0, 2, 3).reshape(H, -1
                                                              ).norm(dim=-1).clamp_min(1e-8)
                for sl in levels:
                    e = (sc_qk(sl) - teacher_qk if sl > 0 else -teacher_qk)
                    errs.append(e.permute(1, 0, 2, 3).reshape(H, -1).norm(dim=-1) / tnorm)
                calibrator.add("qk", block_idx, metric, errs,
                               mac_per_row=float(N) * N * D, alts=None)

            attn_sc = sc_qk(deploy_stoc_len).to(x.dtype) * self.scale
            attn = attn_sc.softmax(dim=-1)

            a3 = attn.float().reshape(B * H, N, N).contiguous()
            v3 = v.float().transpose(-1, -2).reshape(B * H, D, N).contiguous()
            teacher_av = torch.matmul(a3, v3.transpose(-2, -1))

            def sc_av(sl):
                return sc_matmul(a3, v3, granularity="per_row", mode="bipolar",
                                 sc_prec=sc_prec, group_a=1, group_b=1,
                                 stoc_len=int(sl), halve_bipolar_stoc_len=halve)

            if calibrator.use_operator("av"):
                # Per-row metric over ALL B*H*N rows (global pool = runtime).
                from model.sc_vit import _av_row_score
                rows2 = a3.reshape(B * H * N, N)
                # av's primary is the row amax, NOT the legacy peakiness score
                # (_av_row_score / AV_METRIC_KIND). Two reasons, both hard:
                #  * select_dispatch_metrics labels the primary "amax" whatever
                #    it is, so recording entropy there and letting the selector
                #    flip its sign would export {"metric": "amax", "sign": -1}
                #    and the runtime would deploy the row amax — a DIFFERENT
                #    ranking than the thresholds were cut on, silently.
                #  * entropy is not in scmp_kernels' compute_row_metric set, so
                #    the runtime cannot express it through a table at all.
                # Candidates (l2, crest) are on the same rows, so the selector
                # now has something to switch to: av measured rho = -0.18
                # against sigma-benefit on the first run with attention in the
                # table, i.e. it was ranking rows BACKWARDS.
                metric = _normalize_metric(rows2.abs().amax(dim=-1))
                alts = (_metric_candidates(rows2)
                        if calibrator.metric_select == "auto" else None)
                t2 = teacher_av.reshape(B * H * N, D)
                tnorm = t2.norm(dim=-1).clamp_min(1e-8)
                errs = []
                for sl in levels:
                    s2 = (sc_av(sl).reshape(B * H * N, D) if sl > 0
                          else torch.zeros_like(t2))
                    errs.append((s2 - t2).norm(dim=-1) / tnorm)
                calibrator.add("av", block_idx, metric, errs,
                               mac_per_row=float(N) * D, alts=alts)

            out = sc_av(deploy_stoc_len).reshape(B, H, N, D).to(x.dtype)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))

    return forward


def _make_linear_hook(calibrator, levels, sc_prec, halve, chunk_d):
    """Record, per SCLinear call: the per-row dispatch metric, the per-level SC
    reconstruction error, and the layer's static MACs/row.

    The teacher is recomputed from the layer's own input rather than read off
    ``output``, so the measurement is unaffected by whatever stoc_len the model
    happens to be patched at while calibrating.
    """
    def hook(module: SCLinear, inputs, output):
        op = module.operator
        block_idx = module.block_idx
        if op is None or block_idx is None or not calibrator.use_operator(op):
            return
        x = inputs[0]
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.shape[0] == 0:
            return
        with torch.no_grad():
            xf = x_flat.to(torch.float32).contiguous()
            w = module.weight.to(torch.float32).contiguous()
            bias = (module.bias.to(torch.float32)
                    if module.bias is not None else None)
            teacher = F.linear(xf, w, bias)

            # Protected channels: sigma must describe the computation that will
            # actually be DEPLOYED, i.e. protected columns pinned at
            # protect_sl and only the residual columns moving with the level.
            # Measuring sigma on the full-width matmul instead would price a
            # cost the runtime never pays, and would rank rows by an outlier
            # the dispatch no longer sees.
            prot = calibrator.protected_channel_indices.get((op, block_idx))
            if prot:
                pi = torch.as_tensor(prot, dtype=torch.long, device=xf.device)
                mask = torch.ones(xf.shape[1], dtype=torch.bool, device=xf.device)
                mask[pi] = False
                ri = mask.nonzero(as_tuple=True)[0]
                x_p, w_p = xf.index_select(1, pi).contiguous(), w.index_select(1, pi).contiguous()
                x_r, w_r = xf.index_select(1, ri).contiguous(), w.index_select(1, ri).contiguous()
                base = sc_matmul(
                    x_p, w_p, granularity="per_row", mode="bipolar",
                    sc_prec=sc_prec,
                    stoc_len=int(calibrator.protect_channel_stoc_len),
                    chunk_d=(chunk_d if (chunk_d > 0
                                         and x_p.shape[-1] > chunk_d) else 0),
                    halve_bipolar_stoc_len=halve)
            else:
                x_r, w_r, base = xf, w, None

            metric = _normalize_metric(x_r.abs().amax(dim=-1))
            alts = (_metric_candidates(x_r)
                    if calibrator.metric_select == "auto" else None)
            level_errors = []
            for sl in levels:
                if sl <= 0:
                    # Pruned rung: the residual matmul is not issued. With
                    # protected channels their fixed-price product still is,
                    # which is exactly what the runtime does.
                    sc_rows = (torch.zeros_like(teacher) if base is None
                               else base + (bias if bias is not None else 0))
                else:
                    sc_rows = sc_matmul(
                        x_r, w_r,
                        granularity="per_row", mode="bipolar",
                        sc_prec=sc_prec, stoc_len=int(sl),
                        chunk_d=(chunk_d if (chunk_d > 0
                                             and x_r.shape[-1] > chunk_d) else 0),
                        halve_bipolar_stoc_len=halve,
                    )
                    if base is not None:
                        sc_rows = sc_rows + base
                    if bias is not None:
                        sc_rows = sc_rows + bias
                level_errors.append(_relative_l2_rows(sc_rows, teacher))
            # MACs/row is charged on the RESIDUAL width: the protected columns
            # are a separate fixed-price term, already accounted for by the
            # budget compensation in export().
            mac_scale = float(x_r.shape[1]) / float(max(xf.shape[1], 1))
            calibrator.add(op, block_idx, metric, level_errors,
                           mac_per_row=module.mac_per_row * mac_scale,
                           alts=alts)
    return hook


# =====================================================================
# Args + main
# =====================================================================


def _expand_levels(value: str) -> list:
    levels = [int(x.strip()) for x in value.split(",") if x.strip()]
    if len(levels) < 2:
        raise ValueError(f"Need at least 2 stoc_len levels, got {levels}")
    for i in range(len(levels) - 1):
        if levels[i] <= levels[i + 1]:
            raise ValueError(f"Levels must be strictly descending, got {levels}")
    return levels


def _build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default=DEFAULT_ROOT)
    p.add_argument("--mp_levels", required=True,
                   help="Descending comma-separated stoc_len levels "
                        "(halved space when --halve 1), e.g. '128,96,64,48'.")
    p.add_argument("--budget_ratio", type=float, default=0.5,
                   help="Target avg stoc_len / budget_ref_stoc_len.")
    p.add_argument("--budget_ref_stoc_len", type=int, default=None,
                   help="Reference stoc_len. Defaults to max(mp_levels) — so "
                        "the ratio is relative to THIS ladder's top rung, not "
                        "an absolute scale.")
    p.add_argument("--sc_prec", type=int, default=8)
    p.add_argument("--halve", type=int, default=1,
                   help="halve_bipolar_stoc_len. Must match deployment.")
    p.add_argument("--chunk_d", type=int, default=128,
                   help="chunk_d for SC linear calls. Must match deployment; "
                        "0 collapses ViT-L accuracy (see model/sc_vit.py).")
    p.add_argument("--operators", default=DEFAULT_OPERATORS)
    p.add_argument("--sc_attention", type=int, default=1,
                   help="Run qk/av on SC during calibration so the activation "
                        "trajectory matches deployment. They are never given "
                        "an MP table either way.")
    p.add_argument("--attn_stoc_len", type=int, default=0,
                   help="Uniform stoc_len for the non-dispatchable ops "
                        "(qk/av). 0 -> the deployment average.")
    p.add_argument("--n_calib_images", type=int, default=256)
    p.add_argument("--calib_seed", type=int, default=1234,
                   help="Shuffle seed for the calibration subset. Kept "
                        "distinct from the eval seed (0) so calibration and "
                        "measurement do not draw the same images.")
    p.add_argument("--calib_offset", type=int, default=0,
                   help="Start offset into the shuffled val index.")
    p.add_argument("--eval_seed", type=int, default=0,
                   help="Seed the EVAL subset is drawn with. Those images are "
                        "removed from the calibration pool. Must match the "
                        "--seed passed to cls/eval.py.")
    p.add_argument("--eval_exclude_n", type=int, default=2048,
                   help="How many eval images to exclude (match --max_images "
                        "at eval time). 0 disables the exclusion.")
    p.add_argument("--batch_size", type=int, default=16,
                   help="MUST match the eval batch size: the runtime "
                        "classifier min-max normalizes the metric per call, so "
                        "calibration and deployment row pools must have the "
                        "same shape.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--layer_buckets", type=int, default=0,
                   help="0 (default) = one bucket per block.")
    p.add_argument("--max_units_per_call", type=int, default=512)
    p.add_argument("--min_bucket_units", type=int, default=64)
    p.add_argument("--budget-scope", dest="budget_scope",
                   choices=["per_bucket", "global"], default="global")
    p.add_argument("--group-ladder-k", dest="group_ladder_k", type=int,
                   default=0,
                   help="Per-group ladders. 0 (default) = off: every bucket "
                        "shares the --mp_levels ladder. k>0 = calibrate sigma "
                        "on the FULL (dense) --mp_levels ladder but deploy only "
                        "the k rungs each (op x layer-bucket) actually puts "
                        "mass on, then re-solve one shared lambda against those "
                        "ladders so the budget still holds. Use a dense "
                        "--mp_levels (8-12 rungs) with k=3 or 4.")
    p.add_argument("--escape-gate-k", dest="escape_gate_k", type=float,
                   default=None,
                   help="Absolute escape gate: rows whose NORMALIZED dispatch "
                        "metric exceeds mu_b + k*sigma_b run at "
                        "--escape-stoc-len regardless of the band thresholds. "
                        "mu/sigma are the per-bucket metric stats this "
                        "calibrator already exports, so the gate is a wrapper-"
                        "only setting. Unset (default) = off.")
    p.add_argument("--escape-stoc-len", dest="escape_stoc_len", type=int,
                   default=0,
                   help="Stream length escaped rows run at. 0 -> the ladder "
                        "ceiling (max of --mp_levels).")
    p.add_argument("--protect-channel-frac", dest="protect_channel_frac",
                   type=float, default=0.0,
                   help="Fraction of INPUT channels per linear to pin at "
                        "--protect-channel-stoc-len for every row, split off "
                        "along the contraction dim (exact: a matmul is "
                        "additive over that dim). 0 (default) = off. On ViT "
                        "the target is mlp_fc2, whose post-GELU rows carry "
                        "sparse outliers over a near-zero bulk.")
    p.add_argument("--protect-channel-metric", dest="protect_channel_metric",
                   choices=["act", "act_weight", "weight"], default="act",
                   help="Channel salience score. 'act' (default): the "
                        "per-channel activation amax — the channel that SETS "
                        "the shared per-row quantization scale. 'act_weight': "
                        "a_c^2*||W[:,c]||^2, the output-energy proxy.")
    p.add_argument("--protect-channel-stoc-len",
                   dest="protect_channel_stoc_len", type=int, default=0,
                   help="Stream length for protected channels. 0 -> the "
                        "ladder ceiling.")
    p.add_argument("--protect-channel-stat-images",
                   dest="protect_channel_stat_images", type=int, default=64,
                   help="Images used for the per-channel activation-amax "
                        "pre-pass that selects protected channels.")
    p.add_argument("--protect-compensate-budget",
                   dest="protect_compensate_budget", type=int, default=1,
                   help="1 (default): lower the residual per-row budget so the "
                        "MAC-weighted average still lands on the nominal "
                        "target once protected channels are charged. 0: let "
                        "protection overshoot the budget (NOT iso-compute).")
    p.add_argument("--qk_granularity", dest="qk_granularity",
                   choices=["per_head", "per_row"], default="per_head",
                   help="Quantization scope for Q@K^T, and hence the MP "
                        "dispatch unit. 'per_head' (legacy) gives only 16 "
                        "units per block on ViT-L/14. 'per_row' matches the "
                        "av convention and scmp_llm. MUST match the eval "
                        "invocation.")
    p.add_argument("--budget-weight", dest="budget_weight",
                   choices=["rows", "macs"], default="macs",
                   help="'macs' (default): iso-budget = iso-compute, the "
                        "shipped configuration. 'rows' reproduces the "
                        "row-weighted budget bug, for ablation only.")
    p.add_argument("--calib-on-sc", dest="calib_on_sc", type=int, default=1,
                   help="1 (default): calibration forward runs in SC at a "
                        "uniform deployment-average stoc_len, so metric and "
                        "sigma are measured on the noisy activation "
                        "distribution actually deployed. 0: FP trajectory "
                        "(ablation).")
    p.add_argument("--calib-sc-stoclen", dest="calib_sc_stoclen", type=int,
                   default=0,
                   help="stoc_len for the --calib-on-sc forward. 0 -> "
                        "round(budget_ratio * budget_ref_stoc_len).")
    p.add_argument("--metric-select", dest="metric_select",
                   default="amax", choices=["amax", "auto"],
                   help="act_global_v2 per-operator dispatch-metric selection: "
                        "'amax' (legacy, byte-identical) or 'auto' (pick the "
                        "signed amax/l2/crest candidate with best Spearman rho "
                        "vs sigma-benefit; exports 'dispatch_metrics').")
    p.add_argument("--metric-select-margin", dest="metric_select_margin",
                   type=float, default=0.05,
                   help="min |rho| improvement over amax required to switch.")
    p.add_argument("--argmin-pricing", dest="argmin_pricing",
                   default="rep", choices=["rep", "macs"],
                   help="Per-row cost pricing in the argmin. 'macs' was tested "
                        "and REJECTED on scmp_cnn (-17 pt with auto metrics); "
                        "keep 'rep' unless ablating. NOTE on a ViT this knob is "
                        "a NO-OP: every op sees the same rows/forward (257*B), "
                        "so rep_g and mac_per_row differ by one global constant "
                        "that lambda absorbs.")
    p.add_argument("--objective", dest="objective", default="sigma",
                   choices=["sigma", "sigma2", "delta_sigma2"],
                   help="Allocation objective (mp_objectives.py, shared with "
                        "scmp_llm). 'sigma' (default): minimize sum of "
                        "relative-L2 recon error — byte-identical to the "
                        "pre-2026-07-25 calibrator. 'sigma2': minimize sum of "
                        "sigma^2, a propagated-loss surrogate that penalizes "
                        "the low-precision cliff superlinearly. "
                        "'delta_sigma2': same, but priced on the INCREASE over "
                        "each row's error at the top rung, so irreducible error "
                        "at max stoc_len is not chased.")
    p.add_argument("--cross-layer-weight", dest="cross_layer_weight",
                   default="uniform",
                   choices=["uniform", "macs", "measured", "macs_measured"],
                   help="Per-group importance weight W_g for --budget-scope "
                        "global. 'uniform' (default): W_g=1, i.e. act_global — "
                        "benefit counted per ROW while cost is per MAC, which "
                        "pins the cheap out_proj at the ladder ceiling and "
                        "starves mlp_fc1 (MP_REPORT.md). 'macs': W_g=MACs/row, "
                        "so benefit and cost share one currency and every "
                        "operator faces the same cycles-per-sigma exchange "
                        "rate. 'macs_measured': phase-2 — that, times a measured "
                        "per-group dLoss from a knock-down probe. 'measured': "
                        "dLoss alone, which RE-INTRODUCES the cost-side "
                        "favouritism macs removes (rep_g already carries mac_g) "
                        "— ablation only. Implies --budget-scope global.")
    p.add_argument("--measure-marg-frac", dest="measure_marg_frac", type=float,
                   default=0.25,
                   help="Knock-down step for the measured probe, as a fraction "
                        "of the baseline stoc_len. Small ON PURPOSE: knocking a "
                        "group to the ladder floor saturates past the collapse "
                        "cliff, so dL stops being a derivative (scmp_llm's "
                        "measured vs measured_marg finding).")
    p.add_argument("--measure-layer-buckets", dest="measure_layer_buckets",
                   type=int, default=4,
                   help="Layer buckets for the PROBE only (cost is |ops| x this "
                        "+ 1 forwards). Independent of --layer_buckets, which "
                        "bucket the threshold table; the probe result is stored "
                        "per block and averaged onto the table's bucketing.")
    p.add_argument("--measure-images", dest="measure_images", type=int,
                   default=128,
                   help="Calibration images per probe forward.")
    p.add_argument("--variants", default="",
                   help="Export SEVERAL tables from ONE sigma collection, so an "
                        "objective/W_g comparison carries zero calibration-"
                        "sample noise. Comma-separated "
                        "'<objective>[:<uniform|macs>]' entries, or the literal "
                        "'all' for every combination. Each table is written to "
                        "<output_json stem>__<objective>_w<W_g>.json with its "
                        "own wrapper. Empty (default) = one table at "
                        "--output_json from --objective/--cross-layer-weight.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_summary_csv", default=None)
    return p



def main():
    args = _build_parser().parse_args()

    # ---- pure-config validation, BEFORE the CUDA gate ----
    # A bad flag combination should be reportable from the login node; nothing
    # here touches a device, so there is no reason to make the user burn a Slurm
    # allocation to find out the ladder or the scope is wrong.
    levels = _expand_levels(args.mp_levels)
    halve = bool(args.halve)
    sc_prec = int(args.sc_prec)
    halved_cap = 2 ** (sc_prec - 1) if halve else 2 ** sc_prec
    if max(levels) > halved_cap:
        raise ValueError(
            f"max(mp_levels)={max(levels)} exceeds cap {halved_cap} "
            f"for sc_prec={sc_prec} halve={halve}.")
    if (args.cross_layer_weight != "uniform"
            and args.budget_scope != "global"):
        raise SystemExit(
            f"--cross-layer-weight {args.cross_layer_weight} requires "
            f"--budget-scope global: the per_bucket solve pins every group to "
            f"the same average independently, so W_g cannot move budget across "
            f"operators and would be silently ignored.")

    if not torch.cuda.is_available():
        raise RuntimeError("Calibration requires CUDA (sc kernels are GPU-only).")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    seed_all(args.seed)

    calib_on_sc = bool(args.calib_on_sc)
    ref_sl = args.budget_ref_stoc_len or max(levels)
    calib_sc_sl = int(args.calib_sc_stoclen) or max(
        1, int(round(args.budget_ratio * ref_sl)))
    if not args.attn_stoc_len:
        args.attn_stoc_len = calib_sc_sl

    # Build the SC model, then decide whether the forward TRAJECTORY is SC or
    # FP. The modules stay patched either way so the hooks still fire; only
    # what flows between blocks changes.
    model = _build_sc_model(args, device)
    set_sc_enabled(calib_on_sc)

    total = total_blocks()
    if not total:
        raise RuntimeError("patch_model tagged no blocks — nothing to calibrate.")

    operators = {x.strip() for x in args.operators.split(",") if x.strip()}
    attn_ops = operators & {"qk", "av"}
    if attn_ops and not args.sc_attention:
        raise ValueError(
            f"--operators includes {sorted(attn_ops)} but --sc_attention 0; "
            f"attention must be SC-patched to be calibrated.")
    present = {m.operator for m in model.modules()
               if isinstance(m, SCLinear) and m.operator} | (
                   {"qk", "av"} if args.sc_attention else set())
    missing = operators - present
    if missing:
        raise ValueError(
            f"--operators names {sorted(missing)} but the patched model only "
            f"exposes {sorted(present)}.")

    layer_buckets = int(args.layer_buckets) or total
    print(f"[calib] blocks={total} operators={sorted(operators)} "
          f"layer_buckets={layer_buckets} levels={levels} "
          f"budget={args.budget_ratio:.4f}x{ref_sl}="
          f"{args.budget_ratio * ref_sl:.1f} "
          f"scope={args.budget_scope} weight={args.budget_weight} "
          f"objective={args.objective} W_g={args.cross_layer_weight} "
          f"halve={halve} sc_prec={sc_prec} metric={args.metric_select} "
          f"calib_on_sc={calib_on_sc}"
          + (f" calib_sc_stoclen={calib_sc_sl}" if calib_on_sc else ""),
          flush=True)

    loader, run_forward, calib_indices = _build_calib_iter(args, device)

    calibrator = ThresholdCalibrator(
        levels=levels,
        operators=operators,
        total_blocks=total,
        layer_buckets=layer_buckets,
        budget_ratio=args.budget_ratio,
        budget_ref_stoc_len=args.budget_ref_stoc_len,
        max_units_per_call=args.max_units_per_call,
        min_bucket_units=args.min_bucket_units,
        rng_seed=args.seed,
        budget_scope=args.budget_scope,
        budget_weight=args.budget_weight,
        metric_select=args.metric_select,
        metric_select_margin=args.metric_select_margin,
        argmin_pricing=args.argmin_pricing,
        objective=args.objective,
        cross_layer_weight=args.cross_layer_weight,
        group_ladder_k=args.group_ladder_k,
        escape_gate_k=args.escape_gate_k,
        escape_stoc_len=args.escape_stoc_len,
    )

    # ---- protected input channels (selected BEFORE the sigma pass) ---------
    # Order matters: the dispatch metric and every per-level sigma are measured
    # on the RESIDUAL channels, so the protected set has to be known first.
    if args.protect_channel_frac > 0.0:
        prot_sl = int(args.protect_channel_stoc_len or max(levels))
        if prot_sl > max(levels):
            raise SystemExit(
                f"[calib] --protect-channel-stoc-len {prot_sl} exceeds the "
                f"ladder ceiling {max(levels)}; protected channels would run "
                f"longer streams than any rung on the grid.")
        n_stat = max(1, int(round(args.protect_channel_stat_images
                                  / args.batch_size)))
        stat_batches = [b for i, b in enumerate(loader) if i < n_stat]
        t_p = time.time()
        act_scales = _collect_channel_act_scales(model, stat_batches, run_forward)
        idxs, dims = _select_protected_channels(
            model, act_scales, args.protect_channel_frac,
            args.protect_channel_metric)
        calibrator.protected_channel_indices = idxs
        calibrator.protected_channel_dims = dims
        calibrator.protect_channel_frac = float(args.protect_channel_frac)
        calibrator.protect_channel_stoc_len = prot_sl
        calibrator.protect_channel_metric = args.protect_channel_metric
        calibrator.protect_compensate_budget = bool(args.protect_compensate_budget)
        n_ch = sum(len(v) for v in idxs.values())
        print(f"[calib] protected channels: {len(idxs)} modules, {n_ch} channels "
              f"({args.protect_channel_frac:.1%} each) at stoc_len={prot_sl}, "
              f"metric={args.protect_channel_metric}, "
              f"compensate={bool(args.protect_compensate_budget)} "
              f"[{time.time() - t_p:.0f}s]", flush=True)
        del stat_batches, act_scales

    linear_hook = _make_linear_hook(calibrator, levels, sc_prec, halve,
                                    int(args.chunk_d))
    hooks = [m.register_forward_hook(linear_hook)
             for m in model.modules() if isinstance(m, SCLinear)]
    print(f"[calib] hooked {len(hooks)} SCLinear modules", flush=True)

    # qk / av are not nn.Linear, so they cannot be reached with a forward hook.
    # Swap in a calibration forward that records them at the SAME granularity
    # model/sc_vit.py dispatches at (qk per head, av per row pooled globally).
    if attn_ops:
        from model.sc_vit import _discover_blocks, _find_attn
        n_attn = 0
        for bi, blk in enumerate(_discover_blocks(model)):
            attn = _find_attn(blk)
            if attn is None:
                continue
            fwd = make_calib_attention_forward(
                calibrator, levels, sc_prec, halve, calib_sc_sl, bi,
                qk_granularity=args.qk_granularity)
            attn.forward = fwd.__get__(attn, type(attn))
            n_attn += 1
        print(f"[calib] installed calibration forward on {n_attn} attention "
              f"modules for {sorted(attn_ops)}", flush=True)

    t0 = time.time()
    try:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                run_forward(model, batch)
                print(f"[calib] batch {i + 1}/{len(loader)} "
                      f"({time.time() - t0:.1f}s)", flush=True)
    finally:
        for h in hooks:
            h.remove()

    # ---- solve + export, once per requested (objective, W_g) variant ----
    # The sigma collection above is INDEPENDENT of the objective and of W_g:
    # only the Lagrangian solve reads them. So N variants exported here share
    # ONE sigma draw, which removes the calibration-sample noise (~0.33 pt on
    # top-1 at n_calib=256, MP_REPORT.md) from any comparison between them —
    # every difference is attributable to the objective alone. Independent
    # calibrations per variant would confound the two.
    variants = _parse_variants(args, calibrator)

    # ---- phase 2: measured cross-layer sensitivity (only if a variant wants it)
    if any(w in ("measured", "macs_measured") for _o, w in variants):
        n_probe_batches = max(1, int(round(args.measure_images / args.batch_size)))
        probe_batches = []
        for i, batch in enumerate(loader):
            if i >= n_probe_batches:
                break
            imgs, labels = batch[0], batch[1]
            probe_batches.append((imgs.to(device, non_blocking=True),
                                  labels.to(device, non_blocking=True)))
        n_img = sum(int(l.numel()) for _x, l in probe_batches)
        probe_sl = max(1, int(round(calib_sc_sl * (1.0 - args.measure_marg_frac))))
        if probe_sl >= calib_sc_sl:
            probe_sl = max(1, calib_sc_sl - 1)
        mb = int(args.measure_layer_buckets) or 1
        print(f"\n[measure] phase-2 knock-down probe: {len(operators)} ops x "
              f"{mb} buckets + 1 = {len(operators) * mb + 1} forwards over "
              f"{n_img} images", flush=True)
        t_m = time.time()
        raw, L0 = _measure_group_sensitivity(
            model, probe_batches, operators, total, mb,
            baseline_sl=calib_sc_sl, probe_sl=probe_sl)
        norm = _normalize_group_weights(raw)
        calibrator.block_weights = {
            (op, b): norm[(op, lb)]
            for (op, lb) in norm
            for b in range(total) if _bucket_index(b, total, mb) == lb}
        calibrator.measure_meta = {
            "baseline_stoc_len": int(calib_sc_sl), "probe_stoc_len": int(probe_sl),
            "marg_frac": float(args.measure_marg_frac), "layer_buckets": mb,
            "n_images": n_img, "baseline_loss": float(L0),
            "raw_delta_loss": {f"{op}:l{lb}": float(v) for (op, lb), v in raw.items()},
            "normalized": {f"{op}:l{lb}": float(v) for (op, lb), v in norm.items()},
        }
        print(f"[measure] normalized W_g (mean 1): "
              + " ".join(f"{op}:l{lb}={v:.3f}" for (op, lb), v in sorted(norm.items()))
              + f"   [{time.time() - t_m:.0f}s]", flush=True)
        del probe_batches

    snapshot = {k: list(rec["metrics"]) for k, rec in calibrator.records.items()}
    out_root = Path(args.output_json)
    out_root.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for objective, clw in variants:
        # Restore the as-collected amax metrics: select_dispatch_metrics()
        # REBINDS rec["metrics"] to the chosen candidate, and the choice is
        # objective-dependent, so each variant must start from the raw draw.
        for k, metrics in snapshot.items():
            calibrator.records[k]["metrics"] = list(metrics)
        calibrator.objective = objective
        calibrator.cross_layer_weight = clw
        calibrator.dispatch_metric_selection = {}

        tag = f"{objective}_w{clw}"
        out_json = (out_root if len(variants) == 1
                    else out_root.with_name(f"{out_root.stem}__{tag}.json"))
        print(f"\n[calib] === variant objective={objective} W_g={clw} "
              f"-> {out_json.name} ===", flush=True)

        rho = calibrator.metric_fidelity_rho()
        print("[calib] metric-fidelity rho(metric, objective-benefit) per op "
              "(~1 good; low/neg => metric misranks -> headroom):")
        for op in sorted(rho, key=lambda o: rho[o]["rho"]
                         if rho[o]["rho"] == rho[o]["rho"] else 9):
            d = rho[op]
            print(f"    {op:<11} rho={d['rho']:+.3f}  n={d['n']:>8}  "
                  f"benefit_mean={d['benefit_mean']:.4f}")

        # act_global_v2: per-operator signed dispatch-metric selection. Rewrites
        # the stored metric to the deployed candidate so thresholds build on it.
        chosen = calibrator.select_dispatch_metrics()
        if chosen:
            print(f"[calib] metric-select auto -> switched metric on "
                  f"{len(chosen)} op(s):")
            for op in sorted(chosen):
                c = chosen[op]
                print(f"    {op:<11} metric={c['metric']:<5} "
                      f"sign={c['sign']:+.0f} |rho|={c['rho_signed']:.3f} "
                      f"(amax rho={c['rho_amax']:+.3f})")
            rho_post = calibrator.metric_fidelity_rho()
            print("[calib] post-selection rho(deployed metric, benefit):")
            for op in sorted(rho_post, key=lambda o: rho_post[o]["rho"]
                             if rho_post[o]["rho"] == rho_post[o]["rho"] else 9):
                d = rho_post[op]
                print(f"    {op:<11} rho={d['rho']:+.3f}  n={d['n']:>8}")
        elif calibrator.metric_select == "auto":
            print("[calib] metric-select auto -> no op cleared the margin; "
                  "keeping amax everywhere.")

        payload, summary_rows = calibrator.export()
        payload["metric_fidelity_rho"] = {op: rho[op]["rho"] for op in rho}
        if chosen:
            payload["dispatch_metrics"] = {
                op: {"metric": c["metric"], "sign": c["sign"]}
                for op, c in chosen.items()}
        payload["task"] = "cls"
        payload["model"] = "dinov2_vitl14_lc"
        payload["sc_prec"] = sc_prec
        payload["halve_bipolar_stoc_len"] = halve
        payload["operators"] = sorted(operators)
        payload["n_calib_images"] = args.n_calib_images
        payload["calib_batch_size"] = args.batch_size
        payload["total_blocks"] = int(total)
        payload["seed"] = int(args.seed)
        payload["calib_seed"] = int(args.calib_seed)
        payload["eval_excluded"] = {"seed": int(args.eval_seed),
                                    "n": int(args.eval_exclude_n)}
        payload["calib_indices"] = calib_indices
        payload["mp_levels"] = args.mp_levels
        payload["chunk_d"] = int(args.chunk_d)
        # Ops that consume budget but carry no MP table (see module docstring).
        payload["uniform_ops"] = {op: int(args.attn_stoc_len)
                                  for op in ("qk", "av")
                                  if op not in operators}
        method = "act_global" if args.budget_scope == "global" else "act"
        if args.budget_weight == "macs":
            method += "_fw"
        if args.argmin_pricing == "macs":
            method += "_kkt"
        if clw != "uniform":
            method += f"_w{clw}"
        if objective != "sigma":
            method += f"_{objective}"
        if calib_on_sc:
            method += "_onsc"
            payload["calib_on_sc"] = True
            payload["calib_sc_stoclen"] = calib_sc_sl
        if args.group_ladder_k > 0:
            method += f"_gl{args.group_ladder_k}"
        if args.protect_channel_frac > 0.0:
            method += f"_pc{args.protect_channel_frac:g}"
        if args.escape_gate_k is not None:
            method += f"_esc{args.escape_gate_k:g}"
        if args.qk_granularity != "per_head":
            method += f"_qk{args.qk_granularity}"
        payload["qk_granularity"] = args.qk_granularity
        payload["method"] = method
        payload["calib_command"] = " ".join(sys.argv)
        if len(variants) > 1:
            # One sigma draw, many tables: say so in the bundle itself, so a
            # cell can never be mistaken for an independent calibration.
            payload["variant"] = {"tag": tag, "shared_sigma_draw": True,
                                  "siblings": [f"{o}_w{c}" for o, c in variants]}
        if clw in ("measured", "macs_measured") and calibrator.measure_meta:
            payload["measured_probe"] = calibrator.measure_meta

        print(f"[calib] expected_avg_stoc_len (row-weighted) = "
              f"{payload['expected_avg_stoc_len']:.2f}")
        print(f"[calib] expected_flop_avg_stoc_len (iso-compute check) = "
              f"{payload['expected_flop_avg_stoc_len']:.2f} "
              f"(target {args.budget_ratio * calibrator.budget_ref_stoc_len:.2f})")

        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[calib] wrote {out_json}")

        # Deployment wrapper: what model.sc_vit.load_adaptive_mp() consumes.
        # threshold_table_path is stored RELATIVE so a calibration bundle can be
        # moved or copied between machines without rewriting it.
        wrapper = out_json.with_name(out_json.stem + "_wrapper.json")
        wrapper_spec = {
            "type": "AdaptiveMPConfig",
            "stoc_len_levels": levels,
            "threshold_table_path": out_json.name,
        }
        if args.escape_gate_k is not None:
            wrapper_spec["escape_gate_k"] = float(args.escape_gate_k)
            wrapper_spec["escape_stoc_len"] = int(args.escape_stoc_len
                                                  or max(levels))
        with open(wrapper, "w") as f:
            json.dump(wrapper_spec, f, indent=2)
        print(f"[calib] wrote {wrapper}  <- pass this to --adaptive_mp_table")

        summary_csv = (args.output_summary_csv if len(variants) == 1
                       else None) or (str(out_json.with_suffix("")) + "_summary.csv")
        _write_summary_csv(summary_csv, summary_rows)
        print(f"[calib] wrote {summary_csv}")
        written.append((tag, out_json, payload))

    if len(written) > 1:
        _print_variant_comparison(written, calibrator,
                                  args.budget_ratio * calibrator.budget_ref_stoc_len)


def _parse_variants(args, calibrator) -> list:
    """``--variants`` -> [(objective, cross_layer_weight), ...].

    Empty (default) yields the single pair from ``--objective`` /
    ``--cross-layer-weight``, so the one-table behaviour is unchanged.
    ``all`` expands to every (objective x W_g) combination.
    """
    spec = (args.variants or "").strip()
    if not spec:
        return [(args.objective, args.cross_layer_weight)]
    objectives = ["sigma", "sigma2", "delta_sigma2"]
    weights = ["uniform", "macs", "measured", "macs_measured"]
    if spec == "all":
        # 'all' stays the objective x {uniform, macs} grid: the measured weights
        # need a probe, so pulling them in silently would change the run's cost.
        return [(o, w) for w in ("uniform", "macs") for o in objectives]
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        obj, _, clw = item.partition(":")
        clw = clw or "uniform"
        if obj not in objectives or clw not in weights:
            raise SystemExit(
                f"--variants entry {item!r} must be "
                f"'<objective>[:<W_g>]' with objective in {objectives} and "
                f"W_g in {weights} (or the literal 'all').")
        if (obj, clw) not in out:
            out.append((obj, clw))
    if not out:
        raise SystemExit("--variants parsed to nothing")
    if any(w != "uniform" for _o, w in out) and args.budget_scope != "global":
        raise SystemExit("--variants with W_g=macs requires --budget-scope global")
    return out


def _print_variant_comparison(written: list, calibrator, target: float):
    """Per-operator MAC-weighted allocation across variants of ONE sigma draw.

    This is the cheap preview of what the eval will show: if two variants
    allocate the same cycles per operator, running both is wasted GPU time.
    """
    ops = sorted({k[0] for k in calibrator.records})
    hdr = f'{"variant":<26}' + "".join(f"{o:>11}" for o in ops) + f'{"realized":>10}'
    print(f"\n[calib] === allocation across variants (one shared sigma draw, "
          f"target {target:.1f}) ===")
    print(hdr)
    print("-" * len(hdr))
    for tag, _path, payload in written:
        per, num, den = {}, 0.0, 0.0
        for bkey, fit in payload["buckets"].items():
            op, _t, lpart = bkey.split(":")
            mac = payload["mac_per_row"][f"{op}:{lpart}"]
            w = fit["num_units"] * mac
            e = per.setdefault(op, [0.0, 0.0])
            e[0] += fit["avg_stoc_len"] * w
            e[1] += w
            num += fit["avg_stoc_len"] * w
            den += w
        row = f"{tag:<26}" + "".join(
            f'{per[o][0] / per[o][1]:>11.1f}' if o in per else f'{"-":>11}'
            for o in ops)
        print(row + f'{num / den if den else float("nan"):>10.2f}')


if __name__ == "__main__":
    main()
