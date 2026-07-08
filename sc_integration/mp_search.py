"""Backend-agnostic MP-budget swap search.

Finds a per-(op, block) ``stoc_len`` map under a fixed main-budget constraint
via local-proxy iterative pairwise swaps. This is the algorithmic core
ported from ``cls/experiments/mp_budget_swap_search.py``; the cls/det
drivers subclass :class:`MPSearchBackend` to plug in their model-specific
schedule loaders, model + loader builders, and per-block patching adapters.

Algorithm summary:

  1. **Initial allocation** (``build_initial_sl_map``): equal-MAC DP over
     searchable units assigns each unit a discrete level from a per-unit
     allowed-level set, picking the assignment whose stoc-len sum is closest
     to the budget while maximizing a sensitivity-weighted utility (or
     minimizing deviation from a user-provided baseline / saved sl_map).
  2. **Iterative pairwise swap** (``iterative_swap_search``): each round,
     score every ±1-step move per searchable unit on a local proxy
     (``raw_mse`` or ``comp_residual``), then pick the best zero-budget
     swap (one upgrade + one same-step downgrade). Optionally re-score the
     top-K pairs jointly with downstream lookahead before applying.

Equal-MAC constraint: the DP states track ``sum_of_stoc_lens`` rather than
``sum_of_bitops``. This requires all searchable units to share the same
per-unit MAC. Drivers must restrict their search to (op, block) tuples
that all have the same MAC (e.g. cls's ``proj+mlp_fc1+mlp_fc2`` trick;
det's ``proj+mlp_fc1+mlp_fc2`` within a single attention type).

Supported proxies:
  * ``raw_mse``        — mean block output MSE to FP
  * ``comp_residual``  — residual MSE after closed-form ridge comp
                         (matches QwT-SC's per-block correction shape)

See ``cls/experiments/mp_budget_swap_search.py`` for the historical
reference implementation that this module replaces.
"""
from __future__ import annotations

import copy
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scmp_kernels.mp import RidgeFitter

from sc_integration.mp_linear import MPConfig


# ============================================================================
# Backend interface
# ============================================================================


class MPSearchBackend:
    """Per-model adapter for the MP search.

    Subclass and implement the methods below. The cls and det drivers each
    provide their own :class:`MPSearchBackend` subclass that knows about
    the model's op vocabulary, schedule format, and patching mechanics.

    Class attributes:
      op_names: tuple of all SC op names this backend supports. The cls
        backend uses 5 names (``mlp_fc1, mlp_fc2, qk, av, proj`` — proj
        is fused qkv_proj+out_proj); det uses 6 (``mlp_fc1, mlp_fc2,
        qkv_proj, out_proj, qk, av``). The algorithm is name-agnostic.
      n_blocks: number of transformer blocks.
    """

    op_names: tuple[str, ...] = ()
    n_blocks: int = 0

    def op_macs(self, op: str, block_idx: int) -> float:
        """MAC count for one (op, block) instance. Equal-MAC DP requires
        this to be constant across (op, block) tuples in the search set —
        ``build_initial_sl_map`` raises if not."""
        raise NotImplementedError

    def comp_macs(self) -> float:
        """MAC count for one comp matmul (per block). Used only for the
        eff_sl summary; not on the search hot path."""
        raise NotImplementedError

    def load_sensitivity(self) -> dict[tuple[str, int], float]:
        """Load the per-(op, block) sensitivity score (l2 of FP-vs-SC block
        output, or any monotone proxy). Used by the DP utility and as a
        block-weight for the proxy."""
        raise NotImplementedError

    def build_active_ops(self, sc_config: str) -> set[tuple[str, int]]:
        """Set of (op, block_idx) tuples that are SC-active under the given
        schedule config name (e.g. 'skip_worst30'). The complement stays
        FP for the duration of the search."""
        raise NotImplementedError

    def load_model(self, device: torch.device) -> nn.Module:
        """Build a fresh FP model on the given device."""
        raise NotImplementedError

    def get_blocks(self, model: nn.Module) -> list[nn.Module]:
        """Return the model's transformer blocks as a flat list."""
        raise NotImplementedError

    def build_loader(self, n_images: int, seed: int,
                     batch_size: int, workers: int) -> DataLoader:
        """Build a small calibration loader yielding ``n_images`` images.
        Per-driver semantics: cls draws from ImageNet-1k val with seeded
        shuffle; det draws from COCO val (deterministic order)."""
        raise NotImplementedError

    def patch_model_with_sl_map(self, model: nn.Module, sc_prec: int,
                                sl_map: dict) -> dict:
        """In-place patch every block of ``model`` according to ``sl_map``.
        Used once per swap-search iteration to capture block-0 inputs from
        the SC-patched model. Returns swap-counts per op (diagnostic)."""
        raise NotImplementedError

    def patch_block_with_spec(self, block: nn.Module, sc_prec: int,
                              block_spec: dict) -> nn.Module:
        """Return a deepcopy of ``block`` with SC submodules installed
        according to a single-block ``block_spec`` (dict[op_name, entry]).
        Used in the inner loop of ``evaluate_block_moves`` to score
        candidate moves."""
        raise NotImplementedError


# ============================================================================
# sl_map / entry / unit data structures
# ============================================================================
# An sl_map is dict[op_name, list[entry]] where len(list) == n_blocks.
# An entry is either:
#   * int: scalar stoc_len for that (op, block).
#   * list[int] of length n_bins: per-row (or per-head) stoc_lens. The
#     algorithm preserves the bin structure but the level-set (the unique
#     values) is what becomes the MPConfig at inference.
# A "unit" is a tuple identifying one searchable slot:
#   * (op, block_idx): scalar slot.
#   * (op, block_idx, bin_idx): one row of a fraction-bins entry.


def empty_sl_map(op_names: Sequence[str], n_blocks: int) -> dict[str, list[int]]:
    return {op: [0] * n_blocks for op in op_names}


def entry_bins(entry) -> list[int]:
    if isinstance(entry, list):
        return [int(x) for x in entry]
    return [int(entry)]


def entry_avg_sl(entry) -> float:
    bins = entry_bins(entry)
    return float(sum(bins)) / float(len(bins))


def entry_levels_fractions(entry) -> tuple[list[int], list[float]]:
    bins = entry_bins(entry)
    total = len(bins)
    counts = Counter(bins)
    levels = sorted(counts.keys(), reverse=True)
    fracs = [counts[sl] / total for sl in levels]
    return levels, fracs


def make_entry(level: int, n_bins: int) -> int | list[int]:
    if n_bins <= 1:
        return int(level)
    return [int(level)] * int(n_bins)


def ensure_entry(sl_map: dict[str, list], op: str, bi: int, n_bins: int) -> None:
    if isinstance(sl_map[op][bi], list):
        return
    sl_map[op][bi] = make_entry(int(sl_map[op][bi]), n_bins)


def unit_parent(unit: tuple) -> tuple[str, int]:
    return (str(unit[0]), int(unit[1]))


def get_unit_level(sl_map: dict[str, list], unit: tuple) -> int:
    op, bi = unit_parent(unit)
    bins = entry_bins(sl_map[op][bi])
    if len(unit) >= 3:
        idx = int(unit[2])
        if len(bins) == 1:
            return int(bins[0])
        if idx >= len(bins):
            return int(bins[-1])
        return int(bins[idx])
    return int(bins[0])


def set_unit_level(sl_map: dict[str, list], unit: tuple,
                   value: int, n_bins: int) -> None:
    op, bi = unit_parent(unit)
    if len(unit) >= 3:
        ensure_entry(sl_map, op, bi, n_bins)
        bins = list(entry_bins(sl_map[op][bi]))
        bins[int(unit[2])] = int(value)
        sl_map[op][bi] = bins
    else:
        sl_map[op][bi] = int(value)


def entry_to_mpconfig(entry) -> Optional[MPConfig]:
    levels_all, fracs_all = entry_levels_fractions(entry)
    kept = [(int(sl), float(frac)) for sl, frac in zip(levels_all, fracs_all)
            if int(sl) > 0]
    if not kept:
        return None
    levels = [sl for sl, _frac in kept]
    kept_fracs = [frac for _sl, frac in kept]
    s = sum(kept_fracs)
    if s <= 0:
        return None
    kept_fracs = [x / s for x in kept_fracs]
    return MPConfig(levels, kept_fracs)


def build_search_units(
    *,
    active_ops: set[tuple[str, int]],
    search_ops: set[str],
    n_bins: int,
    bin_ops: set[str],
) -> list[tuple]:
    units: list[tuple] = []
    for op, bi in sorted(active_ops, key=lambda x: (x[1], x[0])):
        if op not in search_ops:
            continue
        if n_bins > 1 and op in bin_ops:
            for bin_idx in range(n_bins):
                units.append((op, bi, bin_idx))
        else:
            units.append((op, bi))
    return units


def build_level_spaces(
    *,
    search_units: list[tuple],
    global_levels: list[int],
    op_min_levels: dict[str, int],
    op_max_levels: dict[str, int],
) -> dict[tuple, list[int]]:
    spaces: dict[tuple, list[int]] = {}
    for unit in search_units:
        op, _bi = unit_parent(unit)
        lo = op_min_levels.get(op, global_levels[0])
        hi = op_max_levels.get(op, global_levels[-1])
        allowed = [sl for sl in global_levels if lo <= sl <= hi]
        if not allowed:
            raise ValueError(f"no valid levels for {unit!r} in [{lo}, {hi}]")
        spaces[unit] = allowed
    return spaces


# ============================================================================
# Metrics (main_sl, eff_sl, allocation summaries)
# ============================================================================


def compute_main_sl(sl_map: dict[str, list],
                    active_ops: set[tuple[str, int]],
                    op_macs_fn: Callable[[str, int], float]) -> float:
    total_macs = 0.0
    total_bitops = 0.0
    for op, bi in active_ops:
        sl = entry_avg_sl(sl_map[op][bi])
        m = float(op_macs_fn(op, bi))
        total_macs += m
        total_bitops += m * float(sl)
    return total_bitops / total_macs if total_macs > 0 else 0.0


def compute_eff_sl(main_sl: float, active_ops: set[tuple[str, int]],
                   op_macs_fn: Callable[[str, int], float],
                   comp_macs: float, n_blocks: int,
                   comp_sl: int = 256) -> float:
    main_macs = sum(op_macs_fn(op, bi) for op, bi in active_ops)
    total_comp = float(n_blocks) * float(comp_macs)
    return ((main_sl * main_macs + comp_sl * total_comp) /
            (main_macs + total_comp))


def summarize_allocation(sl_map: dict[str, list],
                         active_ops: set[tuple[str, int]],
                         op_macs_fn: Callable[[str, int], float]) -> dict[str, float]:
    level_macs: Counter = Counter()
    for op, bi in active_ops:
        levels, fracs = entry_levels_fractions(sl_map[op][bi])
        for sl, frac in zip(levels, fracs):
            level_macs[int(sl)] += float(op_macs_fn(op, bi)) * float(frac)
    total = sum(level_macs.values())
    if total == 0:
        return {}
    return {
        str(sl): round(100.0 * macs / total, 1)
        for sl, macs in sorted(level_macs.items(), reverse=True)
    }


def summarize_per_op(sl_map: dict[str, list],
                     active_ops: set[tuple[str, int]]) -> dict[str, dict[str, float]]:
    by_op: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for op, bi in active_ops:
        levels, fracs = entry_levels_fractions(sl_map[op][bi])
        for sl, frac in zip(levels, fracs):
            by_op[op][int(sl)] += float(frac)
        totals[op] += 1.0
    out: dict[str, dict[str, float]] = {}
    for op in sorted(by_op):
        n = float(totals[op])
        out[op] = {
            str(sl): round(float(by_op[op][sl]) / n, 4)
            for sl in sorted(by_op[op], reverse=True)
        }
    return out


# ============================================================================
# Initial allocation: equal-MAC DP
# ============================================================================
# Both DP variants assume search slots share the same per-unit MAC, so
# matching the main-budget reduces to matching the sum of assigned
# stoc_len values. The DP state is `cur_sum_sl` (running total).


def choose_initial_levels_exact(
    *,
    search_units: list[tuple],
    spaces: dict[tuple, list[int]],
    target_sum_sl: int,
    sens_map: dict[tuple[str, int], float],
) -> tuple[dict[tuple, int], int, float]:
    """DP that maximizes a sensitivity-weighted utility ``sum sens * sl``.
    Returns the assignment whose stoc_len sum is closest to ``target_sum_sl``;
    ties broken by higher utility."""
    states: dict[int, tuple[float, Optional[tuple[int, int]]]] = {0: (0.0, None)}
    history: list[dict[int, tuple[int, int]]] = []

    for _idx, unit in enumerate(search_units):
        next_states: dict[int, tuple[float, tuple[int, int]]] = {}
        choices: dict[int, tuple[int, int]] = {}
        sens = sens_map.get(unit_parent(unit), 0.0)
        for cur_sum, (cur_util, _prev) in states.items():
            for level in spaces[unit]:
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

    assignment: dict[tuple, int] = {}
    cur_sum = best_sum
    for idx in range(len(search_units) - 1, -1, -1):
        prev_sum, level = history[idx][cur_sum]
        assignment[search_units[idx]] = level
        cur_sum = prev_sum
    return assignment, int(best_sum), float(best_util)


def choose_initial_levels_from_base(
    *,
    search_units: list[tuple],
    spaces: dict[tuple, list[int]],
    target_sum_sl: int,
    base_levels: dict[tuple, int],
    sens_map: dict[tuple[str, int], float],
) -> tuple[dict[tuple, int], int, float]:
    """DP that minimizes ``sum (1+sens) * |level - base|`` while matching the
    budget. Used by ``uniform_repair`` and ``map_repair`` modes — keeps the
    initial allocation close to a known-good baseline."""
    states: dict[int, tuple[float, Optional[tuple[int, int]]]] = {0: (0.0, None)}
    history: list[dict[int, tuple[int, int]]] = []

    for unit in search_units:
        next_states: dict[int, tuple[float, tuple[int, int]]] = {}
        choices: dict[int, tuple[int, int]] = {}
        sens = sens_map.get(unit_parent(unit), 0.0)
        base = int(base_levels[unit])
        for cur_sum, (cur_score, _prev) in states.items():
            for level in spaces[unit]:
                new_sum = cur_sum + level
                delta = abs(level - base)
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

    assignment: dict[tuple, int] = {}
    cur_sum = best_sum
    for idx in range(len(search_units) - 1, -1, -1):
        prev_sum, level = history[idx][cur_sum]
        assignment[search_units[idx]] = level
        cur_sum = prev_sum
    return assignment, int(best_sum), float(best_score)


def _nearest_level(target: int, levels: list[int]) -> int:
    return min(levels, key=lambda x: abs(x - target))


def build_initial_sl_map(
    *,
    op_names: Sequence[str],
    active_ops: set[tuple[str, int]],
    search_ops: set[str],
    search_units: list[tuple],
    fixed_ops: dict[str, int],
    target_main_sl: int,
    level_spaces: dict[tuple, list[int]],
    sens_map: dict[tuple[str, int], float],
    op_macs_fn: Callable[[str, int], float],
    n_blocks: int,
    init_mode: str = "sensitivity_dp",
    init_level: Optional[int] = None,
    init_sl_map: Optional[dict[str, list]] = None,
    n_bins: int = 1,
    ilp_util_mode: str = "sens",
    ilp_floor_top_k: int = 0,
    ilp_floor_min_sl: int = 192,
    ilp_deep_cut_threshold: int = 32,
    ilp_deep_cut_penalty: float = 1.0,
) -> tuple[dict[str, list[int]], dict[str, object]]:
    """Build a starting sl_map that satisfies the budget exactly (or as
    closely as the level set permits).

    Init modes:
      * sensitivity_dp: maximize ``sum sens * sl`` while matching target.
        Equal-MAC DP; raises if search-unit MACs spread > 1.5×.
      * uniform_repair: start from an all-``init_level`` baseline and only
        deviate where required to hit the budget; deviations on
        high-sensitivity slots are penalized more. Equal-MAC DP.
      * map_repair: start from ``init_sl_map`` and stay close to it
        (used to initialize a higher-budget search from a lower-budget
        result). Equal-MAC DP.
      * ilp: scipy.optimize.milp formulation; handles **arbitrary** MAC
        spread (qk/av_window 562M ↔ qk/av_global 57.7G ↔ mlp 55.4G all in
        the same search). Supports per-unit ``floor_top_k`` hard lower
        bounds for the most-sensitive slots. See ``mp_ilp.py``.
    """
    sl_map = empty_sl_map(op_names, n_blocks)
    fixed_total = 0.0
    total_macs = 0.0
    # Per-unit effective MAC. Equal-MAC DP variants require this to be
    # near-constant across search_units; ILP mode handles arbitrary spread.
    search_mac_list = [
        float(op_macs_fn(*unit_parent(unit))) /
        (n_bins if len(unit) >= 3 else 1.0)
        for unit in search_units
    ]
    if search_units:
        macs_min = min(search_mac_list)
        macs_max = max(search_mac_list)
        mac_spread_tol = 1.5  # allow up to 50% max/min ratio for the DP modes
        # ILP mode bypasses this check — multiple-choice knapsack handles
        # arbitrary MAC heterogeneity natively. The DP modes need it
        # because they track ``sum_of_stoc_lens`` instead of bit-ops, which
        # is only valid when MACs are (approximately) uniform.
        if init_mode != "ilp" and macs_max / max(macs_min, 1e-9) > mac_spread_tol:
            raise ValueError(
                "MP-budget swap search: search-unit per-unit MACs span "
                f"[{macs_min:.3g}, {macs_max:.3g}], ratio "
                f"{macs_max/max(macs_min,1e-9):.2f}× > tolerance "
                f"{mac_spread_tol}×. Either restrict --search_ops to ops "
                "whose MACs are within 1.5× of each other (cls's "
                "`proj,mlp_fc1,mlp_fc2`; det 1280²: mlp_fc1,mlp_fc2[,qkv_proj]) "
                "or switch to --init_mode ilp which handles arbitrary spread.")
        search_mac = sum(search_mac_list) / len(search_mac_list)
        if init_mode != "ilp" and macs_max / max(macs_min, 1e-9) > 1.05:
            print(f"[mp_search] heterogeneous MACs in search units "
                  f"(spread {macs_max/macs_min:.2f}×); using mean MAC "
                  f"{search_mac:.3g} for budget DP. Actual bitop budget "
                  "may drift up to spread% from target.")
    else:
        search_mac = 0.0

    for op, bi in active_ops:
        total_macs += float(op_macs_fn(op, bi))
        if op in search_ops:
            sl_map[op][bi] = make_entry(0, n_bins if n_bins > 1 else 1)
            continue
        sl = fixed_ops.get(op, target_main_sl)
        sl_map[op][bi] = int(sl)
        fixed_total += float(op_macs_fn(op, bi)) * float(sl)

    target_total = float(target_main_sl) * total_macs
    if search_units:
        remaining = target_total - fixed_total
        target_sum_sl = int(round(remaining / search_mac)) if search_mac > 0 else 0
        min_sum = sum(min(level_spaces[unit]) for unit in search_units)
        max_sum = sum(max(level_spaces[unit]) for unit in search_units)
        clipped_target = min(max(target_sum_sl, min_sum), max_sum)
        ilp_meta: Optional[dict] = None
        if init_mode == "uniform_repair":
            base_target = target_main_sl if init_level is None else int(init_level)
            base_levels: dict[tuple, int] = {}
            for unit in search_units:
                allowed = level_spaces[unit]
                if base_target <= min(allowed):
                    base_levels[unit] = min(allowed)
                elif base_target >= max(allowed):
                    base_levels[unit] = max(allowed)
                elif base_target in allowed:
                    base_levels[unit] = base_target
                else:
                    base_levels[unit] = _nearest_level(base_target, allowed)
            assignment, chosen_sum, util = choose_initial_levels_from_base(
                search_units=search_units, spaces=level_spaces,
                target_sum_sl=clipped_target, base_levels=base_levels,
                sens_map=sens_map,
            )
        elif init_mode == "map_repair":
            if init_sl_map is None:
                raise ValueError("init_mode=map_repair requires init_sl_map")
            base_levels = {}
            for unit in search_units:
                allowed = level_spaces[unit]
                base_target = get_unit_level(init_sl_map, unit)
                if base_target <= min(allowed):
                    base_levels[unit] = min(allowed)
                elif base_target >= max(allowed):
                    base_levels[unit] = max(allowed)
                elif base_target in allowed:
                    base_levels[unit] = base_target
                else:
                    base_levels[unit] = _nearest_level(base_target, allowed)
            assignment, chosen_sum, util = choose_initial_levels_from_base(
                search_units=search_units, spaces=level_spaces,
                target_sum_sl=clipped_target, base_levels=base_levels,
                sens_map=sens_map,
            )
        elif init_mode == "sensitivity_dp":
            assignment, chosen_sum, util = choose_initial_levels_exact(
                search_units=search_units, spaces=level_spaces,
                target_sum_sl=clipped_target, sens_map=sens_map,
            )
        elif init_mode == "ilp":
            # ILP mode: budget is in bit-ops, no equal-MAC requirement.
            # ``chosen_sum`` is reported as Σ stoc_len (unweighted) for
            # parity with the DP meta fields; the actual bit-op-level
            # check is in ``ilp_meta['budget_gap_signed']``.
            from sc_integration.mp_ilp import ilp_init_levels
            target_bitops = float(target_main_sl) * total_macs - fixed_total
            # Build base_levels for ILP from_base mode (analogous to the
            # uniform_repair / map_repair DPs). For sens / sens_mac modes
            # base_levels is unused.
            base_levels_for_ilp: Optional[dict] = None
            if ilp_util_mode == "from_base":
                if init_sl_map is not None:
                    # map_repair-equivalent: per-unit base from existing sl_map
                    base_levels_for_ilp = {}
                    for unit in search_units:
                        allowed = level_spaces[unit]
                        base_target = get_unit_level(init_sl_map, unit)
                        if base_target <= min(allowed):
                            base_levels_for_ilp[unit] = min(allowed)
                        elif base_target >= max(allowed):
                            base_levels_for_ilp[unit] = max(allowed)
                        elif base_target in allowed:
                            base_levels_for_ilp[unit] = base_target
                        else:
                            base_levels_for_ilp[unit] = _nearest_level(base_target, allowed)
                else:
                    # uniform_repair-equivalent: single global base
                    base_target = (target_main_sl
                                   if init_level is None else int(init_level))
                    base_levels_for_ilp = {}
                    for unit in search_units:
                        allowed = level_spaces[unit]
                        if base_target <= min(allowed):
                            base_levels_for_ilp[unit] = min(allowed)
                        elif base_target >= max(allowed):
                            base_levels_for_ilp[unit] = max(allowed)
                        elif base_target in allowed:
                            base_levels_for_ilp[unit] = base_target
                        else:
                            base_levels_for_ilp[unit] = _nearest_level(base_target, allowed)
            assignment, chosen_bitops, util, ilp_meta = ilp_init_levels(
                search_units=search_units,
                spaces=level_spaces,
                target_bitops=target_bitops,
                sens_map=sens_map,
                op_macs_fn=op_macs_fn,
                util_mode=ilp_util_mode,
                floor_top_k=int(ilp_floor_top_k),
                floor_min_sl=int(ilp_floor_min_sl),
                base_levels=base_levels_for_ilp,
                deep_cut_threshold=int(ilp_deep_cut_threshold),
                deep_cut_penalty=float(ilp_deep_cut_penalty),
            )
            chosen_sum = sum(int(sl) for sl in assignment.values())
        else:
            raise ValueError(f"unknown init_mode: {init_mode}")
        for unit, sl in assignment.items():
            set_unit_level(sl_map, unit, int(sl), n_bins)
    else:
        clipped_target = target_main_sl
        chosen_sum = 0
        util = 0.0
        target_sum_sl = 0
        ilp_meta = None

    actual_main = compute_main_sl(sl_map, active_ops, op_macs_fn)
    meta = {
        "init_mode": init_mode,
        "init_level": init_level,
        "init_from_map": init_sl_map is not None,
        "search_slots": len(search_units),
        "fixed_bitops": fixed_total,
        "target_total_bitops": target_total,
        "target_search_sum_sl": None if not search_units else target_sum_sl,
        "clipped_target_search_sum_sl": None if not search_units else clipped_target,
        "chosen_search_sum_sl": chosen_sum,
        "utility": util,
        "actual_main_sl": actual_main,
    }
    if ilp_meta is not None:
        meta["ilp_meta"] = ilp_meta
    return sl_map, meta


# ============================================================================
# Block-level proxy scoring
# ============================================================================


def _forward_block_batched(block: nn.Module, x_cpu: torch.Tensor,
                           device: torch.device, chunk: int) -> torch.Tensor:
    outs = []
    for start in range(0, x_cpu.size(0), chunk):
        xb = x_cpu[start:start + chunk].to(device=device, dtype=torch.float32,
                                           non_blocking=True)
        y = block(xb)
        outs.append(y.detach().float().cpu())
    return torch.cat(outs, dim=0)


def _capture_block_inputs(model: nn.Module, blocks: list[nn.Module],
                          loader: DataLoader, device: torch.device,
                          n_search: int, cache_dtype: torch.dtype
                          ) -> list[torch.Tensor]:
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


def _compute_proxy_score(
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


def build_current_spec(sl_map: dict[str, list], op_names: Sequence[str],
                       block_idx: int) -> dict[str, object]:
    spec: dict[str, object] = {}
    for op in op_names:
        entry = sl_map[op][block_idx]
        spec[op] = list(entry) if isinstance(entry, list) else int(entry)
    return spec


def _apply_unit_to_spec(spec: dict[str, object], unit: tuple, new_sl: int) -> None:
    op = str(unit[0])
    if len(unit) >= 3:
        bins = list(entry_bins(spec[op]))
        bins[int(unit[2])] = int(new_sl)
        spec[op] = bins
    else:
        spec[op] = int(new_sl)


def _evaluate_block_moves(
    *,
    backend: MPSearchBackend,
    block_idx: int,
    fp_block: nn.Module,
    x_cpu: torch.Tensor,
    current_spec: dict[str, object],
    search_units_in_block: list[tuple],
    spaces: dict[tuple, list[int]],
    sc_prec: int,
    proxy: str,
    ridge: float,
    device: torch.device,
    fwd_chunk: int,
    block_weight: float,
) -> dict[str, object]:
    fp_block_cpu = copy.deepcopy(fp_block).cpu().eval()
    fp_block_dev = copy.deepcopy(fp_block_cpu).to(device).eval()
    y_fp_cpu = _forward_block_batched(fp_block_dev, x_cpu, device, fwd_chunk)
    del fp_block_dev
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    current_blk = backend.patch_block_with_spec(fp_block_cpu, sc_prec, current_spec).to(device).eval()
    y_cur_cpu = _forward_block_batched(current_blk, x_cpu, device, fwd_chunk)
    current_score = _compute_proxy_score(
        proxy=proxy, x_cpu=x_cpu, y_fp_cpu=y_fp_cpu, y_sc_cpu=y_cur_cpu,
        device=device, ridge=ridge,
    )
    del current_blk, y_cur_cpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    candidates = []
    for unit in search_units_in_block:
        allowed = spaces[unit]
        op = str(unit[0])
        cur_sl = (entry_bins(current_spec[op])[int(unit[2])]
                  if len(unit) >= 3 else int(entry_bins(current_spec[op])[0]))
        pos = allowed.index(cur_sl)
        for next_pos in (pos - 1, pos + 1):
            if not (0 <= next_pos < len(allowed)):
                continue
            new_sl = int(allowed[next_pos])
            cand_spec = {
                k: list(v) if isinstance(v, list) else v
                for k, v in current_spec.items()
            }
            if len(unit) >= 3:
                bins = list(entry_bins(cand_spec[op]))
                bins[int(unit[2])] = new_sl
                cand_spec[op] = bins
            else:
                cand_spec[op] = new_sl
            cand_blk = backend.patch_block_with_spec(fp_block_cpu, sc_prec, cand_spec).to(device).eval()
            y_new_cpu = _forward_block_batched(cand_blk, x_cpu, device, fwd_chunk)
            score = _compute_proxy_score(
                proxy=proxy, x_cpu=x_cpu, y_fp_cpu=y_fp_cpu, y_sc_cpu=y_new_cpu,
                device=device, ridge=ridge,
            )
            candidates.append({
                "slot": unit,
                "op": op,
                "block": block_idx,
                "bin": None if len(unit) < 3 else int(unit[2]),
                "old_sl": cur_sl,
                "new_sl": new_sl,
                "delta_sl": new_sl - cur_sl,
                "delta_score": block_weight * (score - current_score),
                "score": score,
            })
            del cand_blk, y_new_cpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del fp_block_cpu, y_fp_cpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "block": block_idx,
        "weight": block_weight,
        "current_score": current_score,
        "candidates": candidates,
    }


def _build_block_weight(active_ops: set[tuple[str, int]],
                        sens_map: dict[tuple[str, int], float]) -> dict[int, float]:
    weights: dict[int, float] = defaultdict(float)
    for op, bi in active_ops:
        weights[bi] += sens_map.get((op, bi), 0.0)
    return dict(weights)


def _score_block_range(
    *,
    backend: MPSearchBackend,
    start_block: int,
    end_block: int,
    fp_blocks: list[nn.Module],
    x_start_cpu: torch.Tensor,
    block_specs: dict[int, dict[str, object]],
    sc_prec: int,
    proxy: str,
    ridge: float,
    device: torch.device,
    fwd_chunk: int,
    block_weight: dict[int, float],
) -> float:
    total = 0.0
    x_fp_cpu = x_start_cpu
    x_sc_cpu = x_start_cpu

    for bi in range(start_block, end_block + 1):
        fp_blk_dev = copy.deepcopy(fp_blocks[bi]).to(device).eval()
        y_fp_cpu = _forward_block_batched(fp_blk_dev, x_fp_cpu, device, fwd_chunk)
        del fp_blk_dev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sc_blk = backend.patch_block_with_spec(fp_blocks[bi], sc_prec, block_specs[bi]).to(device).eval()
        y_sc_cpu = _forward_block_batched(sc_blk, x_sc_cpu, device, fwd_chunk)
        score = _compute_proxy_score(
            proxy=proxy, x_cpu=x_sc_cpu, y_fp_cpu=y_fp_cpu, y_sc_cpu=y_sc_cpu,
            device=device, ridge=ridge,
        )
        total += float(block_weight.get(bi, 1.0)) * float(score)
        del sc_blk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        x_fp_cpu = y_fp_cpu
        x_sc_cpu = y_sc_cpu

    return float(total)


def _refine_pair_candidates(
    *,
    backend: MPSearchBackend,
    sl_map: dict[str, list],
    fp_blocks: list[nn.Module],
    x_by_block: list[torch.Tensor],
    upgrades_by_step: dict[int, list[dict[str, object]]],
    downgrades_by_step: dict[int, list[dict[str, object]]],
    block_weight: dict[int, float],
    sc_prec: int,
    proxy: str,
    ridge: float,
    device: torch.device,
    fwd_chunk: int,
    pair_eval_topk: int,
    lookahead_blocks: int,
) -> Optional[dict[str, object]]:
    if pair_eval_topk <= 0:
        return None

    nb = len(fp_blocks)
    base_specs = {bi: build_current_spec(sl_map, backend.op_names, bi) for bi in range(nb)}
    current_cache: dict[tuple[int, int], float] = {}
    best_pair = None
    best_delta = 0.0

    for step, upgrades in upgrades_by_step.items():
        downgrades = downgrades_by_step.get(step, [])
        if not downgrades:
            continue

        top_upgrades = sorted(upgrades, key=lambda x: float(x["delta_score"]))[:pair_eval_topk]
        top_downgrades = sorted(downgrades, key=lambda x: float(x["delta_score"]))[:pair_eval_topk]

        for up in top_upgrades:
            for down in top_downgrades:
                if up["slot"] == down["slot"]:
                    continue

                start_block = min(int(up["block"]), int(down["block"]))
                end_block = min(
                    nb - 1,
                    max(int(up["block"]), int(down["block"])) + int(lookahead_blocks),
                )
                key = (start_block, end_block)
                if key not in current_cache:
                    current_cache[key] = _score_block_range(
                        backend=backend,
                        start_block=start_block, end_block=end_block,
                        fp_blocks=fp_blocks,
                        x_start_cpu=x_by_block[start_block],
                        block_specs=base_specs,
                        sc_prec=sc_prec, proxy=proxy, ridge=ridge,
                        device=device, fwd_chunk=fwd_chunk,
                        block_weight=block_weight,
                    )

                pair_specs = {
                    bi: {
                        k: (list(v) if isinstance(v, list) else int(v))
                        for k, v in spec.items()
                    }
                    for bi, spec in base_specs.items()
                }
                _apply_unit_to_spec(pair_specs[int(up["block"])], tuple(up["slot"]), int(up["new_sl"]))
                _apply_unit_to_spec(pair_specs[int(down["block"])], tuple(down["slot"]), int(down["new_sl"]))

                pair_score = _score_block_range(
                    backend=backend,
                    start_block=start_block, end_block=end_block,
                    fp_blocks=fp_blocks,
                    x_start_cpu=x_by_block[start_block],
                    block_specs=pair_specs,
                    sc_prec=sc_prec, proxy=proxy, ridge=ridge,
                    device=device, fwd_chunk=fwd_chunk,
                    block_weight=block_weight,
                )
                pair_delta = float(pair_score - current_cache[key])
                if best_pair is None or pair_delta < best_delta:
                    best_delta = pair_delta
                    best_pair = {
                        # Same step-vs-bucket_key split as the main loop:
                        # ``step`` is always int SL units; ``bucket_key`` is
                        # whatever the bucket dict was keyed on (int or tuple).
                        "step": int(abs(int(up["delta_sl"]))),
                        "bucket_key": step,
                        "upgrade": up,
                        "downgrade": down,
                        "delta_score": pair_delta,
                        "pair_scope": [start_block, end_block],
                    }

    return best_pair


# ============================================================================
# Main entry point: iterative_swap_search
# ============================================================================


def _build_mac_classes(
    search_units: list[tuple],
    op_macs_fn: Callable[[str, int], float],
    tol: float,
) -> dict[tuple, int]:
    """Cluster ``search_units`` into MAC equivalence classes.

    Two units land in the same class iff their per-unit MACs differ by at
    most ``tol`` (relative). Sort by MAC, scan once, start a new class
    whenever the next MAC exceeds ``(1+tol)`` of the previous one. This
    guarantees within-class ``max/min ≤ (1+tol)``.

    Returns ``{unit: class_id}``. Class ids are 0-indexed integers.

    With ``tol=0.0`` (default for backward-compat), only exact-MAC
    matches share a class — equivalent to the old ``Δ_sl``-only bucket
    when search_ops are restricted to a single MAC.
    """
    if not search_units:
        return {}
    if tol < 0:
        raise ValueError(f"mac_bucket_tol must be ≥ 0, got {tol}")
    units_sorted = sorted(search_units,
                          key=lambda u: op_macs_fn(*unit_parent(u)))
    classes: dict[tuple, int] = {}
    cur_cls = 0
    last_mac: Optional[float] = None
    for u in units_sorted:
        m = float(op_macs_fn(*unit_parent(u)))
        if last_mac is not None and m / max(last_mac, 1e-9) > 1.0 + tol:
            cur_cls += 1
        classes[u] = cur_cls
        last_mac = m
    return classes


def iterative_swap_search(
    *,
    backend: MPSearchBackend,
    active_ops: set[tuple[str, int]],
    search_ops: set[str],
    search_units: list[tuple],
    sl_map: dict[str, list],
    spaces: dict[tuple, list[int]],
    loader: DataLoader,
    device: torch.device,
    sc_prec: int,
    proxy: str,
    ridge: float,
    max_iters: int,
    fwd_chunk: int,
    cache_dtype: torch.dtype,
    n_bins: int,
    pair_eval_topk: int,
    lookahead_blocks: int,
    mac_bucket_tol: float = 0.0,
    drift_warn_threshold: float = 0.01,
    log_fn: Callable[[str], None] = print,
) -> tuple[dict[str, list], list[dict[str, object]], dict[str, object]]:
    """Iterative pairwise swap search. Each round: score every ±1-step
    move per unit on the proxy, pick the best zero-budget upgrade+downgrade
    pair (optionally re-scored jointly with downstream lookahead), apply
    if it improves the global proxy. Stops when no swap improves.

    Pair bucketing:
      ``mac_bucket_tol == 0`` (default): bucket by ``|Δ_sl|`` only — the
        original equal-MAC behavior. Backward-compatible with all existing
        DP-based init modes that enforce equal MAC across search_units.
      ``mac_bucket_tol > 0``: bucket by ``(mac_class, |Δ_sl|)``, where two
        units share a MAC class iff their MACs differ by ≤ ``mac_bucket_tol``
        (relative). Designed for ``init_mode=ilp`` where search_units span
        multiple MAC magnitudes (e.g. det's qk/av_window 562M ↔ mlp 55.4G).
        Within a class, swap pairs are still ±1-step zero-budget; across
        classes, no pair forms (cross-class budget redistribution is the
        ILP init's job, not swap's).

    Drift watch:
      Whenever an accepted pair shifts ``main_sl`` by more than
      ``drift_warn_threshold`` (relative to the post-init main_sl), log a
      warning. With ``mac_bucket_tol > 0`` the per-pair drift is bounded
      by ``tol`` and statistically near-zero, but the cumulative drift
      across many swaps can grow.

    Returns ``(final_sl_map, history, last_iter_meta)``.
    """
    sens_map = backend.load_sensitivity()
    block_weight = _build_block_weight(active_ops, sens_map)
    search_units = list(search_units)
    history: list[dict[str, object]] = []
    iteration_meta: dict[str, object] = {}
    n_loader = getattr(loader.dataset, "__len__", lambda: 0)()
    mac_classes = _build_mac_classes(search_units, backend.op_macs,
                                      mac_bucket_tol)
    initial_main_sl = compute_main_sl(sl_map, active_ops, backend.op_macs)
    drift_warned = False

    for it in range(max_iters):
        t0 = time.time()
        model_sc = backend.load_model(device).eval()
        patch_stats = backend.patch_model_with_sl_map(model_sc, sc_prec, sl_map)
        blocks_sc = backend.get_blocks(model_sc)
        x_by_block = _capture_block_inputs(model_sc, blocks_sc, loader, device,
                                           n_loader, cache_dtype)
        del model_sc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_fp = backend.load_model(torch.device("cpu")).eval()
        fp_blocks = backend.get_blocks(model_fp)

        # Bucket key: (mac_class, |Δ_sl|) when mac_bucket_tol > 0, else
        # |Δ_sl| alone (legacy equal-MAC behavior). Both are valid dict keys;
        # the consumer code (`upgrades_by_step.items()` + `.get(key)`) is
        # key-type-agnostic.
        upgrades_by_step: dict = defaultdict(list)
        downgrades_by_step: dict = defaultdict(list)
        block_reports = []
        search_blocks = sorted({int(unit[1]) for unit in search_units})
        for bi in search_blocks:
            units_here = [unit for unit in search_units if int(unit[1]) == bi]
            if not units_here:
                continue
            report = _evaluate_block_moves(
                backend=backend,
                block_idx=bi,
                fp_block=fp_blocks[bi],
                x_cpu=x_by_block[bi],
                current_spec=build_current_spec(sl_map, backend.op_names, bi),
                search_units_in_block=units_here,
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
                if mac_bucket_tol > 0:
                    cls = mac_classes.get(tuple(cand["slot"]), -1)
                    key: object = (cls, step)
                else:
                    key = step
                if cand["delta_sl"] > 0:
                    upgrades_by_step[key].append(cand)
                elif cand["delta_sl"] < 0:
                    downgrades_by_step[key].append(cand)

        best_pair = None
        best_delta = 0.0
        used_pair_refine = False
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
                            # Step in SL units (always int) — the bucket
                            # key may be a tuple when mac_bucket_tol > 0,
                            # but the history entry wants a plain int step.
                            "step": int(abs(int(up["delta_sl"]))),
                            "bucket_key": step,
                            "upgrade": up,
                            "downgrade": down,
                            "delta_score": pair_delta,
                            "pair_scope": [
                                min(int(up["block"]), int(down["block"])),
                                max(int(up["block"]), int(down["block"])),
                            ],
                        }

        if pair_eval_topk > 0:
            refined_pair = _refine_pair_candidates(
                backend=backend,
                sl_map=sl_map,
                fp_blocks=fp_blocks,
                x_by_block=x_by_block,
                upgrades_by_step=upgrades_by_step,
                downgrades_by_step=downgrades_by_step,
                block_weight=block_weight,
                sc_prec=sc_prec,
                proxy=proxy,
                ridge=ridge,
                device=device,
                fwd_chunk=fwd_chunk,
                pair_eval_topk=pair_eval_topk,
                lookahead_blocks=lookahead_blocks,
            )
            if refined_pair is not None:
                best_pair = refined_pair
                best_delta = float(refined_pair["delta_score"])
                used_pair_refine = True

        del model_fp, fp_blocks, x_by_block
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        actual_main = compute_main_sl(sl_map, active_ops, backend.op_macs)
        iteration_meta = {
            "iter": it,
            "actual_main_sl": actual_main,
            "patch_stats": patch_stats,
            "block_reports": block_reports,
            "pair_eval_topk": int(pair_eval_topk),
            "lookahead_blocks": int(lookahead_blocks),
            "used_pair_refine": bool(used_pair_refine),
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
        set_unit_level(sl_map, tuple(up["slot"]), int(up["new_sl"]), n_bins)
        set_unit_level(sl_map, tuple(down["slot"]), int(down["new_sl"]), n_bins)
        main_sl_after = compute_main_sl(sl_map, active_ops, backend.op_macs)
        # Drift watch: cumulative deviation from post-init main_sl.
        # When mac_bucket_tol > 0, individual pairs may not be perfectly
        # zero-budget (within ±tol of MAC), and many swaps can compound.
        if initial_main_sl > 0:
            drift_frac = abs(main_sl_after - initial_main_sl) / initial_main_sl
            if drift_frac > drift_warn_threshold and not drift_warned:
                log_fn(
                    f"[swap_search] WARN cumulative main_sl drift {drift_frac:.3%} "
                    f"exceeds threshold {drift_warn_threshold:.1%} (init {initial_main_sl:.2f} → "
                    f"now {main_sl_after:.2f}); MAC-tol pairs accumulating budget skew."
                )
                drift_warned = True
        history.append({
            "iter": it,
            "delta_score": float(best_pair["delta_score"]),
            "step": int(best_pair["step"]),
            "pair_scope": list(best_pair.get("pair_scope", [])),
            "used_pair_refine": bool(used_pair_refine),
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
            "main_sl_after": main_sl_after,
        })

    iteration_meta["initial_main_sl"] = float(initial_main_sl)
    iteration_meta["drift_warned"] = bool(drift_warned)
    iteration_meta["mac_bucket_tol"] = float(mac_bucket_tol)
    return sl_map, history, iteration_meta
