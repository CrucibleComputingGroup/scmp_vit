"""ILP-based initial level allocation for the MP-budget swap search.

Backend: ``scipy.optimize.milp`` (HiGHS). Replaces the equal-MAC DP in
``sc_integration.mp_search.choose_initial_levels_exact`` for the
heterogeneous-MAC regime where searchable ops have widely different
per-unit MACs (e.g. det at 1280²: ``qkv_proj`` 38G, ``out_proj`` 12.7G,
``qk/av_window`` 562M, ``qk/av_global`` 57.7G — spread > 100×).

Formulation: multiple-choice knapsack

  Variables
    x_{u, ell}  ∈ {0, 1}   one (unit, level) pair
    s_pos, s_neg  ≥ 0      budget-gap slack (handles |Σ bops − B|)

  Constraints
    Σ_ell x_{u, ell} = 1                              ∀ u
    Σ MAC_u · ell · x_{u, ell} − s_pos + s_neg = B
    [optional] Σ_{ell ≥ floor_min_sl} x_{u, ell} = 1  for u in top-K by sens

  Objective (lex via big-M)
    min  M·(s_pos + s_neg)  −  Σ sens_u · factor_u · ell · x_{u, ell}

    factor_u = MAC_u  for ``util_mode = "sens_mac"`` → pure-sens ranking
              ("high sens always gets high SL regardless of MAC cost")
    factor_u = 1      for ``util_mode = "sens"``      → cost-aware ranking
              (Lagrangian threshold becomes sens/MAC; high sens with huge
               MAC may lose to low sens with low MAC — the original DP
               behaves this way)

Equivalence with ``choose_initial_levels_exact``: when all MACs are equal
and ``util_mode = "sens"``, this ILP yields the same optimum (modulo
tiebreak determinism) as the equal-MAC DP. Verified by the smoke test
in this module.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def unit_parent(unit: tuple) -> tuple[str, int]:
    return (str(unit[0]), int(unit[1]))


def ilp_init_levels(
    *,
    search_units: list[tuple],
    spaces: dict[tuple, list[int]],
    target_bitops: float,
    sens_map: dict[tuple[str, int], float],
    op_macs_fn: Callable[[str, int], float],
    util_mode: str = "sens_mac",
    floor_top_k: int = 0,
    floor_min_sl: int = 192,
    util_dominate: Optional[float] = None,
) -> tuple[dict[tuple, int], float, float, dict[str, object]]:
    """Solve the ILP and return ``(assignment, chosen_bitops, utility, meta)``.

    ``assignment`` maps each unit → chosen integer level (one of ``spaces[u]``).
    ``chosen_bitops`` is ``Σ MAC · level`` over ``assignment``.
    ``utility`` is ``Σ sens · factor · level`` (sign convention: larger = better).
    ``meta`` carries solver diagnostics (slack, time, util_mode).
    """
    if util_mode not in ("sens_mac", "sens"):
        raise ValueError(f"util_mode must be 'sens_mac' or 'sens', got {util_mode!r}")

    if not search_units:
        return {}, 0.0, 0.0, {
            "util_mode": util_mode, "n_units": 0, "n_vars": 0,
            "budget_slack": 0.0, "solve_time_s": 0.0,
        }

    # ------------------------------------------------------------------
    # 1. Index variables: one binary per (unit, level) pair.
    # ------------------------------------------------------------------
    rows: list[tuple[float, int, float, float, tuple]] = []  # (sens, ell, mac, bops, unit)
    var_idx: dict[tuple[tuple, int], int] = {}
    for u in search_units:
        sens = float(sens_map.get(unit_parent(u), 0.0))
        mac = float(op_macs_fn(*unit_parent(u)))
        for ell in spaces[u]:
            var_idx[(u, int(ell))] = len(rows)
            rows.append((sens, int(ell), mac, mac * float(ell), u))
    n_x = len(rows)
    n_total = n_x + 2  # + s_pos, s_neg

    # ------------------------------------------------------------------
    # 2. Objective coefficients (we minimize, so utility takes negative sign).
    # ------------------------------------------------------------------
    util_terms = np.zeros(n_x)
    for i, (sens, ell, mac, _, _) in enumerate(rows):
        factor = mac if util_mode == "sens_mac" else 1.0
        util_terms[i] = sens * factor * ell

    # Auto-compute big-M so slack penalty dominates utility.
    if util_dominate is None:
        max_possible_util = float(np.sum(np.abs(util_terms)))
        util_dominate = max(1.0, max_possible_util * 100.0)

    c = np.zeros(n_total)
    c[:n_x] = -util_terms
    c[n_x] = util_dominate      # s_pos
    c[n_x + 1] = util_dominate  # s_neg

    # ------------------------------------------------------------------
    # 3. one-of-K + budget constraints.
    # ------------------------------------------------------------------
    n_units = len(search_units)
    A = np.zeros((n_units + 1, n_total))
    b = np.zeros(n_units + 1)
    for ui, u in enumerate(search_units):
        for ell in spaces[u]:
            A[ui, var_idx[(u, int(ell))]] = 1.0
        b[ui] = 1.0
    last = n_units
    for i, (_, _, _, bops, _) in enumerate(rows):
        A[last, i] = bops
    A[last, n_x] = -1.0       # − s_pos
    A[last, n_x + 1] = +1.0   # + s_neg
    b[last] = float(target_bitops)
    constraints = [LinearConstraint(A, b, b)]

    # ------------------------------------------------------------------
    # 4. Hard floor for top-K most-sensitive units (optional).
    # ------------------------------------------------------------------
    floor_units: list[tuple] = []
    if floor_top_k > 0 and floor_min_sl > 0:
        ranked = sorted(
            search_units,
            key=lambda u: -float(sens_map.get(unit_parent(u), 0.0)),
        )
        for u in ranked[:int(floor_top_k)]:
            allowed = [ell for ell in spaces[u] if int(ell) >= int(floor_min_sl)]
            if not allowed:
                # No allowed level meets the floor — skip this unit silently
                # (its level_space precludes the floor; usually means
                # op_max_levels was tighter). We don't raise so floors stay
                # opt-in / best-effort.
                continue
            floor_units.append(u)
        if floor_units:
            A_floor = np.zeros((len(floor_units), n_total))
            for fi, u in enumerate(floor_units):
                for ell in spaces[u]:
                    if int(ell) >= int(floor_min_sl):
                        A_floor[fi, var_idx[(u, int(ell))]] = 1.0
            b_floor = np.ones(len(floor_units))
            constraints.append(LinearConstraint(A_floor, b_floor, b_floor))

    # ------------------------------------------------------------------
    # 5. Bounds + integrality (binary x, continuous slack).
    # ------------------------------------------------------------------
    integrality = np.concatenate([np.ones(n_x), np.zeros(2)])  # 1 = integer
    ub = np.concatenate([np.ones(n_x), [np.inf, np.inf]])
    lb = np.zeros(n_total)
    bounds = Bounds(lb=lb, ub=ub)

    # ------------------------------------------------------------------
    # 6. Solve.
    # ------------------------------------------------------------------
    import time as _time
    t0 = _time.time()
    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
    solve_time = _time.time() - t0
    if not res.success:
        raise RuntimeError(
            f"ILP init failed: {res.message}. "
            f"n_vars={n_total}, n_constraints={sum(c.A.shape[0] for c in constraints)}, "
            f"target_bitops={target_bitops:.3g}, floor_top_k={floor_top_k}, "
            f"floor_min_sl={floor_min_sl}."
        )

    # ------------------------------------------------------------------
    # 7. Decode solution.
    # ------------------------------------------------------------------
    x = res.x[:n_x]
    s_pos = float(res.x[n_x])
    s_neg = float(res.x[n_x + 1])

    assignment: dict[tuple, int] = {}
    chosen_bops = 0.0
    utility = 0.0
    for (u, ell), i in var_idx.items():
        if x[i] > 0.5:
            assignment[u] = int(ell)
            chosen_bops += rows[i][3]
            utility += util_terms[i]

    # Sanity check: every unit got exactly one level.
    if len(assignment) != n_units:
        raise RuntimeError(
            f"ILP returned partial assignment: got {len(assignment)} of "
            f"{n_units} units. Solver may be returning a fractional solution; "
            f"check integrality settings."
        )

    meta = {
        "util_mode": util_mode,
        "n_units": n_units,
        "n_vars": n_total,
        "n_constraints": sum(con.A.shape[0] for con in constraints),
        "n_floor_units": len(floor_units),
        "floor_top_k": int(floor_top_k),
        "floor_min_sl": int(floor_min_sl),
        "budget_slack_pos": s_pos,
        "budget_slack_neg": s_neg,
        "budget_gap_signed": s_neg - s_pos,  # actual − target (positive = under-budget)
        "solve_time_s": round(solve_time, 4),
        "utility": float(utility),
        "chosen_bitops": float(chosen_bops),
        "target_bitops": float(target_bitops),
    }
    return assignment, float(chosen_bops), float(utility), meta


# ---------------------------------------------------------------------------
# Smoke test: equal-MAC ILP must agree with the equal-MAC DP.
# ---------------------------------------------------------------------------

def _smoke_test_equal_mac() -> None:
    """Sanity: with all MACs equal and util_mode='sens', the ILP optimum
    matches choose_initial_levels_exact."""
    import sys
    from pathlib import Path
    _HERE = Path(__file__).resolve().parent
    if str(_HERE.parent) not in sys.path:
        sys.path.insert(0, str(_HERE.parent))
    from sc_integration.mp_search import choose_initial_levels_exact

    # 6 units, all MAC = 1G, levels = [64, 96, 128, 192, 256], random sens
    import random
    random.seed(42)
    units = [("op_a", i) for i in range(3)] + [("op_b", i) for i in range(3)]
    spaces = {u: [64, 96, 128, 192, 256] for u in units}
    sens_map = {u: random.uniform(0.1, 5.0) for u in units}

    target_sum_sl = 6 * 128  # average 128
    mac = 1e9
    target_bitops = mac * target_sum_sl

    # DP reference
    dp_assign, dp_sum, dp_util = choose_initial_levels_exact(
        search_units=units, spaces=spaces,
        target_sum_sl=target_sum_sl, sens_map=sens_map,
    )

    # ILP with util_mode=sens (matches DP utility exactly)
    ilp_assign, ilp_bops, ilp_util, ilp_meta = ilp_init_levels(
        search_units=units, spaces=spaces, target_bitops=target_bitops,
        sens_map=sens_map, op_macs_fn=lambda op, bi: mac,
        util_mode="sens",
    )
    ilp_sum = sum(ilp_assign[u] for u in units)

    print(f"[smoke] DP   sum={dp_sum} util={dp_util:.4f} assign={dp_assign}")
    print(f"[smoke] ILP  sum={ilp_sum} util={ilp_util:.4f} assign={ilp_assign}")
    print(f"[smoke] ILP  meta={ilp_meta}")
    assert ilp_sum == dp_sum, f"sum mismatch: ILP {ilp_sum} != DP {dp_sum}"
    assert abs(ilp_util - dp_util) < 1e-6, f"util mismatch: {ilp_util} vs {dp_util}"
    print("[smoke] ✓ ILP matches DP on equal-MAC reference.")

    # Heterogeneous MAC: just verify it solves & respects budget.
    macs = {("op_a", i): 1e9 for i in range(3)}
    macs.update({("op_b", i): 4e9 for i in range(3)})  # 4× MAC spread
    target_bitops_het = sum(m for m in macs.values()) * 128
    ilp_assign2, ilp_bops2, ilp_util2, ilp_meta2 = ilp_init_levels(
        search_units=units, spaces=spaces, target_bitops=target_bitops_het,
        sens_map=sens_map,
        op_macs_fn=lambda op, bi: macs[(op, bi)],
        util_mode="sens_mac",
    )
    print(f"[smoke] ILP-het assign={ilp_assign2}")
    print(f"[smoke] ILP-het meta={ilp_meta2}")
    gap_frac = abs(ilp_meta2["budget_gap_signed"]) / target_bitops_het
    assert gap_frac < 0.01, f"budget gap too large: {gap_frac:.3%}"
    print(f"[smoke] ✓ heterogeneous-MAC ILP solves, gap={gap_frac:.4%}.")

    # Hard floor: all units in top-2 by sens must hit ≥ 192.
    ilp_assign3, _, _, ilp_meta3 = ilp_init_levels(
        search_units=units, spaces=spaces, target_bitops=target_bitops,
        sens_map=sens_map, op_macs_fn=lambda op, bi: mac,
        util_mode="sens", floor_top_k=2, floor_min_sl=192,
    )
    top2 = sorted(units, key=lambda u: -sens_map[u])[:2]
    for u in top2:
        assert ilp_assign3[u] >= 192, f"floor violated for {u}: {ilp_assign3[u]}"
    print(f"[smoke] ✓ hard floor respected: top-2 sens get ≥ 192.")


if __name__ == "__main__":
    _smoke_test_equal_mac()
