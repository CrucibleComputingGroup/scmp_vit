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
    min  M·(s_pos + s_neg)  +  utility_term(util_mode)

    util_mode = "sens" (default)         → cost-aware reward
        utility_term  =  − Σ sens_u · ell · x_{u, ell}
        Lagrangian threshold becomes sens/MAC; cheap-but-sensitive ops
        win their share before expensive-but-equally-sensitive ones.
        Matches the original equal-MAC DP exactly when MACs are uniform.

    util_mode = "sens_mac"               → pure-sens reward
        utility_term  =  − Σ sens_u · MAC_u · ell · x_{u, ell}
        "high sens gets high SL regardless of MAC cost". Empirically
        worse on det heterogeneous-MAC search (lost ~13 mAP at
        target=128 vs uniform-128 baseline); the bipolar 256/64 corner
        solution scrapes too much precision off mid-sens slots.

    util_mode = "from_base"              → symmetric baseline-anchor
        utility_term  =  Σ (1 + sens_u) · |ell_u − base_u|

        Symmetric anchor: ANY deviation from base costs (1 + sens_u)·|Δ|,
        in either direction. High-sens units are anchored hard at base
        (expensive to move). Low-sens units are cheap to move and act as
        the budget-compensation pool when the budget forces deviation.

        Pair this with ``floor_top_k`` to selectively upgrade the K most
        sensitive units while keeping everyone else as close to base as
        possible — the ILP picks the cheapest (lowest-sens) units to
        downgrade if the floor pushes the budget past target.

        Mirrors cls's symmetric DP ``choose_initial_levels_from_base``.
        Requires ``base_levels`` arg. Linearized via two non-negative
        slack vars per unit (δ+ = upgrade, δ- = downgrade).

Equivalence with equal-MAC DPs: ``util_mode="sens"`` matches
``choose_initial_levels_exact`` exactly when MACs are uniform.
``util_mode="from_base"`` matches the symmetric DP
``choose_initial_levels_from_base`` when MACs are uniform.
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
    util_mode: str = "sens",
    floor_top_k: int = 0,
    floor_min_sl: int = 192,
    base_levels: Optional[dict[tuple, int]] = None,
    util_dominate: Optional[float] = None,
    deep_cut_threshold: int = 32,
    deep_cut_penalty: float = 1.0,
) -> tuple[dict[tuple, int], float, float, dict[str, object]]:
    """Solve the ILP and return ``(assignment, chosen_bitops, utility, meta)``.

    ``assignment`` maps each unit → chosen integer level (one of ``spaces[u]``).
    ``chosen_bitops`` is ``Σ MAC · level`` over ``assignment``.
    ``utility`` is the value of the chosen objective term (signed so that
    larger = better — matches the DP convention).
    ``meta`` carries solver diagnostics (slack, time, util_mode).

    ``base_levels`` is required when ``util_mode == "from_base"`` and ignored
    otherwise; provide one integer baseline level per search unit (typically
    derived from ``init_level`` clipped to each unit's ``spaces[u]``).
    """
    if util_mode not in ("sens", "sens_mac", "from_base"):
        raise ValueError(
            f"util_mode must be 'sens', 'sens_mac', or 'from_base', got {util_mode!r}")
    if util_mode == "from_base" and base_levels is None:
        raise ValueError("util_mode='from_base' requires base_levels (one int per search unit)")

    if not search_units:
        return {}, 0.0, 0.0, {
            "util_mode": util_mode, "n_units": 0, "n_vars": 0,
            "budget_slack": 0.0, "solve_time_s": 0.0,
        }

    n_units = len(search_units)
    use_dev = (util_mode == "from_base")
    # PWL downgrade: split δ-_u into step1 (≤ deep_cut_threshold, cost 1×)
    # and step2 (the rest, cost α×). Active only when α > 1 — otherwise
    # use a single δ- slack (cheaper, identical answer at α=1).
    use_pwl_dn = use_dev and float(deep_cut_penalty) > 1.0 + 1e-9

    # ------------------------------------------------------------------
    # 1. Index variables.
    #    [0,                  n_x):                  binary x_{u,ℓ}
    #    [n_x,                n_x + 2):              budget slack s+, s-
    #    [n_x + 2,            n_x + 2 + n_units):    δ+_u             (from_base)
    #    [n_x + 2 + n_units,  n_x + 2 + 2·n_units):  δ-_u (or δ-_step1 if PWL)
    #    [n_x + 2 + 2·n_units,n_x + 2 + 3·n_units):  δ-_step2  (PWL only)
    # ------------------------------------------------------------------
    rows: list[tuple[float, int, float, float, tuple]] = []  # (sens, ell, mac, bops, unit)
    var_idx: dict[tuple[tuple, int], int] = {}
    sens_per_unit: dict[tuple, float] = {}
    for u in search_units:
        sens = float(sens_map.get(unit_parent(u), 0.0))
        sens_per_unit[u] = sens
        mac = float(op_macs_fn(*unit_parent(u)))
        for ell in spaces[u]:
            var_idx[(u, int(ell))] = len(rows)
            rows.append((sens, int(ell), mac, mac * float(ell), u))
    n_x = len(rows)
    s_pos_idx = n_x
    s_neg_idx = n_x + 1
    dev_pos_base = n_x + 2  # only used when use_dev
    dev_neg_base = n_x + 2 + n_units  # δ-_u  (or δ-_step1 in PWL mode)
    dev_neg2_base = n_x + 2 + 2 * n_units  # δ-_step2 (PWL only)
    n_dev_blocks = (3 if use_pwl_dn else 2) if use_dev else 0
    n_total = n_x + 2 + n_dev_blocks * n_units

    # ------------------------------------------------------------------
    # 2. Objective coefficients (we minimize).
    # ------------------------------------------------------------------
    util_terms = np.zeros(n_x)  # signed utility (sens*factor*ell), sign-flipped on insertion
    for i, (sens, ell, mac, _, _) in enumerate(rows):
        factor = mac if util_mode == "sens_mac" else 1.0
        util_terms[i] = sens * factor * ell

    c = np.zeros(n_total)
    if use_dev:
        # from_base (symmetric, cls-style): both upgrade (δ+) and downgrade
        # (δ-) cost = (1 + sens_u)·|Δ|. High-sens units are anchored at
        # base; low-sens / noise units are the cheap pool the ILP can
        # move when budget or floor constraints force deviation.
        #
        # PWL deep-cut penalty (use_pwl_dn): downgrade Δ split into
        #   step1 ∈ [0, deep_cut_threshold]                cost = (1+sens)·δ-_step1
        #   step2 ∈ [0, max_dev_below - deep_cut_threshold] cost = α·(1+sens)·δ-_step2
        # LP fills cheaper step1 first → no explicit ordering needed. With
        # α > 1, ILP prefers two-shallow-cuts over one-deep-cut except for
        # exceptionally low-sens units (math: α=1.2 → deep cut chosen iff
        # sens_min < 0.91·(1+sens_avg) − 1).
        for ui, u in enumerate(search_units):
            w = 1.0 + sens_per_unit[u]
            c[dev_pos_base + ui] = w   # upgrade (linear)
            c[dev_neg_base + ui] = w   # downgrade step1 (cheap)
            if use_pwl_dn:
                c[dev_neg2_base + ui] = float(deep_cut_penalty) * w  # step2 (penalty)
        # Auto big-M: dominate the worst-case total deviation cost.
        if util_dominate is None:
            scale = max(float(deep_cut_penalty), 1.0)
            max_dev = max(
                (max(spaces[u]) - min(spaces[u])) * (1.0 + abs(sens_per_unit[u]))
                for u in search_units
            )
            util_dominate = max(1.0, max_dev * scale * n_units * 100.0)
    else:
        # sens / sens_mac: minimize − Σ sens·factor·ℓ·x.
        c[:n_x] = -util_terms
        if util_dominate is None:
            max_possible_util = float(np.sum(np.abs(util_terms)))
            util_dominate = max(1.0, max_possible_util * 100.0)

    c[s_pos_idx] = util_dominate
    c[s_neg_idx] = util_dominate

    # ------------------------------------------------------------------
    # 3. one-of-K + budget + (optional) deviation linearization.
    #    Deviation linearization: for each unit u
    #        Σ_ℓ ℓ·x_{u,ℓ}  −  δ+_u  +  δ-_u  =  base_u
    #    Then |Σ_ℓ ℓ·x − base_u| = δ+_u + δ-_u  in any optimal soln.
    # ------------------------------------------------------------------
    n_eq_rows = n_units + 1 + (n_units if use_dev else 0)
    A = np.zeros((n_eq_rows, n_total))
    b = np.zeros(n_eq_rows)
    for ui, u in enumerate(search_units):
        for ell in spaces[u]:
            A[ui, var_idx[(u, int(ell))]] = 1.0
        b[ui] = 1.0
    budget_row = n_units
    for i, (_, _, _, bops, _) in enumerate(rows):
        A[budget_row, i] = bops
    A[budget_row, s_pos_idx] = -1.0
    A[budget_row, s_neg_idx] = +1.0
    b[budget_row] = float(target_bitops)
    if use_dev:
        for ui, u in enumerate(search_units):
            row = n_units + 1 + ui
            for ell in spaces[u]:
                A[row, var_idx[(u, int(ell))]] = float(ell)
            A[row, dev_pos_base + ui] = -1.0
            A[row, dev_neg_base + ui] = +1.0   # δ-_step1
            if use_pwl_dn:
                A[row, dev_neg2_base + ui] = +1.0  # δ-_step2
            base_u = int(base_levels[u])
            # Clip base to spaces (safety): can't anchor outside allowed range.
            if base_u < min(spaces[u]):
                base_u = min(spaces[u])
            elif base_u > max(spaces[u]):
                base_u = max(spaces[u])
            b[row] = float(base_u)

    # ------------------------------------------------------------------
    # 4. Hard floor for top-K most-sensitive units (optional, all modes).
    #    Build floor rows up-front so they merge into the same A matrix
    #    as the equality constraints. This avoids passing multiple
    #    LinearConstraint objects to scipy.optimize.milp, which trips a
    #    HiGHS malloc bug in scipy <1.17 (observed on gl1811's qwt_d2 env,
    #    scipy 1.15.2).
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
                continue
            floor_units.append(u)
    if floor_units:
        A_floor = np.zeros((len(floor_units), n_total))
        for fi, u in enumerate(floor_units):
            for ell in spaces[u]:
                if int(ell) >= int(floor_min_sl):
                    A_floor[fi, var_idx[(u, int(ell))]] = 1.0
        b_floor = np.ones(len(floor_units))
        # Merge into main A/b — floor rows are equality (= 1).
        A = np.vstack([A, A_floor])
        b = np.concatenate([b, b_floor])
    constraints = [LinearConstraint(A, b, b)]

    # ------------------------------------------------------------------
    # 5. Bounds + integrality (binary x, continuous slack).
    # ------------------------------------------------------------------
    integrality = np.zeros(n_total)
    integrality[:n_x] = 1.0  # 1 = integer
    ub = np.full(n_total, np.inf)
    ub[:n_x] = 1.0           # x_{u,ℓ} ∈ [0, 1] (with integrality → {0, 1})
    lb = np.zeros(n_total)
    if use_pwl_dn:
        # Cap step1 at deep_cut_threshold and step2 at the remaining
        # downgrade range (base_u - min_level - threshold). Per-unit
        # bounds — a unit whose smallest space level is base_u itself
        # will simply not downgrade at all (ub2 = 0).
        thr = float(deep_cut_threshold)
        for ui, u in enumerate(search_units):
            base_u = int(base_levels[u])
            if base_u < min(spaces[u]):
                base_u = min(spaces[u])
            elif base_u > max(spaces[u]):
                base_u = max(spaces[u])
            max_dev_below = max(0.0, float(base_u) - float(min(spaces[u])))
            ub[dev_neg_base + ui] = min(thr, max_dev_below)
            ub[dev_neg2_base + ui] = max(0.0, max_dev_below - thr)
    bounds = Bounds(lb=lb, ub=ub)

    # ------------------------------------------------------------------
    # 6. Solve.
    # ------------------------------------------------------------------
    import os as _os
    import time as _time
    # Threads and gap come from env so tuning doesn't require code changes.
    _n_threads = int(_os.environ.get("MILP_THREADS",
                                     min(8, max(1, (_os.cpu_count() or 4) // 2))))
    _rel_gap = float(_os.environ.get("MILP_REL_GAP", "1e-3"))
    _time_limit = float(_os.environ.get("MILP_TIME_LIMIT", "120.0"))

    # Try highspy direct (HiGHS Python bindings) so we can actually pass
    # parallel/threads — scipy.optimize.milp's options dict silently
    # drops those keys (verified: scipy 1.17 only honors a small subset).
    # Fallback to scipy.milp if highspy isn't installed in this env.
    try:
        import highspy  # type: ignore
        _have_highspy = True
    except ImportError:
        _have_highspy = False

    t0 = _time.time()
    if _have_highspy:
        from scipy.sparse import csc_matrix
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("parallel", "on")
        h.setOptionValue("threads", _n_threads)
        h.setOptionValue("presolve", "on")
        h.setOptionValue("mip_rel_gap", _rel_gap)
        h.setOptionValue("time_limit", _time_limit)
        # Build LP (minimize). Constraints are equalities A x = b.
        A_csc = csc_matrix(A)
        lp = highspy.HighsLp()
        lp.num_col_ = n_total
        lp.num_row_ = A.shape[0]
        lp.col_cost_ = c.tolist()
        lp.col_lower_ = lb.tolist()
        lp.col_upper_ = ub.tolist()
        lp.row_lower_ = b.tolist()
        lp.row_upper_ = b.tolist()
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A_csc.indptr.astype(int).tolist()
        lp.a_matrix_.index_ = A_csc.indices.astype(int).tolist()
        lp.a_matrix_.value_ = A_csc.data.astype(float).tolist()
        lp.integrality_ = [
            highspy.HighsVarType.kInteger if g >= 0.5
            else highspy.HighsVarType.kContinuous
            for g in integrality
        ]
        h.passModel(lp)
        run_status = h.run()
        sol = h.getSolution()
        info = h.getInfo()
        model_status = h.getModelStatus()
        ok_statuses = {highspy.HighsModelStatus.kOptimal,
                       highspy.HighsModelStatus.kTimeLimit}
        success = model_status in ok_statuses and len(sol.col_value) == n_total

        class _Res:
            pass
        res = _Res()
        res.success = success
        res.x = np.array(sol.col_value) if success else None
        res.message = f"HiGHS status: {model_status.name}"
        backend = f"highspy (threads={_n_threads}, gap≤{_rel_gap})"
    else:
        options = {"presolve": "on", "mip_rel_gap": _rel_gap,
                   "time_limit": _time_limit}
        res = milp(c, constraints=constraints, integrality=integrality,
                   bounds=bounds, options=options)
        backend = "scipy.optimize.milp (no parallel — install highspy)"
    solve_time = _time.time() - t0
    if not res.success:
        raise RuntimeError(
            f"ILP init failed: {res.message}. backend={backend}, "
            f"util_mode={util_mode}, n_vars={n_total}, "
            f"n_constraints={sum(con.A.shape[0] for con in constraints)}, "
            f"target_bitops={target_bitops:.3g}, floor_top_k={floor_top_k}, "
            f"floor_min_sl={floor_min_sl}."
        )

    # ------------------------------------------------------------------
    # 7. Decode solution.
    # ------------------------------------------------------------------
    x = res.x[:n_x]
    s_pos = float(res.x[s_pos_idx])
    s_neg = float(res.x[s_neg_idx])

    assignment: dict[tuple, int] = {}
    chosen_bops = 0.0
    utility = 0.0
    total_dev = 0.0
    for (u, ell), i in var_idx.items():
        if x[i] > 0.5:
            assignment[u] = int(ell)
            chosen_bops += rows[i][3]
            if not use_dev:
                utility += util_terms[i]

    if use_dev:
        # utility for symmetric from_base = − Σ (1+sens_u)·(δ+ + δ-_step1 + α·δ-_step2)
        # Negative; closer to 0 = better.
        alpha = float(deep_cut_penalty) if use_pwl_dn else 1.0
        for ui, u in enumerate(search_units):
            dev_pos = float(res.x[dev_pos_base + ui])
            dev_neg1 = float(res.x[dev_neg_base + ui])
            dev_neg2 = float(res.x[dev_neg2_base + ui]) if use_pwl_dn else 0.0
            total_dev += dev_pos + dev_neg1 + dev_neg2
            utility -= (1.0 + sens_per_unit[u]) * (dev_pos + dev_neg1 + alpha * dev_neg2)

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
        "backend": backend,
        "budget_slack_pos": s_pos,
        "budget_slack_neg": s_neg,
        "budget_gap_signed": s_neg - s_pos,
        "solve_time_s": round(solve_time, 4),
        "utility": float(utility),
        "chosen_bitops": float(chosen_bops),
        "target_bitops": float(target_bitops),
    }
    if use_dev:
        meta["total_abs_deviation_sl"] = float(total_dev)
        meta["mean_abs_deviation_sl"] = float(total_dev / max(1, n_units))
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

    # Symmetric from_base: any deviation costs (1+sens)·|Δ|. Sens-low units
    # are cheap to move; sens-high are anchored. base = 128, target = base
    # × n → "stay at base" is cost-0 trivial optimum.
    base_levels = {u: 128 for u in units}
    ilp_assign4, ilp_bops4, ilp_util4, ilp_meta4 = ilp_init_levels(
        search_units=units, spaces=spaces, target_bitops=target_bitops,
        sens_map=sens_map, op_macs_fn=lambda op, bi: mac,
        util_mode="from_base", base_levels=base_levels,
    )
    ilp_sum4 = sum(ilp_assign4[u] for u in units)
    print(f"[smoke] ILP-fb (sym, equal-MAC) sum={ilp_sum4}  "
          f"util={ilp_util4:.4f}  assign={ilp_assign4}")
    assert all(v == 128 for v in ilp_assign4.values()), (
        f"expected all-128 in degenerate case, got {ilp_assign4}")
    print("[smoke] ✓ sym from_base trivially holds at base when budget == base × n.")

    # Symmetric direction check: sens-low is cheaper to move regardless of
    # direction. Force deviations via off-budget targets.
    units_dir = [("hi", i) for i in range(3)] + [("lo", i) for i in range(3)]
    sens_dir = {("hi", i): 0.5 for i in range(3)}
    sens_dir.update({("lo", i): -0.2 for i in range(3)})  # (1+sens) > 0
    spaces_dir = {u: [64, 96, 128, 192, 256] for u in units_dir}
    base_dir = {u: 128 for u in units_dir}

    # (a) Forced upgrade: budget = (128*6 + 64) × 1G — sym picks LO (cheap).
    bops_up = mac * (6 * 128 + 64)
    ilp_a, _, _, _ = ilp_init_levels(
        search_units=units_dir, spaces=spaces_dir, target_bitops=bops_up,
        sens_map=sens_dir, op_macs_fn=lambda op, bi: mac,
        util_mode="from_base", base_levels=base_dir,
    )
    upgraders = [u for u in units_dir if ilp_a[u] > 128]
    print(f"[smoke] ILP-fb sym up-shift: assign={ilp_a}, upgraders={upgraders}")
    assert all(u[0] == "lo" for u in upgraders), (
        f"sym anchor: expected lo-sens (cheap) upgrade, got {upgraders}")
    print(f"[smoke] ✓ symmetric from_base routes forced-upgrade to sens-low (cheap)")

    # (b) Forced downgrade: budget = (128*6 − 64) × 1G — sym picks LO (cheap).
    bops_dn = mac * (6 * 128 - 64)
    ilp_b, _, _, _ = ilp_init_levels(
        search_units=units_dir, spaces=spaces_dir, target_bitops=bops_dn,
        sens_map=sens_dir, op_macs_fn=lambda op, bi: mac,
        util_mode="from_base", base_levels=base_dir,
    )
    downgraders = [u for u in units_dir if ilp_b[u] < 128]
    print(f"[smoke] ILP-fb sym dn-shift: assign={ilp_b}, downgraders={downgraders}")
    assert all(u[0] == "lo" for u in downgraders), (
        f"sym anchor: expected lo-sens (cheap) downgrade, got {downgraders}")
    print(f"[smoke] ✓ symmetric from_base routes forced-downgrade to sens-low (cheap)")


if __name__ == "__main__":
    _smoke_test_equal_mac()
