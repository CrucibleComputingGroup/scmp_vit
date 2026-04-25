"""Build linear/attn MP-spec dicts from CLI args (det side).

Mirror of ``cls/eval.py``'s ``_build_linear_mp_spec`` / ``_build_attn_mp_spec``,
adapted to det's argparse names. Produces specs that ``det/sc_patch/
sc_model_eva.py:sc_patch_eva`` accepts as ``linear_mp_spec`` / ``attn_mp_spec``.

The spec types come from the shared ``sc_integration.mp_linear`` package
(``MPConfig`` for fixed multi-level, ``RangeMPConfig`` for range-based
per-weight-group, ``AdaptiveMPConfig`` for time-varying — kept here for
parity even though ViTDet has no native timestep).
"""
from __future__ import annotations


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_csv_ops(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def add_mp_args(p):
    """Register the MP CLI flags on an argparse.ArgumentParser ``p``.

    Mirrors the cls-side flag set so ``det/sc_eval.py`` and
    ``det/experiments/qwt_det_compensate.py`` accept the same MP CLI
    surface as ``cls/eval.py`` and ``cls/experiments/qwt_sc_overnight.py``.
    """
    p.add_argument("--mp_levels", default="",
                   help="Comma-separated SC stoc_lens for fixed multi-level MP "
                        "on linear ops (qkv_proj/out_proj/mlp_fc1/mlp_fc2). "
                        "Empty = disabled.")
    p.add_argument("--mp_fractions", default="",
                   help="Comma-separated row fractions matching --mp_levels.")
    p.add_argument("--mp_ops", default="",
                   help="Comma-separated linear op names targeted by "
                        "--mp_levels. Accepts mlp_fc1, mlp_fc2, qkv_proj, "
                        "out_proj, or 'proj' as alias for both proj ops.")
    p.add_argument("--qk_mp_levels", default="",
                   help="Stoc_lens for QK matmul MP. Empty = disabled.")
    p.add_argument("--qk_mp_fractions", default="")
    p.add_argument("--av_mp_levels", default="",
                   help="Stoc_lens for AV matmul MP. Empty = disabled.")
    p.add_argument("--av_mp_fractions", default="")
    # Range-based MP (per-weight-group routing by magnitude)
    p.add_argument("--range_mp", type=int, default=0)
    p.add_argument("--range_mp_levels", default="256,128")
    p.add_argument("--range_mp_threshold", type=float, default=0.3)
    p.add_argument("--range_mp_ops", default="")
    p.add_argument("--range_mp_group_size", type=int, default=0)
    # Adaptive MP (time-varying; ViTDet has no native timestep but kept for
    # parity with the cls CLI surface)
    p.add_argument("--adaptive_mp", type=int, default=0)
    p.add_argument("--adaptive_mp_levels", default="")
    p.add_argument("--adaptive_mp_alpha", type=float, default=0.3)
    p.add_argument("--adaptive_mp_beta", type=float, default=0.05)
    p.add_argument("--adaptive_mp_enable_pruning", type=int, default=1)
    p.add_argument("--adaptive_mp_ops", default="")


def build_attn_mp_spec(args) -> dict:
    """{"qk": MPConfig|AdaptiveMPConfig, "av": ...} from CLI."""
    from sc_integration.mp_linear import MPConfig, AdaptiveMPConfig
    spec: dict = {}
    adaptive_ops = (set(_parse_csv_ops(args.adaptive_mp_ops))
                    if bool(args.adaptive_mp) else set())
    adaptive_levels = (_parse_csv_ints(args.adaptive_mp_levels)
                       if bool(args.adaptive_mp) and args.adaptive_mp_levels
                       else [])
    adaptive_cfg = (
        AdaptiveMPConfig(
            stoc_len_levels=adaptive_levels,
            alpha=args.adaptive_mp_alpha,
            beta=args.adaptive_mp_beta,
            enable_pruning=bool(args.adaptive_mp_enable_pruning))
        if adaptive_levels else None)

    if "qk" in adaptive_ops and adaptive_cfg is not None:
        spec["qk"] = adaptive_cfg
    elif args.qk_mp_levels:
        spec["qk"] = MPConfig(
            stoc_len_levels=_parse_csv_ints(args.qk_mp_levels),
            level_fractions=(_parse_csv_floats(args.qk_mp_fractions)
                             if args.qk_mp_fractions else None),
        )
    if "av" in adaptive_ops and adaptive_cfg is not None:
        spec["av"] = adaptive_cfg
    elif args.av_mp_levels:
        spec["av"] = MPConfig(
            stoc_len_levels=_parse_csv_ints(args.av_mp_levels),
            level_fractions=(_parse_csv_floats(args.av_mp_fractions)
                             if args.av_mp_fractions else None),
        )
    return spec


def build_linear_mp_spec(args) -> dict:
    """{op_name: {"fixed": MPConfig, "adaptive": AdaptiveMPConfig,
    "range": RangeMPConfig, "range_group_size": int}} from CLI flags."""
    from sc_integration.mp_linear import (
        MPConfig, AdaptiveMPConfig, RangeMPConfig,
    )

    fixed_levels = _parse_csv_ints(args.mp_levels) if args.mp_levels else []
    fixed_fracs = (_parse_csv_floats(args.mp_fractions)
                   if args.mp_fractions else None)
    fixed_ops = _parse_csv_ops(args.mp_ops)
    range_on = bool(args.range_mp)
    range_levels = _parse_csv_ints(args.range_mp_levels) if range_on else []
    range_ops = _parse_csv_ops(args.range_mp_ops) or (fixed_ops if range_on else [])
    adaptive_on = bool(args.adaptive_mp)
    adaptive_levels = (_parse_csv_ints(args.adaptive_mp_levels)
                       if adaptive_on else [])
    _raw_adaptive_ops = (_parse_csv_ops(args.adaptive_mp_ops)
                         or (fixed_ops if adaptive_on else []))
    # qk/av are routed through build_attn_mp_spec, not here.
    adaptive_ops = [op for op in _raw_adaptive_ops if op not in ("qk", "av")]

    mp_fixed = (MPConfig(stoc_len_levels=fixed_levels,
                         level_fractions=fixed_fracs)
                if fixed_levels else None)
    mp_range = (RangeMPConfig(stoc_len_levels=range_levels,
                              base_threshold=args.range_mp_threshold)
                if range_on else None)
    mp_adaptive = (AdaptiveMPConfig(
                       stoc_len_levels=adaptive_levels,
                       alpha=args.adaptive_mp_alpha,
                       beta=args.adaptive_mp_beta,
                       enable_pruning=bool(args.adaptive_mp_enable_pruning))
                   if adaptive_levels else None)

    if not mp_fixed and not mp_range and not mp_adaptive:
        return {}

    spec: dict = {}
    if mp_fixed:
        for op in fixed_ops:
            spec.setdefault(op, {})["fixed"] = mp_fixed
    if mp_adaptive:
        for op in adaptive_ops:
            spec.setdefault(op, {})["adaptive"] = mp_adaptive
    if mp_range:
        for op in range_ops:
            entry = spec.setdefault(op, {})
            entry["range"] = mp_range
            entry["range_group_size"] = args.range_mp_group_size
    return spec


def mp_args_to_dict(args) -> dict:
    """Snapshot of MP CLI flags suitable for the JSON 'config' block."""
    return {
        "mp_levels": args.mp_levels,
        "mp_fractions": args.mp_fractions,
        "mp_ops": args.mp_ops,
        "qk_mp_levels": args.qk_mp_levels,
        "qk_mp_fractions": args.qk_mp_fractions,
        "av_mp_levels": args.av_mp_levels,
        "av_mp_fractions": args.av_mp_fractions,
        "range_mp": int(args.range_mp),
        "range_mp_levels": args.range_mp_levels,
        "range_mp_threshold": args.range_mp_threshold,
        "range_mp_ops": args.range_mp_ops,
        "range_mp_group_size": args.range_mp_group_size,
        "adaptive_mp": int(args.adaptive_mp),
        "adaptive_mp_levels": args.adaptive_mp_levels,
        "adaptive_mp_alpha": args.adaptive_mp_alpha,
        "adaptive_mp_beta": args.adaptive_mp_beta,
        "adaptive_mp_enable_pruning": int(args.adaptive_mp_enable_pruning),
        "adaptive_mp_ops": args.adaptive_mp_ops,
    }
