"""Stochastic-computing integration for ViT models, backed by ``scmp_kernels``.

This replaces the retired in-tree SC stack (``sc/`` + ``sc_integration/``, now
under ``deprecated/``). Every SC matmul goes through the single unified
``scmp_kernels.sc_matmul`` entry point:

===============  ==========================================================
op               call
===============  ==========================================================
Q @ K^T          ``sc_matmul(q, k, granularity="per_head")``   3D, per-head
                 (max, min) — one call for all (B, H).
attn @ V         ``sc_matmul(a, v, granularity="per_row")``    3D batched —
                 one call for all (B, H); the legacy code looped B*H times.
nn.Linear        ``sc_matmul(x, w, granularity="per_row", chunk_d=...)``
                 ``chunk_d`` keeps the cum_indicator table in L2 for wide
                 MLP layers (ViT-L fc2 has D_in=4096).
===============  ==========================================================

Softmax, layernorms, activations and the classifier head stay FP.

Mixed precision reuses the kernel-side dispatch (``scmp_kernels.mp``) rather
than a vit_sc-local fork, so calibration tables produced for scmp_llm/scmp_cnn
transfer here unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ._bootstrap import ensure_kernels_on_path, kernels_provenance  # noqa: F401

ensure_kernels_on_path()

from scmp_kernels import sc_matmul                              # noqa: E402
from scmp_kernels import trace as sc_trace                      # noqa: E402
from scmp_kernels.mp import (                                   # noqa: E402
    MPConfig,
    AdaptiveMPConfig,
    RangeMPConfig,
    compute_row_metric,
    classify_rows_by_metric,
    adaptive_classify_rows,
    classify_groups_by_range,
    AutoMPBudgetLogger,
    set_current_block_idx,
    get_current_block_idx,
)

__all__ = [
    "SCLinear", "patch_model", "parse_sc_ops", "normalize_sc_ops_per_block",
    "SC_OP_NAMES", "LINEAR_OP_NAMES", "make_sc_attention_forward",
    "set_av_metric_kind", "set_qk_granularity", "set_sc_defaults", "sc_defaults",
    "set_vit_timestep", "get_vit_timestep", "load_adaptive_mp",
    "read_mp_table_provenance", "set_group_stoclen", "get_group_stoclen",
    "MPConfig", "AdaptiveMPConfig", "RangeMPConfig",
]


# ---------------------------------------------------------------------------
# Global SC defaults (knobs that are not per-module)
# ---------------------------------------------------------------------------

_DEFAULTS = {
    # D-chunking for the per_row linear path. This is NOT just a cache
    # optimization — it is REQUIRED for correctness on wide MLP layers.
    # Measured 2026-07-25 on DINOv2 ViT-L/14, SC on mlp_fc1+fc2, sc_prec=8,
    # stoc_len=256, 256 images:  chunk_d=128 -> top-1 0.8438 (== FP),
    # chunk_d=0 -> top-1 0.0000. The unchunked path accumulates the whole
    # D_in=4096 against one cum_indicator table, and post-GELU activations put
    # ~2/3 of their mass at |x_int| <= 2, so the SC noise on the near-zero bulk
    # dominates and the block output norm inflates (~2.5x by block 22),
    # compounding through the residual stream until logits are garbage.
    # This is the documented "SC on MLP collapses to ~0%" failure in the old
    # README — it was a property of the UNCHUNKED path, not of SC on MLP.
    "chunk_d": 128,
    # Width past which running unchunked is known to be dangerous.
    "chunk_d_warn_width": 1024,
    # uSystolic/HUB sign-magnitude cycle halving. Bipolar magnitudes only carry
    # sc_prec-1 bits, so a half-length stream is free. OFF by default to keep
    # numbers comparable with the legacy vit_sc runs.
    "halve_bipolar_stoc_len": False,
}


def set_sc_defaults(**kw):
    """Override global SC defaults (``chunk_d``, ``halve_bipolar_stoc_len``)."""
    for k, v in kw.items():
        if k not in _DEFAULTS:
            raise KeyError(f"unknown SC default {k!r}; valid: {sorted(_DEFAULTS)}")
        _DEFAULTS[k] = v


def sc_defaults() -> dict:
    return dict(_DEFAULTS)


def _resolve_stoc_len(stoc_len, sc_prec: int) -> int:
    """Effective UNIFORM stream length the kernel will actually run.

    ``None`` resolves to the grid ceiling, and an explicit request is clamped to
    it. Under ``halve_bipolar_stoc_len`` the ceiling is ``2**(sc_prec-1)``, not
    ``2**sc_prec``: a bipolar magnitude only spans ``sc_prec-1`` bits, so the
    kernel drops ``rng_levels`` to the halved grid and a longer stream buys no
    resolution — it just burns ~2x the cycles and records them in the precision
    trace. Mirrors scmp_llm's min-clamp (model/sc_common.py), which this port
    was missing: ``--halve 1`` with no ``--stoc_len`` used to issue a 256-cycle
    stream on a 128-level grid.

    MP ladder rungs are NOT routed through here — like scmp_llm they are passed
    to the kernel verbatim, already in halved space, and validated against the
    cap by the calibrator and by ``load_adaptive_mp``.
    """
    cap = 2 ** (sc_prec - 1) if _DEFAULTS["halve_bipolar_stoc_len"] else 2 ** sc_prec
    if stoc_len is None:
        return int(cap)
    return int(min(int(stoc_len), cap))


# Calibration tables are bucketed by (operator, timestep, layer). A ViT has no
# native timestep, so it stays at (0, 1) — the single-bucket case — unless a
# caller drives it explicitly (kept for parity with the diffusion consumers).
_VIT_TIMESTEP = [0, 1]
_TOTAL_BLOCKS: int | None = None
_CHUNK_WARNED: set = set()
_SC_ENABLED = [True]
_EMPTY_LONG = torch.empty(0, dtype=torch.long)


def _channel_index_tensor(indices, width: int, device) -> torch.Tensor:
    """Sorted, de-duplicated, in-range protected-channel indices."""
    vals = sorted({int(i) for i in (indices or []) if 0 <= int(i) < width})
    if not vals:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(vals, dtype=torch.long, device=device)


def _complement_channel_indices(width: int, selected: torch.Tensor,
                                device) -> torch.Tensor:
    if selected.numel() == 0:
        return torch.arange(width, dtype=torch.long, device=device)
    mask = torch.ones(width, dtype=torch.bool, device=device)
    mask[selected] = False
    return mask.nonzero(as_tuple=True)[0]


def set_sc_enabled(flag: bool):
    """Globally bypass SC in every patched module (falls back to FP).

    The patched modules stay in place, so hooks registered on them keep firing
    — which is what the MP calibrator needs to measure an FP-trajectory
    ablation without rebuilding the model. Mirrors scmp_llm's
    ``config.use_sc_linear``.
    """
    _SC_ENABLED[0] = bool(flag)


def sc_enabled() -> bool:
    return _SC_ENABLED[0]


# Calibration-only per-(operator, block) UNIFORM stoc_len override. Mirrors
# scmp_llm's ``config.sc_group_stoclen``: used by the measured-ΔLoss knock-down
# probe to force ONE group to a different stream length while everything else
# stays at its own ``stoc_len``, with no MP dispatch. A pure forward measurement.
_GROUP_STOCLEN: list = [None]


def set_group_stoclen(mapping: dict | None):
    """Force ``{(operator, block_idx): stoc_len}`` uniformly, bypassing MP.

    Keys absent from the map fall back to the module's own ``stoc_len``, so a
    probe only has to name the group it knocks down. ``None`` clears the
    override. ``stoc_len <= 0`` zeroes that group's output (issues no matmul).
    """
    _GROUP_STOCLEN[0] = dict(mapping) if mapping else None


def get_group_stoclen() -> dict | None:
    return _GROUP_STOCLEN[0]


def total_blocks() -> int | None:
    """Block count recorded by the last ``patch_model`` call."""
    return _TOTAL_BLOCKS


def set_vit_timestep(t: int, total: int | None = None):
    """Set the (timestep, total_timesteps) pair used for table bucketing."""
    _VIT_TIMESTEP[0] = int(t)
    if total is not None:
        _VIT_TIMESTEP[1] = max(int(total), 1)


def get_vit_timestep() -> tuple[int, int]:
    return tuple(_VIT_TIMESTEP)


def _resolve_mp_table_path(path: str):
    """(wrapper_path, table_path|None, wrapper_spec) for an MP wrapper JSON.

    Relative ``threshold_table_path`` values resolve against the wrapper's own
    directory, so a calibration bundle stays relocatable.
    """
    import json
    from pathlib import Path

    wrapper_path = Path(path).expanduser().resolve()
    with open(wrapper_path) as f:
        spec = json.load(f)
    table = spec.get("threshold_table_path")
    table_path = None
    if table:
        table_path = Path(table)
        if not table_path.is_absolute():
            table_path = wrapper_path.parent / table_path
    return wrapper_path, table_path, spec


def read_mp_table_provenance(path: str) -> dict:
    """Calibration provenance recorded in an MP wrapper's threshold table.

    Returns the subset of ``calibrate_mp_thresholds.py``'s payload that a
    deployment has to agree with (``sc_prec``, ``halve_bipolar_stoc_len``,
    ``chunk_d``, ``stoc_len_levels``, ``method``, ``total_blocks``), or ``{}``
    for a fractions-only wrapper that carries no table. Cheap enough to call
    before building the model — it only parses JSON.
    """
    import json

    _wrapper, table_path, _spec = _resolve_mp_table_path(path)
    if table_path is None:
        return {}
    with open(table_path) as f:
        payload = json.load(f)
    keys = ("sc_prec", "halve_bipolar_stoc_len", "chunk_d", "stoc_len_levels",
            "method", "total_blocks", "objective", "cross_layer_weight",
            "qk_granularity", "group_ladder_k")
    return {k: payload[k] for k in keys if k in payload}


def load_adaptive_mp(path: str, *,
                     expect_sc_prec: int | None = None,
                     expect_halve: bool | None = None,
                     expect_chunk_d: int | None = None) -> AdaptiveMPConfig:
    """Build an ``AdaptiveMPConfig`` from a scmp_llm-style wrapper JSON.

    The wrapper names the ladder plus the calibrated table produced by
    ``calibrate_mp_thresholds.py``::

        {"type": "AdaptiveMPConfig", "stoc_len_levels": [128, 64, 32],
         "threshold_table_path": "<table>.json"}

    ``expect_*`` are the deploying run's SC settings. When given, they are
    checked against the table's recorded calibration settings and a mismatch
    RAISES. Thresholds are fitted on a specific quantization grid: deploying a
    ``halve=1`` table at ``halve=0`` reinterprets every ladder rung as a nominal
    (unhalved) cycle count, which changes both the RNG grid the sigmas were
    measured on and the realized compute — silently, since the run still
    produces plausible accuracy. ``cls/eval.py`` used to default ``--halve 0``
    while the calibrator defaulted ``--halve 1``, so the default pair was itself
    a mismatch; this is the guard that makes that impossible.
    """
    wrapper_path, table_path, spec = _resolve_mp_table_path(path)
    levels = [int(x) for x in spec["stoc_len_levels"]]
    # Absolute escape gate (R7): rows whose NORMALIZED dispatch metric exceeds
    # mu_b + k*sigma_b escape to ``escape_stoc_len`` regardless of the band
    # thresholds. mu/sigma are the per-bucket ``metric_mean``/``metric_std``
    # the calibrator already writes into every bucket, so an existing table
    # needs no recalibration to gain the gate — only the wrapper changes.
    # Absent ⇒ None ⇒ byte-identical to the pre-gate classifier.
    # The gate is applied AFTER the provenance checks below, by re-running
    # load_threshold_table with escape_gate_k set: __post_init__ rejects a gate
    # without a table, and this loader deliberately validates the table's
    # sc_prec/halve/chunk_d against the deployment BEFORE reading it.
    escape_gate_k = spec.get("escape_gate_k")
    cfg = AdaptiveMPConfig(
        stoc_len_levels=levels,
        enable_pruning=bool(spec.get("enable_pruning", True)),
        target_fractions=spec.get("target_fractions"),
    )
    if table_path is not None:
        prov = read_mp_table_provenance(str(wrapper_path))
        checks = (
            ("sc_prec", expect_sc_prec, prov.get("sc_prec"), int),
            ("halve_bipolar_stoc_len", expect_halve,
             prov.get("halve_bipolar_stoc_len"), bool),
            ("chunk_d", expect_chunk_d, prov.get("chunk_d"), int),
        )
        for name, want, got, cast in checks:
            if want is None or got is None:
                continue
            if cast(want) != cast(got):
                raise ValueError(
                    f"{table_path.name}: MP table was calibrated with "
                    f"{name}={cast(got)!r} but this run deploys "
                    f"{name}={cast(want)!r}. Calibrated thresholds and ladder "
                    f"rungs do not transfer across that change — re-run the "
                    f"calibration, or match the deployment to the table.")
        halve = prov.get("halve_bipolar_stoc_len")
        sc_prec = prov.get("sc_prec")
        if halve is not None and sc_prec is not None and bool(halve):
            cap = 2 ** (int(sc_prec) - 1)
            if max(levels) > cap:
                raise ValueError(
                    f"{table_path.name}: halved ladder top rung {max(levels)} "
                    f"exceeds the halved grid cap {cap} for sc_prec="
                    f"{int(sc_prec)}. Ladder rungs are cycle counts in halved "
                    f"space; a rung above the cap runs a longer stream on the "
                    f"same grid for no resolution gain.")
        cfg.load_threshold_table(str(table_path))
        if escape_gate_k is not None:
            cfg.escape_gate_k = float(escape_gate_k)
            cfg.escape_stoc_len = int(spec.get("escape_stoc_len", max(levels)))
            if cfg.escape_stoc_len <= 0:
                raise ValueError(
                    f"{wrapper_path}: escape_stoc_len must be a positive cycle "
                    f"count, got {cfg.escape_stoc_len}")
            cfg.threshold_table_path = str(table_path)
            # Recompute t_esc = mu_b + k*sigma_b per bucket now that k is set.
            cfg.load_threshold_table(str(table_path))
            if not (cfg.bucket_escape_thresholds
                    or cfg.operator_default_escape_thresholds):
                raise ValueError(
                    f"{table_path.name}: escape_gate_k={escape_gate_k} was "
                    f"requested but the table carries no metric_mean/metric_std "
                    f"in any bucket — the gate constants cannot be derived. "
                    f"Recalibrate, or drop escape_gate_k from the wrapper.")
    elif escape_gate_k is not None:
        raise ValueError(
            f"{wrapper_path}: escape_gate_k needs a calibrated threshold table "
            f"(the gate constants are the table's per-bucket metric stats); a "
            f"fractions-only wrapper has none.")
    elif not cfg.target_fractions:
        raise ValueError(
            f"{wrapper_path}: AdaptiveMPConfig needs 'threshold_table_path' or "
            f"'target_fractions' — the kernel has no closed-form threshold mode.")
    return cfg


# ---------------------------------------------------------------------------
# SCLinear
# ---------------------------------------------------------------------------

class SCLinear(nn.Module):
    """Drop-in SC replacement that *wraps* an existing ``nn.Linear``.

    Weight/bias tensors are reused (not copied), so this can be swapped onto a
    pretrained model in place: ``mlp.fc1 = SCLinear(mlp.fc1, sc_prec)``.

    Mixed-precision knobs (all optional, evaluated in this order):

    * ``range_mp_cfg`` — per-**weight**-row-group levels, dispatched once at
      construction time from the wrapped weight (static, input-independent).
    * ``adaptive_mp_cfg`` — per-**input**-row levels from a calibrated
      threshold table (``AdaptiveMPConfig``); wins over ``mp_cfg``.
    * ``mp_cfg`` — per-**input**-row fixed-quantile levels (``MPConfig``).

    With none of them set the layer runs uniform SC at ``stoc_len``.
    """

    def __init__(self, linear: nn.Linear, sc_prec: int = 8,
                 mode: str = "bipolar",
                 *,
                 stoc_len: int | None = None,
                 chunk_d: int | None = None,
                 mp_cfg: MPConfig | None = None,
                 adaptive_mp_cfg: AdaptiveMPConfig | None = None,
                 range_mp_cfg: RangeMPConfig | None = None,
                 range_mp_group_size: int = 0,
                 operator: str | None = None,
                 block_idx: int | None = None,
                 metric: str = "amax",
                 metric_sign: float = 1.0,
                 protect_spec=None):
        super().__init__()
        # (channel indices, stoc_len) for the no-MP protected-channel path.
        # Ignored when an MP config is attached — that path reads the protected
        # set off the table instead, so the two can never disagree.
        self.protect_spec = protect_spec
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.sc_prec = int(sc_prec)
        self.mode = mode
        self.stoc_len = stoc_len
        self.chunk_d = _DEFAULTS["chunk_d"] if chunk_d is None else int(chunk_d)

        self.mp_cfg = mp_cfg
        self.adaptive_mp_cfg = adaptive_mp_cfg
        self.range_mp_cfg = range_mp_cfg
        self.range_mp_group_size = int(range_mp_group_size)
        self.operator = operator
        self.block_idx = block_idx
        self.metric = metric
        self.metric_sign = float(metric_sign)

        # Loud warning for the known-catastrophic config (see _DEFAULTS above).
        # Silent 0% accuracy is the worst possible failure mode, so say it once
        # per (operator, width) rather than let a sweep quietly produce garbage.
        if (self.mode == "bipolar" and self.chunk_d <= 0
                and self.in_features >= _DEFAULTS["chunk_d_warn_width"]):
            key = (operator, self.in_features)
            if key not in _CHUNK_WARNED:
                _CHUNK_WARNED.add(key)
                import warnings
                warnings.warn(
                    f"SCLinear({operator}, D_in={self.in_features}) built with "
                    f"chunk_d=0. Unchunked SC on wide layers is known to "
                    f"collapse accuracy to ~0% on ViT MLPs. Pass chunk_d=128 "
                    f"unless you are deliberately reproducing that failure.",
                    RuntimeWarning, stacklevel=2)

        # Static per-weight-group dispatch is input-independent: resolve once.
        # classify_groups_by_range returns one stoc_len per group, where group g
        # owns the contiguous output rows [g*gsize, (g+1)*gsize).
        self._range_groups = None
        self._range_group_size = 0
        if range_mp_cfg is not None:
            gsize = (self.range_mp_group_size
                     if 0 < self.range_mp_group_size < self.out_features
                     else self.out_features)
            self._range_group_size = gsize
            self._range_groups = list(classify_groups_by_range(
                self.weight.detach(), gsize, range_mp_cfg, operator=operator))

    @property
    def mac_per_row(self) -> float:
        """MACs one input row costs. Static for a ViT (no KV-cache growth), so
        the MP calibrator can price the compute budget analytically instead of
        reading a runtime trace the way the LLM has to."""
        return float(self.in_features) * float(self.out_features)

    def extra_repr(self) -> str:
        kind = ("range" if self.range_mp_cfg is not None else
                "adaptive" if self.adaptive_mp_cfg is not None else
                "fixed-mp" if self.mp_cfg is not None else "uniform")
        return (f"in={self.in_features}, out={self.out_features}, "
                f"sc_prec={self.sc_prec}, mode={self.mode}, "
                f"stoc_len={_resolve_stoc_len(self.stoc_len, self.sc_prec)}, "
                f"mp={kind}, op={self.operator}")

    # -- kernel call ------------------------------------------------------
    def _mm(self, x2: torch.Tensor, w: torch.Tensor, stoc_len: int):
        """One SC matmul at an explicit stream length. ``x2`` is 2D."""
        return sc_matmul(
            x2, w,
            granularity="per_row", mode=self.mode, sc_prec=self.sc_prec,
            stoc_len=int(stoc_len), group_a=1, group_b=1,
            chunk_d=(self.chunk_d if (self.chunk_d > 0
                                      and self.mode == "bipolar"
                                      and x2.shape[-1] > self.chunk_d) else 0),
            halve_bipolar_stoc_len=_DEFAULTS["halve_bipolar_stoc_len"],
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _SC_ENABLED[0]:
            return nn.functional.linear(x, self.weight, self.bias)
        orig_shape = x.shape
        orig_dtype = x.dtype
        x2 = x.reshape(-1, orig_shape[-1]).float().contiguous()
        w = self.weight.float().contiguous()
        if x2.shape[0] == 0:
            out = x.new_empty((*orig_shape[:-1], self.out_features))
            return out

        if sc_trace._ENABLED:
            sc_trace.set_context(self.operator,
                                 self.block_idx if self.block_idx is not None
                                 else get_current_block_idx(),
                                 None)

        base_sl = _resolve_stoc_len(self.stoc_len, self.sc_prec)

        gmap = _GROUP_STOCLEN[0]
        if gmap is not None:
            # Knock-down probe: uniform at an explicit stoc_len, MP bypassed, so
            # the measured ΔLoss is attributable to THIS group's precision alone.
            sl = int(gmap.get((self.operator, self.block_idx), base_sl))
            y = (torch.zeros(x2.shape[0], self.out_features,
                             dtype=torch.float32, device=x2.device)
                 if sl <= 0 else self._mm(x2, w, sl))
            if self.bias is not None:
                y = y + self.bias.float()
            return y.to(orig_dtype).reshape(*orig_shape[:-1], self.out_features)

        if self._range_groups is not None:
            y = self._forward_range(x2, w)
        elif self.adaptive_mp_cfg is not None or self.mp_cfg is not None:
            y = self._forward_row_mp(x2, w, base_sl)
        elif self.protect_spec is not None:
            y = self._forward_protect_uniform(x2, w, base_sl)
        else:
            y = self._mm(x2, w, base_sl)

        if self.bias is not None:
            y = y + self.bias.float()
        return y.to(orig_dtype).reshape(*orig_shape[:-1], self.out_features)

    # -- protected channels WITHOUT mixed precision -----------------------
    def _forward_protect_uniform(self, x2, w, base_sl):
        """Uniform SC everywhere except a few input channels held at a high
        stream length.

        This exists to answer whether protected channels are an MP feature at
        all. The qk `per_row` result showed that one of this study's two big
        wins was a plain quantization-scope fix that helps uniform SC just as
        much as MP; protection attacks the same failure (an outlier setting the
        shared per-row scale) on the linear side, so it may equally be a
        general SC fix wearing an MP costume. Without this path the question is
        unanswerable, because protection only ever ran inside the MP dispatch.

        Iso-compute is the caller's job: protecting a fraction p of channels at
        ``protect_sl`` while the rest run at ``base_sl`` costs
        ``p*protect_sl + (1-p)*base_sl``, so the comparison cell must be placed
        at its trace-measured average, not at ``base_sl``.
        """
        prot, prot_sl = self.protect_spec
        pi = _channel_index_tensor(prot, x2.shape[1], x2.device)
        if pi.numel() == 0:
            return self._mm(x2, w, base_sl)
        ri = _complement_channel_indices(x2.shape[1], pi, x2.device)
        y = self._mm(x2.index_select(1, pi).contiguous(),
                     w.index_select(1, pi).contiguous(), int(prot_sl))
        if ri.numel() > 0:
            y = y + self._mm(x2.index_select(1, ri).contiguous(),
                             w.index_select(1, ri).contiguous(), base_sl)
        return y

    # -- per-input-row mixed precision ------------------------------------
    def _forward_row_mp(self, x2, w, base_sl):
        cfg = self.adaptive_mp_cfg or self.mp_cfg
        adaptive = self.adaptive_mp_cfg

        # Protected input channels run at a fixed high stream length for EVERY
        # row, split off along the contraction dim. Exact, because a matmul is
        # additive over that dim: y = x[:, P] @ W[:, P]^T + x[:, R] @ W[:, R]^T,
        # so the two partial products just sum. Row dispatch then sees only the
        # residual channels — which is the point: one outlier channel otherwise
        # sets the shared per-row quantization scale and swamps the rest.
        prot_idx = _EMPTY_LONG.to(x2.device)
        rest_idx = None
        if adaptive is not None:
            prot = adaptive.get_protected_channels(
                operator=self.operator, block_idx=self.block_idx)
            prot_idx = _channel_index_tensor(prot, x2.shape[1], x2.device)
            if prot_idx.numel() > 0:
                rest_idx = _complement_channel_indices(
                    x2.shape[1], prot_idx, x2.device)

        if rest_idx is not None and rest_idx.numel() == 0:
            metric = torch.zeros(x2.shape[0], dtype=torch.float32,
                                 device=x2.device)
        else:
            src = x2 if rest_idx is None else x2.index_select(1, rest_idx)
            if adaptive is not None:
                # act_global_v2: the calibration table may have picked a per-op
                # metric other than amax (and possibly a negative sign, i.e. an
                # inverted ranking). Absent from the table => ("amax", +1).
                name, sign = adaptive.get_dispatch_metric(self.operator)
                metric = compute_row_metric(src, name) * sign
            else:
                metric = compute_row_metric(src, self.metric) * self.metric_sign

        if adaptive is not None:
            assignment = adaptive_classify_rows(
                metric, adaptive,
                operator=self.operator, block_idx=self.block_idx,
                total_blocks=_TOTAL_BLOCKS,
                timestep=_VIT_TIMESTEP[0], total_timesteps=_VIT_TIMESTEP[1])
            # Per-bucket ladder (+ the escape rung when the gate is on), NOT
            # the global list — with per-group ladders the two differ and the
            # global one would mis-price the budget baseline.
            levels = adaptive.classify_level_values(
                operator=self.operator, block_idx=self.block_idx,
                total_blocks=_TOTAL_BLOCKS,
                timestep=_VIT_TIMESTEP[0], total_timesteps=_VIT_TIMESTEP[1])
        else:
            assignment = classify_rows_by_metric(
                metric, cfg.stoc_len_levels, cfg.level_fractions)
            levels = cfg.stoc_len_levels

        max_sl = max(int(s) for s in levels)
        width = max(x2.shape[1], 1)
        prot_scale = float(prot_idx.numel()) / float(width)
        rest_scale = 1.0 - prot_scale
        macs_per_row = x2.shape[1] * self.out_features
        baseline = x2.shape[0] * macs_per_row * max_sl
        actual = 0.0

        y = torch.zeros(x2.shape[0], self.out_features,
                        dtype=torch.float32, device=x2.device)
        if prot_idx.numel() > 0:
            prot_sl = int(getattr(adaptive, "protected_channel_stoc_len", None)
                          or max_sl)
            y += self._mm(x2.index_select(1, prot_idx).contiguous(),
                          w.index_select(1, prot_idx).contiguous(), prot_sl)
            actual += x2.shape[0] * macs_per_row * prot_scale * prot_sl
            x_disp = x2.index_select(1, rest_idx).contiguous()
            w_disp = w.index_select(1, rest_idx).contiguous()
        else:
            x_disp, w_disp = x2, w

        if rest_scale > 0.0:
            for sl, idx in assignment.level_row_indices.items():
                if idx.numel() == 0:
                    continue
                sl = int(sl)
                if sl <= 0:        # level 0 == pruned row, issues no matmul
                    continue
                actual += idx.numel() * macs_per_row * rest_scale * sl
                out_sub = self._mm(x_disp.index_select(0, idx).contiguous(),
                                   w_disp, sl)
                if prot_idx.numel() > 0:
                    y[idx] += out_sub
                else:
                    y[idx] = out_sub
        AutoMPBudgetLogger.record(
            self.block_idx if self.block_idx is not None else get_current_block_idx(),
            self.operator or "linear", baseline, actual)
        return y

    # -- per-weight-group (range) mixed precision -------------------------
    def _forward_range(self, x2, w):
        """Output rows are grouped in contiguous blocks; each group gets its own
        stream length, so one kernel call per *distinct* level (groups sharing a
        level are gathered into a single call)."""
        g = self._range_group_size
        y = torch.zeros(x2.shape[0], self.out_features,
                        dtype=torch.float32, device=x2.device)
        by_level: dict[int, list[int]] = {}
        for gi, sl in enumerate(self._range_groups):
            by_level.setdefault(int(sl), []).extend(
                range(gi * g, min((gi + 1) * g, self.out_features)))
        for sl, rows in by_level.items():
            if sl <= 0 or not rows:
                continue
            idx = torch.as_tensor(rows, dtype=torch.long, device=x2.device)
            y[:, idx] = self._mm(x2, w.index_select(0, idx).contiguous(), sl)
        return y


# ---------------------------------------------------------------------------
# SC attention
# ---------------------------------------------------------------------------

AV_METRIC_KIND = "entropy"          # "amax" | "entropy" | "ipr"


def set_av_metric_kind(kind: str):
    """Row-peakiness metric used to allocate precision across attn@V rows.

    ``entropy`` / ``ipr`` are already oriented so HIGHER ⇒ flatter row ⇒ needs
    more precision. ``amax`` is the legacy metric and runs the other way.
    """
    global AV_METRIC_KIND
    if kind not in ("amax", "entropy", "ipr"):
        raise ValueError(f"unknown AV metric: {kind}")
    AV_METRIC_KIND = kind


def _av_row_score(attn_row: torch.Tensor) -> torch.Tensor:
    if AV_METRIC_KIND == "amax":
        return attn_row.amax(-1)
    if AV_METRIC_KIND == "entropy":
        safe = attn_row.clamp(min=1e-9)
        return -(attn_row * safe.log()).sum(-1)
    return 1.0 / (attn_row.pow(2).sum(-1) + 1e-9)          # ipr


def _classify_heads(q: torch.Tensor, cfg) -> dict[int, list[int]]:
    """Bucket attention heads by stoc_len. metric = per-head |q|.amax.

    Legacy ``qk_granularity="per_head"`` path. The kernel's per_head mode takes
    ONE (max, min) over each head's whole (N, D) slice, so a head is the finest
    unit that can be re-quantized independently — hence 16 units per block on
    ViT-L/14, which is a very coarse population to fit thresholds on. Prefer
    ``per_row`` (see ``_sc_qk``).
    """
    metric = q.float().abs().amax(dim=(0, 2, 3))            # (H,)
    if isinstance(cfg, AdaptiveMPConfig):
        assignment = adaptive_classify_rows(
            metric, cfg, operator="qk", block_idx=get_current_block_idx(),
            total_blocks=_TOTAL_BLOCKS,
            timestep=_VIT_TIMESTEP[0], total_timesteps=_VIT_TIMESTEP[1])
    else:
        assignment = classify_rows_by_metric(
            metric, cfg.stoc_len_levels, cfg.level_fractions)
    buckets: dict[int, list[int]] = {}
    for sl, heads in assignment.level_row_indices.items():
        if heads.numel():
            buckets.setdefault(int(sl), []).extend(h.item() for h in heads)
    return buckets


def _padded_level_gather(level_of_row: torch.Tensor, sl: int, BH: int, N: int):
    """Gather index + validity mask selecting this level's rows in every slice.

    ``level_of_row`` is a flat (BH*N,) tensor of assigned stream lengths. For
    stream length ``sl`` it returns ``(gather_idx, valid)`` both shaped
    ``(BH, M)`` where ``M`` is the largest per-slice count, so ONE batched
    ``sc_matmul`` covers every (b, h) slice instead of a BH-long Python loop
    (BH is 512 at batch 32 on ViT-L/14 — the loop dominated MP attention).

    Padding repeats an already-selected row rather than using row 0 or a zero
    row: under ``per_row`` granularity each row carries its own (max, min), and
    an all-zero row has a degenerate range. A repeated row is always in-range,
    and its output is discarded by ``valid``.
    """
    mask = (level_of_row == sl).reshape(BH, N)
    counts = mask.sum(dim=1)
    M = int(counts.max().item())
    if M == 0:
        return None, None, 0
    # rank of each selected row within its slice; unselected -> large sentinel
    order = torch.argsort(mask.to(torch.int8), dim=1, descending=True,
                          stable=True)                       # (BH, N)
    gather_idx = order[:, :M].contiguous()                   # (BH, M)
    ar = torch.arange(M, device=level_of_row.device).unsqueeze(0)
    valid = ar < counts.unsqueeze(1)                         # (BH, M)
    # Pad slots point at the slice's first selected row (or row 0 when the
    # slice selected nothing); both are in-range and both get masked out.
    first = gather_idx[:, :1].expand(-1, M)
    gather_idx = torch.where(valid, gather_idx, first)
    return gather_idx, valid, M


def _batched_level_matmul(a3, b3, level_of_row, levels, *, sc_prec, out_dim,
                          granularity, group_a=1, group_b=1):
    """Per-row MP over a 3D batch: one batched kernel call per stream length.

    ``a3`` is (BH, N, K); ``b3`` is the per-slice right operand in the kernel's
    (BH, out_dim, K) row-major layout. Returns ``(out, actual_row_cycles)``
    where ``out`` is (BH, N, out_dim) and the second value is
    ``Σ rows·stoc_len`` for the budget logger.
    """
    BH, N = a3.shape[0], a3.shape[1]
    K = a3.shape[2]
    out = torch.zeros(BH, N, out_dim, dtype=torch.float32, device=a3.device)
    row_cycles = 0.0
    slice_base = torch.arange(BH, device=a3.device).unsqueeze(1) * N
    for sl in levels:
        sl = int(sl)
        if sl <= 0:                 # pruned rows issue no matmul
            continue
        gather_idx, valid, M = _padded_level_gather(level_of_row, sl, BH, N)
        if gather_idx is None:
            continue
        a_sel = a3.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, K))
        out_sel = sc_matmul(
            a_sel.contiguous(), b3,
            granularity=granularity, mode="bipolar", sc_prec=sc_prec,
            group_a=group_a, group_b=group_b, stoc_len=sl,
            halve_bipolar_stoc_len=_DEFAULTS["halve_bipolar_stoc_len"])
        dst = (slice_base + gather_idx).reshape(-1)
        vflat = valid.reshape(-1)
        out.reshape(BH * N, out_dim)[dst[vflat]] = out_sel.reshape(
            BH * M, out_dim)[vflat]
        row_cycles += float(int(valid.sum().item()) * sl)
    return out, row_cycles


def _classify_flat_rows(metric: torch.Tensor, cfg, operator: str):
    """Classify a globally-pooled flat row metric; returns (assignment, levels).

    Pooling is GLOBAL over all B*H*N rows, never per (b, h): per-slice min-max
    erases cross-head scale, and the calibrator records one normalized metric
    pool per call, so a per-slice dispatch would put the thresholds on a
    different scale than they were fitted on.
    """
    if isinstance(cfg, AdaptiveMPConfig):
        assignment = adaptive_classify_rows(
            metric, cfg, operator=operator, block_idx=get_current_block_idx(),
            total_blocks=_TOTAL_BLOCKS,
            timestep=_VIT_TIMESTEP[0], total_timesteps=_VIT_TIMESTEP[1])
        levels = cfg.classify_level_values(
            operator=operator, block_idx=get_current_block_idx(),
            total_blocks=_TOTAL_BLOCKS,
            timestep=_VIT_TIMESTEP[0], total_timesteps=_VIT_TIMESTEP[1])
    else:
        assignment = classify_rows_by_metric(
            metric, cfg.stoc_len_levels, cfg.level_fractions)
        levels = cfg.stoc_len_levels
    return assignment, levels


def _level_of_row(assignment, n_rows: int, device) -> torch.Tensor:
    lv = torch.zeros(n_rows, dtype=torch.long, device=device)
    for sl, rows in assignment.level_row_indices.items():
        if rows.numel():
            lv[rows] = int(sl)
    return lv


# Quantization scope for Q@K^T. "per_head" is the legacy vit_sc setting (one
# (max, min) per head); "per_row" gives each QUERY ROW its own range, which is
# what makes per-query-row MP dispatch exact — a row subset re-quantized under
# per_head would see a different range than the full head did. scmp_llm runs
# attention at per_row for the same reason.
QK_GRANULARITY = "per_head"


def set_qk_granularity(kind: str):
    global QK_GRANULARITY
    if kind not in ("per_head", "per_row"):
        raise ValueError(f"unknown qk granularity: {kind!r}")
    QK_GRANULARITY = kind


def _sc_qk(q, k, sc_prec, stoc_len, mp_cfg):
    """Q @ K^T -> (B, H, N, N), unscaled.

    Granularity follows ``QK_GRANULARITY``. Under ``per_row`` and an MP config
    the dispatch unit is one QUERY ROW pooled globally over B*H*N — the same
    unit and pool ``av`` uses, and the one the calibrator must match.
    """
    B, H, N, D = q.shape
    qf = q.float().reshape(B * H, N, D).contiguous()
    kf = k.float().reshape(B * H, N, D).contiguous()

    if sc_trace._ENABLED:
        sc_trace.set_context("qk", get_current_block_idx(), None)

    # Knock-down probe: force this (op, block) to a uniform stoc_len with MP
    # bypassed, so measured dLoss is attributable to its precision alone.
    # Without this the probe silently cannot reach qk/av and every attention
    # group scores dL = 0 — which reads as "insensitive", not as "not measured".
    gmap = _GROUP_STOCLEN[0]
    if gmap is not None:
        sl = int(gmap.get(("qk", get_current_block_idx()),
                          _resolve_stoc_len(stoc_len, sc_prec)))
        if sl <= 0:
            return torch.zeros(B, H, N, N, dtype=torch.float32, device=q.device)
        return sc_matmul(qf, kf, granularity=QK_GRANULARITY, mode="bipolar",
                         sc_prec=sc_prec, group_a=1, group_b=1, stoc_len=sl,
                         halve_bipolar_stoc_len=_DEFAULTS["halve_bipolar_stoc_len"]
                         ).reshape(B, H, N, N)

    if mp_cfg is None:
        out = sc_matmul(qf, kf, granularity=QK_GRANULARITY, mode="bipolar",
                        sc_prec=sc_prec, group_a=1, group_b=1,
                        stoc_len=_resolve_stoc_len(stoc_len, sc_prec),
                        halve_bipolar_stoc_len=_DEFAULTS["halve_bipolar_stoc_len"])
        return out.reshape(B, H, N, N)

    if QK_GRANULARITY == "per_row":
        # Per-QUERY-ROW MP. metric = ||q_row||_inf over D, pooled globally.
        metric = qf.abs().amax(dim=-1).reshape(B * H * N)
        if isinstance(mp_cfg, AdaptiveMPConfig):
            name, sign = mp_cfg.get_dispatch_metric("qk")
            metric = compute_row_metric(qf.reshape(B * H * N, D), name) * sign
        assignment, levels = _classify_flat_rows(metric, mp_cfg, "qk")
        lv = _level_of_row(assignment, B * H * N, q.device)
        out, row_cycles = _batched_level_matmul(
            qf, kf, lv, levels, sc_prec=sc_prec, out_dim=N,
            granularity="per_row")
        max_sl = max(int(s) for s in levels)
        AutoMPBudgetLogger.record(
            get_current_block_idx(), "qk",
            float(B * H * N) * N * D * max_sl, row_cycles * N * D)
        return out.reshape(B, H, N, N)

    # Legacy per-head mixed precision: one kernel call per stoc_len bucket.
    buckets = _classify_heads(q, mp_cfg)
    max_sl = max(int(s) for s in mp_cfg.stoc_len_levels)
    baseline = B * H * N * N * D * max_sl
    actual = 0.0
    out = torch.zeros(B, H, N, N, dtype=torch.float32, device=q.device)
    for sl, heads in buckets.items():
        if sl <= 0 or not heads:
            continue
        idx = torch.as_tensor(heads, dtype=torch.long, device=q.device)
        actual += B * len(heads) * N * N * D * sl
        sub = sc_matmul(
            q.float().index_select(1, idx).reshape(-1, N, D).contiguous(),
            k.float().index_select(1, idx).reshape(-1, N, D).contiguous(),
            granularity="per_head", mode="bipolar", sc_prec=sc_prec,
            stoc_len=sl,
            halve_bipolar_stoc_len=_DEFAULTS["halve_bipolar_stoc_len"],
        ).reshape(B, len(heads), N, N)
        out.index_copy_(1, idx, sub)
    AutoMPBudgetLogger.record(get_current_block_idx(), "qk", baseline, actual)
    return out


def _sc_av(attn, v, sc_prec, stoc_len, mp_cfg):
    """attn @ V -> (B, H, N, D). per_row granularity, batched over B*H."""
    B, H, N, D = v.shape
    a3 = attn.float().reshape(B * H, N, N).contiguous()
    v3 = v.float().transpose(-1, -2).reshape(B * H, D, N).contiguous()

    if sc_trace._ENABLED:
        sc_trace.set_context("av", get_current_block_idx(), None)

    gmap = _GROUP_STOCLEN[0]          # knock-down probe; see _sc_qk
    if gmap is not None:
        sl = int(gmap.get(("av", get_current_block_idx()),
                          _resolve_stoc_len(stoc_len, sc_prec)))
        if sl <= 0:
            return torch.zeros(B, H, N, D, dtype=torch.float32, device=v.device)
        return sc_matmul(a3, v3, granularity="per_row", mode="bipolar",
                         sc_prec=sc_prec, group_a=1, group_b=1, stoc_len=sl,
                         halve_bipolar_stoc_len=_DEFAULTS["halve_bipolar_stoc_len"]
                         ).reshape(B, H, N, D)

    if mp_cfg is None:
        # One batched call for every (B, H) — the legacy path looped B*H times.
        out = sc_matmul(a3, v3, granularity="per_row", mode="bipolar",
                        sc_prec=sc_prec, group_a=1, group_b=1,
                        stoc_len=_resolve_stoc_len(stoc_len, sc_prec),
                        halve_bipolar_stoc_len=_DEFAULTS["halve_bipolar_stoc_len"])
        return out.reshape(B, H, N, D)

    # Per-attn-row MP.
    #
    # The metric is pooled and normalized GLOBALLY over all B*H*N rows, not
    # per (b, h). Two reasons, and they are the same reason scmp_llm does it:
    # per-slice min-max erases cross-head scale (a flat head and a peaked head
    # both get their own row spanning [0,1]), and — decisive here — the MP
    # calibrator records one normalized metric pool per call, so calibrating
    # globally and dispatching per-slice would put the thresholds on a
    # different scale than they were fitted on.
    #
    # Dispatch is ONE batched kernel call per stream length (padded row gather
    # across slices), not the old BH-long Python loop — BH is 512 at batch 32
    # on ViT-L/14 and that loop, not the SC work, dominated MP attention.
    # A calibrated table ranks av rows by a compute_row_metric candidate (amax
    # by default, or whatever rho-selection picked). The legacy peakiness score
    # is kept ONLY for a fixed MPConfig: it is not in the kernel's metric set,
    # so a table can never name it, and calibrating on it while the table
    # declares "amax" would cut the bands on a different ranking than the one
    # dispatched.
    rows2 = a3.reshape(B * H * N, N)
    if isinstance(mp_cfg, AdaptiveMPConfig):
        name, sign = mp_cfg.get_dispatch_metric("av")
        metric = compute_row_metric(rows2, name) * sign
    else:
        metric = _av_row_score(rows2).reshape(B * H * N)
    assignment, levels = _classify_flat_rows(metric, mp_cfg, "av")
    lv = _level_of_row(assignment, B * H * N, a3.device)
    out, row_cycles = _batched_level_matmul(
        a3, v3, lv, levels, sc_prec=sc_prec, out_dim=D, granularity="per_row")
    max_sl = max(int(s) for s in levels)
    AutoMPBudgetLogger.record(
        get_current_block_idx(), "av",
        float(B * H * N) * D * N * max_sl, row_cycles * D * N)
    return out.reshape(B, H, N, D)


def make_sc_attention_forward(sc_prec: int, sc_qk: bool, sc_av: bool,
                              *, stoc_len: int | None = None,
                              av_stoc_len: int | None = "inherit",
                              qk_mp_cfg=None, av_mp_cfg=None,
                              block_idx: int | None = None,
                              qk_backend: str = "sc", av_backend: str = "sc",
                              int_cfg=None):
    """Build a ``forward(self, x, attn_bias=None)`` for a DINOv2-style Attention.

    Only Q@K^T and attn@V are quantized; softmax and the qkv/proj linears are
    left to the caller (patch them with ``SCLinear`` separately).

    ``qk_backend`` / ``av_backend`` select per-op between ``"sc"``, ``"int"``
    (RTN fake-quant at ``int_cfg``) and ``"fp"``. They exist because qk and av
    share ONE attention forward: a hybrid schedule that sends qk to INT while
    av stays on SC cannot be expressed by choosing an SC forward or an INT
    forward for the block. ``sc_qk``/``sc_av`` still gate whether the op is
    quantized at all, so the default (both "sc") is the original behaviour.
    """
    if qk_backend not in ("sc", "int", "fp") or av_backend not in ("sc", "int", "fp"):
        raise ValueError(f"attention backends must be sc|int|fp, got "
                         f"qk={qk_backend!r} av={av_backend!r}")
    if (qk_backend == "int" or av_backend == "int") and int_cfg is None:
        raise ValueError("attention backend 'int' needs an int_cfg")

    def forward(self, x, attn_bias=None):
        from .int_vit import fake_quant, _fake_quant_perhead
        B, N, C = x.shape
        H = self.num_heads
        D = C // H
        if block_idx is not None:
            set_current_block_idx(block_idx)
        qkv = self.qkv(x).reshape(B, N, 3, H, D)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if sc_qk and qk_backend == "sc":
            attn = _sc_qk(q, k, sc_prec, stoc_len, qk_mp_cfg).to(x.dtype) * self.scale
        elif sc_qk and qk_backend == "int":
            def _qa(t):
                t3 = t.float().reshape(B * H, N, D)
                return (_fake_quant_perhead(t3, int_cfg.bits, int_cfg.sym)
                        if int_cfg.attn_scope == "per_head"
                        else fake_quant(t3, int_cfg.bits, int_cfg.sym, dim=-1))
            attn = (_qa(q) @ _qa(k).transpose(-2, -1)).reshape(B, H, N, N)
            attn = attn.to(x.dtype) * self.scale
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if sc_av and av_backend == "sc":
            _av_sl = stoc_len if av_stoc_len == "inherit" else av_stoc_len
            out = _sc_av(attn, v, sc_prec, _av_sl, av_mp_cfg).to(x.dtype)
        elif sc_av and av_backend == "int":
            a3 = fake_quant(attn.float().reshape(B * H, N, N),
                            int_cfg.bits, int_cfg.sym, dim=-1)
            v3 = fake_quant(v.float().transpose(-1, -2).reshape(B * H, D, N),
                            int_cfg.bits, int_cfg.sym, dim=-1)
            out = (a3 @ v3.transpose(-2, -1)).reshape(B, H, N, D).to(x.dtype)
        else:
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))

    return forward


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------

SC_OP_NAMES = ("mlp_fc1", "mlp_fc2", "qk", "av", "proj")
LINEAR_OP_NAMES = ("mlp_fc1", "mlp_fc2", "qkv_proj", "out_proj")


def parse_sc_ops(spec) -> set[str]:
    """Normalize an ``sc_ops`` spec to a set of canonical names.

    Accepts a set/list of names, a 5-element 0/1 vector in ``SC_OP_NAMES``
    order, or a comma-separated string.
    """
    if spec is None:
        return set()
    if isinstance(spec, str):
        return parse_sc_ops([s.strip() for s in spec.split(",") if s.strip()])
    items = list(spec)
    if len(items) == len(SC_OP_NAMES) and all(isinstance(x, (int, bool)) for x in items):
        return {n for n, v in zip(SC_OP_NAMES, items) if v}
    unknown = [x for x in items if x not in SC_OP_NAMES]
    if unknown:
        raise ValueError(f"unknown sc op(s): {unknown} (valid: {SC_OP_NAMES})")
    return set(items)


def normalize_sc_ops_per_block(spec, n_blocks: int) -> dict[str, tuple]:
    """Expand a ``{op: [0/1] * n_blocks}`` spec; missing ops default to off."""
    if spec is None:
        return {k: tuple([0] * n_blocks) for k in SC_OP_NAMES}
    if not isinstance(spec, dict):
        raise TypeError("sc_ops_per_block must be a dict[op_name, list]")
    extra = set(spec) - set(SC_OP_NAMES)
    if extra:
        raise ValueError(f"unknown op names in sc_ops_per_block: {sorted(extra)}")
    out = {}
    for k in SC_OP_NAMES:
        v = spec.get(k, 0)
        if isinstance(v, (int, bool)):
            out[k] = tuple([int(bool(v))] * n_blocks)
            continue
        v = list(v)
        if len(v) == 1:
            out[k] = tuple([int(bool(v[0]))] * n_blocks)
        elif len(v) == n_blocks:
            out[k] = tuple(int(bool(x)) for x in v)
        else:
            raise ValueError(
                f"sc_ops_per_block[{k!r}] length {len(v)} != n_blocks {n_blocks}")
    return out


def _discover_blocks(model):
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        return list(model.backbone.blocks)
    if hasattr(model, "blocks"):
        return list(model.blocks)
    raise RuntimeError("could not locate transformer blocks on model")


def _find_attn(block):
    for m in block.modules():
        if hasattr(m, "qkv") and hasattr(m, "proj") and hasattr(m, "num_heads"):
            return m
    return None


def _normalize_linear_mp_spec(spec) -> dict[str, tuple]:
    """``{op|'proj': cfg-or-dict}`` -> ``{op: (fixed, adaptive, range, gsize)}``."""
    out: dict[str, tuple] = {}
    if not spec:
        return out
    for k, v in spec.items():
        if k == "proj":
            targets = ("qkv_proj", "out_proj")
        elif k in LINEAR_OP_NAMES:
            targets = (k,)
        else:
            raise ValueError(f"unknown linear_mp_spec key {k!r}; expected one of "
                             f"{LINEAR_OP_NAMES + ('proj',)}")
        if isinstance(v, MPConfig):
            entry = (v, None, None, 0)
        elif isinstance(v, AdaptiveMPConfig):
            entry = (None, v, None, 0)
        elif isinstance(v, RangeMPConfig):
            entry = (None, None, v, 0)
        elif isinstance(v, dict):
            entry = (v.get("fixed"), v.get("adaptive"), v.get("range"),
                     int(v.get("range_group_size", 0)))
        else:
            raise TypeError(f"linear_mp_spec[{k!r}] has unsupported type "
                            f"{type(v).__name__}")
        for op in targets:
            out[op] = entry
    return out


def _mp_kwargs(linear_mp, op_name, block_idx):
    base = dict(operator=op_name, block_idx=block_idx)
    if not linear_mp or op_name not in linear_mp:
        return base
    fixed, adaptive, rng, gsize = linear_mp[op_name]
    return dict(base, mp_cfg=fixed, adaptive_mp_cfg=adaptive,
                range_mp_cfg=rng, range_mp_group_size=gsize)


def patch_model(model, sc_prec: int = 8,
                sc_qk: bool = True, sc_av: bool = False, sc_mlp: bool = False,
                sc_qkv_proj: bool = False, sc_out_proj: bool = False,
                *,
                stoc_len: int | None = None,
                mlp_skip_first_k: int = 0, mlp_skip_last_k: int = 0,
                sc_mlp_mode: str = "bipolar", sc_proj_mode: str = "bipolar",
                mlp_fc_mask: str = "both",
                mlp_sc_spec=None,
                sc_ops=None,
                sc_ops_per_block=None,
                linear_mp_spec=None,
                attn_mp_spec=None,
                hybrid_schedule=None,
                int_cfg=None,
                protect_channels=None,
                chunk_d: int | None = None) -> dict:
    """Swap SC ops into a DINOv2-style ViT in place. Returns patch counts.

    Op selection priority: ``sc_ops_per_block`` > ``sc_ops`` > per-op booleans.
    ``mlp_sc_spec`` (a list of ``(block_idx, "fc1"|"fc2")``) overrides the MLP
    selection when given.

    ``hybrid_schedule`` is ``{op_name: [backend per block]}`` with backends
    ``"sc"`` / ``"int"`` / ``"fp"``. It does not choose WHICH ops are quantized
    — that is still ``sc_ops`` — only which datapath each selected (op, block)
    runs on, so a hybrid cell and its uniform comparator cover exactly the same
    ops. Groups routed to ``int`` leave the SC budget entirely; they are priced
    on the energy axis, not against the SC average.
    """
    hyb = dict(hybrid_schedule or {})

    def _backend(op, bi):
        v = hyb.get(op)
        if v is None:
            return "sc"
        b = v[bi] if not isinstance(v, str) else v
        if b not in ("sc", "int", "fp"):
            raise ValueError(f"hybrid_schedule[{op!r}][{bi}] = {b!r}; "
                             f"expected sc|int|fp")
        return b
    if sc_ops is not None:
        ops = parse_sc_ops(sc_ops)
        sc_qk = "qk" in ops
        sc_av = "av" in ops
        sc_qkv_proj = sc_out_proj = "proj" in ops
        if mlp_sc_spec is None:
            want1, want2 = "mlp_fc1" in ops, "mlp_fc2" in ops
            sc_mlp = want1 or want2
            mlp_fc_mask = ("both" if want1 and want2
                           else "fc1" if want1 else "fc2" if want2 else "both")

    linear_mp = _normalize_linear_mp_spec(linear_mp_spec)
    qk_mp_cfg = (attn_mp_spec or {}).get("qk")
    av_mp_cfg = (attn_mp_spec or {}).get("av")
    for name, cfg in (("qk", qk_mp_cfg), ("av", av_mp_cfg)):
        if cfg is not None and not isinstance(cfg, (MPConfig, AdaptiveMPConfig)):
            raise TypeError(f"attn_mp_spec[{name!r}] must be MPConfig or "
                            f"AdaptiveMPConfig, got {type(cfg).__name__}")

    stats = {"attn": 0, "mlp_fc1": 0, "mlp_fc2": 0, "qkv_proj": 0, "out_proj": 0,
             "hybrid_int": 0, "hybrid_fp": 0}
    blocks = _discover_blocks(model)
    global _TOTAL_BLOCKS
    _TOTAL_BLOCKS = len(blocks)          # layer-bucket denominator for MP tables

    # stoc_len may be a scalar (uniform) or a per-op dict. The per-op form is
    # what makes "spend cycles where the MACs aren't" expressible: on ViT-L/14
    # attention is ~4% of SC MACs, so pinning qk/av to the ceiling costs almost
    # nothing against a budget dominated by MLP (64%) and proj (32%).
    def _sl(op):
        if isinstance(stoc_len, dict):
            return stoc_len.get(op, stoc_len.get("default"))
        return stoc_len

    # {(op, block): [channel idx]} + the stream length they run at, for the
    # no-MP protected-channel control.
    prot_map, prot_sl = (protect_channels or (None, 0))

    def _protect_for(op, bi):
        if not prot_map:
            return None
        idx = prot_map.get((op, bi), prot_map.get(f"{op}:b{bi}"))
        return (idx, prot_sl) if idx else None

    def _lin(mod, op, bi, mode):
        backend = _backend(op, bi)
        if backend == "fp":
            stats["hybrid_fp"] += 1
            return mod
        if backend == "int":
            from .int_vit import IntLinear
            if int_cfg is None:
                raise ValueError(f"hybrid_schedule sends {op}:b{bi} to INT but "
                                 f"no int_cfg was given")
            stats["hybrid_int"] += 1
            return IntLinear(mod, int_cfg, op, bi)
        return SCLinear(mod, sc_prec, mode=mode, stoc_len=_sl(op),
                        chunk_d=chunk_d, protect_spec=_protect_for(op, bi),
                        **_mp_kwargs(linear_mp, op, bi))

    # ---- per-block per-op vectors (highest priority) ----
    if sc_ops_per_block is not None:
        pbm = normalize_sc_ops_per_block(sc_ops_per_block, len(blocks))
        for i, blk in enumerate(blocks):
            attn = _find_attn(blk)
            q_on, a_on, p_on = pbm["qk"][i], pbm["av"][i], pbm["proj"][i]
            if attn is not None and (q_on or a_on):
                qb, ab = _backend("qk", i), _backend("av", i)
                stats["hybrid_int"] += int(q_on and qb == "int") + int(a_on and ab == "int")
                fwd = make_sc_attention_forward(
                    sc_prec, bool(q_on), bool(a_on), stoc_len=stoc_len,
                    qk_mp_cfg=qk_mp_cfg if (q_on and qb == "sc") else None,
                    av_mp_cfg=av_mp_cfg if (a_on and ab == "sc") else None,
                    block_idx=i, qk_backend=qb, av_backend=ab, int_cfg=int_cfg)
                attn.forward = fwd.__get__(attn, type(attn))
                stats["attn"] += 1
            if attn is not None and p_on:
                if isinstance(attn.qkv, nn.Linear):
                    attn.qkv = _lin(attn.qkv, "qkv_proj", i, sc_proj_mode)
                    stats["qkv_proj"] += 1
                if isinstance(attn.proj, nn.Linear):
                    attn.proj = _lin(attn.proj, "out_proj", i, sc_proj_mode)
                    stats["out_proj"] += 1
            mlp = getattr(blk, "mlp", None)
            if mlp is not None:
                if pbm["mlp_fc1"][i] and isinstance(getattr(mlp, "fc1", None), nn.Linear):
                    mlp.fc1 = _lin(mlp.fc1, "mlp_fc1", i, sc_mlp_mode)
                    stats["mlp_fc1"] += 1
                if pbm["mlp_fc2"][i] and isinstance(getattr(mlp, "fc2", None), nn.Linear):
                    mlp.fc2 = _lin(mlp.fc2, "mlp_fc2", i, sc_mlp_mode)
                    stats["mlp_fc2"] += 1
        return stats

    # ---- attention ----
    if sc_qk or sc_av:
        for i, blk in enumerate(blocks):
            attn = _find_attn(blk)
            if attn is None:
                continue
            qb, ab = _backend("qk", i), _backend("av", i)
            stats["hybrid_int"] += int(sc_qk and qb == "int") + int(sc_av and ab == "int")
            fwd = make_sc_attention_forward(
                sc_prec, sc_qk, sc_av, stoc_len=stoc_len,
                qk_mp_cfg=qk_mp_cfg if (sc_qk and qb == "sc") else None,
                av_mp_cfg=av_mp_cfg if (sc_av and ab == "sc") else None,
                block_idx=i, qk_backend=qb, av_backend=ab, int_cfg=int_cfg)
            attn.forward = fwd.__get__(attn, type(attn))
            stats["attn"] += 1

    # ---- qkv / out projections ----
    if sc_qkv_proj or sc_out_proj:
        for i, blk in enumerate(blocks):
            attn = _find_attn(blk)
            if attn is None:
                continue
            if sc_qkv_proj and isinstance(attn.qkv, nn.Linear):
                attn.qkv = _lin(attn.qkv, "qkv_proj", i, sc_proj_mode)
                stats["qkv_proj"] += 1
            if sc_out_proj and isinstance(attn.proj, nn.Linear):
                attn.proj = _lin(attn.proj, "out_proj", i, sc_proj_mode)
                stats["out_proj"] += 1

    # ---- MLP ----
    if sc_mlp:
        if mlp_sc_spec is not None:
            selected: dict[int, set] = {}
            for bi, fc in mlp_sc_spec:
                if fc not in ("fc1", "fc2"):
                    raise ValueError(f"bad linear name in mlp_sc_spec: {fc!r}")
                selected.setdefault(int(bi), set()).add(fc)
            for bi, blk in enumerate(blocks):
                mlp = getattr(blk, "mlp", None)
                want = selected.get(bi, set())
                if mlp is None or not want:
                    continue
                if "fc1" in want and isinstance(getattr(mlp, "fc1", None), nn.Linear):
                    mlp.fc1 = _lin(mlp.fc1, "mlp_fc1", bi, sc_mlp_mode)
                    stats["mlp_fc1"] += 1
                if "fc2" in want and isinstance(getattr(mlp, "fc2", None), nn.Linear):
                    mlp.fc2 = _lin(mlp.fc2, "mlp_fc2", bi, sc_mlp_mode)
                    stats["mlp_fc2"] += 1
        else:
            skip = set(range(mlp_skip_first_k))
            if mlp_skip_last_k > 0:
                skip |= set(range(len(blocks) - mlp_skip_last_k, len(blocks)))
            for bi, blk in enumerate(blocks):
                mlp = getattr(blk, "mlp", None)
                if mlp is None or bi in skip:
                    continue
                if mlp_fc_mask in ("both", "fc1") and isinstance(getattr(mlp, "fc1", None), nn.Linear):
                    mlp.fc1 = _lin(mlp.fc1, "mlp_fc1", bi, sc_mlp_mode)
                    stats["mlp_fc1"] += 1
                if mlp_fc_mask in ("both", "fc2") and isinstance(getattr(mlp, "fc2", None), nn.Linear):
                    mlp.fc2 = _lin(mlp.fc2, "mlp_fc2", bi, sc_mlp_mode)
                    stats["mlp_fc2"] += 1

    return stats
