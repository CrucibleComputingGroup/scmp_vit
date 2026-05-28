"""SCLinear and helpers — shared between cls/ (DINOv2) and det/ (EVA-ViTDet).

Lives in ``sc_integration`` so neither cls nor det has to reach across into the
other. Backed by the Triton SC kernels in ``vit_sc/sc/`` and the noise-model
surrogate in ``sc_integration.noise_matmul``.

Public surface:
  * ``SCLinear``                   — drop-in nn.Linear replacement.
  * ``set_noise_model(bool)``      — toggle the calibrated Gaussian surrogate.
  * ``USE_NOISE_MODEL``            — read at call time via this module
                                     (``sc_linear.USE_NOISE_MODEL``); avoid
                                     ``from … import USE_NOISE_MODEL`` because
                                     the snapshot won't reflect later toggles.
  * ``_sc_linear`` / ``_sc_linear_mp`` / ``_mm_one_bucket`` /
    ``_get_config`` / ``_CFG_CACHE`` — internal helpers also used by
    ``HeadAlignedSCLinear`` and the cls attention forward.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Need vit_sc/sc/ on the path to find sc_triton + config_helpers.
_VIT_SC = Path(__file__).resolve().parents[1]   # vit_sc/
_SC_DIR = _VIT_SC / "sc"
if str(_SC_DIR) not in sys.path:
    sys.path.insert(0, str(_SC_DIR))

from scmp_kernels.sc import sc_matmul  # unified granularity dispatcher
from scmp_kernels.sc.config_helpers import make_sobol_simple_config  # noqa: E402

from sc_integration.noise_matmul import noisy_sc_matmul_grouped
from sc_integration.mp_linear import (
    MPConfig, RangeMPConfig, AdaptiveMPConfig, DispatchEntry,
    build_range_dispatch, classify_input_rows, classify_input_rows_adaptive,
    sc_prec_for_stoc_len, get_current_block_idx,
)
from scmp_kernels.mp import AutoMPBudgetLogger


# Module-level toggle: when True, SC matmuls use the calibrated Gaussian-noise
# surrogate (sc_integration/noise_matmul.py) instead of the real bitstream /
# enable-signal kernels. Orders of magnitude faster — for sensitivity sweeps.
USE_NOISE_MODEL = False


def set_noise_model(enabled: bool):
    global USE_NOISE_MODEL
    USE_NOISE_MODEL = bool(enabled)


# ---------------------------------------------------------------------------
# External timestep state (for ViT / any non-diffusion caller).
# ---------------------------------------------------------------------------
# Diffusion callers already have a native (t, T) via ``SCController``. ViT cls
# and EVA-det have no diffusion step; when they want AdaptiveMPConfig they set
# ``(t, T)`` externally and every SCLinear with an adaptive_mp_cfg reads from
# this module. Kept here (not on SCLinear instances) so a whole patched model
# can be driven by one setter.
_VIT_TIMESTEP: int = 0
_VIT_TOTAL_TIMESTEPS: int = 1


def set_vit_timestep(t: int, total: int | None = None):
    """Set the current external timestep for adaptive MP in SCLinear.

    ``t`` is the current timestep, ``total`` (if given) updates the total.
    ViT has no diffusion step so the caller picks both — e.g. ``(0, 1)`` to
    disable any timestep dependence, or sweep ``t ∈ [0, T-1]`` to emulate
    a schedule for sensitivity studies.
    """
    global _VIT_TIMESTEP, _VIT_TOTAL_TIMESTEPS
    _VIT_TIMESTEP = int(t)
    if total is not None:
        _VIT_TOTAL_TIMESTEPS = max(int(total), 1)


def get_vit_timestep() -> tuple[int, int]:
    return _VIT_TIMESTEP, _VIT_TOTAL_TIMESTEPS


_CFG_CACHE: dict = {}


def _get_config(D, sc_prec):
    key = (D, sc_prec)
    if key not in _CFG_CACHE:
        _CFG_CACHE[key] = make_sobol_simple_config(D, D, sc_prec)
    return _CFG_CACHE[key]


def _sym_range(t: torch.Tensor) -> tuple[float, float]:
    m = float(t.detach().abs().max().item())
    m = max(m, 1e-6)
    return m, -m


def _sc_linear(x: torch.Tensor, weight: torch.Tensor, bias, sc_prec: int,
               mode: str = "bipolar", cfg_override: dict = None,
               stoc_len: int | None = None, chunk_d: int = 0):
    """y = x @ weight^T + bias via SC.

    mode="bipolar": per-row symmetric quant (|max|), q_max = 2^(sc_prec-1)-1.
        Good for near-zero-mean tensors (post-LN activations, weights).
    mode="unipolar": per-row asymmetric quant (scale = range/q_max, zero-point),
        q_max = 2^sc_prec - 1 → ~2× grid resolution. Good for heavy-tailed /
        asymmetric tensors (post-GELU activations for fc2).

    ``cfg_override``: when not None, use this SC config instead of the cached
    default — lets the comp use an alternative Sobol scrambling per block.
    ``stoc_len``: explicit bit-stream length (arbitrary int). When None,
    kernels fall back to ``2**sc_prec``.
    ``chunk_d``: when > 0 and mode == "bipolar" (and no cfg_override), use the
    per-row MLP fast path ``sc_matmul(..., granularity="per_row", chunk_d=...)``
    with internal D-chunking. For wide MLP layers (det's fc1 D=1408, fc2 D=6144) this:
      * shrinks the cum_indicator table from O(D) → O(chunk_d) so it fits in
        L2 (avoids the ~800 MB table for fc2 at sc_prec=8);
      * fuses per-row quant kernels (~10× fewer launches than the generic
        grouped path);
      * is the same fast path Q-DiT's SCMlp uses.
    Ignored when ``cfg_override`` is set (the comp pool is sized for full D).
    """
    orig_shape = x.shape
    D_in = orig_shape[-1]
    x2 = x.reshape(-1, D_in).float()
    w = weight.float()  # (D_out, D_in)
    if USE_NOISE_MODEL:
        y = noisy_sc_matmul_grouped(x2, w, group_a=1, group_b=1,
                                    mode=mode, sc_prec=sc_prec,
                                    stoc_len=stoc_len)
    elif (chunk_d > 0 and mode == "bipolar"
          and D_in > chunk_d and cfg_override is None):
        cfg = _get_config(chunk_d, sc_prec)
        y = sc_matmul(
            x2, w, granularity="per_row",
            mode=mode, sc_prec=sc_prec, config=cfg,
            group_a=1, group_b=1, chunk_d=chunk_d,
            stoc_len=stoc_len,
        )
    else:
        cfg = cfg_override if cfg_override is not None else _get_config(D_in, sc_prec)
        y = sc_matmul(x2, w, granularity="per_row", group_a=1, group_b=1,
                      mode=mode, sc_prec=sc_prec, config=cfg,
                      stoc_len=stoc_len)
    if bias is not None:
        y = y + bias.float()
    y = y.to(x.dtype).reshape(*orig_shape[:-1], w.shape[0])
    return y


def _mm_one_bucket(x_sub: torch.Tensor, w_sub: torch.Tensor,
                   stoc_len: int, sc_prec: int, mode: str,
                   chunk_d: int = 0) -> torch.Tensor:
    """One SC matmul call at a chosen ``stoc_len``. Picks the noise-surrogate
    backend when ``USE_NOISE_MODEL`` is on, otherwise the Triton kernel.
    Both accept ``stoc_len`` as a kwarg. ``chunk_d`` enables the MLP fast
    path (same semantics as ``_sc_linear``).
    """
    if USE_NOISE_MODEL:
        return noisy_sc_matmul_grouped(
            x_sub, w_sub, group_a=1, group_b=1,
            mode=mode, sc_prec=sc_prec, stoc_len=stoc_len,
        )
    D_in = x_sub.shape[-1]
    if chunk_d > 0 and mode == "bipolar" and D_in > chunk_d:
        cfg = _get_config(chunk_d, sc_prec)
        return sc_matmul(
            x_sub, w_sub, granularity="per_row",
            mode=mode, sc_prec=sc_prec, config=cfg,
            group_a=1, group_b=1, chunk_d=chunk_d,
            stoc_len=stoc_len,
        )
    cfg = _get_config(D_in, sc_prec)
    return sc_matmul(
        x_sub, w_sub, granularity="per_row", group_a=1, group_b=1,
        mode=mode, sc_prec=sc_prec, config=cfg, stoc_len=stoc_len,
    )


def _sc_linear_mp(x: torch.Tensor, weight: torch.Tensor, bias,
                  sc_prec: int, mode: str,
                  range_entries: list[DispatchEntry] | None = None,
                  mp_cfg: MPConfig | None = None,
                  adaptive_mp_cfg: AdaptiveMPConfig | None = None,
                  operator: str | None = None,
                  chunk_d: int = 0) -> torch.Tensor:
    """MP-aware SC linear: ``y = x @ weight^T + bias``.

    * ``mp_cfg``         → per-input-row stoc_len buckets (fixed quantile).
    * ``adaptive_mp_cfg``→ per-input-row buckets with timestep-aware thresholds
      ``α·progress(t) + β``; the (t, T) pair is read from
      ``sc_linear.get_vit_timestep()``. When both ``mp_cfg`` and
      ``adaptive_mp_cfg`` are set, adaptive wins.
    * ``range_entries``  → per-weight-output-group stoc_len buckets (static).
    * Any row + range combo → effective stoc_len = ``min(row_sl, weight_sl)``.
    * Neither row config → uniform, equivalent to ``_sc_linear``.
    """
    orig_shape = x.shape
    D_in = orig_shape[-1]
    x_flat = x.reshape(-1, D_in).float()
    w = weight.float()
    M = x_flat.shape[0]
    D_out = w.shape[0]

    if range_entries is None and mp_cfg is None and adaptive_mp_cfg is None:
        y = _mm_one_bucket(x_flat, w, 2 ** sc_prec, sc_prec, mode,
                           chunk_d=chunk_d)
        if bias is not None:
            y = y + bias.float()
        return y.to(x.dtype).reshape(*orig_shape[:-1], D_out)

    # Per-row stoc_len from dynamic MP (or a uniform sentinel otherwise).
    if adaptive_mp_cfg is not None:
        t, T = get_vit_timestep()
        assignment = classify_input_rows_adaptive(
            x_flat, adaptive_mp_cfg, t, T, operator=operator,
        )
        row_stoc_lens = torch.empty(M, dtype=torch.long, device=x.device)
        for sl, rows in assignment.level_row_indices.items():
            if rows.numel():
                row_stoc_lens[rows] = int(sl)
    elif mp_cfg is not None:
        assignment = classify_input_rows(x_flat, mp_cfg)
        row_stoc_lens = torch.empty(M, dtype=torch.long, device=x.device)
        for sl, rows in assignment.level_row_indices.items():
            if rows.numel():
                row_stoc_lens[rows] = int(sl)
    else:
        row_stoc_lens = torch.full((M,), 2 ** sc_prec,
                                   dtype=torch.long, device=x.device)

    result = torch.zeros(M, D_out, dtype=torch.float32, device=x.device)

    if range_entries is None and (adaptive_mp_cfg is not None or mp_cfg is not None):
        levels = (adaptive_mp_cfg.stoc_len_levels if adaptive_mp_cfg is not None
                  else mp_cfg.stoc_len_levels)
        max_stoc_len = max(levels)
        compute_baseline = M * D_out * max_stoc_len
        compute_actual = 0.0
        for sl, rows in assignment.level_row_indices.items():
            if int(sl) <= 0 or rows.numel() == 0:
                continue
            compute_actual += int(sl) * rows.numel() * D_out
        if operator is not None:
            AutoMPBudgetLogger.record(
                get_current_block_idx(), operator,
                compute_baseline, compute_actual,
            )

    if range_entries is None:
        # Dynamic MP only: full-width weight, bucket by row stoc_len.
        for sl in torch.unique(row_stoc_lens).tolist():
            if sl <= 0:
                continue
            rows = torch.nonzero(row_stoc_lens == sl, as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                continue
            x_sub = x_flat.index_select(0, rows).contiguous()
            sp = sc_prec_for_stoc_len(int(sl))
            sub = _mm_one_bucket(x_sub, w, int(sl), sp, mode,
                                 chunk_d=chunk_d)
            result.index_copy_(0, rows, sub)
    else:
        # Range dispatch, optionally combined with dynamic MP.
        for entry in range_entries:
            if entry.stoc_len <= 0:
                continue
            weight_sl = torch.tensor(entry.stoc_len, dtype=torch.long,
                                     device=x.device)
            eff = torch.minimum(row_stoc_lens, weight_sl)
            for sl in torch.unique(eff).tolist():
                if sl <= 0:
                    continue
                rows = torch.nonzero(eff == sl, as_tuple=False).squeeze(-1)
                if rows.numel() == 0:
                    continue
                x_sub = x_flat.index_select(0, rows).contiguous()
                sp = sc_prec_for_stoc_len(int(sl))
                sub = _mm_one_bucket(x_sub, entry.weight, int(sl), sp, mode,
                                     chunk_d=chunk_d)
                result[rows.unsqueeze(1), entry.out_indices.unsqueeze(0)] = sub

    if bias is not None:
        result = result + bias.float()
    return result.to(x.dtype).reshape(*orig_shape[:-1], D_out)


class SCLinear(nn.Module):
    """Drop-in replacement for nn.Linear that runs matmul via SC.

    mode selects symmetric (bipolar) or asymmetric zero-point (unipolar) quant.

    Optional mixed-precision knobs (ported from scmp_llm):
      * ``mp_cfg`` — per-input-row MPConfig (fixed-quantile).
      * ``adaptive_mp_cfg`` — per-input-row AdaptiveMPConfig. Reads (t, T) at
        forward time from ``sc_linear.get_vit_timestep()`` — callers update it
        via ``set_vit_timestep``. When both ``mp_cfg`` and ``adaptive_mp_cfg``
        are set, adaptive wins.
      * ``range_mp_cfg`` — per-weight-row-group RangeMPConfig (static
        dispatch built once from the wrapped weight).
      * ``range_mp_group_size`` — rows per group (0 / >=out_features ⇒
        per-tensor).
      * ``operator`` — name passed to RangeMPConfig and AdaptiveMPConfig for
        per-op thresholds / (α, β).
    """

    def __init__(self, linear: nn.Linear, sc_prec: int, mode: str = "bipolar",
                 cfg_override: dict = None,
                 mp_cfg: MPConfig | None = None,
                 adaptive_mp_cfg: AdaptiveMPConfig | None = None,
                 range_mp_cfg: RangeMPConfig | None = None,
                 range_mp_group_size: int = 0,
                 operator: str | None = None,
                 stoc_len: int | None = None,
                 chunk_d: int = 0):
        super().__init__()
        self.sc_prec = sc_prec
        self.stoc_len = stoc_len
        self.mode = mode
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        # Stored as a plain dict (not a buffer): used to pick an alternative
        # Sobol scrambling for the comp's SC matmul, baked at calibration time.
        self.cfg_override = cfg_override
        self.mp_cfg = mp_cfg
        self.adaptive_mp_cfg = adaptive_mp_cfg
        self.operator = operator
        self.chunk_d = chunk_d
        if range_mp_cfg is not None:
            self.range_entries = build_range_dispatch(
                self.weight.detach(), range_mp_cfg,
                operator=operator, group_size=range_mp_group_size,
            )
        else:
            self.range_entries = None

    def forward(self, x):
        if (self.range_entries is None and self.mp_cfg is None
                and self.adaptive_mp_cfg is None):
            # Uniform path — keep cfg_override support (residual compensator).
            return _sc_linear(x, self.weight, self.bias, self.sc_prec,
                              mode=self.mode, cfg_override=self.cfg_override,
                              stoc_len=self.stoc_len, chunk_d=self.chunk_d)
        return _sc_linear_mp(
            x, self.weight, self.bias, self.sc_prec, self.mode,
            range_entries=self.range_entries, mp_cfg=self.mp_cfg,
            adaptive_mp_cfg=self.adaptive_mp_cfg, operator=self.operator,
            chunk_d=self.chunk_d,
        )
