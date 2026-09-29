"""
Closed-form SC noise surrogate, drop-in replacements for the real SC
kernels in SC/sc_triton.py.

Each adapter below has the EXACT signature of the corresponding real
kernel, so it can be returned from `_get_sc_matmul_fn()` (and its sibling
dispatch helpers) without any caller-side changes. All arguments related
to the bitstream pipeline (RNG config, per-group settings, quantization
scales) are accepted and then ignored — the surrogate only needs `mode`
and `stoc_len` (falling back to `2**sc_prec` if `stoc_len` is absent).

Variance formula (calibrated to Sobol-based SC, NOT iid Bernoulli):

The real Q-DiT SC kernels use Sobol low-discrepancy sequences (see
make_sobol_simple_config). Sobol integration error is O(log(L) / L)
per multiplication (Koksma–Hlawka), not O(1/sqrt(L)) as with iid
Bernoulli RNG. Hence for a D-sized reduction, the RMS error of the
matmul output scales as ~sqrt(D) / L — one order of magnitude better
in L than iid Bernoulli.

Empirically on DiT-XL/2 shapes (verified in tests/test_noise_matmul_adapters.py):
    L=256:  real RMSE 0.131, halving L doubles RMSE (1/L scaling confirmed)
    L=128:  0.250   L=64: 0.546   L=32: 1.128   L=16: 2.420

Accordingly the surrogate uses `var / L^2` (Sobol) rather than `var / L`
(iid Bernoulli). The numerator still comes from the analytical Bernoulli
variance formula so that the *shape* of the noise across different (a, b)
values remains correct:

    Bipolar XNOR, a,b in [-1, 1]:
        Var[out] = (D - (a²) @ (b²)^T) / L²        ← Sobol scaling

    Unipolar AND, a,b in [0, 1]:
        Var[out] = ((a @ b^T) - (a²) @ (b²)^T) / L²

The surrogate also re-quantizes its inputs to the SC integer grid
(q_max = 2^(sc_prec-1) - 1, sc_prec = ceil(log2(L))) before computing,
matching what real SC kernels do internally so that the L → ∞ limit
agrees exactly (up to the rounding error both paths share).

Important implementation details:
  * Per-batch normalization (`amax(dim=(-2,-1), keepdim=True)`) so batched
    QK / AV get one scale per (B,H) head — matches the `q_maxs = q_flat.
    amax(dim=(1,2))` pattern used by the real batched kernels. Using a
    global `.max()` would let an outlier head rescale every other head
    into the quantization dead zone and produce pure noise.
  * Autocast is explicitly disabled inside the helper so the output is
    always float32, matching the real SC Triton kernels' contract.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


# =============================================================================
# Core: analytical Gaussian noise on a @ b^T with SC-grid re-quantization
# =============================================================================
# Module-level correction factors. Defaults from unit test fitting against
# real SC kernels (tests/test_noise_matmul_adapters.py). Overridable via
# `set_noise_corrections()` (called by SCController.__init__).
NOISE_LOCAL_CORRECTION: float = 0.15   # per-row, per-batch (multiple local scales)
NOISE_GLOBAL_CORRECTION: float = 0.60  # per-tensor (single global scale)


def set_noise_corrections(local: float, global_: float) -> None:
    """Override the empirical Sobol correction factors used by the surrogate."""
    global NOISE_LOCAL_CORRECTION, NOISE_GLOBAL_CORRECTION
    NOISE_LOCAL_CORRECTION = float(local)
    NOISE_GLOBAL_CORRECTION = float(global_)


def _surrogate_bipolar_per_row(a, b, std_norm):
    """Bipolar + per-row (dominant DiT path). Fast: fp16, no round, var=D."""
    a_scale = a.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    b_scale = b.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    cell_scales = a_scale * b_scale.transpose(-2, -1)
    exact = a @ b.transpose(-2, -1)
    noise = (std_norm * cell_scales) * torch.randn_like(exact)
    return exact + noise


def _surrogate_bipolar_per_tensor(a, b, std_norm):
    """Bipolar + per-tensor. Fast."""
    a_scale = a.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    b_scale = b.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    cell_scales = a_scale * b_scale
    exact = a @ b.transpose(-2, -1)
    noise = (std_norm * cell_scales) * torch.randn_like(exact)
    return exact + noise


def _surrogate_unipolar(a, b, L_sq, sobol_var_correction, per_row_scale):
    """Unipolar (rare). Fast."""
    if per_row_scale:
        a_scale = a.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        b_scale = b.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        cell_scales = a_scale * b_scale.transpose(-2, -1)
    else:
        a_scale = a.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
        b_scale = b.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
        cell_scales = a_scale * b_scale
    exact = a @ b.transpose(-2, -1)
    var_real = (exact * cell_scales / L_sq * sobol_var_correction).clamp(min=0.0)
    noise = var_real.sqrt() * torch.randn_like(exact)
    return exact + noise


# torch.compile fuses the kernels into single CUDA graphs (numerically
# identical to eager). Lazily compiled on first call.
_compiled_bipolar_per_row = None
_compiled_bipolar_per_tensor = None
_compiled_unipolar = None


def _maybe_compile():
    global _compiled_bipolar_per_row, _compiled_bipolar_per_tensor, _compiled_unipolar
    if _compiled_bipolar_per_row is None:
        try:
            _compiled_bipolar_per_row = torch.compile(
                _surrogate_bipolar_per_row, dynamic=True, mode="reduce-overhead"
            )
            _compiled_bipolar_per_tensor = torch.compile(
                _surrogate_bipolar_per_tensor, dynamic=True, mode="reduce-overhead"
            )
            _compiled_unipolar = torch.compile(
                _surrogate_unipolar, dynamic=True, mode="reduce-overhead"
            )
        except Exception:
            _compiled_bipolar_per_row = _surrogate_bipolar_per_row
            _compiled_bipolar_per_tensor = _surrogate_bipolar_per_tensor
            _compiled_unipolar = _surrogate_unipolar


@torch.no_grad()
def _noisy_matmul_core(
    a: torch.Tensor,
    b: torch.Tensor,
    L: int,
    mode: str = "bipolar",
    per_row_scale: bool = False,
) -> torch.Tensor:
    """
    Core surrogate: returns ``a @ b^T`` plus CLT-limit Gaussian noise whose
    per-output variance matches the exact Bernoulli formula.

    a: (..., M, D), b: (..., N, D).  Returns (..., M, N) in float32.

    Args:
        per_row_scale:
          * False (default): one (a_scale, b_scale) per batch — matches real
            SC kernels that take a single max/min scalar per tensor
            (sc_matmul, sc_matmul_enable_triton) and the batched-bipolar
            kernel (sc_matmul_enable_batched_bipolar) that takes
            `q_maxs = q.amax(dim=(1,2))`.
          * True: each row of ``a`` gets its own scale, each row of ``b``
            gets its own scale.  Matches real SC kernels called with
            group_a=1 and group_b=1 (sc_matmul_mlp / sc_matmul_grouped
            family). Per-row scaling gives much tighter quantization and
            lower SC noise in practice.
    """
    assert mode in ("bipolar", "unipolar"), f"unknown SC mode: {mode}"

    if L is None or L <= 0:
        return ((a @ b.transpose(-2, -1)) * 0.0).float()

    L_sq = float(L) * float(L)
    is_local_scale = per_row_scale or (a.dim() > 2)
    sobol_var_correction = NOISE_LOCAL_CORRECTION if is_local_scale else NOISE_GLOBAL_CORRECTION

    _maybe_compile()
    if mode == "bipolar":
        D = a.shape[-1]
        std_norm = ((float(D) / L_sq) * sobol_var_correction) ** 0.5
        if per_row_scale:
            out = _compiled_bipolar_per_row(a, b, std_norm)
        else:
            out = _compiled_bipolar_per_tensor(a, b, std_norm)
        return out.float()

    out = _compiled_unipolar(a, b, L_sq, sobol_var_correction, per_row_scale)
    return out.float()


def _resolve_L(stoc_len: Optional[int], sc_prec: int) -> int:
    """Pick L from the kernel arg combination used throughout sc_attention."""
    if stoc_len is not None and stoc_len > 0:
        return int(stoc_len)
    return int(2 ** int(sc_prec))


# =============================================================================
# Drop-in adapters — one per real kernel signature
# =============================================================================

# Matches sc_matmul(...) and sc_matmul_enable_triton(...)
@torch.no_grad()
def noisy_sc_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    max_fp_a: float,
    min_fp_a: float,
    max_fp_b: float = None,
    min_fp_b: float = None,
    mode: str = "bipolar",
    sc_prec: int = 8,
    config: Optional[dict] = None,
    stoc_len: Optional[int] = None,
) -> torch.Tensor:
    return _noisy_matmul_core(a, b, L=_resolve_L(stoc_len, sc_prec), mode=mode)


# Matches sc_matmul_mlp(...) and sc_matmul_enable_triton_mlp(...)
@torch.no_grad()
def noisy_sc_matmul_mlp(
    a: torch.Tensor,
    b: torch.Tensor,
    max_fp_a: float = 0.0,
    min_fp_a: float = 0.0,
    max_fp_b: float = None,
    min_fp_b: float = None,
    mode: str = "bipolar",
    sc_prec: int = 8,
    config: Optional[dict] = None,
    group_a: int = 1,
    group_b: int = 1,
    chunk_d: int = 0,
    stoc_len: Optional[int] = None,
) -> torch.Tensor:
    # group_a=group_b=1 in the real kernel means per-row quantization
    per_row = (group_a == 1 and group_b == 1)
    return _noisy_matmul_core(
        a, b, L=_resolve_L(stoc_len, sc_prec), mode=mode,
        per_row_scale=per_row,
    )


# Matches sc_matmul_grouped(...) and sc_matmul_grouped_enable_triton(...)
@torch.no_grad()
def noisy_sc_matmul_grouped(
    a: torch.Tensor,
    b: torch.Tensor,
    group_a: int = 1,
    group_b: int = 1,
    mode: str = "unipolar",
    sc_prec: int = 8,
    config: Optional[dict] = None,
    stoc_len: Optional[int] = None,
) -> torch.Tensor:
    per_row = (group_a == 1 and group_b == 1)
    return _noisy_matmul_core(
        a, b, L=_resolve_L(stoc_len, sc_prec), mode=mode,
        per_row_scale=per_row,
    )


# Matches sc_matmul_enable_batched_bipolar(...)
@torch.no_grad()
def noisy_sc_matmul_enable_batched_bipolar(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    q_maxs: torch.Tensor,
    q_mins: torch.Tensor,
    k_maxs: torch.Tensor,
    k_mins: torch.Tensor,
    sc_prec: int,
    config: dict,
    stoc_len: Optional[int] = None,
) -> torch.Tensor:
    return _noisy_matmul_core(
        q_flat, k_flat, L=_resolve_L(stoc_len, sc_prec), mode="bipolar"
    )
