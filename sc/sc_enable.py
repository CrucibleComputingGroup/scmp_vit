"""
Enable-signal stochastic computing matrix multiplication.

Implements the enable-signal (conditional BSG) mechanism from UnarySim's FSUMul:
operand B's RNG index advances only when operand A's current bit is 1.
This preserves Sobol low-discrepancy properties for the bits that actually
contribute to the AND result, improving multiplication accuracy.

Two methods are provided:
- cycle_by_cycle: Exact simulation matching UnarySim FSUMul semantics (reference)
- k_shortcut: Vectorized equivalent using prefix-sum lookup tables (fast)

Supports:
- Bipolar mode (sign-magnitude): exact sign from integer, unipolar AND+enable on magnitudes
- Unipolar mode (asymmetric): direct AND+enable with zero-point correction
"""

import torch
import numpy as np
from typing import Optional

from sng import RNGPool, SNGBank


# =============================================================================
# Helper functions
# =============================================================================

def _compute_boundary(mag: torch.Tensor, max_val: float, max_rng_val: int) -> torch.Tensor:
    """
    Compute integer boundary for stochastic comparison.

    boundary = round(mag * max_rng_val / max_val)
    Maps [0, max_val] -> [0, max_rng_val].

    Args:
        mag: Integer magnitudes in [0, max_val]
        max_val: Maximum integer value (q_max)
        max_rng_val: Maximum RNG value (2^sc_prec = stoc_len)

    Returns:
        Integer boundary tensor, same shape as mag
    """
    return (mag.float() * max_rng_val / max_val).round().long()


def _build_cum_indicator_table(rng_b: torch.Tensor, max_rng_val: int) -> torch.Tensor:
    """
    Build cumulative indicator table for B's RNG sequence.

    cum_indicator[d, k, v] = |{i < k : v > rng_b[d, i]}|

    For each dimension d and each prefix length k, counts how many of the
    first k RNG values are strictly less than each possible boundary v.

    If A has k ones in its bitstream for dimension d,
    how many of B's enabled bits are 1 (given boundary v)?

    Args:
        rng_b: (D, stoc_len) int tensor, per-dimension RNG sequences for B
        max_rng_val: Maximum boundary value (2^sc_prec = stoc_len)

    Returns:
        (D, stoc_len+1, max_rng_val+1) int32 tensor
    """
    D, stoc_len = rng_b.shape
    device = rng_b.device

    # creates a 1D tensor [0, 1, 2, ..., max_rng_val]
    v_range = torch.arange(max_rng_val + 1, device=device)  # (max_rng_val+1,)

    # rng_b.unsqueeze(2) adds a trailing dimension to rng_b.
    # Shape goes from (D, stoc_len) → (D, stoc_len, 1).
    # This prepares it for broadcasting against the v dimension
    rng_b_exp = rng_b.unsqueeze(2)  # (D, stoc_len, 1)

    # delta is independent of A's bitstream, only depends on B's RNG and the boundary v.
    # delta[d, t, v] = 1 if v > rng_b[d, t] else 0  (strictly greater than)
    # (V) -> (1, 1, V)
    delta = (v_range.unsqueeze(0).unsqueeze(0) > rng_b_exp).int()  # (D, stoc_len, max_rng_val+1)

    # Cumulative sum along time dimension
    cum_inner = delta.cumsum(dim=1)  # (D, stoc_len, max_rng_val+1)

    # Prepend zeros for k=0
    zeros = torch.zeros(D, 1, max_rng_val + 1, dtype=torch.int32, device=device)
    cum = torch.cat([zeros, cum_inner], dim=1)  # (D, stoc_len+1, max_rng_val+1)

    return cum


def _compute_k_table(rng_a: torch.Tensor, max_rng_val: int) -> torch.Tensor:
    """
    Compute popcount table for A's RNG sequence.

    k_table[d, v] = |{t : v > rng_a[d, t]}| for v in [0, max_rng_val]

    For each dimension d and each possible boundary v, counts how many
    of A's RNG values are strictly less than v (i.e., how many 1-bits A would have).

    Args:
        rng_a: (D, stoc_len) int tensor, per-dimension RNG sequences for A
        max_rng_val: Maximum boundary value (2^sc_prec = stoc_len)

    Returns:
        (D, max_rng_val+1) int32 tensor
    """
    D, stoc_len = rng_a.shape
    device = rng_a.device

    # indicator[d, t, v] = 1 if v > rng_a[d, t]  (strictly greater than)
    v_range = torch.arange(max_rng_val + 1, device=device)  # (max_rng_val+1,)
    indicator = (v_range.unsqueeze(0).unsqueeze(0) > rng_a.unsqueeze(2))  # (D, stoc_len, max_rng_val+1)

    # Sum over time dimension
    k_table = indicator.sum(dim=1).int()  # (D, max_rng_val+1)

    return k_table


# =============================================================================
# Core multiplication functions
# =============================================================================

# debug purpose
def _enable_mul_cycle_by_cycle(
    mag_a: torch.Tensor,
    mag_b: torch.Tensor,
    rng_a: torch.Tensor,
    rng_b: torch.Tensor,
    max_val: float,
    sc_prec: int,
    return_per_dim: bool = False,
) -> torch.Tensor:
    """
    Enable-signal multiplication via exact cycle-by-cycle simulation.

    Matches UnarySim FSUMul semantics: B's RNG index advances only when
    A's current bit is 1 (AND gate with enable signal).

    Args:
        mag_a: (N, D) int tensor, magnitudes in [0, max_val]
        mag_b: (M, D) int tensor, magnitudes in [0, max_val]
        rng_a: (D, stoc_len) int tensor, per-dimension RNG sequences for A
        rng_b: (D, stoc_len) int tensor, per-dimension RNG sequences for B
        max_val: Maximum fp value (q_max)
        sc_prec: SC precision bits
        return_per_dim: If True return (N, M, D); if False return (N, M)

    Returns:
        Raw AND counts (not yet decoded)
    """
    N, D = mag_a.shape
    M = mag_b.shape[0]
    stoc_len = 2 ** sc_prec
    max_rng_val = 2 ** sc_prec
    device = mag_a.device

    # Compute boundaries
    boundary_a = _compute_boundary(mag_a, max_val, max_rng_val)  # (N, D)
    boundary_b = _compute_boundary(mag_b, max_val, max_rng_val)  # (M, D)

    # Enable indices per (n, d) — tracks where B's RNG is
    enable_idx = torch.zeros(N, D, dtype=torch.long, device=device)

    # Accumulate counts
    if return_per_dim:
        counts = torch.zeros(N, M, D, dtype=torch.long, device=device)
    else:
        counts = torch.zeros(N, M, dtype=torch.long, device=device)

    # Dimension index for advanced indexing
    d_indices = torch.arange(D, device=device).unsqueeze(0).expand(N, D)  # (N, D)

    for t in range(stoc_len):
        # A's bits at time t
        rng_a_t = rng_a[:, t]  # (D,)
        a_bits = (boundary_a > rng_a_t.unsqueeze(0)).long()  # (N, D)

        # B's RNG values at current enable indices
        # enable_idx[n,d] <= t < stoc_len, so always a valid index
        # Both d_indices and enable_idx have shape (N, D)                                                                                                  
        # Result[i, j] = rng_b[d_indices[i,j], enable_idx[i,j]] 
        rng_b_at_idx = rng_b[d_indices, enable_idx]  # (N, D)

        # B's bits for each (n, m, d)
        b_bits = (boundary_b.unsqueeze(0) > rng_b_at_idx.unsqueeze(1)).long()  # (N, M, D)

        # Output = a_bit AND b_bit
        output_bits = a_bits.unsqueeze(1) * b_bits  # (N, M, D)

        if return_per_dim:
            counts += output_bits
        else:
            counts += output_bits.sum(dim=2)

        # Update enable indices (advance only where A=1)
        enable_idx += a_bits

    return counts


def _enable_mul_k_shortcut(
    mag_a: torch.Tensor,
    mag_b: torch.Tensor,
    rng_a: torch.Tensor,
    rng_b: torch.Tensor,
    max_val: float,
    sc_prec: int,
    return_per_dim: bool = False,
) -> torch.Tensor:
    """
    Enable-signal multiplication via k-shortcut (prefix-sum lookup tables).

    Mathematically equivalent to cycle-by-cycle but eliminates the sequential
    dependency by precomputing lookup tables.

    Key insight: if k_a = popcount(A's bitstream), then
        count = |{i in 0..k_a-1 : rng_b[i] <= boundary_b}|
    which can be looked up from a precomputed cumulative indicator table.

    Args:
        mag_a: (N, D) int tensor, magnitudes in [0, max_val]
        mag_b: (M, D) int tensor, magnitudes in [0, max_val]
        rng_a: (D, stoc_len) int tensor, per-dimension RNG sequences for A
        rng_b: (D, stoc_len) int tensor, per-dimension RNG sequences for B
        max_val: Maximum fp value (q_max)
        sc_prec: SC precision bits
        return_per_dim: If True return (N, M, D); if False return (N, M)

    Returns:
        Raw AND counts (not yet decoded)
    """
    N, D = mag_a.shape
    M = mag_b.shape[0]
    max_rng_val = 2 ** sc_prec
    device = mag_a.device

    # Compute boundaries
    boundary_a = _compute_boundary(mag_a, max_val, max_rng_val)  # (N, D)
    boundary_b = _compute_boundary(mag_b, max_val, max_rng_val)  # (M, D)

    # Build lookup tables
    cum_indicator = _build_cum_indicator_table(rng_b, max_rng_val)  # (D, stoc_len+1, max_rng_val+1)
    k_table = _compute_k_table(rng_a, max_rng_val)  # (D, max_rng_val+1)

    if return_per_dim:
        # Vectorized lookup for all (n, m, d)
        # k_a[n, d] = k_table[d, boundary_a[n, d]]
        d_idx = torch.arange(D, device=device)
        k_a = k_table[d_idx.unsqueeze(0), boundary_a]  # (N, D)

        # count[n, m, d] = cum_indicator[d, k_a[n,d], boundary_b[m,d]]
        d_exp = d_idx.unsqueeze(0).unsqueeze(0).expand(N, M, D)  # (N, M, D)
        k_exp = k_a.unsqueeze(1).expand(N, M, D)  # (N, M, D)
        b_exp = boundary_b.unsqueeze(0).expand(N, M, D)  # (N, M, D)

        counts = cum_indicator[d_exp, k_exp, b_exp].long()  # (N, M, D)
    else:
        # Accumulate over dimensions to avoid materializing full (N, M, D)
        d_idx = torch.arange(D, device=device)
        k_a = k_table[d_idx.unsqueeze(0), boundary_a]  # (N, D)

        counts = torch.zeros(N, M, dtype=torch.long, device=device)
        for d in range(D):
            k_idx = k_a[:, d].long()  # (N,)
            b_idx = boundary_b[:, d].long()  # (M,)
            # cum_indicator[d, k_idx, :] → (N, max_rng_val+1)
            # then index by b_idx → (N, M)
            row_indexed = cum_indicator[d][k_idx]  # (N, max_rng_val+1)
            counts += row_indexed[:, b_idx].long()  # (N, M)

    return counts


# =============================================================================
# Bipolar and unipolar wrappers
# =============================================================================

def _sc_matmul_enable_bipolar(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
    rand_seqs_a_t, rand_seqs_b_t, N, D, M, stoc_len, method,
):
    """
    Enable-signal bipolar SC matmul using sign-magnitude decomposition.

    Sign is extracted exactly from the quantized integer. Magnitudes are
    multiplied using unipolar AND with enable-signal gating. Final result
    is reconstructed by applying signs and summing over dimensions.
    """
    q_max = 2 ** (sc_prec - 1) - 1
    q_min = -(2 ** (sc_prec - 1))

    # Per-operand symmetric scales
    abs_max_a = max(abs(max_fp_a), abs(min_fp_a), 1e-5)
    abs_max_b = max(abs(max_fp_b), abs(min_fp_b), 1e-5)
    scale_a = abs_max_a / q_max
    scale_b = abs_max_b / q_max

    # Quantize FP -> int with separate scales
    a_int = (a / scale_a).round().clamp(q_min, q_max)
    b_int = (b / scale_b).round().clamp(q_min, q_max)

    # Sign-magnitude decomposition
    sign_a = torch.sign(a_int)  # (N, D), values in {-1, 0, 1}
    sign_b = torch.sign(b_int)  # (M, D)
    mag_a = a_int.abs()  # (N, D), values in [0, q_max]
    mag_b = b_int.abs()  # (M, D)

    # Select method
    mul_fn = _enable_mul_cycle_by_cycle if method == "cycle_by_cycle" else _enable_mul_k_shortcut

    # Enable-signal multiplication on magnitudes (per-dimension)
    counts = mul_fn(mag_a, mag_b, rand_seqs_a_t, rand_seqs_b_t,
                    float(q_max), sc_prec, return_per_dim=True)  # (N, M, D)

    # Decode: counts -> integer product approximation
    decoded = counts.float() * float(q_max * q_max) / float(stoc_len)  # (N, M, D)

    # Apply signs per dimension
    signed = decoded * sign_a.unsqueeze(1) * sign_b.unsqueeze(0)  # (N, M, D)

    # Sum over dimensions and scale
    result = signed.sum(dim=2) * (scale_a * scale_b)

    return result


def _sc_matmul_enable_unipolar(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
    rand_seqs_a_t, rand_seqs_b_t, N, D, M, stoc_len, method,
):
    """
    Enable-signal unipolar SC matmul with asymmetric quantization.

    Same zero-point correction as standard unipolar sc_matmul.
    """
    q_max = 2 ** sc_prec - 1
    q_min = 0

    # Per-operand asymmetric scales and zero points
    range_a = max(max_fp_a - min_fp_a, 1e-5)
    scale_a = range_a / q_max
    zp_a = round(-min_fp_a / scale_a)
    zp_a = max(q_min, min(q_max, zp_a))

    range_b = max(max_fp_b - min_fp_b, 1e-5)
    scale_b = range_b / q_max
    zp_b = round(-min_fp_b / scale_b)
    zp_b = max(q_min, min(q_max, zp_b))

    # Quantize FP -> int (non-negative)
    a_int = (a / scale_a + zp_a).round().clamp(q_min, q_max)
    b_int = (b / scale_b + zp_b).round().clamp(q_min, q_max)

    # Select method
    mul_fn = _enable_mul_cycle_by_cycle if method == "cycle_by_cycle" else _enable_mul_k_shortcut

    # Enable-signal multiplication (sum over dims)
    counts = mul_fn(a_int, b_int, rand_seqs_a_t, rand_seqs_b_t,
                    float(q_max), sc_prec, return_per_dim=False)  # (N, M)

    # Decode: counts -> integer product approximation
    sc_raw = counts.float() * float(q_max * q_max) / float(stoc_len)

    # Zero-point correction (same as standard unipolar)
    zp_a_f = float(zp_a)
    zp_b_f = float(zp_b)
    a_sum = a_int.sum(dim=-1, keepdim=True)  # (N, 1)
    b_sum = b_int.sum(dim=-1, keepdim=True)  # (M, 1)

    correction = -zp_b_f * a_sum - zp_a_f * b_sum.transpose(-2, -1) + D * zp_a_f * zp_b_f
    corrected = sc_raw + correction

    # Dequantize
    result = corrected * (scale_a * scale_b)

    return result


# =============================================================================
# Public API
# =============================================================================

@torch.no_grad()
def sc_matmul_enable(
    a: torch.Tensor,
    b: torch.Tensor,
    max_fp_a: float,
    min_fp_a: float,
    max_fp_b: float = None,
    min_fp_b: float = None,
    mode: str = "bipolar",
    sc_prec: int = 8,
    config: Optional[dict] = None,
    method: str = "k_shortcut",
) -> torch.Tensor:
    """
    Enable-signal stochastic computing matrix multiplication: a @ b^T.

    FP-in, FP-out. Uses the enable-signal (conditional BSG) mechanism where
    B's RNG index advances only when A's current bit is 1.

    Args:
        a: Left operand, shape (N, D) or (B, N, D). FP values.
        b: Right operand, shape (M, D) or (B, M, D). FP values.
        max_fp_a: Max FP value for operand a.
        min_fp_a: Min FP value for operand a.
        max_fp_b: Max FP value for operand b. If None, uses max_fp_a.
        min_fp_b: Min FP value for operand b. If None, uses min_fp_a.
        mode: "bipolar" (symmetric sign-magnitude) or "unipolar" (asymmetric AND).
        sc_prec: SC precision. stoc_len = 2^sc_prec.
        config: Optional SC RNG/SNG config dict. If None, uses sobol_simple.
        method: "k_shortcut" (fast vectorized) or "cycle_by_cycle" (exact reference).

    Returns:
        Result tensor in FP, shape (N, M) or (B, N, M).
    """
    if max_fp_b is None:
        max_fp_b = max_fp_a
    if min_fp_b is None:
        min_fp_b = min_fp_a

    if a.dim() == 3:
        return _sc_matmul_enable_batched(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, mode, sc_prec, config, method
        )

    assert a.dim() == 2 and b.dim() == 2, f"Expected 2D tensors, got a:{a.dim()}D, b:{b.dim()}D"
    assert a.shape[1] == b.shape[1], f"Embedding dim mismatch: a={a.shape[1]}, b={b.shape[1]}"
    assert method in ("k_shortcut", "cycle_by_cycle"), f"Unknown method: {method}"

    N, D = a.shape
    M = b.shape[0]
    stoc_len = 2 ** sc_prec

    # Build config
    if config is None:
        from config_helpers import make_sobol_simple_config
        config = make_sobol_simple_config(D, D, sc_prec)

    # Ensure float32
    device = a.device
    a = a.float()
    b = b.float()

    # Build RNG pool and SNG banks
    rng_pool = RNGPool(config["rng_pool"], sc_prec)
    sng_a = SNGBank(rng_pool, config["sng"]["q"])
    sng_b = SNGBank(rng_pool, config["sng"]["k"])

    rand_seqs_a = sng_a.get_all_sequences(stoc_len)
    rand_seqs_b = sng_b.get_all_sequences(stoc_len)
    rand_seqs_a_t = torch.tensor(rand_seqs_a, dtype=torch.long, device=device)
    rand_seqs_b_t = torch.tensor(rand_seqs_b, dtype=torch.long, device=device)

    if mode == "bipolar":
        result = _sc_matmul_enable_bipolar(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
            rand_seqs_a_t, rand_seqs_b_t, N, D, M, stoc_len, method,
        )
    elif mode == "unipolar":
        result = _sc_matmul_enable_unipolar(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
            rand_seqs_a_t, rand_seqs_b_t, N, D, M, stoc_len, method,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'bipolar' or 'unipolar'.")

    return result


def _sc_matmul_enable_batched(
    a: torch.Tensor,
    b: torch.Tensor,
    max_fp_a: float,
    min_fp_a: float,
    max_fp_b: float,
    min_fp_b: float,
    mode: str,
    sc_prec: int,
    config: Optional[dict],
    method: str,
) -> torch.Tensor:
    """Batched enable-signal SC matrix multiplication."""
    B = a.shape[0]
    results = []
    for i in range(B):
        result = sc_matmul_enable(
            a[i], b[i], max_fp_a, min_fp_a, max_fp_b, min_fp_b,
            mode, sc_prec, config, method,
        )
        results.append(result)
    return torch.stack(results, dim=0)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    from config_helpers import make_sobol_simple_config

    print("Testing sc_matmul_enable()...")
    print("=" * 60)

    N, D, M = 8, 64, 8
    config = make_sobol_simple_config(D, D, 8)

    # --- Test bipolar mode ---
    print("\n  Bipolar mode (sign-magnitude, enable-signal):")
    torch.manual_seed(42)
    a = torch.randn(N, D) * 3.0
    b = torch.randn(M, D) * 3.0

    max_fp = max(a.abs().max().item(), b.abs().max().item())

    # Test both methods
    result_cbc = sc_matmul_enable(a, b, max_fp, -max_fp, mode="bipolar", sc_prec=8,
                                   config=config, method="cycle_by_cycle")
    result_ks = sc_matmul_enable(a, b, max_fp, -max_fp, mode="bipolar", sc_prec=8,
                                  config=config, method="k_shortcut")
    result_gt = a @ b.T

    # Verify methods match
    cbc_ks_diff = (result_cbc - result_ks).abs().max().item()
    print(f"    cycle_by_cycle vs k_shortcut max diff: {cbc_ks_diff:.6e}")
    assert cbc_ks_diff < 1e-3, f"Methods don't match! diff={cbc_ks_diff}"

    # Compare with ground truth
    max_dot = D * max_fp * max_fp
    rmse = ((result_ks - result_gt) ** 2).mean().sqrt().item() / max_dot
    print(f"    RMSE (normalized): {rmse:.6f}")
    print(f"    SC sample:  {result_ks[0, :3].tolist()}")
    print(f"    GT sample:  {result_gt[0, :3].tolist()}")
    bipolar_pass = rmse < 0.1
    print(f"    [{'PASS' if bipolar_pass else 'FAIL'}]")

    # --- Test unipolar mode ---
    print("\n  Unipolar mode (asymmetric, enable-signal):")
    torch.manual_seed(42)
    a_uni = torch.rand(N, D)
    b_uni = torch.rand(M, D)

    result_cbc_uni = sc_matmul_enable(a_uni, b_uni, 1.0, 0.0, mode="unipolar", sc_prec=8,
                                       config=config, method="cycle_by_cycle")
    result_ks_uni = sc_matmul_enable(a_uni, b_uni, 1.0, 0.0, mode="unipolar", sc_prec=8,
                                      config=config, method="k_shortcut")
    result_gt_uni = a_uni @ b_uni.T

    cbc_ks_diff_uni = (result_cbc_uni - result_ks_uni).abs().max().item()
    print(f"    cycle_by_cycle vs k_shortcut max diff: {cbc_ks_diff_uni:.6e}")
    assert cbc_ks_diff_uni < 1e-3, f"Methods don't match! diff={cbc_ks_diff_uni}"

    max_dot_uni = D * 1.0
    rmse_uni = ((result_ks_uni - result_gt_uni) ** 2).mean().sqrt().item() / max_dot_uni
    print(f"    RMSE (normalized): {rmse_uni:.6f}")
    print(f"    SC sample:  {result_ks_uni[0, :3].tolist()}")
    print(f"    GT sample:  {result_gt_uni[0, :3].tolist()}")
    unipolar_pass = rmse_uni < 0.1
    print(f"    [{'PASS' if unipolar_pass else 'FAIL'}]")

    print("=" * 60)
    all_pass = bipolar_pass and unipolar_pass
    print(f"sc_matmul_enable: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
