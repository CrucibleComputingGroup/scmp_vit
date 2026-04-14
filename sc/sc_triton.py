"""
Triton GPU-accelerated stochastic computing matrix multiplication.

This module provides a drop-in replacement for matmul_sc() using Triton kernels
with bit-packed representation for maximum performance.

Supports:
- Bipolar mode (XNOR gate): for symmetric quantization, values in [-max, max]
- Unipolar mode (AND gate): for asymmetric quantization, values in [0, max]

Supports all config types:
- LFSR with per-element scrambling
- Fully independent LFSRs
- Sobol sequences (simple and DSE)
"""
from __future__ import annotations

import json
import os
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
import numpy as np
from typing import Optional

# Import from new architecture
from sng import RNGPool, SNGBank
from sc import FP8_E4M3_MAX, FP8_E5M2_MAX, INT8_MAX


# =============================================================================
# RNG Sequence Cache
# =============================================================================

_rng_seq_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _config_cache_key(config: dict, sc_prec: int, device: torch.device) -> str:
    """Create a hashable cache key from config, precision, and device."""
    return json.dumps(config, sort_keys=True) + f"|{sc_prec}|{device}"


def _get_cached_sequences(
    config: dict, sc_prec: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get RNG sequences from cache, or generate and cache on miss.

    Returns:
        (rand_seqs_a_t, rand_seqs_b_t): int32 tensors on the given device.
    """
    key = _config_cache_key(config, sc_prec, device)
    if key not in _rng_seq_cache:
        stoc_len = 2 ** sc_prec
        rng_pool = RNGPool(config["rng_pool"], sc_prec)
        sng_a = SNGBank(rng_pool, config["sng"]["q"])
        sng_b = SNGBank(rng_pool, config["sng"]["k"])
        rand_seqs_a_t = torch.tensor(
            sng_a.get_all_sequences(stoc_len), dtype=torch.int32, device=device)
        rand_seqs_b_t = torch.tensor(
            sng_b.get_all_sequences(stoc_len), dtype=torch.int32, device=device)
        _rng_seq_cache[key] = (rand_seqs_a_t, rand_seqs_b_t)
    return _rng_seq_cache[key]


def clear_rng_cache():
    """Clear the cached RNG sequences and enable tables to free GPU memory."""
    _rng_seq_cache.clear()
    _enable_table_cache.clear()
    _k_table_cache.clear()


# =============================================================================
# Kernel 1: bin_to_stoc_packed_kernel
# Converts binary values to packed stochastic bitstreams
# =============================================================================

@triton.jit
def bin_to_stoc_packed_kernel(
    values_ptr,      # (M, N) float32 input matrix
    packed_ptr,      # (M, N, NUM_PACKS) int64 output (packed bits)
    rng_seqs_ptr,    # (N, stoc_len) int32 per-element RNG sequences
    max_val,         # float: max value for normalization
    max_rng_val,     # int: 2^sc_prec - 1
    M: tl.constexpr,
    N: tl.constexpr,
    stoc_len: tl.constexpr,
    NUM_PACKS: tl.constexpr,
):
    """
    Convert one matrix element to a packed stochastic bitstream.

    Each program (thread) handles one (m, n) element.
    Output: NUM_PACKS int64 values containing stoc_len packed bits.
    """
    # Get element indices
    m = tl.program_id(0)
    n = tl.program_id(1)

    # Bounds check
    if m >= M or n >= N:
        return

    # Load the binary value
    val_offset = m * N + n
    val = tl.load(values_ptr + val_offset)

    # Compute boundary for stochastic comparison
    # prob_1 = (val / max_val + 1) / 2  maps [-max_val, max_val] to [0, 1]
    # boundary = prob_1 * max_rng_val
    prob_bipolar = val / max_val
    prob_1 = (prob_bipolar + 1.0) / 2.0
    boundary = libdevice.nearbyint(prob_1 * max_rng_val).to(tl.int32)

    # Process each pack - use per-element RNG sequence
    for pack_idx in tl.static_range(NUM_PACKS):
        # Load 64 RNG values for this pack from THIS element's sequence
        bit_offsets = pack_idx * 64 + tl.arange(0, 64)
        mask = bit_offsets < stoc_len
        # Access rng_seqs[n, bit_offsets] = rng_seqs_ptr + n * stoc_len + bit_offsets
        rng_vals = tl.load(rng_seqs_ptr + n * stoc_len + bit_offsets, mask=mask, other=max_rng_val + 1)

        # Compare against boundary: 1 if rng_val <= boundary
        bits = tl.where(boundary > rng_vals, 1, 0).to(tl.int64)

        # Pack bits: sum of (bit << position)
        bit_positions = tl.arange(0, 64).to(tl.int64)
        packed_bits = tl.sum(bits << bit_positions)

        # Store packed result
        out_offset = (m * N + n) * NUM_PACKS + pack_idx
        tl.store(packed_ptr + out_offset, packed_bits)


# =============================================================================
# Kernel 1b: bin_to_stoc_packed_unipolar_kernel
# Converts non-negative values to packed stochastic bitstreams (unipolar)
# =============================================================================

@triton.jit
def bin_to_stoc_packed_unipolar_kernel(
    values_ptr,      # (M, N) float32 input matrix, values in [0, max_val]
    packed_ptr,      # (M, N, NUM_PACKS) int64 output (packed bits)
    rng_seqs_ptr,    # (N, stoc_len) int32 per-element RNG sequences
    max_val,         # float: max value for normalization
    max_rng_val,     # int: 2^sc_prec - 1
    M: tl.constexpr,
    N: tl.constexpr,
    stoc_len: tl.constexpr,
    NUM_PACKS: tl.constexpr,
):
    """
    Convert one matrix element to a packed stochastic bitstream (unipolar).

    Unipolar encoding: prob = val / max_val, maps [0, max_val] to [0, 1].
    Used with AND gate for multiplication.

    Each program (thread) handles one (m, n) element.
    Output: NUM_PACKS int64 values containing stoc_len packed bits.
    """
    # Get element indices
    m = tl.program_id(0)
    n = tl.program_id(1)

    # Bounds check
    if m >= M or n >= N:
        return

    # Load the value
    val_offset = m * N + n
    val = tl.load(values_ptr + val_offset)

    # Compute boundary for stochastic comparison
    # Unipolar: prob = val / max_val, maps [0, max_val] to [0, 1]
    prob_1 = val / max_val
    boundary = libdevice.nearbyint(prob_1 * max_rng_val).to(tl.int32)

    # Process each pack - use per-element RNG sequence
    for pack_idx in tl.static_range(NUM_PACKS):
        bit_offsets = pack_idx * 64 + tl.arange(0, 64)
        mask = bit_offsets < stoc_len
        rng_vals = tl.load(rng_seqs_ptr + n * stoc_len + bit_offsets, mask=mask, other=max_rng_val + 1)

        # Compare against boundary: 1 if rng_val <= boundary
        bits = tl.where(boundary > rng_vals, 1, 0).to(tl.int64)

        # Pack bits: sum of (bit << position)
        bit_positions = tl.arange(0, 64).to(tl.int64)
        packed_bits = tl.sum(bits << bit_positions)

        # Store packed result
        out_offset = (m * N + n) * NUM_PACKS + pack_idx
        tl.store(packed_ptr + out_offset, packed_bits)


# =============================================================================
# Kernel 2: xnor_matmul_kernel
# Computes XNOR-based matrix multiplication using packed streams
# =============================================================================

@triton.jit
def xnor_matmul_kernel(
    Q_packed_ptr,    # (Q_l, Q_e, NUM_PACKS) int64
    K_packed_ptr,    # (K_l, K_e, NUM_PACKS) int64
    output_ptr,      # (Q_l, K_l) float32
    Q_l: tl.constexpr,
    Q_e: tl.constexpr,
    K_l: tl.constexpr,
    NUM_PACKS: tl.constexpr,
    stoc_len: tl.constexpr,
    max_val_squared,  # float: max_val^2 for decoding
):
    """
    Compute one output element of the SC matrix multiplication.

    Each program computes output[i, j] = sum_e decode(XNOR(Q[i,e], K[j,e]))
    """
    # Get output indices
    i = tl.program_id(0)  # Q row
    j = tl.program_id(1)  # K row

    # Bounds check
    if i >= Q_l or j >= K_l:
        return

    # Accumulate dot product (scalar)
    dot_product = 0.0

    # Loop over embedding dimension
    for e in tl.static_range(Q_e):
        xnor_ones = 0

        # Process each pack - load all packs for this embedding dim
        pack_offsets = tl.arange(0, NUM_PACKS)
        q_base = (i * Q_e + e) * NUM_PACKS
        k_base = (j * Q_e + e) * NUM_PACKS

        q_packs = tl.load(Q_packed_ptr + q_base + pack_offsets)
        k_packs = tl.load(K_packed_ptr + k_base + pack_offsets)

        # XNOR = NOT(XOR) for bipolar multiplication
        xnor_results = ~(q_packs ^ k_packs)

        # Count ones using popcount and sum
        popcounts = libdevice.popc(xnor_results)
        xnor_ones = tl.sum(popcounts)

        # Decode bipolar: (2 * xnor_ones / stoc_len - 1) * max_val^2
        prob_1 = xnor_ones.to(tl.float32) / stoc_len
        bipolar_val = 2.0 * prob_1 - 1.0
        decoded = bipolar_val * max_val_squared

        dot_product = dot_product + decoded

    # Store result
    out_offset = i * K_l + j
    tl.store(output_ptr + out_offset, dot_product)


# =============================================================================
# Kernel 3: and_matmul_kernel
# Computes AND-based matrix multiplication using packed streams (unipolar)
# =============================================================================

@triton.jit
def and_matmul_kernel(
    A_packed_ptr,    # (A_l, A_e, NUM_PACKS) int64
    B_packed_ptr,    # (B_l, B_e, NUM_PACKS) int64
    output_ptr,      # (A_l, B_l) float32
    A_l: tl.constexpr,
    A_e: tl.constexpr,
    B_l: tl.constexpr,
    NUM_PACKS: tl.constexpr,
    stoc_len: tl.constexpr,
    max_val_squared,  # float: max_val^2 for decoding
):
    """
    Compute one output element of the unipolar SC matrix multiplication.

    Uses AND gate: P(output=1) = P(a=1) * P(b=1)
    Decode: (ones / stoc_len) * max_val^2

    Each program computes output[i, j] = sum_e decode(AND(A[i,e], B[j,e]))
    """
    # Get output indices
    i = tl.program_id(0)  # A row
    j = tl.program_id(1)  # B row

    # Bounds check
    if i >= A_l or j >= B_l:
        return

    # Accumulate dot product (scalar)
    dot_product = 0.0

    # Loop over embedding dimension
    for e in tl.static_range(A_e):
        # Process each pack - load all packs for this embedding dim
        pack_offsets = tl.arange(0, NUM_PACKS)
        a_base = (i * A_e + e) * NUM_PACKS
        b_base = (j * A_e + e) * NUM_PACKS

        a_packs = tl.load(A_packed_ptr + a_base + pack_offsets)
        b_packs = tl.load(B_packed_ptr + b_base + pack_offsets)

        # AND gate for unipolar multiplication
        and_results = a_packs & b_packs

        # Count ones using popcount and sum
        popcounts = libdevice.popc(and_results)
        and_ones = tl.sum(popcounts)

        # Decode unipolar: (ones / stoc_len) * max_val^2
        prob_1 = and_ones.to(tl.float32) / stoc_len
        decoded = prob_1 * max_val_squared

        dot_product = dot_product + decoded

    # Store result
    out_offset = i * B_l + j
    tl.store(output_ptr + out_offset, dot_product)


# =============================================================================
# Kernel 2b: xnor_matmul_tiled_kernel (tiled version)
# =============================================================================

@triton.jit
def xnor_matmul_tiled_kernel(
    Q_packed_ptr,    # (Q_l, Q_e, NUM_PACKS) int64
    K_packed_ptr,    # (K_l, K_e, NUM_PACKS) int64
    output_ptr,      # (Q_l, K_l) float32
    Q_l, Q_e, K_l,
    NUM_PACKS: tl.constexpr,
    stoc_len: tl.constexpr,
    max_val_squared,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Tiled XNOR matmul: each program computes a BLOCK_M x BLOCK_N output tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    m_mask = m_offsets < Q_l
    n_mask = n_offsets < K_l

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for e in range(Q_e):
        e_ones = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)

        for p in tl.static_range(NUM_PACKS):
            # Load Q tile: Q_packed[m_offsets, e, p] -> [BLOCK_M]
            q_idx = (m_offsets * Q_e + e) * NUM_PACKS + p
            q_packs = tl.load(Q_packed_ptr + q_idx, mask=m_mask, other=0)

            # Load K tile: K_packed[n_offsets, e, p] -> [BLOCK_N]
            k_idx = (n_offsets * Q_e + e) * NUM_PACKS + p
            k_packs = tl.load(K_packed_ptr + k_idx, mask=n_mask, other=0)

            # XNOR: broadcast [BLOCK_M, 1] x [1, BLOCK_N] -> [BLOCK_M, BLOCK_N]
            xnor_results = ~(q_packs[:, None] ^ k_packs[None, :])
            e_ones += libdevice.popc(xnor_results)

        # Decode this embedding dimension
        prob = e_ones.to(tl.float32) / stoc_len
        decoded = (2.0 * prob - 1.0) * max_val_squared
        acc += decoded

    # Store tile
    out_offsets = m_offsets[:, None] * K_l + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offsets, acc, mask=out_mask)


# =============================================================================
# Kernel 3b: and_matmul_tiled_kernel (tiled version)
# =============================================================================

@triton.jit
def and_matmul_tiled_kernel(
    A_packed_ptr,    # (A_l, A_e, NUM_PACKS) int64
    B_packed_ptr,    # (B_l, B_e, NUM_PACKS) int64
    output_ptr,      # (A_l, B_l) float32
    A_l, A_e, B_l,
    NUM_PACKS: tl.constexpr,
    stoc_len: tl.constexpr,
    max_val_squared,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Tiled AND matmul: each program computes a BLOCK_M x BLOCK_N output tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < A_l
    n_mask = n_offsets < B_l

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for e in range(A_e):
        e_ones = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)

        for p in tl.static_range(NUM_PACKS):
            a_idx = (m_offsets * A_e + e) * NUM_PACKS + p
            a_packs = tl.load(A_packed_ptr + a_idx, mask=m_mask, other=0)

            b_idx = (n_offsets * A_e + e) * NUM_PACKS + p
            b_packs = tl.load(B_packed_ptr + b_idx, mask=n_mask, other=0)

            and_results = a_packs[:, None] & b_packs[None, :]
            e_ones += libdevice.popc(and_results)

        prob = e_ones.to(tl.float32) / stoc_len
        decoded = prob * max_val_squared
        acc += decoded

    out_offsets = m_offsets[:, None] * B_l + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offsets, acc, mask=out_mask)


# =============================================================================
# Fused SNG+Matmul Kernels (no intermediate packed tensors)
# =============================================================================

@triton.jit
def fused_xnor_matmul_kernel(
    a_ptr,           # (N, D) float32 quantized values
    b_ptr,           # (M, D) float32 quantized values
    rng_a_ptr,       # (D, stoc_len) int32
    rng_b_ptr,       # (D, stoc_len) int32
    output_ptr,      # (N, M) float32
    N, D, M,
    stoc_len: tl.constexpr,
    NUM_PACKS: tl.constexpr,
    max_val,         # float: q_max for boundary computation
    max_rng_val,     # int: 2^sc_prec - 1
    max_val_squared, # float: q_max^2
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused SNG encoding + XNOR matmul. No intermediate packed tensors.

    For each tile, generates packed bitstreams on-the-fly and immediately
    computes XNOR + popcount, avoiding global memory writes/reads for
    the packed representation.

    BLOCK_K tiles the embedding dimension so multiple dims are loaded per
    iteration, improving data reuse and reducing loop overhead.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < N
    n_mask = n_offsets < M

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Tile the embedding dimension by BLOCK_K
    num_k_blocks = (D + BLOCK_K - 1) // BLOCK_K
    for k_block in range(num_k_blocks):
        k_start = k_block * BLOCK_K
        # Process each embedding dim within this K-tile
        for ki in tl.static_range(BLOCK_K):
            e = k_start + ki
            if e < D:
                # Load values for this embedding dim
                a_vals = tl.load(a_ptr + m_offsets * D + e, mask=m_mask, other=0.0)
                b_vals = tl.load(b_ptr + n_offsets * D + e, mask=n_mask, other=0.0)

                # Compute boundaries (bipolar encoding)
                prob_a = (a_vals / max_val + 1.0) / 2.0
                boundary_a = libdevice.nearbyint(prob_a * max_rng_val).to(tl.int32)
                prob_b = (b_vals / max_val + 1.0) / 2.0
                boundary_b = libdevice.nearbyint(prob_b * max_rng_val).to(tl.int32)

                e_ones = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)

                for p in tl.static_range(NUM_PACKS):
                    bit_offsets = p * 64 + tl.arange(0, 64)
                    bit_mask = bit_offsets < stoc_len

                    # Load RNG values for element e, this pack
                    rng_a_vals = tl.load(rng_a_ptr + e * stoc_len + bit_offsets,
                                          mask=bit_mask, other=max_rng_val + 1)
                    rng_b_vals = tl.load(rng_b_ptr + e * stoc_len + bit_offsets,
                                          mask=bit_mask, other=max_rng_val + 1)

                    # SNG for a: compare boundary_a[BLOCK_M] against rng[64]
                    a_bits = tl.where(boundary_a[:, None] > rng_a_vals[None, :],
                                      tl.full([1], 1, dtype=tl.int64),
                                      tl.full([1], 0, dtype=tl.int64))
                    bit_positions = tl.arange(0, 64).to(tl.int64)
                    a_packed = tl.sum(a_bits << bit_positions[None, :], axis=1)  # [BLOCK_M]

                    b_bits = tl.where(boundary_b[:, None] > rng_b_vals[None, :],
                                      tl.full([1], 1, dtype=tl.int64),
                                      tl.full([1], 0, dtype=tl.int64))
                    b_packed = tl.sum(b_bits << bit_positions[None, :], axis=1)  # [BLOCK_N]

                    # XNOR + popcount
                    xnor = ~(a_packed[:, None] ^ b_packed[None, :])
                    e_ones += libdevice.popc(xnor)

                # Decode bipolar
                prob = e_ones.to(tl.float32) / stoc_len
                decoded = (2.0 * prob - 1.0) * max_val_squared
                acc += decoded

    out_offsets = m_offsets[:, None] * M + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offsets, acc, mask=out_mask)


@triton.jit
def fused_and_matmul_kernel(
    a_ptr,           # (N, D) float32 quantized values
    b_ptr,           # (M, D) float32 quantized values
    rng_a_ptr,       # (D, stoc_len) int32
    rng_b_ptr,       # (D, stoc_len) int32
    output_ptr,      # (N, M) float32
    N, D, M,
    stoc_len: tl.constexpr,
    NUM_PACKS: tl.constexpr,
    max_val,         # float: q_max for boundary computation
    max_rng_val,     # int: 2^sc_prec - 1
    max_val_squared, # float: q_max^2
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused SNG encoding + AND matmul (unipolar). No intermediate packed tensors.

    BLOCK_K tiles the embedding dimension for better loop unrolling.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < N
    n_mask = n_offsets < M

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    num_k_blocks = (D + BLOCK_K - 1) // BLOCK_K
    for k_block in range(num_k_blocks):
        k_start = k_block * BLOCK_K
        for ki in tl.static_range(BLOCK_K):
            e = k_start + ki
            if e < D:
                a_vals = tl.load(a_ptr + m_offsets * D + e, mask=m_mask, other=0.0)
                b_vals = tl.load(b_ptr + n_offsets * D + e, mask=n_mask, other=0.0)

                # Unipolar encoding: prob = val / max_val
                boundary_a = libdevice.nearbyint(a_vals / max_val * max_rng_val).to(tl.int32)
                boundary_b = libdevice.nearbyint(b_vals / max_val * max_rng_val).to(tl.int32)

                e_ones = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)

                for p in tl.static_range(NUM_PACKS):
                    bit_offsets = p * 64 + tl.arange(0, 64)
                    bit_mask = bit_offsets < stoc_len

                    rng_a_vals = tl.load(rng_a_ptr + e * stoc_len + bit_offsets,
                                          mask=bit_mask, other=max_rng_val + 1)
                    rng_b_vals = tl.load(rng_b_ptr + e * stoc_len + bit_offsets,
                                          mask=bit_mask, other=max_rng_val + 1)

                    a_bits = tl.where(boundary_a[:, None] > rng_a_vals[None, :],
                                      tl.full([1], 1, dtype=tl.int64),
                                      tl.full([1], 0, dtype=tl.int64))
                    bit_positions = tl.arange(0, 64).to(tl.int64)
                    a_packed = tl.sum(a_bits << bit_positions[None, :], axis=1)

                    b_bits = tl.where(boundary_b[:, None] > rng_b_vals[None, :],
                                      tl.full([1], 1, dtype=tl.int64),
                                      tl.full([1], 0, dtype=tl.int64))
                    b_packed = tl.sum(b_bits << bit_positions[None, :], axis=1)

                    # AND + popcount
                    and_result = a_packed[:, None] & b_packed[None, :]
                    e_ones += libdevice.popc(and_result)

                # Decode unipolar
                prob = e_ones.to(tl.float32) / stoc_len
                decoded = prob * max_val_squared
                acc += decoded

    out_offsets = m_offsets[:, None] * M + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offsets, acc, mask=out_mask)


# =============================================================================
# Fused SNG+Matmul Host Functions
# =============================================================================

def fused_xnor_matmul(
    a_int: torch.Tensor, b_int: torch.Tensor,
    rng_a: torch.Tensor, rng_b: torch.Tensor,
    q_max: float, sc_prec: int,
) -> torch.Tensor:
    """Fused bipolar SNG encoding + XNOR matmul. No intermediate packed tensors."""
    N, D = a_int.shape
    M = b_int.shape[0]
    stoc_len = 2 ** sc_prec
    NUM_PACKS = stoc_len // 64
    max_rng_val = 2 ** sc_prec
    q_max_sq = float(q_max * q_max)

    output = torch.empty(N, M, dtype=torch.float32, device=a_int.device)

    BLOCK_M = 32
    BLOCK_N = 32
    # Choose BLOCK_K as the largest divisor of D among {8, 4, 2, 1}, or 8 if D>=8
    if D >= 8 and D % 8 == 0:
        BLOCK_K = 8
    elif D >= 4 and D % 4 == 0:
        BLOCK_K = 4
    elif D % 2 == 0:
        BLOCK_K = 2
    else:
        BLOCK_K = 1
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N))
    fused_xnor_matmul_kernel[grid](
        a_int, b_int, rng_a, rng_b, output,
        N, D, M, stoc_len, NUM_PACKS,
        float(q_max), max_rng_val, q_max_sq,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return output


def fused_and_matmul(
    a_int: torch.Tensor, b_int: torch.Tensor,
    rng_a: torch.Tensor, rng_b: torch.Tensor,
    q_max: float, sc_prec: int,
) -> torch.Tensor:
    """Fused unipolar SNG encoding + AND matmul. No intermediate packed tensors."""
    N, D = a_int.shape
    M = b_int.shape[0]
    stoc_len = 2 ** sc_prec
    NUM_PACKS = stoc_len // 64
    max_rng_val = 2 ** sc_prec
    q_max_sq = float(q_max * q_max)

    output = torch.empty(N, M, dtype=torch.float32, device=a_int.device)

    BLOCK_M = 32
    BLOCK_N = 32
    if D >= 8 and D % 8 == 0:
        BLOCK_K = 8
    elif D >= 4 and D % 4 == 0:
        BLOCK_K = 4
    elif D % 2 == 0:
        BLOCK_K = 2
    else:
        BLOCK_K = 1
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N))
    fused_and_matmul_kernel[grid](
        a_int, b_int, rng_a, rng_b, output,
        N, D, M, stoc_len, NUM_PACKS,
        float(q_max), max_rng_val, q_max_sq,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return output


# =============================================================================
# Host functions
# =============================================================================

def bin_to_stoc_packed(values: torch.Tensor, rng_seqs: torch.Tensor,
                        max_val: float, sc_prec: int) -> torch.Tensor:
    """
    Convert a matrix of values to packed stochastic bitstreams.

    Args:
        values: (M, N) float32 tensor
        rng_seqs: (N, stoc_len) int32 tensor with per-element RNG sequences
        max_val: Maximum absolute value for normalization
        sc_prec: SC precision (determines stoc_len = 2^sc_prec)

    Returns:
        packed: (M, N, NUM_PACKS) int64 tensor with packed bitstreams
    """
    M, N = values.shape
    stoc_len = 2 ** sc_prec
    NUM_PACKS = stoc_len // 64
    max_rng_val = 2 ** sc_prec

    # Allocate output
    packed = torch.empty((M, N, NUM_PACKS), dtype=torch.int64, device=values.device)

    # Launch kernel
    grid = (M, N)
    bin_to_stoc_packed_kernel[grid](
        values, packed, rng_seqs,
        max_val, max_rng_val,
        M, N, stoc_len, NUM_PACKS,
    )

    return packed


def xnor_matmul(Q_packed: torch.Tensor, K_packed: torch.Tensor,
                Q_l: int, Q_e: int, K_l: int, stoc_len: int,
                max_val_squared: float) -> torch.Tensor:
    """
    Compute XNOR-based matrix multiplication using packed streams.

    Args:
        Q_packed: (Q_l, Q_e, NUM_PACKS) int64 tensor
        K_packed: (K_l, K_e, NUM_PACKS) int64 tensor
        Q_l, Q_e, K_l: Matrix dimensions
        stoc_len: Stochastic stream length
        max_val_squared: max_val^2 for decoding

    Returns:
        output: (Q_l, K_l) float32 tensor
    """
    NUM_PACKS = Q_packed.shape[2]

    # Allocate output
    output = torch.empty((Q_l, K_l), dtype=torch.float32, device=Q_packed.device)

    # Use tiled kernel for better GPU utilization
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(Q_l, BLOCK_M), triton.cdiv(K_l, BLOCK_N))
    xnor_matmul_tiled_kernel[grid](
        Q_packed, K_packed, output,
        Q_l, Q_e, K_l, NUM_PACKS, stoc_len,
        max_val_squared, BLOCK_M, BLOCK_N,
    )

    return output


def bin_to_stoc_packed_unipolar(values: torch.Tensor, rng_seqs: torch.Tensor,
                                 max_val: float, sc_prec: int) -> torch.Tensor:
    """
    Convert a matrix of non-negative values to packed stochastic bitstreams (unipolar).

    Args:
        values: (M, N) float32 tensor, values in [0, max_val]
        rng_seqs: (N, stoc_len) int32 tensor with per-element RNG sequences
        max_val: Maximum value for normalization
        sc_prec: SC precision (determines stoc_len = 2^sc_prec)

    Returns:
        packed: (M, N, NUM_PACKS) int64 tensor with packed bitstreams
    """
    M, N = values.shape
    stoc_len = 2 ** sc_prec
    NUM_PACKS = stoc_len // 64
    max_rng_val = 2 ** sc_prec

    # Allocate output
    packed = torch.empty((M, N, NUM_PACKS), dtype=torch.int64, device=values.device)

    # Launch kernel
    grid = (M, N)
    bin_to_stoc_packed_unipolar_kernel[grid](
        values, packed, rng_seqs,
        max_val, max_rng_val,
        M, N, stoc_len, NUM_PACKS,
    )

    return packed


def and_matmul(A_packed: torch.Tensor, B_packed: torch.Tensor,
               A_l: int, A_e: int, B_l: int, stoc_len: int,
               max_val_squared: float) -> torch.Tensor:
    """
    Compute AND-based matrix multiplication using packed streams (unipolar).

    Args:
        A_packed: (A_l, A_e, NUM_PACKS) int64 tensor
        B_packed: (B_l, B_e, NUM_PACKS) int64 tensor
        A_l, A_e, B_l: Matrix dimensions
        stoc_len: Stochastic stream length
        max_val_squared: max_val^2 for decoding

    Returns:
        output: (A_l, B_l) float32 tensor
    """
    NUM_PACKS = A_packed.shape[2]

    # Allocate output
    output = torch.empty((A_l, B_l), dtype=torch.float32, device=A_packed.device)

    # Use tiled kernel for better GPU utilization
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(A_l, BLOCK_M), triton.cdiv(B_l, BLOCK_N))
    and_matmul_tiled_kernel[grid](
        A_packed, B_packed, output,
        A_l, A_e, B_l, NUM_PACKS, stoc_len,
        max_val_squared, BLOCK_M, BLOCK_N,
    )

    return output


# =============================================================================
# Enable-Signal Kernels (Table-Lookup Matmul)
# =============================================================================

@triton.jit
def build_cum_indicator_kernel(
    rng_b_ptr,       # (D, stoc_len) int32 — per-dimension RNG sequences for B
    cum_ptr,         # (D, stoc_len+1, V) int16 — output cumulative indicator table
    D: tl.constexpr,
    stoc_len: tl.constexpr,
    V: tl.constexpr,          # max_rng_val + 1 = 2^sc_prec
):
    """
    Build cumulative indicator table for B's RNG sequence.

    cum[d, k, v] = |{i < k : rng_b[d, i] <= v}|

    One program per dimension d.
    """
    d = tl.program_id(0)
    if d >= D:
        return

    v_range = tl.arange(0, V)
    cum_stride_d = (stoc_len + 1) * V  # stride for d dimension

    # Initialize cum[d, 0, :] = 0
    tl.store(cum_ptr + d * cum_stride_d + v_range,
             tl.zeros([V], dtype=tl.int16))

    # Build prefix sums
    running = tl.zeros([V], dtype=tl.int16)
    for k in range(stoc_len):
        r = tl.load(rng_b_ptr + d * stoc_len + k)
        delta = tl.where(v_range > r,
                         tl.full([V], 1, dtype=tl.int16),
                         tl.zeros([V], dtype=tl.int16))
        running = running + delta
        offset = d * cum_stride_d + (k + 1) * V
        tl.store(cum_ptr + offset + v_range, running)


@triton.jit
def compute_k_table_kernel(
    rng_a_ptr,       # (D, stoc_len) int32 — per-dimension RNG sequences for A
    k_table_ptr,     # (D, V) int16 — output k-table
    D: tl.constexpr,
    stoc_len: tl.constexpr,
    V: tl.constexpr,          # max_rng_val + 1
):
    """
    Compute popcount table for A's RNG sequence.

    k_table[d, v] = |{t : rng_a[d, t] <= v}|

    One program per dimension d.
    """
    d = tl.program_id(0)
    if d >= D:
        return

    v_range = tl.arange(0, V)

    # Count how many rng_a values are <= each v
    counts = tl.zeros([V], dtype=tl.int16)
    for t in range(stoc_len):
        r = tl.load(rng_a_ptr + d * stoc_len + t)
        counts += tl.where(v_range > r,
                           tl.full([V], 1, dtype=tl.int16),
                           tl.zeros([V], dtype=tl.int16))

    tl.store(k_table_ptr + d * V + v_range, counts)


@triton.jit
def enable_matmul_bipolar_kernel(
    cum_ptr,           # (D, stoc_len+1, V) int16
    k_table_ptr,       # (D, V) int16
    boundary_a_ptr,    # (N, D) int16
    boundary_b_ptr,    # (M, D) int16
    sign_a_ptr,        # (N, D) int8
    sign_b_ptr,        # (M, D) int8
    output_ptr,        # (N, M) float32
    N: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    stoc_len: tl.constexpr,
    V: tl.constexpr,
    q_max_sq,          # float: q_max^2 for decoding
):
    """
    Enable-signal matmul kernel (bipolar sign-magnitude).

    Each program computes output[n, m] = sum_d decode(enable_count[n,m,d]) * sign_a * sign_b
    """
    n = tl.program_id(0)
    m = tl.program_id(1)
    if n >= N or m >= M:
        return

    cum_stride_d = (stoc_len + 1) * V
    dot_product = 0.0

    for d in tl.static_range(D):
        ba = tl.load(boundary_a_ptr + n * D + d).to(tl.int32)
        bb = tl.load(boundary_b_ptr + m * D + d).to(tl.int32)

        # k = k_table[d, ba]
        k = tl.load(k_table_ptr + d * V + ba).to(tl.int32)

        # count = cum_indicator[d, k, bb]
        count = tl.load(cum_ptr + d * cum_stride_d + k * V + bb).to(tl.float32)

        # Decode and apply signs
        decoded = count * q_max_sq / stoc_len
        sa = tl.load(sign_a_ptr + n * D + d).to(tl.float32)
        sb = tl.load(sign_b_ptr + m * D + d).to(tl.float32)
        dot_product += decoded * sa * sb

    tl.store(output_ptr + n * M + m, dot_product)


@triton.jit
def enable_matmul_unipolar_kernel(
    cum_ptr,           # (D, stoc_len+1, V) int16
    k_table_ptr,       # (D, V) int16
    boundary_a_ptr,    # (N, D) int32
    boundary_b_ptr,    # (M, D) int32
    output_ptr,        # (N, M) float32
    N: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    stoc_len: tl.constexpr,
    V: tl.constexpr,
    q_max_sq,          # float: q_max^2 for decoding
):
    """
    Enable-signal matmul kernel (unipolar).

    Each program computes output[n, m] = sum_d decode(enable_count[n,m,d])
    """
    n = tl.program_id(0)
    m = tl.program_id(1)
    if n >= N or m >= M:
        return

    cum_stride_d = (stoc_len + 1) * V
    dot_product = 0.0

    for d in tl.static_range(D):
        ba = tl.load(boundary_a_ptr + n * D + d).to(tl.int32)
        bb = tl.load(boundary_b_ptr + m * D + d).to(tl.int32)

        k = tl.load(k_table_ptr + d * V + ba).to(tl.int32)
        count = tl.load(cum_ptr + d * cum_stride_d + k * V + bb).to(tl.float32)

        dot_product += count * q_max_sq / stoc_len

    tl.store(output_ptr + n * M + m, dot_product)


# =============================================================================
# Enable-Signal Tiled Kernels
# =============================================================================

@triton.jit
def enable_matmul_bipolar_tiled_kernel(
    cum_ptr,           # (D, stoc_len+1, V) int16
    k_table_ptr,       # (D, V) int16
    boundary_a_ptr,    # (D, N) int16 — transposed for coalesced access
    boundary_b_ptr,    # (D, M) int16 — transposed for coalesced access
    sign_a_ptr,        # (D, N) int8 — transposed for coalesced access
    sign_b_ptr,        # (D, M) int8 — transposed for coalesced access
    output_ptr,        # (N, M) float32
    N, M, D,
    stoc_len: tl.constexpr,
    V: tl.constexpr,
    q_max_sq,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Tiled enable-signal matmul (bipolar sign-magnitude).

    BLOCK_K tiles the D dimension with static_range for compiler unrolling,
    allowing better instruction scheduling and memory access pipelining.
    Boundary/sign tensors use (D, N) layout for coalesced thread access.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < N
    n_mask = n_offsets < M
    gather_mask = m_mask[:, None] & n_mask[None, :]

    cum_stride_d = (stoc_len + 1) * V
    scale = q_max_sq / stoc_len
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    num_k_blocks = (D + BLOCK_K - 1) // BLOCK_K
    for k_block in range(num_k_blocks):
        k_start = k_block * BLOCK_K
        for ki in tl.static_range(BLOCK_K):
            d = k_start + ki
            if d < D:
                # Load signs as int8, cast for zero-check and arithmetic
                sa_i8 = tl.load(sign_a_ptr + d * N + m_offsets, mask=m_mask, other=0)
                # Skip if all sign_a are zero — no contribution to acc
                if tl.sum(tl.abs(sa_i8).to(tl.int32)) > 0:
                    sb_i8 = tl.load(sign_b_ptr + d * M + n_offsets, mask=n_mask, other=0)
                    # Skip if all sign_b are zero
                    if tl.sum(tl.abs(sb_i8).to(tl.int32)) > 0:
                        sa = sa_i8.to(tl.float32)
                        sb = sb_i8.to(tl.float32)
                        # Load boundaries (coalesced — (D,N) layout)
                        ba = tl.load(boundary_a_ptr + d * N + m_offsets, mask=m_mask, other=0).to(tl.int32)
                        bb = tl.load(boundary_b_ptr + d * M + n_offsets, mask=n_mask, other=0).to(tl.int32)

                        # k_table lookup: k[d, ba] -> [BLOCK_M]
                        k_vals = tl.load(k_table_ptr + d * V + ba, mask=m_mask, other=0).to(tl.int32)

                        # cum_indicator gather: cum[d, k_vals[i], bb[j]] -> [BLOCK_M, BLOCK_N]
                        cum_offsets = (d * cum_stride_d
                                       + k_vals[:, None].to(tl.int64) * V
                                       + bb[None, :].to(tl.int64))
                        counts = tl.load(cum_ptr + cum_offsets, mask=gather_mask, other=0).to(tl.float32)

                        acc += counts * sa[:, None] * sb[None, :]

    # Apply loop-invariant scale once (enables FMA fusion in the inner loop)
    acc *= scale

    out_offsets = m_offsets[:, None] * M + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offsets, acc, mask=out_mask)


@triton.jit
def enable_matmul_unipolar_tiled_kernel(
    cum_ptr,           # (D, stoc_len+1, V) int16
    k_table_ptr,       # (D, V) int16
    boundary_a_ptr,    # (D, N) int16 — transposed for coalesced access
    boundary_b_ptr,    # (D, M) int16 — transposed for coalesced access
    output_ptr,        # (N, M) float32
    N, M, D,
    stoc_len: tl.constexpr,
    V: tl.constexpr,
    q_max_sq,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Tiled enable-signal matmul (unipolar).

    BLOCK_K tiles the D dimension with static_range for compiler unrolling.
    Boundary tensors use (D, N) layout for coalesced thread access.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < N
    n_mask = n_offsets < M
    gather_mask = m_mask[:, None] & n_mask[None, :]

    cum_stride_d = (stoc_len + 1) * V
    scale = q_max_sq / stoc_len
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    num_k_blocks = (D + BLOCK_K - 1) // BLOCK_K
    for k_block in range(num_k_blocks):
        k_start = k_block * BLOCK_K
        for ki in tl.static_range(BLOCK_K):
            d = k_start + ki
            if d < D:
                ba = tl.load(boundary_a_ptr + d * N + m_offsets, mask=m_mask, other=0).to(tl.int32)
                bb = tl.load(boundary_b_ptr + d * M + n_offsets, mask=n_mask, other=0).to(tl.int32)

                k_vals = tl.load(k_table_ptr + d * V + ba, mask=m_mask, other=0).to(tl.int32)

                cum_offsets = (d * cum_stride_d
                               + k_vals[:, None].to(tl.int64) * V
                               + bb[None, :].to(tl.int64))
                counts = tl.load(cum_ptr + cum_offsets, mask=gather_mask, other=0).to(tl.float32)

                acc += counts

    # Apply loop-invariant scale once
    acc *= scale

    out_offsets = m_offsets[:, None] * M + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offsets, acc, mask=out_mask)


# =============================================================================
# Enable-Signal Compact Kernels (no cum_indicator table, O(D*V) memory)
# =============================================================================

@triton.jit
def enable_matmul_compact_bipolar_kernel(
    rng_b_ptr,         # (D, stoc_len) int32
    k_table_ptr,       # (D, V) int16
    boundary_a_ptr,    # (N, D) int16
    boundary_b_ptr,    # (M, D) int16
    sign_a_ptr,        # (N, D) int8
    sign_b_ptr,        # (M, D) int8
    output_ptr,        # (N, M) float32
    N, M, D,
    stoc_len: tl.constexpr,
    V: tl.constexpr,
    q_max_sq,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BATCH_T: tl.constexpr,
):
    """
    Compact enable-signal matmul (bipolar). Computes cum_indicator on-the-fly.

    For each (n, m, d): count = |{t < k : rng_b[d, t] <= bb}|
    where k = k_table[d, ba], ba = boundary_a[n, d], bb = boundary_b[m, d].

    BATCH_T vectorizes the inner t-loop: loads BATCH_T RNG values at once and
    compares them against bb in a vectorized manner, reducing loop iterations
    from stoc_len to stoc_len/BATCH_T.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < N
    n_mask = n_offsets < M

    scale = q_max_sq / stoc_len
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for d in range(D):
        # Load boundaries
        ba = tl.load(boundary_a_ptr + m_offsets * D + d, mask=m_mask, other=0).to(tl.int32)
        bb = tl.load(boundary_b_ptr + n_offsets * D + d, mask=n_mask, other=0).to(tl.int32)

        # k = k_table[d, ba] -> [BLOCK_M]
        k_vals = tl.load(k_table_ptr + d * V + ba, mask=m_mask, other=0).to(tl.int32)

        # Batch-vectorized inner loop: load BATCH_T RNG values at once
        counts = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
        num_batches = stoc_len // BATCH_T
        for tb in range(num_batches):
            t_base = tb * BATCH_T
            t_indices = t_base + tl.arange(0, BATCH_T)  # [BATCH_T]

            # Load BATCH_T RNG values: rng_b[d, t_base:t_base+BATCH_T]
            rng_vals = tl.load(rng_b_ptr + d * stoc_len + t_indices)  # [BATCH_T]

            # t_mask: which t indices are < k_vals[m]? -> [BLOCK_M, BATCH_T]
            t_ok = t_indices[None, :] < k_vals[:, None]  # [BLOCK_M, BATCH_T]

            # r_mask: which rng vals are <= bb[n]? -> [BATCH_T, BLOCK_N]
            r_ok = bb[None, :] > rng_vals[:, None]  # [BATCH_T, BLOCK_N]

            # For each (m, n): sum over BATCH_T where both conditions hold
            # t_ok: [BLOCK_M, BATCH_T], r_ok: [BATCH_T, BLOCK_N]
            # We need: counts[m, n] += sum_t(t_ok[m, t] & r_ok[t, n])
            # Compute by iterating over BATCH_T (static_range for unrolling)
            for ti in tl.static_range(BATCH_T):
                t_m = t_ok[:, ti]   # [BLOCK_M] bool
                r_n = r_ok[ti, :]   # [BLOCK_N] bool
                counts += (t_m[:, None] & r_n[None, :]).to(tl.int32)

        # Signs (int8 → float32)
        sa = tl.load(sign_a_ptr + m_offsets * D + d, mask=m_mask, other=0).to(tl.float32)
        sb = tl.load(sign_b_ptr + n_offsets * D + d, mask=n_mask, other=0).to(tl.float32)

        acc += counts.to(tl.float32) * scale * sa[:, None] * sb[None, :]

    out_offsets = m_offsets[:, None] * M + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offsets, acc, mask=out_mask)


@triton.jit
def enable_matmul_compact_unipolar_kernel(
    rng_b_ptr,         # (D, stoc_len) int32
    k_table_ptr,       # (D, V) int16
    boundary_a_ptr,    # (N, D) int32
    boundary_b_ptr,    # (M, D) int32
    output_ptr,        # (N, M) float32
    N, M, D,
    stoc_len: tl.constexpr,
    V: tl.constexpr,
    q_max_sq,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BATCH_T: tl.constexpr,
):
    """
    Compact enable-signal matmul (unipolar). No cum_indicator table needed.

    BATCH_T vectorizes the inner t-loop for reduced iteration count.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < N
    n_mask = n_offsets < M

    scale = q_max_sq / stoc_len
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for d in range(D):
        ba = tl.load(boundary_a_ptr + m_offsets * D + d, mask=m_mask, other=0).to(tl.int32)
        bb = tl.load(boundary_b_ptr + n_offsets * D + d, mask=n_mask, other=0).to(tl.int32)
        k_vals = tl.load(k_table_ptr + d * V + ba, mask=m_mask, other=0).to(tl.int32)

        counts = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
        num_batches = stoc_len // BATCH_T
        for tb in range(num_batches):
            t_base = tb * BATCH_T
            t_indices = t_base + tl.arange(0, BATCH_T)

            rng_vals = tl.load(rng_b_ptr + d * stoc_len + t_indices)

            t_ok = t_indices[None, :] < k_vals[:, None]
            r_ok = bb[None, :] > rng_vals[:, None]

            for ti in tl.static_range(BATCH_T):
                t_m = t_ok[:, ti]
                r_n = r_ok[ti, :]
                counts += (t_m[:, None] & r_n[None, :]).to(tl.int32)

        acc += counts.to(tl.float32) * scale

    out_offsets = m_offsets[:, None] * M + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offsets, acc, mask=out_mask)


# =============================================================================
# Enable-Signal Compact Kernels for MLP
# Two versions:
#   - "dot" kernel: tl.dot vectorized, single program per (m,n) tile.
#     Used by enable_matmul_compact() for attention (small D).
#   - "splitd" kernel: tl.dot + split-D parallelism + transposed inputs.
#     Used by enable_matmul_compact_mlp() for MLP (large D).
#     Inputs are (D,N)/(D,M) layout for coalesced memory access.
#     Grid z-axis splits D into BLOCK_D chunks with atomic accumulation.
# =============================================================================

@triton.jit
def enable_matmul_compact_bipolar_dot_kernel(
    rng_b_ptr,         # (D, stoc_len) int32
    k_table_ptr,       # (D, V) int16
    boundary_a_ptr,    # (N, D) int16
    boundary_b_ptr,    # (M, D) int16
    sign_a_ptr,        # (N, D) int8
    sign_b_ptr,        # (M, D) int8
    output_ptr,        # (N, M) float32
    N, M, D,
    stoc_len: tl.constexpr,
    V: tl.constexpr,
    q_max_sq,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BATCH_T: tl.constexpr,
):
    """
    Compact enable-signal matmul (bipolar). tl.dot vectorized, no split-D.
    For small D (attention). Inputs in (N, D) / (M, D) row-major layout.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < N
    n_mask = n_offsets < M

    scale = q_max_sq / stoc_len
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    num_batches: tl.constexpr = stoc_len // BATCH_T

    for d in range(D):
        ba = tl.load(boundary_a_ptr + m_offsets * D + d, mask=m_mask, other=0).to(tl.int32)
        bb = tl.load(boundary_b_ptr + n_offsets * D + d, mask=n_mask, other=0).to(tl.int32)
        k_vals = tl.load(k_table_ptr + d * V + ba, mask=m_mask, other=0).to(tl.int32)

        sa = tl.load(sign_a_ptr + m_offsets * D + d, mask=m_mask, other=0).to(tl.float32)
        sb = tl.load(sign_b_ptr + n_offsets * D + d, mask=n_mask, other=0).to(tl.float32)

        counts = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
        for tb in range(num_batches):
            t_base = tb * BATCH_T
            t_indices = t_base + tl.arange(0, BATCH_T)
            rng_vals = tl.load(rng_b_ptr + d * stoc_len + t_indices)
            t_ok = (t_indices[None, :] < k_vals[:, None]).to(tl.int8)
            r_ok = (bb[None, :] > rng_vals[:, None]).to(tl.int8)
            counts += tl.dot(t_ok, r_ok, out_dtype=tl.int32)

        acc += counts.to(tl.float32) * scale * sa[:, None] * sb[None, :]

    out_offsets = m_offsets[:, None] * M + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offsets, acc, mask=out_mask)


@triton.jit
def enable_matmul_compact_unipolar_dot_kernel(
    rng_b_ptr,         # (D, stoc_len) int32
    k_table_ptr,       # (D, V) int16
    boundary_a_ptr,    # (N, D) int32
    boundary_b_ptr,    # (M, D) int32
    output_ptr,        # (N, M) float32
    N, M, D,
    stoc_len: tl.constexpr,
    V: tl.constexpr,
    q_max_sq,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BATCH_T: tl.constexpr,
):
    """
    Compact enable-signal matmul (unipolar). tl.dot vectorized, no split-D.
    For small D (attention). Inputs in (N, D) / (M, D) row-major layout.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < N
    n_mask = n_offsets < M

    scale = q_max_sq / stoc_len
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    num_batches: tl.constexpr = stoc_len // BATCH_T

    for d in range(D):
        ba = tl.load(boundary_a_ptr + m_offsets * D + d, mask=m_mask, other=0).to(tl.int32)
        bb = tl.load(boundary_b_ptr + n_offsets * D + d, mask=n_mask, other=0).to(tl.int32)
        k_vals = tl.load(k_table_ptr + d * V + ba, mask=m_mask, other=0).to(tl.int32)

        counts = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
        for tb in range(num_batches):
            t_base = tb * BATCH_T
            t_indices = t_base + tl.arange(0, BATCH_T)
            rng_vals = tl.load(rng_b_ptr + d * stoc_len + t_indices)
            t_ok = (t_indices[None, :] < k_vals[:, None]).to(tl.int8)
            r_ok = (bb[None, :] > rng_vals[:, None]).to(tl.int8)
            counts += tl.dot(t_ok, r_ok, out_dtype=tl.int32)

        acc += counts.to(tl.float32) * scale

    out_offsets = m_offsets[:, None] * M + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offsets, acc, mask=out_mask)


# --- Split-D kernels for MLP (transposed inputs, atomic accumulation) ---

@triton.jit
def enable_matmul_compact_bipolar_mlp_kernel(
    rng_b_ptr,         # (D, stoc_len) int32
    k_table_ptr,       # (D, V) int16
    boundary_a_ptr,    # (D, N) int16  -- TRANSPOSED for coalesced access
    boundary_b_ptr,    # (D, M) int16  -- TRANSPOSED
    sign_a_ptr,        # (D, N) int8 -- TRANSPOSED
    sign_b_ptr,        # (D, M) int8 -- TRANSPOSED
    output_ptr,        # (N, M) float32
    N, M, D,
    stoc_len: tl.constexpr,
    V: tl.constexpr,
    q_max_sq,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BATCH_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compact enable-signal matmul for MLP (bipolar). Split-D + tl.dot.

    - Inputs transposed to (D, N)/(D, M) for coalesced memory access.
    - Grid z-axis splits D into BLOCK_D chunks for parallelism.
    - Uses tl.atomic_add for cross-chunk accumulation.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_d = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < N
    n_mask = n_offsets < M

    scale = q_max_sq / stoc_len
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    d_start = pid_d * BLOCK_D
    num_batches: tl.constexpr = stoc_len // BATCH_T

    for d_off in range(BLOCK_D):
        d = d_start + d_off
        if d < D:
            # Coalesced loads: (D, N) layout, load row d, columns m_offsets
            ba = tl.load(boundary_a_ptr + d * N + m_offsets, mask=m_mask, other=0).to(tl.int32)
            bb = tl.load(boundary_b_ptr + d * M + n_offsets, mask=n_mask, other=0).to(tl.int32)
            k_vals = tl.load(k_table_ptr + d * V + ba, mask=m_mask, other=0).to(tl.int32)

            sa = tl.load(sign_a_ptr + d * N + m_offsets, mask=m_mask, other=0).to(tl.float32)
            sb = tl.load(sign_b_ptr + d * M + n_offsets, mask=n_mask, other=0).to(tl.float32)

            counts = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
            for tb in range(num_batches):
                t_base = tb * BATCH_T
                t_indices = t_base + tl.arange(0, BATCH_T)
                rng_vals = tl.load(rng_b_ptr + d * stoc_len + t_indices)
                t_ok = (t_indices[None, :] < k_vals[:, None]).to(tl.int8)
                r_ok = (bb[None, :] > rng_vals[:, None]).to(tl.int8)
                counts += tl.dot(t_ok, r_ok, out_dtype=tl.int32)

            acc += counts.to(tl.float32) * scale * sa[:, None] * sb[None, :]

    out_offsets = m_offsets[:, None] * M + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.atomic_add(output_ptr + out_offsets, acc, mask=out_mask)


@triton.jit
def enable_matmul_compact_unipolar_mlp_kernel(
    rng_b_ptr,         # (D, stoc_len) int32
    k_table_ptr,       # (D, V) int16
    boundary_a_ptr,    # (D, N) int32  -- TRANSPOSED
    boundary_b_ptr,    # (D, M) int32  -- TRANSPOSED
    output_ptr,        # (N, M) float32
    N, M, D,
    stoc_len: tl.constexpr,
    V: tl.constexpr,
    q_max_sq,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BATCH_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compact enable-signal matmul for MLP (unipolar). Split-D + tl.dot.
    Transposed inputs for coalesced access, atomic accumulation.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_d = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < N
    n_mask = n_offsets < M

    scale = q_max_sq / stoc_len
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    d_start = pid_d * BLOCK_D
    num_batches: tl.constexpr = stoc_len // BATCH_T

    for d_off in range(BLOCK_D):
        d = d_start + d_off
        if d < D:
            ba = tl.load(boundary_a_ptr + d * N + m_offsets, mask=m_mask, other=0).to(tl.int32)
            bb = tl.load(boundary_b_ptr + d * M + n_offsets, mask=n_mask, other=0).to(tl.int32)
            k_vals = tl.load(k_table_ptr + d * V + ba, mask=m_mask, other=0).to(tl.int32)

            counts = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
            for tb in range(num_batches):
                t_base = tb * BATCH_T
                t_indices = t_base + tl.arange(0, BATCH_T)
                rng_vals = tl.load(rng_b_ptr + d * stoc_len + t_indices)
                t_ok = (t_indices[None, :] < k_vals[:, None]).to(tl.int8)
                r_ok = (bb[None, :] > rng_vals[:, None]).to(tl.int8)
                counts += tl.dot(t_ok, r_ok, out_dtype=tl.int32)

            acc += counts.to(tl.float32) * scale

    out_offsets = m_offsets[:, None] * M + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.atomic_add(output_ptr + out_offsets, acc, mask=out_mask)


# =============================================================================
# Fused Quantization Kernels
# Replaces scattered PyTorch elementwise ops (div, round, clamp, sign, abs,
# boundary computation) with a single kernel pass per operand.
# =============================================================================

@triton.jit
def fused_quant_bipolar_kernel(
    fp_ptr,            # (rows, cols) float32 input
    boundary_ptr,      # (rows, cols) int16 output
    sign_ptr,          # (rows, cols) int8 output
    inv_scale,         # float: 1.0 / scale = q_norm / abs_max
    q_clip,            # int: clamp upper bound (e.g. 125 for 8-bit)
    q_clip_min,        # int: clamp lower bound (e.g. -125 for 8-bit)
    q_norm,            # int: normalization q_max for boundary (e.g. 127)
    max_rng_val,       # int: 2^sc_prec - 1
    rows, cols,
    BLOCK: tl.constexpr,
):
    """
    Fused bipolar quantization: FP -> (boundary, sign) in one kernel.

    For each element x:
      x_int = round(clamp(x * inv_scale, q_clip_min, q_clip))
      sign = int8(sign(x_int))  (-1, 0, or 1)
      boundary = int16(abs(x_int) * max_rng_val / q_norm)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    total = rows * cols
    mask = offsets < total

    x = tl.load(fp_ptr + offsets, mask=mask, other=0.0)

    # Quantize: x_int = round(clamp(x * inv_scale, q_clip_min, q_clip))
    x_scaled = x * inv_scale
    # round
    x_rounded = libdevice.nearbyint(x_scaled)
    # clamp to [-q_clip, q_clip] (narrower than full range to avoid degenerate SC probs)
    x_clamped = tl.minimum(tl.maximum(x_rounded, q_clip_min.to(tl.float32)), q_clip.to(tl.float32))

    # Sign: -1, 0, or +1 as int8
    sign_val = tl.where(x_clamped > 0.0, tl.full(x_clamped.shape, 1, dtype=tl.int8),
                        tl.where(x_clamped < 0.0, tl.full(x_clamped.shape, -1, dtype=tl.int8),
                                 tl.full(x_clamped.shape, 0, dtype=tl.int8)))

    # Boundary: abs(x_int) * max_rng_val / q_norm (q_norm=127, not q_clip)
    mag = tl.abs(x_clamped)
    boundary = libdevice.nearbyint(mag * (max_rng_val / q_norm)).to(tl.int16)

    tl.store(boundary_ptr + offsets, boundary, mask=mask)
    tl.store(sign_ptr + offsets, sign_val, mask=mask)


@triton.jit
def fused_quant_bipolar_perrow_kernel(
    fp_ptr,            # (rows, cols) float32 input
    boundary_ptr,      # (rows, cols) int16 output
    sign_ptr,          # (rows, cols) int8 output
    scale_ptr,         # (rows,) float32 output — per-row scale for dequant
    q_clip,            # int: clamp bound (e.g. 125)
    q_norm,            # int: normalization for boundary (e.g. 127)
    max_rng_val,       # int: 2^sc_prec - 1
    rows, cols,
    COLS_PAD: tl.constexpr,
):
    """
    Fused per-row bipolar quantization: FP -> (boundary, sign, scale) in one
    kernel launch. One program per row.

    Replaces _grouped_symmetric_quant + boundary computation (~20+ PyTorch ops)
    with a single Triton kernel.
    """
    row = tl.program_id(0)
    if row >= rows:
        return

    col_offsets = tl.arange(0, COLS_PAD)
    mask = col_offsets < cols

    # Load entire row
    x = tl.load(fp_ptr + row * cols + col_offsets, mask=mask, other=0.0)

    # Per-row abs max → scale
    abs_max = tl.max(tl.abs(x))
    abs_max = tl.maximum(abs_max, 1e-5)
    scale = abs_max / q_clip
    inv_scale = 1.0 / scale
    tl.store(scale_ptr + row, scale)

    # Quantize: round(clamp(x / scale, -q_clip, q_clip))
    x_scaled = x * inv_scale
    x_rounded = libdevice.nearbyint(x_scaled)
    x_clamped = tl.minimum(tl.maximum(x_rounded, (-q_clip).to(tl.float32)), q_clip.to(tl.float32))

    # Sign: -1, 0, +1 as int8
    sign_val = tl.where(x_clamped > 0.0, tl.full(x_clamped.shape, 1, dtype=tl.int8),
                        tl.where(x_clamped < 0.0, tl.full(x_clamped.shape, -1, dtype=tl.int8),
                                 tl.full(x_clamped.shape, 0, dtype=tl.int8)))

    # Boundary: round(|x_int| * max_rng_val / q_norm)
    mag = tl.abs(x_clamped)
    boundary = libdevice.nearbyint(mag * (max_rng_val / q_norm)).to(tl.int16)

    tl.store(boundary_ptr + row * cols + col_offsets, boundary, mask=mask)
    tl.store(sign_ptr + row * cols + col_offsets, sign_val, mask=mask)


def fused_quantize_bipolar_perrow(
    fp_tensor: torch.Tensor,
    sc_prec: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused per-row bipolar quantization in one kernel launch.

    Matches _grouped_symmetric_quant(x, G=1, q_max, clip_margin=0) followed by
    boundary = (|x_int| * max_rng_val / q_max).round().short().

    Returns:
        boundary: (rows, cols) int16
        sign: (rows, cols) int8
        scale_row: (rows,) float32 — per-row dequantization scale
    """
    rows, cols = fp_tensor.shape
    q_max = 2 ** (sc_prec - 1) - 1      # 127: q_clip = q_max (clip_margin=0)
    max_rng_val = 2 * (q_max + 1)       # 256 = stoc_len (matches MLP path)

    boundary = torch.empty(rows, cols, dtype=torch.int16, device=fp_tensor.device)
    sign = torch.empty(rows, cols, dtype=torch.int8, device=fp_tensor.device)
    scale_row = torch.empty(rows, dtype=torch.float32, device=fp_tensor.device)

    COLS_PAD = triton.next_power_of_2(cols)
    fused_quant_bipolar_perrow_kernel[(rows,)](
        fp_tensor, boundary, sign, scale_row,
        q_max, q_max, max_rng_val,
        rows, cols, COLS_PAD,
    )
    return boundary, sign, scale_row


@triton.jit
def fused_quant_unipolar_kernel(
    fp_ptr,            # (rows, cols) float32 input
    boundary_ptr,      # (rows, cols) int32 output
    inv_scale,         # float: 1.0 / scale
    zp,                # float: zero-point
    q_max,             # int: 2^sc_prec - 1
    max_rng_val,       # int: 2^sc_prec - 1
    rows, cols,
    BLOCK: tl.constexpr,
):
    """
    Fused unipolar quantization: FP -> boundary in one kernel.

    For each element x:
      x_int = round(clamp(x * inv_scale + zp, 0, q_max))
      boundary = int(x_int * max_rng_val / q_max)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    total = rows * cols
    mask = offsets < total

    x = tl.load(fp_ptr + offsets, mask=mask, other=0.0)

    # Quantize
    x_scaled = x * inv_scale + zp
    x_rounded = libdevice.nearbyint(x_scaled)
    x_clamped = tl.minimum(tl.maximum(x_rounded, 0.0), q_max.to(tl.float32))

    # Boundary
    boundary = libdevice.nearbyint(x_clamped * (max_rng_val / q_max)).to(tl.int32)

    tl.store(boundary_ptr + offsets, boundary, mask=mask)


@triton.jit
def fused_quant_unipolar_with_sum_kernel(
    fp_ptr,            # (rows, cols) float32 input
    boundary_ptr,      # (rows, cols) int32 output
    row_sum_ptr,       # (rows,) float32 output — sum of x_int per row
    inv_scale,         # float: 1.0 / scale
    zp,                # float: zero-point
    q_max,             # int: 2^sc_prec - 1
    max_rng_val,       # int: 2^sc_prec - 1
    rows, cols,
    COLS_BLOCK: tl.constexpr,
):
    """
    Fused unipolar quantization + per-row sum in one kernel.

    Each program handles one row: quantize all cols, compute boundary,
    and reduce the row's x_int sum for zero-point correction.
    """
    row = tl.program_id(0)
    if row >= rows:
        return

    col_offsets = tl.arange(0, COLS_BLOCK)
    col_mask = col_offsets < cols
    base = row * cols

    x = tl.load(fp_ptr + base + col_offsets, mask=col_mask, other=0.0)

    # Quantize
    x_scaled = x * inv_scale + zp
    x_rounded = libdevice.nearbyint(x_scaled)
    x_clamped = tl.minimum(tl.maximum(x_rounded, 0.0), q_max.to(tl.float32))

    # Boundary
    boundary = libdevice.nearbyint(x_clamped * (max_rng_val / q_max)).to(tl.int32)
    tl.store(boundary_ptr + base + col_offsets, boundary, mask=col_mask)

    # Row sum of x_int (for zero-point correction)
    row_sum = tl.sum(x_clamped, axis=0)
    tl.store(row_sum_ptr + row, row_sum)


def fused_quantize_bipolar(
    fp_tensor: torch.Tensor,
    abs_max: float,
    sc_prec: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Fused bipolar quantization: FP -> (boundary, sign) in one kernel launch.

    Returns:
        boundary: (rows, cols) int16
        sign: (rows, cols) int8
        scale: float (for dequantization)
    """
    rows, cols = fp_tensor.shape
    q_norm = 2 ** (sc_prec - 1) - 1     # 127: for SNG boundary normalization
    q_clip = q_norm - 2                  # 125: quantization range & clamp
    max_rng_val = 2 ** sc_prec - 1
    abs_max = max(abs_max, 1e-5)
    scale = abs_max / q_clip
    inv_scale = 1.0 / scale

    boundary = torch.empty(rows, cols, dtype=torch.int16, device=fp_tensor.device)
    sign = torch.empty(rows, cols, dtype=torch.int8, device=fp_tensor.device)

    total = rows * cols
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    fused_quant_bipolar_kernel[grid](
        fp_tensor, boundary, sign,
        inv_scale, q_clip, -q_clip, q_clip, max_rng_val,
        rows, cols, BLOCK,
    )
    return boundary, sign, scale


def fused_quantize_unipolar(
    fp_tensor: torch.Tensor,
    fp_max: float,
    fp_min: float,
    sc_prec: int,
    compute_sum: bool = False,
) -> tuple[torch.Tensor, float, float, float, torch.Tensor | None]:
    """
    Fused unipolar quantization: FP -> boundary in one kernel launch.

    Returns:
        boundary: (rows, cols) int32
        scale: float
        zp: float (zero-point)
        row_sum: (rows,) float32 if compute_sum else None
    """
    rows, cols = fp_tensor.shape
    q_max = 2 ** sc_prec - 1
    max_rng_val = 2 ** sc_prec - 1
    q_lo, q_hi = 2, q_max - 2  # map min->2, max->253 for 8-bit
    range_fp = max(fp_max - fp_min, 1e-5)
    scale = range_fp / (q_hi - q_lo)
    inv_scale = 1.0 / scale
    zp = round(-fp_min / scale) + q_lo
    zp = max(q_lo, min(q_hi, zp))
    zp_f = float(zp)

    boundary = torch.empty(rows, cols, dtype=torch.int32, device=fp_tensor.device)

    if compute_sum:
        # Use per-row kernel that also computes row sums
        row_sum = torch.empty(rows, dtype=torch.float32, device=fp_tensor.device)
        # Round cols up to power of 2 for tl.arange
        COLS_BLOCK = triton.next_power_of_2(cols)
        grid = (rows,)
        fused_quant_unipolar_with_sum_kernel[grid](
            fp_tensor, boundary, row_sum,
            inv_scale, zp_f, q_max, max_rng_val,
            rows, cols, COLS_BLOCK,
        )
        return boundary, scale, zp_f, row_sum
    else:
        total = rows * cols
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)
        fused_quant_unipolar_kernel[grid](
            fp_tensor, boundary,
            inv_scale, zp_f, q_max, max_rng_val,
            rows, cols, BLOCK,
        )
        return boundary, scale, zp_f, None


# Threshold: use compact path when cum_indicator would exceed this many bytes.
# On Blackwell (RTX PRO 6000 measured) the table kernel's gather beats the
# compact kernel's tl.dot inner loop — table ~25 s/it vs compact ~28 s/it e2e.
# On older cards with smaller L2 (e.g. 4080), compact was ~6% faster because
# the 18 MB cum_indicator polluted L2 (commit d60e442). Default now prefers
# table; set SC_FORCE_COMPACT=1 to force compact (recover 4080-era behaviour).
_COMPACT_ENABLE_THRESHOLD_BYTES = 1 << 40  # default: always table
if os.environ.get("SC_FORCE_COMPACT", "0") == "1":
    _COMPACT_ENABLE_THRESHOLD_BYTES = 0
if os.environ.get("SC_FORCE_TABLE", "0") == "1":
    _COMPACT_ENABLE_THRESHOLD_BYTES = 1 << 40  # no-op: already default

# =============================================================================
# Batched Kernels — one launch for all B*H heads
# =============================================================================

@triton.jit
def fused_quant_bipolar_batched_kernel(
    fp_ptr,            # (BH, rows, cols) float32 input
    boundary_ptr,      # (BH, cols, rows) int16 output — transposed layout
    sign_ptr,          # (BH, cols, rows) int8 output — transposed layout
    inv_scale_ptr,     # (BH,) float32 — per-head inv_scale
    q_max,             # int: 2^(sc_prec-1) - 1
    q_min,             # int: -(2^(sc_prec-1))
    max_rng_val,       # int: 2^sc_prec - 1
    slice_size,        # int: rows * cols (elements per head)
    rows,              # int: number of rows (N for q, M for k)
    cols,              # int: number of cols (D)
    BLOCK: tl.constexpr,
):
    """Batched bipolar quantization with fused transpose.

    Reads from (BH, rows, cols) and writes to (BH, cols, rows) layout,
    eliminating the 4 separate .transpose().contiguous() calls."""
    batch_id = tl.program_id(1)
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < slice_size

    # Load from (BH, rows, cols) layout — linear access
    base_in = batch_id * slice_size
    x = tl.load(fp_ptr + base_in + offsets, mask=mask, other=0.0)
    inv_scale = tl.load(inv_scale_ptr + batch_id)

    x_scaled = x * inv_scale
    x_rounded = libdevice.nearbyint(x_scaled)
    x_clamped = tl.minimum(tl.maximum(x_rounded, q_min.to(tl.float32)), q_max.to(tl.float32))

    sign_val = tl.where(x_clamped > 0.0, tl.full(x_clamped.shape, 1, dtype=tl.int8),
                        tl.where(x_clamped < 0.0, tl.full(x_clamped.shape, -1, dtype=tl.int8),
                                 tl.full(x_clamped.shape, 0, dtype=tl.int8)))
    mag = tl.abs(x_clamped)
    boundary = libdevice.nearbyint(mag * (max_rng_val / q_max)).to(tl.int16)

    # Compute transposed store offsets: (row, col) → (col, row)
    # linear offset → (row_idx, col_idx) → store at col_idx * rows + row_idx
    row_idx = offsets // cols
    col_idx = offsets % cols
    base_out = batch_id * slice_size
    store_offsets = base_out + col_idx * rows + row_idx

    tl.store(boundary_ptr + store_offsets, boundary, mask=mask)
    tl.store(sign_ptr + store_offsets, sign_val, mask=mask)


@triton.jit
def enable_matmul_bipolar_batched_kernel(
    cum_ptr,           # (D, stoc_len+1, V) int16 — shared across heads
    k_table_ptr,       # (D, V) int16 — shared across heads
    boundary_a_ptr,    # (BH, D, N) int16 — transposed for coalesced access
    boundary_b_ptr,    # (BH, D, M) int16 — transposed for coalesced access
    sign_a_ptr,        # (BH, D, N) int8 — transposed for coalesced access
    sign_b_ptr,        # (BH, D, M) int8 — transposed for coalesced access
    output_ptr,        # (BH, N, M) float32
    scale_ptr,         # (BH,) float32 — per-head (scale_a * scale_b)
    N, M, D,
    stoc_len: tl.constexpr,
    V: tl.constexpr,
    q_max_sq,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Batched tiled enable-signal matmul (bipolar). One launch for all heads.
    Boundary/sign tensors use (BH, D, N/M) layout for coalesced thread access."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    batch_id = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < N
    n_mask = n_offsets < M
    gather_mask = m_mask[:, None] & n_mask[None, :]

    # Per-head base offsets (D*N = N*D, same total stride)
    ba_base = batch_id * D * N
    bb_base = batch_id * D * M

    cum_stride_d = (stoc_len + 1) * V
    inner_scale = q_max_sq / stoc_len
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    num_k_blocks = (D + BLOCK_K - 1) // BLOCK_K
    for k_block in range(num_k_blocks):
        k_start = k_block * BLOCK_K
        for ki in tl.static_range(BLOCK_K):
            d = k_start + ki
            if d < D:
                # Load signs as int8 for early-exit check
                sa_i8 = tl.load(sign_a_ptr + ba_base + d * N + m_offsets, mask=m_mask, other=0)
                # Skip if all sign_a are zero — no contribution to acc
                if tl.sum(tl.abs(sa_i8).to(tl.int32)) > 0:
                    sb_i8 = tl.load(sign_b_ptr + bb_base + d * M + n_offsets, mask=n_mask, other=0)
                    # Skip if all sign_b are zero
                    if tl.sum(tl.abs(sb_i8).to(tl.int32)) > 0:
                        sa = sa_i8.to(tl.float32)
                        sb = sb_i8.to(tl.float32)
                        # Coalesced loads — (D, N/M) layout
                        ba = tl.load(boundary_a_ptr + ba_base + d * N + m_offsets,
                                     mask=m_mask, other=0).to(tl.int32)
                        bb = tl.load(boundary_b_ptr + bb_base + d * M + n_offsets,
                                     mask=n_mask, other=0).to(tl.int32)

                        k_vals = tl.load(k_table_ptr + d * V + ba, mask=m_mask, other=0).to(tl.int32)

                        cum_offsets = (d * cum_stride_d
                                       + k_vals[:, None].to(tl.int64) * V
                                       + bb[None, :].to(tl.int64))
                        counts = tl.load(cum_ptr + cum_offsets, mask=gather_mask, other=0).to(tl.float32)

                        acc += counts * sa[:, None] * sb[None, :]

    # Apply combined inner_scale × head_scale in one multiply
    head_scale = tl.load(scale_ptr + batch_id)
    acc *= inner_scale * head_scale

    out_base = batch_id * N * M
    out_offsets = m_offsets[:, None] * M + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_base + out_offsets, acc, mask=out_mask)


def sc_matmul_enable_batched_bipolar(
    q_flat: torch.Tensor,       # (BH, N, D) float32
    k_flat: torch.Tensor,       # (BH, N, D) float32
    q_maxs: torch.Tensor,       # (BH,) float32 — per-head max
    q_mins: torch.Tensor,       # (BH,) float32 — per-head min
    k_maxs: torch.Tensor,       # (BH,) float32
    k_mins: torch.Tensor,       # (BH,) float32
    sc_prec: int,
    config: dict,
    stoc_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Batched bipolar enable-signal SC matmul for all heads in one launch.

    Replaces the per-head loop with two kernel launches:
    1. fused_quant_bipolar_batched_kernel (for q and k)
    2. enable_matmul_bipolar_batched_kernel

    Args:
        q_flat, k_flat: (BH, N, D) float32 tensors
        q_maxs, q_mins, k_maxs, k_mins: (BH,) per-head ranges
        sc_prec: SC precision
        config: SC RNG config dict
        stoc_len: Stochastic stream length. If None, uses 2^sc_prec.

    Returns:
        output: (BH, N, N) float32
    """
    if stoc_len is None:
        stoc_len = 2 ** sc_prec

    q_flat = q_flat.contiguous()
    k_flat = k_flat.contiguous()
    BH, N, D = q_flat.shape
    M = N  # QK matmul: N x N
    device = q_flat.device

    q_max = 2 ** (sc_prec - 1) - 1
    q_min = -(2 ** (sc_prec - 1))
    max_rng_val = 2 ** sc_prec
    q_max_sq = float(q_max * q_max)

    # --- Per-head inv_scale on GPU ---
    abs_max_q = torch.maximum(q_maxs.abs(), q_mins.abs()).clamp(min=1e-5)
    abs_max_k = torch.maximum(k_maxs.abs(), k_mins.abs()).clamp(min=1e-5)
    scale_q = abs_max_q / q_max
    scale_k = abs_max_k / q_max
    inv_scale_q = 1.0 / scale_q  # (BH,)
    inv_scale_k = 1.0 / scale_k  # (BH,)

    # --- Batched fused quantization (writes directly in transposed (BH, D, N/M) layout) ---
    slice_size = N * D
    boundary_q = torch.empty(BH, D, N, dtype=torch.int16, device=device)
    sign_q = torch.empty(BH, D, N, dtype=torch.int8, device=device)
    boundary_k = torch.empty(BH, D, M, dtype=torch.int16, device=device)
    sign_k = torch.empty(BH, D, M, dtype=torch.int8, device=device)

    BLOCK = 1024
    grid_quant = (triton.cdiv(slice_size, BLOCK), BH)
    fused_quant_bipolar_batched_kernel[grid_quant](
        q_flat, boundary_q, sign_q, inv_scale_q,
        q_max, q_min, max_rng_val, slice_size, N, D, BLOCK,
    )
    fused_quant_bipolar_batched_kernel[grid_quant](
        k_flat, boundary_k, sign_k, inv_scale_k,
        q_max, q_min, max_rng_val, slice_size, M, D, BLOCK,
    )

    # --- Get cached tables (shared across all heads) ---
    rand_seqs_a_t, rand_seqs_b_t = _get_cached_sequences(config, sc_prec, device)
    V = 2 ** sc_prec + 1
    cum_table_bytes = D * (stoc_len + 1) * V * 2
    use_compact = cum_table_bytes > _COMPACT_ENABLE_THRESHOLD_BYTES

    if use_compact:
        # Compact path not batched yet — fall back to per-head loop
        # Compact kernel expects (N, D) layout; transpose from (D, N)
        k_table = _get_cached_k_table(config, sc_prec, device, rand_seqs_a_t, stoc_len)
        rng_b_prefix = rand_seqs_b_t[:, :stoc_len].contiguous() if stoc_len < rand_seqs_b_t.shape[1] else rand_seqs_b_t
        output = torch.empty(BH, N, M, dtype=torch.float32, device=device)
        for i in range(BH):
            sc_raw = enable_matmul_compact(
                rng_b_prefix, k_table,
                boundary_q[i].t().contiguous(), boundary_k[i].t().contiguous(),
                sign_q[i].t().contiguous(), sign_k[i].t().contiguous(),
                N, M, D, stoc_len, q_max_sq, is_bipolar=True,
            )
            output[i] = sc_raw * (scale_q[i] * scale_k[i]).item()
        return output

    cum_indicator, k_table = _get_cached_enable_tables(
        config, sc_prec, device, rand_seqs_a_t, rand_seqs_b_t, stoc_len)

    # V from actual table layout (may be padded to power-of-2)
    V_actual = cum_indicator.shape[2]

    # Per-head output scale on GPU
    out_scale = scale_q * scale_k  # (BH,)

    # --- Batched matmul kernel ---
    # boundary/sign already in (BH, D, N/M) layout from fused-transpose quant kernel
    output = torch.empty(BH, N, M, dtype=torch.float32, device=device)

    # Adaptive tile size: smaller tiles for small N/M to increase GPU occupancy
    if N <= 64 or M <= 64:
        BLOCK_M = 16
        BLOCK_N = 16
    else:
        BLOCK_M = 32
        BLOCK_N = 32
    # BLOCK_K=4 reduces register pressure vs 8, improving occupancy
    if D >= 4 and D % 4 == 0:
        BLOCK_K = 4
    elif D % 2 == 0:
        BLOCK_K = 2
    else:
        BLOCK_K = 1

    # Warp count tuning: more warps for larger tiles, fewer for smaller
    nw = 8 if BLOCK_M == 32 else 2
    grid_mm = (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N), BH)
    enable_matmul_bipolar_batched_kernel[grid_mm](
        cum_indicator, k_table,
        boundary_q, boundary_k,
        sign_q, sign_k,
        output, out_scale,
        N, M, D,
        stoc_len, V_actual, q_max_sq,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=nw,
    )

    return output


# =============================================================================
# Enable-Signal Host Functions
# =============================================================================

_enable_table_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_k_table_cache: dict[str, torch.Tensor] = {}


def _enable_table_cache_key(config: dict, sc_prec: int, device: torch.device) -> str:
    """Cache key for enable tables (same as RNG cache key)."""
    return json.dumps(config, sort_keys=True) + f"|{sc_prec}|{device}|enable"


def build_enable_tables(
    rng_a: torch.Tensor,
    rng_b: torch.Tensor,
    sc_prec: int,
    stoc_len: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build lookup tables for enable-signal multiplication.

    Args:
        rng_a: (D, max_stoc_len) int32 tensor — per-dimension RNG sequences for A
        rng_b: (D, max_stoc_len) int32 tensor — per-dimension RNG sequences for B
        sc_prec: SC precision (controls quantization grid: V = 2^sc_prec + 1)
        stoc_len: Stochastic stream length. If None, uses 2^sc_prec.
                  When < 2^sc_prec, uses a prefix of the RNG sequences.

    Returns:
        cum_indicator: (D, stoc_len+1, V) int16 tensor
        k_table: (D, V) int16 tensor
    """
    if stoc_len is None:
        stoc_len = 2 ** sc_prec
    D = rng_a.shape[0]
    V = 2 ** sc_prec + 1  # boundary range [0, stoc_len] needs stoc_len+1 entries
    # Triton tl.arange requires power-of-2 sizes; pad V for the kernel
    V_PADDED = triton.next_power_of_2(V)
    device = rng_a.device

    # Use prefix of RNG sequences for shorter stoc_len
    rng_a_prefix = rng_a[:, :stoc_len].contiguous()
    rng_b_prefix = rng_b[:, :stoc_len].contiguous()

    cum_indicator = torch.zeros(D, stoc_len + 1, V_PADDED, dtype=torch.int16, device=device)
    k_table = torch.zeros(D, V_PADDED, dtype=torch.int16, device=device)

    # Launch table-build kernels
    build_cum_indicator_kernel[(D,)](
        rng_b_prefix, cum_indicator,
        D, stoc_len, V_PADDED,
    )
    compute_k_table_kernel[(D,)](
        rng_a_prefix, k_table,
        D, stoc_len, V_PADDED,
    )

    return cum_indicator, k_table


def _get_cached_enable_tables(
    config: dict, sc_prec: int, device: torch.device,
    rng_a: torch.Tensor, rng_b: torch.Tensor,
    stoc_len: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get or build+cache enable tables (cum_indicator + k_table)."""
    if stoc_len is None:
        stoc_len = 2 ** sc_prec
    key = _enable_table_cache_key(config, sc_prec, device) + f"|sl={stoc_len}"
    if key not in _enable_table_cache:
        _enable_table_cache[key] = build_enable_tables(rng_a, rng_b, sc_prec, stoc_len)
    return _enable_table_cache[key]


def _get_cached_k_table(
    config: dict, sc_prec: int, device: torch.device,
    rng_a: torch.Tensor,
    stoc_len: Optional[int] = None,
) -> torch.Tensor:
    """Get or build+cache k_table only (for compact path)."""
    if stoc_len is None:
        stoc_len = 2 ** sc_prec
    key = _enable_table_cache_key(config, sc_prec, device) + f"|k_only|sl={stoc_len}"
    if key not in _k_table_cache:
        _k_table_cache[key] = build_k_table_only(rng_a, sc_prec, stoc_len)
    return _k_table_cache[key]


def enable_matmul_triton(
    cum_indicator: torch.Tensor,
    k_table: torch.Tensor,
    boundary_a: torch.Tensor,
    boundary_b: torch.Tensor,
    sign_a: torch.Tensor,
    sign_b: torch.Tensor,
    N: int, M: int, D: int,
    stoc_len: int,
    q_max_sq: float,
    is_bipolar: bool,
) -> torch.Tensor:
    """
    Launch enable-signal matmul kernel.

    Args:
        cum_indicator: (D, stoc_len+1, V) int16
        k_table: (D, V) int16
        boundary_a: (N, D) int16
        boundary_b: (M, D) int16
        sign_a: (N, D) int8 (bipolar only)
        sign_b: (M, D) int8 (bipolar only)
        N, M, D: matrix dimensions
        stoc_len: stochastic stream length
        q_max_sq: q_max^2 for decoding
        is_bipolar: True for bipolar mode

    Returns:
        output: (N, M) float32
    """
    V = cum_indicator.shape[2]
    output = torch.empty(N, M, dtype=torch.float32, device=boundary_a.device)

    # Transpose boundary/sign to (D, N/M) for coalesced kernel access
    boundary_a = boundary_a.t().contiguous()
    boundary_b = boundary_b.t().contiguous()
    if is_bipolar:
        sign_a = sign_a.t().contiguous()
        sign_b = sign_b.t().contiguous()

    # Adaptive tile size: smaller tiles for small N/M to increase GPU occupancy
    if N <= 64 or M <= 64:
        BLOCK_M = 16
        BLOCK_N = 16
    else:
        BLOCK_M = 32
        BLOCK_N = 32
    # BLOCK_K=4 reduces register pressure vs 8, improving occupancy
    if D >= 4 and D % 4 == 0:
        BLOCK_K = 4
    elif D % 2 == 0:
        BLOCK_K = 2
    else:
        BLOCK_K = 1
    # Warp count tuning: more warps for larger tiles, fewer for smaller
    nw = 8 if BLOCK_M == 32 else 2
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N))
    if is_bipolar:
        enable_matmul_bipolar_tiled_kernel[grid](
            cum_indicator, k_table,
            boundary_a, boundary_b,
            sign_a, sign_b,
            output,
            N, M, D, stoc_len, V,
            q_max_sq, BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=nw,
        )
    else:
        enable_matmul_unipolar_tiled_kernel[grid](
            cum_indicator, k_table,
            boundary_a, boundary_b,
            output,
            N, M, D, stoc_len, V,
            q_max_sq, BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=nw,
        )

    return output


def build_k_table_only(
    rng_a: torch.Tensor,
    sc_prec: int,
    stoc_len: Optional[int] = None,
) -> torch.Tensor:
    """Build only the k_table (not cum_indicator). Used by compact enable path.

    Args:
        rng_a: (D, max_stoc_len) int32 tensor.
        sc_prec: SC precision (controls V = 2^sc_prec + 1).
        stoc_len: Stream length. If None, uses 2^sc_prec.
                  When < 2^sc_prec, uses prefix of rng_a.
    """
    if stoc_len is None:
        stoc_len = 2 ** sc_prec
    D = rng_a.shape[0]
    V = 2 ** sc_prec + 1
    V_PADDED = triton.next_power_of_2(V)
    rng_a_prefix = rng_a[:, :stoc_len].contiguous()
    k_table = torch.zeros(D, V_PADDED, dtype=torch.int16, device=rng_a.device)
    compute_k_table_kernel[(D,)](rng_a_prefix, k_table, D, stoc_len, V_PADDED)
    return k_table


def enable_matmul_compact(
    rng_b: torch.Tensor,
    k_table: torch.Tensor,
    boundary_a: torch.Tensor,
    boundary_b: torch.Tensor,
    sign_a: torch.Tensor,
    sign_b: torch.Tensor,
    N: int, M: int, D: int,
    stoc_len: int,
    q_max_sq: float,
    is_bipolar: bool,
) -> torch.Tensor:
    """
    Compact enable-signal matmul for attention (small D). No split-D.

    Uses rng_b (D, stoc_len) int32 directly, computing counts on-the-fly.
    Inputs in (N, D) / (M, D) row-major layout.
    """
    V = k_table.shape[1]
    output = torch.empty(N, M, dtype=torch.float32, device=boundary_a.device)

    # tl.dot requires K >= 32 on Blackwell (>= 16 on older archs), so the compact
    # dot kernel needs BATCH_T=32. When stoc_len < 32 the inner loop would iterate
    # zero times and silently zero the output. For those tiny streams, build a
    # small cum_indicator on the fly (memory is trivial: D*(stoc_len+1)*V*2 bytes,
    # e.g. ~41 KB for D=72, sl=16, V=257) and dispatch through the table kernel.
    if stoc_len < 32:
        cum_indicator = torch.empty(
            D, stoc_len + 1, V, dtype=torch.int16, device=rng_b.device
        )
        build_cum_indicator_kernel[(D,)](rng_b, cum_indicator, D, stoc_len, V)
        return enable_matmul_triton(
            cum_indicator, k_table, boundary_a, boundary_b, sign_a, sign_b,
            N, M, D, stoc_len, q_max_sq, is_bipolar,
        )

    BLOCK_M = 32
    BLOCK_N = 32
    BATCH_T = 32
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N))
    if is_bipolar:
        enable_matmul_compact_bipolar_dot_kernel[grid](
            rng_b, k_table,
            boundary_a, boundary_b,
            sign_a, sign_b,
            output,
            N, M, D, stoc_len, V,
            q_max_sq, BLOCK_M, BLOCK_N, BATCH_T,
        )
    else:
        enable_matmul_compact_unipolar_dot_kernel[grid](
            rng_b, k_table,
            boundary_a, boundary_b,
            output,
            N, M, D, stoc_len, V,
            q_max_sq, BLOCK_M, BLOCK_N, BATCH_T,
        )

    return output


def enable_matmul_compact_mlp(
    rng_b: torch.Tensor,
    k_table: torch.Tensor,
    boundary_a: torch.Tensor,
    boundary_b: torch.Tensor,
    sign_a: torch.Tensor,
    sign_b: torch.Tensor,
    N: int, M: int, D: int,
    stoc_len: int,
    q_max_sq: float,
    is_bipolar: bool,
) -> torch.Tensor:
    """
    Chunked cum_indicator matmul for MLP layers (large D).

    Instead of computing counts on-the-fly in O(stoc_len) per element,
    builds cum_indicator in D-chunks and uses O(1) table lookup per element.
    Each chunk's cum_indicator fits in ~34MB (vs ~608MB for full D).

    Algorithm:
      For each D-chunk [d_start, d_end):
        1. Build cum_indicator for rng_b[d_start:d_end] — O(D_CHUNK * stoc_len * V)
        2. Run fast tiled matmul with O(1) lookup — O(N * M * D_CHUNK)
        3. Accumulate partial result into output
    """
    V = k_table.shape[1]
    device = boundary_a.device
    output = torch.zeros(N, M, dtype=torch.float32, device=device)

    # D_CHUNK chosen to keep cum_indicator under ~34MB:
    # D_CHUNK * (stoc_len+1) * V * 2 bytes
    D_CHUNK = 128

    # Reusable buffer for cum_indicator (allocated once, reused per chunk)
    cum_buf = torch.zeros(D_CHUNK, stoc_len + 1, V, dtype=torch.int16, device=device)
    # Reusable buffer for partial output
    partial = torch.empty(N, M, dtype=torch.float32, device=device)

    # Adaptive tile size: smaller tiles for small N/M to increase GPU occupancy
    if N <= 64 or M <= 64:
        BLOCK_M = 16
        BLOCK_N = 16
    else:
        BLOCK_M = 32
        BLOCK_N = 32
    if D_CHUNK >= 4 and D_CHUNK % 4 == 0:
        BLOCK_K = 4
    elif D_CHUNK % 2 == 0:
        BLOCK_K = 2
    else:
        BLOCK_K = 1
    # Warp count tuning: more warps for larger tiles, fewer for smaller
    nw = 8 if BLOCK_M == 32 else 2
    grid_mm = (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N))

    for d_start in range(0, D, D_CHUNK):
        d_end = min(d_start + D_CHUNK, D)
        d_len = d_end - d_start

        # Build cum_indicator for this chunk of D dimensions
        rng_b_chunk = rng_b[d_start:d_end].contiguous()
        if d_len < D_CHUNK:
            # Last chunk: allocate smaller buffer
            cum_chunk = torch.zeros(d_len, stoc_len + 1, V, dtype=torch.int16, device=device)
        else:
            cum_chunk = cum_buf
            cum_chunk.zero_()
        build_cum_indicator_kernel[(d_len,)](
            rng_b_chunk, cum_chunk,
            d_len, stoc_len, V,
        )

        # Slice boundaries/signs for this D-chunk, transpose to (D, N/M) for coalesced access
        ba_chunk = boundary_a[:, d_start:d_end].t().contiguous()
        bb_chunk = boundary_b[:, d_start:d_end].t().contiguous()
        k_tab_chunk = k_table[d_start:d_end].contiguous()

        # Run fast tiled matmul with O(1) cum_indicator lookup
        if is_bipolar:
            sa_chunk = sign_a[:, d_start:d_end].t().contiguous()
            sb_chunk = sign_b[:, d_start:d_end].t().contiguous()
            enable_matmul_bipolar_tiled_kernel[grid_mm](
                cum_chunk, k_tab_chunk,
                ba_chunk, bb_chunk,
                sa_chunk, sb_chunk,
                partial,
                N, M, d_len, stoc_len, V,
                q_max_sq, BLOCK_M, BLOCK_N, BLOCK_K,
                num_warps=nw,
            )
        else:
            enable_matmul_unipolar_tiled_kernel[grid_mm](
                cum_chunk, k_tab_chunk,
                ba_chunk, bb_chunk,
                partial,
                N, M, d_len, stoc_len, V,
                q_max_sq, BLOCK_M, BLOCK_N, BLOCK_K,
                num_warps=nw,
            )

        output += partial

    return output


@torch.no_grad()
def sc_matmul_enable_triton(
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
    """
    Enable-signal SC matmul on GPU using Triton table-lookup kernels.

    Same interface as sc_matmul_enable() but uses Triton for acceleration.
    Pipeline: quantize → build tables (GPU) → matmul (GPU) → dequantize.

    Args:
        a: Left operand, shape (N, D) or (B, N, D). FP values.
        b: Right operand, shape (M, D) or (B, M, D). FP values.
        max_fp_a: Max FP value for operand a.
        min_fp_a: Min FP value for operand a.
        max_fp_b: Max FP value for operand b. If None, uses max_fp_a.
        min_fp_b: Min FP value for operand b. If None, uses min_fp_a.
        mode: "bipolar" (symmetric sign-magnitude) or "unipolar" (asymmetric AND).
        sc_prec: SC precision. Controls quantization grid (q_max, V, max_rng_val).
        config: Optional SC RNG/SNG config dict. If None, uses sobol_simple.
        stoc_len: Stochastic stream length. If None, uses 2^sc_prec.
                  Shorter stoc_len = fewer iterations = proportional speedup.

    Returns:
        Result tensor in FP, shape (N, M) or (B, N, M).
    """
    if stoc_len is None:
        stoc_len = 2 ** sc_prec

    if max_fp_b is None:
        max_fp_b = max_fp_a
    if min_fp_b is None:
        min_fp_b = min_fp_a

    if a.dim() == 3:
        return _sc_matmul_enable_triton_batched(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, mode, sc_prec, config
        )

    assert a.dim() == 2 and b.dim() == 2, f"Expected 2D, got a:{a.dim()}D b:{b.dim()}D"
    assert a.shape[1] == b.shape[1], f"Dim mismatch: a={a.shape[1]}, b={b.shape[1]}"

    N, D = a.shape
    M = b.shape[0]
    max_rng_val = 2 ** sc_prec

    if config is None:
        from config_helpers import make_sobol_simple_config
        config = make_sobol_simple_config(D, D, sc_prec)

    device = a.device
    if device.type != 'cuda':
        a = a.cuda()
        b = b.cuda()
    a = a.float()
    b = b.float()

    # Get cached RNG sequences
    rand_seqs_a_t, rand_seqs_b_t = _get_cached_sequences(config, sc_prec, a.device)

    # Choose compact vs table-based path based on memory
    V = 2 ** sc_prec + 1
    cum_table_bytes = D * (stoc_len + 1) * V * 2  # int16
    use_compact = cum_table_bytes > _COMPACT_ENABLE_THRESHOLD_BYTES

    # Use cached enable tables to avoid rebuilding every call
    if use_compact:
        k_table = _get_cached_k_table(config, sc_prec, a.device, rand_seqs_a_t, stoc_len)
        rng_b_for_compact = rand_seqs_b_t[:, :stoc_len].contiguous() if stoc_len < rand_seqs_b_t.shape[1] else rand_seqs_b_t
    else:
        cum_indicator, k_table = _get_cached_enable_tables(
            config, sc_prec, a.device, rand_seqs_a_t, rand_seqs_b_t, stoc_len)
        rng_b_for_compact = None

    if mode == "bipolar":
        result = _sc_matmul_enable_triton_bipolar(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
            cum_indicator if not use_compact else None,
            k_table, max_rng_val, N, D, M, stoc_len,
            rng_b=rng_b_for_compact,
        )
    elif mode == "unipolar":
        result = _sc_matmul_enable_triton_unipolar(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
            cum_indicator if not use_compact else None,
            k_table, max_rng_val, N, D, M, stoc_len,
            rng_b=rng_b_for_compact,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if device.type != 'cuda':
        result = result.to(device)

    return result


def _sc_matmul_enable_triton_bipolar(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
    cum_indicator, k_table, max_rng_val, N, D, M, stoc_len,
    rng_b=None,
):
    """Bipolar enable-signal SC matmul via Triton (sign-magnitude)."""
    q_max = 2 ** (sc_prec - 1) - 1

    # Fused quantization: FP -> (boundary, sign) in one kernel each
    abs_max_a = max(abs(max_fp_a), abs(min_fp_a), 1e-5)
    abs_max_b = max(abs(max_fp_b), abs(min_fp_b), 1e-5)
    boundary_a, sign_a, scale_a = fused_quantize_bipolar(a, abs_max_a, sc_prec)
    boundary_b, sign_b, scale_b = fused_quantize_bipolar(b, abs_max_b, sc_prec)

    q_max_sq = float(q_max * q_max)

    if rng_b is not None:
        sc_raw = enable_matmul_compact(
            rng_b, k_table, boundary_a, boundary_b,
            sign_a, sign_b, N, M, D, stoc_len, q_max_sq, is_bipolar=True,
        )
    else:
        sc_raw = enable_matmul_triton(
            cum_indicator, k_table, boundary_a, boundary_b,
            sign_a, sign_b, N, M, D, stoc_len, q_max_sq, is_bipolar=True,
        )

    return sc_raw * (scale_a * scale_b)


def _sc_matmul_enable_triton_unipolar(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
    cum_indicator, k_table, max_rng_val, N, D, M, stoc_len,
    rng_b=None,
):
    """Unipolar enable-signal SC matmul via Triton (asymmetric + zero-point)."""
    q_max_sq = float((2 ** sc_prec - 1) ** 2)

    # Fused quantization + row-sum in one kernel each
    boundary_a, scale_a, zp_a_f, a_sum = fused_quantize_unipolar(
        a, max_fp_a, min_fp_a, sc_prec, compute_sum=True)
    boundary_b, scale_b, zp_b_f, b_sum = fused_quantize_unipolar(
        b, max_fp_b, min_fp_b, sc_prec, compute_sum=True)

    if rng_b is not None:
        sc_raw = enable_matmul_compact(
            rng_b, k_table, boundary_a, boundary_b,
            None, None, N, M, D, stoc_len, q_max_sq, is_bipolar=False,
        )
    else:
        sc_raw = enable_matmul_triton(
            cum_indicator, k_table, boundary_a, boundary_b,
            None, None, N, M, D, stoc_len, q_max_sq, is_bipolar=False,
        )

    # Zero-point correction (a_sum/b_sum already computed by fused kernel)
    correction = (-zp_b_f * a_sum[:, None]
                  - zp_a_f * b_sum[None, :]
                  + D * zp_a_f * zp_b_f)
    corrected = sc_raw + correction

    return corrected * (scale_a * scale_b)


def _sc_matmul_enable_triton_batched(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, mode, sc_prec, config,
):
    """Batched enable-signal SC matmul via Triton with CUDA streams."""
    B, N, D = a.shape
    M = b.shape[1]
    output = torch.empty(B, N, M, dtype=torch.float32, device=a.device)

    streams = [torch.cuda.Stream() for _ in range(B)]
    for i in range(B):
        with torch.cuda.stream(streams[i]):
            output[i] = sc_matmul_enable_triton(
                a[i], b[i], max_fp_a, min_fp_a, max_fp_b, min_fp_b,
                mode, sc_prec, config,
            )
    for s in streams:
        s.synchronize()
    return output


# =============================================================================
# Enable-Signal SC Matmul for MLP Layers (large D, compact path only)
# =============================================================================


def _sc_matmul_enable_triton_bipolar_mlp(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
    k_table, rng_b, N, D, M, stoc_len,
    group_a=0, group_b=0,
):
    """Bipolar enable-signal SC matmul for MLP with per-row-group quantization."""
    q_max = 2 ** (sc_prec - 1) - 1    # 127: for scale, boundary norm, and dequant
    q_max_sq = float(q_max * q_max)

    # Default: per-row for activation, per-channel for weight
    if group_a <= 0:
        group_a = 1
    if group_b <= 0:
        group_b = 1

    scale_a_row, a_int, sign_a = _grouped_symmetric_quant(a, group_a, q_max, clip_margin=0)
    scale_b_row, b_int, sign_b = _grouped_symmetric_quant(b, group_b, q_max, clip_margin=0)

    # Convert quantized integers to boundaries for enable-signal lookup
    max_rng_val = 2 * (q_max + 1)  # = 2^sc_prec = stoc_len
    boundary_a = (a_int.abs() * max_rng_val / q_max).round().short()
    boundary_b = (b_int.abs() * max_rng_val / q_max).round().short()

    sc_raw = enable_matmul_compact_mlp(
        rng_b, k_table, boundary_a, boundary_b,
        sign_a, sign_b, N, M, D, stoc_len, q_max_sq, is_bipolar=True,
    )

    # Per-element dequantization with row-group scales
    return sc_raw * (scale_a_row[:, None] * scale_b_row[None, :])


def _sc_matmul_bipolar_mlp_chunked(
    a, b, sc_prec, k_table, rng_b, chunk_d,
    stoc_len=None,
):
    """
    Bipolar SC matmul for MLP with internal chunk_d loop.

    Handles the entire D-chunking internally, replacing the Python loop in
    sc_mlp.py. Key optimizations vs calling sc_matmul_enable_triton_mlp in a loop:
    - Build cum_indicator ONCE (all chunks share same config/RNG)
    - Use fused per-row quant kernel (1 launch vs ~12 PyTorch ops per chunk)
    - No .item() GPU sync calls (bipolar doesn't need max/min)
    - Minimal Python overhead per chunk

    Total kernel launches: 1 (build) + num_chunks * 3 (quant_a + quant_b + matmul)
    vs old: num_chunks * ~38 launches + 4 syncs each
    """
    N, D = a.shape
    M = b.shape[0]
    if stoc_len is None:
        stoc_len = 2 ** sc_prec
    q_max = 2 ** (sc_prec - 1) - 1
    q_max_sq = float(q_max * q_max)

    V = k_table.shape[1]
    device = a.device
    output = torch.zeros(N, M, dtype=torch.float32, device=device)

    # Build cum_indicator ONCE — all chunks share the same RNG sequences
    cum_indicator = torch.zeros(chunk_d, stoc_len + 1, V, dtype=torch.int16, device=device)
    build_cum_indicator_kernel[(chunk_d,)](
        rng_b, cum_indicator,
        chunk_d, stoc_len, V,
    )

    # Tiled matmul params — adaptive tile size for small N/M
    if N <= 64 or M <= 64:
        BLOCK_M = 16
        BLOCK_N = 16
    else:
        BLOCK_M = 32
        BLOCK_N = 32
    if chunk_d >= 4 and chunk_d % 4 == 0:
        BLOCK_K = 4
    elif chunk_d % 2 == 0:
        BLOCK_K = 2
    else:
        BLOCK_K = 1
    # Warp count tuning: more warps for larger tiles, fewer for smaller
    nw = 8 if BLOCK_M == 32 else 2
    grid_mm = (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N))

    # Reusable buffer for partial matmul output
    partial = torch.empty(N, M, dtype=torch.float32, device=device)

    # Per-row scale accumulators for dequantization
    # Each chunk has its own per-row scales; we accumulate via outer product
    for d_start in range(0, D, chunk_d):
        d_end = min(d_start + chunk_d, D)
        d_len = d_end - d_start

        # Slice input chunks (make contiguous for kernel addressing)
        a_chunk = a[:, d_start:d_end].contiguous()
        b_chunk = b[:, d_start:d_end].contiguous()

        # Fused per-row quant: 1 kernel launch each (vs ~12 PyTorch ops each)
        boundary_a, sign_a, scale_a = fused_quantize_bipolar_perrow(a_chunk, sc_prec)
        boundary_b, sign_b, scale_b = fused_quantize_bipolar_perrow(b_chunk, sc_prec)

        # Handle last chunk if smaller than chunk_d
        if d_len < chunk_d:
            cum_chunk = torch.zeros(d_len, stoc_len + 1, V, dtype=torch.int16, device=device)
            rng_b_chunk = rng_b[:d_len].contiguous()
            build_cum_indicator_kernel[(d_len,)](
                rng_b_chunk, cum_chunk,
                d_len, stoc_len, V,
            )
            k_tab_chunk = k_table[:d_len].contiguous()
        else:
            cum_chunk = cum_indicator
            k_tab_chunk = k_table

        # Transpose boundary/sign to (D, N/M) for coalesced kernel access
        boundary_a_t = boundary_a.t().contiguous()
        boundary_b_t = boundary_b.t().contiguous()
        sign_a_t = sign_a.t().contiguous()
        sign_b_t = sign_b.t().contiguous()

        # Fast tiled matmul with O(1) cum_indicator lookup
        enable_matmul_bipolar_tiled_kernel[grid_mm](
            cum_chunk, k_tab_chunk,
            boundary_a_t, boundary_b_t,
            sign_a_t, sign_b_t,
            partial,
            N, M, d_len, stoc_len, V,
            q_max_sq, BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=nw,
        )

        # Accumulate with per-chunk dequantization scales
        output += partial * (scale_a[:, None] * scale_b[None, :])

    return output


def _sc_matmul_enable_triton_unipolar_mlp(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
    k_table, rng_b, N, D, M, stoc_len,
):
    """Unipolar enable-signal SC matmul for MLP via compact MLP kernel."""
    q_max_sq = float((2 ** sc_prec - 1) ** 2)

    boundary_a, scale_a, zp_a_f, a_sum = fused_quantize_unipolar(
        a, max_fp_a, min_fp_a, sc_prec, compute_sum=True)
    boundary_b, scale_b, zp_b_f, b_sum = fused_quantize_unipolar(
        b, max_fp_b, min_fp_b, sc_prec, compute_sum=True)

    sc_raw = enable_matmul_compact_mlp(
        rng_b, k_table, boundary_a, boundary_b,
        None, None, N, M, D, stoc_len, q_max_sq, is_bipolar=False,
    )

    correction = (-zp_b_f * a_sum[:, None]
                  - zp_a_f * b_sum[None, :]
                  + D * zp_a_f * zp_b_f)
    corrected = sc_raw + correction

    return corrected * (scale_a * scale_b)


def _sc_matmul_enable_triton_mlp_batched(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, mode, sc_prec, config,
):
    """Batched enable-signal SC matmul for MLP via CUDA streams."""
    B, N, D = a.shape
    M = b.shape[1]
    output = torch.empty(B, N, M, dtype=torch.float32, device=a.device)

    streams = [torch.cuda.Stream() for _ in range(B)]
    for i in range(B):
        with torch.cuda.stream(streams[i]):
            output[i] = sc_matmul_enable_triton_mlp(
                a[i], b[i], max_fp_a, min_fp_a, max_fp_b, min_fp_b,
                mode, sc_prec, config,
            )
    for s in streams:
        s.synchronize()
    return output


@torch.no_grad()
def sc_matmul_enable_triton_mlp(
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
    """
    Enable-signal SC matmul for MLP layers.

    Args:
        chunk_d: If > 0, split D into chunks of this size for reduced SC error.
                 When chunk_d > 0 and mode == "bipolar", uses optimized internal
                 chunking that builds cum_indicator once and uses fused per-row
                 quantization (~10x fewer kernel launches vs external loop).
        group_a: rows per quantization group for a (1 = per-row, default).
        group_b: rows per quantization group for b (1 = per-row/per-channel, default).
        stoc_len: Stochastic stream length. If None, uses 2^sc_prec.
    """
    if stoc_len is None:
        stoc_len = 2 ** sc_prec

    if max_fp_b is None:
        max_fp_b = max_fp_a
    if min_fp_b is None:
        min_fp_b = min_fp_a

    if a.dim() == 3:
        return _sc_matmul_enable_triton_mlp_batched(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, mode, sc_prec, config
        )

    assert a.dim() == 2 and b.dim() == 2, f"Expected 2D, got a:{a.dim()}D b:{b.dim()}D"
    assert a.shape[1] == b.shape[1], f"Dim mismatch: a={a.shape[1]}, b={b.shape[1]}"

    N, D = a.shape
    M = b.shape[0]

    device = a.device
    if device.type != 'cuda':
        a = a.cuda()
        b = b.cuda()
    a = a.float()
    b = b.float()

    # Fast path: bipolar + chunk_d → optimized internal chunking
    if mode == "bipolar" and chunk_d > 0 and D > chunk_d:
        # Config is for chunk_d dimensions (not full D)
        if config is None:
            from config_helpers import make_sobol_simple_config
            config = make_sobol_simple_config(chunk_d, chunk_d, sc_prec)

        rand_seqs_a_t, rand_seqs_b_t = _get_cached_sequences(config, sc_prec, a.device)
        k_table = _get_cached_k_table(config, sc_prec, a.device, rand_seqs_a_t, stoc_len)
        rng_b = rand_seqs_b_t[:, :stoc_len].contiguous() if stoc_len < rand_seqs_b_t.shape[1] else rand_seqs_b_t

        result = _sc_matmul_bipolar_mlp_chunked(
            a, b, sc_prec, k_table, rng_b, chunk_d,
            stoc_len=stoc_len,
        )

        if device.type != 'cuda':
            result = result.to(device)
        return result

    # Standard path (no chunk_d, or unipolar, or D <= chunk_d)
    if config is None:
        from config_helpers import make_sobol_simple_config
        config = make_sobol_simple_config(D, D, sc_prec)

    rand_seqs_a_t, rand_seqs_b_t = _get_cached_sequences(config, sc_prec, a.device)
    k_table = _get_cached_k_table(config, sc_prec, a.device, rand_seqs_a_t, stoc_len)
    rng_b = rand_seqs_b_t[:, :stoc_len].contiguous() if stoc_len < rand_seqs_b_t.shape[1] else rand_seqs_b_t

    if mode == "bipolar":
        result = _sc_matmul_enable_triton_bipolar_mlp(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
            k_table, rng_b, N, D, M, stoc_len,
            group_a=group_a, group_b=group_b,
        )
    elif mode == "unipolar":
        result = _sc_matmul_enable_triton_unipolar_mlp(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
            k_table, rng_b, N, D, M, stoc_len,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if device.type != 'cuda':
        result = result.to(device)

    return result


# =============================================================================
# Enable-Signal Grouped Quantization
# =============================================================================

@torch.no_grad()
def sc_matmul_grouped_enable_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    group_a: int = 1,
    group_b: int = 1,
    mode: str = "unipolar",
    sc_prec: int = 8,
    config: Optional[dict] = None,
    stoc_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Enable-signal SC matmul with per-row-group quantization: a @ b^T.

    Same as sc_matmul_grouped but uses enable-signal table-lookup instead
    of packed AND matmul.

    Args:
        a: (N, D) left operand.
        b: (M, D) right operand.
        group_a: rows per quantization group for a (1 = per-row, N = per-tensor).
        group_b: rows per quantization group for b (1 = per-row, M = per-tensor).
        mode: "bipolar" or "unipolar".
        sc_prec: SC precision (controls quantization grid).
        config: RNG/SNG config dict.
        stoc_len: Stochastic stream length. If None, uses 2^sc_prec.

    Returns:
        (N, M) result tensor in FP.
    """
    if stoc_len is None:
        stoc_len = 2 ** sc_prec

    assert a.dim() == 2 and b.dim() == 2, f"Expected 2D, got a:{a.dim()}D b:{b.dim()}D"
    assert a.shape[1] == b.shape[1], f"Inner dim mismatch: {a.shape[1]} vs {b.shape[1]}"

    N, D = a.shape
    M = b.shape[0]
    max_rng_val = 2 ** sc_prec

    if config is None:
        from config_helpers import make_sobol_simple_config
        config = make_sobol_simple_config(D, D, sc_prec)

    device = a.device
    if device.type != 'cuda':
        a = a.cuda()
        b = b.cuda()
    a = a.float()
    b = b.float()

    # Get cached RNG sequences
    rand_seqs_a_t, rand_seqs_b_t = _get_cached_sequences(config, sc_prec, a.device)

    # Choose compact vs table-based path based on memory, use cached tables
    V = 2 ** sc_prec + 1
    cum_table_bytes = D * (stoc_len + 1) * V * 2
    use_compact = cum_table_bytes > _COMPACT_ENABLE_THRESHOLD_BYTES

    if mode == "bipolar":
        # Bipolar grouped quantization path
        if use_compact:
            k_table = _get_cached_k_table(config, sc_prec, a.device, rand_seqs_a_t, stoc_len)
            rng_b = rand_seqs_b_t[:, :stoc_len].contiguous() if stoc_len < rand_seqs_b_t.shape[1] else rand_seqs_b_t
            result = _sc_matmul_bipolar_grouped_enable(
                a, b, group_a, group_b, sc_prec,
                None, k_table, max_rng_val, N, D, M, stoc_len,
                rng_b=rng_b,
            )
        else:
            cum_indicator, k_table = _get_cached_enable_tables(
                config, sc_prec, a.device, rand_seqs_a_t, rand_seqs_b_t, stoc_len)
            result = _sc_matmul_bipolar_grouped_enable(
                a, b, group_a, group_b, sc_prec,
                cum_indicator, k_table, max_rng_val, N, D, M, stoc_len,
            )
    elif mode == "unipolar":
        # Unipolar grouped quantization path
        if use_compact:
            k_table = _get_cached_k_table(config, sc_prec, a.device, rand_seqs_a_t, stoc_len)
            rng_b = rand_seqs_b_t[:, :stoc_len].contiguous() if stoc_len < rand_seqs_b_t.shape[1] else rand_seqs_b_t
            result = _sc_matmul_unipolar_grouped_enable(
                a, b, group_a, group_b, sc_prec,
                None, k_table, max_rng_val, N, D, M, stoc_len,
                rng_b=rng_b,
            )
        else:
            cum_indicator, k_table = _get_cached_enable_tables(
                config, sc_prec, a.device, rand_seqs_a_t, rand_seqs_b_t, stoc_len)
            result = _sc_matmul_unipolar_grouped_enable(
                a, b, group_a, group_b, sc_prec,
                cum_indicator, k_table, max_rng_val, N, D, M, stoc_len,
            )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if device.type != 'cuda':
        result = result.to(device)

    return result


def _sc_matmul_bipolar_grouped_enable(
    a, b, group_a, group_b, sc_prec,
    cum_indicator, k_table, max_rng_val, N, D, M, stoc_len,
    rng_b=None,
):
    """Bipolar enable-signal SC matmul with per-row-group quantization.

    Uses _grouped_symmetric_quant for host-side quantization,
    then enable-signal table-lookup kernel with sign handling.
    """
    q_max = 2 ** (sc_prec - 1) - 1

    # Per-row-group symmetric quantization
    scale_a_row, a_int, sign_a = _grouped_symmetric_quant(a, group_a, q_max, clip_margin=0)
    scale_b_row, b_int, sign_b = _grouped_symmetric_quant(b, group_b, q_max, clip_margin=0)

    # Compute boundaries for enable-signal lookup
    abs_a_int = a_int.abs()
    abs_b_int = b_int.abs()
    boundary_a = (abs_a_int * max_rng_val / q_max).round().short()  # (N, D)
    boundary_b = (abs_b_int * max_rng_val / q_max).round().short()  # (M, D)

    q_max_sq = float(q_max * q_max)

    # Enable-signal matmul with sign handling (bipolar)
    if rng_b is not None:
        sc_raw = enable_matmul_compact(
            rng_b, k_table, boundary_a, boundary_b,
            sign_a, sign_b, N, M, D, stoc_len, q_max_sq, is_bipolar=True,
        )
    else:
        sc_raw = enable_matmul_triton(
            cum_indicator, k_table, boundary_a, boundary_b,
            sign_a, sign_b, N, M, D, stoc_len, q_max_sq, is_bipolar=True,
        )

    # Per-element dequantization (no zero-point correction needed for symmetric)
    result_fp = sc_raw * (scale_a_row[:, None] * scale_b_row[None, :])

    return result_fp


def _sc_matmul_unipolar_grouped_enable(
    a, b, group_a, group_b, sc_prec,
    cum_indicator, k_table, max_rng_val, N, D, M, stoc_len,
    rng_b=None,
):
    """
    Unipolar enable-signal SC matmul with per-row-group quantization.

    Uses _grouped_asymmetric_quant for host-side quantization (same as standard),
    then enable-signal table-lookup kernel instead of packed AND matmul.
    """
    q_max = 2 ** sc_prec - 1

    # Per-row-group asymmetric quantization (reuse existing helper)
    scale_a_row, zp_a_row, a_int = _grouped_asymmetric_quant(a, group_a, q_max)
    scale_b_row, zp_b_row, b_int = _grouped_asymmetric_quant(b, group_b, q_max)

    # Compute boundaries for enable-signal lookup
    boundary_a = (a_int * max_rng_val / q_max).round().short()  # (N, D)
    boundary_b = (b_int * max_rng_val / q_max).round().short()  # (M, D)

    q_max_sq = float(q_max * q_max)

    # Enable-signal matmul: compact or table-based
    if rng_b is not None:
        sc_raw = enable_matmul_compact(
            rng_b, k_table, boundary_a, boundary_b,
            None, None, N, M, D, stoc_len, q_max_sq, is_bipolar=False,
        )
    else:
        sc_raw = enable_matmul_triton(
            cum_indicator, k_table, boundary_a, boundary_b,
            None, None, N, M, D, stoc_len, q_max_sq, is_bipolar=False,
        )

    # Per-element zero-point correction (same as standard grouped)
    a_sum = a_int.sum(dim=1)   # (N,)
    b_sum = b_int.sum(dim=1)   # (M,)

    correction = (
        -zp_b_row[None, :] * a_sum[:, None]
        - zp_a_row[:, None] * b_sum[None, :]
        + D * zp_a_row[:, None] * zp_b_row[None, :]
    )
    corrected = sc_raw + correction

    # Per-element dequantization
    result_fp = corrected * (scale_a_row[:, None] * scale_b_row[None, :])

    return result_fp


# =============================================================================
# sc_matmul: FP-in, FP-out entry point (supports bipolar and unipolar)
# =============================================================================

@torch.no_grad()
def sc_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    max_fp_a: float,
    min_fp_a: float,
    max_fp_b: float = None,
    min_fp_b: float = None,
    mode: str = "bipolar",
    sc_prec: int = 8,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """
    Stochastic computing matrix multiplication: a @ b^T.

    FP-in, FP-out. Internally quantizes to integers using Q-DiT-compatible
    symmetric (bipolar) or asymmetric (unipolar) quantization, performs SC
    matmul, and dequantizes back to FP.

    Each operand has its own quantization range for better precision.
    If max_fp_b/min_fp_b are not provided, falls back to using a's range
    for both operands (shared range, legacy behavior).

    Args:
        a: Left operand, shape (N, D) or (B, N, D). FP values.
        b: Right operand, shape (M, D) or (B, M, D). FP values.
        max_fp_a: Max FP value for operand a.
        min_fp_a: Min FP value for operand a.
        max_fp_b: Max FP value for operand b. If None, uses max_fp_a.
        min_fp_b: Min FP value for operand b. If None, uses min_fp_a.
        mode: "bipolar" (symmetric, XNOR gate) or "unipolar" (asymmetric, AND gate).
        sc_prec: SC precision. stoc_len = 2^sc_prec.
        config: Optional SC RNG/SNG config dict. If None, uses sobol_simple.

    Returns:
        Result tensor in FP, shape (N, M) or (B, N, M).
    """
    if max_fp_b is None:
        max_fp_b = max_fp_a
    if min_fp_b is None:
        min_fp_b = min_fp_a

    if a.dim() == 3:
        return _sc_matmul_batched(a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, mode, sc_prec, config)

    assert a.dim() == 2 and b.dim() == 2, f"Expected 2D tensors, got a:{a.dim()}D, b:{b.dim()}D"
    assert a.shape[1] == b.shape[1], f"Embedding dim mismatch: a={a.shape[1]}, b={b.shape[1]}"

    N, D = a.shape
    M = b.shape[0]
    stoc_len = 2 ** sc_prec

    # Build config
    if config is None:
        from config_helpers import make_sobol_simple_config
        config = make_sobol_simple_config(D, D, sc_prec)

    # Ensure tensors are on CUDA and float32
    device = a.device
    if device.type != 'cuda':
        a = a.cuda()
        b = b.cuda()
    a = a.float()
    b = b.float()

    # Get cached RNG sequences
    rand_seqs_a_t, rand_seqs_b_t = _get_cached_sequences(config, sc_prec, a.device)

    if mode == "bipolar":
        result = _sc_matmul_bipolar(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
            rand_seqs_a_t, rand_seqs_b_t, N, D, M, stoc_len,
        )
    elif mode == "unipolar":
        result = _sc_matmul_unipolar(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
            rand_seqs_a_t, rand_seqs_b_t, N, D, M, stoc_len,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'bipolar' or 'unipolar'.")

    if device.type != 'cuda':
        result = result.to(device)

    return result


@torch.no_grad()
def sc_matmul_mlp(
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
) -> torch.Tensor:
    """
    SC matmul for MLP layers. Delegates to sc_matmul (packed bitstream
    kernels work fine for all D sizes). chunk_d is ignored (no chunking needed).
    """
    if max_fp_a == 0.0:
        max_fp_a = a.max().item()
    if min_fp_a == 0.0:
        min_fp_a = a.min().item()
    if max_fp_b is None:
        max_fp_b = b.max().item()
    if min_fp_b is None:
        min_fp_b = b.min().item()
    return sc_matmul(a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b,
                     mode, sc_prec, config)


def _sc_matmul_bipolar(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
    rand_seqs_a_t, rand_seqs_b_t, N, D, M, stoc_len,
):
    """
    Bipolar SC matmul (symmetric quantization + XNOR).

    Each operand gets its own symmetric scale:
        q_max = 2^(sc_prec-1) - 1
        scale_a = abs_max_a / q_max
        scale_b = abs_max_b / q_max
        dequantize: result_fp = sc_raw * (scale_a * scale_b)
    """
    q_norm = 2 ** (sc_prec - 1) - 1     # 127: for SNG boundary normalization
    q_clip = q_norm - 2                  # 125: quantization range & clamp

    # Per-operand symmetric scales (use q_clip so dequant amplifies to compensate SC)
    abs_max_a = max(abs(max_fp_a), abs(min_fp_a), 1e-5)
    abs_max_b = max(abs(max_fp_b), abs(min_fp_b), 1e-5)
    scale_a = abs_max_a / q_clip
    scale_b = abs_max_b / q_clip

    # Quantize FP -> int with separate scales (clamp to ±q_clip)
    a_int = (a / scale_a).round().clamp(-q_clip, q_clip)
    b_int = (b / scale_b).round().clamp(-q_clip, q_clip)

    # Fused SNG + XNOR matmul (boundary normalization uses q_clip=120, so 120→full range)
    sc_raw = fused_xnor_matmul(a_int, b_int, rand_seqs_a_t, rand_seqs_b_t,
                                float(q_clip), sc_prec)

    # Dequantize: sc_raw is in integer-product domain
    result_fp = sc_raw * (scale_a * scale_b)

    return result_fp


def _sc_matmul_unipolar(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
    rand_seqs_a_t, rand_seqs_b_t, N, D, M, stoc_len,
):
    """
    Unipolar SC matmul (asymmetric quantization + AND).

    Each operand gets its own scale and zero_point:
        scale_a = (max_a - min_a) / q_max,  zp_a = round(-min_a / scale_a)
        scale_b = (max_b - min_b) / q_max,  zp_b = round(-min_b / scale_b)

    Dequantization with per-operand zero-point correction:
        SC AND computes: sum_e(a_int_e * b_int_e)
        Real dot product: sum_e((a_int_e - zp_a)(b_int_e - zp_b))
        = sum(a_int*b_int) - zp_b*sum(a_int) - zp_a*sum(b_int) + D*zp_a*zp_b
        result_fp = corrected * (scale_a * scale_b)
    """
    q_max = 2 ** sc_prec - 1
    q_lo, q_hi = 2, q_max - 2  # map min->2, max->253 for 8-bit

    # Per-operand asymmetric scales and zero points
    range_a = max(max_fp_a - min_fp_a, 1e-5)
    scale_a = range_a / (q_hi - q_lo)
    zp_a = round(-min_fp_a / scale_a) + q_lo
    zp_a = max(q_lo, min(q_hi, zp_a))

    range_b = max(max_fp_b - min_fp_b, 1e-5)
    scale_b = range_b / (q_hi - q_lo)
    zp_b = round(-min_fp_b / scale_b) + q_lo
    zp_b = max(q_lo, min(q_hi, zp_b))

    # Quantize FP -> int (non-negative) with separate scales
    a_int = (a / scale_a + zp_a).round().clamp(q_lo, q_hi)
    b_int = (b / scale_b + zp_b).round().clamp(q_lo, q_hi)

    # Fused SNG + AND matmul (no intermediate packed tensors)
    sc_raw = fused_and_matmul(a_int, b_int, rand_seqs_a_t, rand_seqs_b_t,
                               float(q_max), sc_prec)

    # Per-operand zero-point correction:
    # Real dot = sum_e((a_int - zp_a)(b_int - zp_b))
    # = sum(a_int*b_int) - zp_b*sum(a_int) - zp_a*sum(b_int) + D*zp_a*zp_b
    zp_a_f = float(zp_a)
    zp_b_f = float(zp_b)
    a_sum = a_int.sum(dim=-1, keepdim=True)  # (N, 1)
    b_sum = b_int.sum(dim=-1, keepdim=True)  # (M, 1)

    # sc_raw: (N, M), a_sum: (N, 1), b_sum.T: (1, M)
    correction = -zp_b_f * a_sum - zp_a_f * b_sum.transpose(-2, -1) + D * zp_a_f * zp_b_f
    corrected = sc_raw + correction

    # Dequantize: multiply by per-operand scales
    result_fp = corrected * (scale_a * scale_b)

    return result_fp


def _sc_matmul_unipolar_grouped(
    a, b, group_a, group_b, sc_prec,
    rand_seqs_a_t, rand_seqs_b_t, N, D, M, stoc_len,
):
    """
    Unipolar SC matmul with per-row-group quantization.

    Same SNG encoding and AND matmul kernels as _sc_matmul_unipolar, but
    quantization scales and zero-points are computed per group of rows instead
    of per-tensor.  This fixes the quality collapse when value distributions
    vary across rows (e.g. softmax attention where most values ≈ 1/N).

    a: (N, D), b: (M, D).  Computes a @ b^T -> (N, M).
    group_a: rows per quantization group for a.
    group_b: rows per quantization group for b.
    """
    q_max = 2 ** sc_prec - 1

    # --- per-row-group asymmetric quantization for a ---
    scale_a_row, zp_a_row, a_int = _grouped_asymmetric_quant(a, group_a, q_max)

    # --- per-row-group asymmetric quantization for b ---
    scale_b_row, zp_b_row, b_int = _grouped_asymmetric_quant(b, group_b, q_max)

    # --- Fused SNG + AND matmul (no intermediate packed tensors) ---
    sc_raw = fused_and_matmul(a_int, b_int, rand_seqs_a_t, rand_seqs_b_t,
                               float(q_max), sc_prec)

    # --- per-element zero-point correction ---
    # output[i, j] uses zp_a for row i's group and zp_b for row j's group
    a_sum = a_int.sum(dim=1)   # (N,)
    b_sum = b_int.sum(dim=1)   # (M,)

    # correction[i,j] = -zp_b[j]*a_sum[i] - zp_a[i]*b_sum[j] + D*zp_a[i]*zp_b[j]
    correction = (
        -zp_b_row[None, :] * a_sum[:, None]
        - zp_a_row[:, None] * b_sum[None, :]
        + D * zp_a_row[:, None] * zp_b_row[None, :]
    )
    corrected = sc_raw + correction

    # --- per-element dequantization ---
    result_fp = corrected * (scale_a_row[:, None] * scale_b_row[None, :])

    return result_fp


def _grouped_symmetric_quant(x, G, q_max, clip_margin=0):
    """Per-row-group symmetric quantization for bipolar mode.

    Args:
        x: (rows, cols) float tensor
        G: number of rows per quantization group
        q_max: max quantized value (e.g. 127 for 8-bit bipolar)
        clip_margin: headroom levels for outliers (default 2).
                     scale uses q_max - clip_margin, clamp uses q_max.
                     Set to 0 to disable clipping (standard quantization).

    Returns:
        scale_row: (rows,) per-row scale
        x_int:     (rows, cols) quantized values in [-q_max, q_max]
        sign:      (rows, cols) sign bits (+1/-1)
    """
    rows, cols = x.shape
    q_clip = q_max - clip_margin

    if G >= rows:
        # Single group (per-tensor) — fast path
        abs_max = x.abs().max().clamp(min=1e-5)
        scale = abs_max / q_clip  # use q_clip so dequant amplifies
        x_int = (x / scale).round().clamp(-q_max, q_max)
        sign = torch.sign(x_int).to(torch.int8)
        sign[sign == 0] = 1  # Handle zeros as positive
        return scale.expand(rows), x_int, sign

    num_full = rows // G
    rem = rows % G

    parts_scale = []
    parts_int = []
    parts_sign = []

    if num_full > 0:
        x_full = x[:num_full * G].reshape(num_full, G, cols)
        gabs_max = x_full.abs().amax(dim=(1, 2)).clamp(min=1e-5)  # (num_full,)
        gscale = gabs_max / q_clip  # use q_clip so dequant amplifies

        # Expand scales for broadcasting
        gscale_exp = gscale[:, None, None].expand(num_full, G, cols)
        x_full_quant = (x_full / gscale_exp).round().clamp(-q_max, q_max)
        x_full_sign = torch.sign(x_full_quant).to(torch.int8)
        x_full_sign[x_full_sign == 0] = 1

        parts_scale.append(gscale.repeat_interleave(G))
        parts_int.append(x_full_quant.reshape(num_full * G, cols))
        parts_sign.append(x_full_sign.reshape(num_full * G, cols))

    if rem > 0:
        x_rem = x[num_full * G:]
        rabs_max = x_rem.abs().max().clamp(min=1e-5)
        rscale = rabs_max / q_clip  # use q_clip so dequant amplifies
        x_rem_quant = (x_rem / rscale).round().clamp(-q_max, q_max)
        x_rem_sign = torch.sign(x_rem_quant).to(torch.int8)
        x_rem_sign[x_rem_sign == 0] = 1
        
        parts_scale.append(rscale.expand(rem))
        parts_int.append(x_rem_quant)
        parts_sign.append(x_rem_sign)

    scale_row = torch.cat(parts_scale)  # (rows,)
    x_int = torch.cat(parts_int, dim=0) # (rows, cols)
    sign = torch.cat(parts_sign, dim=0) # (rows, cols)

    return scale_row, x_int, sign


def _grouped_asymmetric_quant(x, G, q_max):
    """Per-row-group asymmetric quantization.

    Args:
        x: (rows, cols) float tensor
        G: number of rows per quantization group
        q_max: max quantized value (e.g. 255 for 8-bit)

    Returns:
        scale_row: (rows,) per-row scale
        zp_row:    (rows,) per-row zero-point
        x_int:     (rows, cols) quantized values in [0, q_max]
    """
    rows, cols = x.shape

    if G >= rows:
        # Single group (per-tensor) — fast path
        x_max = x.max()
        x_min = x.min()
        range_x = (x_max - x_min).clamp(min=1e-5)
        scale = range_x / q_max
        zp = (-x_min / scale).round().clamp(0, q_max)
        x_int = (x / scale + zp).round().clamp(0, q_max)
        return scale.expand(rows), zp.expand(rows), x_int

    num_full = rows // G
    rem = rows % G

    parts_scale = []
    parts_zp = []

    if num_full > 0:
        x_full = x[:num_full * G].reshape(num_full, G, cols)
        gmax = x_full.amax(dim=(1, 2))      # (num_full,)
        gmin = x_full.amin(dim=(1, 2))
        grange = (gmax - gmin).clamp(min=1e-5)
        gscale = grange / q_max              # (num_full,)
        gzp = (-gmin / gscale).round().clamp(0, q_max)
        parts_scale.append(gscale.repeat_interleave(G))
        parts_zp.append(gzp.repeat_interleave(G))

    if rem > 0:
        x_rem = x[num_full * G:]
        rmax = x_rem.max()
        rmin = x_rem.min()
        rrange = (rmax - rmin).clamp(min=1e-5)
        rscale = rrange / q_max
        rzp = (-rmin / rscale).round().clamp(0, q_max)
        parts_scale.append(rscale.expand(rem))
        parts_zp.append(rzp.expand(rem))

    scale_row = torch.cat(parts_scale)       # (rows,)
    zp_row = torch.cat(parts_zp)             # (rows,)
    x_int = (x / scale_row[:, None] + zp_row[:, None]).round().clamp(0, q_max)

    return scale_row, zp_row, x_int


@torch.no_grad()
def sc_matmul_grouped(
    a: torch.Tensor,
    b: torch.Tensor,
    group_a: int = 1,
    group_b: int = 1,
    mode: str = "unipolar",
    sc_prec: int = 8,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """
    SC matmul with per-row-group quantization: a @ b^T.

    Same Triton kernels as sc_matmul (single launch for SNG encoding, single
    launch for AND matmul).  Only the host-side quantization and dequantization
    use per-row-group scales/zero-points instead of per-tensor.

    Args:
        a: (N, D) left operand.
        b: (M, D) right operand.
        group_a: rows per quantization group for a (1 = per-row, N = per-tensor).
        group_b: rows per quantization group for b (1 = per-row, M = per-tensor).
        mode: only "unipolar" supported for grouped quantization.
        sc_prec: SC precision (stoc_len = 2^sc_prec).
        config: RNG/SNG config dict.

    Returns:
        (N, M) result tensor in FP.
    """
    assert a.dim() == 2 and b.dim() == 2, f"Expected 2D, got a:{a.dim()}D b:{b.dim()}D"
    assert a.shape[1] == b.shape[1], f"Inner dim mismatch: {a.shape[1]} vs {b.shape[1]}"
    assert mode == "unipolar", "Grouped quantization only supports unipolar mode"

    N, D = a.shape
    M = b.shape[0]
    stoc_len = 2 ** sc_prec

    if config is None:
        from config_helpers import make_sobol_simple_config
        config = make_sobol_simple_config(D, D, sc_prec)

    device = a.device
    if device.type != 'cuda':
        a = a.cuda()
        b = b.cuda()
    a = a.float()
    b = b.float()

    # Get cached RNG sequences
    rand_seqs_a_t, rand_seqs_b_t = _get_cached_sequences(config, sc_prec, a.device)

    result = _sc_matmul_unipolar_grouped(
        a, b, group_a, group_b, sc_prec,
        rand_seqs_a_t, rand_seqs_b_t, N, D, M, stoc_len,
    )

    if device.type != 'cuda':
        result = result.to(device)

    return result


def _sc_matmul_batched(
    a: torch.Tensor,
    b: torch.Tensor,
    max_fp_a: float,
    min_fp_a: float,
    max_fp_b: float,
    min_fp_b: float,
    mode: str,
    sc_prec: int,
    config: Optional[dict],
) -> torch.Tensor:
    """
    Batched SC matrix multiplication using CUDA streams for parallelism.

    Args:
        a: (B, N, D) tensor
        b: (B, M, D) tensor

    Returns:
        (B, N, M) tensor
    """
    B, N, D = a.shape
    M = b.shape[1]
    output = torch.empty(B, N, M, dtype=torch.float32, device=a.device)

    # Use CUDA streams for parallel execution across batch
    streams = [torch.cuda.Stream() for _ in range(B)]
    for i in range(B):
        with torch.cuda.stream(streams[i]):
            output[i] = sc_matmul(a[i], b[i], max_fp_a, min_fp_a,
                                   max_fp_b, min_fp_b, mode, sc_prec, config)
    # Sync all streams
    for s in streams:
        s.synchronize()
    return output


# =============================================================================
# Drop-in replacement API (legacy, for testing/benchmarking)
# =============================================================================

def matmul_sc_triton(
    Q_l: int,
    Q_e: int,
    K_l: int,
    K_e: int,
    config: Optional[dict] = None,
    binary_prec: str = "fp8_e4m3",
    sc_prec: int = 8,
    input_seed: Optional[int] = None,
    verbose: bool = False,
):
    """
    Perform matrix multiplication using stochastic computing on GPU.

    This is a drop-in replacement for matmul_sc() in sc.py.

    Args:
        Q_l: Token length of Q
        Q_e: Embedding dimension of Q
        K_l: Token length of K
        K_e: Embedding dimension of K
        config: RNG/SNG configuration dict with structure:
            {
                "rng_pool": [
                    {"type": "lfsr", "seed": 125, "taps": [7,5,3,0]},
                    {"type": "sobol", "seed_type": "q"},
                    ...
                ],
                "sng": {
                    "q": [{"rng_id": 0, "scramble": None}, ...],
                    "k": [{"rng_id": 1, "scramble": [7,6,5,4,3,2,1,0]}, ...],
                }
            }
        binary_prec: Precision for binary representation. Options:
            - "fp8_e4m3": FP8 E4M3 format, max=448 (default)
            - "fp8_e5m2": FP8 E5M2 format, max=57344
            - "int8": Signed 8-bit integer, max=127
        sc_prec: Precision for stochastic computing (default is 8)
        input_seed: Numpy random seed for reproducible Q/K generation
        verbose: Enable debug prints (default is False)

    Returns:
        A tuple containing:
        - QK_sc: SC-computed QK matrix (numpy array)
        - QK_actual: Actual QK matrix (ground truth, numpy array)
        - rmse: RMSE on normalized values (comparable across precisions)
    """
    assert Q_e == K_e, "Embedding dimensions must match for Q @ K^T"

    # Set seed for reproducibility
    if input_seed is not None:
        np.random.seed(input_seed)

    # Get max value based on precision
    if binary_prec == "fp8_e4m3":
        max_val = FP8_E4M3_MAX
    elif binary_prec == "fp8_e5m2":
        max_val = FP8_E5M2_MAX
    elif binary_prec == "int8":
        max_val = INT8_MAX
    else:
        raise ValueError(f"Unsupported binary precision: {binary_prec}")

    # Generate random Q and K matrices
    if binary_prec == "int8":
        Q_np = np.random.randint(-max_val, max_val + 1, size=(Q_l, Q_e)).astype(np.float32)
        K_np = np.random.randint(-max_val, max_val + 1, size=(K_l, K_e)).astype(np.float32)
    else:
        Q_np = np.random.uniform(-max_val, max_val, size=(Q_l, Q_e)).astype(np.float32)
        K_np = np.random.uniform(-max_val, max_val, size=(K_l, K_e)).astype(np.float32)

    # Compute actual matrix multiplication: Q @ K^T -> (Q_l, K_l)
    QK_actual = Q_np @ K_np.T

    # Stochastic computing parameters
    stoc_len = 2 ** sc_prec

    # Use default config if not provided
    if config is None:
        from config_helpers import make_default_config
        config = make_default_config(Q_e, K_e, sc_prec)

    # Build RNG pool and SNG banks
    rng_pool = RNGPool(config["rng_pool"], sc_prec)
    sng_q = SNGBank(rng_pool, config["sng"]["q"])
    sng_k = SNGBank(rng_pool, config["sng"]["k"])

    # Get all per-element random sequences (with scrambling already applied)
    rand_seqs_q = sng_q.get_all_sequences(stoc_len)  # (Q_e, stoc_len)
    rand_seqs_k = sng_k.get_all_sequences(stoc_len)  # (K_e, stoc_len)

    if verbose:
        print(f"Q[0,:3] = {Q_np[0, :3]}")
        print(f"K[0,:3] = {K_np[0, :3]}")
        print(f"stoc_len = {stoc_len}, max_rng_val = {2**sc_prec - 1}")
        print(f"RNG pool size: {len(rng_pool)}")
        print(f"Q SNGs: {len(sng_q)}, K SNGs: {len(sng_k)}")

    # Transfer to GPU
    device = 'cuda'
    Q = torch.from_numpy(Q_np).to(device)
    K = torch.from_numpy(K_np).to(device)
    rand_seqs_q_t = torch.tensor(rand_seqs_q, dtype=torch.int32, device=device)
    rand_seqs_k_t = torch.tensor(rand_seqs_k, dtype=torch.int32, device=device)

    # Phase 1: Convert to packed stochastic streams
    Q_packed = bin_to_stoc_packed(Q, rand_seqs_q_t, max_val, sc_prec)
    K_packed = bin_to_stoc_packed(K, rand_seqs_k_t, max_val, sc_prec)

    # Phase 2: XNOR matrix multiplication
    max_val_squared = float(max_val * max_val)
    QK_sc_t = xnor_matmul(Q_packed, K_packed, Q_l, Q_e, K_l, stoc_len, max_val_squared)

    # Transfer back to CPU
    QK_sc = QK_sc_t.cpu().numpy()

    if verbose:
        print(f"\nTriton SC result sample: {QK_sc[0, :3]}")
        print(f"Actual result sample: {QK_actual[0, :3]}")

    # Compute RMSE on normalized values (standard in SC papers)
    max_dot_product = Q_e * (max_val ** 2)
    QK_sc_norm = QK_sc / max_dot_product
    QK_actual_norm = QK_actual / max_dot_product
    rmse = np.sqrt(np.mean((QK_sc_norm - QK_actual_norm) ** 2))

    return QK_sc, QK_actual, rmse

# OUTDATED
def matmul_sc_triton_from_saved(
    Q_l: int,
    Q_e: int,
    K_l: int,
    K_e: int,
    operation: str = "matmul",
    sc_prec: int = 8,
    binary_prec: str = "int8",
    config_path: Optional[str] = None,
    input_seed: Optional[int] = None,
    verbose: bool = False,
):
    """
    Load best saved config and run SC matmul on GPU.

    Args:
        Q_l, Q_e, K_l, K_e: Matrix dimensions
        operation: Operation type for config lookup (default: "matmul")
        sc_prec: SC precision for config lookup (default: 8)
        binary_prec: Binary precision for config lookup (default: "int8")
        config_path: Direct path to config file (overrides lookup)
        input_seed: Numpy random seed for reproducibility
        verbose: Enable debug prints

    Returns:
        Tuple of (QK_sc, QK_actual, rmse) or None if no config found
    """
    from config_helpers import load_config, load_best_config

    if config_path:
        config, metadata = load_config(config_path)
    else:
        config, metadata = load_best_config(operation, sc_prec, binary_prec)

    if config is None:
        if verbose:
            print(f"No saved config found for {operation}/{sc_prec}bit/{binary_prec}")
        return None

    if verbose and metadata:
        print(f"Loaded config with error: {metadata.get('error', 'N/A')}")

    return matmul_sc_triton(
        Q_l, Q_e, K_l, K_e,
        config=config,
        binary_prec=binary_prec,
        sc_prec=sc_prec,
        input_seed=input_seed,
        verbose=verbose,
    )


# =============================================================================
# Test and Benchmark
# =============================================================================

def test_all_configs():
    """Test GPU implementation against CPU for all config types."""
    from sc import matmul_sc
    from config_helpers import (
        make_default_config, make_random_config,
        make_fully_independent_config, make_sobol_simple_config,make_sobol_dse_config,
    )

    print("Testing GPU vs CPU for all config types...")
    print("=" * 60)

    configs = [
        ("default (shared RNG + reverse)", make_default_config(64, 64, 8)),
        ("two_rng (Q->RNG0, K->RNG1)", make_random_config(64, 64, 8)),
        ("fully_independent", make_fully_independent_config(64, 64, 8)),
        ("sobol_simple", make_sobol_simple_config(64, 64, 8)),
        ("sobol_dse", make_sobol_dse_config(64, 64, 8)),
    ]

    all_passed = True
    for name, config in configs:
        # Run both with same seed
        _, _, rmse_cpu = matmul_sc(4, 64, 4, 64, config=config, input_seed=42)
        _, _, rmse_gpu = matmul_sc_triton(4, 64, 4, 64, config=config, input_seed=42)

        diff = abs(rmse_cpu - rmse_gpu)
        passed = diff < 0.01  # Allow small numerical differences
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"  {name}:")
        print(f"    CPU RMSE: {rmse_cpu:.6f}")
        print(f"    GPU RMSE: {rmse_gpu:.6f}")
        print(f"    Diff: {diff:.6f} [{status}]")

    print("=" * 60)
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def benchmark_comparison():
    """Compare performance: CPU vs GPU across config types."""
    import time
    from sc import matmul_sc
    from config_helpers import (
        make_default_config, make_random_config,
        make_fully_independent_config, make_sobol_simple_config, make_sobol_dse_config,
    )

    print("\nPerformance Benchmark: CPU vs GPU")
    print("=" * 60)

    Q_l, Q_e, K_l, K_e = 64, 64, 64, 64
    n_warmup = 3
    n_runs = 20

    configs = [
        ("default", make_default_config(Q_e, K_e, 8)),
        ("two_rng", make_random_config(Q_e, K_e, 8)),
        ("fully_independent", make_fully_independent_config(Q_e, K_e, 8)),
        ("sobol_simple", make_sobol_simple_config(Q_e, K_e, 8)),
        ("sobol_dse", make_sobol_dse_config(Q_e, K_e, 8)),
    ]

    for name, config in configs:
        # CPU warmup and timing
        for _ in range(n_warmup):
            matmul_sc(Q_l, Q_e, K_l, K_e, config=config)
        t0 = time.time()
        for _ in range(n_runs):
            matmul_sc(Q_l, Q_e, K_l, K_e, config=config)
        cpu_time = (time.time() - t0) / n_runs * 1000  # ms

        # GPU warmup and timing
        for _ in range(n_warmup):
            matmul_sc_triton(Q_l, Q_e, K_l, K_e, config=config)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_runs):
            matmul_sc_triton(Q_l, Q_e, K_l, K_e, config=config)
        torch.cuda.synchronize()
        gpu_time = (time.time() - t0) / n_runs * 1000  # ms

        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"  {name}:")
        print(f"    CPU: {cpu_time:.2f} ms")
        print(f"    GPU: {gpu_time:.2f} ms")
        print(f"    Speedup: {speedup:.1f}x")

    print("=" * 60)


def test_sc_matmul():
    """Test the new sc_matmul() FP-in/FP-out entry point."""
    from config_helpers import make_sobol_simple_config

    print("Testing sc_matmul() FP-in/FP-out entry point...")
    print("=" * 60)

    N, D, M = 8, 64, 8
    config = make_sobol_simple_config(D, D, 8)

    # --- Test bipolar mode ---
    print("\n  Bipolar mode (symmetric, XNOR):")
    torch.manual_seed(42)
    a = torch.randn(N, D, device='cuda') * 3.0  # FP values in ~ [-9, 9]
    b = torch.randn(M, D, device='cuda') * 3.0

    max_fp = max(a.abs().max().item(), b.abs().max().item())
    result_sc = sc_matmul(a, b, max_fp, -max_fp, mode="bipolar", sc_prec=8, config=config)
    result_gt = a @ b.T

    # Compute RMSE normalized by max possible dot product
    max_dot = D * max_fp * max_fp
    rmse = ((result_sc - result_gt) ** 2).mean().sqrt().item() / max_dot
    print(f"    RMSE (normalized): {rmse:.6f}")
    print(f"    SC sample:  {result_sc[0, :3].tolist()}")
    print(f"    GT sample:  {result_gt[0, :3].tolist()}")
    bipolar_pass = rmse < 0.05
    print(f"    [{('PASS' if bipolar_pass else 'FAIL')}]")

    # --- Test unipolar mode ---
    print("\n  Unipolar mode (asymmetric, AND):")
    torch.manual_seed(42)
    # Post-softmax-like values in [0, 1]
    a_uni = torch.rand(N, D, device='cuda')
    b_uni = torch.rand(M, D, device='cuda')

    result_sc_uni = sc_matmul(a_uni, b_uni, 1.0, 0.0, mode="unipolar", sc_prec=8, config=config)
    result_gt_uni = a_uni @ b_uni.T

    max_dot_uni = D * 1.0 * 1.0
    rmse_uni = ((result_sc_uni - result_gt_uni) ** 2).mean().sqrt().item() / max_dot_uni
    print(f"    RMSE (normalized): {rmse_uni:.6f}")
    print(f"    SC sample:  {result_sc_uni[0, :3].tolist()}")
    print(f"    GT sample:  {result_gt_uni[0, :3].tolist()}")
    unipolar_pass = rmse_uni < 0.05
    print(f"    [{('PASS' if unipolar_pass else 'FAIL')}]")

    # --- Test batched ---
    print("\n  Batched bipolar mode (B=4):")
    a_batch = torch.randn(4, N, D, device='cuda') * 2.0
    b_batch = torch.randn(4, M, D, device='cuda') * 2.0
    max_fp_b = max(a_batch.abs().max().item(), b_batch.abs().max().item())

    result_sc_b = sc_matmul(a_batch, b_batch, max_fp_b, -max_fp_b, mode="bipolar", sc_prec=8, config=config)
    result_gt_b = torch.bmm(a_batch, b_batch.transpose(-2, -1))

    max_dot_b = D * max_fp_b * max_fp_b
    rmse_b = ((result_sc_b - result_gt_b) ** 2).mean().sqrt().item() / max_dot_b
    print(f"    RMSE (normalized): {rmse_b:.6f}")
    batch_pass = rmse_b < 0.05
    print(f"    [{('PASS' if batch_pass else 'FAIL')}]")

    print("=" * 60)
    all_pass = bipolar_pass and unipolar_pass and batch_pass
    print(f"sc_matmul: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return all_pass


if __name__ == "__main__":
    print("Testing Triton SC Matrix Multiplication...")
    print("=" * 60)

    # Basic test (legacy API)
    print("\nBasic test (4x64 @ 4x64):")
    QK_sc, QK_actual, rmse = matmul_sc_triton(4, 64, 4, 64, verbose=True, sc_prec=8)
    print(f"\nRMSE: {rmse:.6f}")

    # Test all config types (legacy API)
    print("\n")
    test_all_configs()

    # Test new sc_matmul entry point
    print("\n")
    test_sc_matmul()

    # Benchmark
    benchmark_comparison()
