"""
SC Matrix Multiplication wrapper for Q-DiT integration.

This module provides wrapper functions for stochastic computing matrix
multiplication, adapting the SC triton kernels for use with PyTorch tensors
in the diffusion transformer.
"""

import sys
from pathlib import Path

import torch
import numpy as np
from typing import Optional, Tuple

# Make the sibling flat SC module layout (vit_sc/sc) importable.
SC_PATH = Path(__file__).resolve().parent.parent / "sc"
if str(SC_PATH) not in sys.path:
    sys.path.insert(0, str(SC_PATH))

from sc_triton import bin_to_stoc_packed, xnor_matmul  # noqa: E402
from sng import RNGPool, SNGBank  # noqa: E402
from config_helpers import make_sobol_simple_config  # noqa: E402


def sc_matmul_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    sc_prec: int = 8,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """
    Stochastic computing matrix multiplication for attention: q @ k^T

    Performs SC-based matrix multiplication suitable for attention scores.
    Input tensors should already be quantized to integer range.

    Args:
        q: Query tensor, shape (N, D) or (B, N, D), values in [-quant_max, quant_max]
        k: Key tensor, shape (M, D) or (B, M, D), values in [-quant_max, quant_max]
        sc_prec: SC precision (determines stoc_len and quant_max)
        config: Optional SC config dict. If None, uses sobol_simple config.

    Returns:
        Attention scores tensor, shape (N, M) or (B, N, M)

    Note:
        The result is scaled by quant_max^2 internally by SC, so the output
        needs to be rescaled by the caller based on the original scales.
    """
    # Handle batched input
    if q.dim() == 3:
        return _sc_matmul_qk_batched(q, k, sc_prec, config)

    assert q.dim() == 2 and k.dim() == 2, f"Expected 2D tensors, got q:{q.dim()}D, k:{k.dim()}D"
    assert q.shape[1] == k.shape[1], f"Embedding dim mismatch: q={q.shape[1]}, k={k.shape[1]}"

    N, D = q.shape
    M = k.shape[0]

    quant_max = 2 ** (sc_prec - 1) - 1
    stoc_len = 2 ** sc_prec

    # Use default config if not provided
    if config is None:
        config = make_sobol_simple_config(D, D, sc_prec)

    # Ensure tensors are on CUDA and float32 for SC kernels
    device = q.device
    if device.type != 'cuda':
        q = q.cuda()
        k = k.cuda()

    q = q.float()
    k = k.float()

    # Build RNG pool and SNG banks
    rng_pool = RNGPool(config["rng_pool"], sc_prec)
    sng_q = SNGBank(rng_pool, config["sng"]["q"])
    sng_k = SNGBank(rng_pool, config["sng"]["k"])

    # Get per-element random sequences
    rand_seqs_q = sng_q.get_all_sequences(stoc_len)  # (D, stoc_len)
    rand_seqs_k = sng_k.get_all_sequences(stoc_len)  # (D, stoc_len)

    # Transfer RNG sequences to GPU
    rand_seqs_q_t = torch.tensor(rand_seqs_q, dtype=torch.int32, device=q.device)
    rand_seqs_k_t = torch.tensor(rand_seqs_k, dtype=torch.int32, device=k.device)

    # Convert to packed stochastic streams
    Q_packed = bin_to_stoc_packed(q, rand_seqs_q_t, float(quant_max), sc_prec)
    K_packed = bin_to_stoc_packed(k, rand_seqs_k_t, float(quant_max), sc_prec)

    # XNOR matrix multiplication
    max_val_squared = float(quant_max * quant_max)
    result = xnor_matmul(Q_packed, K_packed, N, D, M, stoc_len, max_val_squared)

    # Move back to original device if needed
    if device.type != 'cuda':
        result = result.to(device)

    return result


def _sc_matmul_qk_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    sc_prec: int,
    config: Optional[dict],
) -> torch.Tensor:
    """
    Batched SC matrix multiplication.

    Args:
        q: Query tensor, shape (B, N, D)
        k: Key tensor, shape (B, M, D)
        sc_prec: SC precision
        config: SC config dict

    Returns:
        Attention scores tensor, shape (B, N, M)
    """
    B, N, D = q.shape
    M = k.shape[1]

    results = []
    for i in range(B):
        result = sc_matmul_qk(q[i], k[i], sc_prec, config)
        results.append(result)

    return torch.stack(results, dim=0)


def sc_matmul_qk_multihead(
    q: torch.Tensor,
    k: torch.Tensor,
    sc_prec: int = 8,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """
    SC matrix multiplication for multi-head attention: q @ k^T

    Handles the (B, H, N, D) shape used in multi-head attention.

    Args:
        q: Query tensor, shape (B, H, N, D), values in [-quant_max, quant_max]
        k: Key tensor, shape (B, H, N, D), values in [-quant_max, quant_max]
        sc_prec: SC precision
        config: Optional SC config dict

    Returns:
        Attention scores tensor, shape (B, H, N, N)
    """
    assert q.dim() == 4 and k.dim() == 4, f"Expected 4D tensors, got q:{q.dim()}D, k:{k.dim()}D"

    B, H, N, D = q.shape
    assert k.shape == (B, H, N, D), f"Shape mismatch: q={q.shape}, k={k.shape}"

    # Reshape to (B*H, N, D) for batched processing
    q_flat = q.reshape(B * H, N, D)
    k_flat = k.reshape(B * H, N, D)

    # Create config if not provided (only once for efficiency)
    if config is None:
        config = make_sobol_simple_config(D, D, sc_prec)

    # Batched SC matmul
    result_flat = _sc_matmul_qk_batched(q_flat, k_flat, sc_prec, config)

    # Reshape back to (B, H, N, N)
    result = result_flat.reshape(B, H, N, N)

    return result


def quantize_for_sc(
    x: torch.Tensor,
    sc_prec: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor for SC computation (per-tensor).

    Converts floating-point tensor to integer representation suitable for SC.

    Args:
        x: Input tensor (any shape)
        sc_prec: SC precision

    Returns:
        Tuple of (quantized_tensor, scale):
        - quantized_tensor: Integer values in [-quant_max, quant_max]
        - scale: Scale factor to recover original range
    """
    quant_max = 2 ** (sc_prec - 1) - 1

    # Per-tensor quantization (symmetric)
    x_absmax = x.abs().max()
    if x_absmax == 0:
        x_absmax = torch.tensor(1.0, device=x.device, dtype=x.dtype)

    scale = x_absmax / quant_max
    x_quant = (x / scale).round().clamp(-quant_max, quant_max)

    return x_quant, scale


def quantize_for_sc_per_head(
    x: torch.Tensor,
    sc_prec: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor for SC computation with per-head scaling.

    Better accuracy for multi-head attention by using separate scale per head.

    Args:
        x: Input tensor, shape (B, H, N, D)
        sc_prec: SC precision

    Returns:
        Tuple of (quantized_tensor, scale):
        - quantized_tensor: Integer values in [-quant_max, quant_max], shape (B, H, N, D)
        - scale: Scale factor per head, shape (B, H, 1, 1)
    """
    assert x.dim() == 4, f"Expected 4D tensor (B, H, N, D), got {x.dim()}D"

    quant_max = 2 ** (sc_prec - 1) - 1

    # Per-head max: compute max over N and D dimensions
    x_absmax = x.abs().amax(dim=(2, 3), keepdim=True)  # (B, H, 1, 1)
    # eg. x_absmax = 1.27
    x_absmax = x_absmax.clamp(min=1e-6)  # Avoid division by zero

    scale = x_absmax / quant_max  # (B, H, 1, 1)
    # eg. scale = 1.27/127 = 0.01
    x_quant = (x / scale).round().clamp(-quant_max, quant_max)
    # eg. x_quant = 127
    return x_quant, scale


def dequantize_sc_result(
    result: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize SC result back to original scale.

    Args:
        result: SC matmul result (already scaled by quant_max^2 internally)
        q_scale: Scale used for Q quantization
        k_scale: Scale used for K quantization

    Returns:
        Dequantized result in original scale
    """
    # SC internally computes: (q_int @ k_int^T) * quant_max^2
    # We need to rescale by: (q_scale * k_scale) / quant_max^2
    # But SC already handles the quant_max^2, so just multiply by scales
    return result * (q_scale * k_scale)
