"""Stochastic Computing core operations."""
from __future__ import annotations

from typing import Optional

import numpy as np
from sng import RNGPool, SNGBank

# Max absolute values for different precisions
FP8_E4M3_MAX = 448   # FP8 E4M3 (OCP standard)
FP8_E5M2_MAX = 57344  # FP8 E5M2
INT8_MAX = 127        # Signed 8-bit integer (symmetric)


def matmul_sc(Q_l: int, Q_e: int, K_l: int, K_e: int,
              config: Optional[dict] = None,
              binary_prec: str = "fp8_e4m3",
              sc_prec: int = 8,
              input_seed: Optional[int] = None,
              verbose: bool = False):
    """
    Perform stochastic computing matrix multiplication for QK matmul.

    Args:
        Q_l: Token length of Q
        Q_e: Embedding dimension of Q
        K_l: Token length of K
        K_e: Embedding dimension of K
        config: RNG/SNG configuration dict with structure:
            {
                "rng_pool": [
                    {"type": "lfsr", "seed": 125, "taps": [7,5,3,0]},
                    ...
                ],
                "sng": {
                    "q": [{"rng_id": 0, "scramble": None}, ...],  # One per Q_e
                    "k": [{"rng_id": 0, "scramble": [7,6,5,4,3,2,1,0]}, ...],  # One per K_e
                }
            }
        binary_prec: Precision for binary representation. Options:
            - "fp8_e4m3": FP8 E4M3 format, max=448 (default)
            - "fp8_e5m2": FP8 E5M2 format, max=57344
            - "int8": Signed 8-bit integer, max=127
        sc_prec: Precision for stochastic computing (default is 8)
        input_seed: Numpy random seed for reproducible Q/K generation (default None)
        verbose: Enable debug prints (default is False)

    Returns:
        A tuple containing:
        - QK_sc: SC-computed QK matrix
        - QK_actual: Actual QK matrix (ground truth)
        - rmse: RMSE on normalized values (comparable across precisions)
    """
    assert Q_e == K_e, "Embedding dimensions must match for Q @ K^T"

    # Set seed for reproducibility if provided
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
        # For INT8, generate integers
        Q = np.random.randint(-max_val, max_val + 1, size=(Q_l, Q_e)).astype(np.float64)
        K = np.random.randint(-max_val, max_val + 1, size=(K_l, K_e)).astype(np.float64)
    else:
        # For floating point formats, generate uniform floats
        Q = np.random.uniform(-max_val, max_val, size=(Q_l, Q_e))
        K = np.random.uniform(-max_val, max_val, size=(K_l, K_e))

    # Compute actual matrix multiplication: Q @ K^T -> (Q_l, K_l)
    QK_actual = Q @ K.T

    # Stochastic computing version
    stoc_len = 2 ** sc_prec

    # Use default config if not provided
    if config is None:
        from config_helpers import make_default_config
        config = make_default_config(Q_e, K_e, sc_prec)

    # Build RNG pool and SNG banks
    rng_pool = RNGPool(config["rng_pool"], sc_prec)
    sng_q = SNGBank(rng_pool, config["sng"]["q"])
    sng_k = SNGBank(rng_pool, config["sng"]["k"])

    # Get all random sequences: shape (n_elements, stoc_len)
    rand_seqs_q = sng_q.get_all_sequences(stoc_len)
    rand_seqs_k = sng_k.get_all_sequences(stoc_len)

    # Debug prints
    if verbose:
        print(f"Q[0,:3] = {Q[0, :3]}")
        print(f"K[0,:3] = {K[0, :3]}")
        print(f"stoc_len = {stoc_len}, max_rng_val = {2**sc_prec}")
        print(f"RNG pool size: {len(rng_pool)}")
        print(f"Q SNGs: {len(sng_q)}, K SNGs: {len(sng_k)}")

    # Vectorized bin_to_stoc for entire matrices
    max_rng_val = 2 ** sc_prec

    # Compute boundaries for all elements
    Q_prob = (Q / max_val + 1) / 2  # (Q_l, Q_e), values in [0, 1]
    K_prob = (K / max_val + 1) / 2  # (K_l, K_e), values in [0, 1]
    Q_boundary = np.round(Q_prob * max_rng_val).astype(np.int32)  # (Q_l, Q_e)
    K_boundary = np.round(K_prob * max_rng_val).astype(np.int32)  # (K_l, K_e)

    # Generate stochastic streams using per-element sequences
    # rand_seqs_q: (Q_e, stoc_len), Q_boundary: (Q_l, Q_e)
    # Q_stoc[i, k, t] = 1 if Q_boundary[i, k] > rand_seqs_q[k, t] else 0
    Q_stoc = (Q_boundary[:, :, None] > rand_seqs_q[None, :, :]).astype(np.int8)  # (Q_l, Q_e, stoc_len)
    K_stoc = (K_boundary[:, :, None] > rand_seqs_k[None, :, :]).astype(np.int8)  # (K_l, K_e, stoc_len)

    # Vectorized XNOR and matmul
    # XNOR: result[i, j, k, t] = 1 if Q_stoc[i, k, t] == K_stoc[j, k, t] else 0
    xnor_result = (Q_stoc[:, None, :, :] == K_stoc[None, :, :, :]).astype(np.int8)  # (Q_l, K_l, Q_e, stoc_len)

    # Count ones per product stream
    xnor_ones = xnor_result.sum(axis=-1)  # (Q_l, K_l, Q_e)

    # Decode each product
    prod_max_val = max_val * max_val
    prob_1 = xnor_ones / stoc_len
    decoded_prods = ((2 * prob_1 - 1) * prod_max_val).astype(np.int32)  # (Q_l, K_l, Q_e)

    # Sum over embedding dimension
    QK_sc = decoded_prods.sum(axis=-1).astype(np.float64)  # (Q_l, K_l)

    # Debug for first element
    if verbose:
        q_val, k_val = Q[0, 0], K[0, 0]
        expected_prod = q_val * k_val
        q_ones = Q_stoc[0, 0, :].sum()
        k_ones = K_stoc[0, 0, :].sum()
        xnor_ones_00 = xnor_ones[0, 0, 0]
        decoded_prod = decoded_prods[0, 0, 0]

        q_prob1 = ((q_val / max_val) + 1) / 2
        k_prob1 = ((k_val / max_val) + 1) / 2
        expected_xnor = q_prob1 * k_prob1 + (1 - q_prob1) * (1 - k_prob1)

        print(f"\nQ[0,0]={q_val:.2f}, K[0,0]={k_val:.2f}")
        print(f"Q stream ones: {q_ones}/{stoc_len} = {q_ones/stoc_len:.3f} (expected {q_prob1:.3f})")
        print(f"K stream ones: {k_ones}/{stoc_len} = {k_ones/stoc_len:.3f} (expected {k_prob1:.3f})")
        print(f"Expected product: {expected_prod:.2f}")
        print(f"XNOR ones: {xnor_ones_00}/{stoc_len} = {xnor_ones_00/stoc_len:.3f} (expected {expected_xnor:.3f})")
        print(f"Decoded product: {decoded_prod}")

    # Compute RMSE on normalized values (standard in SC papers)
    # Normalize to [-1, 1] range for comparable error across precisions
    # Max possible value of dot product: Q_e * max_val^2
    max_dot_product = Q_e * (max_val ** 2)
    QK_sc_norm = QK_sc / max_dot_product
    QK_actual_norm = QK_actual / max_dot_product
    rmse = np.sqrt(np.mean((QK_sc_norm - QK_actual_norm) ** 2))

    return QK_sc, QK_actual, rmse


if __name__ == "__main__":
    from config_helpers import make_default_config, make_independent_config, make_sobol_simple_config

    print("Testing with default config (shared RNG, K uses reverse scramble):")
    config = make_default_config(64, 64, 8)
    QK_sc, QK_actual, rmse = matmul_sc(4, 64, 4, 64, config=config, verbose=True)
    print(f"RMSE: {rmse:.6f}\n")

    print("Testing with independent config (different taps for Q and K):")
    config = make_independent_config(64, 64, 8,
                                      q_taps=[7, 5, 3, 0], k_taps=[7, 6, 5, 2],
                                      q_seed=125, k_seed=157)
    QK_sc, QK_actual, rmse = matmul_sc(4, 64, 4, 64, config=config, verbose=False)
    print(f"RMSE: {rmse:.6f}")


    config = make_sobol_simple_config(64, 64, 8)
    QK_sc, QK_actual, rmse = matmul_sc(4, 64, 4, 64, config=config, verbose=False)
    print(f"RMSE: {rmse:.6f}")
