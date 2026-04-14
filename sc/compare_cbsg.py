"""
Comparison experiment: C-BSG (UnarySim) vs Independent Generation (Your approach)

This script compares:
1. Accuracy (RMSE, max error, bias)
2. Simulation speed
3. Correlation (SCC) between bitstreams

Based on the uSystolic paper's claim that C-BSG achieves SCC ≈ 0 for more accurate multiplication.
"""

import sys
import os
import numpy as np
import torch
import time
from typing import Tuple, Optional

# Add paths
sys.path.insert(0, '/home/allenjin/Codes/scmp_llm/SC')
sys.path.insert(0, '/home/allenjin/Codes')

# =============================================================================
# Method 1: Your Independent Generation Approach
# =============================================================================

def independent_gen_dot_product(
    a: np.ndarray,  # (D,) values in [-max_val, max_val]
    b: np.ndarray,  # (D,) values in [-max_val, max_val]
    max_val: float,
    sc_prec: int = 8,
    rng_type: str = "sobol",
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Your approach: Generate bitstreams independently, then XNOR.

    Returns:
        result: SC-computed dot product
        a_bits: (D, stoc_len) bitstream for a
        b_bits: (D, stoc_len) bitstream for b
    """
    D = len(a)
    stoc_len = 2 ** sc_prec
    max_rng = stoc_len  # = 2^sc_prec (matching UnarySim convention)

    # Map to probability [0, 1] for bipolar
    a_prob = (a / max_val + 1) / 2  # [-1,1] -> [0,1]
    b_prob = (b / max_val + 1) / 2

    # Quantize to integer boundaries
    a_boundary = np.round(a_prob * max_rng).astype(np.int32)
    b_boundary = np.round(b_prob * max_rng).astype(np.int32)

    # Generate RNG sequences
    if rng_type == "sobol":
        # Use different Sobol dimensions for a and b
        from rng import Sobol
        rng_a = Sobol(sc_prec, seed_type="q")
        rng_b = Sobol(sc_prec, seed_type="k")
        seq_a = rng_a.simulate(stoc_len)  # (stoc_len,)
        seq_b = rng_b.simulate(stoc_len)
    elif rng_type == "lfsr":
        from rng import LFSR
        from lfsr_taps import get_lfsr_taps
        taps = get_lfsr_taps(sc_prec)
        rng_a = LFSR(seed=1, length=sc_prec, taps=taps)
        rng_b = LFSR(seed=127, length=sc_prec, taps=taps)  # Different seed
        seq_a = rng_a.simulate(stoc_len)
        seq_b = rng_b.simulate(stoc_len)
    else:
        # System random
        np.random.seed(42)
        seq_a = np.random.permutation(stoc_len).astype(np.int32)
        np.random.seed(123)
        seq_b = np.random.permutation(stoc_len).astype(np.int32)

    # Generate bitstreams: bit = 1 if boundary > rng_val (strictly greater than)
    # Shape: (D, stoc_len)
    a_bits = (a_boundary[:, None] > seq_a[None, :]).astype(np.int32)
    b_bits = (b_boundary[:, None] > seq_b[None, :]).astype(np.int32)

    # XNOR multiplication (bipolar)
    xnor = (a_bits == b_bits).astype(np.int32)  # (D, stoc_len)

    # Sum over stoc_len to get counts, then decode
    xnor_counts = xnor.sum(axis=1)  # (D,)

    # Decode each element: (2 * count / stoc_len - 1) * max_val^2
    decoded = (2 * xnor_counts / stoc_len - 1) * (max_val ** 2)

    # Sum over D for dot product
    result = decoded.sum()

    return result, a_bits, b_bits


# =============================================================================
# Method 2: C-BSG (Conditional Bitstream Generation) - UnarySim style
# =============================================================================

def cbsg_dot_product(
    a: np.ndarray,  # (D,) values in [-max_val, max_val]
    b: np.ndarray,  # (D,) values in [-max_val, max_val]
    max_val: float,
    sc_prec: int = 8,
    rng_type: str = "sobol",
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    C-BSG approach: One bitstream enables RNG advancement for the other.

    Based on uSystolic/UnarySim FSUMul implementation.
    'a' bitstream is generated normally.
    'b' bitstream uses C-BSG: RNG advances only when a_bit = 1.

    For bipolar, we use the two-path approach:
    - Path 1: a=1 AND b_generated
    - Path 2: a=0 AND (1 - b_generated_inv)

    Returns:
        result: SC-computed dot product
        a_bits: (D, stoc_len) bitstream for a
        b_bits: (D, stoc_len) bitstream for b (C-BSG generated)
    """
    D = len(a)
    stoc_len = 2 ** sc_prec
    max_rng = stoc_len - 1

    # Map to probability [0, 1] for bipolar
    a_prob = (a / max_val + 1) / 2
    b_prob = (b / max_val + 1) / 2

    # Quantize
    a_boundary = np.round(a_prob * max_rng).astype(np.int32)
    b_boundary = np.round(b_prob * max_rng).astype(np.int32)

    # Generate RNG sequence for a (standard)
    if rng_type == "sobol":
        from rng import Sobol
        rng_a = Sobol(sc_prec, seed_type="q")
        seq_a = rng_a.simulate(stoc_len)
        # For b, we'll use a separate sequence but with C-BSG logic
        rng_b = Sobol(sc_prec, seed_type="k")
        seq_b_full = rng_b.simulate(stoc_len * 2)  # Extra for wraparound
    elif rng_type == "lfsr":
        from rng import LFSR
        from lfsr_taps import get_lfsr_taps
        taps = get_lfsr_taps(sc_prec)
        rng_a = LFSR(seed=1, length=sc_prec, taps=taps)
        seq_a = rng_a.simulate(stoc_len)
        rng_b = LFSR(seed=127, length=sc_prec, taps=taps)
        seq_b_full = rng_b.simulate(stoc_len * 2)
    else:
        np.random.seed(42)
        seq_a = np.random.permutation(stoc_len).astype(np.int32)
        np.random.seed(123)
        seq_b_full = np.tile(np.random.permutation(stoc_len), 2).astype(np.int32)

    # Generate a bitstream (standard)
    a_bits = (a_boundary[:, None] >= seq_a[None, :]).astype(np.int32)  # (D, stoc_len)

    # C-BSG for b: simulate cycle-by-cycle with conditional RNG advancement
    # This is the key difference from independent generation
    b_bits = np.zeros((D, stoc_len), dtype=np.int32)

    for d in range(D):
        rng_idx_i1 = 0  # RNG index for path 1 (a=1)
        rng_idx_i0 = 0  # RNG index for path 2 (a=0)

        for cycle in range(stoc_len):
            a_bit = a_bits[d, cycle]

            if a_bit == 1:
                # Path 1: generate b bit, advance RNG
                b_bit = 1 if b_boundary[d] >= seq_b_full[rng_idx_i1 % len(seq_b_full)] else 0
                rng_idx_i1 += 1
                b_bits[d, cycle] = b_bit
            else:
                # Path 2: generate inverted b bit for (1-a)(1-b), advance RNG
                b_bit_inv = 1 if b_boundary[d] >= seq_b_full[rng_idx_i0 % len(seq_b_full)] else 0
                rng_idx_i0 += 1
                b_bits[d, cycle] = 1 - b_bit_inv  # Store inverted for later XNOR

    # XNOR for bipolar (now with C-BSG correlation management)
    xnor = (a_bits == b_bits).astype(np.int32)
    xnor_counts = xnor.sum(axis=1)
    decoded = (2 * xnor_counts / stoc_len - 1) * (max_val ** 2)
    result = decoded.sum()

    return result, a_bits, b_bits


# =============================================================================
# Method 3: Your Triton Implementation (for speed comparison)
# =============================================================================

def triton_dot_product(
    a: np.ndarray,
    b: np.ndarray,
    max_val: float,
    sc_prec: int = 8,
) -> float:
    """
    Your Triton-accelerated implementation.
    """
    try:
        from sc_triton import sc_matmul
        from config_helpers import make_sobol_simple_config

        D = len(a)
        config = make_sobol_simple_config(D, D, sc_prec)

        # Convert to torch tensors and reshape for matmul
        a_t = torch.tensor(a, dtype=torch.float32, device='cuda').unsqueeze(0)  # (1, D)
        b_t = torch.tensor(b, dtype=torch.float32, device='cuda').unsqueeze(0)  # (1, D)

        result = sc_matmul(a_t, b_t, max_val, -max_val, max_val, -max_val,
                          mode="bipolar", sc_prec=sc_prec, config=config)

        return result[0, 0].item()
    except Exception as e:
        print(f"Triton not available: {e}")
        return None


# =============================================================================
# Correlation (SCC) Measurement
# =============================================================================

def compute_scc(a_bits: np.ndarray, b_bits: np.ndarray) -> float:
    """
    Compute Stochastic Cross Correlation (SCC) between two bitstreams.

    SCC = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))
    where:
        a = count(a=1, b=1)
        b = count(a=1, b=0)
        c = count(a=0, b=1)
        d = count(a=0, b=0)

    Returns average SCC across all D elements.
    """
    D, stoc_len = a_bits.shape
    sccs = []

    for d in range(D):
        a_bit = a_bits[d]
        b_bit = b_bits[d]

        # Count pairs
        a_cnt = np.sum((a_bit == 1) & (b_bit == 1))  # (1,1)
        b_cnt = np.sum((a_bit == 1) & (b_bit == 0))  # (1,0)
        c_cnt = np.sum((a_bit == 0) & (b_bit == 1))  # (0,1)
        d_cnt = np.sum((a_bit == 0) & (b_bit == 0))  # (0,0)

        # SCC formula
        ad_bc = a_cnt * d_cnt - b_cnt * c_cnt
        denom = np.sqrt((a_cnt + b_cnt) * (c_cnt + d_cnt) *
                        (a_cnt + c_cnt) * (b_cnt + d_cnt) + 1e-10)

        if denom > 1e-10:
            scc = ad_bc / denom
        else:
            scc = 0.0

        sccs.append(scc)

    return np.mean(sccs), np.std(sccs)


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(
    D: int = 64,
    max_val: float = 127.0,
    sc_prec: int = 8,
    n_trials: int = 100,
    rng_type: str = "sobol",
    verbose: bool = True,
):
    """
    Run comparison experiment.

    Args:
        D: Vector dimension
        max_val: Maximum absolute value (like INT8 range)
        sc_prec: SC precision (stoc_len = 2^sc_prec)
        n_trials: Number of random vector pairs to test
        rng_type: "sobol", "lfsr", or "random"
    """
    print(f"\n{'='*70}")
    print(f"Comparison: C-BSG (UnarySim) vs Independent Generation (Your Approach)")
    print(f"{'='*70}")
    print(f"Parameters: D={D}, max_val={max_val}, sc_prec={sc_prec}, stoc_len={2**sc_prec}")
    print(f"RNG type: {rng_type}, Trials: {n_trials}")
    print(f"{'='*70}\n")

    # Storage for results
    results = {
        "independent": {"errors": [], "scc_mean": [], "scc_std": [], "time": 0},
        "cbsg": {"errors": [], "scc_mean": [], "scc_std": [], "time": 0},
        "triton": {"errors": [], "time": 0},
    }

    np.random.seed(42)

    for trial in range(n_trials):
        # Generate random vectors
        a = np.random.uniform(-max_val, max_val, size=D).astype(np.float32)
        b = np.random.uniform(-max_val, max_val, size=D).astype(np.float32)

        # Ground truth
        gt = np.dot(a, b)

        # Method 1: Independent Generation (Your approach)
        t0 = time.time()
        result_ind, a_bits_ind, b_bits_ind = independent_gen_dot_product(
            a, b, max_val, sc_prec, rng_type)
        results["independent"]["time"] += time.time() - t0
        results["independent"]["errors"].append(result_ind - gt)
        scc_mean, scc_std = compute_scc(a_bits_ind, b_bits_ind)
        results["independent"]["scc_mean"].append(scc_mean)
        results["independent"]["scc_std"].append(scc_std)

        # Method 2: C-BSG (UnarySim approach)
        t0 = time.time()
        result_cbsg, a_bits_cbsg, b_bits_cbsg = cbsg_dot_product(
            a, b, max_val, sc_prec, rng_type)
        results["cbsg"]["time"] += time.time() - t0
        results["cbsg"]["errors"].append(result_cbsg - gt)
        scc_mean, scc_std = compute_scc(a_bits_cbsg, b_bits_cbsg)
        results["cbsg"]["scc_mean"].append(scc_mean)
        results["cbsg"]["scc_std"].append(scc_std)

        # Method 3: Triton (speed only, first few trials)
        if trial < 10:
            t0 = time.time()
            result_triton = triton_dot_product(a, b, max_val, sc_prec)
            if result_triton is not None:
                results["triton"]["time"] += time.time() - t0
                results["triton"]["errors"].append(result_triton - gt)

        if verbose and trial % 20 == 0:
            print(f"Trial {trial}: GT={gt:.2f}, Ind={result_ind:.2f}, C-BSG={result_cbsg:.2f}")

    # Compute statistics
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")

    # Normalize errors by max possible dot product for fair comparison
    max_dot = D * max_val * max_val

    for method in ["independent", "cbsg"]:
        errors = np.array(results[method]["errors"])
        errors_norm = errors / max_dot

        rmse = np.sqrt(np.mean(errors ** 2))
        rmse_norm = np.sqrt(np.mean(errors_norm ** 2))
        bias = np.mean(errors)
        max_err = np.max(np.abs(errors))
        std_err = np.std(errors)

        avg_time = results[method]["time"] / n_trials * 1000  # ms

        avg_scc = np.mean(results[method]["scc_mean"])
        std_scc = np.mean(results[method]["scc_std"])

        method_name = "Independent Gen" if method == "independent" else "C-BSG (UnarySim)"
        print(f"{method_name}:")
        print(f"  RMSE:           {rmse:.4f} (normalized: {rmse_norm:.6f})")
        print(f"  Bias:           {bias:.4f}")
        print(f"  Max Error:      {max_err:.4f}")
        print(f"  Std Error:      {std_err:.4f}")
        print(f"  Avg SCC:        {avg_scc:.6f} (std: {std_scc:.6f})")
        print(f"  Time per trial: {avg_time:.3f} ms")
        print()

    if len(results["triton"]["errors"]) > 0:
        errors = np.array(results["triton"]["errors"])
        errors_norm = errors / max_dot
        rmse = np.sqrt(np.mean(errors ** 2))
        rmse_norm = np.sqrt(np.mean(errors_norm ** 2))
        avg_time = results["triton"]["time"] / len(results["triton"]["errors"]) * 1000

        print(f"Triton (GPU, {len(results['triton']['errors'])} trials):")
        print(f"  RMSE:           {rmse:.4f} (normalized: {rmse_norm:.6f})")
        print(f"  Time per trial: {avg_time:.3f} ms")
        print()

    # Speedup calculation
    print(f"{'='*70}")
    print("SPEED COMPARISON")
    print(f"{'='*70}")
    cbsg_time = results["cbsg"]["time"] / n_trials * 1000
    ind_time = results["independent"]["time"] / n_trials * 1000
    print(f"Independent Gen: {ind_time:.3f} ms")
    print(f"C-BSG:           {cbsg_time:.3f} ms")
    print(f"Speedup (Ind over C-BSG): {cbsg_time / ind_time:.2f}x")

    if len(results["triton"]["errors"]) > 0:
        triton_time = results["triton"]["time"] / len(results["triton"]["errors"]) * 1000
        print(f"Triton GPU:      {triton_time:.3f} ms")
        print(f"Speedup (Triton over C-BSG): {cbsg_time / triton_time:.1f}x")

    return results


def run_precision_sweep():
    """Run experiment across different SC precisions."""
    print("\n" + "="*70)
    print("PRECISION SWEEP: Accuracy vs SC Precision")
    print("="*70 + "\n")

    precisions = [4, 6, 8, 10]
    D = 64
    max_val = 127.0
    n_trials = 50

    print(f"{'Precision':<10} {'Ind RMSE':<15} {'C-BSG RMSE':<15} {'Ind SCC':<12} {'C-BSG SCC':<12}")
    print("-" * 70)

    for prec in precisions:
        results = run_experiment(D=D, max_val=max_val, sc_prec=prec,
                                 n_trials=n_trials, verbose=False)

        max_dot = D * max_val * max_val

        ind_rmse = np.sqrt(np.mean(np.array(results["independent"]["errors"])**2)) / max_dot
        cbsg_rmse = np.sqrt(np.mean(np.array(results["cbsg"]["errors"])**2)) / max_dot
        ind_scc = np.mean(results["independent"]["scc_mean"])
        cbsg_scc = np.mean(results["cbsg"]["scc_mean"])

        print(f"{prec:<10} {ind_rmse:<15.6f} {cbsg_rmse:<15.6f} {ind_scc:<12.6f} {cbsg_scc:<12.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare C-BSG vs Independent Generation")
    parser.add_argument("--D", type=int, default=64, help="Vector dimension")
    parser.add_argument("--max_val", type=float, default=127.0, help="Max absolute value")
    parser.add_argument("--sc_prec", type=int, default=8, help="SC precision")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--rng", type=str, default="sobol", choices=["sobol", "lfsr", "random"])
    parser.add_argument("--sweep", action="store_true", help="Run precision sweep")

    args = parser.parse_args()

    if args.sweep:
        run_precision_sweep()
    else:
        run_experiment(
            D=args.D,
            max_val=args.max_val,
            sc_prec=args.sc_prec,
            n_trials=args.n_trials,
            rng_type=args.rng,
        )
