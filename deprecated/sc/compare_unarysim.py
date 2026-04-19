"""
Comparison: UnarySim's C-BSG vs Your Independent Generation

Uses UnarySim's actual FSUMul implementation directly.
"""

import sys
import os
import numpy as np
import torch
import time

# Add paths
sys.path.insert(0, '/home/allenjin/Codes/scmp_llm/SC')
sys.path.insert(0, '/home/allenjin/Codes')  # For UnarySim

from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.kernel import FSUMul
from UnarySim.metric import ProgError

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Method 1: UnarySim's C-BSG Dot Product
# =============================================================================

def unarysim_dot_product(
    a: np.ndarray,  # (D,) values in [-1, 1] (bipolar normalized)
    b: np.ndarray,  # (D,) values in [-1, 1] (bipolar normalized)
    sc_prec: int = 8,
    rng_type: str = "sobol",
) -> float:
    """
    Use UnarySim's FSUMul to compute element-wise multiplication,
    then sum for dot product.

    FSUMul uses C-BSG for accurate multiplication.
    """
    D = len(a)
    stoc_len = 2 ** sc_prec

    hwcfg = {
        "width": sc_prec,
        "mode": "bipolar",
        "dimr": 1,
        "rng": rng_type,
        "static": True  # Use static multiplier (C-BSG)
    }
    swcfg = {
        "rtype": torch.float,
        "stype": torch.float,
        "btype": torch.float
    }

    # Convert to torch tensors
    a_t = torch.tensor(a, dtype=torch.float32).to(device)
    b_t = torch.tensor(b, dtype=torch.float32).to(device)

    # Quantize to bitwidth precision
    a_t = (a_t * (2**sc_prec)).round() / (2**sc_prec)
    b_t = (b_t * (2**sc_prec)).round() / (2**sc_prec)

    # Create FSUMul with 'a' as the static multiplier (uses C-BSG)
    dut_mul = FSUMul(a_t, hwcfg, swcfg).to(device)

    # Generate bitstream for 'b' (the streaming input)
    b_source = BinGen(b_t, hwcfg, swcfg)().to(device)
    b_rng = RNG(hwcfg, swcfg)().to(device)
    b_bsg = BSGen(b_source, b_rng, swcfg).to(device)

    # Track output with ProgError
    expected_product = a_t * b_t
    output_pe = ProgError(expected_product, {"scale": 1, "mode": "bipolar"}).to(device)

    # Run cycle-by-cycle simulation
    with torch.no_grad():
        for cycle in range(stoc_len):
            b_bits = b_bsg(torch.tensor([cycle]).to(device))  # Generate b bitstream
            out_bits = dut_mul(b_bits)  # C-BSG multiplication
            output_pe.Monitor(out_bits)

    # Get final progressive precision (decoded product per element)
    pp, pe = output_pe()
    products = pp.cpu().numpy()  # (D,) element-wise products

    # Sum for dot product (need to scale back)
    dot_product = products.sum()

    return dot_product, pe.cpu().numpy()


# =============================================================================
# Method 2: Your Independent Generation (simplified for comparison)
# =============================================================================

def independent_dot_product(
    a: np.ndarray,  # (D,) values in [-1, 1]
    b: np.ndarray,  # (D,) values in [-1, 1]
    sc_prec: int = 8,
    rng_type: str = "sobol",
) -> float:
    """
    Your approach: Generate bitstreams independently, then XNOR.
    """
    D = len(a)
    stoc_len = 2 ** sc_prec
    max_rng = stoc_len  # = 2^sc_prec (matching UnarySim convention)

    # Quantize
    a = np.round(a * (2**sc_prec)) / (2**sc_prec)
    b = np.round(b * (2**sc_prec)) / (2**sc_prec)

    # Map to probability [0, 1] for bipolar
    a_prob = (a + 1) / 2
    b_prob = (b + 1) / 2
    a_boundary = np.round(a_prob * max_rng).astype(np.int32)
    b_boundary = np.round(b_prob * max_rng).astype(np.int32)

    # Generate RNG sequences using UnarySim's RNG for fair comparison
    hwcfg_a = {"width": sc_prec, "dimr": 1, "rng": rng_type}
    hwcfg_b = {"width": sc_prec, "dimr": 2, "rng": rng_type}  # Different dimension
    swcfg = {"rtype": torch.float}

    seq_a = RNG(hwcfg_a, swcfg)().numpy().astype(np.int32)
    seq_b = RNG(hwcfg_b, swcfg)().numpy().astype(np.int32)

    # Generate bitstreams: bit = 1 if boundary > rng_val (strictly greater than)
    a_bits = (a_boundary[:, None] > seq_a[None, :]).astype(np.int32)
    b_bits = (b_boundary[:, None] > seq_b[None, :]).astype(np.int32)

    # XNOR multiplication (bipolar)
    xnor = (a_bits == b_bits).astype(np.int32)
    xnor_counts = xnor.sum(axis=1)  # (D,)

    # Decode: (2 * count / stoc_len - 1) gives product in [-1, 1]
    products = (2 * xnor_counts / stoc_len - 1)

    return products.sum(), (products - a * b)


# =============================================================================
# Method 3: Your Triton GPU Implementation
# =============================================================================

def triton_dot_product(
    a: np.ndarray,  # (D,) values in [-1, 1]
    b: np.ndarray,  # (D,) values in [-1, 1]
    sc_prec: int = 8,
) -> float:
    """Your Triton implementation."""
    try:
        from sc_triton import sc_matmul
        from config_helpers import make_sobol_simple_config

        D = len(a)
        config = make_sobol_simple_config(D, D, sc_prec)

        a_t = torch.tensor(a, dtype=torch.float32, device='cuda').unsqueeze(0)
        b_t = torch.tensor(b, dtype=torch.float32, device='cuda').unsqueeze(0)

        # sc_matmul expects actual values, not normalized
        # For normalized [-1,1] values, max_val = 1.0
        result = sc_matmul(a_t, b_t, 1.0, -1.0, 1.0, -1.0,
                          mode="bipolar", sc_prec=sc_prec, config=config)

        return result[0, 0].item()
    except Exception as e:
        print(f"Triton error: {e}")
        return None


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(
    D: int = 64,
    sc_prec: int = 8,
    n_trials: int = 50,
    rng_type: str = "sobol",
):
    """Compare UnarySim C-BSG vs Your approach."""

    print(f"\n{'='*70}")
    print(f"Comparison: UnarySim C-BSG vs Independent Generation")
    print(f"{'='*70}")
    print(f"D={D}, sc_prec={sc_prec}, stoc_len={2**sc_prec}, rng={rng_type}")
    print(f"Trials: {n_trials}")
    print(f"{'='*70}\n")

    results = {
        "unarysim": {"errors": [], "time": 0},
        "independent": {"errors": [], "time": 0},
        "triton": {"errors": [], "time": 0},
    }

    np.random.seed(42)

    for trial in range(n_trials):
        # Generate random vectors in [-1, 1] (bipolar normalized)
        a = np.random.uniform(-1, 1, size=D).astype(np.float32)
        b = np.random.uniform(-1, 1, size=D).astype(np.float32)

        # Ground truth dot product
        gt = np.dot(a, b)

        # Method 1: UnarySim C-BSG
        t0 = time.time()
        result_unarysim, pe_unarysim = unarysim_dot_product(a, b, sc_prec, rng_type)
        results["unarysim"]["time"] += time.time() - t0
        results["unarysim"]["errors"].append(result_unarysim - gt)

        # Method 2: Independent Generation
        t0 = time.time()
        result_ind, pe_ind = independent_dot_product(a, b, sc_prec, rng_type)
        results["independent"]["time"] += time.time() - t0
        results["independent"]["errors"].append(result_ind - gt)

        # Method 3: Triton (first few trials only for speed)
        if trial < 5:
            t0 = time.time()
            result_triton = triton_dot_product(a, b, sc_prec)
            if result_triton is not None:
                results["triton"]["time"] += time.time() - t0
                results["triton"]["errors"].append(result_triton - gt)

        if trial % 10 == 0:
            print(f"Trial {trial}: GT={gt:.4f}, UnarySim={result_unarysim:.4f}, "
                  f"Ind={result_ind:.4f}")

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")

    # Max possible dot product for normalization (D elements, each product in [-1,1])
    max_dot = D

    for method, name in [("unarysim", "UnarySim C-BSG"),
                         ("independent", "Independent Gen")]:
        errors = np.array(results[method]["errors"])
        rmse = np.sqrt(np.mean(errors ** 2))
        rmse_norm = rmse / max_dot
        bias = np.mean(errors)
        max_err = np.max(np.abs(errors))
        avg_time = results[method]["time"] / n_trials * 1000

        print(f"{name}:")
        print(f"  RMSE:           {rmse:.6f} (normalized: {rmse_norm:.6f})")
        print(f"  Bias:           {bias:.6f}")
        print(f"  Max Error:      {max_err:.6f}")
        print(f"  Time per trial: {avg_time:.2f} ms")
        print()

    if results["triton"]["errors"]:
        errors = np.array(results["triton"]["errors"])
        rmse = np.sqrt(np.mean(errors ** 2))
        avg_time = results["triton"]["time"] / len(results["triton"]["errors"]) * 1000
        print(f"Triton GPU ({len(results['triton']['errors'])} trials):")
        print(f"  RMSE:           {rmse:.6f}")
        print(f"  Time per trial: {avg_time:.2f} ms")
        print()

    # Speed comparison
    print(f"{'='*70}")
    print("SPEED COMPARISON")
    print(f"{'='*70}")
    unarysim_time = results["unarysim"]["time"] / n_trials * 1000
    ind_time = results["independent"]["time"] / n_trials * 1000
    print(f"UnarySim C-BSG:  {unarysim_time:.2f} ms (cycle-accurate)")
    print(f"Independent Gen: {ind_time:.2f} ms (vectorized)")
    print(f"Speedup (Ind over UnarySim): {unarysim_time / ind_time:.1f}x")

    if results["triton"]["errors"]:
        triton_time = results["triton"]["time"] / len(results["triton"]["errors"]) * 1000
        print(f"Triton GPU:      {triton_time:.2f} ms")
        print(f"Speedup (Triton over UnarySim): {unarysim_time / triton_time:.1f}x")


def run_precision_sweep():
    """Compare accuracy across precisions."""
    print("\n" + "="*70)
    print("PRECISION SWEEP")
    print("="*70 + "\n")

    D = 64
    n_trials = 30

    print(f"{'Prec':<6} {'UnarySim RMSE':<16} {'Indep RMSE':<16} {'UnarySim Time':<14} {'Indep Time':<12}")
    print("-" * 70)

    for prec in [4, 6, 8]:
        np.random.seed(42)

        unarysim_errors = []
        ind_errors = []
        unarysim_time = 0
        ind_time = 0

        for trial in range(n_trials):
            a = np.random.uniform(-1, 1, size=D).astype(np.float32)
            b = np.random.uniform(-1, 1, size=D).astype(np.float32)
            gt = np.dot(a, b)

            t0 = time.time()
            result_u, _ = unarysim_dot_product(a, b, prec, "sobol")
            unarysim_time += time.time() - t0
            unarysim_errors.append(result_u - gt)

            t0 = time.time()
            result_i, _ = independent_dot_product(a, b, prec, "sobol")
            ind_time += time.time() - t0
            ind_errors.append(result_i - gt)

        u_rmse = np.sqrt(np.mean(np.array(unarysim_errors)**2))
        i_rmse = np.sqrt(np.mean(np.array(ind_errors)**2))
        u_time = unarysim_time / n_trials * 1000
        i_time = ind_time / n_trials * 1000

        print(f"{prec:<6} {u_rmse:<16.6f} {i_rmse:<16.6f} {u_time:<14.2f} {i_time:<12.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--sc_prec", type=int, default=8)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--rng", type=str, default="sobol")
    parser.add_argument("--sweep", action="store_true")

    args = parser.parse_args()

    if args.sweep:
        run_precision_sweep()
    else:
        run_experiment(args.D, args.sc_prec, args.n_trials, args.rng)
