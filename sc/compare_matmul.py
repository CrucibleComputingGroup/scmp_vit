"""
Comparison: UnarySim vs Your Triton for actual matrix multiplication.

Tests on realistic attention-sized operations: (N, D) @ (M, D)^T
"""

import sys
import os
import numpy as np
import torch
import time

sys.path.insert(0, '/home/allenjin/Codes/scmp_llm/SC')
sys.path.insert(0, '/home/allenjin/Codes')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def triton_matmul(Q: np.ndarray, K: np.ndarray, sc_prec: int = 8) -> np.ndarray:
    """Your Triton SC matmul: Q @ K^T"""
    from sc_triton import sc_matmul
    from config_helpers import make_sobol_simple_config

    N, D = Q.shape
    M = K.shape[0]
    config = make_sobol_simple_config(D, D, sc_prec)

    Q_t = torch.tensor(Q, dtype=torch.float32, device='cuda')
    K_t = torch.tensor(K, dtype=torch.float32, device='cuda')

    max_val = max(np.abs(Q).max(), np.abs(K).max())

    result = sc_matmul(Q_t, K_t, max_val, -max_val, max_val, -max_val,
                      mode="bipolar", sc_prec=sc_prec, config=config)
    return result.cpu().numpy()


def unarysim_matmul(Q: np.ndarray, K: np.ndarray, sc_prec: int = 8) -> np.ndarray:
    """
    UnarySim FSULinear for matrix multiplication.

    This simulates Q @ K^T row by row using FSULinear (which uses C-BSG).
    Note: This is slow because FSULinear is designed for neural network inference,
    not for fast simulation.
    """
    from UnarySim.stream import RNG, BinGen, BSGen
    from UnarySim.kernel import FSUMul
    from UnarySim.metric import ProgError

    N, D = Q.shape
    M = K.shape[0]
    stoc_len = 2 ** sc_prec

    # Normalize to [-1, 1]
    max_val = max(np.abs(Q).max(), np.abs(K).max())
    Q_norm = Q / max_val
    K_norm = K / max_val

    hwcfg = {
        "width": sc_prec,
        "mode": "bipolar",
        "dimr": 1,
        "rng": "sobol",
        "static": True
    }
    swcfg = {"rtype": torch.float, "stype": torch.float, "btype": torch.float}

    result = np.zeros((N, M), dtype=np.float32)

    # For each (i, j) output, compute dot product Q[i] · K[j]
    for i in range(N):
        for j in range(M):
            q_vec = torch.tensor(Q_norm[i], dtype=torch.float32).to(device)
            k_vec = torch.tensor(K_norm[j], dtype=torch.float32).to(device)

            # Quantize
            q_vec = (q_vec * (2**sc_prec)).round() / (2**sc_prec)
            k_vec = (k_vec * (2**sc_prec)).round() / (2**sc_prec)

            # Use FSUMul with q as static multiplier
            dut_mul = FSUMul(q_vec, hwcfg, swcfg).to(device)

            # Generate bitstream for k
            k_source = BinGen(k_vec, hwcfg, swcfg)().to(device)
            k_rng = RNG(hwcfg, swcfg)().to(device)
            k_bsg = BSGen(k_source, k_rng, swcfg).to(device)

            # Progressive error tracker
            expected = q_vec * k_vec
            pe_tracker = ProgError(expected, {"scale": 1, "mode": "bipolar"}).to(device)

            # Run simulation
            with torch.no_grad():
                for cycle in range(stoc_len):
                    k_bits = k_bsg(torch.tensor([cycle]).to(device))
                    out_bits = dut_mul(k_bits)
                    pe_tracker.Monitor(out_bits)

            # Get result and sum for dot product
            pp, _ = pe_tracker()
            products = pp.cpu().numpy()
            result[i, j] = products.sum() * (max_val ** 2)  # Scale back

    return result


def run_matmul_comparison(N: int = 8, M: int = 8, D: int = 64, sc_prec: int = 8, n_trials: int = 3):
    """Compare on matrix multiplication."""
    print(f"\n{'='*70}")
    print(f"Matrix Multiplication Comparison: ({N}, {D}) @ ({M}, {D})^T")
    print(f"sc_prec={sc_prec}, stoc_len={2**sc_prec}")
    print(f"{'='*70}\n")

    np.random.seed(42)

    triton_errors = []
    triton_time = 0

    unarysim_errors = []
    unarysim_time = 0

    for trial in range(n_trials):
        # Generate random matrices
        Q = np.random.uniform(-1, 1, size=(N, D)).astype(np.float32)
        K = np.random.uniform(-1, 1, size=(M, D)).astype(np.float32)

        # Ground truth
        gt = Q @ K.T

        # Triton
        torch.cuda.synchronize()
        t0 = time.time()
        result_triton = triton_matmul(Q, K, sc_prec)
        torch.cuda.synchronize()
        triton_time += time.time() - t0
        triton_errors.append(result_triton - gt)

        print(f"Trial {trial}: Triton done in {(time.time()-t0)*1000:.1f}ms")

        # UnarySim (only run if small enough)
        if N * M <= 64:  # Too slow for large matrices
            t0 = time.time()
            result_unarysim = unarysim_matmul(Q, K, sc_prec)
            unarysim_time += time.time() - t0
            unarysim_errors.append(result_unarysim - gt)
            print(f"         UnarySim done in {(time.time()-t0)*1000:.1f}ms")

    # Results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")

    max_val = N * D  # Max possible dot product magnitude

    # Triton
    errors = np.concatenate([e.flatten() for e in triton_errors])
    rmse = np.sqrt(np.mean(errors ** 2))
    rmse_norm = rmse / max_val
    avg_time = triton_time / n_trials * 1000
    print(f"Triton GPU:")
    print(f"  RMSE:           {rmse:.6f} (normalized: {rmse_norm:.6f})")
    print(f"  Time per matmul: {avg_time:.2f} ms")
    print()

    # UnarySim
    if unarysim_errors:
        errors = np.concatenate([e.flatten() for e in unarysim_errors])
        rmse = np.sqrt(np.mean(errors ** 2))
        rmse_norm = rmse / max_val
        avg_time = unarysim_time / n_trials * 1000
        print(f"UnarySim C-BSG:")
        print(f"  RMSE:           {rmse:.6f} (normalized: {rmse_norm:.6f})")
        print(f"  Time per matmul: {avg_time:.2f} ms")
        print(f"\nSpeedup (Triton over UnarySim): {unarysim_time/triton_time:.1f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--sc_prec", type=int, default=8)
    parser.add_argument("--n_trials", type=int, default=3)
    args = parser.parse_args()

    run_matmul_comparison(args.N, args.M, args.D, args.sc_prec, args.n_trials)
