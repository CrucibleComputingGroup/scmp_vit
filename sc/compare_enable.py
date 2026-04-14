"""
Accuracy comparison: enable-signal SC vs standard SC vs FP ground truth.

Measures:
- RMSE (normalized by ground truth magnitude)
- Pearson correlation with FP ground truth
- Max absolute error
- Verifies cycle_by_cycle matches k_shortcut exactly

Sweeps across sc_prec values, dimensions D, and both modes.
"""

import torch
import numpy as np
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from sc_enable import sc_matmul_enable
from config_helpers import make_sobol_simple_config


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Pearson correlation between two flattened tensors."""
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    x_centered = x_flat - x_flat.mean()
    y_centered = y_flat - y_flat.mean()
    num = (x_centered * y_centered).sum()
    den = (x_centered.norm() * y_centered.norm()).clamp(min=1e-8)
    return (num / den).item()


def run_comparison(
    N: int = 8,
    M: int = 8,
    D: int = 64,
    sc_prec: int = 8,
    mode: str = "bipolar",
    seed: int = 42,
    use_cuda: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run a single accuracy comparison.

    Returns dict with metrics for enable-SC and standard SC.
    """
    torch.manual_seed(seed)
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    config = make_sobol_simple_config(D, D, sc_prec)

    if mode == "bipolar":
        a = torch.randn(N, D, device=device) * 3.0
        b = torch.randn(M, D, device=device) * 3.0
        max_fp_a = a.abs().max().item()
        min_fp_a = -max_fp_a
        max_fp_b = b.abs().max().item()
        min_fp_b = -max_fp_b
    else:  # unipolar
        a = torch.rand(N, D, device=device)
        b = torch.rand(M, D, device=device)
        max_fp_a = a.max().item()
        min_fp_a = a.min().item()
        max_fp_b = b.max().item()
        min_fp_b = b.min().item()

    # Ground truth
    gt = a @ b.T

    # Enable-SC: cycle-by-cycle
    result_cbc = sc_matmul_enable(
        a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b,
        mode=mode, sc_prec=sc_prec, config=config, method="cycle_by_cycle",
    )

    # Enable-SC: k-shortcut
    result_ks = sc_matmul_enable(
        a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b,
        mode=mode, sc_prec=sc_prec, config=config, method="k_shortcut",
    )

    # Standard SC (import here to handle optional CUDA dependency)
    try:
        from sc_triton import sc_matmul
        if device == "cpu":
            # sc_matmul needs CUDA, run on CPU by temporarily moving
            a_cuda = a.cuda()
            b_cuda = b.cuda()
        else:
            a_cuda = a
            b_cuda = b
        result_std = sc_matmul(
            a_cuda, b_cuda, max_fp_a, min_fp_a, max_fp_b, min_fp_b,
            mode=mode, sc_prec=sc_prec, config=config,
        )
        if device == "cpu":
            result_std = result_std.cpu()
        has_standard = True
    except Exception as e:
        if verbose:
            print(f"    [Standard SC unavailable: {e}]")
        result_std = None
        has_standard = False

    # Verify cycle-by-cycle == k-shortcut
    cbc_ks_diff = (result_cbc - result_ks).abs().max().item()

    # Metrics for enable-SC
    gt_mag = gt.abs().mean().item()
    norm_factor = max(gt_mag, 1e-8)

    enable_rmse = ((result_ks - gt) ** 2).mean().sqrt().item() / norm_factor
    enable_corr = pearson_corr(result_ks, gt)
    enable_max_err = (result_ks - gt).abs().max().item()

    metrics = {
        "mode": mode,
        "sc_prec": sc_prec,
        "D": D,
        "N": N,
        "M": M,
        "cbc_ks_max_diff": cbc_ks_diff,
        "enable_rmse": enable_rmse,
        "enable_corr": enable_corr,
        "enable_max_err": enable_max_err,
    }

    # Metrics for standard SC
    if has_standard:
        std_rmse = ((result_std - gt) ** 2).mean().sqrt().item() / norm_factor
        std_corr = pearson_corr(result_std, gt)
        std_max_err = (result_std - gt).abs().max().item()
        metrics["std_rmse"] = std_rmse
        metrics["std_corr"] = std_corr
        metrics["std_max_err"] = std_max_err

    if verbose:
        print(f"    cbc==ks: max_diff={cbc_ks_diff:.2e}  "
              f"{'OK' if cbc_ks_diff < 1e-3 else 'MISMATCH!'}")
        print(f"    Enable-SC: RMSE={enable_rmse:.4f}  corr={enable_corr:.4f}  "
              f"max_err={enable_max_err:.2f}")
        if has_standard:
            print(f"    Standard:  RMSE={std_rmse:.4f}  corr={std_corr:.4f}  "
                  f"max_err={std_max_err:.2f}")
            rmse_ratio = enable_rmse / max(std_rmse, 1e-8)
            print(f"    Enable/Std RMSE ratio: {rmse_ratio:.3f} "
                  f"({'better' if rmse_ratio < 1 else 'worse'})")

    return metrics


def sweep_comparison(
    use_cuda: bool = False,
    verbose: bool = True,
):
    """
    Run full parameter sweep comparing enable-SC vs standard SC.
    """
    print("=" * 70)
    print("Enable-Signal SC vs Standard SC Accuracy Comparison")
    print("=" * 70)

    sc_precs = [4, 6, 8]
    dims = [32, 64, 128]
    modes = ["bipolar", "unipolar"]
    N, M = 8, 8

    all_metrics = []

    for mode in modes:
        print(f"\n{'='*70}")
        print(f"Mode: {mode}")
        print(f"{'='*70}")

        for sc_prec in sc_precs:
            for D in dims:
                print(f"\n  sc_prec={sc_prec}, D={D}:")
                metrics = run_comparison(
                    N=N, M=M, D=D, sc_prec=sc_prec, mode=mode,
                    seed=42, use_cuda=use_cuda, verbose=verbose,
                )
                all_metrics.append(metrics)

    # Summary table
    print(f"\n{'='*70}")
    print("Summary Table")
    print(f"{'='*70}")

    has_std = "std_rmse" in all_metrics[0]

    header = f"{'Mode':<10} {'Prec':>4} {'D':>4} | {'En-RMSE':>8} {'En-Corr':>8}"
    if has_std:
        header += f" | {'St-RMSE':>8} {'St-Corr':>8} | {'Ratio':>6}"
    print(header)
    print("-" * len(header))

    for m in all_metrics:
        line = (f"{m['mode']:<10} {m['sc_prec']:>4} {m['D']:>4} | "
                f"{m['enable_rmse']:>8.4f} {m['enable_corr']:>8.4f}")
        if has_std and "std_rmse" in m:
            ratio = m["enable_rmse"] / max(m["std_rmse"], 1e-8)
            line += (f" | {m['std_rmse']:>8.4f} {m['std_corr']:>8.4f} "
                     f"| {ratio:>6.3f}")
        print(line)

    # Check all cbc==ks
    all_match = all(m["cbc_ks_max_diff"] < 1e-3 for m in all_metrics)
    print(f"\ncycle_by_cycle == k_shortcut: {'ALL MATCH' if all_match else 'MISMATCH DETECTED!'}")

    return all_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare enable-signal SC vs standard SC")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for computation")
    parser.add_argument("--quick", action="store_true", help="Quick test (single config)")
    args = parser.parse_args()

    if args.quick:
        print("Quick test: bipolar sc_prec=8 D=64")
        run_comparison(sc_prec=8, D=64, mode="bipolar", use_cuda=args.cuda)
        print("\nQuick test: unipolar sc_prec=8 D=64")
        run_comparison(sc_prec=8, D=64, mode="unipolar", use_cuda=args.cuda)
    else:
        sweep_comparison(use_cuda=args.cuda)
