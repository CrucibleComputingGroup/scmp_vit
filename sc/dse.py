"""Design Space Exploration (DSE) module for Stochastic Computing.

This module provides general-purpose DSE functions that can work with
any stochastic computing operation using the new RNG/SNG architecture.

Tap configurations are loaded from pre-computed file (lfsr_taps_data.json).
Run `python lfsr_taps.py <bits>` to generate tap data for a given bit width.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Any, Optional
from multiprocessing import Pool, cpu_count

from config_helpers import (
    get_taps_choices,
    make_random_config,
    make_single_rng_reverse_config,
    make_fully_independent_config,
    make_sobol_simple_config,
    make_sobol_dse_config,
    save_best_config,
)


# Config generator types
CONFIG_TWO_RNG = "two_rng"
CONFIG_SINGLE_RNG = "single_rng"
CONFIG_FULLY_INDEPENDENT = "fully_independent"
CONFIG_SOBOL_SIMPLE = "sobol_simple"
CONFIG_SOBOL_DSE = "sobol_dse"


def _eval_config(args):
    """Worker function for parallel config evaluation."""
    sc_func, config, sc_prec, n_runs, fixed_kwargs = args
    errors = []
    for _ in range(n_runs):
        result = sc_func(config=config, sc_prec=sc_prec, **fixed_kwargs)
        error = result[-1] if isinstance(result, tuple) else result
        errors.append(error)
    return (config, np.mean(errors))


def dse_search(
    sc_func: Callable[..., tuple[Any, Any, float]],
    config_type: str,
    q_e: int,
    k_e: int,
    sc_prec: int = 8,
    n_configs: int = 1000,
    n_runs_per_config: int = 1000,
    use_scrambling: bool = True,
    fixed_kwargs: Optional[dict] = None,
    verbose: bool = True,
    n_workers: Optional[int] = None,
    save_result: bool = True,
    operation: str = "matmul",
    binary_prec: str = "int8",
    input_seed: Optional[int] = None,
) -> dict:
    """
    Unified DSE search over RNG/SNG configurations.

    Args:
        sc_func: A callable that returns a tuple where the last element is the error.
        config_type: One of:
            - "two_rng": 2 RNGs (Q->RNG_0, K->RNG_1), per-element scrambling
            - "single_rng": 1 RNG shared, K's scramble = reverse(Q's scramble)
            - "fully_independent": Each SNG has its own RNG
        q_e: Number of Q embedding elements
        k_e: Number of K embedding elements
        sc_prec: Stochastic computing precision (LFSR bit width)
        n_configs: Number of random configurations to try
        n_runs_per_config: Number of runs per configuration for averaging
        use_scrambling: Whether to include scrambling (for two_rng and single_rng)
        fixed_kwargs: Additional fixed keyword arguments to pass to sc_func
        verbose: Enable progress prints
        n_workers: Number of parallel workers (default: cpu_count())
        save_result: Whether to save the best config to file
        operation: Operation type for saving (default: "matmul")
        binary_prec: Binary precision for saving (default: "fp8_e4m3")
        input_seed: Numpy seed for reproducibility

    Returns:
        Dict with best configuration, error, and optionally saved filepath
    """
    if fixed_kwargs is None:
        fixed_kwargs = {}
    if n_workers is None:
        n_workers = cpu_count()

    # Load tap choices only for LFSR-based configs
    taps_choices = None
    if config_type in (CONFIG_TWO_RNG, CONFIG_SINGLE_RNG, CONFIG_FULLY_INDEPENDENT):
        taps_choices = get_taps_choices(sc_prec)

    # Config generator based on type
    def make_config():
        if config_type == CONFIG_TWO_RNG:
            return make_random_config(q_e, k_e, sc_prec, taps_choices, use_scrambling)
        elif config_type == CONFIG_SINGLE_RNG:
            return make_single_rng_reverse_config(q_e, k_e, sc_prec, taps_choices, use_scrambling)
        elif config_type == CONFIG_FULLY_INDEPENDENT:
            return make_fully_independent_config(q_e, k_e, sc_prec, taps_choices)
        elif config_type == CONFIG_SOBOL_SIMPLE:
            # Sobol with optimal Q/K seeds (SCC≈0) - deterministic, no DSE needed
            return make_sobol_simple_config(q_e, k_e, sc_prec)
        elif config_type == CONFIG_SOBOL_DSE:
            # Sobol with random seeds + optional scrambling for DSE search
            return make_sobol_dse_config(q_e, k_e, sc_prec, use_scrambling)
        else:
            raise ValueError(f"Unknown config_type: {config_type}")

    if verbose:
        print(f"Running DSE [{config_type}]: {n_configs} configs x {n_runs_per_config} runs")
        if taps_choices:
            print(f"  Tap configs: {len(taps_choices)}, Workers: {n_workers}")
        else:
            print(f"  Using Sobol sequences, Workers: {n_workers}")

    # Generate all configs; for Sobol DSE, include simple config as baseline
    # so the result is guaranteed to be at least as good as sobol_simple.
    configs = [make_config() for _ in range(n_configs)]
    if config_type == CONFIG_SOBOL_DSE:
        configs[0] = make_sobol_simple_config(q_e, k_e, sc_prec)

    # Prepare args for parallel evaluation
    args_list = [
        (sc_func, cfg, sc_prec, n_runs_per_config, fixed_kwargs)
        for cfg in configs
    ]

    # Run evaluation with progress tracking
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = []
            for i, result in enumerate(pool.imap(_eval_config, args_list)):
                results.append(result)
                if verbose and (i + 1) % 100 == 0:
                    best_so_far = min(r[1] for r in results)
                    print(f"  Progress: {i+1}/{n_configs}, best so far: {best_so_far:.6f}")
    else:
        results = []
        for i, args in enumerate(args_list):
            results.append(_eval_config(args))
            if verbose and (i + 1) % 100 == 0:
                best_so_far = min(r[1] for r in results)
                print(f"  Progress: {i+1}/{n_configs}, best so far: {best_so_far:.2f}")

    # Find best
    best_config = None
    best_error = float('inf')
    for cfg, error in results:
        if error < best_error:
            best_error = error
            best_config = cfg

    if verbose:
        print(f"  Done! Best error: {best_error:.6f}")

    result = {
        "best_config": best_config,
        "best_error": best_error,
        "all_results": results,
        "saved_filepath": None,
    }

    # Save best config if requested
    if save_result and best_config is not None:
        dse_params = {
            "config_type": config_type,
            "n_configs": n_configs,
            "n_runs_per_config": n_runs_per_config,
            "use_scrambling": use_scrambling,
        }
        filepath = save_best_config(
            config=best_config,
            error=best_error,
            operation=operation,
            sc_prec=sc_prec,
            binary_prec=binary_prec,
            operand_sizes={"q": q_e, "k": k_e},
            input_seed=input_seed,
            dse_params=dse_params,
            notes=f"Found via DSE [{config_type}]",
        )
        result["saved_filepath"] = str(filepath)
        if verbose:
            print(f"  Saved best config to: {filepath}")

    return result


if __name__ == "__main__":
    import time
    from sc import matmul_sc

    Q_e = K_e = 256
    N_CONFIGS = 2000
    N_RUNS = 5000

    fixed_params = {
        "Q_l": 1,
        "Q_e": Q_e,
        "K_l": 1,
        "K_e": K_e,
        "verbose": False,
    }

    print("=" * 60)
    print(f"DSE Comparison: {N_CONFIGS} configs x {N_RUNS} runs each")
    print("=" * 60)

    results = {}

    # # Test 1: Single RNG + reverse scramble (paper's approach)
    # print("\n[1] Single RNG + Reverse Scramble")
    # print("    Q and K share same RNG, K's scramble = reverse(Q's scramble)")
    # t0 = time.time()
    # result = dse_search(
    #     sc_func=matmul_sc, config_type=CONFIG_SINGLE_RNG,
    #     q_e=Q_e, k_e=K_e, sc_prec=8,
    #     n_configs=N_CONFIGS, n_runs_per_config=N_RUNS,
    #     use_scrambling=True, fixed_kwargs=fixed_params,
    # )
    # results["1_single_rng"] = result["best_error"]
    # print(f"  Time: {time.time() - t0:.2f}s")

    # # Test 2: Two RNGs (Q→RNG_0, K→RNG_1)
    # print("\n[2] Two RNGs (Q->RNG_0, K->RNG_1)")
    # print("    Each SNG can have its own scramble")
    # t0 = time.time()
    # result = dse_search(
    #     sc_func=matmul_sc, config_type=CONFIG_TWO_RNG,
    #     q_e=Q_e, k_e=K_e, sc_prec=8,
    #     n_configs=N_CONFIGS, n_runs_per_config=N_RUNS,
    #     use_scrambling=True, fixed_kwargs=fixed_params,
    # )
    # results["2_two_rng"] = result["best_error"]
    # print(f"  Time: {time.time() - t0:.2f}s")

    # # Test 3: Fully Independent (each SNG has own RNG)
    # print("\n[3] Fully Independent (128 RNGs)")
    # print("    Each RNG has random taps and seed")
    # t0 = time.time()
    # result = dse_search(
    #     sc_func=matmul_sc, config_type=CONFIG_FULLY_INDEPENDENT,
    #     q_e=Q_e, k_e=K_e, sc_prec=8,
    #     n_configs=N_CONFIGS, n_runs_per_config=N_RUNS,
    #     fixed_kwargs=fixed_params,
    # )
    # results["3_fully_independent"] = result["best_error"]
    # print(f"  Time: {time.time() - t0:.2f}s")

    # Test 4: Simple Sobol with optimal seeds (no DSE needed)
    print("\n[4] Sobol Simple (optimal Q/K seeds)")
    print("    Q seed=[1,1,1,...], K seed=[1,1,1,1,9,1,41,255], SCC≈0")
    t0 = time.time()
    result = dse_search(
        sc_func=matmul_sc, config_type=CONFIG_SOBOL_SIMPLE,
        q_e=Q_e, k_e=K_e, sc_prec=8,
        n_configs=1, n_runs_per_config=N_RUNS,  # Only 1 config since it's deterministic
        fixed_kwargs=fixed_params,
    )
    results["4_sobol_simple"] = result["best_error"]
    print(f"  Time: {time.time() - t0:.2f}s")

    # Test 5: Sobol with DSE (random seeds + scramble search)
    print("\n[5] Sobol DSE (random seeds + scramble search)")
    print("    Random valid Sobol seeds for Q/K, optional per-element scrambling")
    t0 = time.time()
    result = dse_search(
        sc_func=matmul_sc, config_type=CONFIG_SOBOL_DSE,
        q_e=Q_e, k_e=K_e, sc_prec=8,
        n_configs=N_CONFIGS, n_runs_per_config=N_RUNS,
        use_scrambling=True, fixed_kwargs=fixed_params,
    )
    results["5_sobol_dse"] = result["best_error"]
    print(f"  Time: {time.time() - t0:.2f}s")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, error in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name}: {error:.4f}")
