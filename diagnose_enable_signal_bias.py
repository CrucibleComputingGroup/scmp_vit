#!/usr/bin/env python3
"""Diagnose the enable-signal table lookup approximation error.

The enable-signal approach in sc_triton.py computes a PROXY for the joint
enable count that introduces a systematic bias:

  CORRECT:  count = |{t : rng_a[d,t] < ba  AND  rng_b[d,t] < bb}|
  ACTUAL:   count = |{t < k_table[d,ba] : rng_b[d,t] < bb}|
            i.e.   "first k natural-order positions where rng_b < bb"

These differ because the Sobol sequence is NOT sorted in natural time order.
The set {t : rng_a[d,t] < ba} ≠ {0, 1, ..., k-1} in general.

This script computes both quantities for real Sobol sequences and shows the
per-dimension and aggregate error.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sc"))

import numpy as np
from sng import RNGPool, SNG, SNGBank
from config_helpers import make_sobol_simple_config

SC_PREC = 8
STOC_LEN = 2 ** SC_PREC  # 256
D = 64  # head dimension for QK


def main():
    config = make_sobol_simple_config(D, D, SC_PREC)
    rng_pool = RNGPool(config["rng_pool"], SC_PREC)
    sng_a = SNGBank(rng_pool, config["sng"]["q"])
    sng_b = SNGBank(rng_pool, config["sng"]["k"])

    rng_a = sng_a.get_all_sequences(STOC_LEN)  # (D, STOC_LEN)
    rng_b = sng_b.get_all_sequences(STOC_LEN)  # (D, STOC_LEN)

    # Show that Sobol is NOT sorted
    print("=== Sobol sequence (dim 0, first 16 values) ===")
    print(f"  rng_a[0]: {rng_a[0, :16]}")
    print(f"  rng_b[0]: {rng_b[0, :16]}")
    print(f"  Sorted?   {np.all(np.diff(rng_a[0]) > 0)}")
    print()

    # Compute error for a range of boundary pairs
    ba_vals = [32, 64, 96, 128, 160, 192, 224]
    bb_vals = [32, 64, 96, 128, 160, 192, 224]

    print("=== Per-dimension error: correct_count - approx_count ===")
    print(f"{'ba':>5} {'bb':>5} | {'mean_err':>8} {'std_err':>8} {'max_err':>8} | {'mean_correct':>12} {'rel_err%':>10}")
    print("-" * 80)

    for ba in ba_vals:
        for bb in bb_vals:
            errors = []
            correct_counts = []
            for d in range(D):
                # CORRECT: positions where rng_a < ba AND rng_b < bb
                correct = np.sum((rng_a[d] < ba) & (rng_b[d] < bb))

                # APPROXIMATE (what code does): first k positions where rng_b < bb
                k = np.sum(rng_a[d] < ba)  # k_table value
                approx = np.sum(rng_b[d, :k] < bb)

                errors.append(correct - approx)
                correct_counts.append(correct)

            errors = np.array(errors, dtype=float)
            correct_counts = np.array(correct_counts, dtype=float)
            mean_correct = correct_counts.mean()
            if mean_correct > 0:
                rel_err = 100 * np.abs(errors).mean() / mean_correct
            else:
                rel_err = 0
            print(f"{ba:5d} {bb:5d} | {errors.mean():8.2f} {errors.std():8.2f} {np.abs(errors).max():8.0f} | {mean_correct:12.1f} {rel_err:10.2f}")

    # Aggregate: simulate a full dot product
    print("\n=== Aggregate dot product error for random boundary vectors ===")
    rng = np.random.RandomState(42)
    n_trials = 1000

    rel_errors = []
    for _ in range(n_trials):
        ba_vec = rng.randint(0, 257, size=D)  # per-dim boundaries
        bb_vec = rng.randint(0, 257, size=D)

        correct_sum = 0.0
        approx_sum = 0.0
        for d in range(D):
            ba, bb = int(ba_vec[d]), int(bb_vec[d])
            correct = np.sum((rng_a[d] < ba) & (rng_b[d] < bb))
            k = np.sum(rng_a[d] < ba)
            approx = np.sum(rng_b[d, :k] < bb)

            # In bipolar mode, decode = count * q_max^2 / stoc_len * sign_a * sign_b
            # Signs are random, so let's use uniform signs
            s = 1 if rng.random() < 0.5 else -1
            correct_sum += correct * s
            approx_sum += approx * s

        if abs(correct_sum) > 1:
            rel_errors.append(abs(correct_sum - approx_sum) / abs(correct_sum))

    rel_errors = np.array(rel_errors)
    print(f"  D = {D}")
    print(f"  Trials: {n_trials}")
    print(f"  Mean |relative error|: {rel_errors.mean():.4f} ({rel_errors.mean()*100:.2f}%)")
    print(f"  Median |relative error|: {np.median(rel_errors):.4f}")
    print(f"  95th percentile: {np.percentile(rel_errors, 95):.4f}")
    print(f"  Max: {rel_errors.max():.4f}")

    # Now for D=1024 (MLP fc1)
    print(f"\n=== Same analysis for D=1024 (MLP fc1 inner dim) ===")
    config_mlp = make_sobol_simple_config(1024, 1024, SC_PREC)
    rng_pool_mlp = RNGPool(config_mlp["rng_pool"], SC_PREC)
    sng_a_mlp = SNGBank(rng_pool_mlp, config_mlp["sng"]["q"])
    sng_b_mlp = SNGBank(rng_pool_mlp, config_mlp["sng"]["k"])
    rng_a_mlp = sng_a_mlp.get_all_sequences(STOC_LEN)
    rng_b_mlp = sng_b_mlp.get_all_sequences(STOC_LEN)

    # Check: is it the same sequence broadcast?
    if np.array_equal(rng_a_mlp[0], rng_a_mlp[1]):
        print("  NOTE: All dimensions share the SAME base sequence (just scrambled)")
        # Check if scrambles are applied
        n_unique = len(set(tuple(rng_a_mlp[d]) for d in range(min(32, 1024))))
        print(f"  Unique sequences among first 32 dims: {n_unique}")
    else:
        print("  Different sequences per dimension")

    rel_errors_mlp = []
    for _ in range(n_trials):
        ba_vec = rng.randint(0, 257, size=1024)
        bb_vec = rng.randint(0, 257, size=1024)

        correct_sum = 0.0
        approx_sum = 0.0
        for d in range(1024):
            ba, bb = int(ba_vec[d]), int(bb_vec[d])
            correct = np.sum((rng_a_mlp[d] < ba) & (rng_b_mlp[d] < bb))
            k = np.sum(rng_a_mlp[d] < ba)
            approx = np.sum(rng_b_mlp[d, :k] < bb)
            s = 1 if rng.random() < 0.5 else -1
            correct_sum += correct * s
            approx_sum += approx * s

        if abs(correct_sum) > 1:
            rel_errors_mlp.append(abs(correct_sum - approx_sum) / abs(correct_sum))

    rel_errors_mlp = np.array(rel_errors_mlp)
    print(f"  D = 1024")
    print(f"  Mean |relative error|: {rel_errors_mlp.mean():.4f} ({rel_errors_mlp.mean()*100:.2f}%)")
    print(f"  Median |relative error|: {np.median(rel_errors_mlp):.4f}")
    print(f"  95th percentile: {np.percentile(rel_errors_mlp, 95):.4f}")
    print(f"  Max: {rel_errors_mlp.max():.4f}")


    # === Verify: are all dimensions using the SAME sequence? ===
    print(f"\n=== Sobol diversity check (D=64) ===")
    same_count = sum(1 for d in range(1, D) if np.array_equal(rng_a[0], rng_a[d]))
    print(f"  Dims with rng_a identical to dim 0: {same_count}/{D-1}")
    same_b = sum(1 for d in range(1, D) if np.array_equal(rng_b[0], rng_b[d]))
    print(f"  Dims with rng_b identical to dim 0: {same_b}/{D-1}")
    print(f"  rng_a[0] == rng_b[0]? {np.array_equal(rng_a[0], rng_b[0])}")
    # Check divergence point
    for t in range(STOC_LEN):
        if rng_a[0, t] != rng_b[0, t]:
            print(f"  rng_a[0] and rng_b[0] first differ at t={t}: a={rng_a[0,t]}, b={rng_b[0,t]}")
            break

    # === Show what the CORRECT fix would produce ===
    print(f"\n=== Error with sorted-rng_a fix (D=64) ===")
    # If we sort rng_b by rng_a ordering, the enable-signal approach becomes exact.
    errors_fixed = []
    for ba in [32, 64, 128, 192]:
        for bb in [32, 64, 128, 192]:
            for d in range(D):
                # Sort rng_b by the argsort of rng_a
                sort_order = np.argsort(rng_a[d])
                rng_b_sorted_by_a = rng_b[d, sort_order]

                # Now "first k" in sorted order = positions where rng_a < ba
                k = np.sum(rng_a[d] < ba)
                approx_fixed = np.sum(rng_b_sorted_by_a[:k] < bb)
                correct = np.sum((rng_a[d] < ba) & (rng_b[d] < bb))

                errors_fixed.append(correct - approx_fixed)

    errors_fixed = np.array(errors_fixed)
    print(f"  Max |error| after fix: {np.abs(errors_fixed).max()}")
    print(f"  Mean |error| after fix: {np.abs(errors_fixed).mean():.6f}")


if __name__ == "__main__":
    main()
