#!/usr/bin/env python3
"""Prove the enable-signal table lookup bias — NumPy (CPU) + GPU experiments.

Experiments:
  1 (CPU):  NumPy reference SC matmul — correct joint count vs table-lookup
            approximation vs sorted-fix. Shows the error and that fix zeroes it.
  2 (GPU):  Same comparison using real Triton kernels: enable-signal vs packed
            XNOR (ground truth). Shows the kernels diverge exactly as predicted.
  3 (GPU):  End-to-end model accuracy: original SC vs fixed SC vs FP.
"""
import sys, os, time, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sc"))
sys.path.insert(0, str(ROOT))

import numpy as np
from config_helpers import make_sobol_simple_config


# ── Pure-NumPy SC matmul (reference) ─────────────────────────────────────────

def numpy_bipolar_sc_matmul(a, b, rng_a, rng_b, sc_prec=8, use_fix=False):
    """
    NumPy bipolar SC matmul: a (N,D) @ b (M,D)^T → (N,M).

    Per-row symmetric quantization, enable-signal counting.

    If use_fix=True: sort rng_b by rng_a per dimension before counting.
    """
    N, D = a.shape
    M = b.shape[0]
    q_max = 2 ** (sc_prec - 1) - 1  # 127
    max_rng_val = 2 ** sc_prec       # 256
    stoc_len = rng_a.shape[1]

    # Per-row symmetric quantization (clip_margin=0)
    abs_max_a = np.maximum(np.abs(a).max(axis=1), 1e-5)  # (N,)
    scale_a = abs_max_a / q_max
    a_int = np.clip(np.round(a / scale_a[:, None]), -q_max, q_max)
    sign_a = np.sign(a_int).astype(np.int8)
    sign_a[sign_a == 0] = 1

    abs_max_b = np.maximum(np.abs(b).max(axis=1), 1e-5)  # (M,)
    scale_b = abs_max_b / q_max
    b_int = np.clip(np.round(b / scale_b[:, None]), -q_max, q_max)
    sign_b = np.sign(b_int).astype(np.int8)
    sign_b[sign_b == 0] = 1

    # Boundaries (unipolar magnitude)
    boundary_a = np.round(np.abs(a_int) * max_rng_val / q_max).astype(np.int32)  # (N,D)
    boundary_b = np.round(np.abs(b_int) * max_rng_val / q_max).astype(np.int32)  # (M,D)

    q_max_sq = float(q_max * q_max)

    # Precompute per-dim sort order if fix is enabled
    if use_fix:
        sort_orders = [np.argsort(rng_a[d, :stoc_len]) for d in range(D)]
        rng_b_eff = np.empty_like(rng_b[:, :stoc_len])
        for d in range(D):
            rng_b_eff[d] = rng_b[d, :stoc_len][sort_orders[d]]
    else:
        rng_b_eff = rng_b[:, :stoc_len]

    # Build k_table[d, v] = |{t : rng_a[d,t] < v}| for v in [0, max_rng_val]
    # Since all dims share the same sequence, k_table is the same for all d.
    # But we compute per-dim for generality.
    # Instead of full table, compute k(ba) = sum(rng_a[d] < ba) on the fly.

    result = np.zeros((N, M), dtype=np.float64)

    for d in range(D):
        ra = rng_a[d, :stoc_len]
        rb = rng_b_eff[d]

        for n in range(N):
            ba = boundary_a[n, d]
            sa = sign_a[n, d]
            k = int(np.sum(ra < ba))

            for m in range(M):
                bb = boundary_b[m, d]
                sb = sign_b[m, d]

                # Enable-signal count: first k positions where rng_b < bb
                count = int(np.sum(rb[:k] < bb))

                decoded = count * q_max_sq / stoc_len
                result[n, m] += decoded * sa * sb

    # Dequantize
    result *= scale_a[:, None] * scale_b[None, :]
    return result


def numpy_correct_sc_matmul(a, b, rng_a, rng_b, sc_prec=8):
    """Ground-truth SC: count positions where BOTH rng_a < ba AND rng_b < bb."""
    N, D = a.shape
    M = b.shape[0]
    q_max = 2 ** (sc_prec - 1) - 1
    max_rng_val = 2 ** sc_prec
    stoc_len = rng_a.shape[1]

    abs_max_a = np.maximum(np.abs(a).max(axis=1), 1e-5)
    scale_a = abs_max_a / q_max
    a_int = np.clip(np.round(a / scale_a[:, None]), -q_max, q_max)
    sign_a = np.sign(a_int).astype(np.int8)
    sign_a[sign_a == 0] = 1

    abs_max_b = np.maximum(np.abs(b).max(axis=1), 1e-5)
    scale_b = abs_max_b / q_max
    b_int = np.clip(np.round(b / scale_b[:, None]), -q_max, q_max)
    sign_b = np.sign(b_int).astype(np.int8)
    sign_b[sign_b == 0] = 1

    boundary_a = np.round(np.abs(a_int) * max_rng_val / q_max).astype(np.int32)
    boundary_b = np.round(np.abs(b_int) * max_rng_val / q_max).astype(np.int32)

    q_max_sq = float(q_max * q_max)

    result = np.zeros((N, M), dtype=np.float64)

    for d in range(D):
        ra = rng_a[d, :stoc_len]
        rb = rng_b[d, :stoc_len]

        for n in range(N):
            ba = boundary_a[n, d]
            sa = sign_a[n, d]

            for m in range(M):
                bb = boundary_b[m, d]
                sb = sign_b[m, d]

                # CORRECT joint count
                count = int(np.sum((ra < ba) & (rb < bb)))

                decoded = count * q_max_sq / stoc_len
                result[n, m] += decoded * sa * sb

    result *= scale_a[:, None] * scale_b[None, :]
    return result


# ── Vectorized versions (much faster for larger dims) ────────────────────────

def vec_enable_signal_matmul(a, b, rng_a, rng_b, sc_prec=8, use_fix=False):
    """Vectorized enable-signal SC matmul (approx or fixed)."""
    N, D = a.shape
    M = b.shape[0]
    q_max = 2 ** (sc_prec - 1) - 1
    max_rng_val = 2 ** sc_prec
    stoc_len = rng_a.shape[1]

    abs_max_a = np.maximum(np.abs(a).max(axis=1, keepdims=True), 1e-5)
    scale_a = abs_max_a / q_max
    a_int = np.clip(np.round(a / scale_a), -q_max, q_max)
    sign_a = np.sign(a_int).astype(np.float64)
    sign_a[sign_a == 0] = 1

    abs_max_b = np.maximum(np.abs(b).max(axis=1, keepdims=True), 1e-5)
    scale_b = abs_max_b / q_max
    b_int = np.clip(np.round(b / scale_b), -q_max, q_max)
    sign_b = np.sign(b_int).astype(np.float64)
    sign_b[sign_b == 0] = 1

    boundary_a = np.round(np.abs(a_int) * max_rng_val / q_max).astype(np.int32)  # (N,D)
    boundary_b = np.round(np.abs(b_int) * max_rng_val / q_max).astype(np.int32)  # (M,D)

    q_max_sq = float(q_max * q_max)

    result = np.zeros((N, M), dtype=np.float64)

    for d in range(D):
        ra = rng_a[d, :stoc_len]
        if use_fix:
            sort_idx = np.argsort(ra)
            rb = rng_b[d, :stoc_len][sort_idx]
        else:
            rb = rng_b[d, :stoc_len]

        # k_a[n] = number of positions where ra < boundary_a[n,d]
        # Since ra is a permutation of [0..255], k = boundary_a value (roughly)
        # But compute exactly:
        k_a = np.array([np.sum(ra < boundary_a[n, d]) for n in range(N)])  # (N,)

        # count[n,m] = |{t < k_a[n] : rb[t] < boundary_b[m,d]}|
        # Precompute prefix-count for rb
        # prefix[k] = number of rb[:k] < each possible bb
        # Instead: for each unique k, compute counts
        counts = np.zeros((N, M), dtype=np.float64)
        for n in range(N):
            k = k_a[n]
            if k > 0:
                rb_prefix = rb[:k]
                for m in range(M):
                    counts[n, m] = np.sum(rb_prefix < boundary_b[m, d])

        decoded = counts * q_max_sq / stoc_len
        result += decoded * sign_a[:, d:d+1] * sign_b[:, d:d+1].T

    result *= (scale_a * scale_b.T)
    return result


def vec_correct_matmul(a, b, rng_a, rng_b, sc_prec=8):
    """Vectorized correct SC matmul (joint counting)."""
    N, D = a.shape
    M = b.shape[0]
    q_max = 2 ** (sc_prec - 1) - 1
    max_rng_val = 2 ** sc_prec
    stoc_len = rng_a.shape[1]

    abs_max_a = np.maximum(np.abs(a).max(axis=1, keepdims=True), 1e-5)
    scale_a = abs_max_a / q_max
    a_int = np.clip(np.round(a / scale_a), -q_max, q_max)
    sign_a = np.sign(a_int).astype(np.float64)
    sign_a[sign_a == 0] = 1

    abs_max_b = np.maximum(np.abs(b).max(axis=1, keepdims=True), 1e-5)
    scale_b = abs_max_b / q_max
    b_int = np.clip(np.round(b / scale_b), -q_max, q_max)
    sign_b = np.sign(b_int).astype(np.float64)
    sign_b[sign_b == 0] = 1

    boundary_a = np.round(np.abs(a_int) * max_rng_val / q_max).astype(np.int32)
    boundary_b = np.round(np.abs(b_int) * max_rng_val / q_max).astype(np.int32)

    q_max_sq = float(q_max * q_max)

    result = np.zeros((N, M), dtype=np.float64)

    for d in range(D):
        ra = rng_a[d, :stoc_len]
        rb = rng_b[d, :stoc_len]

        counts = np.zeros((N, M), dtype=np.float64)
        for n in range(N):
            a_mask = ra < boundary_a[n, d]  # (stoc_len,)
            for m in range(M):
                counts[n, m] = np.sum(a_mask & (rb < boundary_b[m, d]))

        decoded = counts * q_max_sq / stoc_len
        result += decoded * sign_a[:, d:d+1] * sign_b[:, d:d+1].T

    result *= (scale_a * scale_b.T)
    return result


# ── Experiment 1: CPU pure-NumPy proof ──────────────────────────────────────

def exp1_cpu():
    """Pure NumPy comparison: enable-signal (buggy) vs correct vs fixed."""
    from sng import RNGPool, SNGBank

    print("=" * 72)
    print("EXP 1: NumPy reference — enable-signal vs correct vs sorted-fix")
    print("=" * 72)

    sc_prec = 8
    stoc_len = 256
    np.random.seed(42)

    for label, N, D, M in [
        ("QK-head   D=64", 16, 64, 16),
        ("small-proj D=128", 16, 128, 16),
    ]:
        print(f"\n--- {label}: ({N},{D}) @ ({M},{D})^T ---")

        config = make_sobol_simple_config(D, D, sc_prec)
        rng_pool = RNGPool(config["rng_pool"], sc_prec)
        sng_a = SNGBank(rng_pool, config["sng"]["q"])
        sng_b = SNGBank(rng_pool, config["sng"]["k"])
        rng_a = sng_a.get_all_sequences(stoc_len)  # (D, stoc_len)
        rng_b = sng_b.get_all_sequences(stoc_len)

        # Random matrices
        a = np.random.randn(N, D).astype(np.float64) * 0.5
        b = np.random.randn(M, D).astype(np.float64) * 0.5
        fp = a @ b.T

        t0 = time.time()
        res_approx = vec_enable_signal_matmul(a, b, rng_a, rng_b, sc_prec, use_fix=False)
        t1 = time.time()
        res_correct = vec_correct_matmul(a, b, rng_a, rng_b, sc_prec)
        t2 = time.time()
        res_fixed = vec_enable_signal_matmul(a, b, rng_a, rng_b, sc_prec, use_fix=True)
        t3 = time.time()

        fp_scale = np.abs(fp).mean()

        err_approx  = np.abs(res_approx - fp).mean() / fp_scale
        err_correct = np.abs(res_correct - fp).mean() / fp_scale
        err_fixed   = np.abs(res_fixed - fp).mean() / fp_scale

        gap_approx_vs_correct = np.abs(res_approx - res_correct).mean() / fp_scale
        gap_fixed_vs_correct  = np.abs(res_fixed - res_correct).mean() / fp_scale

        print(f"  FP result scale:                      {fp_scale:.4f}")
        print(f"  |enable_approx  - FP| / scale:        {err_approx:.6f}  ({t1-t0:.1f}s)")
        print(f"  |correct_joint  - FP| / scale:        {err_correct:.6f}  ({t2-t1:.1f}s)")
        print(f"  |sorted_fix     - FP| / scale:        {err_fixed:.6f}  ({t3-t2:.1f}s)")
        print(f"  |enable_approx - correct| / scale:    {gap_approx_vs_correct:.6f}  ← TABLE LOOKUP ERROR")
        print(f"  |sorted_fix    - correct| / scale:    {gap_fixed_vs_correct:.6f}  ← SHOULD BE ZERO")

        # Check: is fixed == correct?
        max_diff = np.abs(res_fixed - res_correct).max()
        print(f"  max |fixed - correct|:                {max_diff:.2e}  "
              f"({'EXACT MATCH' if max_diff < 1e-10 else 'MISMATCH!'})")

        # Decompose error into mean (bias) and std (variance) per output element
        err_per_elem = res_approx - fp
        bias = err_per_elem.mean()
        std = err_per_elem.std()
        err_per_elem_c = res_correct - fp
        bias_c = err_per_elem_c.mean()
        std_c = err_per_elem_c.std()
        print(f"  enable_approx error:  bias={bias/fp_scale:.6f}  std={std/fp_scale:.6f}")
        print(f"  correct_joint error:  bias={bias_c/fp_scale:.6f}  std={std_c/fp_scale:.6f}")
        print(f"  --> approx has {'LARGER' if abs(bias) > abs(bias_c) else 'smaller'} bias")


# ── Experiment 2: GPU Triton kernel comparison ──────────────────────────────

def exp2_gpu(device="cuda"):
    """Compare enable-signal (Triton) vs packed XNOR (Triton ground truth)."""
    import torch
    torch.set_grad_enabled(False)

    from sc_triton import (
        sc_matmul_grouped_enable_triton,
        build_enable_tables,
        _get_cached_sequences,
        _grouped_symmetric_quant,
        enable_matmul_triton,
    )
    from sc_triton import _sc_matmul_bipolar as xnor_matmul_bipolar

    print("\n" + "=" * 72)
    print("EXP 2: Triton kernels — enable-signal vs packed-XNOR vs fixed-tables")
    print("=" * 72)

    sc_prec = 8
    stoc_len = 256

    for label, N, D, M in [
        ("QK (head)", 258, 64, 258),
        ("AV (head)", 258, 258, 64),
        ("qkv_proj", 258, 1024, 3072),
        ("out_proj", 258, 1024, 1024),
        ("fc1", 258, 1024, 4096),
        ("fc2", 258, 4096, 1024),
    ]:
        print(f"\n--- {label}: ({N},{D}) @ ({M},{D})^T ---")

        torch.manual_seed(42)
        a = torch.randn(N, D, device=device) * 0.5
        b = torch.randn(M, D, device=device) * 0.5

        config = make_sobol_simple_config(D, D, sc_prec)
        rng_a_t, rng_b_t = _get_cached_sequences(config, sc_prec, device)

        # ── A. Original enable-signal (what the code uses) ──
        result_enable = sc_matmul_grouped_enable_triton(
            a, b, group_a=1, group_b=1,
            mode="bipolar", sc_prec=sc_prec, config=config,
        )

        # ── B. Packed XNOR (correct SC, same Sobol sequences) ──
        max_a, min_a = a.max().item(), a.min().item()
        max_b, min_b = b.max().item(), b.min().item()
        result_xnor = xnor_matmul_bipolar(
            a, b, max_a, min_a, max_b, min_b, sc_prec,
            rng_a_t, rng_b_t, N, D, M, stoc_len,
        )

        # ── C. Fixed enable tables (sort rng_b by rng_a) ──
        rng_b_sorted = torch.empty_like(rng_b_t)
        for d in range(D):
            sort_idx = torch.argsort(rng_a_t[d, :stoc_len])
            rng_b_sorted[d, :stoc_len] = rng_b_t[d, :stoc_len][sort_idx]
            if rng_b_t.shape[1] > stoc_len:
                rng_b_sorted[d, stoc_len:] = rng_b_t[d, stoc_len:]
        cum_fixed, k_table_fixed = build_enable_tables(rng_a_t, rng_b_sorted, sc_prec, stoc_len)

        q_max = 2 ** (sc_prec - 1) - 1
        max_rng_val = 2 ** sc_prec
        scale_a_row, a_int, sign_a_t = _grouped_symmetric_quant(a.float(), 1, q_max, clip_margin=0)
        scale_b_row, b_int, sign_b_t = _grouped_symmetric_quant(b.float(), 1, q_max, clip_margin=0)
        boundary_a_t = (a_int.abs() * max_rng_val / q_max).round().short()
        boundary_b_t = (b_int.abs() * max_rng_val / q_max).round().short()
        q_max_sq = float(q_max * q_max)

        sc_raw_fixed = enable_matmul_triton(
            cum_fixed, k_table_fixed, boundary_a_t, boundary_b_t,
            sign_a_t, sign_b_t, N, M, D, stoc_len, q_max_sq, is_bipolar=True,
        )
        result_fixed = sc_raw_fixed * (scale_a_row[:, None] * scale_b_row[None, :])

        # ── D. FP ground truth ──
        result_fp = a @ b.t()

        # ── Metrics ──
        fp_scale = result_fp.abs().mean().item()

        def rel(x):
            return x.abs().mean().item() / fp_scale

        print(f"  FP scale:                          {fp_scale:.4f}")
        print(f"  |enable_orig  - FP| / scale:       {rel(result_enable - result_fp):.6f}")
        print(f"  |packed_XNOR  - FP| / scale:       {rel(result_xnor - result_fp):.6f}")
        print(f"  |enable_fixed - FP| / scale:       {rel(result_fixed - result_fp):.6f}")
        print(f"  |enable_orig  - XNOR| / scale:     {rel(result_enable - result_xnor):.6f}  ← TABLE ERROR")
        print(f"  |enable_fixed - XNOR| / scale:     {rel(result_fixed - result_xnor):.6f}  ← AFTER FIX")
        gap_orig = rel(result_enable - result_xnor)
        gap_fixed = rel(result_fixed - result_xnor)
        if gap_fixed > 1e-9:
            print(f"  --> Fix reduces table error by {gap_orig/gap_fixed:.1f}x")
        else:
            print(f"  --> Fix ELIMINATES table error (gap → 0)")

        del result_enable, result_xnor, result_fixed, result_fp
        del cum_fixed, k_table_fixed, a_int, b_int, sign_a_t, sign_b_t
        torch.cuda.empty_cache()


# ── Experiment 3: end-to-end accuracy ────────────────────────────────────────

def exp3_e2e(args, device="cuda"):
    """Patch build_enable_tables, run eval, compare original vs fixed."""
    import torch
    torch.set_grad_enabled(False)
    import sc_triton as sct

    print("\n" + "=" * 72)
    print("EXP 3: End-to-end ViT accuracy — original vs fixed enable-signal")
    print("=" * 72)

    original_build = sct.build_enable_tables

    def fixed_build(rng_a, rng_b, sc_prec, stoc_len=None):
        if stoc_len is None:
            stoc_len = 2 ** sc_prec
        D = rng_a.shape[0]
        rng_b_sorted = torch.empty_like(rng_b)
        for d in range(D):
            sort_idx = torch.argsort(rng_a[d, :stoc_len])
            rng_b_sorted[d, :stoc_len] = rng_b[d, :stoc_len][sort_idx]
            if rng_b.shape[1] > stoc_len:
                rng_b_sorted[d, stoc_len:] = rng_b[d, stoc_len:]
        return original_build(rng_a, rng_b_sorted, sc_prec, stoc_len)

    import random
    from torch.utils.data import DataLoader, Subset
    from eval import load_model, evaluate, build_transform
    from imagenet_parquet import ImageNetParquetVal
    from sc_attention_patch import patch_model, _CFG_CACHE

    data_root = args.data_root

    # Load data once
    ds = ImageNetParquetVal(data_root, transform=build_transform(224))
    if args.n_eval and args.n_eval < len(ds):
        idx = list(range(len(ds)))
        random.Random(0).shuffle(idx)
        ds = Subset(ds, idx[:args.n_eval])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True, drop_last=False)

    model_fp = load_model(device)

    sc_prec = 8
    sc_configs = [
        ("sc_qk", dict(sc_qk=True, sc_av=False, sc_qkv_proj=False, sc_out_proj=False, sc_mlp=False)),
        ("sc_qk_av", dict(sc_qk=True, sc_av=True, sc_qkv_proj=False, sc_out_proj=False, sc_mlp=False)),
        ("full_attn", dict(sc_qk=True, sc_av=True, sc_qkv_proj=True, sc_out_proj=True, sc_mlp=False)),
    ]

    results = {}

    # FP baseline
    print("\n[FP baseline]")
    met = evaluate(model_fp, loader, device=device)
    results["fp"] = met["top1"]
    print(f"  top-1 = {met['top1']:.4f}")

    for cfg_name, sc_kwargs in sc_configs:
        for variant_name, use_fix in [("original", False), ("fixed", True)]:
            tag = f"{cfg_name}_{variant_name}"
            print(f"\n[{tag}]")

            # Clear all caches
            sct.clear_rng_cache()
            sct._enable_table_cache.clear()
            sct._k_table_cache.clear()
            _CFG_CACHE.clear()

            # Apply or remove fix
            sct.build_enable_tables = fixed_build if use_fix else original_build

            # Reload model
            m = load_model(device)
            patch_model(m, sc_prec=sc_prec, **sc_kwargs)
            met = evaluate(m, loader, device=device)
            results[tag] = met["top1"]
            delta = met["top1"] - results["fp"]
            print(f"  top-1 = {met['top1']:.4f}  (Δ vs FP = {delta:+.4f})")
            del m
            torch.cuda.empty_cache()

    sct.build_enable_tables = original_build

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Config':<30} {'top-1':>8} {'Δ vs FP':>10}")
    print("-" * 50)
    for k, v in results.items():
        d = v - results["fp"]
        print(f"{k:<30} {v:>8.4f} {d:>+10.4f}" if k != "fp" else f"{k:<30} {v:>8.4f} {'':>10}")

    # Save
    out_path = ROOT / "results" / "enable_signal_fix_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


# ── Experiment 4: Coherent noise from no per-dimension Sobol diversity ───────

def exp4_coherent_noise():
    """Show that all D dimensions share the SAME Sobol sequence (no scramble),
    causing SC noise to accumulate coherently (∝D) instead of averaging (∝√D).

    Compare: same-sequence (current) vs independent-per-dim (scrambled).
    """
    from sng import RNGPool, SNGBank

    print("\n" + "=" * 72)
    print("EXP 4: Coherent SC noise from shared Sobol sequences")
    print("       (all D dims use the SAME rng — no per-dim diversity)")
    print("=" * 72)

    sc_prec = 8
    stoc_len = 256
    np.random.seed(42)

    for label, N, D, M in [
        ("QK head D=64", 32, 64, 32),
        ("proj D=256", 32, 256, 32),
        ("fc1  D=1024", 32, 1024, 32),
    ]:
        print(f"\n--- {label}: ({N},{D}) @ ({M},{D})^T ---")

        config = make_sobol_simple_config(D, D, sc_prec)
        rng_pool = RNGPool(config["rng_pool"], sc_prec)
        sng_a = SNGBank(rng_pool, config["sng"]["q"])
        sng_b = SNGBank(rng_pool, config["sng"]["k"])
        rng_a = sng_a.get_all_sequences(stoc_len)  # (D, stoc_len)
        rng_b = sng_b.get_all_sequences(stoc_len)

        # Verify: all dims same?
        all_same_a = all(np.array_equal(rng_a[0], rng_a[d]) for d in range(1, D))
        all_same_b = all(np.array_equal(rng_b[0], rng_b[d]) for d in range(1, D))
        print(f"  All D={D} dims share same rng_a: {all_same_a}")
        print(f"  All D={D} dims share same rng_b: {all_same_b}")

        # Random matrices
        a = np.random.randn(N, D).astype(np.float64) * 0.5
        b = np.random.randn(M, D).astype(np.float64) * 0.5
        fp = a @ b.T

        # ── Current: shared Sobol (coherent noise) ──
        q_max = 2 ** (sc_prec - 1) - 1
        max_rng_val = 2 ** sc_prec

        def sc_matmul_numpy_fast(a_, b_, rng_a_, rng_b_):
            """Fast NumPy enable-signal SC matmul (vectorized over n,m)."""
            N_, D_ = a_.shape
            M_ = b_.shape[0]

            abs_max_a = np.maximum(np.abs(a_).max(axis=1, keepdims=True), 1e-5)
            scale_a_ = abs_max_a / q_max
            a_int_ = np.clip(np.round(a_ / scale_a_), -q_max, q_max)
            sign_a_ = np.sign(a_int_).astype(np.float64)
            sign_a_[sign_a_ == 0] = 1

            abs_max_b = np.maximum(np.abs(b_).max(axis=1, keepdims=True), 1e-5)
            scale_b_ = abs_max_b / q_max
            b_int_ = np.clip(np.round(b_ / scale_b_), -q_max, q_max)
            sign_b_ = np.sign(b_int_).astype(np.float64)
            sign_b_[sign_b_ == 0] = 1

            ba = np.round(np.abs(a_int_) * max_rng_val / q_max).astype(np.int32)
            bb = np.round(np.abs(b_int_) * max_rng_val / q_max).astype(np.int32)

            result_ = np.zeros((N_, M_), dtype=np.float64)
            q_max_sq = float(q_max * q_max)

            for d in range(D_):
                ra = rng_a_[d, :stoc_len]
                rb = rng_b_[d, :stoc_len]

                for n in range(N_):
                    k = int(np.sum(ra < ba[n, d]))
                    if k == 0:
                        continue
                    counts_m = np.array([np.sum(rb[:k] < bb[m_, d]) for m_ in range(M_)])
                    result_[n, :] += counts_m * q_max_sq / stoc_len * sign_a_[n, d] * sign_b_[:, d]

            result_ *= (scale_a_ * scale_b_.T)
            return result_

        # Shared Sobol
        res_shared = sc_matmul_numpy_fast(a, b, rng_a, rng_b)

        # ── Independent per-dim: randomly permute time axis per dimension ──
        rng_a_indep = rng_a.copy()
        rng_b_indep = rng_b.copy()
        rng_state = np.random.RandomState(123)
        for d in range(D):
            perm_a = rng_state.permutation(stoc_len)
            perm_b = rng_state.permutation(stoc_len)
            rng_a_indep[d] = rng_a[d, perm_a]
            rng_b_indep[d] = rng_b[d, perm_b]

        res_indep = sc_matmul_numpy_fast(a, b, rng_a_indep, rng_b_indep)

        fp_scale = np.abs(fp).mean()
        err_shared = np.abs(res_shared - fp)
        err_indep = np.abs(res_indep - fp)

        print(f"  FP result scale:                    {fp_scale:.4f}")
        print(f"  |shared_sobol - FP| mean/scale:     {err_shared.mean()/fp_scale:.6f}")
        print(f"  |indep_per_dim - FP| mean/scale:    {err_indep.mean()/fp_scale:.6f}")
        print(f"  ratio (shared / indep):              {err_shared.mean() / err_indep.mean():.2f}x")

        # Per-element error distributions
        errs_sh = (res_shared - fp).flatten()
        errs_in = (res_indep - fp).flatten()
        print(f"  shared:  mean_err={errs_sh.mean()/fp_scale:+.6f}  std={errs_sh.std()/fp_scale:.6f}")
        print(f"  indep:   mean_err={errs_in.mean()/fp_scale:+.6f}  std={errs_in.std()/fp_scale:.6f}")
        if errs_in.std() > 0:
            print(f"  std ratio (shared/indep):            {errs_sh.std()/errs_in.std():.2f}x")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="1",
                        help="Comma-separated: 1=CPU NumPy, 2=GPU Triton, 3=E2E accuracy")
    parser.add_argument("--data_root", type=str,
                        default="/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data")
    parser.add_argument("--n_eval", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    exps = [int(x) for x in args.exp.split(",")]

    if 1 in exps:
        exp1_cpu()

    if 4 in exps:
        exp4_coherent_noise()

    if 2 in exps or 3 in exps:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device == "cuda", "Experiments 2 and 3 need CUDA"
        if 2 in exps:
            exp2_gpu(device)
        if 3 in exps:
            exp3_e2e(args, device)


if __name__ == "__main__":
    main()
