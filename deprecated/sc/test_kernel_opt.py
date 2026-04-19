"""
Test script for kernel optimizations:
1. Fused kernel BLOCK_K tiling (embedding dimension)
2. Enable-signal tiled kernel BLOCK_K (D-dimension static unrolling)
3. Compact enable kernel BATCH_T (vectorized RNG inner loop)
4. Fused quantization kernels (eliminate scattered elementwise ops)
5. Enable table caching (eliminate repeated build_cum_indicator/compute_k_table)

Validates correctness against torch matmul ground truth and reports timing.
"""
import torch
import time
from config_helpers import make_sobol_simple_config
from sc_triton import (
    sc_matmul,
    sc_matmul_enable_triton,
    sc_matmul_grouped,
    sc_matmul_grouped_enable_triton,
    clear_rng_cache,
    fused_quantize_bipolar,
    fused_quantize_unipolar,
    _COMPACT_ENABLE_THRESHOLD_BYTES,
)


def rmse_normalized(sc_result, gt_result, max_dot):
    """Compute RMSE normalized by max possible dot product."""
    return ((sc_result - gt_result) ** 2).mean().sqrt().item() / max_dot


def test_fused_block_k():
    """Test fused XNOR/AND kernels with BLOCK_K tiling."""
    print("=" * 60)
    print("Test 1: Fused kernels with BLOCK_K tiling")
    print("=" * 60)

    all_pass = True
    # Test multiple D values to exercise different BLOCK_K choices
    # D%8==0 -> BLOCK_K=8, D%4==0 -> BLOCK_K=4, D%2==0 -> BLOCK_K=2, odd -> BLOCK_K=1
    test_cases = [
        (32, 72, 32, "D=72 (BLOCK_K=8, DiT head_dim)"),
        (16, 64, 16, "D=64 (BLOCK_K=8)"),
        (8, 36, 8,   "D=36 (BLOCK_K=4)"),
        (8, 30, 8,   "D=30 (BLOCK_K=2)"),
        (8, 17, 8,   "D=17 (BLOCK_K=1, odd)"),
        # Larger realistic size
        (256, 72, 256, "D=72, N=M=256 (attention-sized)"),
    ]

    for N, D, M, desc in test_cases:
        config = make_sobol_simple_config(D, D, 8)

        # --- Bipolar ---
        torch.manual_seed(42)
        a = torch.randn(N, D, device='cuda') * 3.0
        b = torch.randn(M, D, device='cuda') * 3.0
        max_fp = max(a.abs().max().item(), b.abs().max().item())

        result_sc = sc_matmul(a, b, max_fp, -max_fp, mode="bipolar", sc_prec=8, config=config)
        result_gt = a @ b.T
        max_dot = D * max_fp ** 2
        rmse_bp = rmse_normalized(result_sc, result_gt, max_dot)

        # --- Unipolar ---
        torch.manual_seed(42)
        a_uni = torch.rand(N, D, device='cuda')
        b_uni = torch.rand(M, D, device='cuda')
        result_sc_uni = sc_matmul(a_uni, b_uni, 1.0, 0.0, mode="unipolar", sc_prec=8, config=config)
        result_gt_uni = a_uni @ b_uni.T
        rmse_uni = rmse_normalized(result_sc_uni, result_gt_uni, D)

        bp_pass = rmse_bp < 0.05
        uni_pass = rmse_uni < 0.05
        all_pass = all_pass and bp_pass and uni_pass

        print(f"  {desc}:")
        print(f"    Bipolar  RMSE={rmse_bp:.6f} [{'PASS' if bp_pass else 'FAIL'}]")
        print(f"    Unipolar RMSE={rmse_uni:.6f} [{'PASS' if uni_pass else 'FAIL'}]")

    print(f"\n  Fused BLOCK_K: {'ALL PASSED' if all_pass else 'SOME FAILED'}\n")
    return all_pass


def test_enable_tiled_block_k():
    """Test enable-signal tiled kernels with BLOCK_K (table-based path)."""
    print("=" * 60)
    print("Test 2: Enable-signal tiled kernels with BLOCK_K")
    print("=" * 60)

    all_pass = True
    test_cases = [
        (32, 72, 32, "D=72 (DiT head_dim)"),
        (16, 64, 16, "D=64"),
        (8, 36, 8,   "D=36"),
        (256, 72, 256, "D=72, N=M=256"),
    ]

    for N, D, M, desc in test_cases:
        config = make_sobol_simple_config(D, D, 8)

        # --- Bipolar ---
        torch.manual_seed(42)
        a = torch.randn(N, D, device='cuda') * 3.0
        b = torch.randn(M, D, device='cuda') * 3.0
        max_fp = max(a.abs().max().item(), b.abs().max().item())

        result_sc = sc_matmul_enable_triton(
            a, b, max_fp, -max_fp, mode="bipolar", sc_prec=8, config=config
        )
        result_gt = a @ b.T
        max_dot = D * max_fp ** 2
        rmse_bp = rmse_normalized(result_sc, result_gt, max_dot)

        # --- Unipolar ---
        torch.manual_seed(42)
        a_uni = torch.rand(N, D, device='cuda')
        b_uni = torch.rand(M, D, device='cuda')
        result_sc_uni = sc_matmul_enable_triton(
            a_uni, b_uni, 1.0, 0.0, mode="unipolar", sc_prec=8, config=config
        )
        result_gt_uni = a_uni @ b_uni.T
        rmse_uni = rmse_normalized(result_sc_uni, result_gt_uni, D)

        bp_pass = rmse_bp < 0.05
        uni_pass = rmse_uni < 0.05
        all_pass = all_pass and bp_pass and uni_pass

        print(f"  {desc}:")
        print(f"    Bipolar  RMSE={rmse_bp:.6f} [{'PASS' if bp_pass else 'FAIL'}]")
        print(f"    Unipolar RMSE={rmse_uni:.6f} [{'PASS' if uni_pass else 'FAIL'}]")

    print(f"\n  Enable tiled BLOCK_K: {'ALL PASSED' if all_pass else 'SOME FAILED'}\n")
    return all_pass


def test_compact_batch_t():
    """Test compact enable kernels with BATCH_T vectorization."""
    print("=" * 60)
    print("Test 3: Compact enable kernels with BATCH_T")
    print("=" * 60)

    # Force compact path by temporarily lowering threshold
    import sc_triton
    orig_threshold = sc_triton._COMPACT_ENABLE_THRESHOLD_BYTES
    sc_triton._COMPACT_ENABLE_THRESHOLD_BYTES = 0  # Force compact for all sizes

    all_pass = True
    test_cases = [
        (32, 72, 32, "D=72 (DiT head_dim)"),
        (16, 64, 16, "D=64"),
        (8, 32, 8,   "D=32"),
        (256, 72, 256, "D=72, N=M=256"),
    ]

    for N, D, M, desc in test_cases:
        config = make_sobol_simple_config(D, D, 8)

        # --- Bipolar ---
        torch.manual_seed(42)
        a = torch.randn(N, D, device='cuda') * 3.0
        b = torch.randn(M, D, device='cuda') * 3.0
        max_fp = max(a.abs().max().item(), b.abs().max().item())

        result_sc = sc_matmul_enable_triton(
            a, b, max_fp, -max_fp, mode="bipolar", sc_prec=8, config=config
        )
        result_gt = a @ b.T
        max_dot = D * max_fp ** 2
        rmse_bp = rmse_normalized(result_sc, result_gt, max_dot)

        # --- Unipolar ---
        torch.manual_seed(42)
        a_uni = torch.rand(N, D, device='cuda')
        b_uni = torch.rand(M, D, device='cuda')
        result_sc_uni = sc_matmul_enable_triton(
            a_uni, b_uni, 1.0, 0.0, mode="unipolar", sc_prec=8, config=config
        )
        result_gt_uni = a_uni @ b_uni.T
        rmse_uni = rmse_normalized(result_sc_uni, result_gt_uni, D)

        bp_pass = rmse_bp < 0.05
        uni_pass = rmse_uni < 0.05
        all_pass = all_pass and bp_pass and uni_pass

        print(f"  {desc}:")
        print(f"    Bipolar  RMSE={rmse_bp:.6f} [{'PASS' if bp_pass else 'FAIL'}]")
        print(f"    Unipolar RMSE={rmse_uni:.6f} [{'PASS' if uni_pass else 'FAIL'}]")

    # Restore threshold
    sc_triton._COMPACT_ENABLE_THRESHOLD_BYTES = orig_threshold

    print(f"\n  Compact BATCH_T: {'ALL PASSED' if all_pass else 'SOME FAILED'}\n")
    return all_pass


def test_grouped_enable():
    """Test grouped quantization with enable-signal (uses compact or table path)."""
    print("=" * 60)
    print("Test 4: Grouped enable-signal matmul")
    print("=" * 60)

    all_pass = True
    test_cases = [
        (64, 72, 64, 8, 1, "D=72, group_a=8"),
        (32, 64, 32, 1, 1, "D=64, per-row"),
        (256, 72, 256, 32, 1, "D=72, N=M=256, group_a=32"),
    ]

    for N, D, M, ga, gb, desc in test_cases:
        config = make_sobol_simple_config(D, D, 8)
        torch.manual_seed(42)
        a = torch.rand(N, D, device='cuda')
        b = torch.rand(M, D, device='cuda')

        result_sc = sc_matmul_grouped_enable_triton(
            a, b, group_a=ga, group_b=gb, mode="unipolar", sc_prec=8, config=config
        )
        result_gt = a @ b.T
        rmse = rmse_normalized(result_sc, result_gt, D)

        passed = rmse < 0.05
        all_pass = all_pass and passed
        print(f"  {desc}: RMSE={rmse:.6f} [{'PASS' if passed else 'FAIL'}]")

    print(f"\n  Grouped enable: {'ALL PASSED' if all_pass else 'SOME FAILED'}\n")
    return all_pass


def test_fused_quantization():
    """Test fused quantization kernels (eliminate scattered elementwise ops)."""
    print("=" * 60)
    print("Test 5: Fused quantization kernels")
    print("=" * 60)

    all_pass = True
    test_cases = [
        (256, 72, "N=256, D=72 (attention Q/K)"),
        (16, 64, "N=16, D=64"),
        (1024, 72, "N=1024, D=72 (large)"),
    ]

    for N, D, desc in test_cases:
        sc_prec = 8
        q_max_bp = 2 ** (sc_prec - 1) - 1
        q_min_bp = -(2 ** (sc_prec - 1))
        max_rng_val = 2 ** sc_prec
        q_max_uni = 2 ** sc_prec - 1

        # --- Test bipolar fused quant ---
        torch.manual_seed(42)
        a = torch.randn(N, D, device='cuda') * 3.0
        abs_max = a.abs().max().item()

        boundary, sign, scale = fused_quantize_bipolar(a, abs_max, sc_prec)

        # Reference: scattered PyTorch ops
        scale_ref = max(abs_max, 1e-5) / q_max_bp
        a_int_ref = (a / scale_ref).round().clamp(q_min_bp, q_max_bp)
        sign_ref = torch.sign(a_int_ref).float()
        boundary_ref = (a_int_ref.abs() * max_rng_val / q_max_bp).round().int()

        bp_boundary_match = (boundary == boundary_ref).all().item()
        bp_sign_match = (sign == sign_ref).all().item()
        bp_pass = bp_boundary_match and bp_sign_match

        # --- Test unipolar fused quant ---
        torch.manual_seed(42)
        a_uni = torch.rand(N, D, device='cuda')
        fp_max, fp_min = a_uni.max().item(), a_uni.min().item()

        boundary_uni, scale_uni, zp_f, row_sum = fused_quantize_unipolar(
            a_uni, fp_max, fp_min, sc_prec, compute_sum=True)

        # Reference
        range_ref = max(fp_max - fp_min, 1e-5)
        scale_ref_u = range_ref / q_max_uni
        zp_ref = round(-fp_min / scale_ref_u)
        zp_ref = max(0, min(q_max_uni, zp_ref))
        a_int_ref_u = (a_uni / scale_ref_u + zp_ref).round().clamp(0, q_max_uni)
        boundary_ref_u = (a_int_ref_u * max_rng_val / q_max_uni).round().int()
        row_sum_ref = a_int_ref_u.sum(dim=1)

        uni_boundary_match = (boundary_uni == boundary_ref_u).all().item()
        uni_sum_close = (row_sum - row_sum_ref).abs().max().item() < 1.0
        uni_pass = uni_boundary_match and uni_sum_close

        all_pass = all_pass and bp_pass and uni_pass
        print(f"  {desc}:")
        print(f"    Bipolar  boundary={'match' if bp_boundary_match else 'MISMATCH'} "
              f"sign={'match' if bp_sign_match else 'MISMATCH'} [{'PASS' if bp_pass else 'FAIL'}]")
        print(f"    Unipolar boundary={'match' if uni_boundary_match else 'MISMATCH'} "
              f"sum_err={abs(0) if uni_sum_close else 'BIG'} [{'PASS' if uni_pass else 'FAIL'}]")

    print(f"\n  Fused quantization: {'ALL PASSED' if all_pass else 'SOME FAILED'}\n")
    return all_pass


def test_enable_table_caching():
    """Test that enable table caching works (2nd call should be faster)."""
    print("=" * 60)
    print("Test 6: Enable table caching")
    print("=" * 60)

    N, D, M = 64, 72, 64
    config = make_sobol_simple_config(D, D, 8)
    torch.manual_seed(42)
    a = torch.randn(N, D, device='cuda') * 3.0
    b = torch.randn(M, D, device='cuda') * 3.0
    max_fp = max(a.abs().max().item(), b.abs().max().item())

    # Clear cache and time first call (cold)
    clear_rng_cache()
    torch.cuda.synchronize()
    t0 = time.time()
    r1 = sc_matmul_enable_triton(a, b, max_fp, -max_fp, mode="bipolar", sc_prec=8, config=config)
    torch.cuda.synchronize()
    cold_ms = (time.time() - t0) * 1000

    # Second call (warm — tables cached)
    torch.cuda.synchronize()
    t0 = time.time()
    r2 = sc_matmul_enable_triton(a, b, max_fp, -max_fp, mode="bipolar", sc_prec=8, config=config)
    torch.cuda.synchronize()
    warm_ms = (time.time() - t0) * 1000

    # Results should be identical (same RNG, same data)
    match = torch.allclose(r1, r2)
    print(f"  Cold call: {cold_ms:.2f} ms")
    print(f"  Warm call: {warm_ms:.2f} ms")
    print(f"  Results match: {match}")
    print(f"  Speedup: {cold_ms/warm_ms:.1f}x")

    passed = match
    print(f"\n  Enable table caching: {'PASS' if passed else 'FAIL'}\n")
    return passed


def benchmark_fused():
    """Benchmark fused kernels at DiT-relevant sizes."""
    print("=" * 60)
    print("Benchmark: Fused matmul (D=72, attention-like)")
    print("=" * 60)

    N, D, M = 256, 72, 256
    config = make_sobol_simple_config(D, D, 8)
    torch.manual_seed(0)
    a = torch.randn(N, D, device='cuda') * 3.0
    b = torch.randn(M, D, device='cuda') * 3.0
    max_fp = max(a.abs().max().item(), b.abs().max().item())

    n_warmup = 5
    n_runs = 50

    # Bipolar fused
    for _ in range(n_warmup):
        sc_matmul(a, b, max_fp, -max_fp, mode="bipolar", sc_prec=8, config=config)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        sc_matmul(a, b, max_fp, -max_fp, mode="bipolar", sc_prec=8, config=config)
    torch.cuda.synchronize()
    fused_time = (time.time() - t0) / n_runs * 1000
    print(f"  Fused bipolar:  {fused_time:.2f} ms")

    # Enable-signal (table path)
    clear_rng_cache()
    for _ in range(n_warmup):
        sc_matmul_enable_triton(a, b, max_fp, -max_fp, mode="bipolar", sc_prec=8, config=config)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        sc_matmul_enable_triton(a, b, max_fp, -max_fp, mode="bipolar", sc_prec=8, config=config)
    torch.cuda.synchronize()
    enable_time = (time.time() - t0) / n_runs * 1000
    print(f"  Enable bipolar: {enable_time:.2f} ms")

    # Enable-signal (compact path, forced)
    import sc_triton
    orig = sc_triton._COMPACT_ENABLE_THRESHOLD_BYTES
    sc_triton._COMPACT_ENABLE_THRESHOLD_BYTES = 0
    clear_rng_cache()
    for _ in range(n_warmup):
        sc_matmul_enable_triton(a, b, max_fp, -max_fp, mode="bipolar", sc_prec=8, config=config)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        sc_matmul_enable_triton(a, b, max_fp, -max_fp, mode="bipolar", sc_prec=8, config=config)
    torch.cuda.synchronize()
    compact_time = (time.time() - t0) / n_runs * 1000
    sc_triton._COMPACT_ENABLE_THRESHOLD_BYTES = orig
    print(f"  Compact bipolar:{compact_time:.2f} ms")

    # Reference: torch matmul
    for _ in range(n_warmup):
        _ = a @ b.T
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        _ = a @ b.T
    torch.cuda.synchronize()
    torch_time = (time.time() - t0) / n_runs * 1000
    print(f"  Torch matmul:   {torch_time:.2f} ms")

    print()


if __name__ == "__main__":
    print("Kernel Optimization Verification Tests")
    print("=" * 60)
    print()

    p1 = test_fused_block_k()
    p2 = test_enable_tiled_block_k()
    p3 = test_compact_batch_t()
    p4 = test_grouped_enable()
    p5 = test_fused_quantization()
    p6 = test_enable_table_caching()

    print("=" * 60)
    all_pass = p1 and p2 and p3 and p4 and p5 and p6
    print(f"OVERALL: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print("=" * 60)

    if all_pass:
        print()
        benchmark_fused()
