"""
Benchmark: table-based vs compact (on-the-fly) enable-signal SC matmul.

Compares:
  1. Kernel-level timing for attention shapes (small D=72)
  2. Kernel-level timing for MLP shapes (large D=1152)
  3. End-to-end inference via quant_sc_main.py (optional, --e2e flag)

Usage:
    cd SC && python bench_table_vs_compact.py
    cd SC && python bench_table_vs_compact.py --e2e   # also run full inference
    cd SC && python bench_table_vs_compact.py --sc_prec 6  # test different precisions
"""
import argparse
import os
import sys
import time
import subprocess
import json

import torch
import torch.cuda

from config_helpers import make_sobol_simple_config
from sc_triton import (
    _get_cached_sequences,
    _get_cached_enable_tables,
    _get_cached_k_table,
    enable_matmul_triton,
    enable_matmul_compact,
    fused_quantize_bipolar,
    clear_rng_cache,
)


def kernel_benchmark(
    N, M, D, sc_prec, stoc_len, num_warmup=5, num_iters=50, label=""
):
    """Benchmark table vs compact kernel for a given shape."""
    config = make_sobol_simple_config(D, D, sc_prec)
    device = torch.device("cuda")
    q_max = 2 ** (sc_prec - 1) - 1
    q_max_sq = float(q_max * q_max)
    V = 2 ** sc_prec + 1

    # Generate random input
    torch.manual_seed(42)
    a = torch.randn(N, D, device=device) * 3.0
    b = torch.randn(M, D, device=device) * 3.0
    abs_max = max(a.abs().max().item(), b.abs().max().item(), 1e-5)

    # Quantize once (shared between both paths)
    boundary_a, sign_a, scale_a = fused_quantize_bipolar(a, abs_max, sc_prec)
    boundary_b, sign_b, scale_b = fused_quantize_bipolar(b, abs_max, sc_prec)

    # Get RNG sequences
    rand_seqs_a, rand_seqs_b = _get_cached_sequences(config, sc_prec, device)

    # Build table
    cum_indicator, k_table = _get_cached_enable_tables(
        config, sc_prec, device, rand_seqs_a, rand_seqs_b, stoc_len
    )
    rng_b = rand_seqs_b[:, :stoc_len].contiguous()
    k_table_only = _get_cached_k_table(
        config, sc_prec, device, rand_seqs_a, stoc_len
    )

    # ---- Table-based path ----
    # Warmup
    for _ in range(num_warmup):
        enable_matmul_triton(
            cum_indicator, k_table,
            boundary_a, boundary_b, sign_a, sign_b,
            N, M, D, stoc_len, q_max_sq, is_bipolar=True,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        enable_matmul_triton(
            cum_indicator, k_table,
            boundary_a, boundary_b, sign_a, sign_b,
            N, M, D, stoc_len, q_max_sq, is_bipolar=True,
        )
    end.record()
    torch.cuda.synchronize()
    table_ms = start.elapsed_time(end) / num_iters

    # ---- Compact path ----
    # Warmup
    for _ in range(num_warmup):
        enable_matmul_compact(
            rng_b, k_table_only,
            boundary_a, boundary_b, sign_a, sign_b,
            N, M, D, stoc_len, q_max_sq, is_bipolar=True,
        )
    torch.cuda.synchronize()

    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start2.record()
    for _ in range(num_iters):
        enable_matmul_compact(
            rng_b, k_table_only,
            boundary_a, boundary_b, sign_a, sign_b,
            N, M, D, stoc_len, q_max_sq, is_bipolar=True,
        )
    end2.record()
    torch.cuda.synchronize()
    compact_ms = start2.elapsed_time(end2) / num_iters

    # ---- FP matmul baseline ----
    a_f = a.float()
    b_f = b.float()
    for _ in range(num_warmup):
        _ = a_f @ b_f.T
    torch.cuda.synchronize()

    start3 = torch.cuda.Event(enable_timing=True)
    end3 = torch.cuda.Event(enable_timing=True)
    start3.record()
    for _ in range(num_iters):
        _ = a_f @ b_f.T
    end3.record()
    torch.cuda.synchronize()
    fp_ms = start3.elapsed_time(end3) / num_iters

    # Correctness check
    result_table = enable_matmul_triton(
        cum_indicator, k_table,
        boundary_a, boundary_b, sign_a, sign_b,
        N, M, D, stoc_len, q_max_sq, is_bipolar=True,
    )
    result_compact = enable_matmul_compact(
        rng_b, k_table_only,
        boundary_a, boundary_b, sign_a, sign_b,
        N, M, D, stoc_len, q_max_sq, is_bipolar=True,
    )
    max_diff = (result_table - result_compact).abs().max().item()

    # Table memory footprint
    table_mem_mb = cum_indicator.numel() * 2 / (1024 * 1024)
    compact_mem_mb = rng_b.numel() * 4 / (1024 * 1024)  # int32

    speedup = table_ms / compact_ms if compact_ms > 0 else float('inf')

    print(f"  {label}")
    print(f"    Table:    {table_ms:.3f} ms   (mem: {table_mem_mb:.1f} MB cum_indicator)")
    print(f"    Compact:  {compact_ms:.3f} ms   (mem: {compact_mem_mb:.1f} MB rng_b)")
    print(f"    FP32:     {fp_ms:.3f} ms")
    print(f"    Compact/Table speedup: {speedup:.2f}x   (>1 = compact faster)")
    print(f"    Max diff (table vs compact): {max_diff:.2e}")
    print()

    return {
        "label": label, "N": N, "M": M, "D": D,
        "sc_prec": sc_prec, "stoc_len": stoc_len,
        "table_ms": table_ms, "compact_ms": compact_ms, "fp_ms": fp_ms,
        "speedup": speedup, "max_diff": max_diff,
        "table_mem_mb": table_mem_mb, "compact_mem_mb": compact_mem_mb,
    }


def run_e2e_benchmark(sc_prec=8, num_steps=20, batch_size=8):
    """Run full inference twice: once with table path, once with compact path."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qdit_dir = os.path.join(project_root, "Q-DiT")
    script = os.path.join(qdit_dir, "scripts", "quant_sc_main.py")

    base_args = [
        sys.executable, script,
        "--wbits", "8", "--abits", "8",
        "--w_sym", "--a_sym",
        "--timewise", "1.0",
        "--qklayerwise", "1.0",
        "--sc_prec", str(sc_prec),
        "--sc_enable",
        "--image-size", "256",
        "--num-sampling-steps", str(num_steps),
        "--batch-size", str(batch_size),
        "--cfg-scale", "1.5",
        "--results-dir", os.path.join(project_root, "results_bench"),
    ]

    results = {}

    for path_name, env_var in [("table", "SC_FORCE_TABLE"), ("compact", "SC_FORCE_COMPACT")]:
        print(f"\n{'='*60}")
        print(f"End-to-end: {path_name} path (num_steps={num_steps}, batch={batch_size})")
        print(f"{'='*60}")

        env = os.environ.copy()
        env[env_var] = "1"
        # Unset the opposite
        for v in ["SC_FORCE_TABLE", "SC_FORCE_COMPACT"]:
            if v != env_var:
                env.pop(v, None)

        clear_rng_cache()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        proc = subprocess.run(
            base_args, cwd=qdit_dir, env=env,
            capture_output=True, text=True,
        )
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"  FAILED (exit code {proc.returncode})")
            print(f"  stderr: {proc.stderr[-500:]}")
            results[path_name] = {"elapsed_s": -1, "error": proc.stderr[-200:]}
        else:
            print(f"  Wall time: {elapsed:.1f}s")
            results[path_name] = {"elapsed_s": elapsed}

    # Summary
    if results.get("table", {}).get("elapsed_s", -1) > 0 and \
       results.get("compact", {}).get("elapsed_s", -1) > 0:
        t = results["table"]["elapsed_s"]
        c = results["compact"]["elapsed_s"]
        print(f"\n{'='*60}")
        print(f"E2E Summary (sc_prec={sc_prec}, steps={num_steps}, batch={batch_size}):")
        print(f"  Table path:   {t:.1f}s")
        print(f"  Compact path: {c:.1f}s")
        print(f"  Speedup (compact/table): {t/c:.2f}x  (>1 = compact faster)")
        print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark table vs compact SC kernel")
    parser.add_argument("--sc_prec", type=int, default=8)
    parser.add_argument("--e2e", action="store_true", help="Also run end-to-end inference")
    parser.add_argument("--e2e_steps", type=int, default=20, help="Diffusion steps for e2e")
    parser.add_argument("--e2e_batch", type=int, default=8, help="Batch size for e2e")
    parser.add_argument("--iters", type=int, default=50, help="Kernel benchmark iterations")
    args = parser.parse_args()

    sc_prec = args.sc_prec
    stoc_len = 2 ** sc_prec

    print(f"{'='*60}")
    print(f"SC Kernel Benchmark: Table vs Compact")
    print(f"sc_prec={sc_prec}, stoc_len={stoc_len}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"{'='*60}\n")

    # --- Kernel-level benchmarks ---
    all_results = []

    print("--- Attention shapes (D=72, DiT-XL head_dim) ---\n")
    attention_shapes = [
        (32, 32, 72, "N=M=32  (small batch)"),
        (256, 256, 72, "N=M=256 (typical 256px: 1024/4=256 tokens per head)"),
        (512, 512, 72, "N=M=512 (half of 512px tokens)"),
        (1024, 1024, 72, "N=M=1024 (512px: 4096/4=1024 tokens per head)"),
    ]
    for N, M, D, label in attention_shapes:
        r = kernel_benchmark(N, M, D, sc_prec, stoc_len,
                             num_iters=args.iters, label=label)
        all_results.append(r)

    # Test different sc_prec values to see how table size affects perf
    print("--- sc_prec sweep (N=M=256, D=72) ---\n")
    for sp in [4, 5, 6, 7, 8]:
        sl = 2 ** sp
        label = f"sc_prec={sp}, stoc_len={sl}"
        r = kernel_benchmark(256, 256, 72, sp, sl,
                             num_iters=args.iters, label=label)
        all_results.append(r)

    # MLP shapes (compact path is always used, but benchmark table for comparison)
    print("--- MLP shapes (large D, compact always used) ---\n")
    mlp_shapes = [
        (256, 1152, 1152, "D=1152 (DiT-XL hidden, proj/input_proj)"),
        (256, 4608, 1152, "N=256,M=4608,D=1152 (MLP fc1)"),
    ]
    for N, M, D, label in mlp_shapes:
        cum_bytes = D * (stoc_len + 1) * (2**sc_prec + 1) * 2
        print(f"  {label}")
        print(f"    cum_indicator would be {cum_bytes / 1e6:.0f} MB — "
              f"{'compact forced' if cum_bytes > 64*1024*1024 else 'table fits'}")
        # Only benchmark compact for large D (table won't fit or is impractical)
        config = make_sobol_simple_config(min(D, 128), min(D, 128), sc_prec)
        print(f"    (Skipping head-to-head — table doesn't fit for D={D})\n")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "bench_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Kernel results saved to {out_path}\n")

    # --- End-to-end benchmark ---
    if args.e2e:
        e2e_results = run_e2e_benchmark(
            sc_prec=sc_prec,
            num_steps=args.e2e_steps,
            batch_size=args.e2e_batch,
        )
        # Append to saved results
        with open(out_path, "r") as f:
            data = json.load(f)
        data.append({"type": "e2e", **e2e_results})
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
