#!/usr/bin/env python3
"""Verify: k_shortcut table lookup == cycle-by-cycle FSUMul (exact match).

This proves that the enable-signal table lookup in sc_triton.py correctly
implements the FSUMul hardware semantics from uSystolic-Sim, where B's RNG
counter is gated by A's output bit.

Three-way comparison:
  1. cycle-by-cycle (sc_enable.py) — reference FSUMul simulation
  2. k_shortcut (sc_enable.py) — table-lookup acceleration of (1)
  3. Triton enable-signal (sc_triton.py) — production GPU kernel

All three should produce IDENTICAL results (same Sobol sequences, same math).
If they do, the implementation is correct and not the cause of SC degradation.
"""
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sc"))
sys.path.insert(0, str(ROOT))

import torch
torch.set_grad_enabled(False)

from config_helpers import make_sobol_simple_config
from sng import RNGPool, SNGBank
from sc_enable import (
    _enable_mul_cycle_by_cycle,
    _enable_mul_k_shortcut,
    _compute_boundary,
)
from sc_triton import (
    sc_matmul_grouped_enable_triton,
    _get_cached_sequences,
)


def main():
    device = "cuda"
    sc_prec = 8
    stoc_len = 2 ** sc_prec  # 256

    print("=" * 72)
    print("Verify: k_shortcut == cycle-by-cycle == Triton enable-signal")
    print("        (all must match → FSUMul implementation is correct)")
    print("=" * 72)

    for label, N, D, M in [
        ("QK head D=64", 32, 64, 32),
        ("AV D=258", 32, 258, 32),
        ("proj D=1024", 16, 1024, 16),
    ]:
        print(f"\n--- {label}: ({N},{D}) @ ({M},{D})^T ---")

        config = make_sobol_simple_config(D, D, sc_prec)

        # Get Sobol sequences
        rng_pool = RNGPool(config["rng_pool"], sc_prec)
        sng_a = SNGBank(rng_pool, config["sng"]["q"])
        sng_b = SNGBank(rng_pool, config["sng"]["k"])

        import numpy as np
        rng_a_np = sng_a.get_all_sequences(stoc_len)
        rng_b_np = sng_b.get_all_sequences(stoc_len)
        rng_a = torch.tensor(rng_a_np, dtype=torch.int32, device=device)
        rng_b = torch.tensor(rng_b_np, dtype=torch.int32, device=device)

        # Random input matrices
        torch.manual_seed(42)
        a_fp = torch.randn(N, D, device=device) * 0.5
        b_fp = torch.randn(M, D, device=device) * 0.5

        q_max = 2 ** (sc_prec - 1) - 1  # 127
        max_rng_val = 2 ** sc_prec       # 256

        # Per-row symmetric quantization (matching production path)
        abs_max_a = a_fp.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        scale_a = abs_max_a / q_max
        a_int = (a_fp / scale_a).round().clamp(-q_max, q_max)
        sign_a = a_int.sign().to(torch.int8)
        sign_a[sign_a == 0] = 1
        mag_a = a_int.abs().to(torch.int32)

        abs_max_b = b_fp.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        scale_b = abs_max_b / q_max
        b_int = (b_fp / scale_b).round().clamp(-q_max, q_max)
        sign_b = b_int.sign().to(torch.int8)
        sign_b[sign_b == 0] = 1
        mag_b = b_int.abs().to(torch.int32)

        # ── 1. Cycle-by-cycle (exact FSUMul simulation) ──
        counts_cbc = _enable_mul_cycle_by_cycle(
            mag_a, mag_b, rng_a, rng_b,
            max_val=float(q_max), sc_prec=sc_prec,
        )  # (N, M) raw counts

        # Decode: counts * q_max^2 / stoc_len, apply signs, dequant
        q_max_sq = float(q_max * q_max)
        # signs: need sign_a[n,d] * sign_b[m,d] summed over d
        # But cycle-by-cycle returns summed counts, not per-dim.
        # Need per-dim for sign handling. Re-run with return_per_dim=True.
        counts_cbc_pd = _enable_mul_cycle_by_cycle(
            mag_a, mag_b, rng_a, rng_b,
            max_val=float(q_max), sc_prec=sc_prec,
            return_per_dim=True,
        )  # (N, M, D)

        decoded_cbc = counts_cbc_pd.float() * q_max_sq / stoc_len
        # Apply signs per dim
        sign_product = sign_a.float().unsqueeze(1) * sign_b.float().unsqueeze(0)  # (N, M, D)
        result_cbc = (decoded_cbc * sign_product).sum(dim=2)
        result_cbc *= (scale_a * scale_b.t())  # (N, M)

        # ── 2. K-shortcut (table lookup) ──
        counts_ks_pd = _enable_mul_k_shortcut(
            mag_a, mag_b, rng_a, rng_b,
            max_val=float(q_max), sc_prec=sc_prec,
            return_per_dim=True,
        )  # (N, M, D)

        decoded_ks = counts_ks_pd.float() * q_max_sq / stoc_len
        result_ks = (decoded_ks * sign_product).sum(dim=2)
        result_ks *= (scale_a * scale_b.t())

        # ── 3. Triton enable-signal (production kernel) ──
        result_triton = sc_matmul_grouped_enable_triton(
            a_fp.float(), b_fp.float(),
            group_a=1, group_b=1,
            mode="bipolar", sc_prec=sc_prec, config=config,
        )

        # ── 4. FP ground truth ──
        result_fp = a_fp.float() @ b_fp.float().t()

        # ── Comparisons ──
        fp_scale = result_fp.abs().mean().item()

        diff_cbc_ks = (result_cbc - result_ks).abs()
        diff_cbc_triton = (result_cbc - result_triton).abs()
        diff_ks_triton = (result_ks - result_triton).abs()

        print(f"  FP scale: {fp_scale:.4f}")
        print(f"  |cycle-by-cycle - k_shortcut| / scale:  "
              f"mean={diff_cbc_ks.mean().item()/fp_scale:.2e}  "
              f"max={diff_cbc_ks.max().item()/fp_scale:.2e}")
        print(f"  |cycle-by-cycle - Triton|     / scale:  "
              f"mean={diff_cbc_triton.mean().item()/fp_scale:.2e}  "
              f"max={diff_cbc_triton.max().item()/fp_scale:.2e}")
        print(f"  |k_shortcut     - Triton|     / scale:  "
              f"mean={diff_ks_triton.mean().item()/fp_scale:.2e}  "
              f"max={diff_ks_triton.max().item()/fp_scale:.2e}")

        err_cbc = (result_cbc - result_fp).abs().mean().item() / fp_scale
        err_ks = (result_ks - result_fp).abs().mean().item() / fp_scale
        err_triton = (result_triton - result_fp).abs().mean().item() / fp_scale
        print(f"  |cycle-by-cycle - FP| / scale: {err_cbc:.6f}")
        print(f"  |k_shortcut     - FP| / scale: {err_ks:.6f}")
        print(f"  |Triton_enable  - FP| / scale: {err_triton:.6f}")

        match_cbc_ks = diff_cbc_ks.max().item() < 1e-4 * fp_scale
        match_cbc_tri = diff_cbc_triton.max().item() < 1e-4 * fp_scale
        print(f"  cycle-by-cycle == k_shortcut?  {'YES ✓' if match_cbc_ks else 'NO ✗'}")
        print(f"  cycle-by-cycle == Triton?      {'YES ✓' if match_cbc_tri else 'NO ✗'}")

        # Per-dim count comparison (most direct test)
        count_diff = (counts_cbc_pd - counts_ks_pd).abs()
        print(f"  Raw count diff (per-dim): max={count_diff.max().item()}, "
              f"mean={count_diff.float().mean().item():.4f}")

    print("\n" + "=" * 72)
    print("CONCLUSION")
    print("=" * 72)
    print("""
If cycle-by-cycle == k_shortcut (exact match on raw counts), then the
table-lookup correctly implements FSUMul semantics where B's RNG counter
is gated by A's output bit — matching the uSystolic-Sim hardware:

    sobol16 U_sobol_W(.enable(bitI), .sobolSeq(randW))

The enable-signal approach is NOT an approximation of AND/XNOR. It is a
DIFFERENT (and lower-variance) SC multiplication circuit. The "first k
natural-order positions" in rng_b correspond exactly to the first k
enable events in the FSUMul gated clock.
""")


if __name__ == "__main__":
    main()
