"""
Calibrate noise-injection surrogate against real SC matmul.

Goal: figure out whether SC's per-element error grows with the output
magnitude (favours option A: out *= (1 + p2*N)) or is roughly a flat
floor (favours option B: out += p2*sigma*N).

Pipeline:
  1. Generate random Q, K matrices (DiT-attention-ish: standard normal).
  2. Compute exact = Q @ K^T (ground truth).
  3. Compute "real" SC matmul for several stream lengths L using a
     pure-PyTorch bipolar AND-gate reference (independent RNG per D).
  4. Fit p1 of the surrogate to match real SC's RMSE for each L.
  5. Compare three things side-by-side at one chosen L:
        - real SC error scatter:        |err| vs |exact|
        - surrogate A error scatter
        - surrogate B error scatter
  6. Print Pearson r between |err| and |exact| for real SC across L.
     - r >> 0  → error tracks magnitude → choose option A
     - r ~ 0   → flat floor             → choose option B

Outputs (next to this script):
  noise_model_scatter.png      — three-panel scatter
  noise_model_rmse_vs_L.png    — real SC RMSE vs L (sanity check 1/sqrt(L))
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Reference: real SC matmul (bipolar sign-magnitude, AND gate, independent RNG)
# =============================================================================
@torch.no_grad()
def real_sc_matmul(a: torch.Tensor, b: torch.Tensor, L: int, sc_prec: int = 8) -> torch.Tensor:
    """
    Bipolar sign-magnitude SC matmul: a @ b^T with stream length L.

    Inputs:
        a: (N, D)
        b: (M, D)
    Returns:
        (N, M)
    """
    N, D = a.shape
    M = b.shape[0]
    q_max = 2 ** (sc_prec - 1) - 1  # 127 for sc_prec=8

    a_max = a.abs().max().item()
    b_max = b.abs().max().item()
    a_scale = a_max / q_max
    b_scale = b_max / q_max

    a_int = (a / a_scale).round().clamp(-q_max, q_max)
    b_int = (b / b_scale).round().clamp(-q_max, q_max)

    sign_a = torch.sign(a_int)            # (N, D)
    sign_b = torch.sign(b_int)            # (M, D)
    mag_a = a_int.abs().long()            # (N, D)
    mag_b = b_int.abs().long()            # (M, D)

    # Independent RNG per (D,), shared across N for a, across M for b
    rng_a = torch.randint(0, q_max, (D, L))   # (D, L)
    rng_b = torch.randint(0, q_max, (D, L))   # (D, L)

    # Bit streams: (mag > rng) → 1
    a_bits = (mag_a.unsqueeze(-1) > rng_a.unsqueeze(0)).float()  # (N, D, L)
    b_bits = (mag_b.unsqueeze(-1) > rng_b.unsqueeze(0)).float()  # (M, D, L)

    # AND gate, sum over L, then sum over D
    counts = torch.einsum('ndt,mdt->nmd', a_bits, b_bits)  # (N, M, D)

    # Decode to magnitude product
    decoded = counts * (q_max * q_max) / float(L)          # (N, M, D)

    # Apply signs and reduce over embedding dim
    signed = decoded * sign_a.unsqueeze(1) * sign_b.unsqueeze(0)
    result = signed.sum(dim=2) * (a_scale * b_scale)
    return result


# =============================================================================
# Surrogates
# =============================================================================
@torch.no_grad()
def surrogate_A(a: torch.Tensor, b: torch.Tensor, p1: float, p2: float) -> torch.Tensor:
    """Output multiplicative: out *= (1 + p2*N)."""
    a_n = a * (1 + p1 * torch.randn_like(a))
    b_n = b * (1 + p1 * torch.randn_like(b))
    out = a_n @ b_n.transpose(-2, -1)
    if p2 > 0:
        out = out * (1 + p2 * torch.randn_like(out))
    return out


@torch.no_grad()
def surrogate_B(a: torch.Tensor, b: torch.Tensor, p1: float, p2: float) -> torch.Tensor:
    """Output additive: out += p2 * sigma_out * N (sigma_out = std of out)."""
    a_n = a * (1 + p1 * torch.randn_like(a))
    b_n = b * (1 + p1 * torch.randn_like(b))
    out = a_n @ b_n.transpose(-2, -1)
    if p2 > 0:
        sigma = out.detach().std()
        out = out + p2 * sigma * torch.randn_like(out)
    return out


# =============================================================================
# Driver
# =============================================================================
def make_data(seed: int, N: int, M: int, D: int):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(N, D, generator=g)
    b = torch.randn(M, D, generator=g)
    return a, b


def pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm() + 1e-12))


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # ----- Setup -----
    N, M, D = 32, 32, 128
    n_trials = 10
    L_values = [16, 32, 64, 128, 256]

    # ----- 1) Real SC RMSE per L -----
    print("=== Real SC RMSE vs L (10 trials, N=M=32, D=128) ===")
    sc_rmse_per_L = {}
    for L in L_values:
        rmses = []
        for trial in range(n_trials):
            a, b = make_data(trial, N, M, D)
            exact = a @ b.T
            sc_out = real_sc_matmul(a, b, L)
            err = sc_out - exact
            rmses.append(err.pow(2).mean().sqrt().item())
        sc_rmse_per_L[L] = float(np.mean(rmses))
        print(f"  L={L:>4d}  RMSE = {sc_rmse_per_L[L]:.4f}")

    # ----- 2) Fit p1 (no p2) per L by matching RMSE -----
    print("\n=== Best p1 (with p2=0) per L (matching real SC RMSE) ===")
    p1_grid = np.linspace(0.005, 0.30, 30)
    best_p1_per_L = {}
    for L in L_values:
        target = sc_rmse_per_L[L]
        best_p1, best_diff = None, float('inf')
        for p1 in p1_grid:
            rmses = []
            for trial in range(n_trials):
                a, b = make_data(trial, N, M, D)
                exact = a @ b.T
                out = surrogate_A(a, b, p1=float(p1), p2=0.0)
                rmses.append((out - exact).pow(2).mean().sqrt().item())
            diff = abs(np.mean(rmses) - target)
            if diff < best_diff:
                best_diff = diff
                best_p1 = float(p1)
        best_p1_per_L[L] = best_p1
        print(f"  L={L:>4d}  best p1={best_p1:.3f}  (target RMSE {target:.4f})")

    # ----- 3) Pearson r between |err| and |exact| for real SC -----
    print("\n=== Pearson r ( |error|, |exact| ) for real SC ===")
    print("  r >> 0 → error tracks magnitude → favours option A")
    print("  r ~ 0  → flat error floor       → favours option B")
    r_per_L = {}
    for L in L_values:
        rs = []
        for trial in range(n_trials):
            a, b = make_data(trial, N, M, D)
            exact = a @ b.T
            sc_out = real_sc_matmul(a, b, L)
            mag = exact.flatten().abs()
            err = (sc_out - exact).flatten().abs()
            rs.append(pearson_r(mag, err))
        r_per_L[L] = float(np.mean(rs))
        print(f"  L={L:>4d}  r = {r_per_L[L]:+.3f}")

    # Same correlation for surrogates A and B (sanity)
    L_show = 64
    p1_match = best_p1_per_L[L_show]
    print(f"\n=== Same correlation for surrogates (at L={L_show}, p1={p1_match:.3f}, p2=0.05) ===")
    rs_A, rs_B = [], []
    for trial in range(n_trials):
        a, b = make_data(trial, N, M, D)
        exact = a @ b.T
        mag = exact.flatten().abs()
        err_A = (surrogate_A(a, b, p1_match, 0.05) - exact).flatten().abs()
        err_B = (surrogate_B(a, b, p1_match, 0.05) - exact).flatten().abs()
        rs_A.append(pearson_r(mag, err_A))
        rs_B.append(pearson_r(mag, err_B))
    print(f"  surrogate A  r = {np.mean(rs_A):+.3f}")
    print(f"  surrogate B  r = {np.mean(rs_B):+.3f}")

    # ----- 4) Three-panel scatter at L_show -----
    a, b = make_data(0, N, M, D)
    exact = a @ b.T
    mag = exact.flatten().abs()
    sc_out = real_sc_matmul(a, b, L_show)
    err_sc = (sc_out - exact).flatten().abs()
    out_A = surrogate_A(a, b, p1=p1_match, p2=0.05)
    err_A = (out_A - exact).flatten().abs()
    out_B = surrogate_B(a, b, p1=p1_match, p2=0.05)
    err_B = (out_B - exact).flatten().abs()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    common_ymax = max(err_sc.max().item(), err_A.max().item(), err_B.max().item()) * 1.05
    common_xmax = mag.max().item() * 1.05

    axes[0].scatter(mag, err_sc, s=4, alpha=0.4, color='black')
    axes[0].set_title(f'Real SC (L={L_show}, r={r_per_L[L_show]:+.2f})')

    axes[1].scatter(mag, err_A, s=4, alpha=0.4, color='#1f77b4')
    axes[1].set_title(f'A: out *= (1 + p2·N)\np1={p1_match:.3f}, p2=0.05, r={np.mean(rs_A):+.2f}')

    axes[2].scatter(mag, err_B, s=4, alpha=0.4, color='#d62728')
    axes[2].set_title(f'B: out += p2·σ·N\np1={p1_match:.3f}, p2=0.05, r={np.mean(rs_B):+.2f}')

    for ax in axes:
        ax.set_xlabel('|exact|')
        ax.set_xlim(0, common_xmax)
        ax.set_ylim(0, common_ymax)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('|error|')

    fig.suptitle('Per-element |error| vs |exact|: does the error track the output magnitude?')
    fig.tight_layout()
    scatter_path = os.path.join(out_dir, 'noise_model_scatter.png')
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved scatter plot to {scatter_path}")

    # ----- 5) RMSE vs L sanity check -----
    fig, ax = plt.subplots(figsize=(7, 5))
    Ls = list(L_values)
    sc_rmses = [sc_rmse_per_L[L] for L in Ls]
    ref_idx = 2
    ref = sc_rmses[ref_idx]
    ref_L = Ls[ref_idx]
    theory = [ref * math.sqrt(ref_L / L) for L in Ls]

    ax.loglog(Ls, sc_rmses, 'o-', label='Real SC RMSE', color='black')
    ax.loglog(Ls, theory, '--', label=r'$1/\sqrt{L}$ scaling', color='gray')
    ax.set_xlabel('Stream length L')
    ax.set_ylabel('RMSE')
    ax.set_title('Real SC RMSE vs L (sanity)')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    rmse_path = os.path.join(out_dir, 'noise_model_rmse_vs_L.png')
    fig.savefig(rmse_path, dpi=150)
    plt.close(fig)
    print(f"Saved RMSE-vs-L plot to {rmse_path}")

    # ----- Recommendation -----
    avg_r = float(np.mean(list(r_per_L.values())))
    print("\n=== Verdict ===")
    print(f"Mean Pearson r across L: {avg_r:+.3f}")
    if avg_r > 0.3:
        print("→ Real SC error grows with output magnitude. Use option A (multiplicative).")
    elif avg_r < 0.1:
        print("→ Real SC error is roughly a flat floor. Use option B (additive).")
    else:
        print("→ Mixed regime. Option A with small p2, or A+B combined, would be safer.")


if __name__ == "__main__":
    main()
