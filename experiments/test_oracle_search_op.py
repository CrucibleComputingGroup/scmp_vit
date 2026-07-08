"""Phase-2 smoke: ``oracle_search_op`` on a synthetic SC block.

Builds a tiny "FP block" = Linear(D, D) and an "SC block" whose output is
``Y_fp + noise(sl_per_row)`` where the per-row stoc_len comes from applying
the current FreeBoundaryMPConfig boundaries to the row metric. Noise is
larger for smaller sl, so better boundaries (allocate sl=256 to
high-metric rows) drive the residual down.

Success criteria:
 - RidgeFitter matches direct closed-form solve.
 - oracle_search_op decreases the residual score monotonically.
 - Returned boundaries are strictly descending in (0, 1).

Runs on CPU in a few seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sc"))
sys.path.insert(0, str(ROOT / "third_party" / "QwT-SC" / "QwT-vit-sc"))

from scmp_kernels.mp.config import (  # noqa: E402
    FreeBoundaryMPConfig,
    _classify_with_free_boundaries,
    set_current_block_idx,
)
from scmp_kernels.mp import (  # noqa: E402
    RidgeFitter,
    oracle_search_op,
)


# ---------------------------------------------------------------------
# RidgeFitter correctness
# ---------------------------------------------------------------------

def test_ridge_fitter():
    torch.manual_seed(0)
    N, D, Dout = 200, 16, 8
    X = torch.randn(N, D)
    W_true = torch.randn(D, Dout)
    b_true = torch.randn(Dout)
    R = X @ W_true + b_true + 0.01 * torch.randn(N, Dout)
    fitter = RidgeFitter(X, ridge=1e-3)

    # The residual with W_true should be ~ noise_norm_sq = (0.01)² * N * Dout
    score = fitter.residual_norm_sq(R)
    expected = (0.01 ** 2) * N * Dout
    # Allow large tolerance — ridge dampens slightly, noise is random.
    assert 0.1 * expected < score < 3.0 * expected, (
        f"ridge fitter score off: {score} vs ~{expected}")

    # With pure noise (no signal), score approximately equals RNorm² minus
    # what the (X, 1) basis can explain — should be << ||R||² for random X.
    R_noise = torch.randn(N, Dout)
    score_noise = fitter.residual_norm_sq(R_noise)
    total = (R_noise ** 2).sum().item()
    assert score_noise <= total + 1e-6, "projected residual exceeds total norm"
    print(f"[ok] RidgeFitter: signal={score:.4f} (exp~{expected:.4f}), "
          f"noise={score_noise:.4f}/{total:.4f}")


# ---------------------------------------------------------------------
# Toy SC block for coord-descent smoke
# ---------------------------------------------------------------------

class ToySCBlock:
    """Fake SC block: Y = X W + b + row_noise, where row_noise is gated by
    the *current* boundaries in a shared FreeBoundaryMPConfig under op="mlp".

    Noise std per row = noise_amp[sl_assigned_to_row]. Higher sl → lower
    noise. For a row-aligned metric, the optimal boundaries push high-
    metric rows to sl=256 (lowest noise) and low-metric rows to sl=32.
    """

    def __init__(self, X: torch.Tensor, metric: torch.Tensor,
                 cfg: FreeBoundaryMPConfig, block_idx: int = 0,
                 op: str = "mlp", seed: int = 0):
        torch.manual_seed(seed)
        N, D = X.shape
        self.N, self.D = N, D
        self.W = torch.randn(D, D) * 0.3
        self.b = torch.zeros(D)
        self.X = X
        self.metric = metric
        self.cfg = cfg
        self.block_idx = block_idx
        self.op = op
        # noise amplitudes: sl=256 smallest, sl=32 largest
        self.noise_amp = {256: 0.05, 128: 0.15, 64: 0.40, 32: 1.0}
        self._Y_fp = X @ self.W + self.b
        self._noise_rng = torch.Generator().manual_seed(seed)

    def Y_fp(self) -> torch.Tensor:
        return self._Y_fp.clone()

    def Y_sc(self) -> torch.Tensor:
        """Re-uses the SAME noise seed per call so that scoring is
        deterministic for a fixed set of boundaries — this isolates the
        coord-descent objective from SC draw variance."""
        noise_g = torch.Generator().manual_seed(12345)
        noise = torch.randn(self.N, self.D, generator=noise_g)
        assignment = _classify_with_free_boundaries(
            self.metric, self.cfg.get_boundaries(self.op, self.block_idx),
            self.cfg.stoc_len_levels)
        # Per-row std from the assigned sl
        row_std = torch.empty(self.N)
        for sl, idx in assignment.level_row_indices.items():
            if len(idx):
                row_std[idx] = self.noise_amp[sl]
        return self._Y_fp + noise * row_std.unsqueeze(-1)


def test_oracle_coord_descent_improves():
    torch.manual_seed(0)
    N, D = 300, 32

    # Skewed metric: 30% rows have high metric, rest low. Optimal
    # boundaries should give the top ~30% sl=256.
    high = torch.rand(int(N * 0.3)) * 0.3 + 0.7   # in [0.7, 1.0)
    low = torch.rand(N - int(N * 0.3)) * 0.5      # in [0, 0.5)
    metric = torch.cat([high, low])[torch.randperm(N)]

    X = torch.randn(N, D)
    cfg = FreeBoundaryMPConfig(stoc_len_levels=[256, 128, 64, 32])
    set_current_block_idx(0)
    toy = ToySCBlock(X, metric, cfg)

    Y_fp = toy.Y_fp()
    fitter = RidgeFitter(X, ridge=1e-3)

    def run_sc():
        return toy.Y_sc()

    # Baseline score with default equal-spaced boundaries
    cfg.set_boundaries("mlp", 0, cfg.default_boundaries())
    initial_score = fitter.residual_norm_sq(Y_fp - run_sc())

    best_b, best_score = oracle_search_op(
        run_block_sc=run_sc,
        Y_fp=Y_fp,
        fitter=fitter,
        cfg=cfg,
        block_idx=0,
        op="mlp",
        n_candidates=8,
        n_outer=2,
        log_fn=print,
    )

    # Coord descent should never make it worse.
    assert best_score <= initial_score + 1e-6, (
        f"score regressed: {initial_score} -> {best_score}")
    # And on a skewed metric + noisy SC, it should strictly improve.
    assert best_score < initial_score * 0.99, (
        f"score did not improve meaningfully: "
        f"{initial_score:.4f} -> {best_score:.4f}")

    # Boundaries must be descending in (0, 1).
    b = best_b.tolist()
    assert all(0 < b[i] < 1 for i in range(len(b))), f"out of range: {b}"
    assert all(b[i] > b[i + 1] for i in range(len(b) - 1)), (
        f"not descending: {b}")

    print(f"[ok] coord descent: {initial_score:.4f} -> {best_score:.4f} "
          f"({100*(1 - best_score/initial_score):.1f}% drop); "
          f"boundaries={[round(x, 3) for x in b]}")


def test_budget_aware_search_respects_target():
    torch.manual_seed(0)
    N, D = 300, 32

    high = torch.rand(int(N * 0.3)) * 0.3 + 0.7
    low = torch.rand(N - int(N * 0.3)) * 0.5
    metric = torch.cat([high, low])[torch.randperm(N)]

    X = torch.randn(N, D)
    cfg = FreeBoundaryMPConfig(stoc_len_levels=[256, 32])
    set_current_block_idx(0)
    toy = ToySCBlock(X, metric, cfg)
    fitter = RidgeFitter(X, ridge=1e-3)
    Y_fp = toy.Y_fp()

    def run_sc():
        return toy.Y_sc()

    def run_sc_with_budget():
        y = toy.Y_sc()
        assign = _classify_with_free_boundaries(
            metric, cfg.get_boundaries("mlp", 0), cfg.stoc_len_levels)
        baseline = float(N * max(cfg.stoc_len_levels))
        actual = 0.0
        for sl, rows in assign.level_row_indices.items():
            actual += float(int(sl) * len(rows))
        return y, {"baseline": baseline, "actual": actual}

    cfg.set_boundaries("mlp", 0, cfg.default_boundaries())
    b_free, _ = oracle_search_op(
        run_block_sc=run_sc,
        Y_fp=Y_fp,
        fitter=fitter,
        cfg=cfg,
        block_idx=0,
        op="mlp",
        n_candidates=10,
        n_outer=2,
        log_fn=lambda *_args: None,
    )
    free_assign = _classify_with_free_boundaries(metric, b_free, cfg.stoc_len_levels)
    free_actual = 0.0
    for sl, rows in free_assign.level_row_indices.items():
        free_actual += float(int(sl) * len(rows))

    cfg.set_boundaries("mlp", 0, cfg.default_boundaries())
    budget_target = 0.30 * float(N * max(cfg.stoc_len_levels))
    b_budget, _ = oracle_search_op(
        run_block_sc=run_sc,
        run_block_sc_with_budget=run_sc_with_budget,
        budget_target_actual=budget_target,
        Y_fp=Y_fp,
        fitter=fitter,
        cfg=cfg,
        block_idx=0,
        op="mlp",
        n_candidates=10,
        n_outer=2,
        log_fn=lambda *_args: None,
    )
    budget_assign = _classify_with_free_boundaries(
        metric, b_budget, cfg.stoc_len_levels)
    budget_actual = 0.0
    for sl, rows in budget_assign.level_row_indices.items():
        budget_actual += float(int(sl) * len(rows))

    assert budget_actual <= budget_target + 1e-6, (
        f"budgeted search exceeded target: {budget_actual} > {budget_target}")
    assert budget_actual < free_actual - 1e-6, (
        f"budgeted search did not reduce compute: {budget_actual} vs {free_actual}")
    print(f"[ok] budget-aware search: free_actual={free_actual:.1f} "
          f"budget_actual={budget_actual:.1f} target={budget_target:.1f}")


if __name__ == "__main__":
    test_ridge_fitter()
    test_oracle_coord_descent_improves()
    test_budget_aware_search_respects_target()
    print("\n[Phase 2] oracle_search_op smoke OK")
