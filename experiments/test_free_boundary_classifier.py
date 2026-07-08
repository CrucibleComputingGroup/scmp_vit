"""Phase-1 smoke: FreeBoundaryMPConfig dispatch + classifier correctness.

Runs on CPU in <1s. Asserts the free-boundary classifier splits rows into
the expected level sizes and respects the block-idx global.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sc"))

from scmp_kernels.mp import (  # noqa: E402
    FreeBoundaryMPConfig,
    adaptive_classify_rows,
    set_current_block_idx,
)


def test_default_boundaries_equal_spacing():
    cfg = FreeBoundaryMPConfig(stoc_len_levels=[256, 128, 64, 32])
    # Uniform metric in [0, 1] — after normalization stays uniform.
    metric = torch.linspace(0.0, 1.0, 100)
    set_current_block_idx(0)
    # Default boundaries are equal spacing: 0.75, 0.5, 0.25
    # Expected split: 25 rows → sl=256 (above 0.75), 25 → 128, 25 → 64,
    # 25 → 32.
    assign = adaptive_classify_rows(metric, 0, 1, cfg, operator="qk")
    counts = {sl: len(idx) for sl, idx in assign.level_row_indices.items()}
    for sl in [256, 128, 64, 32]:
        assert counts[sl] == 25, f"default spacing wrong for sl={sl}: {counts}"
    print(f"[ok] default equal spacing → {counts}")


def test_custom_boundaries_heavy_top():
    cfg = FreeBoundaryMPConfig(stoc_len_levels=[256, 128, 64, 32])
    # Heavy-top boundaries: only top 10% → sl=256, next 10% → 128,
    # next 30% → 64, rest 50% → 32.
    #   above 0.90 → level 0 (10 rows)
    #   [0.80, 0.90) → level 1 (10)
    #   [0.50, 0.80) → level 2 (30)
    #   <0.50 → level 3 (50)
    cfg.set_boundaries("av", block_idx=3,
                       boundaries=torch.tensor([0.90, 0.80, 0.50]))
    metric = torch.linspace(0.0, 1.0, 100)

    set_current_block_idx(3)
    assign = adaptive_classify_rows(metric, 0, 1, cfg, operator="av")
    counts = {sl: len(idx) for sl, idx in assign.level_row_indices.items()}
    # Boundaries are on metric_norm ∈ [0, 1]; linspace includes 0 and 1.
    # The strict < comparison gives 10/10/30/50 with off-by-one tolerance.
    assert abs(counts[256] - 10) <= 1, f"sl=256 count off: {counts}"
    assert abs(counts[128] - 10) <= 1, f"sl=128 count off: {counts}"
    assert abs(counts[64] - 30) <= 1, f"sl=64 count off: {counts}"
    assert abs(counts[32] - 50) <= 1, f"sl=32 count off: {counts}"
    print(f"[ok] heavy-top boundaries (block=3, av) → {counts}")

    # Different block index must fall back to default (equal) spacing.
    set_current_block_idx(0)
    assign2 = adaptive_classify_rows(metric, 0, 1, cfg, operator="av")
    counts2 = {sl: len(idx) for sl, idx in assign2.level_row_indices.items()}
    for sl in [256, 128, 64, 32]:
        assert counts2[sl] == 25, (
            f"block=0 should use default, got {counts2}")
    print(f"[ok] unset (block=0, av) falls back to default → {counts2}")


def test_flat_metric_degenerate():
    cfg = FreeBoundaryMPConfig(stoc_len_levels=[256, 64, 16])
    metric = torch.full((50,), 0.5)  # all equal
    set_current_block_idx(0)
    assign = adaptive_classify_rows(metric, 0, 1, cfg, operator="qk")
    counts = {sl: len(idx) for sl, idx in assign.level_row_indices.items()}
    assert counts[256] == 50 and counts[64] == 0 and counts[16] == 0, (
        f"flat metric should go fully to sl=256, got {counts}")
    print(f"[ok] flat metric → all sl=256 (counts={counts})")


if __name__ == "__main__":
    test_default_boundaries_equal_spacing()
    test_custom_boundaries_heavy_top()
    test_flat_metric_degenerate()
    print("\n[Phase 1] free-boundary classifier smoke OK")
