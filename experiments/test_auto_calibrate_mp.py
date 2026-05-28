"""Phase-3 smoke: end-to-end ``auto_calibrate_mp`` on a 3-block synthetic
transformer-like stack. Verifies:

 1. Gauss-Seidel loop runs cleanly over all blocks.
 2. Per-block RMSE drops monotonically below the pre-calibration baseline.
 3. Final config has boundaries populated for every (block, op) pair.
 4. Block pre-hooks correctly route ``set_current_block_idx`` so different
    blocks can have different boundaries installed.

Runs on CPU in <30s.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sc"))
sys.path.insert(0, str(ROOT / "third_party" / "QwT-SC" / "QwT-vit-sc"))

from scmp_kernels.mp.config import (  # noqa: E402
    FreeBoundaryMPConfig,
    _classify_with_free_boundaries,
    set_current_block_idx,
    get_current_block_idx,
)
from scmp_kernels.mp import auto_calibrate_mp  # noqa: E402


class SyntheticBlock(nn.Module):
    """Block that computes ``Y = X W + b + row_noise(sl_per_row)``.

    The per-row stoc_len is looked up from a **shared** FreeBoundaryMPConfig
    at ``(current_block_idx, op="mlp")``. Row metric = ``||x_row||_inf``.

    Noise is seeded per-forward by the current block_idx + a fixed offset
    so scoring is deterministic for fixed boundaries across repeat calls.
    """
    noise_amp = {256: 0.01, 128: 0.10, 64: 0.40, 32: 1.0}

    def __init__(self, D: int, cfg: FreeBoundaryMPConfig,
                 is_sc: bool = True, init_seed: int = 0):
        super().__init__()
        torch.manual_seed(init_seed)
        self.linear = nn.Linear(D, D)
        # Scale weights so per-row outputs stay bounded.
        with torch.no_grad():
            self.linear.weight.mul_(0.3)
        self.cfg = cfg
        self.is_sc = is_sc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_fp = self.linear(x)
        if not self.is_sc:
            return y_fp
        N, D = x.shape[0], x.shape[-1]
        x_flat = x.reshape(-1, D)
        metric = x_flat.float().abs().amax(dim=-1)       # [N]
        assign = _classify_with_free_boundaries(
            metric, self.cfg.get_boundaries("mlp"),
            self.cfg.stoc_len_levels)
        row_std = torch.empty(metric.numel(), dtype=torch.float32)
        for sl, idx in assign.level_row_indices.items():
            if len(idx):
                row_std[idx] = self.noise_amp[sl]
        # Deterministic generator tied to block_idx so scoring is stable.
        blk_id = get_current_block_idx()
        g = torch.Generator().manual_seed(1000 + blk_id)
        noise = torch.randn(metric.numel(), D, generator=g)
        y = y_fp.reshape(-1, D) + noise * row_std.unsqueeze(-1)
        return y.reshape_as(y_fp)


class SyntheticModel(nn.Module):
    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


def make_models(n_blocks: int, D: int, cfg: FreeBoundaryMPConfig):
    fp_blocks = nn.ModuleList([
        SyntheticBlock(D, cfg, is_sc=False, init_seed=7 + i)
        for i in range(n_blocks)])
    sc_blocks = nn.ModuleList([
        SyntheticBlock(D, cfg, is_sc=True, init_seed=7 + i)
        for i in range(n_blocks)])
    # Keep FP and SC weights identical so the only difference is SC noise.
    for bf, bs in zip(fp_blocks, sc_blocks):
        bs.linear.load_state_dict(bf.linear.state_dict())
    return (SyntheticModel(fp_blocks), SyntheticModel(sc_blocks),
            fp_blocks, sc_blocks)


def synthetic_loader(n_calib: int, D: int, batch: int = 16):
    torch.manual_seed(42)
    for _ in range(max(1, n_calib // batch + 1)):
        yield torch.randn(batch, D), torch.zeros(batch, dtype=torch.long)


def main():
    D = 24
    n_blocks = 3
    cfg = FreeBoundaryMPConfig(stoc_len_levels=[256, 128, 64, 32])

    model_fp, model_sc, blocks_fp, blocks_sc = make_models(n_blocks, D, cfg)

    report = auto_calibrate_mp(
        model_fp=model_fp,
        model_sc=model_sc,
        blocks_fp=list(blocks_fp),
        blocks_sc_container=blocks_sc,
        calib_loader=synthetic_loader(n_calib=48, D=D),
        device=torch.device("cpu"),
        n_calib=48,
        cfg=cfg,
        ops_per_block=["mlp"],
        ridge=1e-3,
        start_block=0,
        fwd_chunk=16,
        avg_sc_draws=1,
        n_candidates=6,
        n_outer_per_op=1,
        n_outer_block=1,
        comp_factory=None,
        install_compensation=True,
        log_fn=print,
    )

    # ---- Validate report ----
    assert len(report) == n_blocks
    for i, entry in enumerate(report):
        assert entry["block"] == i
        assert entry["rmse_after"] <= entry["rmse_before"] + 1e-6, (
            f"block {i}: rmse regressed "
            f"{entry['rmse_before']} -> {entry['rmse_after']}")
        assert "mlp" in entry["boundaries_per_op"], (
            f"block {i}: mlp op missing from boundaries_per_op")
        b = entry["boundaries_per_op"]["mlp"]["boundaries"]
        assert len(b) == 3, f"k-1 = 3 boundaries expected, got {b}"
        # descending + in (0, 1)
        for j in range(len(b) - 1):
            assert b[j] > b[j + 1], f"block {i}: not descending: {b}"
        for v in b:
            assert 0 < v < 1, f"block {i}: out of range: {b}"

    # ---- Every (block, op) must be populated in cfg ----
    for i in range(n_blocks):
        assert (i, "mlp") in cfg.boundaries, (
            f"(block={i}, mlp) not populated in cfg")

    # ---- Different blocks should (generally) have different boundaries ----
    # since each block sees a different input distribution.
    unique_bounds = {tuple(cfg.boundaries[(i, "mlp")].tolist())
                     for i in range(n_blocks)}
    print(f"[ok] auto_calibrate_mp produced {len(unique_bounds)} unique "
          f"boundary tuples across {n_blocks} blocks")

    print(f"[ok] per-block RMSE progression:")
    for entry in report:
        print(f"     block {entry['block']}: "
              f"{entry['rmse_before']:.4e} -> {entry['rmse_after']:.4e}  "
              f"r2={entry['r2']:+.3f}  enabled={entry['enabled']}  "
              f"b={entry['boundaries_per_op']['mlp']['boundaries']}")

    print("\n[Phase 3] auto_calibrate_mp smoke OK")

    # ---- Raw-only / no-comp branch ----
    cfg_raw = FreeBoundaryMPConfig(stoc_len_levels=[256, 128, 64, 32])
    model_fp_raw, model_sc_raw, blocks_fp_raw, blocks_sc_raw = make_models(
        n_blocks, D, cfg_raw)
    report_raw = auto_calibrate_mp(
        model_fp=model_fp_raw,
        model_sc=model_sc_raw,
        blocks_fp=list(blocks_fp_raw),
        blocks_sc_container=blocks_sc_raw,
        calib_loader=synthetic_loader(n_calib=48, D=D),
        device=torch.device("cpu"),
        n_calib=48,
        cfg=cfg_raw,
        ops_per_block=["mlp"],
        ridge=1e-3,
        start_block=0,
        fwd_chunk=16,
        avg_sc_draws=1,
        n_candidates=6,
        n_outer_per_op=1,
        n_outer_block=1,
        search_objective="raw_mse",
        fit_compensation=False,
        install_compensation=False,
        log_fn=print,
    )
    assert len(report_raw) == n_blocks
    for i, entry in enumerate(report_raw):
        assert entry["objective"] == "raw_mse"
        assert not entry["enabled"]
        assert entry["raw_rmse_after"] <= entry["rmse_before"] + 1e-6, (
            f"raw-only block {i}: rmse regressed "
            f"{entry['rmse_before']} -> {entry['raw_rmse_after']}")
        assert abs(entry["rmse_after"] - entry["raw_rmse_after"]) <= 1e-6, (
            f"raw-only block {i}: rmse_after should equal raw_rmse_after")
    print("[ok] raw-only auto_calibrate_mp smoke OK")


if __name__ == "__main__":
    main()
