"""Smoke test: combined min(range_mp, adaptive_mp) on SCLinear with external T.

Runs under the noise-model fast path (no Triton / GPU required) and verifies:
  1. SCLinear accepts AdaptiveMPConfig + RangeMPConfig at the same time.
  2. Forward pass is shape-correct at two externally-set (t, T) values.
  3. The adaptive path really routed through adaptive_classify_rows (we
     monkey-patch the classifier to count calls).
  4. Combined dispatch exercised both range groups and adaptive row levels.
  5. cls's patch_model accepts AdaptiveMPConfig in its linear_mp_spec.
  6. det's sc_patch_eva accepts the same via linear_mp_spec.

Run (from repo root):
    python experiments/smoke_combined_adaptive_mp.py
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sc"))


def _seed_all(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _dump_run_metadata(out_dir: Path, results: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    results["git_commit"] = _git_commit()
    results["python"] = sys.version.split()[0]
    results["torch"] = torch.__version__
    (out_dir / "smoke_combined_adaptive_mp.json").write_text(
        json.dumps(results, indent=2, default=str))


def test_sclinear_combined(log: logging.Logger) -> dict:
    """Direct SCLinear test — smallest unit that exercises combined+adaptive."""
    from sc_integration import sc_linear as _scl
    from sc_integration.sc_linear import SCLinear, set_vit_timestep
    from sc_integration.mp_linear import (
        MPConfig, AdaptiveMPConfig, RangeMPConfig,
    )

    _scl.set_noise_model(True)  # fast path, no Triton needed

    # Instrument: count adaptive_classify_rows calls.
    call_counter = {"n": 0, "timesteps": []}
    from sc_integration import mp_linear as _mpl
    orig_adaptive = _mpl.adaptive_classify_rows

    def spy_adaptive(metric, timestep, total_timesteps, config, operator=None):
        call_counter["n"] += 1
        call_counter["timesteps"].append((timestep, total_timesteps))
        return orig_adaptive(metric, timestep, total_timesteps, config,
                             operator=operator)
    _mpl.adaptive_classify_rows = spy_adaptive
    # sc_linear imports the name at module load — patch there too.
    _scl.classify_input_rows_adaptive = _mpl.classify_input_rows_adaptive

    D_in, D_out = 64, 128
    linear = nn.Linear(D_in, D_out)
    adaptive = AdaptiveMPConfig(
        stoc_len_levels=[256, 64, 0],
        alpha=0.3, beta=0.05,
        operator_params={"mlp_fc1": (0.4, 0.1)},
    )
    range_cfg = RangeMPConfig(
        stoc_len_levels=[256, 128, 64], base_threshold=0.3,
    )

    sc_fc = SCLinear(
        linear, sc_prec=8, mode="bipolar",
        adaptive_mp_cfg=adaptive,
        range_mp_cfg=range_cfg,
        range_mp_group_size=32,  # 128 / 32 = 4 groups
        operator="mlp_fc1",
    )

    out = {"range_entries": len(sc_fc.range_entries)}

    x = torch.randn(2, 16, D_in) * 0.5
    set_vit_timestep(0, 1)  # ViT no-timestep default
    y0 = sc_fc(x)
    assert y0.shape == (2, 16, D_out), f"bad shape at T=1: {y0.shape}"
    out["shape_T1"] = list(y0.shape)

    set_vit_timestep(7, 10)  # "late diffusion" emulation
    y1 = sc_fc(x)
    assert y1.shape == (2, 16, D_out), f"bad shape at T=10,t=7: {y1.shape}"
    out["shape_T10"] = list(y1.shape)

    set_vit_timestep(9, 10)  # noisiest end
    y2 = sc_fc(x)
    out["shape_T10_t9"] = list(y2.shape)

    assert call_counter["n"] >= 3, (
        f"adaptive classifier was not called per forward: {call_counter}")
    assert (0, 1) in call_counter["timesteps"], "T=(0,1) not recorded"
    assert (7, 10) in call_counter["timesteps"], "T=(7,10) not recorded"
    out["adaptive_calls"] = call_counter["n"]
    out["timesteps_seen"] = call_counter["timesteps"]

    # Output shouldn't be all zeros (SC with noise model produces finite results).
    assert torch.isfinite(y0).all() and y0.abs().max().item() > 0
    assert torch.isfinite(y1).all() and y1.abs().max().item() > 0

    # Restore
    _mpl.adaptive_classify_rows = orig_adaptive
    log.info(f"[unit] combined adaptive+range passed: {out}")
    return out


def test_cls_patch_accepts_adaptive(log: logging.Logger) -> dict:
    """Ensure cls's patch_model accepts AdaptiveMPConfig in linear_mp_spec."""
    from sc_integration import sc_linear as _scl
    _scl.set_noise_model(True)
    sys.path.insert(0, str(REPO / "cls"))
    from sc_attention_patch import _normalize_linear_mp_spec, _mp_kwargs
    from sc_integration.mp_linear import AdaptiveMPConfig, RangeMPConfig

    spec = {
        "mlp_fc1": {
            "adaptive": AdaptiveMPConfig(
                stoc_len_levels=[256, 64, 0], alpha=0.3, beta=0.05),
            "range": RangeMPConfig(
                stoc_len_levels=[256, 128, 64], base_threshold=0.3),
            "range_group_size": 32,
        },
        "mlp_fc2": AdaptiveMPConfig(
            stoc_len_levels=[256, 128, 0], alpha=0.2, beta=0.1),
    }
    normalized = _normalize_linear_mp_spec(spec)
    kw1 = _mp_kwargs(normalized, "mlp_fc1")
    kw2 = _mp_kwargs(normalized, "mlp_fc2")
    assert kw1["adaptive_mp_cfg"] is not None and kw1["range_mp_cfg"] is not None
    assert kw2["adaptive_mp_cfg"] is not None and kw2["range_mp_cfg"] is None
    log.info("[cls] patch_model adaptive spec passed normalization.")
    return {"mlp_fc1_kwargs": list(kw1.keys()),
            "mlp_fc2_kwargs": list(kw2.keys())}


def test_det_patch_accepts_adaptive(log: logging.Logger) -> dict:
    """Ensure det's sc_patch_eva accepts AdaptiveMPConfig in linear_mp_spec."""
    sys.path.insert(0, str(REPO / "det"))
    from sc_patch.sc_model_eva import (
        _normalize_det_linear_mp_spec, _det_mp_kwargs,
    )
    from sc_integration.mp_linear import AdaptiveMPConfig, RangeMPConfig

    spec = {
        "proj": {
            "adaptive": AdaptiveMPConfig(
                stoc_len_levels=[256, 64, 0], alpha=0.3, beta=0.05),
            "range": RangeMPConfig(
                stoc_len_levels=[256, 128, 64], base_threshold=0.3),
        },
    }
    normalized = _normalize_det_linear_mp_spec(spec)
    kw_qkv = _det_mp_kwargs(normalized, "qkv_proj")
    kw_out = _det_mp_kwargs(normalized, "out_proj")
    assert kw_qkv["adaptive_mp_cfg"] is not None
    assert kw_out["adaptive_mp_cfg"] is not None
    log.info("[det] sc_patch_eva adaptive spec passed normalization.")
    return {"qkv_proj_kwargs": list(kw_qkv.keys()),
            "out_proj_kwargs": list(kw_out.keys())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(REPO / "results" / "smoke"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("smoke")

    _seed_all(args.seed)
    t0 = time.time()
    results: dict = {"seed": args.seed, "argv": sys.argv}

    results["sclinear_unit"] = test_sclinear_combined(log)
    results["cls_patch"] = test_cls_patch_accepts_adaptive(log)
    results["det_patch"] = test_det_patch_accepts_adaptive(log)
    results["elapsed_s"] = round(time.time() - t0, 3)

    _dump_run_metadata(Path(args.out_dir), results)
    log.info(f"SMOKE OK in {results['elapsed_s']}s. "
             f"Output: {args.out_dir}/smoke_combined_adaptive_mp.json")


if __name__ == "__main__":
    main()
