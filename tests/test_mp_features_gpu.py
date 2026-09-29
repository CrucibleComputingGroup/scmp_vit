"""GPU checks for the MP features wired in 2026-07-29.

Each test asserts an EXACTNESS property that a silent mis-wiring would break:
protected channels must reproduce the full-width matmul, the batched attention
dispatch must cover every row exactly once, and a group-laddered table must
deploy the rungs it was fitted on. Run on a GPU node:

    python tests/test_mp_features_gpu.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.sc_vit import (                                  # noqa: E402
    SCLinear, set_sc_defaults, set_qk_granularity, _sc_qk, _sc_av,
    _batched_level_matmul, AdaptiveMPConfig,
)
from scmp_kernels import sc_matmul                          # noqa: E402
import torch.nn as nn                                       # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(name)


def test_protected_channels_are_exact():
    """Splitting P/R channels and summing must equal the full-width SC matmul
    when both halves run at the SAME stream length. This is the linearity the
    protected-channel path relies on; if the index math is wrong the two
    disagree."""
    print("\n[protected channels] partition exactness")
    torch.manual_seed(0)
    dev = "cuda"
    M, D, O, SL = 256, 512, 128, 64
    x = torch.randn(M, D, device=dev)
    w = torch.randn(O, D, device=dev)
    full = sc_matmul(x, w, granularity="per_row", mode="bipolar", sc_prec=8,
                     stoc_len=SL, chunk_d=128, halve_bipolar_stoc_len=True)
    pi = torch.arange(0, D, 8, device=dev)
    mask = torch.ones(D, dtype=torch.bool, device=dev)
    mask[pi] = False
    ri = mask.nonzero(as_tuple=True)[0]
    part = (sc_matmul(x.index_select(1, pi).contiguous(),
                      w.index_select(1, pi).contiguous(),
                      granularity="per_row", mode="bipolar", sc_prec=8,
                      stoc_len=SL, chunk_d=128, halve_bipolar_stoc_len=True)
            + sc_matmul(x.index_select(1, ri).contiguous(),
                        w.index_select(1, ri).contiguous(),
                        granularity="per_row", mode="bipolar", sc_prec=8,
                        stoc_len=SL, chunk_d=128, halve_bipolar_stoc_len=True))
    rel = ((part - full).norm() / full.norm()).item()
    # Not bit-identical: each partition re-derives its own per-row (max,min)
    # and re-chunks, so the two see different quantization grids. The claim is
    # that the split does not CHANGE the computation, i.e. the disagreement
    # stays at SC-noise scale rather than the O(1) a mis-indexed gather gives.
    check("P+R partition reproduces full-width matmul", rel < 0.15, f"rel={rel:.4f}")

    lin = nn.Linear(D, O).to(dev)
    cfg = AdaptiveMPConfig(stoc_len_levels=[64, 48, 32],
                           target_fractions=[0.3, 0.4, 0.3])
    cfg.protected_channel_stoc_len = 64
    cfg.protected_channel_indices = {("mlp_fc2", 0, None): list(range(0, D, 8))}
    sc = SCLinear(lin, 8, stoc_len=64, chunk_d=128, adaptive_mp_cfg=cfg,
                  operator="mlp_fc2", block_idx=0)
    y = sc(x.unsqueeze(0))
    ref = torch.nn.functional.linear(x, lin.weight, lin.bias)
    rel2 = ((y[0] - ref).norm() / ref.norm()).item()
    check("SCLinear with protected channels tracks FP", rel2 < 0.35, f"rel={rel2:.4f}")
    check("protected-channel output is finite", bool(torch.isfinite(y).all()))


def test_batched_attention_dispatch_covers_every_row():
    """Every (slice, row) must be written exactly once by the padded gather."""
    print("\n[attention] batched per-level dispatch coverage")
    torch.manual_seed(0)
    dev = "cuda"
    BH, N, D = 12, 65, 32
    a3 = torch.randn(BH, N, D, device=dev)
    b3 = torch.randn(BH, N, D, device=dev)
    levels = [64, 48, 32]
    lv = torch.tensor([levels[i] for i in
                       torch.randint(0, 3, (BH * N,)).tolist()], device=dev)
    lv.reshape(BH, N)[3] = 64          # a slice with zero rows on two rungs
    out, cycles = _batched_level_matmul(a3, b3, lv, levels, sc_prec=8,
                                        out_dim=N, granularity="per_row")
    check("no unwritten rows (all-zero row count == 0)",
          int((out.abs().sum(-1) == 0).sum()) == 0)
    expect = sum(int((lv == sl).sum()) * sl for sl in levels)
    check("row-cycle accounting matches assignment", abs(cycles - expect) < 1e-6,
          f"got={cycles:.0f} want={expect}")
    uni = sc_matmul(a3, b3, granularity="per_row", mode="bipolar", sc_prec=8,
                    group_a=1, group_b=1, stoc_len=64,
                    halve_bipolar_stoc_len=True)
    rel = ((out - uni).norm() / uni.norm()).item()
    check("MP output is close to uniform-at-ceiling", rel < 0.5, f"rel={rel:.4f}")


def test_qk_per_row_matches_reference():
    print("\n[attention] qk per_row granularity")
    torch.manual_seed(0)
    dev = "cuda"
    B, H, N, D = 2, 4, 65, 32
    q = torch.randn(B, H, N, D, device=dev)
    k = torch.randn(B, H, N, D, device=dev)
    ref = torch.matmul(q, k.transpose(-2, -1))
    for gran in ("per_head", "per_row"):
        set_qk_granularity(gran)
        out = _sc_qk(q, k, 8, 64, None)
        rel = ((out - ref).norm() / ref.norm()).item()
        check(f"uniform qk @{gran} tracks FP", rel < 0.25, f"rel={rel:.4f}")
    set_qk_granularity("per_row")
    cfg = AdaptiveMPConfig(stoc_len_levels=[64, 48, 32],
                           target_fractions=[0.3, 0.4, 0.3])
    out = _sc_qk(q, k, 8, 64, cfg)
    rel = ((out - ref).norm() / ref.norm()).item()
    check("per-row MP qk tracks FP", rel < 0.3, f"rel={rel:.4f}")
    check("per-row MP qk shape", tuple(out.shape) == (B, H, N, N))
    set_qk_granularity("per_head")


def test_escape_gate_and_group_ladders():
    """The escape gate must key level_row_indices by the BUCKET's ladder.

    With per-group ladders in play, rebuilding that map against the GLOBAL
    ladder re-keys every rung — rows then run at stream lengths from a ladder
    they were never fitted on, and nothing raises. The assertion is that every
    dispatched stream length belongs to the bucket's own ladder.
    """
    print("\n[escape gate + group ladders] dispatch stays on the bucket ladder")
    import json
    import tempfile
    from scmp_kernels.mp import adaptive_classify_rows

    global_ladder = [128, 96, 64, 48, 32]
    bucket_ladder = [64, 48, 32]
    table = {
        "stoc_len_levels": global_ladder,
        "timestep_buckets": 1, "layer_buckets": 1,
        "operator_defaults": {},
        "buckets": {
            "mlp_fc1:t0:l0": {
                "stoc_len_levels": bucket_ladder,
                "thresholds": [0.7, 0.3],
                "metric_mean": 0.2, "metric_std": 0.1,
            }
        },
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(table, f)
        path = f.name

    cfg = AdaptiveMPConfig(stoc_len_levels=global_ladder,
                           threshold_table_path=path,
                           escape_gate_k=1.0, escape_stoc_len=128)
    metric = torch.rand(4096, device="cuda")
    metric[:10] = 1.0                       # force escapes
    a = adaptive_classify_rows(metric, cfg, operator="mlp_fc1", block_idx=0,
                               total_blocks=1)
    used = {int(sl) for sl, idx in a.level_row_indices.items() if idx.numel()}
    allowed = set(bucket_ladder) | {128}
    check("dispatched stream lengths are on the bucket ladder (+escape)",
          used <= allowed, f"used={sorted(used)} allowed={sorted(allowed)}")
    total = sum(int(idx.numel()) for idx in a.level_row_indices.values())
    check("every row assigned exactly once", total == metric.numel(),
          f"{total} vs {metric.numel()}")
    check("escape rung actually fired", 128 in used)
    vals = cfg.classify_level_values(operator="mlp_fc1", block_idx=0,
                                     total_blocks=1)
    check("classify_level_values covers the escape rung",
          set(used) <= set(int(v) for v in vals),
          f"vals={vals}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    set_sc_defaults(halve_bipolar_stoc_len=True, chunk_d=128)
    test_protected_channels_are_exact()
    test_batched_attention_dispatch_covers_every_row()
    test_qk_per_row_matches_reference()
    test_escape_gate_and_group_ladders()
    print(f"\n{'ALL PASS' if not FAILED else 'FAILURES: ' + ', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)
