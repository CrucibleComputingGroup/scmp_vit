"""Extended QwT-SC compensator with K-seed bank, polarity flip, and W-scale
search. Also records per-block correlation diagnostics so we can tell *why*
a variant was selected.

The hardware framing is unchanged from the parent driver: each chosen
non-default option is at most O(log K) config bits per block (a tiny LUT in
the SC accelerator's per-block scheduler).

Usage::

    python experiments/qwt_sc_overnight.py --sc_config qk_only \\
        --n_calib 256 --n_eval 500 --batch_size 8 \\
        --comp_mode sc --n_variants 8 --polarity_flip \\
        --w_scales 0.5,0.75,1.0 \\
        --out_json results/qwt_sc_qk_only_overnight.json
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]   # vit_sc/
CLS = HERE / "cls"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CLS))
QWT_SC_LIB = HERE / "third_party" / "QwT-SC" / "QwT-vit-sc"
if not (QWT_SC_LIB / "qwt_sc").exists():
    raise RuntimeError(f"qwt_sc submodule missing at {QWT_SC_LIB}")
sys.path.insert(0, str(QWT_SC_LIB))
sys.path.insert(0, str(HERE / "sc"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from eval import (build_transform, load_model, seed_all,
                  _build_attn_mp_spec, _build_linear_mp_spec)
from imagenet_parquet import ImageNetParquetVal
from sc_attention_patch import patch_model, set_noise_model, SC_OP_NAMES, SCLinear

from qwt_sc import calibrate_qwt
from config_helpers import (
    make_sobol_simple_config,
    make_sobol_antithetic_config,
    make_sobol_altseed_config,
)
from sc_integration.head_aligned_comp import HeadAlignedSCLinear


SENSITIVITY_JSON = HERE / "results" / "sensitivity_all_ops.json"

# Per-op MACs per block (DINOv2 ViT-L/14: D=1024, 16 heads, head_dim=64, seq=257)
_SEQ = 257
_OP_MACS = {
    "qk":      16 * _SEQ * 64 * _SEQ,
    "av":      16 * _SEQ * _SEQ * 64,
    "proj":    _SEQ * 1024 * 3072 + _SEQ * 1024 * 1024,  # qkv + out
    "mlp_fc1": _SEQ * 1024 * 4096,
    "mlp_fc2": _SEQ * 4096 * 1024,
}
_COMP_MACS = _SEQ * 1024 * 1024


def _avg_sl(levels_str, fracs_str, default):
    if not levels_str:
        return default
    levels = [int(x) for x in levels_str.split(",")]
    if fracs_str:
        fracs = [float(x) for x in fracs_str.split(",")]
    else:
        fracs = [1.0 / len(levels)] * len(levels)
    return sum(l * f for l, f in zip(levels, fracs))


def compute_stoc_len_stats(args, fine_schedule, preset, n_blocks=24):
    base_sl = 2 ** args.sc_prec
    mp_ops = {x.strip() for x in args.mp_ops.split(",") if x.strip()}
    sl_linear = _avg_sl(args.mp_levels, args.mp_fractions, base_sl)
    op_sl = {
        "qk":      _avg_sl(args.qk_mp_levels, args.qk_mp_fractions, base_sl),
        "av":      _avg_sl(args.av_mp_levels, args.av_mp_fractions, base_sl),
        "proj":    sl_linear if mp_ops & {"qkv_proj", "out_proj", "proj"} else base_sl,
        "mlp_fc1": sl_linear if "mlp_fc1" in mp_ops else base_sl,
        "mlp_fc2": sl_linear if "mlp_fc2" in mp_ops else base_sl,
    }
    if fine_schedule is not None:
        active = {op: sum(fine_schedule.get(op, [0] * n_blocks)) for op in _OP_MACS}
    elif preset is not None:
        active = {
            "qk": n_blocks if preset.get("sc_qk") else 0,
            "av": n_blocks if preset.get("sc_av") else 0,
            "proj": n_blocks if (preset.get("sc_qkv_proj") or preset.get("sc_out_proj")) else 0,
            "mlp_fc1": n_blocks if preset.get("sc_mlp") else 0,
            "mlp_fc2": n_blocks if preset.get("sc_mlp") else 0,
        }
    else:
        active = {op: 0 for op in _OP_MACS}
    main_macs = sum(_OP_MACS[op] * active[op] for op in _OP_MACS)
    main_bitops = sum(_OP_MACS[op] * active[op] * op_sl[op] for op in _OP_MACS)
    main_sl = main_bitops / main_macs if main_macs > 0 else 0
    comp_sl = 2 ** args.comp_sc_prec if args.comp_mode == "sc" else 0
    comp_macs = n_blocks * _COMP_MACS if args.comp_mode == "sc" else 0
    comp_bitops = comp_macs * comp_sl
    total_macs = main_macs + comp_macs
    eff_sl = (main_bitops + comp_bitops) / total_macs if total_macs > 0 else 0
    return {
        "main_sl": round(main_sl, 1),
        "eff_sl": round(eff_sl, 1),
        "eff_reduction_pct": round((1 - eff_sl / base_sl) * 100, 1),
        "main_macs_B": round(main_macs / 1e9, 2),
        "comp_macs_B": round(comp_macs / 1e9, 2),
    }


def build_skip_worst_k_schedule(k, n_blocks=24):
    with open(SENSITIVITY_JSON) as f:
        data = json.load(f)
    rows = sorted(data["grid"], key=lambda r: r["l2"], reverse=True)
    drop = {(r["op"], int(r["block"])) for r in rows[:k]}
    spec = {name: [1] * n_blocks for name in SC_OP_NAMES}
    for (op, bi) in drop:
        spec[op][bi] = 0
    return spec


SC_PRESETS = {
    "full_attn": dict(sc_qk=True, sc_av=True, sc_qkv_proj=True, sc_out_proj=True, sc_mlp=False),
    "qk_only":   dict(sc_qk=True, sc_av=False, sc_qkv_proj=False, sc_out_proj=False, sc_mlp=False),
    "qk_av":     dict(sc_qk=True, sc_av=True, sc_qkv_proj=False, sc_out_proj=False, sc_mlp=False),
}
FINE_GRAINED_KS = {
    "skip_worst50": 50, "skip_worst40": 40, "skip_worst30": 30,
    "skip_worst20": 20, "all_ops": 0,
}


def build_dataset(data_root, seed, n_images):
    ds = ImageNetParquetVal(data_root, transform=build_transform(224))
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if n_images and n_images < len(ds):
        ds = Subset(ds, idx[:n_images])
    return ds


def get_blocks(model):
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        return model.backbone.blocks
    return model.blocks


@torch.no_grad()
def evaluate(model, loader, device, log_every=20):
    model.eval()
    n = top1 = top5 = 0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        _, p5 = logits.topk(5, 1)
        correct = p5.eq(y.unsqueeze(1))
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(1).sum().item()
        n += y.numel()
        if (i + 1) % log_every == 0:
            dt = time.time() - t0
            print(f"  [{i+1}] top1={top1/n:.4f} top5={top5/n:.4f} "
                  f"({n} imgs, {dt:.1f}s, {n/dt:.1f} img/s)", flush=True)
    dt = time.time() - t0
    return {"n": n, "top1": top1 / n, "top5": top5 / n,
            "elapsed_s": dt, "img_per_s": n / dt if dt > 0 else 0.0}


def build_seed_bank(D, sc_prec, n_variants, rng_seed=42):
    """Build n_variants Sobol configs at dim D. Always includes default + antithetic;
    pads with random K-side seeds."""
    rnd = random.Random(rng_seed)
    configs = [
        ("default",    make_sobol_simple_config(D, D, sc_prec)),
        ("antithetic", make_sobol_antithetic_config(D, D, sc_prec)),
    ]
    n_extra = max(0, n_variants - len(configs))
    for i in range(n_extra):
        seed = [1]
        for j in range(1, sc_prec):
            mx = 1 << (j + 1)
            v = rnd.randint(0, mx // 2 - 1) * 2 + 1
            seed.append(v)
        cfg = make_sobol_altseed_config(D, D, sc_prec, k_seed=seed)
        configs.append((f"alt{i}_{','.join(str(s) for s in seed[:4])}..", cfg))
    return configs[:n_variants]


def make_sc_factory(sc_prec, mode, cfg_override=None, scale=1.0, polarity=1):
    """Factory closing over (cfg, scale, polarity). polarity ∈ {+1, -1} flips W sign."""
    def factory(W, b):
        D_in, D_out = W.shape
        layer = nn.Linear(D_in, D_out, bias=True)
        with torch.no_grad():
            layer.weight.copy_((W.t().contiguous()) * (scale * polarity))
            layer.bias.copy_(b * polarity if polarity == -1 else b)
        return SCLinear(layer, sc_prec=sc_prec, mode=mode, cfg_override=cfg_override)
    return factory


def make_fp_factory(scale=1.0, polarity=1):
    """Factory returning an nn.Linear (FP comp) for --comp_mode fp."""
    def factory(W, b):
        D_in, D_out = W.shape
        layer = nn.Linear(D_in, D_out, bias=True)
        with torch.no_grad():
            layer.weight.copy_((W.t().contiguous()) * (scale * polarity))
            layer.bias.copy_(b * polarity if polarity == -1 else b)
        return layer
    return factory


def make_head_aligned_factory(sc_prec, mode, n_heads=16, cfg_override=None,
                              scale=1.0, polarity=1):
    """Factory: HeadAlignedSCLinear with per-head SC matmul reusing
    _CFG_CACHE[(D_in/n_heads, sc_prec)] — same pool as block per-head QK."""
    def factory(W, b):
        D_in, D_out = W.shape
        layer = nn.Linear(D_in, D_out, bias=True)
        with torch.no_grad():
            layer.weight.copy_((W.t().contiguous()) * (scale * polarity))
            layer.bias.copy_(b * polarity if polarity == -1 else b)
        return HeadAlignedSCLinear(layer, sc_prec=sc_prec, mode=mode,
                                    n_heads=n_heads, cfg_override=cfg_override)
    return factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data")
    ap.add_argument("--sc_prec", type=int, default=8)
    ap.add_argument("--sc_config",
                    choices=list(SC_PRESETS) + list(FINE_GRAINED_KS),
                    default="qk_only")
    ap.add_argument("--sc_ops_per_block_json", default="",
                    help="Path to a pre-built 5-op sc_ops_per_block JSON "
                         "(e.g. from build_skip_worst20_json.py). When set, "
                         "overrides --sc_config's fine-grained lookup.")
    ap.add_argument("--n_calib", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--calib_seed", type=int, default=1)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--r2_threshold", type=float, default=0.0,
                    help="Skip QwT compensation in a block if its fit r2 is "
                         "<= this threshold. Default 0.0 = original behavior.")
    ap.add_argument("--max_block_for_qwt", type=int, default=10**9,
                    help="Hard cap: never compensate blocks with index >= this. "
                         "Protects against catastrophic late-block noise injection "
                         "when a borderline r2 fit is applied to a high-norm output "
                         "(e.g. pre-norm ViT block 23 at r2~0.5).")
    ap.add_argument("--last_block_r2_threshold", type=float, default=-1.0,
                    help="Stricter r2 gate for the LAST block only (negative = "
                         "use --r2_threshold). Block 23's residual-stream norm is "
                         "~20x mid-block, so a borderline r2 fit there can inject "
                         "catastrophic noise into the pre-head embedding. Tighten "
                         "to 0.7 or 0.9 to let only well-fit comps apply.")
    ap.add_argument("--lookahead_veto", action="store_true",
                    help="Per-block 1-step binary lookahead. For each block that "
                         "passes the r2 gate, propagate both (apply comp) and "
                         "(skip comp) through block_fp[i+1] and veto apply when "
                         "skip is closer to the FP reference. Principled "
                         "replacement for max_block_for_qwt — no magic numbers.")
    ap.add_argument("--rmse_margin", type=float, default=0.0,
                    help="Require raw_rmse/after_rmse > this margin as an extra "
                         "gate on top of --r2_threshold. 0 = disabled. 1.5 asks "
                         "for at least 1.5x RMSE improvement from the comp; "
                         "protects against knife-edge r² cases where the LS fit "
                         "barely explains variance.")
    ap.add_argument("--start_block", type=int, default=0)
    ap.add_argument("--avg_sc_draws", type=int, default=1)
    ap.add_argument("--fwd_chunk", type=int, default=32)
    ap.add_argument("--skip_baseline", action="store_true")
    ap.add_argument("--comp_mode", choices=["fp", "sc"], default="sc")
    ap.add_argument("--comp_sc_prec", type=int, default=8)
    ap.add_argument("--comp_sc_mode", choices=["bipolar", "unipolar"], default="bipolar")
    ap.add_argument("--n_variants", type=int, default=1,
                    help="Number of Sobol seed variants in the K-bank "
                         "(includes default + antithetic). 1 disables search.")
    ap.add_argument("--polarity_flip", action="store_true",
                    help="Also try polarity = -1 for each seed variant. "
                         "Doubles variant count. 1 extra config bit per block.")
    ap.add_argument("--w_scales", type=str, default="1.0",
                    help="Comma-separated W magnitudes to try, e.g. '0.5,0.75,1.0'. "
                         "Per-block selection. ⌈log_2 n_scales⌉ extra config bits.")
    ap.add_argument("--head_aligned", action="store_true",
                    help="Add head-aligned SC comp variants (per-head D=64 SC "
                         "matmul reusing _CFG_CACHE[(64,8)] = block QK pool).")
    ap.add_argument("--head_aligned_only", action="store_true",
                    help="Drop the regular SC variants; only use head-aligned "
                         "variants for the SC factories. Implies --head_aligned.")
    ap.add_argument("--n_heads", type=int, default=16,
                    help="Number of heads for head-aligned comp (must divide D=1024).")
    ap.add_argument("--out_json", default="results/qwt_sc_overnight.json")
    # ---- Mixed precision (same CLI as eval.py; piped into patch_model) ----
    ap.add_argument("--mp_levels", default="")
    ap.add_argument("--mp_fractions", default="")
    ap.add_argument("--mp_ops", default="")
    ap.add_argument("--range_mp", type=int, default=0)
    ap.add_argument("--range_mp_levels", default="256,128")
    ap.add_argument("--range_mp_threshold", type=float, default=0.3)
    ap.add_argument("--range_mp_ops", default="")
    ap.add_argument("--range_mp_group_size", type=int, default=0)
    ap.add_argument("--qk_mp_levels", default="")
    ap.add_argument("--qk_mp_fractions", default="")
    ap.add_argument("--av_mp_levels", default="")
    ap.add_argument("--av_mp_fractions", default="")
    # adaptive MP (shared with eval.py; unused here, but _build_*_mp_spec
    # reads these fields unconditionally via args.* attribute access)
    ap.add_argument("--adaptive_mp", type=int, default=0)
    ap.add_argument("--adaptive_mp_levels", default="")
    ap.add_argument("--adaptive_mp_alpha", type=float, default=0.3)
    ap.add_argument("--adaptive_mp_beta", type=float, default=0.05)
    ap.add_argument("--adaptive_mp_enable_pruning", type=int, default=1)
    ap.add_argument("--adaptive_mp_ops", default="")
    args = ap.parse_args()

    os.environ.setdefault("XFORMERS_DISABLED", "1")
    seed_all(args.calib_seed)
    device = torch.device("cuda")
    set_noise_model(False)

    print(f"[env] torch {torch.__version__} cuda {torch.version.cuda} "
          f"dev={torch.cuda.get_device_name(0)}", flush=True)

    fine_schedule = None
    preset = None
    if args.sc_ops_per_block_json:
        with open(args.sc_ops_per_block_json) as f:
            fine_schedule = json.load(f)
        total_on = sum(sum(v) for v in fine_schedule.values())
        cfg_desc = (f"external schedule from {args.sc_ops_per_block_json} "
                    f"({total_on}/120)")
    elif args.sc_config in SC_PRESETS:
        preset = SC_PRESETS[args.sc_config]
        cfg_desc = f"preset {preset}"
    else:
        k = FINE_GRAINED_KS[args.sc_config]
        fine_schedule = build_skip_worst_k_schedule(k=k)
        total_on = sum(sum(v) for v in fine_schedule.values())
        cfg_desc = f"fine_grained skip_worst_K={k} ({total_on}/120)"
    print(f"[cfg] sc_config={args.sc_config} -> {cfg_desc}  sc_prec={args.sc_prec}",
          flush=True)

    print("[data] building loaders", flush=True)
    calib_ds = build_dataset(args.data_root, args.calib_seed, args.n_calib)
    calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    eval_ds = build_dataset(args.data_root, args.eval_seed, args.n_eval)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    print("[model] loading FP + SC copies", flush=True)
    model_fp = load_model(device).eval()
    model_sc = load_model(device).eval()

    linear_mp_spec = _build_linear_mp_spec(args) or None
    attn_mp_spec = _build_attn_mp_spec(args) or None
    if linear_mp_spec:
        print(f"[mp] linear_mp_spec ops={sorted(linear_mp_spec)}", flush=True)
    if attn_mp_spec:
        print(f"[mp] attn_mp_spec ops={sorted(attn_mp_spec)}", flush=True)
    if preset is not None:
        stats = patch_model(model_sc, sc_prec=args.sc_prec,
                            linear_mp_spec=linear_mp_spec,
                            attn_mp_spec=attn_mp_spec, **preset)
    else:
        stats = patch_model(model_sc, sc_prec=args.sc_prec,
                            sc_ops_per_block=fine_schedule,
                            linear_mp_spec=linear_mp_spec,
                            attn_mp_spec=attn_mp_spec)
    print(f"[sc] patched={stats}", flush=True)

    sl_stats = compute_stoc_len_stats(args, fine_schedule, preset)
    print(f"[sl] main_sl={sl_stats['main_sl']}  eff_sl={sl_stats['eff_sl']}  "
          f"reduction={sl_stats['eff_reduction_pct']}%  "
          f"(main={sl_stats['main_macs_B']}B + comp={sl_stats['comp_macs_B']}B MACs)",
          flush=True)

    res_sc_raw = None
    if not args.skip_baseline:
        seed_all(args.eval_seed)
        res_sc_raw = evaluate(model_sc, eval_loader, device)
        print("[result:sc_raw]", res_sc_raw, flush=True)

    # Build the comp variants: cartesian of {seed configs} x {polarities} x {scales}.
    D_resid = 1024  # ViT-L residual dim
    seed_bank = build_seed_bank(D_resid, args.comp_sc_prec, args.n_variants)
    polarities = [1, -1] if args.polarity_flip else [1]
    scales = [float(s) for s in args.w_scales.split(",")]

    variants = []
    if args.comp_mode == "sc":
        if not args.head_aligned_only:
            for sname, cfg in seed_bank:
                for pol in polarities:
                    for sc in scales:
                        name = f"sc/{sname}/p{'+' if pol > 0 else '-'}/s{sc:.2f}"
                        variants.append((name, make_sc_factory(
                            args.comp_sc_prec, args.comp_sc_mode,
                            cfg_override=cfg, scale=sc, polarity=pol)))
        if args.head_aligned or args.head_aligned_only:
            # Head-aligned variants: each chunk uses its size's _CFG_CACHE
            # entry by default (cfg_override=None). Don't search seed bank
            # here — the point is to reuse block per-head SNG.
            for pol in polarities:
                for sc in scales:
                    name = f"head_aligned/p{'+' if pol > 0 else '-'}/s{sc:.2f}/h{args.n_heads}"
                    variants.append((name, make_head_aligned_factory(
                        args.comp_sc_prec, args.comp_sc_mode,
                        n_heads=args.n_heads, cfg_override=None,
                        scale=sc, polarity=pol)))
    else:  # comp_mode == "fp"
        for pol in polarities:
            for sc in scales:
                name = f"fp/p{'+' if pol > 0 else '-'}/s{sc:.2f}"
                variants.append((name, make_fp_factory(scale=sc, polarity=pol)))

    print(f"[comp] variants={len(variants)}  "
          f"(seeds={len(seed_bank)} pols={len(polarities)} "
          f"scales={len(scales)})",
          flush=True)
    for name, _ in variants[:8]:
        print(f"        - {name}", flush=True)
    if len(variants) > 8:
        print(f"        ... ({len(variants)-8} more)", flush=True)

    # Collapse to single-variant if K=1 with no polarity/scale/fp_sc — use comp_factory
    # path so calibrate_qwt skips its variant loop.
    comp_factory = None
    comp_factory_variants = None
    if len(variants) == 1:
        comp_factory = variants[0][1]
        print(f"[comp] single-variant fast path: {variants[0][0]}", flush=True)
    else:
        comp_factory_variants = variants

    print("[calib] starting QwT calibration", flush=True)
    t_cal = time.time()
    seed_all(args.calib_seed)
    blocks_fp = list(get_blocks(model_fp))
    blocks_sc_container = get_blocks(model_sc)
    calib_report = calibrate_qwt(
        model_fp=model_fp, model_sc=model_sc,
        blocks_fp=blocks_fp, blocks_sc_container=blocks_sc_container,
        calib_loader=calib_loader, device=device,
        n_calib=args.n_calib, ridge=args.ridge,
        start_block=args.start_block, fwd_chunk=args.fwd_chunk,
        avg_sc_draws=args.avg_sc_draws,
        comp_factory=comp_factory,
        comp_factory_variants=comp_factory_variants,
        comp_refit_iters=0,
        r2_threshold=args.r2_threshold,
        max_block_for_qwt=args.max_block_for_qwt,
        last_block_r2_threshold=(args.last_block_r2_threshold
                                 if args.last_block_r2_threshold >= 0.0
                                 else None),
        rmse_margin=args.rmse_margin,
        lookahead_veto=args.lookahead_veto,
    )
    calib_s = time.time() - t_cal
    print(f"[calib] done in {calib_s:.1f}s", flush=True)

    # Summarize variant choices and SC-coverage of comp
    chosen = [r.get("variant") for r in calib_report]
    n_fp_comp = sum(1 for c in chosen if c is not None and c.startswith("fp/"))
    n_sc_comp = len(chosen) - n_fp_comp
    n_enabled = sum(1 for r in calib_report if r.get("enabled", True))
    n_blocks_total = len(calib_report)
    print(f"[comp] per-block choices: SC={n_sc_comp}  FP={n_fp_comp}  "
          f"enabled={n_enabled}/{n_blocks_total} (r2_threshold={args.r2_threshold})",
          flush=True)

    # Corrected eff_sl: scale comp_macs by the fraction of blocks that actually
    # run compensation after r2-threshold gating. compute_stoc_len_stats above
    # assumed all blocks get comp; this recomputes the real per-MAC SL average.
    if args.comp_mode == "sc" and n_blocks_total > 0:
        _SEQ_C = 257
        _OP_MACS_C = {
            "qk":      16 * _SEQ_C * 64 * _SEQ_C,
            "av":      16 * _SEQ_C * _SEQ_C * 64,
            "proj":    _SEQ_C * 1024 * 3072 + _SEQ_C * 1024 * 1024,
            "mlp_fc1": _SEQ_C * 1024 * 4096,
            "mlp_fc2": _SEQ_C * 4096 * 1024,
        }
        _COMP_MACS_C = _SEQ_C * 1024 * 1024
        main_macs_full = sl_stats["main_macs_B"] * 1e9
        main_bitops_full = main_macs_full * sl_stats["main_sl"]
        comp_macs_real = n_enabled * _COMP_MACS_C
        comp_bitops_real = comp_macs_real * (2 ** args.comp_sc_prec)
        total_real = main_macs_full + comp_macs_real
        eff_sl_real = ((main_bitops_full + comp_bitops_real) / total_real
                       if total_real > 0 else 0)
        sl_stats["eff_sl_r2corrected"] = round(eff_sl_real, 2)
        sl_stats["n_enabled_blocks"] = n_enabled
        sl_stats["comp_macs_real_B"] = round(comp_macs_real / 1e9, 3)
        print(f"[sl] corrected eff_sl (only {n_enabled} enabled blocks run comp) "
              f"= {eff_sl_real:.2f}  vs nominal {sl_stats['eff_sl']}", flush=True)
    from collections import Counter
    cnt = Counter([c for c in chosen if c is not None])
    for k, v in cnt.most_common():
        print(f"        {k}: {v} blocks", flush=True)

    print("[eval] SC + comp", flush=True)
    seed_all(args.eval_seed)
    res_sc_comp = evaluate(model_sc, eval_loader, device)
    print("[result:sc_comp]", res_sc_comp, flush=True)

    print("[eval] FP reference", flush=True)
    seed_all(args.eval_seed)
    res_fp = evaluate(model_fp, eval_loader, device)
    print("[result:fp]", res_fp, flush=True)

    out = {
        "config": {
            "sc_config": args.sc_config, "sc_prec": args.sc_prec,
            "n_calib": args.n_calib, "n_eval": args.n_eval,
            "calib_seed": args.calib_seed, "eval_seed": args.eval_seed,
            "ridge": args.ridge, "start_block": args.start_block,
            "avg_sc_draws": args.avg_sc_draws,
            "comp_mode": args.comp_mode,
            "comp_sc_prec": args.comp_sc_prec, "comp_sc_mode": args.comp_sc_mode,
            "n_variants": args.n_variants, "polarity_flip": args.polarity_flip,
            "w_scales": args.w_scales,
        },
        "calib": {
            "per_block": calib_report,
            "elapsed_s": calib_s,
            "n_fp_comp_blocks": n_fp_comp,
            "n_sc_comp_blocks": n_sc_comp,
            "variant_counts": dict(cnt),
        },
        "results": {"fp": res_fp, "sc_raw": res_sc_raw, "sc_comp": res_sc_comp},
        "stoc_len": sl_stats,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out_json}", flush=True)

    print("\n=== SUMMARY ===")
    print(f"  FP                : top1={res_fp['top1']:.4f}")
    if res_sc_raw:
        print(f"  SC (raw)          : top1={res_sc_raw['top1']:.4f}")
    print(f"  SC + comp         : top1={res_sc_comp['top1']:.4f}")
    print(f"  main_sl={sl_stats['main_sl']}  eff_sl={sl_stats['eff_sl']}  "
          f"reduction={sl_stats['eff_reduction_pct']}%")
    if args.comp_mode == "sc":
        print(f"  Comp SC coverage  : {n_sc_comp}/{len(chosen)} blocks "
              f"({100*n_sc_comp/max(1,len(chosen)):.0f}% no-FP-mac)")


if __name__ == "__main__":
    main()
