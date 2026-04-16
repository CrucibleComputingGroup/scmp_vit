"""Extended QwT-SC compensator with K-seed bank, polarity flip, W-scale
search, and per-block FP/SC mixed choice. Also records per-block correlation
diagnostics so we can tell *why* a variant was selected.

The hardware framing is unchanged from the parent driver: each chosen
non-default option is at most O(log K) config bits per block (a tiny LUT in
the SC accelerator's per-block scheduler), or in the mixed-FP/SC case, a
single 1-bit "use FP comp" flag per block (24 bits total for ViT-L).

Usage::

    python experiments/qwt_sc_overnight.py --sc_config qk_only \\
        --n_calib 256 --n_eval 500 --batch_size 8 \\
        --comp_mode sc --n_variants 8 --polarity_flip \\
        --w_scales 0.5,0.75,1.0 --mixed_fp_sc \\
        --out_json results/qwt_sc_qk_only_overnight.json
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
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
    """Factory returning an nn.Linear (FP comp) for mixed FP/SC selection."""
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
    ap.add_argument("--data_root", default="./data/imagenet/data")
    ap.add_argument("--sc_prec", type=int, default=8)
    ap.add_argument("--sc_config",
                    choices=list(SC_PRESETS) + list(FINE_GRAINED_KS),
                    default="qk_only")
    ap.add_argument("--n_calib", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--calib_seed", type=int, default=1)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ridge", type=float, default=1e-2)
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
    ap.add_argument("--mixed_fp_sc", action="store_true",
                    help="Per block, also try an FP comp; pick winner. "
                         "1 extra config bit per block. Reports SC-coverage of comp.")
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
    args = ap.parse_args()

    os.environ.setdefault("XFORMERS_DISABLED", "1")
    seed_all(args.calib_seed)
    device = torch.device("cuda")
    set_noise_model(False)

    print(f"[env] torch {torch.__version__} cuda {torch.version.cuda} "
          f"dev={torch.cuda.get_device_name(0)}", flush=True)

    fine_schedule = None
    preset = None
    if args.sc_config in SC_PRESETS:
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

    res_sc_raw = None
    if not args.skip_baseline:
        seed_all(args.eval_seed)
        res_sc_raw = evaluate(model_sc, eval_loader, device)
        print("[result:sc_raw]", res_sc_raw, flush=True)

    # Build the comp variants: cartesian of {seed configs} x {polarities} x {scales}
    # x {fp/sc} (if mixed_fp_sc).
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
        if args.mixed_fp_sc:
            for pol in polarities:
                for sc in scales:
                    name = f"fp/p{'+' if pol > 0 else '-'}/s{sc:.2f}"
                    variants.append((name, make_fp_factory(scale=sc, polarity=pol)))
    else:  # comp_mode == "fp"
        for pol in polarities:
            for sc in scales:
                name = f"fp/p{'+' if pol > 0 else '-'}/s{sc:.2f}"
                variants.append((name, make_fp_factory(scale=sc, polarity=pol)))

    print(f"[comp] variants={len(variants)}  "
          f"(seeds={len(seed_bank)} pols={len(polarities)} "
          f"scales={len(scales)} mixed_fp_sc={args.mixed_fp_sc})",
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
    )
    calib_s = time.time() - t_cal
    print(f"[calib] done in {calib_s:.1f}s", flush=True)

    # Summarize variant choices and SC-coverage of comp
    chosen = [r.get("variant") for r in calib_report]
    n_fp_comp = sum(1 for c in chosen if c is not None and c.startswith("fp/"))
    n_sc_comp = len(chosen) - n_fp_comp
    print(f"[comp] per-block choices: SC={n_sc_comp}  FP={n_fp_comp}", flush=True)
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
            "w_scales": args.w_scales, "mixed_fp_sc": args.mixed_fp_sc,
        },
        "calib": {
            "per_block": calib_report,
            "elapsed_s": calib_s,
            "n_fp_comp_blocks": n_fp_comp,
            "n_sc_comp_blocks": n_sc_comp,
            "variant_counts": dict(cnt),
        },
        "results": {"fp": res_fp, "sc_raw": res_sc_raw, "sc_comp": res_sc_comp},
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
    if args.comp_mode == "sc":
        print(f"  Comp SC coverage  : {n_sc_comp}/{len(chosen)} blocks "
              f"({100*n_sc_comp/max(1,len(chosen)):.0f}% no-FP-mac)")


if __name__ == "__main__":
    main()
