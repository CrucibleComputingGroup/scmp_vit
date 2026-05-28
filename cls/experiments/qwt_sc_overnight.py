"""QwT-SC uniform-path driver. Installs the cross-seed cosine-gated
compensator on a ViT-L/14 with SC matmuls patched in, evaluates on
ImageNet-1k.

The single-recipe admission rule lives in ``qwt_sc/compensation.py``
(``calibrate_qwt``); see that module's docstring for the physical
derivation and pilot evidence. This driver only handles data/model setup
and CLI plumbing.

Usage::

    python experiments/qwt_sc_overnight.py --sc_config qk_only \\
        --n_calib 1024 --n_eval 50000 --batch_size 64 \\
        --cos_threshold 0.5 --last_block_cos_threshold 0.8 \\
        --lookahead_veto \\
        --out_json results/qwt_sc_qk_only_n50k.json
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
from scmp_kernels.sc.config_helpers import (
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
        "eff_reduction_pct": round((1 - eff_sl / base_sl) * 100, 1) if base_sl else 0.0,
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


# ---------- Comp-kernel factories ----------
# The cross-seed gate in qwt_sc.calibrate_qwt decides *whether* to install a
# correction; these factories decide *how* the installed W̄, b̄ is evaluated
# at inference — bare FP Linear, SCLinear with a chosen Sobol cfg, or
# HeadAlignedSCLinear (per-head D=64 reusing the block QK Sobol pool).

def build_seed_bank(D, sc_prec, n_variants, rng_seed=42):
    """Build n_variants Sobol configs at dim D. Always includes default +
    antithetic; pads with deterministic alt-seed variants for exploration."""
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
    """(W, b) -> SCLinear wrapping the ridge-fit (D_in, D_out) matmul. The
    comp kernel then runs through the same XNOR/Sobol pipeline as the block
    SC matmuls — so the whole inference path stays SC (no FP multiplier)."""
    def factory(W, b):
        D_in, D_out = W.shape
        layer = nn.Linear(D_in, D_out, bias=True)
        with torch.no_grad():
            layer.weight.copy_((W.t().contiguous()) * (scale * polarity))
            layer.bias.copy_(b * polarity if polarity == -1 else b)
        return SCLinear(layer, sc_prec=sc_prec, mode=mode, cfg_override=cfg_override)
    return factory


def make_fp_factory(scale=1.0, polarity=1):
    """(W, b) -> nn.Linear FP comp. Debug / baseline only — FP comp
    reintroduces an FP multiplier that the SC hardware story was designed
    to avoid."""
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
    """(W, b) -> HeadAlignedSCLinear. Splits the comp matmul into n_heads
    chunks of size (D_in/n_heads, D_out/n_heads), each reusing the
    _CFG_CACHE[(D_in/n_heads, sc_prec)] entry that the block's per-head QK
    matmul already populated — so the comp adds **zero** new Sobol config
    cache, maximizing SC hardware reuse."""
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
    ap.add_argument("--start_block", type=int, default=0,
                    help="Indices below this are never admitted (pass-through). "
                         "Useful if early-block calib is unreliable for a model.")
    # ---- Cross-seed gate (the single-recipe admission rule) ----
    ap.add_argument("--cos_threshold", type=float, default=0.5,
                    help="Admission threshold τ. A block is admitted iff "
                         "cos(W_A, W_B) > τ, where W_A and W_B are the ridge "
                         "fits on two disjoint calibration batches. Pilot on "
                         "5 regimes shows a bimodal cos distribution (noise "
                         "≈ 0, signal ≥ 0.68) — any τ in [0.2, 0.65] gives "
                         "the same admission decisions. Default 0.5.")
    ap.add_argument("--calib_seed_b", type=int, default=2,
                    help="Seed for the second calibration loader. Must differ "
                         "from --calib_seed to draw disjoint images.")
    ap.add_argument("--norm_floor", type=float, default=0.0,
                    help="Additionally require min(||W_A||,||W_B||) > this. "
                         "0 = disabled. Filters out near-zero-bias blocks "
                         "where cosine is numerically noisy.")
    ap.add_argument("--last_block_cos_threshold", type=float, default=0.8,
                    help="Stricter τ for the last block (block 23 feeds the "
                         "pre-head embedding directly; demand tighter W "
                         "agreement). Default 0.8. Set < --cos_threshold to "
                         "disable the extra-strict rule.")
    ap.add_argument("--lookahead_veto", action="store_true",
                    help="One-step binary lookahead: propagate 'apply' vs "
                         "'skip' through block i+1 and veto apply if skip is "
                         "closer to the FP chain. Rarely fires under the "
                         "cosine gate but is cheap insurance for borderline "
                         "blocks.")
    ap.add_argument("--fwd_chunk", type=int, default=32)
    ap.add_argument("--skip_baseline", action="store_true",
                    help="Skip the raw-SC evaluation (useful when the baseline "
                         "is already recorded from a sibling sweep).")
    # ---- Comp-kernel choice (how the installed W̄, b̄ is executed) ----
    ap.add_argument("--comp_mode", choices=["fp", "sc"], default="sc",
                    help="Kernel for the installed correction matmul. 'sc' "
                         "(default) routes W̄ through SCLinear so the whole "
                         "inference path stays in SC hardware. 'fp' uses a "
                         "bare nn.Linear — debug / baseline only.")
    ap.add_argument("--comp_sc_prec", type=int, default=8,
                    help="SC bit-precision of the comp matmul (when --comp_mode sc).")
    ap.add_argument("--comp_sc_mode", choices=["bipolar", "unipolar"], default="bipolar",
                    help="Per-row SC quant mode of the comp matmul.")
    ap.add_argument("--n_variants", type=int, default=1,
                    help="Size of the Sobol-seed bank for the comp kernel "
                         "(default + antithetic + alt-seeds). Per-block "
                         "selection picks the variant with lowest post-comp "
                         "residual = log2(n_variants) config bits per block. "
                         "1 = no search (default).")
    ap.add_argument("--polarity_flip", action="store_true",
                    help="Add the -1 polarity variant for each Sobol cfg. "
                         "Doubles n_variants (+1 extra config bit per block).")
    ap.add_argument("--w_scales", type=str, default="1.0",
                    help="Comma-separated W-magnitude scales to search over, "
                         "e.g. '0.5,0.75,1.0'. Per-block pick. Extra "
                         "log2(n_scales) config bits.")
    ap.add_argument("--head_aligned", action="store_true",
                    help="Include head-aligned SC comp variants (per-head "
                         "D=64 SC matmul reusing _CFG_CACHE[(64, sc_prec)] = "
                         "block QK pool — 0 extra Sobol config cache).")
    ap.add_argument("--head_aligned_only", action="store_true",
                    help="Drop the full-width SC variants; use only head-"
                         "aligned. Implies --head_aligned.")
    ap.add_argument("--n_heads", type=int, default=16,
                    help="Head count for HeadAlignedSCLinear; must divide D=1024.")
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

    # Build the comp-kernel variants: cartesian of
    # {Sobol seeds} × {polarities} × {scales} × {full-width, head-aligned}.
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
            for pol in polarities:
                for sc in scales:
                    name = f"head_aligned/p{'+' if pol > 0 else '-'}/s{sc:.2f}/h{args.n_heads}"
                    variants.append((name, make_head_aligned_factory(
                        args.comp_sc_prec, args.comp_sc_mode,
                        n_heads=args.n_heads, cfg_override=None,
                        scale=sc, polarity=pol)))
    else:  # fp comp — debug / baseline
        for pol in polarities:
            for sc in scales:
                name = f"fp/p{'+' if pol > 0 else '-'}/s{sc:.2f}"
                variants.append((name, make_fp_factory(scale=sc, polarity=pol)))

    print(f"[comp] mode={args.comp_mode}  variants={len(variants)}  "
          f"(seeds={len(seed_bank)} pols={len(polarities)} scales={len(scales)})", flush=True)
    for name, _ in variants[:8]:
        print(f"        - {name}", flush=True)
    if len(variants) > 8:
        print(f"        ... ({len(variants)-8} more)", flush=True)

    comp_factory = None
    comp_factory_variants = None
    if len(variants) == 1:
        comp_factory = variants[0][1]
    elif len(variants) > 1:
        comp_factory_variants = variants

    print("[calib] starting QwT cross-seed calibration", flush=True)
    t_cal = time.time()
    seed_all(args.calib_seed)
    blocks_fp = list(get_blocks(model_fp))
    blocks_sc_container = get_blocks(model_sc)
    if args.calib_seed_b == args.calib_seed:
        raise ValueError("--calib_seed_b must differ from --calib_seed")
    calib_ds_b = build_dataset(args.data_root, args.calib_seed_b, args.n_calib)
    calib_loader_b = DataLoader(calib_ds_b, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
    last_tau = args.last_block_cos_threshold if args.last_block_cos_threshold > args.cos_threshold else None
    print(f"[calib] τ={args.cos_threshold}  seed_a={args.calib_seed} "
          f"seed_b={args.calib_seed_b}  norm_floor={args.norm_floor}  "
          f"last_τ={last_tau}", flush=True)
    calib_report = calibrate_qwt(
        model_fp=model_fp, model_sc=model_sc,
        blocks_fp=blocks_fp, blocks_sc_container=blocks_sc_container,
        calib_loader_a=calib_loader, calib_loader_b=calib_loader_b,
        device=device, n_calib=args.n_calib, ridge=args.ridge,
        start_block=args.start_block, fwd_chunk=args.fwd_chunk,
        cos_threshold=args.cos_threshold,
        norm_floor=args.norm_floor,
        last_block_cos_threshold=last_tau,
        lookahead_veto=args.lookahead_veto,
        comp_factory=comp_factory,
        comp_factory_variants=comp_factory_variants,
    )
    calib_s = time.time() - t_cal
    print(f"[calib] done in {calib_s:.1f}s", flush=True)

    n_enabled = sum(1 for r in calib_report if r.get("enabled", False))
    n_blocks_total = len(calib_report)
    en_blocks = [r["block"] for r in calib_report if r["enabled"]]
    cos_admitted = [r["cos_ab"] for r in calib_report if r["enabled"]]
    cos_rejected = [r["cos_ab"] for r in calib_report if not r["enabled"]]
    print(f"[gate] admitted={n_enabled}/{n_blocks_total}  blocks={en_blocks}", flush=True)
    if cos_admitted:
        print(f"[gate] cos range admitted=[{min(cos_admitted):+.3f}, "
              f"{max(cos_admitted):+.3f}]", flush=True)
    if cos_rejected:
        print(f"[gate] cos range rejected=[{min(cos_rejected):+.3f}, "
              f"{max(cos_rejected):+.3f}]", flush=True)

    print("[eval] SC + comp", flush=True)
    seed_all(args.eval_seed)
    res_sc_comp = evaluate(model_sc, eval_loader, device)
    print("[result:sc_comp]", res_sc_comp, flush=True)

    print("[eval] FP reference", flush=True)
    seed_all(args.eval_seed)
    res_fp = evaluate(model_fp, eval_loader, device)
    print("[result:fp]", res_fp, flush=True)

    from collections import Counter
    chosen = [r.get("variant") for r in calib_report]
    variant_counts = dict(Counter([c for c in chosen if c is not None]))

    out = {
        "config": {
            "sc_config": args.sc_config, "sc_prec": args.sc_prec,
            "n_calib": args.n_calib, "n_eval": args.n_eval,
            "calib_seed": args.calib_seed, "calib_seed_b": args.calib_seed_b,
            "eval_seed": args.eval_seed,
            "ridge": args.ridge, "start_block": args.start_block,
            "cos_threshold": args.cos_threshold,
            "norm_floor": args.norm_floor,
            "last_block_cos_threshold": args.last_block_cos_threshold,
            "lookahead_veto": bool(args.lookahead_veto),
            "comp_mode": args.comp_mode,
            "comp_sc_prec": args.comp_sc_prec,
            "comp_sc_mode": args.comp_sc_mode,
            "n_variants": args.n_variants,
            "polarity_flip": bool(args.polarity_flip),
            "w_scales": args.w_scales,
            "head_aligned": bool(args.head_aligned or args.head_aligned_only),
            "n_heads": args.n_heads if (args.head_aligned or args.head_aligned_only) else None,
        },
        "calib": {
            "per_block": calib_report,
            "elapsed_s": calib_s,
            "n_enabled_blocks": n_enabled,
            "variant_counts": variant_counts,
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


if __name__ == "__main__":
    main()
