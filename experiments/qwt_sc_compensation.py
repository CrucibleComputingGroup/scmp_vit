"""QwT-style closed-form linear compensation for SC accuracy loss.

Thin driver that couples the ``qwt_sc`` library (living in the QwT-SC fork
submodule at ``third_party/QwT-SC/QwT-vit-sc``) to the DINOv2 ViT-L/14 + SC
kernels in this repo. All of the reusable logic — CompensationBlock,
closed_form_ridge, sequential calibration — lives in the submodule.

Usage examples::

    # The worst-hit baseline (full attention SC):
    python experiments/qwt_sc_compensation.py \\
        --sc_config full_attn --n_calib 256 --n_eval 500 \\
        --out_json results/qwt_sc_full_attn_n500.json

    # State-of-the-art: fine-grained sensitivity scheduling + QwT comp:
    python experiments/qwt_sc_compensation.py \\
        --sc_config skip_worst50 --n_calib 256 --n_eval 500 \\
        --out_json results/qwt_sc_skip_worst50_n500.json
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
# Library lives in the QwT-SC fork, vendored as a git submodule.
QWT_SC_LIB = HERE / "third_party" / "QwT-SC" / "QwT-vit-sc"
if not (QWT_SC_LIB / "qwt_sc").exists():
    raise RuntimeError(
        f"qwt_sc library not found at {QWT_SC_LIB}. Did you forget to run "
        f"`git submodule update --init --recursive`?"
    )
sys.path.insert(0, str(QWT_SC_LIB))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from eval import build_transform, load_model, seed_all
from imagenet_parquet import ImageNetParquetVal
from sc_attention_patch import patch_model, set_noise_model, SC_OP_NAMES, SCLinear

from qwt_sc import calibrate_qwt  # noqa: E402


def make_sc_comp_factory(sc_prec: int, mode: str, cfg_override=None):
    """Return a closure (W, b) -> SCLinear init'd from W, b.

    When ``cfg_override`` is None, SCLinear uses the cached default Sobol
    config (same as the block's D=D_in SC ops). When provided, the comp's SC
    matmul uses ``cfg_override`` — a different Sobol scrambling — to give the
    comp a *different* SC noise realization than the block. Per-block variant
    selection at calibration picks whichever realization correlates *least*
    with the block's residual.
    """
    def factory(W: torch.Tensor, b: torch.Tensor) -> nn.Module:
        D_in, D_out = W.shape
        layer = nn.Linear(D_in, D_out, bias=True)
        with torch.no_grad():
            layer.weight.copy_(W.t().contiguous())
            layer.bias.copy_(b)
        return SCLinear(layer, sc_prec=sc_prec, mode=mode,
                         cfg_override=cfg_override)
    return factory

SENSITIVITY_JSON = HERE / "results" / "sensitivity_all_ops.json"


def build_skip_worst_k_schedule(k: int, n_blocks: int = 24):
    """Re-create e2e_fine_grained.py's sc_skip_worst_k schedule.

    SC every (op, block) in {mlp_fc1, mlp_fc2, qk, av, proj} x n_blocks except
    the top-K worst (op, block) pairs by calibrated-noise surrogate L2.
    """
    with open(SENSITIVITY_JSON) as f:
        data = json.load(f)
    rows = sorted(data["grid"], key=lambda r: r["l2"], reverse=True)
    drop = {(r["op"], int(r["block"])) for r in rows[:k]}
    spec = {name: [1] * n_blocks for name in SC_OP_NAMES}
    for (op, bi) in drop:
        spec[op][bi] = 0
    return spec


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


SC_PRESETS = {
    "full_attn": dict(sc_qk=True, sc_av=True, sc_qkv_proj=True, sc_out_proj=True,
                      sc_mlp=False),
    "qk_only":   dict(sc_qk=True, sc_av=False, sc_qkv_proj=False, sc_out_proj=False,
                      sc_mlp=False),
    "qk_av":     dict(sc_qk=True, sc_av=True, sc_qkv_proj=False, sc_out_proj=False,
                      sc_mlp=False),
}

# Fine-grained per-(op, block) schedules from the sensitivity matrix. Values
# are K = number of worst (op, block) pairs left in FP. Match the README.
FINE_GRAINED_KS = {
    "skip_worst50": 50,   # 74.8% raw baseline (best known)
    "skip_worst40": 40,   # 74.2%
    "skip_worst30": 30,   # 68.8% (knee)
    "skip_worst20": 20,   # 43.8%
    "all_ops":       0,   # 0% (the "everything" config)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data/imagenet/data")
    ap.add_argument("--sc_prec", type=int, default=8)
    ap.add_argument("--sc_config",
                    choices=list(SC_PRESETS) + list(FINE_GRAINED_KS),
                    default="full_attn",
                    help=("preset name. SC_PRESETS={full_attn,qk_only,qk_av} or "
                          "FINE_GRAINED_KS={skip_worst50,40,30,20,all_ops}."))
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
    ap.add_argument("--skip_baseline", action="store_true",
                    help="skip the no-comp SC baseline eval (saves time)")
    ap.add_argument("--comp_mode", choices=["fp", "sc"], default="fp",
                    help="fp = original QwT-SC linear comp; sc = SC matmul comp "
                         "reusing the block's _CFG_CACHE Sobol pool.")
    ap.add_argument("--comp_sc_prec", type=int, default=8)
    ap.add_argument("--comp_sc_mode", choices=["bipolar", "unipolar"],
                    default="bipolar")
    ap.add_argument("--comp_refit_iters", type=int, default=0,
                    help="if >0 and comp_mode=sc, do K rounds of "
                         "noise-aware refit: subtract the comp's measured SC "
                         "noise δ from the LS target before refitting W.")
    ap.add_argument("--comp_variant_select", action="store_true",
                    help="per-block, try {default, antithetic-K} Sobol "
                         "configs for the comp and pick the lower-residual "
                         "one. 1 config-bit per block in hardware. NOTE: "
                         "value-complement antithetic gives only ~−0.15 "
                         "noise correlation in our kernel, so empirical gain "
                         "is small and not strictly Pareto.")
    ap.add_argument("--out_json", default="results/qwt_sc_pilot.json")
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
        cfg_desc = f"fine_grained skip_worst_K={k} (SC ops: {total_on}/120)"
    print(f"[cfg] sc_config={args.sc_config} -> {cfg_desc}  "
          f"sc_prec={args.sc_prec}  n_calib={args.n_calib}  n_eval={args.n_eval}  "
          f"ridge={args.ridge}  start_block={args.start_block}  "
          f"avg_sc_draws={args.avg_sc_draws}", flush=True)

    print("[data] loading calibration subset", flush=True)
    calib_ds = build_dataset(args.data_root, args.calib_seed, args.n_calib)
    calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=False)
    print("[data] loading eval subset", flush=True)
    eval_ds = build_dataset(args.data_root, args.eval_seed, args.n_eval)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True,
                             drop_last=False)

    print("[model] loading FP reference model", flush=True)
    model_fp = load_model(device).eval()
    print("[model] loading SC-to-be-patched model", flush=True)
    model_sc = load_model(device).eval()

    if preset is not None:
        print(f"[sc] patching model_sc with preset {preset}", flush=True)
        stats = patch_model(model_sc, sc_prec=args.sc_prec, **preset)
    else:
        print(f"[sc] patching model_sc with fine-grained schedule "
              f"({args.sc_config})", flush=True)
        stats = patch_model(model_sc, sc_prec=args.sc_prec,
                            sc_ops_per_block=fine_schedule)
    print(f"[sc] patched={stats}", flush=True)

    res_sc_raw = None
    if not args.skip_baseline:
        print("[eval] SC baseline (no compensation)", flush=True)
        seed_all(args.eval_seed)
        res_sc_raw = evaluate(model_sc, eval_loader, device)
        print("[result:sc_raw]", res_sc_raw, flush=True)

    comp_factory = None
    comp_factory_variants = None
    if args.comp_mode == "sc":
        if args.comp_variant_select:
            # The comp matmul has D_in == D_out == 1024 for ViT-L (residual
            # stream). Build both candidate Sobol scramblings at that D and
            # let calibration pick per block.
            from config_helpers import make_sobol_simple_config, make_sobol_antithetic_config
            D = 1024
            cfg_default = make_sobol_simple_config(D, D, args.comp_sc_prec)
            cfg_anti    = make_sobol_antithetic_config(D, D, args.comp_sc_prec)
            comp_factory_variants = [
                ("default",    make_sc_comp_factory(args.comp_sc_prec, args.comp_sc_mode, cfg_override=cfg_default)),
                ("antithetic", make_sc_comp_factory(args.comp_sc_prec, args.comp_sc_mode, cfg_override=cfg_anti)),
            ]
            print(f"[comp] SC compensator: sc_prec={args.comp_sc_prec} "
                  f"mode={args.comp_sc_mode}  variants=2 (default, antithetic)",
                  flush=True)
        else:
            comp_factory = make_sc_comp_factory(
                sc_prec=args.comp_sc_prec, mode=args.comp_sc_mode)
            print(f"[comp] SC compensator: sc_prec={args.comp_sc_prec} "
                  f"mode={args.comp_sc_mode}", flush=True)
    else:
        print(f"[comp] FP compensator (baseline)", flush=True)

    print("[calib] starting QwT-style sequential calibration", flush=True)
    t_cal = time.time()
    seed_all(args.calib_seed)
    blocks_fp = list(get_blocks(model_fp))
    blocks_sc_container = get_blocks(model_sc)
    calib_report = calibrate_qwt(
        model_fp=model_fp, model_sc=model_sc,
        blocks_fp=blocks_fp,
        blocks_sc_container=blocks_sc_container,
        calib_loader=calib_loader,
        device=device,
        n_calib=args.n_calib,
        ridge=args.ridge,
        start_block=args.start_block,
        fwd_chunk=args.fwd_chunk,
        avg_sc_draws=args.avg_sc_draws,
        comp_factory=comp_factory,
        comp_factory_variants=comp_factory_variants,
        comp_refit_iters=args.comp_refit_iters,
    )
    calib_s = time.time() - t_cal
    print(f"[calib] done in {calib_s:.1f}s", flush=True)

    print("[eval] SC + QwT compensation", flush=True)
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
            "batch_size": args.batch_size,
            "comp_mode": args.comp_mode,
            "comp_sc_prec": args.comp_sc_prec,
            "comp_sc_mode": args.comp_sc_mode,
            "comp_refit_iters": args.comp_refit_iters,
            "comp_variant_select": args.comp_variant_select,
        },
        "calib": {"per_block": calib_report, "elapsed_s": calib_s},
        "results": {"fp": res_fp, "sc_raw": res_sc_raw, "sc_comp": res_sc_comp},
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out_json}", flush=True)

    print("\n=== SUMMARY ===")
    print(f"  FP                : top1={res_fp['top1']:.4f}")
    if res_sc_raw:
        print(f"  SC ({args.sc_config})  : top1={res_sc_raw['top1']:.4f}")
    print(f"  SC + QwT comp     : top1={res_sc_comp['top1']:.4f}")
    if res_sc_raw:
        delta = res_sc_comp['top1'] - res_sc_raw['top1']
        gap = res_fp['top1'] - res_sc_raw['top1']
        closed = (delta / gap * 100.0) if gap > 0 else float('nan')
        print(f"  Gain over SC      : +{delta*100:.2f} pt  "
              f"(closes {closed:.1f}% of FP-SC gap)")


if __name__ == "__main__":
    main()
