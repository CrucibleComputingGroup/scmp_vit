"""INT post-training quantization baseline on DINOv2 ViT-L/14, with optional QwT.

Wraps each transformer block's qkv / proj / fc1 / fc2 nn.Linear with
fake-quant adapters from
``third_party/QwT-SC/QwT-cls-RepQ-ViT/quant.QuantLinear`` (UniformQuantizer,
per-tensor asymmetric activations, per-channel weights). Patch embed and
classification head stay FP. QK and AV matmuls stay FP (linear-only INT —
the simplest first cut; the SC sweep's QK-FP variant is the directly
analogous comparison point).

For the QwT variant: reuses ``qwt_sc.compensation.calibrate_qwt`` from the
SC pipeline. Per ``compensation.py`` module docstring the calibrator is
kernel-agnostic — it observes ``model_sc`` only via ``block(x)`` forwards,
so passing the INT-patched model in the ``model_sc`` slot works unchanged.
``comp_factory=None`` falls back to a plain FP nn.Linear residual (no
SC-specific kernel needed in the comp path).

JSON schema mirrors ``cls/results/sweep_2026-04-25/`` so the cells slot
into the same comparison table.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # vit_sc/
CLS = HERE / "cls"
QWT_SC_LIB = HERE / "third_party" / "QwT-SC" / "QwT-vit-sc"
for p in (HERE, CLS, str(QWT_SC_LIB), HERE / "sc"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from _int_quant import QuantLinear
from eval import build_transform, load_model, seed_all
from imagenet_parquet import ImageNetParquetVal
from qwt_sc import calibrate_qwt


def get_blocks(model):
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        return model.backbone.blocks
    return model.blocks


def evaluate(model, loader, device, log_every=20):
    model.eval()
    n = top1 = top5 = 0
    t0 = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            _, p5 = logits.topk(5, dim=1)
            c = p5.eq(y.unsqueeze(1))
            top1 += c[:, 0].sum().item()
            top5 += c.any(dim=1).sum().item()
            n += y.numel()
            if (i + 1) % log_every == 0:
                dt = time.time() - t0
                print(f"  [{i+1}] top1={top1/n:.4f} top5={top5/n:.4f} "
                      f"({n} imgs, {dt:.1f}s, {n/dt:.1f} img/s)", flush=True)
    dt = time.time() - t0
    return {"n": n, "top1": top1 / n, "top5": top5 / n,
            "elapsed_s": dt, "img_per_s": n / dt if dt > 0 else 0.0}


def wrap_int(linear: nn.Linear, w_bit: int, a_bit: int) -> QuantLinear:
    new_m = QuantLinear(
        linear.in_features, linear.out_features,
        input_quant_params={"n_bits": a_bit, "channel_wise": False},
        weight_quant_params={"n_bits": w_bit, "channel_wise": True},
    )
    new_m.weight.data = linear.weight.data.clone()
    if linear.bias is not None:
        new_m.bias = nn.Parameter(linear.bias.data.clone())
    else:
        new_m.bias = None
    new_m.set_quant_state(input_quant=True, weight_quant=True)
    return new_m


def patch_model_int(model, w_bit, a_bit):
    """Replace each block's qkv / proj / fc1 / fc2 nn.Linear with QuantLinear."""
    stats = {"qkv": 0, "proj": 0, "fc1": 0, "fc2": 0}
    blocks = get_blocks(model)
    for blk in blocks:
        attn = None
        for m in blk.modules():
            if (hasattr(m, "qkv") and hasattr(m, "proj")
                    and hasattr(m, "num_heads")):
                attn = m
                break
        if attn is not None:
            if isinstance(attn.qkv, nn.Linear):
                attn.qkv = wrap_int(attn.qkv, w_bit, a_bit)
                stats["qkv"] += 1
            if isinstance(attn.proj, nn.Linear):
                attn.proj = wrap_int(attn.proj, w_bit, a_bit)
                stats["proj"] += 1
        mlp = getattr(blk, "mlp", None)
        if mlp is not None:
            if isinstance(getattr(mlp, "fc1", None), nn.Linear):
                mlp.fc1 = wrap_int(mlp.fc1, w_bit, a_bit)
                stats["fc1"] += 1
            if isinstance(getattr(mlp, "fc2", None), nn.Linear):
                mlp.fc2 = wrap_int(mlp.fc2, w_bit, a_bit)
                stats["fc2"] += 1
    return stats


def init_quantizer_scales(model, loader, device, n_calib_init):
    """One forward pass over ``n_calib_init`` images to trigger
    ``UniformQuantizer.init_quantization_scale`` on every wrapper."""
    model.eval()
    seen = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            model(x)
            seen += x.size(0)
            if seen >= n_calib_init:
                break
    return seen


def build_dataset(data_root, seed, n):
    ds = ImageNetParquetVal(data_root, transform=build_transform(224))
    if n is not None and n < len(ds):
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        ds = Subset(ds, idx[:n])
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",
                    default="/scratch/nbleier_owned_root/nbleier_owned1/"
                            "shared_data/imagenet/data")
    ap.add_argument("--w_bit", type=int, required=True)
    ap.add_argument("--a_bit", type=int, required=True)
    ap.add_argument("--qwt", type=int, default=0,
                    help="0 = INT only, 1 = INT + QwT cross-seed comp")
    ap.add_argument("--n_calib_init", type=int, default=128,
                    help="images for one-shot quantizer scale init")
    ap.add_argument("--n_calib_qwt", type=int, default=1024,
                    help="images per seed for QwT cross-seed calib")
    ap.add_argument("--n_eval", type=int, default=50000,
                    help="0 = full ImageNet val (50000)")
    ap.add_argument("--calib_seed", type=int, default=1)
    ap.add_argument("--calib_seed_b", type=int, default=2)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--cos_threshold", type=float, default=0.5)
    ap.add_argument("--last_block_cos_threshold", type=float, default=0.8)
    ap.add_argument("--norm_floor", type=float, default=0.0)
    ap.add_argument("--lookahead_veto", action="store_true")
    ap.add_argument("--start_block", type=int, default=0)
    ap.add_argument("--fwd_chunk", type=int, default=32)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    os.environ.setdefault("XFORMERS_DISABLED", "1")
    seed_all(args.calib_seed)
    device = torch.device("cuda")
    print(f"[env] torch {torch.__version__} cuda {torch.version.cuda} "
          f"dev={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[cfg] W{args.w_bit}A{args.a_bit}  qwt={args.qwt}  "
          f"n_calib_init={args.n_calib_init}  n_calib_qwt={args.n_calib_qwt}  "
          f"n_eval={args.n_eval}", flush=True)

    n_eval = args.n_eval if args.n_eval > 0 else None
    print("[data] building loaders", flush=True)
    n_calib_loader_a = args.n_calib_qwt if args.qwt else args.n_calib_init
    calib_ds_a = build_dataset(args.data_root, args.calib_seed, n_calib_loader_a)
    calib_loader_a = DataLoader(calib_ds_a, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers,
                                pin_memory=True)
    eval_ds = build_dataset(args.data_root, args.eval_seed, n_eval)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers,
                             pin_memory=True)
    print(f"[data] eval={len(eval_ds)} imgs", flush=True)

    print("[model] loading FP + INT copies", flush=True)
    model_fp = load_model(device).eval()
    model_int = load_model(device).eval()

    stats = patch_model_int(model_int, args.w_bit, args.a_bit)
    print(f"[int] patched={stats}", flush=True)

    print(f"[calib] init quantizer scales on >={args.n_calib_init} images",
          flush=True)
    seen = init_quantizer_scales(model_int, calib_loader_a, device,
                                 args.n_calib_init)
    print(f"[calib] saw {seen} images for scale init", flush=True)

    res_int_raw = None
    calib_report = None
    calib_s = 0.0

    if args.qwt:
        print("[qwt] starting cross-seed calibration", flush=True)
        t_cal = time.time()
        seed_all(args.calib_seed)
        calib_ds_b = build_dataset(args.data_root, args.calib_seed_b,
                                   args.n_calib_qwt)
        calib_loader_b = DataLoader(calib_ds_b, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.workers,
                                    pin_memory=True)
        last_tau = (args.last_block_cos_threshold
                    if args.last_block_cos_threshold > args.cos_threshold
                    else None)
        print(f"[qwt] τ={args.cos_threshold} last_τ={last_tau} "
              f"seed_a={args.calib_seed} seed_b={args.calib_seed_b}",
              flush=True)
        calib_report = calibrate_qwt(
            model_fp=model_fp, model_sc=model_int,
            blocks_fp=list(get_blocks(model_fp)),
            blocks_sc_container=get_blocks(model_int),
            calib_loader_a=calib_loader_a, calib_loader_b=calib_loader_b,
            device=device, n_calib=args.n_calib_qwt, ridge=args.ridge,
            start_block=args.start_block, fwd_chunk=args.fwd_chunk,
            cos_threshold=args.cos_threshold,
            norm_floor=args.norm_floor,
            last_block_cos_threshold=last_tau,
            lookahead_veto=args.lookahead_veto,
            comp_factory=None,
            comp_factory_variants=None,
        )
        calib_s = time.time() - t_cal
        n_enabled = sum(1 for r in calib_report if r["enabled"])
        en_blocks = [r["block"] for r in calib_report if r["enabled"]]
        cos_admitted = [r["cos_ab"] for r in calib_report if r["enabled"]]
        cos_rejected = [r["cos_ab"] for r in calib_report if not r["enabled"]]
        print(f"[qwt] done in {calib_s:.1f}s  "
              f"admitted={n_enabled}/{len(calib_report)}  blocks={en_blocks}",
              flush=True)
        if cos_admitted:
            print(f"[qwt] cos admitted=[{min(cos_admitted):+.3f}, "
                  f"{max(cos_admitted):+.3f}]", flush=True)
        if cos_rejected:
            print(f"[qwt] cos rejected=[{min(cos_rejected):+.3f}, "
                  f"{max(cos_rejected):+.3f}]", flush=True)
    else:
        print("[eval] INT raw (no QwT)", flush=True)
        seed_all(args.eval_seed)
        res_int_raw = evaluate(model_int, eval_loader, device)
        print("[result:int_raw]", res_int_raw, flush=True)

    label = f"INT W{args.w_bit}A{args.a_bit}{' + QwT' if args.qwt else ''}"
    print(f"[eval] {label}", flush=True)
    seed_all(args.eval_seed)
    res_int = evaluate(model_int, eval_loader, device)
    print(f"[result:int]", res_int, flush=True)

    print("[eval] FP reference", flush=True)
    seed_all(args.eval_seed)
    res_fp = evaluate(model_fp, eval_loader, device)
    print("[result:fp]", res_fp, flush=True)

    out = {
        "config": {
            "quant_method": "uniform_per_tensor_act_per_channel_w_linear_only",
            "w_bit": args.w_bit, "a_bit": args.a_bit,
            "qwt": bool(args.qwt),
            "n_calib_init": args.n_calib_init,
            "n_calib_qwt": args.n_calib_qwt if args.qwt else None,
            "n_eval": len(eval_ds),
            "calib_seed": args.calib_seed,
            "calib_seed_b": args.calib_seed_b if args.qwt else None,
            "eval_seed": args.eval_seed,
            "ridge": args.ridge if args.qwt else None,
            "cos_threshold": args.cos_threshold if args.qwt else None,
            "last_block_cos_threshold":
                args.last_block_cos_threshold if args.qwt else None,
            "norm_floor": args.norm_floor if args.qwt else None,
            "lookahead_veto": bool(args.lookahead_veto) if args.qwt else None,
            "start_block": args.start_block,
        },
        "calib": ({
            "per_block": calib_report,
            "elapsed_s": calib_s,
            "n_enabled_blocks":
                sum(1 for r in calib_report if r["enabled"]),
        } if args.qwt else None),
        "results": {"fp": res_fp, "int_raw": res_int_raw, "int": res_int},
        "patch_stats": stats,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out_json}", flush=True)

    print("\n=== SUMMARY ===")
    print(f"  FP                : top1={res_fp['top1']:.4f}")
    if res_int_raw:
        print(f"  INT W{args.w_bit}A{args.a_bit}  : "
              f"top1={res_int_raw['top1']:.4f}")
    print(f"  {label:<18}: top1={res_int['top1']:.4f}")


if __name__ == "__main__":
    main()
