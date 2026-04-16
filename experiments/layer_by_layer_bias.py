#!/usr/bin/env python3
"""Compare SC vs FP output at every layer to detect systematic bias.

Runs the same input through both FP and SC models, hooks each block's
output, and checks: is SC consistently smaller/larger than FP?

If SC has a systematic attenuation (SC < FP), we'd see negative mean
bias at every layer. If it's random noise, the sign would flip.
"""
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sc"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XFORMERS_DISABLED", "1")

import argparse, random, json
import torch
import torch.nn as nn
torch.set_grad_enabled(False)

from torch.utils.data import DataLoader, Subset
from eval import load_model, build_transform
from imagenet_parquet import ImageNetParquetVal
from sc_attention_patch import patch_model, _CFG_CACHE


def collect_block_outputs(model, x, device):
    """Run x through model, capture output of each transformer block + head."""
    outputs = {}

    hooks = []

    # Hook each block
    blocks = None
    # DINOv2 structure: model.backbone.blocks[i] or model.blocks[i]
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'blocks'):
        blocks = model.backbone.blocks
        norm = model.backbone.norm
        head = model.linear_head if hasattr(model, 'linear_head') else model.head
        cls_token_fn = lambda out: out[:, 0]  # CLS token
    else:
        # Try direct DINOv2 LC structure
        blocks = model.backbone.blocks if hasattr(model, 'backbone') else None
        if blocks is None:
            # hub model structure
            for name, mod in model.named_modules():
                if 'blocks' in name and isinstance(mod, nn.ModuleList):
                    blocks = mod
                    break

    if blocks is None:
        print("Could not find blocks in model, trying alternative structure")
        # DINOv2 LC: model.backbone.blocks, model.backbone.norm, model.linear_head
        print([n for n, _ in model.named_children()])
        return {}

    def make_hook(name):
        def hook_fn(module, input, output):
            outputs[name] = output.detach().clone()
        return hook_fn

    for i, block in enumerate(blocks):
        hooks.append(block.register_forward_hook(make_hook(f"block_{i:02d}")))

    # Also hook the norm and head
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'norm'):
        hooks.append(model.backbone.norm.register_forward_hook(make_hook("norm")))

    # Run forward
    with torch.no_grad():
        logits = model(x.to(device))
    outputs["logits"] = logits.detach().clone()

    for h in hooks:
        h.remove()

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data")
    parser.add_argument("--n_images", type=int, default=32,
                        help="Number of images to average over")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sc_prec", type=int, default=8)
    parser.add_argument("--sc_config", type=str, default="full_attn",
                        choices=["qk_only", "qk_av", "full_attn", "everything"])
    args = parser.parse_args()

    device = "cuda"

    # Load data
    ds = ImageNetParquetVal(args.data_root, transform=build_transform(224))
    idx = list(range(len(ds)))
    random.Random(0).shuffle(idx)
    ds = Subset(ds, idx[:args.n_images])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    # SC config
    sc_kwargs = {
        "qk_only": dict(sc_qk=True, sc_av=False, sc_qkv_proj=False, sc_out_proj=False, sc_mlp=False),
        "qk_av": dict(sc_qk=True, sc_av=True, sc_qkv_proj=False, sc_out_proj=False, sc_mlp=False),
        "full_attn": dict(sc_qk=True, sc_av=True, sc_qkv_proj=True, sc_out_proj=True, sc_mlp=False),
        "everything": dict(sc_qk=True, sc_av=True, sc_qkv_proj=True, sc_out_proj=True, sc_mlp=True),
    }[args.sc_config]

    print(f"Config: {args.sc_config}, sc_prec={args.sc_prec}, n_images={args.n_images}")
    print()

    # Load FP model
    model_fp = load_model(device)
    model_fp.eval()

    # Load SC model
    model_sc = load_model(device)
    model_sc.eval()
    _CFG_CACHE.clear()
    patch_model(model_sc, sc_prec=args.sc_prec, **sc_kwargs)

    # Accumulate per-layer statistics
    stats = {}  # layer_name -> {sum_fp, sum_sc, sum_diff, sum_abs_diff, ...}

    n_processed = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        B = images.shape[0]

        out_fp = collect_block_outputs(model_fp, images, device)
        out_sc = collect_block_outputs(model_sc, images, device)

        for name in sorted(out_fp.keys()):
            if name not in out_sc:
                continue
            fp_out = out_fp[name].float()
            sc_out = out_sc[name].float()

            diff = sc_out - fp_out  # positive = SC > FP, negative = SC < FP

            if name not in stats:
                stats[name] = {
                    "sum_fp_mean": 0.0,
                    "sum_sc_mean": 0.0,
                    "sum_fp_norm": 0.0,
                    "sum_sc_norm": 0.0,
                    "sum_diff_mean": 0.0,       # signed mean (bias direction)
                    "sum_abs_diff_mean": 0.0,    # magnitude
                    "sum_fp_abs_mean": 0.0,      # for relative error
                    "sum_ratio": 0.0,            # SC_norm / FP_norm
                    "n": 0,
                }

            s = stats[name]
            s["sum_fp_mean"] += fp_out.mean().item() * B
            s["sum_sc_mean"] += sc_out.mean().item() * B
            s["sum_fp_norm"] += fp_out.norm().item() * B
            s["sum_sc_norm"] += sc_out.norm().item() * B
            s["sum_diff_mean"] += diff.mean().item() * B
            s["sum_abs_diff_mean"] += diff.abs().mean().item() * B
            s["sum_fp_abs_mean"] += fp_out.abs().mean().item() * B
            fp_norm = fp_out.norm().item()
            sc_norm = sc_out.norm().item()
            if fp_norm > 1e-8:
                s["sum_ratio"] += (sc_norm / fp_norm) * B
            s["n"] += B

        n_processed += B
        print(f"  processed {n_processed}/{args.n_images} images", flush=True)

    # Print results
    print()
    print("=" * 100)
    print(f"Layer-by-layer SC vs FP comparison  (config={args.sc_config}, prec={args.sc_prec})")
    print("=" * 100)
    print(f"{'Layer':<14} {'FP_mean':>10} {'SC_mean':>10} {'bias':>10} "
          f"{'|bias|/|FP|':>12} {'‖SC‖/‖FP‖':>10} {'SC<FP?':>8}")
    print("-" * 100)

    all_sc_smaller = True
    bias_directions = []

    for name in sorted(stats.keys()):
        s = stats[name]
        n = s["n"]
        fp_mean = s["sum_fp_mean"] / n
        sc_mean = s["sum_sc_mean"] / n
        bias = s["sum_diff_mean"] / n
        abs_diff = s["sum_abs_diff_mean"] / n
        fp_abs = s["sum_fp_abs_mean"] / n
        ratio = s["sum_ratio"] / n

        rel_bias = bias / fp_abs if fp_abs > 1e-8 else 0.0
        sc_smaller = ratio < 1.0

        if name.startswith("block") or name in ("norm", "logits"):
            bias_directions.append(("smaller" if sc_smaller else "LARGER", name, ratio))
            if not sc_smaller:
                all_sc_smaller = False

        marker = "  ←" if sc_smaller else "  ↑"
        print(f"{name:<14} {fp_mean:>10.4f} {sc_mean:>10.4f} {bias:>+10.6f} "
              f"{rel_bias:>+12.6f} {ratio:>10.6f} {'yes' + marker if sc_smaller else 'NO' + marker:>8}")

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    n_smaller = sum(1 for d, _, _ in bias_directions if d == "smaller")
    n_total = len(bias_directions)
    print(f"  Layers where ‖SC‖ < ‖FP‖: {n_smaller}/{n_total}")

    if n_smaller == n_total:
        print("  >>> SYSTEMATIC ATTENUATION: SC output is smaller than FP at EVERY layer")
        print("  >>> This indicates a multiplicative bias (scale error) in the SC implementation")
    elif n_smaller == 0:
        print("  >>> SC output is LARGER than FP at every layer (systematic amplification)")
    else:
        print(f"  >>> Mixed: SC is sometimes smaller, sometimes larger (random noise, no systematic bias)")

    # Show the norm ratio progression
    print()
    print("  Norm ratio ‖SC‖/‖FP‖ progression through layers:")
    for direction, name, ratio in bias_directions:
        bar_len = int(abs(ratio - 1.0) * 200)
        if ratio < 1.0:
            bar = "█" * min(bar_len, 40)
            print(f"    {name:<14} {ratio:.6f}  {'←' + bar}")
        else:
            bar = "█" * min(bar_len, 40)
            print(f"    {name:<14} {ratio:.6f}  {'→' + bar}")


if __name__ == "__main__":
    main()
