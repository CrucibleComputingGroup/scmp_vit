"""Evaluate DINOv2 ViT-L/14 + linear head on ImageNet-1k val, FP vs SC Q@K^T."""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from imagenet_parquet import ImageNetParquetVal
from sc_attention_patch import patch_model


def build_transform(image_size=224):
    # DINOv2 eval recipe: resize 256 (shorter side), center crop 224,
    # ImageNet mean/std
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None, log_every=10):
    model.eval()
    n = 0
    top1 = 0
    top5 = 0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        _, pred5 = logits.topk(5, dim=1)
        correct = pred5.eq(y.unsqueeze(1))
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        n += y.numel()
        if (i + 1) % log_every == 0:
            dt = time.time() - t0
            print(f"  [{i+1}] top1={top1/n:.4f} top5={top5/n:.4f} "
                  f"({n} imgs, {dt:.1f}s, {n/dt:.1f} img/s)", flush=True)
        if max_batches is not None and (i + 1) >= max_batches:
            break
    dt = time.time() - t0
    return {
        "n": n,
        "top1": top1 / n,
        "top5": top5 / n,
        "elapsed_s": dt,
        "img_per_s": n / dt if dt > 0 else 0.0,
    }


def load_model(device):
    os.environ.setdefault("XFORMERS_DISABLED", "1")  # fall back to SDPA forward
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitl14_lc",
        source="github", trust_repo=True,
    )
    model = model.to(device).eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/home/yjrcs/SC_V/data/imagenet/data")
    p.add_argument("--mode", choices=["fp", "sc"], required=True)
    p.add_argument("--sc_prec", type=int, default=8)
    p.add_argument("--sc_qk", type=int, default=1)
    p.add_argument("--sc_av", type=int, default=0)
    p.add_argument("--sc_mlp", type=int, default=0)
    p.add_argument("--mlp_skip_last_k", type=int, default=0)
    p.add_argument("--mlp_skip_first_k", type=int, default=0)
    p.add_argument("--sc_qkv_proj", type=int, default=0)
    p.add_argument("--sc_out_proj", type=int, default=0)
    p.add_argument("--sc_mlp_mode", choices=["bipolar", "unipolar"],
                   default="bipolar",
                   help="MLP SC quantization: bipolar=symmetric |max|, "
                        "unipolar=asymmetric zero-point (better for post-GELU)")
    p.add_argument("--sc_proj_mode", choices=["bipolar", "unipolar"],
                   default="bipolar",
                   help="qkv/out proj SC quant mode (default bipolar).")
    p.add_argument("--mlp_sc_spec_json", default="",
                   help="Path to JSON file with list of [block_idx, fc_name] "
                        "pairs specifying exactly which MLP linears go SC. "
                        "Overrides --mlp_skip_first_k / --mlp_skip_last_k. "
                        "Sets --sc_mlp 1 implicitly.")
    p.add_argument("--sc_ops", default="",
                   help="Comma-separated op names from "
                        "{mlp_fc1,mlp_fc2,qk,av,proj} OR a 5-element 0/1 "
                        "vector (same order, e.g. '1,0,0,1,0'). Overrides "
                        "per-op --sc_qk/--sc_av/etc flags.")
    p.add_argument("--sc_ops_per_block_json", default="",
                   help="Path to JSON file with per-op per-block vectors: "
                        '{"mlp_fc1":[0,1,...],"mlp_fc2":[...],"qk":[...],'
                        '"av":[...],"proj":[...]}. Each list has length '
                        "n_blocks (24 for DINOv2 ViT-L/14). Highest priority.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_images", type=int, default=0, help="0 = full val")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", default="")
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda")

    print(f"[env] torch {torch.__version__}, cuda {torch.version.cuda}, "
          f"device={torch.cuda.get_device_name(0)}", flush=True)

    print(f"[data] loading parquet val from {args.data_root}", flush=True)
    ds = ImageNetParquetVal(args.data_root, transform=build_transform(224))
    print(f"[data] {len(ds)} images", flush=True)
    if args.max_images and args.max_images < len(ds):
        idx = list(range(len(ds)))
        random.Random(args.seed).shuffle(idx)
        ds = Subset(ds, idx[: args.max_images])
        print(f"[data] subset -> {len(ds)} images", flush=True)

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False,
    )

    print("[model] loading DINOv2 ViT-L/14 + linear head", flush=True)
    model = load_model(device)

    if args.mode == "sc":
        mlp_spec = None
        sc_mlp_flag = bool(args.sc_mlp)
        if args.mlp_sc_spec_json:
            import json as _json
            with open(args.mlp_sc_spec_json) as _f:
                mlp_spec = [tuple(p) for p in _json.load(_f)]
            sc_mlp_flag = True
            print(f"[sc] mlp_sc_spec: {len(mlp_spec)} linears from "
                  f"{args.mlp_sc_spec_json}", flush=True)
        sc_ops_arg = args.sc_ops or None
        if sc_ops_arg:
            from sc_attention_patch import parse_sc_ops
            print(f"[sc] sc_ops={sorted(parse_sc_ops(sc_ops_arg))}", flush=True)
        per_block_arg = None
        if args.sc_ops_per_block_json:
            import json as _json
            with open(args.sc_ops_per_block_json) as _f:
                per_block_arg = _json.load(_f)
            print(f"[sc] sc_ops_per_block from {args.sc_ops_per_block_json}",
                  flush=True)
        stats = patch_model(
            model, sc_prec=args.sc_prec,
            sc_qk=bool(args.sc_qk), sc_av=bool(args.sc_av), sc_mlp=sc_mlp_flag,
            mlp_skip_last_k=args.mlp_skip_last_k,
            mlp_skip_first_k=args.mlp_skip_first_k,
            sc_qkv_proj=bool(args.sc_qkv_proj),
            sc_out_proj=bool(args.sc_out_proj),
            sc_mlp_mode=args.sc_mlp_mode,
            sc_proj_mode=args.sc_proj_mode,
            mlp_sc_spec=mlp_spec,
            sc_ops=sc_ops_arg,
            sc_ops_per_block=per_block_arg,
        )
        print(f"[sc] sc_prec={args.sc_prec} qk={bool(args.sc_qk)} "
              f"av={bool(args.sc_av)} mlp={bool(args.sc_mlp)} "
              f"mlp_mode={args.sc_mlp_mode} proj_mode={args.sc_proj_mode} "
              f"skip_first={args.mlp_skip_first_k} skip_last={args.mlp_skip_last_k} "
              f"patched={stats}",
              flush=True)

    print(f"[eval] mode={args.mode} bs={args.batch_size} "
          f"workers={args.workers}", flush=True)
    res = evaluate(model, loader, device)
    res["mode"] = args.mode
    res["sc_prec"] = args.sc_prec if args.mode == "sc" else None
    print("[result]", res, flush=True)

    if args.out_json:
        import json
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[result] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
