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

# cls/ holds eval/imagenet_parquet/sc_attention_patch; vit_sc/ root holds
# shared sc_integration/ and sc/ packages.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
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


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_csv_ops(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _build_attn_mp_spec(args) -> dict:
    """Assemble {"qk": MPConfig|AdaptiveMPConfig, "av": ...} from CLI.

    Priority: if ``--adaptive_mp 1`` and ``qk`` / ``av`` listed in
    ``--adaptive_mp_ops``, that op gets an ``AdaptiveMPConfig`` built from
    the shared ``--adaptive_mp_*`` flags. Otherwise falls back to the
    legacy static ``--qk_mp_*`` / ``--av_mp_*`` flags.
    """
    from sc_integration.mp_linear import MPConfig, AdaptiveMPConfig
    spec = {}
    adaptive_ops = (set(_parse_csv_ops(args.adaptive_mp_ops))
                    if bool(args.adaptive_mp) else set())
    adaptive_levels = (_parse_csv_ints(args.adaptive_mp_levels)
                       if bool(args.adaptive_mp) and args.adaptive_mp_levels
                       else [])
    adaptive_cfg = (
        AdaptiveMPConfig(
            stoc_len_levels=adaptive_levels,
            alpha=args.adaptive_mp_alpha,
            beta=args.adaptive_mp_beta,
            enable_pruning=bool(args.adaptive_mp_enable_pruning))
        if adaptive_levels else None)

    # QK
    if "qk" in adaptive_ops and adaptive_cfg is not None:
        spec["qk"] = adaptive_cfg
    elif args.qk_mp_levels:
        spec["qk"] = MPConfig(
            stoc_len_levels=_parse_csv_ints(args.qk_mp_levels),
            level_fractions=(_parse_csv_floats(args.qk_mp_fractions)
                             if args.qk_mp_fractions else None),
        )
    # AV
    if "av" in adaptive_ops and adaptive_cfg is not None:
        spec["av"] = adaptive_cfg
    elif args.av_mp_levels:
        spec["av"] = MPConfig(
            stoc_len_levels=_parse_csv_ints(args.av_mp_levels),
            level_fractions=(_parse_csv_floats(args.av_mp_fractions)
                             if args.av_mp_fractions else None),
        )
    return spec


def _build_linear_mp_spec(args) -> dict:
    """Assemble {op_name: {"fixed": MPConfig, "adaptive": AdaptiveMPConfig,
    "range": RangeMPConfig, ...}} from CLI flags. Returns {} if nothing
    is requested.
    """
    from sc_integration.mp_linear import (
        MPConfig, AdaptiveMPConfig, RangeMPConfig,
    )

    fixed_levels = _parse_csv_ints(args.mp_levels) if args.mp_levels else []
    fixed_fracs = _parse_csv_floats(args.mp_fractions) if args.mp_fractions else None
    fixed_ops = _parse_csv_ops(args.mp_ops)
    range_on = bool(args.range_mp)
    range_levels = _parse_csv_ints(args.range_mp_levels) if range_on else []
    range_ops = _parse_csv_ops(args.range_mp_ops) or (fixed_ops if range_on else [])
    adaptive_on = bool(args.adaptive_mp)
    adaptive_levels = (_parse_csv_ints(args.adaptive_mp_levels)
                       if adaptive_on else [])
    _raw_adaptive_ops = (_parse_csv_ops(args.adaptive_mp_ops)
                         or (fixed_ops if adaptive_on else []))
    # qk/av are routed through _build_attn_mp_spec, not here.
    adaptive_ops = [op for op in _raw_adaptive_ops if op not in ("qk", "av")]

    mp_fixed = (MPConfig(stoc_len_levels=fixed_levels,
                         level_fractions=fixed_fracs)
                if fixed_levels else None)
    mp_range = (RangeMPConfig(stoc_len_levels=range_levels,
                              base_threshold=args.range_mp_threshold)
                if range_on else None)
    mp_adaptive = (AdaptiveMPConfig(
                      stoc_len_levels=adaptive_levels,
                      alpha=args.adaptive_mp_alpha,
                      beta=args.adaptive_mp_beta,
                      enable_pruning=bool(args.adaptive_mp_enable_pruning))
                   if adaptive_levels else None)

    if not mp_fixed and not mp_range and not mp_adaptive:
        return {}

    spec: dict = {}
    if mp_fixed:
        for op in fixed_ops:
            spec.setdefault(op, {})["fixed"] = mp_fixed
    if mp_adaptive:
        for op in adaptive_ops:
            spec.setdefault(op, {})["adaptive"] = mp_adaptive
    if mp_range:
        for op in range_ops:
            entry = spec.setdefault(op, {})
            entry["range"] = mp_range
            entry["range_group_size"] = args.range_mp_group_size
    return spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data",
                   help="Path to ImageNet-1k parquet shards.")
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
    # ---- Mixed precision (ported from ../scmp_llm) ----
    p.add_argument("--mp_levels", default="",
                   help="Comma-separated stoc_len levels (descending), e.g. "
                        "'256,128'. Enables fixed per-input-row MP on --mp_ops.")
    p.add_argument("--mp_fractions", default="",
                   help="Comma-separated fractions per level (sums to 1). "
                        "Empty = equal split.")
    p.add_argument("--mp_ops", default="",
                   help="Comma-separated linear ops that get fixed MP. "
                        "Valid: mlp_fc1,mlp_fc2,qkv_proj,out_proj,proj.")
    p.add_argument("--range_mp", type=int, default=0,
                   help="Enable range-based per-weight-group MP on --range_mp_ops.")
    p.add_argument("--range_mp_levels", default="256,128",
                   help="Comma-separated stoc_len levels (descending) for range MP.")
    p.add_argument("--range_mp_threshold", type=float, default=0.3,
                   help="Normalized range threshold. Higher = more groups "
                        "get lower precision.")
    p.add_argument("--range_mp_ops", default="",
                   help="Comma-separated linear ops that get range MP. "
                        "Empty ⇒ same as --mp_ops.")
    p.add_argument("--range_mp_group_size", type=int, default=0,
                   help="Output-rows per range-MP group. 0 = per-tensor.")
    # adaptive (timestep-aware) mixed precision on linear ops
    p.add_argument("--adaptive_mp", type=int, default=0,
                   help="Enable adaptive per-input-row MP (α·progress + β). "
                        "ViT has no native timestep, so (t, T) are driven via "
                        "--vit_timestep / --vit_total_timesteps.")
    p.add_argument("--adaptive_mp_levels", default="",
                   help="Comma-separated stoc_len levels (descending), e.g. "
                        "'256,64,0'. The '0' level enables pruning (requires "
                        "--adaptive_mp_enable_pruning=1).")
    p.add_argument("--adaptive_mp_alpha", type=float, default=0.3,
                   help="Global α for threshold = α·progress + β.")
    p.add_argument("--adaptive_mp_beta", type=float, default=0.05,
                   help="Global β (base offset).")
    p.add_argument("--adaptive_mp_enable_pruning", type=int, default=1,
                   help="Allow stoc_len=0 (row pruning).")
    p.add_argument("--adaptive_mp_ops", default="",
                   help="Comma-separated linear ops that get adaptive MP. "
                        "Empty ⇒ same as --mp_ops. Valid: "
                        "mlp_fc1,mlp_fc2,qkv_proj,out_proj,proj.")
    p.add_argument("--vit_timestep", type=int, default=0,
                   help="External timestep t for adaptive MP (ViT has no "
                        "native timestep). Reports at this (t, T) pair.")
    p.add_argument("--vit_total_timesteps", type=int, default=1,
                   help="External total timesteps T. (0,1) ⇒ progress=0 "
                        "(most conservative, equivalent to threshold=β).")
    # per-head / per-row MP on QK and AV (scmp_llm-compatible)
    p.add_argument("--qk_mp_levels", default="",
                   help="Per-head fixed MP on QK (metric = per-head |Q|.amax). "
                        "Comma-separated stoc_len levels, e.g. '256,128'.")
    p.add_argument("--qk_mp_fractions", default="",
                   help="Comma-separated head fractions per level.")
    p.add_argument("--av_mp_levels", default="",
                   help="Per-attn-row fixed MP on AV (metric = attn_row.amax). "
                        "Comma-separated stoc_len levels.")
    p.add_argument("--av_mp_fractions", default="",
                   help="Comma-separated row fractions per level.")
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

        linear_mp_spec = _build_linear_mp_spec(args)
        if linear_mp_spec:
            print(f"[sc] linear_mp_spec: {sorted(linear_mp_spec)}", flush=True)
        attn_mp_spec = _build_attn_mp_spec(args)
        if attn_mp_spec:
            print(f"[sc] attn_mp_spec: {sorted(attn_mp_spec)}", flush=True)
        if args.adaptive_mp:
            from sc_integration.sc_linear import set_vit_timestep
            set_vit_timestep(args.vit_timestep, args.vit_total_timesteps)
            prog = (args.vit_timestep /
                    max(args.vit_total_timesteps - 1, 1))
            thr = min(args.adaptive_mp_alpha * prog + args.adaptive_mp_beta,
                      0.95)
            print(f"[sc] adaptive_mp t={args.vit_timestep}/T="
                  f"{args.vit_total_timesteps} progress={prog:.3f} "
                  f"threshold={thr:.3f} "
                  f"(α={args.adaptive_mp_alpha}, β={args.adaptive_mp_beta})",
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
            linear_mp_spec=linear_mp_spec or None,
            attn_mp_spec=attn_mp_spec or None,
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
