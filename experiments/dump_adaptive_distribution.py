"""Dump row-count distribution per stoc_len level for each (op, block).

Aggregates across all SCLinear forwards during a short eval pass and reports:
  - per-op fraction of rows at each sl level
  - average effective sl
  - implied savings vs max(levels)
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sc"))
sys.path.insert(0, str(REPO / "cls"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--levels", required=True,
                   help="csv e.g. '256,128,64,32'")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--enable_pruning", type=int, default=0)
    p.add_argument("--ops", default="qkv_proj,out_proj")
    p.add_argument("--t", type=int, default=9)
    p.add_argument("--T", type=int, default=10)
    args = p.parse_args()

    os.environ.setdefault("XFORMERS_DISABLED", "1")
    from imagenet_parquet import ImageNetParquetVal
    from sc_attention_patch import patch_model
    from sc_integration.sc_linear import set_vit_timestep
    from sc_integration import mp_linear as _mpl
    from sc_integration.mp_linear import AdaptiveMPConfig

    device = torch.device("cuda")
    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    ds = Subset(ImageNetParquetVal(
        "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data",
        transform=tfm), list(range(args.n)))
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=2)

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_lc",
                           source="github", trust_repo=True).to(device).eval()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    max_sl = max(levels)
    adap = AdaptiveMPConfig(stoc_len_levels=levels, alpha=args.alpha,
                            beta=args.beta,
                            enable_pruning=bool(args.enable_pruning))
    ops = [x.strip() for x in args.ops.split(",") if x.strip()]
    spec = {op: {"adaptive": adap} for op in ops}
    patch_model(model, sc_prec=8, sc_qk=True, sc_av=False,
                sc_qkv_proj=("qkv_proj" in ops),
                sc_out_proj=("out_proj" in ops),
                linear_mp_spec=spec)

    # Spy on adaptive classifier
    captured = defaultdict(lambda: defaultdict(int))   # op -> {sl: total_rows}
    orig = _mpl.adaptive_classify_rows
    def spy(metric, t, T, cfg, operator=None):
        a = orig(metric, t, T, cfg, operator=operator)
        for sl, rows in a.level_row_indices.items():
            captured[operator or "unknown"][int(sl)] += int(rows.numel())
        return a
    _mpl.adaptive_classify_rows = spy
    from sc_integration import sc_linear as _scl
    _scl.classify_input_rows_adaptive = _mpl.classify_input_rows_adaptive

    set_vit_timestep(args.t, args.T)

    with torch.no_grad():
        for x, y in loader:
            _ = model(x.to(device))
            break

    out_w = {"qkv_proj": 3072, "out_proj": 1024}
    print(f"\nlevels={levels}  α={args.alpha}  β={args.beta}  "
          f"t={args.t}/T={args.T}  threshold={min(args.alpha * args.t / max(args.T-1,1) + args.beta, 0.95):.3f}\n")
    print(f"{'op':<12s}  {'level':>6s}  {'rows':>10s}  {'frac%':>8s}")
    tot_base = 0.0; tot_actual = 0.0
    for op, dist in captured.items():
        tot = sum(dist.values())
        print(f"  -- {op}  (total rows classified: {tot:,}) --")
        for sl in sorted(dist.keys(), reverse=True):
            c = dist[sl]
            print(f"  {op:<12s}  sl={sl:<4d}  {c:>10,}  {100*c/tot:7.2f}%")
        avg_sl = sum(sl*c for sl,c in dist.items()) / max(tot, 1)
        save = 1 - avg_sl / max_sl
        print(f"  {op:<12s}  {'avg_sl':>6s} = {avg_sl:>6.1f}  "
              f"(savings vs sl={max_sl}: {100*save:.2f}%)")
        print()
        of = out_w.get(op, 1024)
        tot_base += tot * of * max_sl
        tot_actual += sum(sl*c for sl,c in dist.items()) * of
    print(f"OVERALL  savings vs sl={max_sl}: {100*(1-tot_actual/tot_base):.2f}%")
    if max_sl < 256:
        # Also vs sl=256 baseline
        base_256 = tot_base * 256 / max_sl
        print(f"OVERALL  savings vs sl=256:  {100*(1-tot_actual/base_256):.2f}%")


if __name__ == "__main__":
    main()
