#!/usr/bin/env python3
"""Rank (op, block) groups by SC fragility and emit a hybrid SC/INT schedule.

The mask answers "which groups should leave SC for INT?". It ranks by the
MARGINAL knock-down loss: run the model uniform-SC at the deployment budget for
a reference loss L0, then re-run with ONE group's stream shortened by a small
step and record dL = L_g - L0. High dL = this group's output degrades fastest
per cycle removed = worst served by SC. Those go to INT.

The step is deliberately SMALL rather than a knock-to-the-floor: past the
collapse cliff dL saturates and stops being a derivative, so a floor probe
ranks by "who collapses" instead of "who is fragile here", which is the wrong
question at exactly the tight budgets where the mask matters (scmp_llm's
`measured` vs `measured_marg` split).

    python cls/build_hybrid_mask.py --stoc_len 32 --frac 0.10 \
        --out results/hybrid/mask10_t32.json

Masks at different --frac off ONE sensitivity run are strictly NESTED, so the
fraction is a genuine dose axis rather than N unrelated selections.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from imagenet_parquet import ImageNetParquetVal          # noqa: E402
from eval import build_transform, load_model, seed_all   # noqa: E402
from model.sc_vit import (                               # noqa: E402
    patch_model, set_sc_defaults, set_qk_granularity, set_group_stoclen,
)

OPS = ("qk", "av", "qkv_proj", "out_proj", "mlp_fc1", "mlp_fc2")


@torch.no_grad()
def measure(model, batches, n_blocks, baseline_sl, probe_sl):
    loss_fn = nn.CrossEntropyLoss()

    def mean_loss(gmap):
        set_group_stoclen(gmap)
        tot = n = 0.0
        for imgs, labels in batches:
            logits = model(imgs)
            tot += float(loss_fn(logits, labels)) * labels.numel()
            n += labels.numel()
        return tot / max(n, 1)

    try:
        L0 = mean_loss({})
        print(f"[sens] baseline L0={L0:.5f}  (uniform sl={baseline_sl}, "
              f"probe sl={probe_sl}, {len(batches)} batches)", flush=True)
        out = {}
        for op in OPS:
            for b in range(n_blocks):
                dl = mean_loss({(op, b): probe_sl}) - L0
                out[f"{op}:b{b}"] = dl
            print(f"[sens] {op:<10} dL range "
                  f"[{min(out[f'{op}:b{b}'] for b in range(n_blocks)):+.5f}, "
                  f"{max(out[f'{op}:b{b}'] for b in range(n_blocks)):+.5f}]",
                  flush=True)
    finally:
        set_group_stoclen(None)
    return out, L0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",
                   default="/scratch/nbleier_owned_root/nbleier_owned1/"
                           "shared_data/imagenet/data")
    p.add_argument("--stoc_len", type=int, required=True,
                   help="Deployment budget (halved cycles) the mask is for. "
                        "Fragility is budget-dependent, so a mask measured at "
                        "one budget does not automatically transfer.")
    p.add_argument("--probe_frac", type=float, default=0.25,
                   help="Marginal step: probe_sl = round(sl*(1-frac)).")
    p.add_argument("--frac", default="0.05,0.10,0.20",
                   help="Comma-separated mask fractions to emit.")
    p.add_argument("--int_bits", type=int, default=7)
    p.add_argument("--int_sym", type=int, default=0)
    p.add_argument("--qk_granularity", default="per_head",
                   choices=["per_head", "per_row"])
    p.add_argument("--n_images", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calib_seed", type=int, default=1234)
    p.add_argument("--eval_exclude_n", type=int, default=2048)
    p.add_argument("--out_dir", default="results/hybrid")
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda")
    set_sc_defaults(halve_bipolar_stoc_len=True, chunk_d=128)
    set_qk_granularity(args.qk_granularity)

    ds = ImageNetParquetVal(args.data_root, transform=build_transform(224))
    import random as _random
    excluded = set()
    if args.eval_exclude_n > 0:
        e = list(range(len(ds)))
        _random.Random(args.seed).shuffle(e)
        excluded = set(e[: args.eval_exclude_n])
    idx = [i for i in range(len(ds)) if i not in excluded]
    _random.Random(args.calib_seed).shuffle(idx)
    sel = idx[: args.n_images]
    loader = DataLoader(Subset(ds, sel), batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
    print(f"[sens] probe pool {len(sel)} imgs ({len(excluded)} eval excluded)",
          flush=True)

    model = load_model(device)
    stats = patch_model(model, sc_prec=8, sc_ops="mlp_fc1,mlp_fc2,qk,av,proj",
                        stoc_len=args.stoc_len, chunk_d=128)
    n_blocks = stats["attn"]
    print(f"[sens] patched {stats}", flush=True)

    batches = [(i.to(device, non_blocking=True), l.to(device, non_blocking=True))
               for i, l in loader]
    probe_sl = max(1, int(round(args.stoc_len * (1.0 - args.probe_frac))))
    if probe_sl >= args.stoc_len:
        probe_sl = max(1, args.stoc_len - 1)
    dl, L0 = measure(model, batches, n_blocks, args.stoc_len, probe_sl)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sens_path = out_dir / f"sens_{args.tag}.json"
    sens_path.write_text(json.dumps({
        "baseline_loss": L0, "stoc_len": args.stoc_len, "probe_stoc_len": probe_sl,
        "probe_frac": args.probe_frac, "n_images": len(sel),
        "qk_granularity": args.qk_granularity, "n_blocks": n_blocks,
        "delta_loss": dl,
    }, indent=2))
    print(f"[sens] wrote {sens_path}", flush=True)

    order = sorted(dl, key=lambda k: -dl[k])
    print("\n[sens] worst 15 groups (highest dL = most SC-fragile):")
    for i, k in enumerate(order[:15], 1):
        print(f"  {i:>3}  {k:<16} dL={dl[k]:+.5f}")

    total = len(order)
    for f in [float(x) for x in args.frac.split(",")]:
        n_int = int(round(f * total))
        chosen = set(order[:n_int])
        sched = {op: ["int" if f"{op}:b{b}" in chosen else "sc"
                      for b in range(n_blocks)] for op in OPS}
        path = out_dir / f"mask{int(round(f * 100)):02d}_{args.tag}.json"
        path.write_text(json.dumps({
            "format": "scmp_vit_hybrid_v1",
            "default": "sc",
            "int_bits": args.int_bits,
            "int_sym": bool(args.int_sym),
            "chunk_size": 128,
            "attn_scope": ("per_head" if args.qk_granularity == "per_head"
                           else "per_row"),
            "source_sensitivity": sens_path.name,
            "frac": f, "n_int_groups": n_int, "n_groups": total,
            "schedule": sched,
        }, indent=2))
        per_op = {op: sum(1 for b in sched[op] if b == "int") for op in OPS}
        print(f"[sens] mask {f:.0%}: {n_int}/{total} groups -> INT  {per_op}"
              f"  -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
