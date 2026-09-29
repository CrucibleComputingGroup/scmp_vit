#!/usr/bin/env python3
"""Rank (op, block) groups by MEASURED SC->INT swap gain, and emit a mask.

WHY A SECOND RANKER. `build_hybrid_mask.py` ranks by SC FRAGILITY: it shortens
one group's stream by a small step and records dL. High dL = "degrades fastest
per cycle removed". That is a PROXY for the decision the mask actually makes,
which is "does this group do better in INT than in SC?". The two come apart
whenever a group is fragile to losing SC cycles but INT cannot help it either --
e.g. a group whose problem is dynamic range rather than quantization step, where
a uniform fixed-point grid is no better than a short stochastic stream.

This script measures the decision quantity directly:

    dL_swap(g) = L(g in INT) - L(uniform SC)

and ranks ASCENDING, so the most NEGATIVE (biggest loss reduction from the swap)
goes to INT first. Groups whose swap makes things worse are never selected,
which the fragility ranker cannot express at all.

MOTIVATION FROM THE DATA. At eff-96 the fragility mask is inert: on the uniform
arm it is worth -0.02 pt, against +2.95 at eff-64 and +0.88 at eff-128. It is
not saturation (eff-128 has less headroom and still gains), not ranking noise
(t48's top-14 sit 2.28 sd above its own tail, the best of the four budgets), not
composition (t48 and t64 masks are both mlp_fc2-dominated at 5.33% of MACs), and
not an execution fault (the trace's absent groups match the mask exactly). What
is left is WHICH BLOCKS get picked -- t48 and t64 share ZERO of their 14 groups.

scmp_llm tested this same idea (`int_swap`, v18) on LLMs and it LOST to the
fragility proxy (4B 12.781 vs 12.572). Two reasons that does not settle it here:
their ranking came from a low-window sensitivity run whose robust version was
never launched, and no ViT has ever been measured.

SCOPE: linear ops only. qk/av are 4% of MACs but they are not `SCLinear`
instances, so the hot-swap below does not reach them. To keep the comparison
one-variable, build the fragility control with `--linear-only` too rather than
comparing a linear-only mask against one that can also spend slots on attention.

    python cls/build_intswap_mask.py --stoc_len 48 --int_bits 7 --tag t48
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
    patch_model, set_sc_defaults, set_qk_granularity, SCLinear,
)
from model.int_vit import IntConfig, IntLinear           # noqa: E402

LINEAR_OPS = ("qkv_proj", "out_proj", "mlp_fc1", "mlp_fc2")


def index_sc_linears(model):
    """{(op, block): (parent_module, attr_name, sclinear)} for every SCLinear.

    Both SCLinear and IntLinear wrap by REFERENCE (`self.weight = linear.weight`),
    so an SCLinear duck-types as IntLinear's `linear` argument and the swap needs
    no weight copy and no re-patch of the model.
    """
    out = {}
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, SCLinear) and child.operator is not None \
                    and child.block_idx is not None:
                out[(child.operator, int(child.block_idx))] = (parent, name, child)
    return out


@torch.no_grad()
def measure(model, batches, index, int_cfg):
    loss_fn = nn.CrossEntropyLoss()

    def mean_loss():
        tot = n = 0.0
        for imgs, labels in batches:
            logits = model(imgs)
            tot += float(loss_fn(logits, labels)) * labels.numel()
            n += labels.numel()
        return tot / max(n, 1)

    L0 = mean_loss()
    print(f"[swap] baseline L0={L0:.5f}  (uniform SC, {len(batches)} batches, "
          f"{len(index)} linear groups)", flush=True)

    out = {}
    for op in LINEAR_OPS:
        keys = sorted(k for k in index if k[0] == op)
        for key in keys:
            parent, name, sc = index[key]
            try:
                setattr(parent, name, IntLinear(sc, int_cfg, key[0], key[1]))
                out[f"{op}:b{key[1]}"] = mean_loss() - L0
            finally:
                setattr(parent, name, sc)      # restore even on failure
        if keys:
            vals = [out[f"{op}:b{k[1]}"] for k in keys]
            print(f"[swap] {op:<10} dL_swap [{min(vals):+.5f}, {max(vals):+.5f}]  "
                  f"helped {sum(1 for v in vals if v < 0)}/{len(vals)}", flush=True)
    return out, L0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",
                   default="/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data")
    p.add_argument("--stoc_len", type=int, required=True)
    p.add_argument("--int_bits", type=int, default=7)
    p.add_argument("--int_sym", type=int, default=0)
    p.add_argument("--frac", default="0.05,0.10,0.20")
    p.add_argument("--qk_granularity", default="per_row")
    p.add_argument("--n_images", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_exclude_n", type=int, default=10000)
    p.add_argument("--out_dir", default="results/hybrid")
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda")
    ds = ImageNetParquetVal(args.data_root, transform=build_transform(224))
    # Probe images come from OUTSIDE the reported eval subset, same rule the
    # calibrator uses: the mask is a fitted object and must not see the test set.
    keep = list(range(len(ds)))
    import random as _r
    _r.Random(args.seed).shuffle(keep)
    sel = keep[args.eval_exclude_n: args.eval_exclude_n + args.n_images]
    loader = DataLoader(Subset(ds, sel), batch_size=args.batch_size,
                        shuffle=False, num_workers=8)
    batches = [(i.to(device, non_blocking=True), l.to(device)) for i, l in loader]

    model = load_model(device)
    # sc_prec is a patch_model argument, NOT a global default: set_sc_defaults
    # only owns chunk_d / halve_bipolar_stoc_len / chunk_d_warn_width.
    set_sc_defaults(halve_bipolar_stoc_len=True, chunk_d=128)
    set_qk_granularity(args.qk_granularity)
    patch_model(model, sc_prec=8, stoc_len=args.stoc_len, chunk_d=128,
                sc_ops="mlp_fc1,mlp_fc2,qk,av,proj")
    model.eval()

    int_cfg = IntConfig(args.int_bits, sym=bool(args.int_sym), chunk_size=128,
                        quantize_attention=True, attn_scope=args.qk_granularity)
    index = index_sc_linears(model)
    dl, L0 = measure(model, batches, index, int_cfg)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sens_path = out_dir / f"sensswap_{args.tag}.json"
    sens_path.write_text(json.dumps({
        "baseline_loss": L0, "stoc_len": args.stoc_len,
        "int_bits": args.int_bits, "int_sym": bool(args.int_sym),
        "n_images": args.n_images, "qk_granularity": args.qk_granularity,
        "ranker": "int_swap", "scope": "linear_ops_only",
        "delta_loss_swap": dl,
    }, indent=2))
    print(f"[swap] wrote {sens_path}", flush=True)

    # ASCENDING: the most negative swap delta is the biggest win from INT.
    order = sorted(dl, key=lambda k: dl[k])
    n_groups = len(dl)
    for f in [float(x) for x in args.frac.split(",")]:
        k = max(1, round(f * n_groups))
        chosen = order[:k]
        sched = {op: ["sc"] * 24 for op in ("qk", "av", *LINEAR_OPS)}
        for key in chosen:
            op, b = key.split(":b")
            sched[op][int(b)] = "int"
        path = out_dir / f"maskswap{int(round(f*100)):02d}_{args.tag}.json"
        path.write_text(json.dumps({
            "format": "scmp_vit_hybrid_v1", "default": "sc",
            "int_bits": args.int_bits, "int_sym": bool(args.int_sym),
            "chunk_size": 128, "attn_scope": args.qk_granularity,
            "source_sensitivity": sens_path.name, "ranker": "int_swap",
            "frac": f, "n_int_groups": k, "n_groups": n_groups,
            "schedule": sched,
        }, indent=2))
        helped = sum(1 for c in chosen if dl[c] < 0)
        print(f"[swap] mask {100*f:.0f}%: {k}/{n_groups} groups -> INT "
              f"({helped} of them measured a GAIN)  -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
