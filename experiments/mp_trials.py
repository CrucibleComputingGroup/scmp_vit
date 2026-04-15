"""Mixed-precision trial matrix (ported-MP port smoke sweep).

Compares MP regimes on the canonical MLP-SC-first-8 regime:
  - uniform SC (stoc_len=256)
  - fixed MPConfig at a few (levels, fractions)
  - range RangeMPConfig at a few (levels, threshold)
  - fixed + range combined

Uses the noise-surrogate backend for speed (already calibrated in repo).
Reports top-1/top-5 and a compute-savings proxy computed from the average
effective stoc_len per MP sub-matmul (assumes rows/groups are equal-size).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import torch
from torch.utils.data import DataLoader, Subset

from eval import build_transform, load_model, seed_all
from imagenet_parquet import ImageNetParquetVal
from sc_attention_patch import (
    patch_model, SC_OP_NAMES, set_noise_model,
)
from sc_integration.mp_linear import MPConfig, RangeMPConfig

DATA_ROOT = "./data/imagenet/data"
N_BLOCKS = 24
SC_BLOCKS_N = 8  # first-N MLP-SC regime (project memory: N ≤ 8)


def mlp_first_n_schedule(n: int) -> dict:
    """sc_ops_per_block dict that SCs mlp_fc1 + mlp_fc2 in the first n blocks."""
    vec = [1] * n + [0] * (N_BLOCKS - n)
    return {
        "mlp_fc1": vec, "mlp_fc2": vec,
        "qk": [0] * N_BLOCKS, "av": [0] * N_BLOCKS,
        "proj": [0] * N_BLOCKS,
    }


def avg_effective_stoc_len(levels: list[int], fractions: list[float]) -> float:
    return sum(l * f for l, f in zip(levels, fractions))


# Each entry defines a (label, linear_mp_spec-builder, expected_avg_sl). The
# linear_mp_spec-builder runs per-config because MPConfig objects are mutable.
def uniform():
    return None, 256.0


def fixed(levels, fractions, ops=("mlp_fc1", "mlp_fc2")):
    def build():
        cfg = MPConfig(stoc_len_levels=list(levels),
                       level_fractions=list(fractions))
        return {op: {"fixed": cfg} for op in ops}
    avg_sl = avg_effective_stoc_len(levels, fractions)
    return build, avg_sl


def rng_mp(levels, threshold, ops=("mlp_fc1", "mlp_fc2"), group_size=0):
    def build():
        cfg = RangeMPConfig(stoc_len_levels=list(levels),
                            base_threshold=threshold)
        return {op: {"range": cfg, "range_group_size": group_size}
                for op in ops}
    # Approximate: range MP is data-dependent. Just use geom mean as a proxy.
    return build, sum(levels) / len(levels)


def combined(fixed_levels, fixed_fracs, range_levels, range_threshold,
             ops=("mlp_fc1", "mlp_fc2"), group_size=0):
    def build():
        fc = MPConfig(stoc_len_levels=list(fixed_levels),
                      level_fractions=list(fixed_fracs))
        rc = RangeMPConfig(stoc_len_levels=list(range_levels),
                           base_threshold=range_threshold)
        return {op: {"fixed": fc, "range": rc, "range_group_size": group_size}
                for op in ops}
    return build, avg_effective_stoc_len(fixed_levels, fixed_fracs)


TRIALS = [
    # (name, description, builder_or_None, avg_sl_proxy)
    ("fp",                 "FP32 baseline (no SC)",                  "FP",  0.0),
    ("sc_uniform_256",     "uniform stoc_len=256",                   None,  256.0),
    ("sc_uniform_128",     "uniform stoc_len=128",                   *fixed([128], [1.0])),
    ("sc_uniform_64",      "uniform stoc_len=64",                    *fixed([64], [1.0])),
    ("fixed_256_128",      "fixed MP [256,128] @ 50/50",             *fixed([256, 128], [0.5, 0.5])),
    ("fixed_256_64",       "fixed MP [256,64] @ 50/50",              *fixed([256, 64], [0.5, 0.5])),
    ("fixed_128_64",       "fixed MP [128,64] @ 50/50",              *fixed([128, 64], [0.5, 0.5])),
    ("fixed_4lvl",         "fixed MP [256,128,64,32] @ 25/25/25/25", *fixed([256, 128, 64, 32], [0.25]*4)),
    ("range_g128_t03",     "range MP [256,128] g=128 t=0.3",         *rng_mp([256, 128], 0.3, group_size=128)),
    ("range_g128_t05",     "range MP [256,128] g=128 t=0.5",         *rng_mp([256, 128], 0.5, group_size=128)),
    ("combined_g128",      "fixed [256,128] + range [256,128] g=128", *combined([256, 128], [0.5, 0.5], [256, 128], 0.3, group_size=128)),
]


@torch.no_grad()
def eval_one(model, loader, device):
    top1 = top5 = n = 0
    t0 = time.time()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        _, p5 = logits.topk(5, 1)
        correct = p5.eq(y.unsqueeze(1))
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(1).sum().item()
        n += y.numel()
    dt = time.time() - t0
    return {"n": n, "top1": top1 / n, "top5": top5 / n,
            "elapsed_s": dt, "img_per_s": n / dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sc_prec", type=int, default=8)
    ap.add_argument("--sc_blocks_n", type=int, default=SC_BLOCKS_N)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--noise_model", type=int, default=1,
                    help="1 = calibrated SC noise surrogate (fast). 0 = real SC kernels.")
    ap.add_argument("--trials", nargs="*", default=[t[0] for t in TRIALS])
    ap.add_argument("--out_json", default="results/mp_trials.json")
    args = ap.parse_args()

    os.environ.setdefault("XFORMERS_DISABLED", "1")
    seed_all(args.seed)
    device = torch.device("cuda")
    set_noise_model(bool(args.noise_model))
    print(f"[cfg] noise_model={bool(args.noise_model)} n_images={args.n_images} "
          f"sc_blocks_n={args.sc_blocks_n} sc_prec={args.sc_prec}", flush=True)

    ds_full = ImageNetParquetVal(DATA_ROOT, transform=build_transform(224))
    idx = list(range(len(ds_full)))
    random.Random(args.seed).shuffle(idx)
    ds = Subset(ds_full, idx[:args.n_images])
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=4, pin_memory=True)

    schedule = mlp_first_n_schedule(args.sc_blocks_n)
    results = {"config": vars(args), "trials": {}}
    label_to_entry = {t[0]: t for t in TRIALS}

    for name in args.trials:
        if name not in label_to_entry:
            print(f"[warn] skipping unknown trial: {name}")
            continue
        _, desc, builder, avg_sl = label_to_entry[name]
        print(f"\n=== {name:<22} {desc}", flush=True)
        m = load_model(device)

        if builder == "FP":
            # no patching
            stats = None
        else:
            lin_mp = builder() if callable(builder) else None
            stats = patch_model(
                m, sc_prec=args.sc_prec,
                sc_mlp_mode="bipolar",
                sc_ops_per_block=schedule,
                linear_mp_spec=lin_mp,
            )
            active = {k: v for k, v in stats.items() if v}
            print(f"  patched {active} | avg_sl_proxy={avg_sl:.1f}", flush=True)

        res = eval_one(m, loader, device)
        print(f"  top1={res['top1']:.4f} top5={res['top5']:.4f}  "
              f"{res['img_per_s']:.1f} img/s ({res['elapsed_s']:.0f}s)",
              flush=True)
        results["trials"][name] = {
            "desc": desc, "avg_sl_proxy": avg_sl,
            "n_patched": stats, **res,
        }
        del m
        torch.cuda.empty_cache()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] wrote {args.out_json}", flush=True)

    # Summary table
    print(f"\n=== SUMMARY (N={args.n_images} imgs, first-{args.sc_blocks_n} MLP-SC) ===")
    print(f"{'trial':<22} {'avg_sl':>7} {'top1':>7} {'top5':>7} {'img/s':>7}")
    for name in args.trials:
        if name not in results["trials"]:
            continue
        r = results["trials"][name]
        print(f"{name:<22} {r['avg_sl_proxy']:>7.1f} "
              f"{r['top1']:>7.4f} {r['top5']:>7.4f} {r['img_per_s']:>7.1f}")


if __name__ == "__main__":
    main()
