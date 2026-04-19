#!/usr/bin/env python
"""Train the adaptive MP allocator on REAL sensitivity data, then evaluate
the learned schedule on DINOv2 ViT-L/14 + ImageNet. Side-by-side with the
hand-crafted sensitivity-proportional baseline at the same FLOP-weighted
target stoc_len.

Output: results/adaptive_mp_dinov2_eval/metrics.json with per-strategy top-1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "experiments"))

from sc_integration.adaptive_mp_allocator import (  # noqa: E402
    AdaptiveMPAllocator,
    load_sensitivity_prior,
    DEFAULT_OPS,
    DEFAULT_LEVELS,
)
from mp_uniform_vs_fine import (  # noqa: E402
    build_active_ops, build_eval_loader, load_sensitivity,
    alloc_proportional, alloc_uniform, compute_main_sl, run_one,
    _OP_MACS,
)
from eval import seed_all  # noqa: E402
from sc_attention_patch import set_noise_model  # noqa: E402


def train_allocator_on_real_sens(
    sens_map: dict,
    target_main_sl: float,
    n_blocks: int = 24,
    n_steps: int = 1500,
    lr: float = 0.05,
    budget_lambda: float = 5e-4,
    entropy_lambda: float = 0.02,
    seed: int = 0,
) -> AdaptiveMPAllocator:
    """Train the adaptive allocator with the REAL per-(op, block) sensitivity
    as the noise-cost weight. Budget is FLOP-weighted to match
    ``compute_main_sl``.

    Key differences from the first pass:
      - tau annealing floor raised (1.5 -> 0.8) to avoid softmax collapse
      - entropy bonus on the level distribution to keep it exploring
      - longer training (1500 vs 400 steps)
      - smaller lr so we don't blow past per-block structure
    """
    torch.manual_seed(seed)
    allocator = AdaptiveMPAllocator(
        n_blocks=n_blocks,
        ops=DEFAULT_OPS,
        levels=DEFAULT_LEVELS,
        embed_dim=32,   # more position-specific capacity
        hidden=64,
    )

    sigma = torch.zeros(n_blocks, len(DEFAULT_OPS))
    for oi, op in enumerate(DEFAULT_OPS):
        for b in range(n_blocks):
            sigma[b, oi] = float(sens_map.get((op, b), 0.0))
    sigma = sigma.clamp(min=1e-3)
    print(f"[train] sigma shape {sigma.shape}  range [{sigma.min():.3f}, {sigma.max():.3f}]")

    flop_w = torch.tensor([float(_OP_MACS.get(op, 1.0)) for op in DEFAULT_OPS])
    flop_w = flop_w / flop_w.sum()

    opt = torch.optim.Adam(allocator.parameters(), lr=lr)
    for step in range(n_steps):
        frac = step / max(1, n_steps - 1)
        tau = 1.5 * (1 - frac) + 0.8 * frac     # floor 0.8 (was 0.3)

        # Pull soft probs directly so we can add entropy bonus
        logits = allocator._logits()            # (B, O, K)
        soft = torch.softmax(logits / tau, dim=-1)
        hard = torch.nn.functional.one_hot(
            soft.argmax(-1), num_classes=len(DEFAULT_LEVELS)
        ).to(soft.dtype)
        probs = (hard - soft).detach() + soft   # STE
        levels_t = torch.tensor(DEFAULT_LEVELS, dtype=torch.float32)
        bits = (probs * levels_t).sum(-1)       # (B, O)

        noise_cost = (sigma / bits).sum()
        avg_bits = (bits * flop_w).sum(dim=-1).mean()
        over = torch.clamp(avg_bits - target_main_sl, min=0.0)
        budget_pen = budget_lambda * over.pow(2)

        # Entropy bonus (maximize entropy → explore)
        ent = -(soft * soft.clamp(min=1e-9).log()).sum(-1).mean()
        ent_bonus = -entropy_lambda * ent        # sign: subtract to maximize

        loss = noise_cost + budget_pen + ent_bonus
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(allocator.parameters(), 1.0)
        opt.step()

        if step % max(1, n_steps // 15) == 0 or step == n_steps - 1:
            # Count distinct schedules across blocks per op (diagnostic)
            argmax_levels = logits.argmax(-1)    # (B, O)
            n_distinct = len({tuple(argmax_levels[b].tolist()) for b in range(n_blocks)})
            print(f"[train] step={step:4d} tau={tau:.2f} "
                  f"loss={float(loss):.3f} noise={float(noise_cost):.3f} "
                  f"avg_bits(flop)={float(avg_bits):.1f} "
                  f"ent={float(ent):.3f} "
                  f"distinct_block_schedules={n_distinct}/{n_blocks}")
    return allocator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sc_config", default="skip_worst40")
    p.add_argument("--target_msl", type=float, default=128.0,
                   help="target FLOP-weighted main stoc_len")
    p.add_argument("--n_eval", type=int, default=500)
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data_root", default="/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data")
    p.add_argument("--sc_prec", type=int, default=8)
    p.add_argument("--n_train_steps", type=int, default=400)
    p.add_argument("--run_dir", default="results/adaptive_mp_dinov2_eval")
    p.add_argument("--with_comp", action="store_true", help="also run with QwT comp")
    args = p.parse_args()

    run_dir = HERE / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XFORMERS_DISABLED", "1")
    set_noise_model(False)
    device = torch.device("cuda")

    # Real sensitivity from prior sweep
    sens_map = load_sensitivity()
    active_ops = build_active_ops(args.sc_config)
    print(f"[cfg] sc_config={args.sc_config} active_ops={len(active_ops)} "
          f"target_msl={args.target_msl}")

    # Shared eval loader
    eval_loader = build_eval_loader(
        args.data_root, args.n_eval, args.eval_seed, args.batch_size,
        workers=args.workers,
    )

    results = {"cfg": vars(args), "runs": {}}

    # 1) FP baseline
    from eval import load_model, evaluate
    print("\n[fp] evaluating baseline")
    model_fp = load_model(device).eval()
    seed_all(args.eval_seed)
    t0 = time.time()
    fp_res = evaluate(model_fp, eval_loader, device)
    del model_fp; torch.cuda.empty_cache()
    fp_res["wall_s"] = time.time() - t0
    results["runs"]["fp"] = fp_res
    print(f"[fp] top1={fp_res['top1']:.4f}  t={fp_res['wall_s']:.0f}s")

    # 2) Train allocator with real sensitivity
    print("\n[train] adaptive MP MLP on real sensitivity")
    allocator = train_allocator_on_real_sens(
        sens_map, args.target_msl,
        n_blocks=24, n_steps=args.n_train_steps,
    )
    sl_map_adaptive = allocator.to_sl_map()
    # Zero out ops that are not in active_ops
    for op in DEFAULT_OPS:
        for b in range(24):
            if (op, b) not in active_ops:
                sl_map_adaptive[op][b] = 0
    actual_msl_adaptive = compute_main_sl(sl_map_adaptive, active_ops)
    print(f"[adaptive] actual_msl={actual_msl_adaptive:.1f}  schedule="
          + json.dumps({op: sl_map_adaptive[op] for op in DEFAULT_OPS}))
    allocator.save(run_dir / "allocator_real_sens.pt")

    # 3) Proportional baseline at matching target_msl
    sl_map_prop = alloc_proportional(
        active_ops, args.target_msl, sens_map,
        min_sl=min(DEFAULT_LEVELS), max_sl=max(DEFAULT_LEVELS),
    )
    actual_msl_prop = compute_main_sl(sl_map_prop, active_ops)
    print(f"[proportional] actual_msl={actual_msl_prop:.1f}")

    # 4) Uniform baseline at target_msl
    # Find the closest quantized level to target
    closest_level = min(DEFAULT_LEVELS, key=lambda q: abs(q - args.target_msl))
    sl_map_uniform = alloc_uniform(active_ops, closest_level)
    actual_msl_uniform = compute_main_sl(sl_map_uniform, active_ops)
    print(f"[uniform] level={closest_level}  actual_msl={actual_msl_uniform:.1f}")

    # 5) Evaluate each
    calib_loader = None
    comp_kwargs = {"n_calib": 256, "ridge": 1e-4, "calib_seed": 1}
    if args.with_comp:
        calib_loader = build_eval_loader(
            args.data_root, 256, 1, args.batch_size, workers=args.workers,
        )

    for tag, sl_map, actual in [
        ("adaptive_mlp", sl_map_adaptive, actual_msl_adaptive),
        ("proportional", sl_map_prop, actual_msl_prop),
        ("uniform", sl_map_uniform, actual_msl_uniform),
    ]:
        print(f"\n[eval] {tag}  main_sl={actual:.1f}")
        t0 = time.time()
        r = run_one(
            sl_map, active_ops, args.sc_prec, eval_loader, device,
            seed=args.eval_seed,
            with_comp=args.with_comp,
            calib_loader=calib_loader,
            comp_kwargs=comp_kwargs,
        )
        # run_one returns dict {"raw": ..., "comp": ..., "patch_stats": ...}
        raw_res = r.get("raw") if isinstance(r, dict) else r
        comp_res = r.get("comp") if isinstance(r, dict) else None
        wall = time.time() - t0
        entry = {
            "sl_map": sl_map,
            "actual_main_sl": actual,
            "raw": raw_res,
            "comp": comp_res,
            "patch_stats": r.get("patch_stats") if isinstance(r, dict) else None,
            "wall_s": wall,
        }
        results["runs"][tag] = entry
        msg = f"[eval] {tag}: raw top1={raw_res['top1']:.4f}"
        if comp_res:
            msg += f"  comp top1={comp_res['top1']:.4f}"
        msg += f"  t={wall:.0f}s"
        print(msg)

    # Save
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== FINAL ===")
    fp_t = results["runs"]["fp"]["top1"]
    print(f"{'strategy':>16}  {'raw top1':>10}  {'Δfp':>8}  {'main_sl':>8}")
    for tag in ["adaptive_mlp", "proportional", "uniform"]:
        e = results["runs"][tag]
        r = e["raw"]
        print(f"{tag:>16}  {r['top1']:>10.4f}  {r['top1']-fp_t:>+8.4f}  "
              f"{e['actual_main_sl']:>8.1f}")


if __name__ == "__main__":
    main()
