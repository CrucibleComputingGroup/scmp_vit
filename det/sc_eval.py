"""SC eval on COCO with an EVA-ViTDet backbone.

Patches the backbone in place (``sc_patch.sc_patch_eva``) according to a
per-(op, block) schedule, then runs the standard Detectron2 COCO eval.

Supported ``--sc_config`` values:
  - ``all_on``           : every (op, block) on.
  - ``skip_last_fc2_K``  : all on except ``mlp_fc2`` for the last K blocks
                           (uses ``--skip-last-k``).
  - ``from_json``        : load from ``--sc_ops_per_block_json``.
  - individual op flags (``--sc-qk`` / ``--sc-av`` / ``--sc-qkv-proj`` /
    ``--sc-out-proj`` / ``--sc-mlp-fc1`` / ``--sc-mlp-fc2``) set that op on
    uniformly across all blocks; convenient for single-op ablations.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from eval_common import load_model_and_loader, set_evaluator_output_dir

# Make ./sc_patch a package importable as ``sc_patch`` (not ``det.sc_patch``).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from sc_patch import (  # noqa: E402
    sc_patch_eva,
    SC_OP_NAMES,
    build_skip_last_fc2_schedule,
    normalize_sc_ops_per_block,
    count_sc_ops,
)


def build_schedule_from_args(args, n_blocks):
    if args.sc_ops_per_block_json:
        with open(args.sc_ops_per_block_json) as f:
            raw = json.load(f)
        return normalize_sc_ops_per_block(raw, n_blocks)

    if args.sc_config == "all_on":
        return {op: [1] * n_blocks for op in SC_OP_NAMES}

    if args.sc_config == "skip_last_fc2_K":
        return build_skip_last_fc2_schedule(n_blocks, args.skip_last_k)

    if args.sc_config == "from_json":
        raise SystemExit("--sc_config from_json requires --sc_ops_per_block_json PATH")

    if args.sc_config == "custom":
        # Assemble from per-op bools.
        sched = {op: [0] * n_blocks for op in SC_OP_NAMES}
        if args.sc_qk:       sched["qk"]       = [1] * n_blocks
        if args.sc_av:       sched["av"]       = [1] * n_blocks
        if args.sc_qkv_proj: sched["qkv_proj"] = [1] * n_blocks
        if args.sc_out_proj: sched["out_proj"] = [1] * n_blocks
        if args.sc_mlp_fc1:  sched["mlp_fc1"]  = [1] * n_blocks
        if args.sc_mlp_fc2:  sched["mlp_fc2"]  = [1] * n_blocks
        return sched

    raise SystemExit(f"unknown --sc_config {args.sc_config}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--sc-prec", type=int, default=8,
                   help="SC precision (stoc_len = 2**sc_prec).")
    p.add_argument("--sc_config",
                   choices=["all_on", "skip_last_fc2_K", "from_json", "custom"],
                   default="all_on")
    p.add_argument("--skip-last-k", type=int, default=4,
                   help="Used when --sc_config=skip_last_fc2_K.")
    p.add_argument("--sc_ops_per_block_json", type=str, default=None,
                   help="Path to a JSON schedule dict "
                        "{op: [0/1]*n_blocks}. Overrides --sc_config.")

    # Per-op bool flags for --sc_config=custom.
    p.add_argument("--sc-qk",       action="store_true")
    p.add_argument("--sc-av",       action="store_true")
    p.add_argument("--sc-qkv-proj", action="store_true")
    p.add_argument("--sc-out-proj", action="store_true")
    p.add_argument("--sc-mlp-fc1",  action="store_true")
    p.add_argument("--sc-mlp-fc2",  action="store_true")

    p.add_argument("--out-tag", type=str, default=None,
                   help="Subdir under results/. Default: auto from sc_config.")
    args = p.parse_args()

    tag = args.out_tag or f"sc_{args.sc_config}_p{args.sc_prec}"
    out_dir = Path(__file__).parent / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1] Loading config + model + weights")
    model, test_loader, evaluator, subset = load_model_and_loader(args.n_eval)
    set_evaluator_output_dir(evaluator, out_dir)

    # Count blocks to size schedules properly.
    n_blocks = len(list(model.backbone.net.blocks))
    print(f"    n_blocks = {n_blocks}")

    sched = build_schedule_from_args(args, n_blocks)
    stats_requested = count_sc_ops(sched)
    print(f"[2] Schedule: {args.sc_config}  (requested per-op SC counts)")
    for op in SC_OP_NAMES:
        on, tot = stats_requested[op]
        print(f"    {op:<10s}  {on:3d}/{tot}")

    print(f"[3] Patching backbone (sc_prec={args.sc_prec})")
    stats = sc_patch_eva(model, sc_prec=args.sc_prec, sc_ops_per_block=sched)
    print(f"    swapped: {stats}")

    # Sanity-check: a module in a "proj"-off op should still be nn.Linear.
    from detectron2.evaluation import inference_on_dataset

    print(f"[4] Running inference on {args.n_eval} images")
    t0 = time.time()
    with torch.no_grad():
        results = inference_on_dataset(model, test_loader, evaluator)
    dt = time.time() - t0
    print(f"    done in {dt:.1f}s ({dt/args.n_eval:.2f}s/img)")

    out = {
        "n_eval": args.n_eval,
        "sc_prec": args.sc_prec,
        "sc_config": args.sc_config,
        "skip_last_k": args.skip_last_k if args.sc_config == "skip_last_fc2_K" else None,
        "n_blocks": n_blocks,
        "schedule": {k: list(v) for k, v in sched.items()},
        "swapped": stats,
        "eval_seconds": dt,
        **results,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    for task, m in results.items():
        print(f"    [{task}] AP={m.get('AP',0):.2f}  "
              f"AP50={m.get('AP50',0):.2f}  AP75={m.get('AP75',0):.2f}")
    print(f"[OK] metrics -> {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
