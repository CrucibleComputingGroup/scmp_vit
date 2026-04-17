"""Preliminary per-(op, block) sensitivity sweep for EVA-ViTDet on COCO.

For each (op, block) combo, SC *only* that single op in that single block;
everything else FP. Measure bbox AP drop vs FP baseline.

This is the detection analog of ``vit_sc/experiments/sensitivity_all_ops.py``
(which targets DINOv2 classification). Uses real SC (no noise surrogate).

Output: results/sensitivity/sensitivity_all_ops.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_DET = _HERE.parent  # vit_sc/det/
if str(_DET) not in sys.path:
    sys.path.insert(0, str(_DET))

from eval_common import load_model_and_loader, set_evaluator_output_dir  # noqa: E402
from sc_patch import sc_patch_eva, SC_OP_NAMES  # noqa: E402


def run_eval(model, test_loader, evaluator, out_dir):
    """Run COCO inference and return results dict."""
    from detectron2.evaluation import inference_on_dataset
    set_evaluator_output_dir(evaluator, out_dir)
    with torch.no_grad():
        results = inference_on_dataset(model, test_loader, evaluator)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_eval", type=int, default=50,
                   help="Number of COCO val images (preliminary → small).")
    p.add_argument("--sc_prec", type=int, default=8)
    p.add_argument("--ops", nargs="*", default=None,
                   help="Subset of ops to sweep (default: all 6). "
                        f"Valid: {SC_OP_NAMES}")
    p.add_argument("--block_start", type=int, default=0,
                   help="First block index (inclusive).")
    p.add_argument("--block_end", type=int, default=-1,
                   help="Last block index (exclusive). -1 = n_blocks.")
    p.add_argument("--out_json",
                   default="results/sensitivity/sensitivity_all_ops.json")
    p.add_argument("--resume", action="store_true",
                   help="Skip (op, block) combos already in out_json.")
    p.add_argument("--d2_datasets", type=str, default="",
                   help="DETECTRON2_DATASETS root.")
    p.add_argument("--ckpt", type=str, default="",
                   help="Path to EVA-ViTDet checkpoint.")
    args = p.parse_args()

    ops_to_sweep = list(SC_OP_NAMES) if args.ops is None else args.ops
    for op in ops_to_sweep:
        if op not in SC_OP_NAMES:
            raise SystemExit(f"unknown op: {op} (valid: {SC_OP_NAMES})")

    out_path = _DET / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = _DET / "results" / "sensitivity" / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume support ---
    done = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        for row in prev.get("grid", []):
            done[(row["op"], int(row["block"]))] = row
        print(f"[resume] loaded {len(done)} existing entries from {out_path}")

    # --- FP baseline ---
    print(f"[1] Loading model + loader (n_eval={args.n_eval})")
    model, test_loader, evaluator, subset = load_model_and_loader(
        args.n_eval, d2_datasets=args.d2_datasets, ckpt=args.ckpt)
    n_blocks = len(list(model.backbone.net.blocks))
    blk_lo = args.block_start
    blk_hi = n_blocks if args.block_end < 0 else min(args.block_end, n_blocks)
    print(f"    n_blocks = {n_blocks},  sweep range [{blk_lo}, {blk_hi})")
    n_combos = len(ops_to_sweep) * (blk_hi - blk_lo) - len(done)
    print(f"    combos to sweep: {n_combos} "
          f"({len(ops_to_sweep)} ops × {n_blocks} blocks"
          f"{f' - {len(done)} done' if done else ''})")

    print("[2] FP baseline")
    t0 = time.time()
    fp_results = run_eval(model, test_loader, evaluator, tmp_dir)
    fp_dt = time.time() - t0
    fp_bbox_ap = fp_results.get("bbox", {}).get("AP", 0.0)
    fp_segm_ap = fp_results.get("segm", {}).get("AP", 0.0)
    print(f"    bbox AP={fp_bbox_ap:.2f}  segm AP={fp_segm_ap:.2f}  "
          f"({fp_dt:.1f}s)")

    # --- Per-(op, block) sweep ---
    grid = list(done.values())  # keep previously computed rows
    t_sweep = time.time()
    combo_i = 0

    for op in ops_to_sweep:
        op_rows = []
        for bi in range(blk_lo, blk_hi):
            if (op, bi) in done:
                op_rows.append(done[(op, bi)])
                continue
            combo_i += 1

            # Reload clean model each time
            del model
            torch.cuda.empty_cache()
            model, test_loader, evaluator, _ = load_model_and_loader(
                args.n_eval, d2_datasets=args.d2_datasets, ckpt=args.ckpt)

            # SC only this one (op, block)
            spec = {k: [0] * n_blocks for k in SC_OP_NAMES}
            spec[op][bi] = 1
            sc_patch_eva(model, sc_prec=args.sc_prec, sc_ops_per_block=spec)

            t0 = time.time()
            results = run_eval(model, test_loader, evaluator, tmp_dir)
            dt = time.time() - t0

            bbox_ap = results.get("bbox", {}).get("AP", 0.0)
            segm_ap = results.get("segm", {}).get("AP", 0.0)
            bbox_drop = fp_bbox_ap - bbox_ap
            segm_drop = fp_segm_ap - segm_ap

            row = {
                "op": op,
                "block": bi,
                "bbox_ap": round(bbox_ap, 4),
                "segm_ap": round(segm_ap, 4),
                "bbox_ap_drop": round(bbox_drop, 4),
                "segm_ap_drop": round(segm_drop, 4),
                "eval_s": round(dt, 1),
            }
            grid.append(row)
            op_rows.append(row)

            eta_s = (time.time() - t_sweep) / combo_i * (n_combos - combo_i)
            print(f"  [{combo_i}/{n_combos}] {op:>10s} blk={bi:>2d}  "
                  f"bbox_AP={bbox_ap:.2f} (Δ={bbox_drop:+.2f})  "
                  f"segm_AP={segm_ap:.2f} (Δ={segm_drop:+.2f})  "
                  f"{dt:.0f}s  ETA {eta_s/60:.0f}min", flush=True)

            # Incremental save after each combo
            _save(out_path, args, n_blocks, fp_bbox_ap, fp_segm_ap, grid)

        if op_rows:
            drops = [r["bbox_ap_drop"] for r in op_rows]
            print(f"[{op:>10s}] bbox_AP_drop min={min(drops):.2f} "
                  f"max={max(drops):.2f}", flush=True)

    _save(out_path, args, n_blocks, fp_bbox_ap, fp_segm_ap, grid)
    total_s = time.time() - t_sweep
    print(f"\n[done] {total_s/60:.1f} min total, wrote {out_path}")

    # --- Ranking ---
    ranked = sorted(grid, key=lambda r: r["bbox_ap_drop"], reverse=True)
    print(f"\nTop-20 most sensitive (op, block) by bbox AP drop:")
    print(f"{'rank':>4s}  {'op':>10s}  {'blk':>3s}  {'bbox_drop':>9s}  {'segm_drop':>9s}")
    for i, r in enumerate(ranked[:20]):
        print(f"{i+1:>4d}  {r['op']:>10s}  {r['block']:>3d}  "
              f"{r['bbox_ap_drop']:>+9.2f}  {r['segm_ap_drop']:>+9.2f}")


def _save(out_path, args, n_blocks, fp_bbox_ap, fp_segm_ap, grid):
    out = {
        "config": {
            "n_eval": args.n_eval,
            "sc_prec": args.sc_prec,
            "n_blocks": n_blocks,
            "op_names": list(SC_OP_NAMES),
        },
        "fp_baseline": {
            "bbox_ap": round(fp_bbox_ap, 4),
            "segm_ap": round(fp_segm_ap, 4),
        },
        "grid": grid,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
