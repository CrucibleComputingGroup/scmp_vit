"""Per-(op, block) SC sensitivity sweep for EVA-ViTDet on COCO.

For each (op, block) pair, SC *only* that single op in that single block at
the given SC length; everything else stays FP. Records bbox/segm AP drop
vs FP baseline.

Length is supplied via `--sc_prec N` (⇒ stoc_len = 2**N) or `--length L`
(arbitrary int, passed as-is to the kernel).

Output: sensitivity/sensitivity_{tag}_n{n_eval}.json
        tag = int{sc_prec} when stoc_len is a power of 2, else len{stoc_len}.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_DET = _HERE.parent
if str(_DET) not in sys.path:
    sys.path.insert(0, str(_DET))

from eval_common import load_model_and_loader, set_evaluator_output_dir  # noqa: E402
from sc_patch import sc_patch_eva, SC_OP_NAMES  # noqa: E402


def run_eval(model, test_loader, evaluator, out_dir):
    from detectron2.evaluation import inference_on_dataset
    set_evaluator_output_dir(evaluator, out_dir)
    with torch.no_grad():
        return inference_on_dataset(model, test_loader, evaluator)


def save(out_path, args, n_blocks, fp_bbox_ap, fp_segm_ap, grid):
    out = {
        "config": {
            "n_eval": args.n_eval,
            "sc_prec": args.sc_prec,
            "stoc_len": args.stoc_len,
            "n_blocks": n_blocks,
            "op_names": list(SC_OP_NAMES),
            "block_start": args.block_start,
            "block_end": args.block_end,
        },
        "fp_baseline": {
            "bbox_ap": round(fp_bbox_ap, 4),
            "segm_ap": round(fp_segm_ap, 4),
        },
        "grid": grid,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sc_prec", type=int,
                   help="SC precision (stoc_len = 2**sc_prec).")
    g.add_argument("--length", "--stoc_len", type=int, dest="length",
                   help="SC stochastic length. sc_prec = ceil(log2(length)).")
    p.add_argument("--n_eval", type=int, default=100,
                   help="Number of COCO val images per config.")
    p.add_argument("--ops", nargs="*", default=None,
                   help=f"Subset of ops (default all). Valid: {SC_OP_NAMES}")
    p.add_argument("--block_start", type=int, default=0)
    p.add_argument("--block_end", type=int, default=-1,
                   help="Exclusive; -1 means n_blocks.")
    p.add_argument("--out_json", type=str, default="",
                   help="Default: sensitivity/sensitivity_{tag}_n{n_eval}.json")
    p.add_argument("--resume", action="store_true",
                   help="Skip (op, block) combos already in out_json.")
    p.add_argument("--d2_datasets", type=str, default="")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--size", type=int, default=0,
                   help="Override image size (square_pad + ResizeShortestEdge). "
                        "Must be multiple of 256. 0 ⇒ config default (1280).")
    p.add_argument("--tmp_dir", type=str, default="",
                   help="COCO evaluator scratch dir. Default sensitivity/_tmp. "
                        "Use a unique value per process when sharding.")
    args = p.parse_args()

    if args.sc_prec is None:
        args.stoc_len = int(args.length)
        args.sc_prec = max(1, int(math.ceil(math.log2(max(args.stoc_len, 2)))))
    else:
        args.stoc_len = 1 << args.sc_prec
    is_pow2 = args.stoc_len == (1 << args.sc_prec)
    print(f"[cfg] sc_prec={args.sc_prec}  stoc_len={args.stoc_len}",
          flush=True)

    ops_to_sweep = list(SC_OP_NAMES) if args.ops is None else args.ops
    for op in ops_to_sweep:
        if op not in SC_OP_NAMES:
            raise SystemExit(f"unknown op: {op} (valid: {SC_OP_NAMES})")

    tag = f"int{args.sc_prec}" if is_pow2 else f"len{args.stoc_len}"
    size_tag = f"_sz{args.size}" if args.size else ""
    default_name = f"sensitivity_{tag}_n{args.n_eval}{size_tag}.json"
    out_path = _DET / (args.out_json or f"sensitivity/{default_name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else _DET / "sensitivity" / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume ---
    done = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        for row in prev.get("grid", []):
            done[(row["op"], int(row["block"]))] = row
        print(f"[resume] loaded {len(done)} entries from {out_path}")

    # --- FP baseline ---
    print(f"[1] loading model + loader (n_eval={args.n_eval})", flush=True)
    model, test_loader, evaluator, _ = load_model_and_loader(
        args.n_eval, d2_datasets=args.d2_datasets, ckpt=args.ckpt,
        size=args.size)
    n_blocks = len(list(model.backbone.net.blocks))
    blk_lo = args.block_start
    blk_hi = n_blocks if args.block_end < 0 else min(args.block_end, n_blocks)
    print(f"    n_blocks={n_blocks}, sweep blocks [{blk_lo}, {blk_hi})",
          flush=True)
    n_combos = len(ops_to_sweep) * (blk_hi - blk_lo) - len(done)
    print(f"    combos to run: {n_combos}", flush=True)

    print("[2] FP baseline", flush=True)
    t0 = time.time()
    fp_results = run_eval(model, test_loader, evaluator, tmp_dir)
    fp_bbox_ap = fp_results.get("bbox", {}).get("AP", 0.0)
    fp_segm_ap = fp_results.get("segm", {}).get("AP", 0.0)
    print(f"    bbox AP={fp_bbox_ap:.2f}  segm AP={fp_segm_ap:.2f}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # --- Sweep ---
    grid = list(done.values())
    t_sweep = time.time()
    combo_i = 0

    for op in ops_to_sweep:
        op_rows = []
        for bi in range(blk_lo, blk_hi):
            if (op, bi) in done:
                op_rows.append(done[(op, bi)])
                continue
            combo_i += 1

            del model
            torch.cuda.empty_cache()
            model, test_loader, evaluator, _ = load_model_and_loader(
                args.n_eval, d2_datasets=args.d2_datasets, ckpt=args.ckpt,
                size=args.size)

            spec = {k: [0] * n_blocks for k in SC_OP_NAMES}
            spec[op][bi] = 1
            sc_patch_eva(model, sc_prec=args.sc_prec, stoc_len=args.stoc_len,
                         sc_ops_per_block=spec)

            t0 = time.time()
            results = run_eval(model, test_loader, evaluator, tmp_dir)
            dt = time.time() - t0

            bbox_ap = results.get("bbox", {}).get("AP", 0.0)
            segm_ap = results.get("segm", {}).get("AP", 0.0)
            row = {
                "op": op,
                "block": bi,
                "bbox_ap": round(bbox_ap, 4),
                "segm_ap": round(segm_ap, 4),
                "bbox_ap_drop": round(fp_bbox_ap - bbox_ap, 4),
                "segm_ap_drop": round(fp_segm_ap - segm_ap, 4),
                "eval_s": round(dt, 1),
            }
            grid.append(row)
            op_rows.append(row)

            eta = (time.time() - t_sweep) / combo_i * (n_combos - combo_i)
            print(f"  [{combo_i}/{n_combos}] {op:>10s} blk={bi:>2d}  "
                  f"bbox_AP={bbox_ap:.2f} (Δ={row['bbox_ap_drop']:+.2f})  "
                  f"segm_AP={segm_ap:.2f} (Δ={row['segm_ap_drop']:+.2f})  "
                  f"{dt:.0f}s  ETA {eta/60:.0f}min", flush=True)

            save(out_path, args, n_blocks, fp_bbox_ap, fp_segm_ap, grid)

        if op_rows:
            drops = [r["bbox_ap_drop"] for r in op_rows]
            print(f"[{op:>10s}] bbox_drop min={min(drops):+.2f} "
                  f"max={max(drops):+.2f}", flush=True)

    save(out_path, args, n_blocks, fp_bbox_ap, fp_segm_ap, grid)
    print(f"\n[done] {(time.time()-t_sweep)/60:.1f} min total, "
          f"wrote {out_path}", flush=True)

    ranked = sorted(grid, key=lambda r: r["bbox_ap_drop"], reverse=True)
    print(f"\nTop-20 most sensitive (op, block) by bbox AP drop:")
    print(f"{'rank':>4s}  {'op':>10s}  {'blk':>3s}  "
          f"{'bbox_drop':>10s}  {'segm_drop':>10s}")
    for i, r in enumerate(ranked[:20]):
        print(f"{i+1:>4d}  {r['op']:>10s}  {r['block']:>3d}  "
              f"{r['bbox_ap_drop']:>+10.2f}  {r['segm_ap_drop']:>+10.2f}")


if __name__ == "__main__":
    main()
