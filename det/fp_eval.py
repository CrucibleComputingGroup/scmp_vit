"""FP baseline COCO eval (bbox + segm mAP) for EVA-ViTDet.

Mirrors QwT-det's ``fp_eval100.py`` but trimmed to a single entry point
driven by argparse, with the per-run output directory rooted at
``vit_sc/det/results/``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from eval_common import load_model_and_loader, set_evaluator_output_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-eval", type=int, default=100,
                   help="Number of COCO val images (first-N).")
    p.add_argument("--out-tag", type=str, default="fp",
                   help="Subdir under results/ (e.g. fp100).")
    p.add_argument("--size", type=int, default=0,
                   help="Override input square size (multiple of 256). 0=default(1280).")
    p.add_argument("--d2_datasets", type=str, default="",
                   help="DETECTRON2_DATASETS root (default: GreatLakes shared_data).")
    p.add_argument("--ckpt", type=str, default="",
                   help="Path to EVA-ViTDet checkpoint (default: GreatLakes shared_data).")
    p.add_argument("--use_soft_nms", action="store_true",
                   help="Enable mmcv linear soft NMS in CascadeROIHeads.")
    p.add_argument("--interp_type", choices=["", "vitdet", "beit"], default="",
                   help="ViT rel-pos interp type ('beit' for off-train res).")
    args = p.parse_args()

    out_dir = Path(__file__).parent / "results" / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1] Loading config + model + weights")
    model, test_loader, evaluator, subset = load_model_and_loader(
        args.n_eval, d2_datasets=args.d2_datasets, ckpt=args.ckpt,
        size=args.size,
        use_soft_nms=args.use_soft_nms, interp_type=args.interp_type)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    params: {n_params:.1f}M   subset: {len(subset)} images")

    set_evaluator_output_dir(evaluator, out_dir)

    from detectron2.evaluation import inference_on_dataset

    print(f"[2] Running inference on {args.n_eval} images")
    t0 = time.time()
    with torch.no_grad():
        results = inference_on_dataset(model, test_loader, evaluator)
    dt = time.time() - t0
    print(f"    done in {dt:.1f}s ({dt/args.n_eval:.2f}s/img)")

    out = {
        "n_eval": args.n_eval,
        "size": args.size or 1280,
        "use_soft_nms": bool(args.use_soft_nms),
        "interp_type": args.interp_type or "vitdet",
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
