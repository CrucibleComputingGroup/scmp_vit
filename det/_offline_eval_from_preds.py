"""Offline COCO eval from saved detectron2 predictions.

Usage:
    python _offline_eval_from_preds.py \\
        --predictions <dir>/sc_comp/checkpoint.pt \\
        --n_eval 5000 --size 1024 \\
        --out_metrics <dir>/metrics_offline.json

Reuses the same subset-aware COCOEvaluator wiring as
``load_model_and_loader`` in eval_common.py, but skips model construction
and weight loading so it runs in seconds without a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
EVA_DET = str(_REPO_ROOT / "third_party" / "QwT-SC" / "QwT-det-RepQ-ViT" / "eva1" / "eva_det")
if EVA_DET not in sys.path:
    sys.path.insert(0, EVA_DET)


def build_evaluator(n_eval: int, start_idx: int, size: int):
    """Mirror the evaluator/subset setup in load_model_and_loader without
    constructing the model."""
    from eval_common import CFG_PATH, _DEFAULT_D2_DATASETS

    os.environ.setdefault("DETECTRON2_DATASETS", _DEFAULT_D2_DATASETS)

    from detectron2.config import LazyConfig
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.evaluation import COCOEvaluator

    cfg = LazyConfig.load(CFG_PATH)
    orig_name = cfg.dataloader.test.dataset.names
    all_items = DatasetCatalog.get(orig_name)
    subset = all_items[start_idx:start_idx + n_eval]
    size_tag = f"_sz{size}" if size else ""
    if start_idx == 0:
        sub_name = f"{orig_name}_first{n_eval}{size_tag}_offline"
    else:
        sub_name = f"{orig_name}_slice_{start_idx}_{n_eval}{size_tag}_offline"
    if sub_name in DatasetCatalog.list():
        DatasetCatalog.remove(sub_name)
        MetadataCatalog.remove(sub_name)
    DatasetCatalog.register(sub_name, lambda subset=subset: subset)
    md = MetadataCatalog.get(orig_name).as_dict()
    md.pop("name", None)
    MetadataCatalog.get(sub_name).set(**md)

    evaluator = COCOEvaluator(orig_name, tasks=("bbox", "segm"),
                              distributed=False, output_dir=None,
                              max_dets_per_image=None)
    subset_ids = set(int(it["image_id"]) for it in subset)
    coco = evaluator._coco_api
    coco.imgs = {k: v for k, v in coco.imgs.items() if k in subset_ids}
    coco.anns = {k: v for k, v in coco.anns.items() if v["image_id"] in subset_ids}
    coco.imgToAnns = {k: v for k, v in coco.imgToAnns.items() if k in subset_ids}
    coco.catToImgs = {c: [i for i in imgs if i in subset_ids]
                      for c, imgs in coco.catToImgs.items()}
    return evaluator, subset


def load_predictions(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "predictions" in obj:
        return list(obj["predictions"])
    if isinstance(obj, list):
        return list(obj)
    raise SystemExit(f"unexpected predictions format: {type(obj)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True,
                   help="checkpoint.pt or instances_predictions.pth")
    p.add_argument("--n_eval", type=int, default=5000)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--out_metrics", required=True)
    args = p.parse_args()

    print(f"[offline_eval] building evaluator (n_eval={args.n_eval}, "
          f"start={args.start_idx}, size={args.size})...", flush=True)
    evaluator, subset = build_evaluator(args.n_eval, args.start_idx, args.size)
    print(f"  subset: {len(subset)} images", flush=True)

    pred_path = Path(args.predictions)
    print(f"[offline_eval] loading {pred_path}...", flush=True)
    preds = load_predictions(pred_path)
    print(f"  {len(preds)} predictions", flush=True)

    evaluator.reset()
    evaluator._predictions = preds
    print(f"[offline_eval] running evaluator.evaluate()...", flush=True)
    res = evaluator.evaluate()

    out = {
        "n_eval": args.n_eval,
        "image_size": args.size,
        "predictions_path": str(pred_path),
        "bbox": res["bbox"],
        "segm": res["segm"],
    }
    Path(args.out_metrics).write_text(json.dumps(out, indent=2))
    print(f"  bbox AP={res['bbox']['AP']:.2f}  "
          f"segm AP={res['segm']['AP']:.2f}", flush=True)
    print(f"  saved to {args.out_metrics}", flush=True)


if __name__ == "__main__":
    main()
