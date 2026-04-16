"""Shared setup used by fp_eval.py and sc_eval.py.

Loads the EVA-ViTDet Detectron2 config + weights and builds a subset-aware
COCO test loader + evaluator.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault(
    "DETECTRON2_DATASETS",
    "/scratch/nbleier_owned_root/nbleier_owned1/shared_data",
)

# Point at the QwT-det fork of EVA-01's ViTDet codebase. The fork is
# EVA-01 + (a) Pillow/mmcv compat patches, (b) a compiled detectron2 C ext,
# (c) MatMul() wrappers in Attention so SC swap is pure module-swap. The
# official EVA-01 source doesn't ship (b) and can't run on the current env
# without rebuilding, so we use the fork as the de-facto D2 base.
EVA_DET = "/home/yjrcs/SC_V/QwT/QwT-det-RepQ-ViT/eva_det"
if EVA_DET not in sys.path:
    sys.path.insert(0, EVA_DET)

CFG_PATH = f"{EVA_DET}/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py"
CKPT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/pretrained/eva_coco_det.pth"


def load_model_and_loader(n_eval: int, disable_act_checkpoint: bool = True):
    """Build FP model + weights + test loader limited to first ``n_eval`` images.

    Returns ``(model, test_loader, evaluator, subset)``. ``evaluator`` is a
    ``COCOEvaluator`` whose internal ``_coco_api`` has been filtered to the
    subset, so AP is computed only against those images.
    """
    from detectron2.config import LazyConfig, instantiate
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.evaluation import COCOEvaluator

    cfg = LazyConfig.load(CFG_PATH)
    if disable_act_checkpoint:
        cfg.model.backbone.net.use_act_checkpoint = False

    model = instantiate(cfg.model).eval().cuda()
    DetectionCheckpointer(model).load(CKPT)

    orig_name = cfg.dataloader.test.dataset.names
    all_items = DatasetCatalog.get(orig_name)
    subset = all_items[:n_eval]
    sub_name = f"{orig_name}_first{n_eval}"
    if sub_name in DatasetCatalog.list():
        DatasetCatalog.remove(sub_name)
        MetadataCatalog.remove(sub_name)
    DatasetCatalog.register(sub_name, lambda: subset)
    md = MetadataCatalog.get(orig_name).as_dict()
    md.pop("name", None)
    MetadataCatalog.get(sub_name).set(**md)
    cfg.dataloader.test.dataset.names = sub_name
    test_loader = instantiate(cfg.dataloader.test)

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
    return model, test_loader, evaluator, subset


def set_evaluator_output_dir(evaluator, out_dir: Path):
    """COCOEvaluator writes coco_instances_results.json to its output_dir.
    The constructor above passes None; this setter plugs a dir in afterwards
    so callers can control where results land.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluator._output_dir = str(out_dir)
