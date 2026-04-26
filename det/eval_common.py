"""Shared setup used by fp_eval.py and sc_eval.py.

Loads the EVA-ViTDet Detectron2 config + weights and builds a subset-aware
COCO test loader + evaluator.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT_D2_DATASETS = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data"
_DEFAULT_CKPT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/pretrained/eva_coco_det.pth"

# Point at the QwT-det fork of EVA-01's ViTDet codebase. The fork is
# EVA-01 + (a) Pillow/mmcv compat patches, (b) a compiled detectron2 C ext,
# (c) MatMul() wrappers in Attention so SC swap is pure module-swap. The
# official EVA-01 source doesn't ship (b) and can't run on the current env
# without rebuilding, so we use the fork as the de-facto D2 base.
_REPO_ROOT = Path(__file__).resolve().parent.parent
EVA_DET = str(_REPO_ROOT / "third_party" / "QwT-SC" / "QwT-det-RepQ-ViT" / "eva1" / "eva_det")
if EVA_DET not in sys.path:
    sys.path.insert(0, EVA_DET)

CFG_PATH = f"{EVA_DET}/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py"


def load_model_and_loader(n_eval: int, disable_act_checkpoint: bool = True,
                          d2_datasets: str = "", ckpt: str = "",
                          start_idx: int = 0, size: int = 0):
    """Build FP model + weights + test loader for COCO val images
    ``[start_idx, start_idx + n_eval)``.

    If ``size`` > 0, override ``backbone.square_pad`` and the test
    ResizeShortestEdge to that size. ``size`` must be a multiple of 256
    (patch_size 16 * window_size 16). Pos params are interpolated at runtime.

    Returns ``(model, test_loader, evaluator, subset)``. ``evaluator`` is a
    ``COCOEvaluator`` whose internal ``_coco_api`` has been filtered to the
    subset, so AP is computed only against those images. ``start_idx`` allows
    disjoint slices for cross-seed calibration; default 0 preserves prior
    behavior.
    """
    os.environ.setdefault("DETECTRON2_DATASETS", d2_datasets or _DEFAULT_D2_DATASETS)
    _ckpt = ckpt or _DEFAULT_CKPT

    from detectron2.config import LazyConfig, instantiate
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.evaluation import COCOEvaluator

    cfg = LazyConfig.load(CFG_PATH)
    if disable_act_checkpoint:
        cfg.model.backbone.net.use_act_checkpoint = False
    if size:
        assert size % 256 == 0, f"size {size} must be a multiple of 256"
        cfg.model.backbone.square_pad = size
        test_aug = cfg.dataloader.test.mapper.augmentations
        assert len(test_aug) == 1
        test_aug[0].short_edge_length = size
        test_aug[0].max_size = size

    model = instantiate(cfg.model).eval().cuda()
    DetectionCheckpointer(model).load(_ckpt)

    orig_name = cfg.dataloader.test.dataset.names
    all_items = DatasetCatalog.get(orig_name)
    subset = all_items[start_idx:start_idx + n_eval]
    size_tag = f"_sz{size}" if size else ""
    if start_idx == 0:
        sub_name = f"{orig_name}_first{n_eval}{size_tag}"
    else:
        sub_name = f"{orig_name}_slice_{start_idx}_{n_eval}{size_tag}"
    if sub_name in DatasetCatalog.list():
        DatasetCatalog.remove(sub_name)
        MetadataCatalog.remove(sub_name)
    DatasetCatalog.register(sub_name, lambda subset=subset: subset)
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
