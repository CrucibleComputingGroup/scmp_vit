"""Shared setup used by fp_eval.py and sc_eval.py.

Loads the EVA-ViTDet Detectron2 config + weights and builds a subset-aware
COCO test loader + evaluator.
"""
from __future__ import annotations

import hashlib
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

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


def _collapse_beit_like_qkv_bias(model) -> int:
    """Fold q_bias / v_bias into self.qkv.bias and disable beit_like_qkv_bias.

    EVA-01's Attention.forward synthesises the qkv bias as
    concat(q_bias, 0, v_bias) at runtime via F.linear(weight=self.qkv.weight, ...),
    bypassing the self.qkv module entirely. This means any SC patching that
    replaces self.qkv with SCLinear is silently a no-op — SCLinear.forward()
    is never called. Mathematically equivalent: copy the synthesised bias
    into self.qkv.bias and turn beit_like off so forward goes through self.qkv(x).
    Mirror of QwT-SC/.../eva_eval/quant/reparam.collapse_beit_like_qkv_bias.
    """
    n = 0
    for _, m in model.named_modules():
        if getattr(m, "beit_like_qkv_bias", False):
            with torch.no_grad():
                eff = torch.cat([m.q_bias, torch.zeros_like(m.v_bias), m.v_bias])
                if m.qkv.bias is None:
                    m.qkv.bias = nn.Parameter(
                        torch.zeros(m.qkv.out_features, device=eff.device,
                                    dtype=eff.dtype))
                m.qkv.bias.data.copy_(eff)
            m.beit_like_qkv_bias = False
            n += 1
    return n


def load_model_and_loader(n_eval: int, disable_act_checkpoint: bool = True,
                          d2_datasets: str = "", ckpt: str = "",
                          start_idx: int = 0, size: int = 0,
                          use_soft_nms: bool = True,
                          interp_type: str = ""):
    """Build FP model + weights + test loader for COCO val images
    ``[start_idx, start_idx + n_eval)``.

    If ``size`` > 0, override ``backbone.square_pad`` and the test
    ResizeShortestEdge to that size. ``size`` must be a multiple of 256
    (patch_size 16 * window_size 16). When ``size != 1280`` we also flip
    ``backbone.net.interp_type`` to ``"beit"`` so the trained pos embed
    gets bicubic-interpolated to the new resolution (mirrors the
    cascade_mask_rcnn_vitdet_eva_1536.py recipe; img_size stays at 1280).

    If ``use_soft_nms`` is True (default), enables linear soft-NMS in the
    cascade roi heads (~+0.2-0.5 AP on COCO val, requires mmcv).

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
        if size != 1280:
            # Trained pos embed is 80x80 (=1280/16); BEIT interp resizes it
            # to (size/16)^2 per forward. Set BEIT_INTERP=0 to fall back to
            # the default "vitdet" interpolation (linear) — useful for
            # isolating whether BEIT interp is the cause of SC accuracy
            # collapse at non-1280 sizes.
            if os.environ.get("BEIT_INTERP", "1") not in ("0", "false", "False", ""):
                cfg.model.backbone.net.interp_type = "beit"
    if use_soft_nms:
        cfg.model.roi_heads.use_soft_nms = True
    if interp_type:
        cfg.model.backbone.net.interp_type = interp_type

    model = instantiate(cfg.model).eval().cuda()
    DetectionCheckpointer(model).load(_ckpt)

    n_collapsed = _collapse_beit_like_qkv_bias(model.backbone.net)
    print(f"[load] collapsed beit_like_qkv_bias on {n_collapsed} blocks "
          f"(SC patching of attn.qkv was previously a no-op)", flush=True)

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


def set_evaluator_output_dir(evaluator, out_dir):
    """COCOEvaluator writes coco_instances_results.json to its output_dir.
    The constructor above passes None; this setter plugs a dir in afterwards
    so callers can control where results land.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluator._output_dir = str(out_dir)


def file_sha256(path) -> str:
    """sha256 of file at ``path`` (or '' if path is empty/None). Used to
    embed schedule/sl_map identity into checkpoint cfg signatures so a
    silently-edited file invalidates resume."""
    if not path:
        return ""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_test_loader_for(items, *, size: int = 0,
                          base_dataset_name: str = "coco_2017_val"):
    """Build a fresh detectron2 test loader over ``items`` (list of D2
    dataset dicts). Mirrors the loader-side cfg overrides in
    :func:`load_model_and_loader` (only ``size`` affects the loader; the
    other knobs are model-side).

    The items list is registered under a unique sub_name in
    DatasetCatalog so multiple loaders can coexist.
    """
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import DatasetCatalog, MetadataCatalog

    cfg = LazyConfig.load(CFG_PATH)
    if size:
        assert size % 256 == 0, f"size {size} must be a multiple of 256"
        test_aug = cfg.dataloader.test.mapper.augmentations
        assert len(test_aug) == 1
        test_aug[0].short_edge_length = size
        test_aug[0].max_size = size

    sub_name = f"{base_dataset_name}_resumeloop_{id(items)}_n{len(items)}"
    if sub_name in DatasetCatalog.list():
        DatasetCatalog.remove(sub_name)
        MetadataCatalog.remove(sub_name)
    DatasetCatalog.register(sub_name, lambda items=items: items)
    md = MetadataCatalog.get(base_dataset_name).as_dict()
    md.pop("name", None)
    MetadataCatalog.get(sub_name).set(**md)
    cfg.dataloader.test.dataset.names = sub_name
    return instantiate(cfg.dataloader.test)


def _atomic_save(state: dict, path: Path):
    import torch
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def checkpointed_eval(model, evaluator, subset, out_dir, *,
                      cfg_sig: dict, save_every: int = 50,
                      resume: bool = False, phase: str = "eval",
                      size: int = 0,
                      base_dataset_name: str = "coco_2017_val"):
    """Run ``model`` over ``subset`` (D2 dataset dicts), feed predictions
    to ``evaluator``, and checkpoint to ``out_dir/checkpoint.pt`` every
    ``save_every`` images. Returns ``evaluator.evaluate()`` results.

    Resumable: when ``resume=True`` and ``checkpoint.pt`` exists with a
    matching ``cfg_sig``, already-evaluated images are skipped and the
    helper appends new ones. Mismatched ``cfg_sig`` aborts with exit 2.

    On SIGINT (Ctrl-C) the current checkpoint is flushed before the
    process exits with code 130, so the partial work isn't lost.

    The caller is responsible for any context managers (e.g.
    ``det_kernel_tuning()``) wrapping the call.
    """
    import torch
    from detectron2.evaluation.evaluator import inference_context

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_evaluator_output_dir(evaluator, out_dir)
    ckpt_path = out_dir / "checkpoint.pt"

    evaluator.reset()
    done_ids: set[int] = set()
    elapsed_prev = 0.0
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        prev_sig = ck.get("config", {})
        if prev_sig != cfg_sig:
            raise SystemExit(
                f"[resume:{phase}] config mismatch — refusing to resume.\n"
                f"    prev: {prev_sig}\n    curr: {cfg_sig}"
            )
        evaluator._predictions = list(ck.get("predictions", []))
        done_ids = {int(p["image_id"]) for p in evaluator._predictions}
        elapsed_prev = float(ck.get("elapsed_seconds", 0.0))
        print(f"[resume:{phase}] {len(done_ids)} images already done "
              f"({elapsed_prev:.1f}s prior wall time)", flush=True)
    elif ckpt_path.exists() and not resume:
        print(f"[warn:{phase}] {ckpt_path} exists but resume not set; "
              f"starting fresh and OVERWRITING.", flush=True)
        ckpt_path.unlink()

    remaining = [it for it in subset if int(it["image_id"]) not in done_ids]
    print(f"[{phase}] remaining to evaluate: {len(remaining)}/{len(subset)}",
          flush=True)

    t0 = time.time()
    if remaining:
        loader = build_test_loader_for(
            remaining, size=size, base_dataset_name=base_dataset_name)

        interrupted = {"flag": False}

        def _on_sigint(signum, frame):
            interrupted["flag"] = True
            print(f"\n[signal:{phase}] SIGINT received; will checkpoint "
                  "after current image", flush=True)

        prev_sigint = signal.signal(signal.SIGINT, _on_sigint)

        def _flush(elapsed_now: float):
            state = {
                "config": cfg_sig,
                "phase": phase,
                "predictions": list(evaluator._predictions),
                "image_ids_done": sorted(
                    {int(p["image_id"]) for p in evaluator._predictions}),
                "elapsed_seconds": elapsed_prev + elapsed_now,
            }
            _atomic_save(state, ckpt_path)

        print(f"[{phase}] inference on {len(remaining)} images "
              f"(save_every={save_every})", flush=True)
        n_done_this_run = 0
        try:
            with torch.no_grad():
                with inference_context(model):
                    for inputs in loader:
                        outputs = model(inputs)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        evaluator.process(inputs, outputs)
                        n_done_this_run += 1
                        total_done = len(evaluator._predictions)
                        if n_done_this_run % save_every == 0:
                            _flush(time.time() - t0)
                            dt = time.time() - t0
                            ips = n_done_this_run / max(dt, 1e-6)
                            eta = ((len(remaining) - n_done_this_run)
                                   / max(ips, 1e-6))
                            print(f"    [{phase} {total_done}/{len(subset)}]  "
                                  f"this-run {n_done_this_run}/{len(remaining)}"
                                  f"  {dt:.0f}s  {ips:.2f} img/s  "
                                  f"ETA {eta/60:.1f}min", flush=True)
                        if interrupted["flag"]:
                            break
        finally:
            # Save first, then restore the prior handler — if a SIGINT lands
            # mid-restore, we don't want it raising during the save.
            _flush(time.time() - t0)
            signal.signal(signal.SIGINT, prev_sigint)

        if interrupted["flag"]:
            print(f"[interrupted:{phase}] saved checkpoint at {ckpt_path}. "
                  "Re-run with --resume to continue.", flush=True)
            sys.exit(130)
        print(f"[{phase}] inference done in {time.time()-t0:.1f}s "
              f"(+{elapsed_prev:.1f}s prior)", flush=True)
    else:
        print(f"[{phase}] nothing to evaluate; all images already done.",
              flush=True)

    return evaluator.evaluate()
