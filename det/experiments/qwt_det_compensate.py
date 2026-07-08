"""QwT cross-seed cosine-gated linear compensation for SC EVA-ViTDet.

Det-side analogue of ``cls/experiments/qwt_sc_overnight.py``. Fits a per-block
residual ``x @ W + b`` on **two disjoint** calibration batches, admits a
block iff the two ridge fits agree in direction (``cos(W_A, W_B) > τ``), then
wraps each admitted SC block with a ``CompensationBlock`` and evaluates on
COCO. The cross-seed admission rule lives in
``third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py`` — see that module's
docstring for the physical derivation. This driver only handles the D2 /
COCO plumbing (image preprocessing through ``model.preprocess_image``,
COCOEvaluator, 1024² fwd_chunk).

Usage::

    python experiments/qwt_det_compensate.py \\
        --sc_prec 7 --sc_ops_per_block_json sensitivity/skip/skip_worst_30_int7.json \\
        --n_calib 64 --n_eval 100 \\
        --calib_seed 1 --calib_seed_b 2 \\
        --cos_threshold 0.5 --last_block_cos_threshold 0.8 --lookahead_veto \\
        --comp_mode sc --comp_sc_prec 8 \\
        --out_json results/qwt_det/run.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_DET = _HERE.parent
_REPO = _DET.parent
_QWT_LIB = _REPO / "third_party" / "QwT-SC" / "QwT-vit-sc"
for p in (_DET, _REPO, _QWT_LIB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_common import (load_model_and_loader, set_evaluator_output_dir,
                         checkpointed_eval, file_sha256)
from mp_spec import (add_mp_args, build_attn_mp_spec, build_linear_mp_spec,
                     mp_args_to_dict)
from sc_patch import sc_patch_eva, SC_OP_NAMES
from sc_integration.sc_linear import SCLinear
from sc_integration.head_aligned_comp import HeadAlignedSCLinear
from scmp_kernels.sc import det_kernel_tuning
from qwt_sc import calibrate_qwt
from qwt_sc.compensation import CompensationBlock


def make_fp_factory(scale: float = 1.0, polarity: int = 1):
    """(W, b) -> nn.Linear FP comp. Debug / baseline only — FP comp
    reintroduces an FP multiplier that the SC hardware story was designed
    to avoid. Production must use ``--comp_mode sc``."""
    def factory(W, b):
        D_in, D_out = W.shape
        layer = nn.Linear(D_in, D_out, bias=True)
        with torch.no_grad():
            layer.weight.copy_((W.t().contiguous()) * (scale * polarity))
            layer.bias.copy_(b * polarity if polarity == -1 else b)
        return layer
    return factory


def make_sc_comp_factory(sc_prec: int, mode: str = "bipolar",
                         scale: float = 1.0, polarity: int = 1):
    """(W, b) -> SCLinear wrapping the ridge-fit (D_in, D_out) matmul.
    Routes the comp through the same XNOR/Sobol pipeline as the block SC
    matmuls — keeps the whole inference path in SC hardware (no FP
    multiplier). Pulls Sobol pool from ``_CFG_CACHE[(D_in, sc_prec)]``."""
    def factory(W, b):
        D_in, D_out = W.shape
        layer = nn.Linear(D_in, D_out, bias=True)
        with torch.no_grad():
            layer.weight.copy_((W.t().contiguous()) * (scale * polarity))
            layer.bias.copy_(b * polarity if polarity == -1 else b)
        return SCLinear(layer, sc_prec=sc_prec, mode=mode)
    return factory


def make_head_aligned_comp_factory(sc_prec: int, mode: str = "bipolar",
                                   n_heads: int = 16, scale: float = 1.0,
                                   polarity: int = 1):
    """(W, b) -> HeadAlignedSCLinear. Splits the comp matmul into
    ``n_heads`` per-head SC matmuls of shape ``(D_in/n_heads, D_out/n_heads)``,
    each reusing ``_CFG_CACHE[(D_in/n_heads, sc_prec)]`` — the same pool
    that the block's per-head Q@K^T matmul already populates. Adds **zero**
    new Sobol config cache entries on EVA-ViTDet (D_in=1408, n_heads=16,
    per-head D=88; (88, 8) is shared with QK)."""
    def factory(W, b):
        D_in, D_out = W.shape
        layer = nn.Linear(D_in, D_out, bias=True)
        with torch.no_grad():
            layer.weight.copy_((W.t().contiguous()) * (scale * polarity))
            layer.bias.copy_(b * polarity if polarity == -1 else b)
        return HeadAlignedSCLinear(layer, sc_prec=sc_prec, mode=mode,
                                   n_heads=n_heads)
    return factory


def parse_sc_ops(spec: str) -> dict:
    """Turn 'qk,av,proj,mlp_fc1' into a sched dict ready for sc_patch_eva."""
    names = [s.strip() for s in spec.split(",") if s.strip()]
    out = {n: 1 for n in names}
    if "proj" in out:
        v = out.pop("proj")
        out.setdefault("qkv_proj", v)
        out.setdefault("out_proj", v)
    return out


class BackboneCalibLoader:
    """Yields (image_tensor,) 1-tuples from a D2 test loader, using the
    model's preprocess_image to get a normalized (B,3,H,W) float tensor.

    Logs the COCO image_ids it yields when ``log_ids`` is set, so the
    caller can verify the two cross-seed loaders are disjoint."""

    def __init__(self, d2_loader, model_for_preproc, device, max_items,
                 tag: str = "", log_ids: bool = False):
        self._loader = d2_loader
        self._pp = model_for_preproc
        self._device = device
        self._max = max_items
        self._tag = tag
        self._log_ids = log_ids
        self.image_ids: list[int] = []

    def __iter__(self):
        n = 0
        ids: list[int] = []
        for batched_inputs in self._loader:
            for it in batched_inputs:
                if "image_id" in it:
                    ids.append(int(it["image_id"]))
            images = self._pp.preprocess_image(batched_inputs).tensor.to(self._device)
            yield (images,)
            n += images.size(0)
            if n >= self._max:
                break
        self.image_ids = ids
        if self._log_ids:
            print(f"[calib:{self._tag}] yielded {len(ids)} image_ids "
                  f"(first 8: {ids[:8]})", flush=True)


@torch.no_grad()
def run_coco_eval(model, evaluator, subset, out_dir, *,
                  cfg_sig, resume, save_every, phase, size):
    """Checkpointed COCO eval. Drops the prior ``inference_on_dataset``
    one-shot in favor of ``checkpointed_eval`` so a 5000-img run can
    survive Ctrl-C / walltime kills mid-eval.

    ``det_kernel_tuning()`` opts in to det-tuned tile sizes for SCMatMul +
    SCLinear (commit 256f8ad: 1.81× e2e on EVA-ViTDet, bit-exact). cls
    path is unaffected — the context manager flips a re-entrant module
    counter that only enable-signal kernels read.
    """
    with det_kernel_tuning():
        return checkpointed_eval(
            model, evaluator, subset, out_dir,
            cfg_sig=cfg_sig, save_every=save_every,
            resume=resume, phase=phase, size=size,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sc_prec", type=int, default=7)
    p.add_argument("--sc_ops", type=str, default="qk,av,proj",
                   help="Comma-separated subset of {qk,av,proj,mlp_fc1,mlp_fc2}.")
    p.add_argument("--sc_ops_per_block_json", type=str, default="",
                   help="Optional per-(op, block) schedule JSON; overrides --sc_ops.")
    p.add_argument("--n_calib", type=int, default=64)
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--eval_start_idx", type=int, default=0,
                   help="COCO val start index for the eval slice. Default 0 "
                        "evaluates [0, n_eval). Use to incrementally extend a "
                        "prior n=N1 run by another N2 imgs: rerun with "
                        "--eval_start_idx N1 --n_eval N2 (output dir must "
                        "differ to avoid skipping). Calib slice is independent, "
                        "driven by --calib_seed{,_b}.")
    p.add_argument("--size", type=int, default=0,
                   help="Override input square size (multiple of 256). "
                        "0=default(1280). Smaller is faster (1024² ≈ 1.7× of "
                        "1280² wall) but lower AP.")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--start_block", type=int, default=0,
                   help="Indices below this are never admitted (pass-through).")
    p.add_argument("--fwd_chunk", type=int, default=2,
                   help="Det images are 1280², so use tiny chunks.")
    # ---- Cross-seed gate (single-recipe admission rule) ----
    p.add_argument("--calib_seed", type=int, default=1,
                   help="Image-range start for calib loader A. Used as "
                        "start_idx into the COCO val list — A draws "
                        "[calib_seed*0, n_calib).")
    p.add_argument("--calib_seed_b", type=int, default=2,
                   help="Image-range tag for calib loader B. Must differ "
                        "from --calib_seed; B draws a disjoint slice from A.")
    p.add_argument("--cos_threshold", type=float, default=0.5,
                   help="Admission threshold τ. A block is admitted iff "
                        "cos(W_A, W_B) > τ. Pilot on cls finds τ "
                        "insensitive in [0.3, 0.65]; default 0.5.")
    p.add_argument("--last_block_cos_threshold", type=float, default=0.8,
                   help="Stricter τ for the last block (block N-1 feeds "
                        "the head directly; demand tighter agreement). "
                        "Default 0.8. Set ≤ --cos_threshold to disable.")
    p.add_argument("--norm_floor", type=float, default=0.0,
                   help="Additionally require min(||W_A||,||W_B||) > this.")
    p.add_argument("--lookahead_veto", action="store_true",
                   help="One-step binary lookahead: propagate 'apply' vs "
                        "'skip' through block i+1 and veto apply if skip "
                        "is closer to the FP chain. Cheap insurance for "
                        "borderline blocks.")
    p.add_argument("--skip_baseline", action="store_true",
                   help="Skip raw-SC eval (use when sibling sweep already "
                        "recorded the baseline).")
    # ---- Comp-kernel choice ----
    p.add_argument("--comp_mode", choices=["fp", "sc"], default="sc",
                   help="Kernel for the installed correction matmul. 'sc' "
                        "(default) routes W̄ through SCLinear so the whole "
                        "inference path stays in SC hardware. 'fp' is "
                        "debug only.")
    p.add_argument("--comp_sc_prec", type=int, default=8)
    p.add_argument("--comp_sc_mode", choices=["bipolar", "unipolar"], default="bipolar")
    p.add_argument("--head_aligned", action="store_true",
                   help="Route the SC comp through HeadAlignedSCLinear "
                        "(per-head SC matmul reusing _CFG_CACHE[(D_head, "
                        "sc_prec)] = block QK pool — zero new Sobol config "
                        "cache). Strongly recommended for production.")
    p.add_argument("--head_aligned_only", action="store_true",
                   help="Same as --head_aligned (kept for parity with cls "
                        "driver; the variant picker isn't exposed here).")
    p.add_argument("--n_heads", type=int, default=16,
                   help="Head count for HeadAlignedSCLinear; must divide "
                        "the residual dim (EVA-ViTDet: 1408 / 16 = 88).")
    p.add_argument("--out_tag", default="qwt_det")
    p.add_argument("--out_json", default="results/qwt_det/run.json")
    p.add_argument("--use_soft_nms", action="store_true",
                   help="Enable mmcv linear soft NMS in CascadeROIHeads.")
    p.add_argument("--interp_type", choices=["", "vitdet", "beit"], default="",
                   help="ViT rel-pos interp type (see sc_eval.py for details).")
    p.add_argument("--stoc_len", type=int, default=0,
                   help="Override SC bitstream length for ALL ops (fast "
                        "uniform path). Default 0 = use 2**sc_prec. Use this "
                        "instead of --mp_levels L --mp_fractions 1.0.")
    p.add_argument("--d2_datasets", type=str, default="",
                   help="DETECTRON2_DATASETS root (default: GreatLakes shared_data).")
    p.add_argument("--ckpt", type=str, default="",
                   help="Path to EVA-ViTDet checkpoint (default: GreatLakes shared_data).")
    p.add_argument("--sl_map_json", default="",
                   help="Per-(op, block) stoc_len map from "
                        "experiments/mp_budget_swap_search.py. When set, "
                        "overrides --sc_ops_per_block_json and --mp_* flags; "
                        "patches the SC model per-block via the sl_map.")
    p.add_argument("--resume", action="store_true",
                   help="If checkpoint.pt exists in the per-phase output dir "
                        "(sc_raw/, sc_comp/), skip already-evaluated images "
                        "and append new ones. Calib is NOT checkpointed: a "
                        "resume mid-sc_comp redoes calib (~minutes) but "
                        "keeps the eval predictions already saved (~hours).")
    p.add_argument("--save_every", type=int, default=50,
                   help="Checkpoint cadence (images). Also flushed on SIGINT.")
    add_mp_args(p)
    args = p.parse_args()

    if args.calib_seed_b == args.calib_seed:
        raise SystemExit("--calib_seed_b must differ from --calib_seed "
                         "(cross-seed gate requires disjoint calib batches)")

    device = torch.device("cuda")
    print(f"[env] torch {torch.__version__} cuda {torch.version.cuda} "
          f"dev={torch.cuda.get_device_name(0)}", flush=True)

    prof: dict[str, float] = {}
    t_total0 = time.time()

    # --- Build SC schedule ---
    if args.sc_ops_per_block_json:
        with open(args.sc_ops_per_block_json) as f:
            sched = json.load(f)
        cfg_desc = f"schedule from {args.sc_ops_per_block_json}"
    else:
        sched = parse_sc_ops(args.sc_ops)
        cfg_desc = f"sc_ops={args.sc_ops}"
    print(f"[cfg] sc_prec={args.sc_prec}  {cfg_desc}  "
          f"n_calib={args.n_calib}  n_eval={args.n_eval}", flush=True)

    # --- Load FP + SC models, both with full eval loaders. ---
    _load_kw = dict(d2_datasets=args.d2_datasets, ckpt=args.ckpt, size=args.size,
                    use_soft_nms=args.use_soft_nms, interp_type=args.interp_type)
    print("[model] loading FP reference", flush=True)
    _t = time.time()
    model_fp, test_loader_fp, evaluator_fp, _ = load_model_and_loader(
        args.n_eval, start_idx=args.eval_start_idx, **_load_kw)
    model_fp.eval()
    prof["model_load_fp"] = time.time() - _t

    print("[model] loading SC model (fresh copy)", flush=True)
    _t = time.time()
    model_sc, test_loader_sc, evaluator_sc, subset_eval = load_model_and_loader(
        args.n_eval, start_idx=args.eval_start_idx, **_load_kw)
    model_sc.eval()
    prof["model_load_sc"] = time.time() - _t

    linear_mp_spec = build_linear_mp_spec(args) or None
    attn_mp_spec = build_attn_mp_spec(args) or None

    _t = time.time()
    if args.sl_map_json:
        # Per-(op, block) sl_map mode — bypass sc_patch_eva for SC patching
        # and use the MP search's per-block patcher.
        from experiments.mp_budget_swap_search import _patch_block_with_spec
        from sc_integration import mp_search as ms
        with open(args.sl_map_json) as f:
            sl_map_d = json.load(f)
        sl_map = sl_map_d.get("sl_map", sl_map_d) if isinstance(sl_map_d, dict) else sl_map_d
        print(f"[sc] patching model_sc per-block from sl_map "
              f"({args.sl_map_json}) sc_prec={args.sc_prec}", flush=True)
        stats = {op: 0 for op in SC_OP_NAMES}
        for bi in range(len(model_sc.backbone.net.blocks)):
            spec = ms.build_current_spec(sl_map, SC_OP_NAMES, bi)
            new_blk = _patch_block_with_spec(
                model_sc.backbone.net.blocks[bi], args.sc_prec, spec)
            model_sc.backbone.net.blocks[bi] = new_blk.cuda()
            for op in SC_OP_NAMES:
                if ms.entry_to_mpconfig(spec.get(op, 0)) is not None:
                    stats[op] += 1
    else:
        if linear_mp_spec:
            print(f"[mp] linear_mp_spec ops={sorted(linear_mp_spec)}", flush=True)
        if attn_mp_spec:
            print(f"[mp] attn_mp_spec ops={sorted(attn_mp_spec)}", flush=True)
        if args.stoc_len and (linear_mp_spec or attn_mp_spec):
            raise SystemExit("--stoc_len conflicts with --mp_levels (both "
                             "compete for stoc_len). Use one or the other.")
        sl_eff = args.stoc_len or None
        print(f"[sc] patching model_sc (sc_prec={args.sc_prec}, "
              f"stoc_len={sl_eff or 2**args.sc_prec})", flush=True)
        stats = sc_patch_eva(model_sc, sc_prec=args.sc_prec, sc_ops_per_block=sched,
                            linear_mp_spec=linear_mp_spec,
                            attn_mp_spec=attn_mp_spec,
                            stoc_len=sl_eff)
    prof["patch"] = time.time() - _t
    print(f"[sc] patched: {stats}", flush=True)

    # --- Per-phase cfg signatures for the resumable eval. sc_raw depends
    # only on the SC patch; sc_comp additionally depends on every calib /
    # comp knob. A mismatch on resume aborts the run rather than silently
    # mixing predictions from different configs.
    sc_eval_sig = {
        "n_eval": args.n_eval,
        "eval_start_idx": args.eval_start_idx,
        "size": args.size,
        "sc_prec": args.sc_prec,
        "sc_ops": args.sc_ops,
        "sc_ops_per_block_json_sha256": file_sha256(args.sc_ops_per_block_json),
        "sl_map_json_sha256": file_sha256(args.sl_map_json),
        "stoc_len": args.stoc_len,
        "use_soft_nms": bool(args.use_soft_nms),
        "interp_type": args.interp_type or "",
        "mp": mp_args_to_dict(args),
    }
    use_head_aligned = bool(args.head_aligned or args.head_aligned_only)
    sc_comp_sig = dict(sc_eval_sig)
    sc_comp_sig.update({
        "n_calib": args.n_calib,
        "calib_seed": args.calib_seed,
        "calib_seed_b": args.calib_seed_b,
        "ridge": args.ridge,
        "start_block": args.start_block,
        "cos_threshold": args.cos_threshold,
        "last_block_cos_threshold": args.last_block_cos_threshold,
        "norm_floor": args.norm_floor,
        "lookahead_veto": bool(args.lookahead_veto),
        "comp_mode": args.comp_mode,
        "comp_sc_prec": args.comp_sc_prec,
        "comp_sc_mode": args.comp_sc_mode,
        "head_aligned": use_head_aligned,
        "n_heads": args.n_heads if use_head_aligned else None,
        "fwd_chunk": args.fwd_chunk,
    })

    # --- SC baseline (optional) ---
    out_dir_base = _DET / "results" / args.out_tag
    out_dir_base.mkdir(parents=True, exist_ok=True)
    res_sc_raw = None
    if not args.skip_baseline:
        print("[eval] SC baseline (no comp)", flush=True)
        t0 = time.time()
        res_sc_raw = run_coco_eval(
            model_sc, evaluator_sc, subset_eval, out_dir_base / "sc_raw",
            cfg_sig=sc_eval_sig, resume=args.resume,
            save_every=args.save_every, phase="sc_raw", size=args.size)
        prof["raw_sc_eval"] = time.time() - t0
        print(f"  bbox AP={res_sc_raw['bbox']['AP']:.2f}  "
              f"segm AP={res_sc_raw['segm']['AP']:.2f}  "
              f"({prof['raw_sc_eval']:.0f}s)", flush=True)

    # --- Two disjoint calib loaders for the cross-seed gate ---
    # COCO val is deterministic in D2; we draw two disjoint contiguous
    # slices to imitate the cls side's two-seed shuffle. The slice anchors
    # come from --calib_seed{,_b} so reruns with different seed pairs use
    # different slices.
    start_a = (max(0, args.calib_seed - 1)) * args.n_calib
    start_b = (max(0, args.calib_seed_b - 1)) * args.n_calib
    if start_b == start_a:
        # Fallback: shift B by n_calib so the slices are disjoint.
        start_b = start_a + args.n_calib
    print(f"[calib] slice A start={start_a}  B start={start_b}  "
          f"size={args.n_calib} each", flush=True)

    _, calib_loader_d2_a, _, subset_a = load_model_and_loader(
        args.n_calib, start_idx=start_a, **_load_kw)
    _, calib_loader_d2_b, _, subset_b = load_model_and_loader(
        args.n_calib, start_idx=start_b, **_load_kw)
    ids_a = {int(it["image_id"]) for it in subset_a}
    ids_b = {int(it["image_id"]) for it in subset_b}
    overlap = ids_a & ids_b
    if overlap:
        raise RuntimeError(f"[calib] cross-seed loaders are not disjoint: "
                           f"{len(overlap)} shared image_ids")
    print(f"[calib] disjoint OK: |A|={len(ids_a)} |B|={len(ids_b)} "
          f"first_a={sorted(ids_a)[:4]} first_b={sorted(ids_b)[:4]}",
          flush=True)

    calib_loader_a = BackboneCalibLoader(calib_loader_d2_a, model_sc, device,
                                         max_items=args.n_calib, tag="A")
    calib_loader_b = BackboneCalibLoader(calib_loader_d2_b, model_sc, device,
                                         max_items=args.n_calib, tag="B")

    # --- Comp kernel choice ---
    # use_head_aligned was defined above when we built sc_comp_sig.
    comp_factory = None
    if args.comp_mode == "sc":
        if use_head_aligned:
            comp_factory = make_head_aligned_comp_factory(
                args.comp_sc_prec, args.comp_sc_mode, n_heads=args.n_heads)
            print(f"[comp] head-aligned SC compensator: sc_prec="
                  f"{args.comp_sc_prec} mode={args.comp_sc_mode} "
                  f"n_heads={args.n_heads}", flush=True)
        else:
            comp_factory = make_sc_comp_factory(args.comp_sc_prec,
                                                args.comp_sc_mode)
            print(f"[comp] full-width SC compensator: sc_prec="
                  f"{args.comp_sc_prec} mode={args.comp_sc_mode}", flush=True)
    else:
        comp_factory = make_fp_factory()
        print(f"[comp] FP compensator (debug)", flush=True)

    # --- Cross-seed calibration ---
    last_tau = (args.last_block_cos_threshold
                if args.last_block_cos_threshold > args.cos_threshold else None)
    print(f"[calib] τ={args.cos_threshold}  last_τ={last_tau}  "
          f"norm_floor={args.norm_floor}  lookahead_veto={args.lookahead_veto}",
          flush=True)
    print(f"[calib] starting QwT cross-seed calibration "
          f"(n_calib={args.n_calib} per side)", flush=True)
    t_cal = time.time()
    blocks_fp = list(model_fp.backbone.net.blocks)
    blocks_sc = model_sc.backbone.net.blocks
    # Calib also benefits from det-tuned kernel tiles — the per-block SC
    # forwards go through the same SCMatMul / SCLinear paths as eval.
    with det_kernel_tuning():
      report = calibrate_qwt(
        model_fp=model_fp.backbone.net,
        model_sc=model_sc.backbone.net,
        blocks_fp=blocks_fp,
        blocks_sc_container=blocks_sc,
        calib_loader_a=calib_loader_a,
        calib_loader_b=calib_loader_b,
        device=device,
        n_calib=args.n_calib,
        ridge=args.ridge,
        start_block=args.start_block,
        fwd_chunk=args.fwd_chunk,
        cos_threshold=args.cos_threshold,
        norm_floor=args.norm_floor,
        last_block_cos_threshold=last_tau,
        lookahead_veto=args.lookahead_veto,
        comp_factory=comp_factory,
      )
    calib_s = time.time() - t_cal
    prof["calib_total"] = calib_s
    prof["per_block_calib"] = sum(float(r.get("dt_s", 0.0)) for r in report)
    # collect_x0 (A+B) is the calib-phase wall not attributed to per-block.
    prof["collect_x0_total"] = max(0.0, calib_s - prof["per_block_calib"])
    print(f"[calib] done in {calib_s:.1f}s", flush=True)

    # Cross-check: log the SNG pool entries that the run actually populated.
    # On EVA-ViTDet the per-head QK matmul should populate (88, sc_prec); the
    # head-aligned comp should reuse it — i.e. (88, sc_prec) appears once.
    try:
        from sc_integration.sc_linear import _CFG_CACHE as _SC_CACHE
        cache_keys = sorted(list(_SC_CACHE.keys()))
        print(f"[comp] _CFG_CACHE.keys(): {cache_keys}", flush=True)
    except Exception as e:
        print(f"[comp] could not introspect _CFG_CACHE: {e}", flush=True)
        cache_keys = None

    n_enabled = sum(1 for r in report if r.get("enabled", False))
    n_blocks_total = len(report)
    en_blocks = [r["block"] for r in report if r["enabled"]]
    cos_admitted = [r["cos_ab"] for r in report if r["enabled"]]
    cos_rejected = [r["cos_ab"] for r in report if not r["enabled"]]
    print(f"[gate] admitted={n_enabled}/{n_blocks_total}  blocks={en_blocks}", flush=True)
    if cos_admitted:
        print(f"[gate] cos range admitted=[{min(cos_admitted):+.3f}, "
              f"{max(cos_admitted):+.3f}]", flush=True)
    if cos_rejected:
        print(f"[gate] cos range rejected=[{min(cos_rejected):+.3f}, "
              f"{max(cos_rejected):+.3f}]", flush=True)

    # --- Evaluate compensated model ---
    print("[eval] SC + QwT compensation", flush=True)
    _, _, evaluator_sc2, _ = load_model_and_loader(
        args.n_eval, start_idx=args.eval_start_idx, **_load_kw)
    t0 = time.time()
    res_sc_comp = run_coco_eval(
        model_sc, evaluator_sc2, subset_eval, out_dir_base / "sc_comp",
        cfg_sig=sc_comp_sig, resume=args.resume,
        save_every=args.save_every, phase="sc_comp", size=args.size)
    prof["comp_eval"] = time.time() - t0
    print(f"  bbox AP={res_sc_comp['bbox']['AP']:.2f}  "
          f"segm AP={res_sc_comp['segm']['AP']:.2f}  "
          f"({prof['comp_eval']:.0f}s)", flush=True)

    # --- Save report ---
    out = {
        "config": {
            "sc_prec": args.sc_prec,
            "sc_ops": args.sc_ops,
            "sc_ops_per_block_json": args.sc_ops_per_block_json,
            "n_calib": args.n_calib, "n_eval": args.n_eval,
            "calib_seed": args.calib_seed,
            "calib_seed_b": args.calib_seed_b,
            "ridge": args.ridge, "start_block": args.start_block,
            "cos_threshold": args.cos_threshold,
            "last_block_cos_threshold": args.last_block_cos_threshold,
            "norm_floor": args.norm_floor,
            "lookahead_veto": bool(args.lookahead_veto),
            "comp_mode": args.comp_mode,
            "comp_sc_prec": args.comp_sc_prec,
            "comp_sc_mode": args.comp_sc_mode,
            "head_aligned": use_head_aligned,
            "n_heads": args.n_heads if use_head_aligned else None,
            "fwd_chunk": args.fwd_chunk,
            "calib_slice_a_start": start_a,
            "calib_slice_b_start": start_b,
            "sc_cfg_cache_keys": [list(k) for k in cache_keys] if cache_keys else None,
            "mp": mp_args_to_dict(args),
        },
        "calib": {
            "per_block": report,
            "elapsed_s": calib_s,
            "n_enabled_blocks": n_enabled,
        },
        "results": {
            "sc_raw": res_sc_raw,
            "sc_comp": res_sc_comp,
        },
    }
    prof["total"] = time.time() - t_total0
    out["profile"] = prof

    def _fmt(k):
        v = prof.get(k)
        return f"{k}={v:.1f}s" if v is not None else f"{k}=NA"
    print("[profile] " + "  ".join(_fmt(k) for k in (
        "model_load_fp", "model_load_sc", "patch", "raw_sc_eval",
        "collect_x0_total", "per_block_calib", "calib_total",
        "comp_eval", "total")), flush=True)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: None)
    print(f"[done] wrote {args.out_json}", flush=True)

    if res_sc_raw:
        d_bbox = res_sc_comp['bbox']['AP'] - res_sc_raw['bbox']['AP']
        d_segm = res_sc_comp['segm']['AP'] - res_sc_raw['segm']['AP']
        print(f"\n=== SUMMARY ===")
        print(f"  SC raw      : bbox AP={res_sc_raw['bbox']['AP']:.2f}  "
              f"segm AP={res_sc_raw['segm']['AP']:.2f}")
        print(f"  SC + QwT    : bbox AP={res_sc_comp['bbox']['AP']:.2f}  "
              f"segm AP={res_sc_comp['segm']['AP']:.2f}")
        print(f"  Δ bbox: {d_bbox:+.2f}   Δ segm: {d_segm:+.2f}")
        print(f"  admitted blocks: {n_enabled}/{n_blocks_total}")


if __name__ == "__main__":
    main()
