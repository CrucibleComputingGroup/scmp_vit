"""QwT closed-form linear compensation for SC EVA-ViTDet.

Det-side analogue of ``cls/experiments/qwt_sc_compensation.py``. Fits a
per-block residual ``x @ W + b`` on a small calibration set, then wraps each
SC block with a ``CompensationBlock`` and evaluates on COCO.

Usage::

    python experiments/qwt_det_compensate.py \\
        --sc_prec 7 --sc_ops qk,av,proj \\
        --n_calib 16 --n_eval 100 \\
        --out_json results/qwt_det_int7_qkavproj.json
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
for p in (_DET, _QWT_LIB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_common import load_model_and_loader, set_evaluator_output_dir
from sc_patch import sc_patch_eva, SC_OP_NAMES
from sc_integration.sc_linear import SCLinear
from qwt_sc import calibrate_qwt
from qwt_sc.compensation import CompensationBlock


def make_sc_comp_factory(sc_prec: int, mode: str = "bipolar"):
    def factory(W, b):
        D_in, D_out = W.shape
        layer = nn.Linear(D_in, D_out, bias=True)
        with torch.no_grad():
            layer.weight.copy_(W.t().contiguous())
            layer.bias.copy_(b)
        return SCLinear(layer, sc_prec=sc_prec, mode=mode)
    return factory


def parse_sc_ops(spec: str) -> dict:
    """Turn 'qk,av,proj,mlp_fc1' into a 40-block sched dict ready for sc_patch_eva."""
    names = [s.strip() for s in spec.split(",") if s.strip()]
    out = {n: 1 for n in names}  # sc_patch normalizes dict[op, scalar] → per-block
    if "proj" in out:
        v = out.pop("proj")
        out.setdefault("qkv_proj", v)
        out.setdefault("out_proj", v)
    return out


class BackboneCalibLoader:
    """Yields (image_tensor,) 1-tuples from a D2 test loader, using the
    model's preprocess_image to get a normalized (B,3,H,W) float tensor."""

    def __init__(self, d2_loader, model_for_preproc, device, max_items):
        self._loader = d2_loader
        self._pp = model_for_preproc
        self._device = device
        self._max = max_items

    def __iter__(self):
        n = 0
        for batched_inputs in self._loader:
            images = self._pp.preprocess_image(batched_inputs).tensor.to(self._device)
            yield (images,)
            n += images.size(0)
            if n >= self._max:
                return


@torch.no_grad()
def run_coco_eval(model, test_loader, evaluator, out_dir):
    from detectron2.evaluation import inference_on_dataset
    set_evaluator_output_dir(evaluator, out_dir)
    return inference_on_dataset(model, test_loader, evaluator)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sc_prec", type=int, default=7)
    p.add_argument("--sc_ops", type=str, default="qk,av,proj",
                   help="Comma-separated subset of {qk,av,proj,mlp_fc1,mlp_fc2}.")
    p.add_argument("--sc_ops_per_block_json", type=str, default="",
                   help="Optional per-(op, block) schedule JSON; overrides --sc_ops.")
    p.add_argument("--n_calib", type=int, default=16)
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--start_block", type=int, default=0)
    p.add_argument("--min_r2", type=float, default=0.0,
                   help="Disable comp on blocks whose calib r2 <= this threshold.")
    p.add_argument("--fwd_chunk", type=int, default=2,
                   help="Det images are 1024², so use tiny chunks.")
    p.add_argument("--avg_sc_draws", type=int, default=1)
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--comp_mode", choices=["fp", "sc"], default="fp")
    p.add_argument("--comp_sc_prec", type=int, default=8)
    p.add_argument("--comp_sc_mode", choices=["bipolar", "unipolar"], default="bipolar")
    p.add_argument("--out_tag", default="qwt_det")
    p.add_argument("--out_json", default="results/qwt_det/run.json")
    p.add_argument("--d2_datasets", type=str, default="",
                   help="DETECTRON2_DATASETS root (default: GreatLakes shared_data).")
    p.add_argument("--ckpt", type=str, default="",
                   help="Path to EVA-ViTDet checkpoint (default: GreatLakes shared_data).")
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[env] torch {torch.__version__} cuda {torch.version.cuda} "
          f"dev={torch.cuda.get_device_name(0)}", flush=True)

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

    # --- Load both models. Use same n_eval loader, we'll slice first n_calib for calib. ---
    _load_kw = dict(d2_datasets=args.d2_datasets, ckpt=args.ckpt)
    print("[model] loading FP reference", flush=True)
    model_fp, test_loader_fp, evaluator_fp, _ = load_model_and_loader(args.n_eval, **_load_kw)
    model_fp.eval()

    print("[model] loading SC model (fresh copy)", flush=True)
    model_sc, test_loader_sc, evaluator_sc, _ = load_model_and_loader(args.n_eval, **_load_kw)
    model_sc.eval()

    print(f"[sc] patching model_sc (sc_prec={args.sc_prec})", flush=True)
    stats = sc_patch_eva(model_sc, sc_prec=args.sc_prec, sc_ops_per_block=sched)
    print(f"[sc] patched: {stats}", flush=True)

    # --- SC baseline (optional) ---
    out_dir_base = _DET / "results" / args.out_tag
    out_dir_base.mkdir(parents=True, exist_ok=True)
    res_sc_raw = None
    if not args.skip_baseline:
        print("[eval] SC baseline (no comp)", flush=True)
        t0 = time.time()
        res_sc_raw = run_coco_eval(model_sc, test_loader_sc, evaluator_sc,
                                    out_dir_base / "sc_raw")
        print(f"  bbox AP={res_sc_raw['bbox']['AP']:.2f}  "
              f"segm AP={res_sc_raw['segm']['AP']:.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # --- Calibration ---
    # Use a fresh loader (not the test_loader_sc we just consumed) via reload.
    calib_model, calib_loader_d2, _, _ = load_model_and_loader(args.n_calib, **_load_kw)
    del calib_model  # only need the loader
    calib_loader = BackboneCalibLoader(calib_loader_d2, model_sc, device,
                                        max_items=args.n_calib)

    print(f"[calib] starting QwT calibration (n_calib={args.n_calib})", flush=True)
    t0 = time.time()
    blocks_fp = list(model_fp.backbone.net.blocks)
    blocks_sc = model_sc.backbone.net.blocks
    # calibrate_qwt invokes `model_sc(imgs)` to populate its block-0 hook.
    # Pass .backbone.net directly — it accepts (B, 3, H, W) tensors.
    comp_factory = None
    if args.comp_mode == "sc":
        comp_factory = make_sc_comp_factory(args.comp_sc_prec, args.comp_sc_mode)
        print(f"[comp] SC compensator: sc_prec={args.comp_sc_prec} "
              f"mode={args.comp_sc_mode}", flush=True)
    else:
        print(f"[comp] FP compensator", flush=True)
    report = calibrate_qwt(
        model_fp=model_fp.backbone.net,
        model_sc=model_sc.backbone.net,
        blocks_fp=blocks_fp,
        blocks_sc_container=blocks_sc,
        calib_loader=calib_loader,
        device=device,
        n_calib=args.n_calib,
        ridge=args.ridge,
        start_block=args.start_block,
        min_r2=args.min_r2,
        fwd_chunk=args.fwd_chunk,
        avg_sc_draws=args.avg_sc_draws,
        comp_factory=comp_factory,
    )
    calib_s = time.time() - t0
    print(f"[calib] done in {calib_s:.1f}s", flush=True)

    # --- Evaluate compensated model ---
    print("[eval] SC + QwT compensation", flush=True)
    # reload evaluator (it has state from previous run)
    _, _, evaluator_sc2, _ = load_model_and_loader(args.n_eval, **_load_kw)
    t0 = time.time()
    res_sc_comp = run_coco_eval(model_sc, test_loader_sc, evaluator_sc2,
                                 out_dir_base / "sc_comp")
    print(f"  bbox AP={res_sc_comp['bbox']['AP']:.2f}  "
          f"segm AP={res_sc_comp['segm']['AP']:.2f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    # --- Save report ---
    out = {
        "config": {
            "sc_prec": args.sc_prec,
            "sc_ops": args.sc_ops,
            "sc_ops_per_block_json": args.sc_ops_per_block_json,
            "n_calib": args.n_calib, "n_eval": args.n_eval,
            "ridge": args.ridge, "start_block": args.start_block,
            "min_r2": args.min_r2,
            "avg_sc_draws": args.avg_sc_draws,
            "fwd_chunk": args.fwd_chunk,
        },
        "calib": {"per_block": report, "elapsed_s": calib_s},
        "results": {
            "sc_raw": res_sc_raw,
            "sc_comp": res_sc_comp,
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: None)
    print(f"[done] wrote {args.out_json}", flush=True)

    if res_sc_raw:
        d = res_sc_comp['bbox']['AP'] - res_sc_raw['bbox']['AP']
        print(f"\n=== SUMMARY ===")
        print(f"  SC raw      : bbox AP={res_sc_raw['bbox']['AP']:.2f}  "
              f"segm AP={res_sc_raw['segm']['AP']:.2f}")
        print(f"  SC + QwT    : bbox AP={res_sc_comp['bbox']['AP']:.2f}  "
              f"segm AP={res_sc_comp['segm']['AP']:.2f}")
        print(f"  Δ bbox: {d:+.2f}")


if __name__ == "__main__":
    main()
