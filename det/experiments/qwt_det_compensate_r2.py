"""Legacy r²-gate det reference driver — for ONE-OFF comparison only.

This driver imports the legacy ``calibrate_qwt`` from the c3d99d5 worktree
(``/tmp/qwt_r2/QwT-vit-sc``) which uses the old single-batch + ``min_r2``
admission rule. It is intentionally kept off the production sweep harness
and is here only to confirm the user's recollection that ``min_r2 > 0.3``
did not collapse on det. Apples-to-apples with the current smoke runs:

  * head-aligned SC compensator (current code)
  * same sc_patch_eva schedule path
  * same n_calib / n_eval / sc_prec / comp_sc_prec
  * single-batch calib (legacy behavior)

DO NOT invoke from production. See docs/DET_GATE_TUNING_PROMPT.md Step 1.
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
# IMPORTANT: legacy QwT-SC worktree first on path so its qwt_sc resolves,
# NOT the current submodule.
_LEGACY_QWT = Path("/home/allenjin/qwt_r2/QwT-vit-sc")
if not _LEGACY_QWT.exists():
    raise SystemExit(
        f"legacy worktree missing: {_LEGACY_QWT}. Run "
        f"`cd {_REPO}/third_party/QwT-SC && git worktree add /home/allenjin/qwt_r2 c3d99d5`"
    )
sys.path.insert(0, str(_LEGACY_QWT))
sys.path.insert(0, str(_DET))
sys.path.insert(0, str(_REPO))

from eval_common import load_model_and_loader, set_evaluator_output_dir
from sc_patch import sc_patch_eva, SC_OP_NAMES
from sc_integration.sc_linear import SCLinear
from sc_integration.head_aligned_comp import HeadAlignedSCLinear

# Resolve legacy qwt_sc explicitly so we don't accidentally import the current one.
import importlib
qwt_sc_legacy = importlib.import_module("qwt_sc")
calibrate_qwt = qwt_sc_legacy.calibrate_qwt


def make_head_aligned_comp_factory(sc_prec: int, mode: str = "bipolar",
                                   n_heads: int = 16):
    def factory(W, b):
        D_in, D_out = W.shape
        layer = nn.Linear(D_in, D_out, bias=True)
        with torch.no_grad():
            layer.weight.copy_(W.t().contiguous())
            layer.bias.copy_(b)
        return HeadAlignedSCLinear(layer, sc_prec=sc_prec, mode=mode, n_heads=n_heads)
    return factory


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
    names = [s.strip() for s in spec.split(",") if s.strip()]
    out = {n: 1 for n in names}
    if "proj" in out:
        v = out.pop("proj")
        out.setdefault("qkv_proj", v)
        out.setdefault("out_proj", v)
    return out


class BackboneCalibLoader:
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
    p.add_argument("--sc_ops", type=str, default="qk,av,proj")
    p.add_argument("--sc_ops_per_block_json", type=str, default="")
    p.add_argument("--n_calib", type=int, default=16)
    p.add_argument("--n_eval", type=int, default=10)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--start_block", type=int, default=0)
    p.add_argument("--min_r2", type=float, default=0.3,
                   help="Legacy r²-gate threshold. 0.3 is the user's hint.")
    p.add_argument("--last_block_r2_threshold", type=float, default=0.5)
    p.add_argument("--lookahead_veto", action="store_true")
    p.add_argument("--fwd_chunk", type=int, default=2)
    p.add_argument("--comp_mode", choices=["fp", "sc"], default="sc")
    p.add_argument("--comp_sc_prec", type=int, default=8)
    p.add_argument("--comp_sc_mode", choices=["bipolar", "unipolar"], default="bipolar")
    p.add_argument("--head_aligned", action="store_true")
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--out_tag", default="ref_r2_03")
    p.add_argument("--out_json", default="results/gate_tuning/ref_r2_03_p7.json")
    p.add_argument("--d2_datasets", type=str, default="")
    p.add_argument("--ckpt", type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[env] torch {torch.__version__} cuda {torch.version.cuda} "
          f"dev={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[ref] using LEGACY r² gate from {_LEGACY_QWT}", flush=True)

    if args.sc_ops_per_block_json:
        with open(args.sc_ops_per_block_json) as f:
            sched = json.load(f)
        cfg_desc = f"schedule from {args.sc_ops_per_block_json}"
    else:
        sched = parse_sc_ops(args.sc_ops)
        cfg_desc = f"sc_ops={args.sc_ops}"
    print(f"[cfg] sc_prec={args.sc_prec}  {cfg_desc}  "
          f"n_calib={args.n_calib}  n_eval={args.n_eval}", flush=True)

    _load_kw = dict(d2_datasets=args.d2_datasets, ckpt=args.ckpt)
    print("[model] loading FP reference", flush=True)
    model_fp, _, _, _ = load_model_and_loader(args.n_eval, **_load_kw)
    model_fp.eval()

    print("[model] loading SC model (fresh copy)", flush=True)
    model_sc, test_loader_sc, evaluator_sc, _ = load_model_and_loader(args.n_eval, **_load_kw)
    model_sc.eval()

    print(f"[sc] patching model_sc (sc_prec={args.sc_prec})", flush=True)
    stats = sc_patch_eva(model_sc, sc_prec=args.sc_prec, sc_ops_per_block=sched)
    print(f"[sc] patched: {stats}", flush=True)

    out_dir_base = _DET / "results" / args.out_tag
    out_dir_base.mkdir(parents=True, exist_ok=True)
    print("[eval] SC baseline (no comp)", flush=True)
    t0 = time.time()
    res_sc_raw = run_coco_eval(model_sc, test_loader_sc, evaluator_sc,
                               out_dir_base / "sc_raw")
    print(f"  bbox AP={res_sc_raw['bbox']['AP']:.2f}  "
          f"segm AP={res_sc_raw['segm']['AP']:.2f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    # Legacy single-batch calib loader.
    _, calib_loader_d2, _, _ = load_model_and_loader(args.n_calib, **_load_kw)
    calib_loader = BackboneCalibLoader(calib_loader_d2, model_sc, device,
                                       max_items=args.n_calib)

    if args.comp_mode == "sc":
        if args.head_aligned:
            comp_factory = make_head_aligned_comp_factory(
                args.comp_sc_prec, args.comp_sc_mode, n_heads=args.n_heads)
            print(f"[comp] head-aligned SC compensator", flush=True)
        else:
            comp_factory = make_sc_comp_factory(args.comp_sc_prec, args.comp_sc_mode)
    else:
        comp_factory = None

    print(f"[calib] starting LEGACY single-batch r²>{args.min_r2} calibration "
          f"(n_calib={args.n_calib})", flush=True)
    t_cal = time.time()
    blocks_fp = list(model_fp.backbone.net.blocks)
    blocks_sc = model_sc.backbone.net.blocks
    last_r2 = (args.last_block_r2_threshold
               if args.last_block_r2_threshold > args.min_r2 else None)
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
        last_block_r2_threshold=last_r2,
        lookahead_veto=args.lookahead_veto,
        fwd_chunk=args.fwd_chunk,
        comp_factory=comp_factory,
    )
    calib_s = time.time() - t_cal
    print(f"[calib] done in {calib_s:.1f}s", flush=True)

    n_enabled = sum(1 for r in report if r.get("enabled", False))
    n_total = len(report)
    en_blocks = [r["block"] for r in report if r["enabled"]]
    print(f"[gate] admitted={n_enabled}/{n_total}  blocks={en_blocks}", flush=True)

    print("[eval] SC + r²-gate compensation", flush=True)
    _, _, evaluator_sc2, _ = load_model_and_loader(args.n_eval, **_load_kw)
    t0 = time.time()
    res_sc_comp = run_coco_eval(model_sc, test_loader_sc, evaluator_sc2,
                                out_dir_base / "sc_comp")
    print(f"  bbox AP={res_sc_comp['bbox']['AP']:.2f}  "
          f"segm AP={res_sc_comp['segm']['AP']:.2f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    out = {
        "config": {
            "gate": "legacy_r2",
            "sc_prec": args.sc_prec,
            "sc_ops": args.sc_ops,
            "sc_ops_per_block_json": args.sc_ops_per_block_json,
            "n_calib": args.n_calib, "n_eval": args.n_eval,
            "ridge": args.ridge, "start_block": args.start_block,
            "min_r2": args.min_r2,
            "last_block_r2_threshold": args.last_block_r2_threshold,
            "lookahead_veto": bool(args.lookahead_veto),
            "comp_mode": args.comp_mode,
            "comp_sc_prec": args.comp_sc_prec,
            "comp_sc_mode": args.comp_sc_mode,
            "head_aligned": bool(args.head_aligned),
            "n_heads": args.n_heads,
            "fwd_chunk": args.fwd_chunk,
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
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: None)
    print(f"[done] wrote {args.out_json}", flush=True)

    d_bbox = res_sc_comp['bbox']['AP'] - res_sc_raw['bbox']['AP']
    print(f"\n=== SUMMARY (legacy r²-gate ref) ===")
    print(f"  SC raw    : bbox AP={res_sc_raw['bbox']['AP']:.2f}")
    print(f"  SC + r²>{args.min_r2}: bbox AP={res_sc_comp['bbox']['AP']:.2f}")
    print(f"  Δ bbox: {d_bbox:+.2f}   admitted: {n_enabled}/{n_total}")


if __name__ == "__main__":
    main()
