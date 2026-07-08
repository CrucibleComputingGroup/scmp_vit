"""Quick exploration of per-block r² and RMSE across SC configs.

Loads FP + SC model once per config, runs QwT-style calibration, dumps
per-block {r2, raw_rmse, after_rmse, enabled} into a CSV, skips eval.
Goal: understand how r²/RMSE depend on SC intensity and block depth.
"""
from __future__ import annotations
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

from eval_common import load_model_and_loader
from sc_patch import sc_patch_eva
from qwt_sc import calibrate_qwt


class Loader:
    def __init__(self, d2_loader, pp, device, max_items):
        self._l = d2_loader; self._pp = pp; self._d = device; self._m = max_items
    def __iter__(self):
        n = 0
        for bi in self._l:
            t = self._pp.preprocess_image(bi).tensor.to(self._d)
            yield (t,)
            n += t.size(0)
            if n >= self._m: return


def run_one(sc_prec, sc_ops_spec, tag, n_calib=8, device=None):
    print(f"\n========== {tag}  (sc_prec={sc_prec}, ops={sc_ops_spec}) ==========")
    if device is None:
        device = torch.device("cuda")

    print("[model] loading FP")
    model_fp, loader, _, _ = load_model_and_loader(n_calib)
    model_fp.eval()
    print("[model] loading SC")
    model_sc, _, _, _ = load_model_and_loader(n_calib)
    model_sc.eval()

    if isinstance(sc_ops_spec, dict):
        sched = sc_ops_spec
    else:
        sched = {n: 1 for n in sc_ops_spec.split(",")}
        if "proj" in sched:
            v = sched.pop("proj")
            sched.setdefault("qkv_proj", v)
            sched.setdefault("out_proj", v)
    stats = sc_patch_eva(model_sc, sc_prec=sc_prec, sc_ops_per_block=sched)
    print(f"[sc] patched={stats}")

    blocks_fp = list(model_fp.backbone.net.blocks)
    blocks_sc = model_sc.backbone.net.blocks

    t0 = time.time()
    report = calibrate_qwt(
        model_fp=model_fp.backbone.net,
        model_sc=model_sc.backbone.net,
        blocks_fp=blocks_fp,
        blocks_sc_container=blocks_sc,
        calib_loader=Loader(loader, model_sc, device, n_calib),
        device=device, n_calib=n_calib, ridge=1e-2, fwd_chunk=2, avg_sc_draws=1,
    )
    dt = time.time() - t0
    print(f"[done] {dt:.1f}s")

    return {"tag": tag, "sc_prec": sc_prec, "ops": str(sc_ops_spec),
            "report": report, "elapsed_s": dt}


def main():
    device = torch.device("cuda")
    configs = [
        (7, "qk", "int7_qk"),
        (7, "qk,av,proj", "int7_qkavproj"),
        (7, "qk,av,proj,mlp_fc1", "int7_qkavprojfc1"),
    ]
    out = []
    for sc_prec, ops, tag in configs:
        res = run_one(sc_prec, ops, tag, n_calib=8, device=device)
        out.append(res)
        # Ends on rolling summary after each
        rs = res["report"]
        print(f"\n[summary {tag}] "
              f"r² min/med/max = "
              f"{min(r['r2'] for r in rs):+.3f} / "
              f"{sorted([r['r2'] for r in rs])[len(rs)//2]:+.3f} / "
              f"{max(r['r2'] for r in rs):+.3f}  "
              f"| after/raw rmse ratio median = "
              f"{sorted([r['rmse_after']/r['rmse_before'] for r in rs])[len(rs)//2]:.3f}")

    # Side-by-side per-block table
    out_path = _DET / "results" / "qwt_det" / "r2_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"configs": [{"tag": o["tag"], "sc_prec": o["sc_prec"],
                                 "ops": o["ops"],
                                 "per_block": o["report"]} for o in out]}, f,
                  indent=2, default=lambda x: None)
    print(f"\n[saved] {out_path}")

    # Print table
    print("\n\n=== PER-BLOCK r²   (rows: config, cols: block) ===")
    header = "     " + "  ".join(f"b{i:02d}" for i in range(0, 40, 5))
    print(header)
    for o in out:
        row = f"{o['tag']:>22s}  " + "  ".join(
            f"{o['report'][i]['r2']:+.2f}" for i in range(0, 40, 5))
        print(row)

    print("\n=== PER-BLOCK raw_rmse / after_rmse ratio (lower = comp reduces error more) ===")
    print(header)
    for o in out:
        row = f"{o['tag']:>22s}  " + "  ".join(
            f"{o['report'][i]['rmse_after']/o['report'][i]['rmse_before']:.2f}"
            for i in range(0, 40, 5))
        print(row)


if __name__ == "__main__":
    main()
