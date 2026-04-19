"""Smoke: verify AdaptiveMPConfig.target_fractions produces the intended
per-level row distribution on real ViT-L/14 proj activations.

Runs a short forward on N imgs with adaptive MP configured with
target_fractions, captures every adaptive_classify_rows call, checks that
the realized fractions are within ±1 row of the target (sort-based bucketing
is exact up to count rounding).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sc"))
sys.path.insert(0, str(REPO / "cls"))


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--levels", default="256,128,64,32")
    p.add_argument("--fractions", default="0.05,0.35,0.40,0.20")
    p.add_argument("--ops", default="qkv_proj,out_proj")
    p.add_argument("--out_json",
                   default=str(REPO / "results" / "smoke" /
                               "custom_fractions_smoke.json"))
    p.add_argument("--tolerance", type=float, default=0.01,
                   help="allowed abs error between realized and target "
                        "fractions; rounding may drift by 1 row / bin.")
    args = p.parse_args()

    os.environ.setdefault("XFORMERS_DISABLED", "1")
    torch.manual_seed(0)

    from imagenet_parquet import ImageNetParquetVal
    from sc_attention_patch import patch_model
    from sc_integration.sc_linear import set_vit_timestep
    from sc_integration import mp_linear as _mpl
    from sc_integration.mp_linear import AdaptiveMPConfig

    levels = [int(x) for x in args.levels.split(",")]
    fractions = [float(x) for x in args.fractions.split(",")]
    ops = [x.strip() for x in args.ops.split(",")]

    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    ds = Subset(ImageNetParquetVal(
        "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data",
        transform=tfm), list(range(args.n)))
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_lc",
                           source="github", trust_repo=True).to(device).eval()

    adap = AdaptiveMPConfig(
        stoc_len_levels=levels, alpha=0.3, beta=0.05,
        enable_pruning=(0 in levels),
        target_fractions=fractions)
    spec = {op: {"adaptive": adap} for op in ops}
    patch_model(model, sc_prec=8, sc_qk=True, sc_av=False,
                sc_qkv_proj=("qkv_proj" in ops),
                sc_out_proj=("out_proj" in ops),
                linear_mp_spec=spec)

    captured = defaultdict(lambda: defaultdict(int))
    orig = _mpl.adaptive_classify_rows
    def spy(metric, t, T, cfg, operator=None):
        a = orig(metric, t, T, cfg, operator=operator)
        for sl, rows in a.level_row_indices.items():
            captured[operator or "_"][int(sl)] += int(rows.numel())
        return a
    _mpl.adaptive_classify_rows = spy
    from sc_integration import sc_linear as _scl
    _scl.classify_input_rows_adaptive = _mpl.classify_input_rows_adaptive

    set_vit_timestep(9, 10)
    with torch.no_grad():
        for x, _ in loader:
            _ = model(x.to(device))
            break

    report = {"target_fractions": fractions, "levels": levels,
              "per_op": {}, "passed": True,
              "git_commit": _git_commit()}
    for op, dist in captured.items():
        tot = sum(dist.values())
        realized = {sl: dist[sl] / tot for sl in levels}
        target = dict(zip(levels, fractions))
        max_err = max(abs(realized[sl] - target[sl]) for sl in levels)
        ok = max_err <= args.tolerance
        report["per_op"][op] = {
            "total_rows": tot,
            "realized": realized,
            "max_err": max_err,
            "ok": ok,
        }
        if not ok:
            report["passed"] = False
        print(f"[{op}] rows={tot}")
        for sl in levels:
            print(f"  sl={sl:<4d}  target={target[sl]:.3f}  "
                  f"realized={realized[sl]:.4f}  err={realized[sl]-target[sl]:+.4f}")
        print(f"  max_err={max_err:.4f}  ok={ok}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSMOKE {'OK' if report['passed'] else 'FAIL'}  "
          f"wrote {out_path}")
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
