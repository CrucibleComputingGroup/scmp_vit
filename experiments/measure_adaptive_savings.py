"""Measure actual compute savings of adaptive MP on ViT cls path.

Runs one forward on N ImageNet images with instrumentation that captures every
adaptive_classify_rows call (per SCLinear × per forward pass), aggregates
per-op baseline vs. actual stoc_len, reports savings.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sc"))
sys.path.insert(0, str(REPO / "cls"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=32, help="num images")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--ops", default="qkv_proj,out_proj",
                   help="ops to apply adaptive MP + measure")
    p.add_argument("--levels", default="256,128",
                   help="stoc_len_levels csv descending")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--timesteps", default="0/1,5/10,9/10",
                   help="comma list of t/T pairs to sweep")
    p.add_argument("--out", default=str(REPO / "results" / "e2e" /
                                        "adaptive_savings.json"))
    args = p.parse_args()

    from imagenet_parquet import ImageNetParquetVal
    from torchvision import transforms
    from sc_attention_patch import patch_model
    from sc_integration.sc_linear import set_vit_timestep
    from sc_integration import mp_linear as _mpl
    from sc_integration.mp_linear import AdaptiveMPConfig

    device = torch.device("cuda")
    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
    ds = ImageNetParquetVal(
        "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data",
        transform=tfm)
    ds = Subset(ds, list(range(args.n)))
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=2)

    import os
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitl14_lc",
        source="github", trust_repo=True,
    ).to(device).eval()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    max_sl = max(levels)
    ops = [x.strip() for x in args.ops.split(",") if x.strip()]
    adaptive = AdaptiveMPConfig(
        stoc_len_levels=levels, alpha=args.alpha, beta=args.beta,
        enable_pruning=(0 in levels),
    )
    spec = {op: {"adaptive": adaptive} for op in ops}

    patch_model(
        model, sc_prec=8, sc_qk=True, sc_av=False, sc_mlp=False,
        sc_qkv_proj=("qkv_proj" in ops), sc_out_proj=("out_proj" in ops),
        sc_proj_mode="bipolar", linear_mp_spec=spec,
    )

    # Spy on adaptive_classify_rows to capture assignments.
    # captures: list of (operator, {sl: count}) per call
    captured: list[tuple[str, dict]] = []
    orig = _mpl.adaptive_classify_rows

    def spy(metric, timestep, total_timesteps, config, operator=None):
        a = orig(metric, timestep, total_timesteps, config, operator=operator)
        dist = {int(sl): int(rows.numel())
                for sl, rows in a.level_row_indices.items()}
        captured.append((operator or "unknown", dist))
        return a
    _mpl.adaptive_classify_rows = spy
    # sc_linear already imported classify_input_rows_adaptive at module load.
    # Re-import so our patched spy is visible.
    from sc_integration import sc_linear as _scl
    _scl.classify_input_rows_adaptive = _mpl.classify_input_rows_adaptive

    results = {}
    # Get ViT weight shapes for out_features (needed for compute = rows * out * sl).
    # For DINOv2 ViT-L/14: qkv_proj → out=3072, out_proj → out=1024.
    op_out_features = {"qkv_proj": 3072, "out_proj": 1024,
                       "mlp_fc1": 4096, "mlp_fc2": 1024}

    for tT in args.timesteps.split(","):
        t_str, T_str = tT.strip().split("/")
        t, T = int(t_str), int(T_str)
        set_vit_timestep(t, T)
        captured.clear()
        with torch.no_grad():
            for x, _y in loader:
                x = x.to(device, non_blocking=True)
                _ = model(x)
                break  # just one batch is plenty

        per_op_base = defaultdict(float)
        per_op_actual = defaultdict(float)
        total_base = 0.0
        total_actual = 0.0
        for op, dist in captured:
            n_rows = sum(dist.values())
            out_f = op_out_features.get(op, 1024)
            base = n_rows * out_f * max_sl
            actual = sum(sl * cnt for sl, cnt in dist.items()) * out_f
            per_op_base[op] += base
            per_op_actual[op] += actual
            total_base += base
            total_actual += actual
        prog = t / max(T - 1, 1)
        thr = min(args.alpha * prog + args.beta, 0.95)
        results[tT.strip()] = {
            "t": t, "T": T, "progress": round(prog, 3),
            "threshold": round(thr, 3),
            "total_baseline_ops": int(total_base),
            "total_actual_ops": int(total_actual),
            "savings_pct": round(
                100 * (1 - total_actual / max(total_base, 1)), 2),
            "per_op": {
                op: {
                    "baseline_ops": int(per_op_base[op]),
                    "actual_ops": int(per_op_actual[op]),
                    "savings_pct": round(
                        100 * (1 - per_op_actual[op] /
                               max(per_op_base[op], 1)), 2),
                }
                for op in per_op_base
            },
            "n_classify_calls": len(captured),
        }
        print(f"[{tT}] prog={prog:.2f} thr={thr:.3f} "
              f"savings={results[tT.strip()]['savings_pct']}% "
              f"(base={total_base:.2e}, actual={total_actual:.2e}) "
              f"calls={len(captured)}")
        for op, d in results[tT.strip()]["per_op"].items():
            print(f"    {op}: savings={d['savings_pct']}%")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
