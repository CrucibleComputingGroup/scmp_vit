"""Operator-level sensitivity with REAL SC kernels (no noise surrogate).

For each op ∈ {mlp_fc1, mlp_fc2, qk, av, proj}, stochasticize ONLY that op
across ALL 24 blocks and measure against the FP baseline:
  - top-1 accuracy
  - logit L2 drift vs FP
  - softmax KL vs FP
  - top-1 prediction flip rate vs FP

Goal: identify which operator cannot tolerate uniform SC at sc_prec=8.

Uses the real SC Triton kernels (noise model OFF) — exactly the kernels used
in end-to-end eval.
"""
import os, sys, random, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]   # vit_sc/
CLS = HERE / "cls"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CLS))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from eval import build_transform, load_model, seed_all
from imagenet_parquet import ImageNetParquetVal
from sc_attention_patch import patch_model, set_noise_model, SC_OP_NAMES

DATA_ROOT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data"
N_IMAGES = 100
BATCH    = 8
SEED     = 0
SC_PREC  = 8
N_BLOCKS = 24
OUT      = str(CLS / "experiments" / "results" / "sensitivity_operator_real_sc.json")


@torch.no_grad()
def run(model, loader, device):
    Ls, Ys = [], []
    for x, y in loader:
        Ls.append(model(x.to(device, non_blocking=True)).float().cpu())
        Ys.append(y)
    return torch.cat(Ls), torch.cat(Ys)


def metrics(L_sc, L_fp, y):
    return {
        "top1": (L_sc.argmax(1) == y).float().mean().item(),
        "top5": L_sc.topk(5, 1).indices.eq(y.unsqueeze(1)).any(1).float().mean().item(),
        "flip": (L_sc.argmax(1) != L_fp.argmax(1)).float().mean().item(),
        "l2":   (L_sc - L_fp).pow(2).sum(1).sqrt().mean().item(),
        "kl":   F.kl_div(F.log_softmax(L_sc, 1), F.log_softmax(L_fp, 1),
                         reduction="batchmean", log_target=True).item(),
    }


def main():
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    seed_all(SEED)
    device = torch.device("cuda")

    print(f"[env] torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    ds_full = ImageNetParquetVal(DATA_ROOT, transform=build_transform(224))
    idx = list(range(len(ds_full)))
    random.Random(SEED).shuffle(idx)
    ds = Subset(ds_full, idx[:N_IMAGES])
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2,
                        pin_memory=True)
    print(f"[data] {N_IMAGES} images (seed={SEED}), batch={BATCH}", flush=True)

    # --- FP baseline ---
    t0 = time.time()
    m = load_model(device)
    L_fp, y = run(m, loader, device)
    del m; torch.cuda.empty_cache()
    fp_top1 = (L_fp.argmax(1) == y).float().mean().item()
    fp_top5 = L_fp.topk(5, 1).indices.eq(y.unsqueeze(1)).any(1).float().mean().item()
    print(f"[fp] top1={fp_top1:.3f} top5={fp_top5:.3f} ({time.time()-t0:.1f}s)",
          flush=True)

    # real SC: ensure noise surrogate is OFF
    set_noise_model(False)

    results = {
        "config": {
            "sc_prec": SC_PREC, "n_images": N_IMAGES, "batch": BATCH,
            "seed": SEED, "n_blocks": N_BLOCKS, "noise_model": False,
            "op_names": list(SC_OP_NAMES),
        },
        "fp": {"top1": fp_top1, "top5": fp_top5},
        "per_op": [],
    }

    t_sweep = time.time()
    for op in SC_OP_NAMES:
        torch.manual_seed(SEED)
        spec = {k: [0] * N_BLOCKS for k in SC_OP_NAMES}
        spec[op] = [1] * N_BLOCKS
        t1 = time.time()
        m = load_model(device)
        stats = patch_model(m, sc_prec=SC_PREC, sc_mlp_mode="bipolar",
                            sc_proj_mode="bipolar", sc_ops_per_block=spec)
        L, _ = run(m, loader, device)
        dt = time.time() - t1
        mt = metrics(L, L_fp, y)
        mt["dtop1"] = mt["top1"] - fp_top1
        row = {"op": op, "patched": stats, "elapsed_s": dt, **mt}
        results["per_op"].append(row)
        print(f"[{op:8s}] top1={mt['top1']:.3f} (Δ={mt['dtop1']:+.3f}) "
              f"top5={mt['top5']:.3f} l2={mt['l2']:.2f} kl={mt['kl']:.3f} "
              f"flip={mt['flip']:.3f} patched={stats} ({dt:.1f}s)",
              flush=True)
        del m; torch.cuda.empty_cache()

    results["elapsed_total_s"] = time.time() - t_sweep
    print(f"[sweep] total {results['elapsed_total_s']:.0f}s", flush=True)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {OUT}", flush=True)

    # Ranked summary
    ranked = sorted(results["per_op"], key=lambda r: -r["l2"])
    print("\nRanked by logit L2 drift (largest = most SC-sensitive):")
    print(f"{'op':<10s} {'top1':>6s} {'Δtop1':>7s} {'l2':>7s} {'kl':>7s} {'flip':>6s}")
    for r in ranked:
        print(f"{r['op']:<10s} {r['top1']:>6.3f} {r['dtop1']:>+7.3f} "
              f"{r['l2']:>7.2f} {r['kl']:>7.3f} {r['flip']:>6.3f}")


if __name__ == "__main__":
    main()
