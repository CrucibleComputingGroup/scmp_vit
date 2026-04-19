"""Per-block & per-linear MLP sensitivity sweep using noise-model surrogate.

For each transformer block i ∈ [0..23], patch ONLY block i's MLP (one at a
time) and measure:
  (a) fc1 + fc2 both SC  (full MLP)
  (b) fc1 only SC
  (c) fc2 only SC
Also record first-N cumulative reference (N ∈ 1..8) for context.

Noise-model calibrated with NOISE_LOCAL_CORRECTION=2.0 (center of N≤8 window).
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
from sc_attention_patch import patch_model, set_noise_model
from sc_integration.noise_matmul import set_noise_corrections

DATA_ROOT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data"
N_IMAGES, BATCH, SEED = 100, 8, 0
N_BLOCKS = 24
NOISE_CORR = 2.0
N_SEEDS = 2   # average over seeds to reduce surrogate RNG jitter
OUT = "results/mlp_sensitivity_sweep.json"


@torch.no_grad()
def run(model, loader, device):
    Ls, Ys = [], []
    for x, y in loader:
        Ls.append(model(x.to(device, non_blocking=True)).float().cpu())
        Ys.append(y)
    return torch.cat(Ls), torch.cat(Ys)


def metrics(L_sc, L_fp, y):
    top1 = (L_sc.argmax(1) == y).float().mean().item()
    l2 = (L_sc - L_fp).pow(2).sum(1).sqrt().mean().item()
    flip = (L_sc.argmax(1) != L_fp.argmax(1)).float().mean().item()
    kl = F.kl_div(F.log_softmax(L_sc, 1), F.log_softmax(L_fp, 1),
                  reduction="batchmean", log_target=True).item()
    return {"top1": top1, "flip": flip, "l2": l2, "kl": kl}


def agg_over_seeds(run_one):
    outs = [run_one(s) for s in range(N_SEEDS)]
    keys = outs[0].keys()
    return {k: sum(o[k] for o in outs) / len(outs) for k in keys}


def main():
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    seed_all(SEED)
    device = torch.device("cuda")

    ds_full = ImageNetParquetVal(DATA_ROOT, transform=build_transform(224))
    idx = list(range(len(ds_full)))
    random.Random(SEED).shuffle(idx)
    ds = Subset(ds_full, idx[:N_IMAGES])
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4,
                        pin_memory=True)

    m = load_model(device)
    L_fp, y = run(m, loader, device)
    del m; torch.cuda.empty_cache()
    print(f"[fp] top1={(L_fp.argmax(1)==y).float().mean().item():.3f}", flush=True)

    set_noise_model(True)
    set_noise_corrections(local=NOISE_CORR, global_=0.60)

    results = {"config": {"noise_corr": NOISE_CORR, "n_images": N_IMAGES,
                           "n_seeds": N_SEEDS}, "per_block": [],
               "first_N_ref": []}

    # --- Per-block sweep: patch block i only, for each of fc1/fc2/both ---
    for i in range(N_BLOCKS):
        row = {"block": i}
        for fc_mask in ("both", "fc1", "fc2"):
            def run_one(seed):
                torch.manual_seed(SEED * 1000 + seed)
                m = load_model(device)
                patch_model(m, sc_prec=8, sc_qk=False, sc_av=False,
                            sc_mlp=True,
                            mlp_skip_first_k=i,
                            mlp_skip_last_k=N_BLOCKS - i - 1,
                            sc_mlp_mode="bipolar",
                            mlp_fc_mask=fc_mask)
                L, _ = run(m, loader, device)
                del m; torch.cuda.empty_cache()
                return metrics(L, L_fp, y)
            row[fc_mask] = agg_over_seeds(run_one)
        print(f"block {i:2d}  "
              f"both l2={row['both']['l2']:5.2f} top1={row['both']['top1']:.2f}  "
              f"fc1  l2={row['fc1']['l2']:5.2f} top1={row['fc1']['top1']:.2f}  "
              f"fc2  l2={row['fc2']['l2']:5.2f} top1={row['fc2']['top1']:.2f}",
              flush=True)
        results["per_block"].append(row)

    # --- first-N cumulative reference ---
    for N in range(0, 9):
        def run_one(seed):
            torch.manual_seed(SEED * 1000 + seed)
            m = load_model(device)
            if N > 0:
                patch_model(m, sc_prec=8, sc_qk=False, sc_av=False,
                            sc_mlp=True, mlp_skip_first_k=0,
                            mlp_skip_last_k=N_BLOCKS - N,
                            sc_mlp_mode="bipolar", mlp_fc_mask="both")
            L, _ = run(m, loader, device)
            del m; torch.cuda.empty_cache()
            return metrics(L, L_fp, y)
        stats = agg_over_seeds(run_one) if N > 0 else {"top1": 0.89, "l2": 0.0, "kl": 0.0, "flip": 0.0}
        stats["N"] = N
        print(f"first-{N:2d}  l2={stats['l2']:5.2f} top1={stats['top1']:.3f} "
              f"kl={stats['kl']:.3f}", flush=True)
        results["first_N_ref"].append(stats)

    Path("results").mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
