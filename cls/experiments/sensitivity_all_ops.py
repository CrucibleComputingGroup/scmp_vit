"""Per-block per-op sensitivity sweep using the calibrated noise surrogate.

For each (op ∈ {mlp_fc1, mlp_fc2, qk, av, proj}, block_idx ∈ [0..23]) combo,
SC *only* that single op in that single block; everything else FP. Measure:
  - top-1, top-5
  - logit L2 drift vs FP
  - softmax KL vs FP
  - top-1 prediction flip rate vs FP
Average over N_SEEDS to reduce surrogate RNG jitter.

Noise-surrogate correction set to NOISE_CORR (calibrated against real-SC MLP
logit_l2 in the N≤8 regime).
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
from sc_integration.noise_matmul import set_noise_corrections

DATA_ROOT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data"
N_IMAGES, BATCH, SEED = 100, 8, 0
N_BLOCKS = 24
N_SEEDS = 2
NOISE_CORR = 2.0
OUT = "results/sensitivity_all_ops.json"


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
        "flip": (L_sc.argmax(1) != L_fp.argmax(1)).float().mean().item(),
        "l2":   (L_sc - L_fp).pow(2).sum(1).sqrt().mean().item(),
        "kl":   F.kl_div(F.log_softmax(L_sc, 1), F.log_softmax(L_fp, 1),
                         reduction="batchmean", log_target=True).item(),
    }


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
                           "n_blocks": N_BLOCKS, "n_seeds": N_SEEDS},
               "grid": []}

    t_sweep = time.time()
    for op in SC_OP_NAMES:
        for bi in range(N_BLOCKS):
            seed_metrics = []
            for s in range(N_SEEDS):
                torch.manual_seed(SEED * 1000 + s)
                spec = {k: [0] * N_BLOCKS for k in SC_OP_NAMES}
                spec[op][bi] = 1
                m = load_model(device)
                patch_model(m, sc_prec=8, sc_mlp_mode="bipolar",
                            sc_proj_mode="bipolar", sc_ops_per_block=spec)
                L, _ = run(m, loader, device)
                seed_metrics.append(metrics(L, L_fp, y))
                del m; torch.cuda.empty_cache()
            keys = seed_metrics[0].keys()
            agg = {k: sum(d[k] for d in seed_metrics) / len(seed_metrics)
                   for k in keys}
            row = {"op": op, "block": bi, **agg}
            results["grid"].append(row)
        print(f"[{op:8s}] l2 min/max = "
              f"{min(r['l2'] for r in results['grid'] if r['op']==op):.2f}"
              f" / {max(r['l2'] for r in results['grid'] if r['op']==op):.2f}",
              flush=True)
    print(f"[sweep] total {time.time()-t_sweep:.0f}s", flush=True)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {OUT}", flush=True)

    # Print a compact text matrix
    print("\nlogit_l2 by (op, block):")
    hdr = "op \\ block  " + " ".join(f"{b:>5d}" for b in range(N_BLOCKS))
    print(hdr)
    for op in SC_OP_NAMES:
        row = [r for r in results["grid"] if r["op"] == op]
        vals = " ".join(f"{r['l2']:>5.2f}" for r in row)
        print(f"{op:<12s}{vals}")


if __name__ == "__main__":
    main()
