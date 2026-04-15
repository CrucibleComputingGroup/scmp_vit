"""Add MLP-SC to first N transformer blocks and compare output vs FP.

For each N in BLOCK_COUNTS, SC-ifies the first N blocks' MLPs (fc1 + fc2,
bipolar symmetric quant, sc_prec=8) and measures:
  - top-1, top-5
  - logit L2 error vs FP
  - top-1 prediction flip rate vs FP
  - softmax KL vs FP (mean)
Attention stays FP everywhere.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from eval import build_transform, load_model, seed_all
from imagenet_parquet import ImageNetParquetVal
from sc_attention_patch import patch_model

DATA_ROOT = "./data/imagenet/data"
N_IMAGES = 100
BATCH = 8
SEED = 0
SC_PREC = 8
MODE = os.environ.get("MODE", "bipolar")
REGION = os.environ.get("REGION", "first")  # "first" | "last"
BLOCK_COUNTS = [0, 1, 2, 4, 8, 16, 24]
OUT = f"results/mlp_onebyone_{MODE}_{REGION}.json"


@torch.no_grad()
def run_model(model, loader, device):
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        all_logits.append(model(x).float().cpu())
        all_y.append(y)
    return torch.cat(all_logits), torch.cat(all_y)


def metrics(logits_sc, logits_fp, y):
    top1 = (logits_sc.argmax(1) == y).float().mean().item()
    top5 = logits_sc.topk(5, 1).indices.eq(y[:, None]).any(1).float().mean().item()
    top1_fp = logits_fp.argmax(1)
    flip = (logits_sc.argmax(1) != top1_fp).float().mean().item()
    l2 = (logits_sc - logits_fp).pow(2).sum(1).sqrt().mean().item()
    kl = F.kl_div(F.log_softmax(logits_sc, 1),
                  F.log_softmax(logits_fp, 1),
                  reduction="batchmean", log_target=True).item()
    return {"top1": top1, "top5": top5, "flip_rate": flip,
            "logit_l2": l2, "softmax_kl": kl}


def main():
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    seed_all(SEED)
    device = torch.device("cuda")

    print(f"[data] loading {DATA_ROOT}", flush=True)
    ds_full = ImageNetParquetVal(DATA_ROOT, transform=build_transform(224))
    idx = list(range(len(ds_full)))
    random.Random(SEED).shuffle(idx)
    ds = Subset(ds_full, idx[:N_IMAGES])
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=4, pin_memory=True)

    print("[fp] FP baseline", flush=True)
    model = load_model(device)
    logits_fp, y = run_model(model, loader, device)
    del model
    torch.cuda.empty_cache()
    fp_m = metrics(logits_fp, logits_fp, y)
    fp_m["n"] = int(y.numel())
    print(f"  FP top1={fp_m['top1']:.3f} top5={fp_m['top5']:.3f}", flush=True)

    results = {"mode": MODE, "sc_prec": SC_PREC, "n_images": N_IMAGES,
               "fp": fp_m, "sweep": []}

    for N in BLOCK_COUNTS:
        print(f"[sc] N={N:2d} MLP blocks patched (mode={MODE})", flush=True)
        model = load_model(device)
        if N > 0:
            if REGION == "first":
                skip_first, skip_last = 0, 24 - N
            elif REGION == "last":
                skip_first, skip_last = 24 - N, 0
            else:
                raise ValueError(f"bad REGION: {REGION}")
            stats = patch_model(
                model, sc_prec=SC_PREC,
                sc_qk=False, sc_av=False,
                sc_mlp=True,
                mlp_skip_first_k=skip_first,
                mlp_skip_last_k=skip_last,
                sc_mlp_mode=MODE,
            )
            print(f"    patched {stats['mlp_fc1']} fc1 + "
                  f"{stats['mlp_fc2']} fc2", flush=True)
        logits_sc, _ = run_model(model, loader, device)
        m = metrics(logits_sc, logits_fp, y)
        m["N_blocks_sc"] = N
        print(f"    top1={m['top1']:.3f} top5={m['top5']:.3f} "
              f"flip={m['flip_rate']:.3f} "
              f"logit_l2={m['logit_l2']:.3f} kl={m['softmax_kl']:.3f}",
              flush=True)
        results["sweep"].append(m)
        del model
        torch.cuda.empty_cache()

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
