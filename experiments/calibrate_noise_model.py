"""Calibrate noise-model correction factor against real-SC data in N<=8 regime.

Real-SC reference points (bipolar, first-N blocks MLP SC'd, N=100 imgs):
    N=1:  logit_l2 = 7.535
    N=2:  logit_l2 = 13.048
    N=4:  logit_l2 = 15.268
    N=8:  logit_l2 = 16.746
Pick a correction factor that fits this small-N regime (the practical one).
"""
import os, sys, random, json
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from eval import build_transform, load_model, seed_all
from imagenet_parquet import ImageNetParquetVal
from sc_attention_patch import patch_model, set_noise_model
from sc_integration.noise_matmul import set_noise_corrections

REF = {1: 7.535, 2: 13.048, 4: 15.268, 8: 16.746}
DATA_ROOT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data"
N_IMAGES, BATCH, SEED = 100, 8, 0


@torch.no_grad()
def run(model, loader, device):
    Ls, Ys = [], []
    for x, y in loader:
        Ls.append(model(x.to(device, non_blocking=True)).float().cpu())
        Ys.append(y)
    return torch.cat(Ls), torch.cat(Ys)


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

    set_noise_model(True)
    results = []
    for local_corr in [0.15, 0.5, 1.0, 2.0, 4.0]:
        set_noise_corrections(local=local_corr, global_=0.60)
        row = {"local_corr": local_corr}
        line = f"corr={local_corr:<4}  "
        for N in (1, 2, 4, 8):
            m = load_model(device)
            patch_model(m, sc_prec=8, sc_qk=False, sc_av=False,
                        sc_mlp=True, mlp_skip_first_k=0,
                        mlp_skip_last_k=24 - N, sc_mlp_mode="bipolar")
            L, _ = run(m, loader, device)
            l2 = (L - L_fp).pow(2).sum(1).sqrt().mean().item()
            top1 = (L.argmax(1) == y).float().mean().item()
            row[f"N{N}"] = {"l2": l2, "top1": top1, "ref_l2": REF[N],
                             "ratio": l2 / REF[N]}
            line += f"N={N} l2={l2:5.2f}/{REF[N]:5.2f} "
            del m; torch.cuda.empty_cache()
        print(line, flush=True)
        results.append(row)
    Path("results").mkdir(exist_ok=True)
    with open("results/noise_model_calibration.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
