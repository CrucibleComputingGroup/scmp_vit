"""Compare real-SC vs noise-model surrogate on MLP-SC all-24-blocks (bipolar).

If the surrogate is well-calibrated, top-1 / logit_l2 / KL should match within
a few percent on a 100-image subset.
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
import sc_attention_patch
from sc_attention_patch import patch_model, set_noise_model

DATA_ROOT = "./data/imagenet/data"
N_IMAGES, BATCH, SEED = 100, 8, 0


@torch.no_grad()
def run(model, loader, device):
    Ls, Ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        Ls.append(model(x).float().cpu())
        Ys.append(y)
    return torch.cat(Ls), torch.cat(Ys)


def metrics(L_sc, L_fp, y):
    top1 = (L_sc.argmax(1) == y).float().mean().item()
    l2 = (L_sc - L_fp).pow(2).sum(1).sqrt().mean().item()
    kl = F.kl_div(F.log_softmax(L_sc, 1), F.log_softmax(L_fp, 1),
                  reduction="batchmean", log_target=True).item()
    return top1, l2, kl


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

    print("[fp] reference", flush=True)
    m = load_model(device)
    import time; t0 = time.time()
    L_fp, y = run(m, loader, device)
    print(f"  FP done in {time.time()-t0:.1f}s", flush=True)
    del m; torch.cuda.empty_cache()

    results = {}
    for path_name, use_noise in [("real_sc", False), ("noise_model", True)]:
        print(f"[{path_name}] all 24 MLP blocks SC (bipolar)", flush=True)
        set_noise_model(use_noise)
        m = load_model(device)
        stats = patch_model(m, sc_prec=8, sc_qk=False, sc_av=False,
                            sc_mlp=True, sc_mlp_mode="bipolar")
        t0 = time.time()
        L, _ = run(m, loader, device)
        dt = time.time() - t0
        top1, l2, kl = metrics(L, L_fp, y)
        print(f"  top1={top1:.3f} logit_l2={l2:.3f} kl={kl:.3f} ({dt:.1f}s)",
              flush=True)
        results[path_name] = {"top1": top1, "logit_l2": l2, "kl": kl,
                               "elapsed_s": dt}
        del m; torch.cuda.empty_cache()
    Path("results").mkdir(exist_ok=True)
    with open("results/validate_noise_model.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
