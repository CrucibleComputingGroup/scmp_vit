"""End-to-end real-SC eval, per-block vector API.

Each config is `dict[op_name, list[0/1] of length N_BLOCKS]` — one vector per
operator. Op names: mlp_fc1, mlp_fc2, qk, av, proj (proj toggles both qkv_proj
and out_proj together).

Configs below probe the questions raised by results/mlp_sensitivity_sweep.json
and the earlier bipolar first/last-N sweep.
"""
import argparse, json, os, random, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import torch
from torch.utils.data import DataLoader, Subset

from eval import build_transform, load_model, seed_all
from imagenet_parquet import ImageNetParquetVal
from sc_attention_patch import patch_model, SC_OP_NAMES, set_noise_model

DATA_ROOT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data"
N_BLOCKS = 24


def ones(n=N_BLOCKS):  return [1] * n
def zeros(n=N_BLOCKS): return [0] * n
def first_k(k, n=N_BLOCKS):  return [1] * k + [0] * (n - k)
def last_k(k, n=N_BLOCKS):   return [0] * (n - k) + [1] * k


def cfg(**ops):
    """Build a complete sc_ops_per_block dict; unlisted ops default all-0."""
    out = {k: zeros() for k in SC_OP_NAMES}
    for k, v in ops.items():
        assert k in SC_OP_NAMES, f"bad op: {k}"
        out[k] = list(v)
    return out


# Configs: each maps a human label to a per-op vector spec.
CONFIGS = {
    # Baseline
    "fp": None,

    # Uniform (all-blocks) per-op
    "qk_only":          cfg(qk=ones()),
    "av_only":          cfg(av=ones()),
    "proj_only":        cfg(proj=ones()),
    "attn_full":        cfg(qk=ones(), av=ones(), proj=ones()),

    # MLP uniform (known-bad at full)
    "mlp_fc1_only":     cfg(mlp_fc1=ones()),
    "mlp_fc2_only":     cfg(mlp_fc2=ones()),
    "mlp_full":         cfg(mlp_fc1=ones(), mlp_fc2=ones()),

    # Sensitivity-driven partial MLP
    "mlp_first8":       cfg(mlp_fc1=first_k(8), mlp_fc2=first_k(8)),
    "mlp_first16":      cfg(mlp_fc1=first_k(16), mlp_fc2=first_k(16)),
    "mlp_fc1_all_fc2_first16": cfg(mlp_fc1=ones(), mlp_fc2=first_k(16)),

    # Full stack with selective MLP skipping
    "attn_plus_fc1_all":       cfg(mlp_fc1=ones(),  qk=ones(), av=ones(), proj=ones()),
    "attn_plus_mlp_first16":   cfg(mlp_fc1=first_k(16), mlp_fc2=first_k(16),
                                    qk=ones(), av=ones(), proj=ones()),
    "everything":              cfg(mlp_fc1=ones(), mlp_fc2=ones(),
                                    qk=ones(), av=ones(), proj=ones()),
}


@torch.no_grad()
def run(model, loader, device):
    top1 = top5 = n = 0; t0 = time.time()
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(x)
        _, p5 = logits.topk(5, 1)
        correct = p5.eq(y.unsqueeze(1))
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(1).sum().item()
        n += y.numel()
    dt = time.time() - t0
    return {"n": n, "top1": top1 / n, "top5": top5 / n,
            "elapsed_s": dt, "img_per_s": n / dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sc_prec", type=int, default=8)
    ap.add_argument("--fp_batch", type=int, default=32)
    ap.add_argument("--sc_batch", type=int, default=8)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--out_json", default="results/e2e_per_block.json")
    args = ap.parse_args()

    os.environ.setdefault("XFORMERS_DISABLED", "1")
    seed_all(args.seed)
    device = torch.device("cuda")
    set_noise_model(False)  # real SC

    ds_full = ImageNetParquetVal(DATA_ROOT, transform=build_transform(224))
    idx = list(range(len(ds_full)))
    random.Random(args.seed).shuffle(idx)
    ds = Subset(ds_full, idx[:args.n_images])

    results = {"n_images": args.n_images, "sc_prec": args.sc_prec,
               "n_blocks": N_BLOCKS, "vec_order": list(SC_OP_NAMES),
               "runs": {}}

    for name in args.configs:
        assert name in CONFIGS, f"unknown config: {name}"
        spec = CONFIGS[name]
        batch = args.fp_batch if spec is None else args.sc_batch
        loader = DataLoader(ds, batch_size=batch, shuffle=False,
                            num_workers=4, pin_memory=True)
        n_sc = 0 if spec is None else sum(sum(v) for v in spec.values())
        print(f"\n=== {name:<26} ({n_sc} SC op-instances across {N_BLOCKS} blocks) ===",
              flush=True)
        m = load_model(device)
        if spec is not None:
            stats = patch_model(m, sc_prec=args.sc_prec,
                                sc_mlp_mode="bipolar", sc_proj_mode="bipolar",
                                sc_ops_per_block=spec)
            active = {k: v for k, v in stats.items() if v}
            print(f"  patched {active}", flush=True)
        res = run(m, loader, device)
        print(f"  top1={res['top1']:.4f} top5={res['top5']:.4f}  "
              f"{res['img_per_s']:.1f} img/s ({res['elapsed_s']:.0f}s)",
              flush=True)
        results["runs"][name] = {"n_sc_ops": n_sc, "spec": spec, **res}
        del m; torch.cuda.empty_cache()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] wrote {args.out_json}", flush=True)

    print(f"\n=== SUMMARY (N={args.n_images}, sc_prec={args.sc_prec}) ===")
    print(f"{'config':<28} {'n_SC':>5} {'top1':>7} {'top5':>7} {'img/s':>7}")
    for name in args.configs:
        r = results["runs"][name]
        print(f"{name:<28} {r['n_sc_ops']:>5d} {r['top1']:>7.4f} "
              f"{r['top5']:>7.4f} {r['img_per_s']:>7.1f}")


if __name__ == "__main__":
    main()
