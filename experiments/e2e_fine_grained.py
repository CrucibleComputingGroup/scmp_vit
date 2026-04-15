"""End-to-end real-SC eval using fine-grained per-(op, block) schedules.

Loads the noise-surrogate sensitivity matrix at results/sensitivity_all_ops.json
and builds ``sc_skip_worstK`` configs: SC every op in every block *except*
the K worst (op, block) pairs (by surrogate logit_l2). Tests whether the
surrogate's ranking actually transfers to real SC accuracy.
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

DATA_ROOT = "./data/imagenet/data"
N_BLOCKS = 24
SENSITIVITY_JSON = "results/sensitivity_all_ops.json"


def load_ranked_worst():
    with open(SENSITIVITY_JSON) as f:
        data = json.load(f)
    rows = data["grid"]
    rows.sort(key=lambda r: r["l2"], reverse=True)
    return [(r["op"], int(r["block"]), float(r["l2"])) for r in rows]


def spec_all_on():
    return {k: [1] * N_BLOCKS for k in SC_OP_NAMES}


def spec_skip_worst_k(ranked, k):
    drop = {(op, bi) for (op, bi, _) in ranked[:k]}
    spec = spec_all_on()
    for (op, bi) in drop:
        spec[op][bi] = 0
    return spec


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
    ap.add_argument("--skip_ks", type=int, nargs="*", default=[10, 20, 30])
    ap.add_argument("--include_everything", action="store_true", default=True)
    ap.add_argument("--out_json", default="results/e2e_fine_grained.json")
    args = ap.parse_args()

    os.environ.setdefault("XFORMERS_DISABLED", "1")
    seed_all(args.seed)
    device = torch.device("cuda")
    set_noise_model(False)   # real SC

    ranked = load_ranked_worst()
    print(f"[sens] loaded {len(ranked)} (op, block) pairs ranked by surrogate l2")
    print("[sens] top 5:",
          [(op, b, round(l, 2)) for (op, b, l) in ranked[:5]])

    configs = [("fp", None)]
    for k in args.skip_ks:
        spec = spec_skip_worst_k(ranked, k)
        configs.append((f"sc_skip_worst{k}", spec))
    if args.include_everything:
        configs.append(("everything", spec_all_on()))

    ds_full = ImageNetParquetVal(DATA_ROOT, transform=build_transform(224))
    idx = list(range(len(ds_full)))
    random.Random(args.seed).shuffle(idx)
    ds = Subset(ds_full, idx[:args.n_images])

    results = {"n_images": args.n_images, "sc_prec": args.sc_prec,
               "ranked_worst_top10": ranked[:10], "runs": {}}

    for name, spec in configs:
        batch = 32 if spec is None else 8
        loader = DataLoader(ds, batch_size=batch, shuffle=False,
                            num_workers=4, pin_memory=True)
        n_sc = 0 if spec is None else sum(sum(v) for v in spec.values())
        print(f"\n=== {name:<22} ({n_sc} SC op-instances) ===", flush=True)
        m = load_model(device)
        if spec is not None:
            stats = patch_model(m, sc_prec=args.sc_prec,
                                sc_mlp_mode="bipolar", sc_proj_mode="bipolar",
                                sc_ops_per_block=spec)
            print(f"  patched {dict((k, v) for k, v in stats.items() if v)}",
                  flush=True)
        res = run(m, loader, device)
        print(f"  top1={res['top1']:.4f} top5={res['top5']:.4f}  "
              f"{res['img_per_s']:.1f} img/s ({res['elapsed_s']:.0f}s)",
              flush=True)
        results["runs"][name] = {"n_sc_ops": n_sc, **res}
        del m; torch.cuda.empty_cache()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== SUMMARY (N={args.n_images}, sc_prec={args.sc_prec}) ===")
    print(f"{'config':<22} {'n_SC':>5} {'top1':>7} {'top5':>7} {'img/s':>7}")
    for name, _ in configs:
        r = results["runs"][name]
        print(f"{name:<22} {r['n_sc_ops']:>5d} {r['top1']:>7.4f} "
              f"{r['top5']:>7.4f} {r['img_per_s']:>7.1f}")
    print(f"[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
