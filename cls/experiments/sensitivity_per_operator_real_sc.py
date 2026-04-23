"""Per-(op, block) sensitivity with REAL SC — one individual operator at a time.

Every transformer block has 6 SC-switchable operators:
    mlp_fc1, mlp_fc2, qkv_proj, out_proj, qk, av

For each (op, block) pair (6 × 24 = 144 configs) we turn SC on for ONLY that
single operator (everything else FP) and measure vs FP:
    top-1, top-5, logit L2 drift, softmax KL, top-1 prediction flip rate.

Real SC kernels, sc_prec=8, 100 ImageNet val images, seed=0.
Note: `sc_ops_per_block` couples qkv_proj + out_proj under "proj", so we patch
individual modules manually here to isolate each one.
"""
import argparse, os, sys, random, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]   # vit_sc/
CLS = HERE / "cls"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CLS))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from eval import build_transform, load_model, seed_all
from imagenet_parquet import ImageNetParquetVal
from sc_attention_patch import (
    SCLinear, make_sc_attention_forward, set_noise_model, _discover_blocks,
)

DATA_ROOT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data"
N_IMAGES = 100
BATCH    = 8
SEED     = 0
N_BLOCKS = 24
# 6 individual operators per block. Order chosen so scariest (mlp_fc2) is last.
OP_NAMES = ("qk", "av", "qkv_proj", "out_proj", "mlp_fc1", "mlp_fc2")


def _find_attn(blk):
    for m in blk.modules():
        if hasattr(m, "qkv") and hasattr(m, "proj") and hasattr(m, "num_heads"):
            return m
    raise RuntimeError("no attention submodule found in block")


def patch_single_op(model, op, bi, sc_prec):
    """Turn SC on for exactly one operator in block bi; leave everything else FP."""
    blocks = _discover_blocks(model)
    if not (0 <= bi < len(blocks)):
        raise IndexError(f"block {bi} out of range (n_blocks={len(blocks)})")
    blk = blocks[bi]
    if op == "mlp_fc1":
        assert isinstance(blk.mlp.fc1, nn.Linear)
        blk.mlp.fc1 = SCLinear(blk.mlp.fc1, sc_prec, mode="bipolar")
    elif op == "mlp_fc2":
        assert isinstance(blk.mlp.fc2, nn.Linear)
        blk.mlp.fc2 = SCLinear(blk.mlp.fc2, sc_prec, mode="bipolar")
    elif op == "qkv_proj":
        attn = _find_attn(blk)
        assert isinstance(attn.qkv, nn.Linear)
        attn.qkv = SCLinear(attn.qkv, sc_prec, mode="bipolar")
    elif op == "out_proj":
        attn = _find_attn(blk)
        assert isinstance(attn.proj, nn.Linear)
        attn.proj = SCLinear(attn.proj, sc_prec, mode="bipolar")
    elif op == "qk":
        attn = _find_attn(blk)
        fwd = make_sc_attention_forward(sc_prec, sc_av=False, sc_qk=True)
        attn.forward = fwd.__get__(attn, type(attn))
    elif op == "av":
        attn = _find_attn(blk)
        fwd = make_sc_attention_forward(sc_prec, sc_av=True, sc_qk=False)
        attn.forward = fwd.__get__(attn, type(attn))
    else:
        raise ValueError(f"unknown op: {op}")


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc_prec", type=int, default=8,
                    help="SC integer precision (stoc_len = 2**sc_prec).")
    ap.add_argument("--out", default="",
                    help="Output JSON path (default: "
                         "cls/sensitivity/"
                         "sensitivity_per_operator_real_sc_p<sc_prec>.json).")
    args = ap.parse_args()
    SC_PREC = args.sc_prec
    OUT = args.out or str(
        CLS / "sensitivity"
        / f"sensitivity_per_operator_real_sc_p{SC_PREC}.json")

    os.environ.setdefault("XFORMERS_DISABLED", "1")
    seed_all(SEED)
    device = torch.device("cuda")

    print(f"[env] torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"gpu={torch.cuda.get_device_name(0)} sc_prec={SC_PREC}", flush=True)

    ds_full = ImageNetParquetVal(DATA_ROOT, transform=build_transform(224))
    idx = list(range(len(ds_full)))
    random.Random(SEED).shuffle(idx)
    ds = Subset(ds_full, idx[:N_IMAGES])
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2,
                        pin_memory=True)
    print(f"[data] {N_IMAGES} images seed={SEED} bs={BATCH}", flush=True)

    t0 = time.time()
    m = load_model(device)
    L_fp, y = run(m, loader, device)
    del m; torch.cuda.empty_cache()
    fp_top1 = (L_fp.argmax(1) == y).float().mean().item()
    fp_top5 = L_fp.topk(5, 1).indices.eq(y.unsqueeze(1)).any(1).float().mean().item()
    print(f"[fp] top1={fp_top1:.3f} top5={fp_top5:.3f} ({time.time()-t0:.1f}s)",
          flush=True)

    set_noise_model(False)  # REAL SC, not surrogate

    results = {
        "config": {
            "sc_prec": SC_PREC, "n_images": N_IMAGES, "batch": BATCH,
            "seed": SEED, "n_blocks": N_BLOCKS, "noise_model": False,
            "op_names": list(OP_NAMES),
        },
        "fp": {"top1": fp_top1, "top5": fp_top5},
        "grid": [],
    }

    t_sweep = time.time()
    n_total = len(OP_NAMES) * N_BLOCKS
    done = 0
    for op in OP_NAMES:
        op_row = []
        for bi in range(N_BLOCKS):
            torch.manual_seed(SEED)
            t1 = time.time()
            m = load_model(device)
            patch_single_op(m, op, bi, SC_PREC)
            L, _ = run(m, loader, device)
            dt = time.time() - t1
            mt = metrics(L, L_fp, y)
            mt["dtop1"] = mt["top1"] - fp_top1
            row = {"op": op, "block": bi, "elapsed_s": dt, **mt}
            results["grid"].append(row)
            op_row.append(row)
            del m; torch.cuda.empty_cache()
            done += 1
            print(f"[{done:3d}/{n_total}] {op:9s} b{bi:02d}  "
                  f"top1={mt['top1']:.3f} Δ={mt['dtop1']:+.3f}  "
                  f"l2={mt['l2']:6.2f} kl={mt['kl']:7.3f} "
                  f"flip={mt['flip']:.3f} ({dt:.1f}s)", flush=True)
        # per-op summary across blocks
        l2s = [r["l2"] for r in op_row]
        d1s = [r["dtop1"] for r in op_row]
        print(f"  [{op:9s}] blocks l2 range = [{min(l2s):.2f}, {max(l2s):.2f}], "
              f"Δtop1 range = [{min(d1s):+.3f}, {max(d1s):+.3f}]", flush=True)

    results["elapsed_total_s"] = time.time() - t_sweep
    print(f"\n[sweep] total {results['elapsed_total_s']:.0f}s", flush=True)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {OUT}", flush=True)

    # --- Compact matrices ---
    def _matrix(key, fmt, label):
        print(f"\n{label} by (op, block):")
        hdr = "op \\ block " + " ".join(f"{b:>6d}" for b in range(N_BLOCKS))
        print(hdr)
        for op in OP_NAMES:
            row = [r for r in results["grid"] if r["op"] == op]
            vals = " ".join(fmt.format(r[key]) for r in row)
            print(f"{op:<11s}{vals}")

    _matrix("l2",    "{:>6.2f}", "logit_l2")
    _matrix("dtop1", "{:>+6.2f}", "Δtop1")
    _matrix("flip",  "{:>6.2f}", "flip_rate")

    # --- Worst individual operators (rank by L2) ---
    worst = sorted(results["grid"], key=lambda r: -r["l2"])[:15]
    print("\nTop-15 most SC-sensitive individual operators (by logit L2):")
    print(f"{'op':<10s} {'block':>5s} {'top1':>6s} {'Δtop1':>7s} {'l2':>7s} "
          f"{'kl':>8s} {'flip':>6s}")
    for r in worst:
        print(f"{r['op']:<10s} {r['block']:>5d} {r['top1']:>6.3f} "
              f"{r['dtop1']:>+7.3f} {r['l2']:>7.2f} {r['kl']:>8.3f} "
              f"{r['flip']:>6.3f}")


if __name__ == "__main__":
    main()
