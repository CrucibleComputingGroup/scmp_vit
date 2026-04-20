"""How bad is within-group spread for AV? Histogram + dump extreme groups."""
from __future__ import annotations
import json, os, sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sc"))
sys.path.insert(0, str(REPO / "cls"))


def main():
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    from imagenet_parquet import ImageNetParquetVal

    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    ds = Subset(ImageNetParquetVal(
        "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data",
        transform=tfm), list(range(32)))
    loader = DataLoader(ds, batch_size=32, num_workers=2)
    device = torch.device("cuda")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_lc",
                           source="github", trust_repo=True).to(device).eval()

    blocks = None
    for m in model.modules():
        if hasattr(m, "blocks") and isinstance(m.blocks, torch.nn.ModuleList):
            blocks = m.blocks; break

    captured = {}
    for b_idx, blk in enumerate(blocks):
        attn = blk.attn
        orig = attn.forward
        def make_new(orig_fwd, idx):
            def new(_self, x, *a, **k):
                B, N, C = x.shape
                H = _self.num_heads
                D = C // H
                qkv = _self.qkv(x).reshape(B, N, 3, H, D)
                q, _k, v = torch.unbind(qkv, 2)
                q = q.transpose(1, 2); _k = _k.transpose(1, 2); v = v.transpose(1, 2)
                attn_sm = ((q @ _k.transpose(-2, -1)) * _self.scale).softmax(dim=-1)
                captured[idx] = attn_sm.float().amax(dim=-1).detach().cpu()
                out = attn_sm @ v
                out = out.transpose(1, 2).reshape(B, N, C)
                return _self.proj(out)
            return new
        attn.forward = make_new(orig, b_idx).__get__(attn, type(attn))

    with torch.no_grad():
        for x, _ in loader:
            _ = model(x.to(device)); break

    # Per-group stats
    records = []
    for b_idx in sorted(captured.keys()):
        amx = captured[b_idx]
        B, H, N = amx.shape
        for b in range(B):
            for h in range(H):
                row = amx[b, h]
                mn = row.mean().item()
                m = row.max().item()
                records.append({"block": b_idx, "b": b, "h": h,
                                "mean": mn, "max": m,
                                "max_over_mean": m / max(mn, 1e-9),
                                "row": row})

    # Histogram of max/mean
    bins = [1.0, 1.5, 2, 3, 5, 10, 20, 50, 100, 1e9]
    labels = ["1.0–1.5", "1.5–2", "2–3", "3–5", "5–10",
              "10–20", "20–50", "50–100", "100+"]
    hist = Counter()
    for r in records:
        mom = r["max_over_mean"]
        for i in range(len(bins)-1):
            if bins[i] <= mom < bins[i+1]:
                hist[labels[i]] += 1
                break
    total = sum(hist.values())
    print(f"\nHistogram of within-group max/mean ({total} groups total):")
    print(f"{'range':<12s}  {'count':>7s}  {'%':>7s}  {'cumulative%':>11s}")
    cumul = 0
    for lbl in labels:
        c = hist.get(lbl, 0)
        cumul += c
        print(f"{lbl:<12s}  {c:>7d}  {100*c/total:>6.2f}%  {100*cumul/total:>10.2f}%")

    # What does quantile assignment look like inside a peaky group?
    # Pick a median-peaky and an extreme-peaky group, show row-amax sorted
    records.sort(key=lambda r: r["max_over_mean"])
    median = records[len(records)//2]
    p95 = records[int(len(records)*0.95)]
    p99 = records[int(len(records)*0.99)]
    extreme = records[-1]
    print(f"\nSample groups (sorted by max/mean):")
    for tag, r in [("p50", median), ("p95", p95), ("p99", p99),
                   ("max", extreme)]:
        row = r["row"]
        sv = torch.sort(row, descending=True).values
        print(f"  {tag}  block={r['block']:2d} b={r['b']:2d} h={r['h']:2d}  "
              f"max/mean={r['max_over_mean']:.2f}  mean={r['mean']:.4f}")
        print(f"     top5: {[round(v, 3) for v in sv[:5].tolist()]}")
        print(f"     middle (ranks 120-125): {[round(v, 3) for v in sv[120:125].tolist()]}")
        print(f"     bot5: {[round(v, 3) for v in sv[-5:].tolist()]}")
        # For adaptive [0.05/0.40/0.45/0.10] reversed, where would each row go?
        # Reversed: top 5% by (-amax) = 5% lowest amax → sl=256
        # 0% → sl=256, 0-40% → sl=128, 40-85% → sl=64, bot 10% (by -amax) = 10% highest amax → sl=32
        # Effective:
        # Lowest 5% amax → sl=256
        # Lowest 5-45% amax → sl=128
        # Lowest 45-90% amax → sl=64
        # Highest 10% amax → sl=32

    # Quantile assignment illustration for sample groups
    fractions = [0.05, 0.35, 0.40, 0.20]
    levels = [256, 128, 64, 32]
    print(f"\nIF we assign with fractions={fractions} levels={levels} (reversed):")
    for tag, r in [("p50 (typical)", median), ("p99 (extreme)", p99)]:
        row = r["row"]
        sl_assign = torch.zeros(len(row), dtype=torch.long)
        # Sort by -amax (reversed) descending → idx[0] has lowest amax (needs sl=256)
        idx = torch.argsort(-row, descending=True)  # same as argsort(row, descending=False)
        offset = 0
        mapping = {}
        for lvl, frac in zip(levels, fractions):
            n = round(frac * len(row))
            chosen = idx[offset:offset+n]
            sl_assign[chosen] = lvl
            offset += n
            mapping[lvl] = chosen
        # What are the actual amax ranges in each sl bucket?
        print(f"  {tag}  (block={r['block']} b={r['b']} h={r['h']}):")
        for lvl in levels:
            mask = sl_assign == lvl
            vals = row[mask]
            print(f"    sl={lvl:<3d}  count={mask.sum().item():>3d}  "
                  f"amax range = [{vals.min().item():.3f}, {vals.max().item():.3f}]  "
                  f"mean {vals.mean().item():.3f}")


if __name__ == "__main__":
    main()
