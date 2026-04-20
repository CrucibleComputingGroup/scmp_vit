"""Are pathological AV groups (high max/mean) also the IMPORTANT ones?

For each (b, h, block) group, measure two things in a pure-FP forward:
  - pathology: max/mean of attn row amax (how outlier-like the group is)
  - importance: ||attn_sm @ V||_F  (how much this group contributes to the
                block's attention output before out_proj mixes heads)

If important groups are disproportionately pathological, any MP scheme that
mishandles pathology will tank accuracy. If uncorrelated, we can trade
pathology handling for savings on the benign majority.
"""
from __future__ import annotations
import json, os, sys
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

    captured = {}  # block_idx → (attn_amax (B,H,N), av_output_per_head (B,H,N,D))
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
                av_out = attn_sm @ v  # (B, H, N, D)
                captured[idx] = (
                    attn_sm.float().amax(dim=-1).detach().cpu(),      # (B, H, N)
                    av_out.float().detach().cpu(),                    # (B, H, N, D)
                )
                out = av_out.transpose(1, 2).reshape(B, N, C)
                return _self.proj(out)
            return new
        attn.forward = make_new(orig, b_idx).__get__(attn, type(attn))

    with torch.no_grad():
        for x, _ in loader:
            _ = model(x.to(device)); break

    records = []
    for b_idx in sorted(captured.keys()):
        amx, av = captured[b_idx]
        B, H, N = amx.shape
        for b in range(B):
            for h in range(H):
                row_amax = amx[b, h]
                pathology = (row_amax.max().item() /
                             max(row_amax.mean().item(), 1e-9))
                # importance: Frobenius norm of this group's AV output
                # = sqrt(Σ_n Σ_d (attn_sm @ V)[n,d]^2)
                importance = av[b, h].pow(2).sum().sqrt().item()
                records.append({
                    "block": b_idx, "b": b, "h": h,
                    "pathology": pathology,
                    "importance": importance,
                })

    # Correlation: do pathological groups have higher importance?
    path = torch.tensor([r["pathology"] for r in records])
    imp  = torch.tensor([r["importance"] for r in records])

    # Pearson
    pm, im = path.mean(), imp.mean()
    cov = ((path - pm) * (imp - im)).mean()
    rho = (cov / (path.std() * imp.std())).item()

    # Rank correlation (Spearman)
    p_rank = path.argsort().argsort().float()
    i_rank = imp.argsort().argsort().float()
    prm, irm = p_rank.mean(), i_rank.mean()
    rho_s = (((p_rank - prm) * (i_rank - irm)).mean() /
             (p_rank.std() * i_rank.std())).item()

    print(f"Pearson(pathology, importance) = {rho:+.3f}")
    print(f"Spearman (rank)               = {rho_s:+.3f}")

    # Bucket groups by pathology severity and report mean importance per bucket
    path_sorted = sorted(records, key=lambda r: r["pathology"])
    buckets = [
        ("p0-50%  (benign)",   path_sorted[:len(records)//2]),
        ("p50-85% (mild)",     path_sorted[len(records)//2:int(len(records)*0.85)]),
        ("p85-97% (moderate)", path_sorted[int(len(records)*0.85):int(len(records)*0.97)]),
        ("p97-100% (pathological)", path_sorted[int(len(records)*0.97):]),
    ]
    print(f"\n{'bucket':<22s}  {'n':>5s}  {'path p50':>9s}  {'path p95':>9s}  "
          f"{'imp mean':>9s}  {'imp sum share':>14s}")
    total_imp = sum(r["importance"] for r in records)
    for name, grp in buckets:
        paths = [r["pathology"] for r in grp]
        imps  = [r["importance"] for r in grp]
        print(f"{name:<22s}  {len(grp):>5d}  "
              f"{sorted(paths)[len(paths)//2]:>9.2f}  "
              f"{sorted(paths)[int(len(paths)*0.95)]:>9.2f}  "
              f"{sum(imps)/len(imps):>9.3f}  "
              f"{100*sum(imps)/total_imp:>12.2f}%")

    # Also: per-block, are pathological groups concentrated in late blocks?
    print(f"\nPer-block pathology-importance correlation (per layer):")
    print(f"{'block':<5s}  {'n':>5s}  {'p50 path':>9s}  {'imp mean':>9s}  "
          f"{'ρ(rank)':>8s}")
    for b_idx in sorted(captured.keys()):
        br = [r for r in records if r["block"] == b_idx]
        ps = torch.tensor([r["pathology"] for r in br])
        is_ = torch.tensor([r["importance"] for r in br])
        p_rank = ps.argsort().argsort().float()
        i_rank = is_.argsort().argsort().float()
        pr_m, ir_m = p_rank.mean(), i_rank.mean()
        rho_b = (((p_rank - pr_m) * (i_rank - ir_m)).mean() /
                 (p_rank.std() * i_rank.std())).item()
        print(f"{b_idx:<5d}  {len(br):>5d}  "
              f"{sorted([r['pathology'] for r in br])[len(br)//2]:>9.2f}  "
              f"{sum(r['importance'] for r in br)/len(br):>9.3f}  "
              f"{rho_b:>+8.3f}")


if __name__ == "__main__":
    main()
