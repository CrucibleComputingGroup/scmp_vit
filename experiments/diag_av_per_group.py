"""Per-(batch, head) distribution of attn row amax for AV.

AV classification is done PER-(b, h) group (each has N=257 query rows), so
the relevant "distribution" for MP sizing is WITHIN each group, not across
all groups pooled. This script captures per-group stats and compares to
the pooled view.
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
            blocks = m.blocks
            break

    # Captured: per-block, the full (B, H, N) attn.amax tensor
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
                captured[idx] = attn_sm.float().amax(dim=-1).detach().cpu()  # (B, H, N)
                out = attn_sm @ v
                out = out.transpose(1, 2).reshape(B, N, C)
                out = _self.proj(out)
                return out
            return new
        attn.forward = make_new(orig, b_idx).__get__(attn, type(attn))

    with torch.no_grad():
        for x, _ in loader:
            _ = model(x.to(device))
            break

    B = H = N = None
    for b_idx in sorted(captured.keys()):
        amx = captured[b_idx]  # (B, H, N)
        B, H, N = amx.shape
        break
    print(f"Shape per block: B={B}, H={H}, N={N} rows/group; "
          f"total groups per block = {B*H}; total rows per block = {B*H*N}")

    # ---------- Per-(b, h) group spread stats aggregated across all groups ----------
    per_group_stats = {"max_over_mean": [], "p99_over_p50": [], "p50_over_p5": []}
    for b_idx in sorted(captured.keys()):
        amx = captured[b_idx]
        for b in range(B):
            for h in range(H):
                row = amx[b, h]  # (N,)
                m = row.max().item()
                mn = row.mean().item()
                p99 = torch.quantile(row, 0.99).item()
                p50 = torch.quantile(row, 0.50).item()
                p5  = torch.quantile(row, 0.05).item()
                per_group_stats["max_over_mean"].append(m / max(mn, 1e-9))
                per_group_stats["p99_over_p50"].append(p99 / max(p50, 1e-9))
                per_group_stats["p50_over_p5"].append(p50 / max(p5, 1e-9))

    def pct(vs, qs=(0.05, 0.50, 0.95)):
        v = torch.tensor(vs)
        return [torch.quantile(v, q).item() for q in qs]

    print("\nWITHIN-group spread statistics (over "
          f"{len(per_group_stats['max_over_mean'])} (b, h, block) groups):")
    print(f"{'stat':<16s}  p5      p50     p95    min      max")
    for stat in ("max_over_mean", "p99_over_p50", "p50_over_p5"):
        vs = per_group_stats[stat]
        q5, q50, q95 = pct(vs)
        print(f"{stat:<16s}  {q5:>6.2f}  {q50:>6.2f}  {q95:>6.2f}  "
              f"{min(vs):>6.2f}   {max(vs):>6.2f}")

    # ---------- Sample groups: show raw histogram per-group ----------
    print("\nSample raw distributions (block 12, batch 0):")
    amx12 = captured[12][0]  # (H, N)
    for h in (0, 3, 7, 11, 15):
        row = amx12[h]
        sorted_vals = torch.sort(row, descending=True).values
        print(f"  head={h:2d}  top3={sorted_vals[:3].tolist()}  "
              f"bot3={sorted_vals[-3:].tolist()}  "
              f"mean={row.mean().item():.4f}  "
              f"p99/p50={torch.quantile(row,0.99).item()/max(torch.quantile(row,0.5).item(),1e-9):.2f}")

    # ---------- Concentration classification ----------
    # Count groups by "peakiness": a row of amax describes the row of attention
    # distribution. Use amax directly as peakiness (value ∈ [1/257, 1]).
    peaky_groups = 0   # high concentration (row.amax > 0.5)
    diffuse_groups = 0 # low concentration (row.amax < 0.05)
    total = 0
    for b_idx in sorted(captured.keys()):
        amx = captured[b_idx]
        # per-group mean amax → how "peaky" that group's attention tends to be
        for b in range(B):
            for h in range(H):
                mean_amax = amx[b, h].mean().item()
                total += 1
                if mean_amax > 0.5: peaky_groups += 1
                elif mean_amax < 0.05: diffuse_groups += 1
    print(f"\nGroup peakiness (mean row-amax per (b,h,block) group):")
    print(f"  peaky (>0.5):   {peaky_groups}/{total}  ({100*peaky_groups/total:.1f}%)")
    print(f"  diffuse (<0.05):{diffuse_groups}/{total}  ({100*diffuse_groups/total:.1f}%)")

    out = REPO / "results" / "e2e" / "av_per_group_distribution.json"
    summary = {
        "shape": {"B": B, "H": H, "N": N, "n_blocks": len(captured),
                  "total_groups": len(per_group_stats["max_over_mean"])},
        "within_group_spread": {
            k: {"p5": round(pct(v)[0],3), "p50": round(pct(v)[1],3),
                "p95": round(pct(v)[2],3), "min": round(min(v),3),
                "max": round(max(v),3)}
            for k, v in per_group_stats.items()},
        "group_peakiness": {
            "peaky_gt_0p5": peaky_groups, "diffuse_lt_0p05": diffuse_groups,
            "total": total},
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
