"""Diagnose how heterogeneous the per-row abs-max is on real ViT-L/14 proj inputs.

If the distribution is flat (max/mean ~ O(1)), adaptive MP has little leverage:
no "outlier tokens" to protect at high precision, no "cold tokens" safe to
drop. If heavy-tailed (max/mean >> 1), adaptive should be able to win big --
the failure is then in the classifier, not the data.
"""
from __future__ import annotations
import json, os, sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sc"))
sys.path.insert(0, str(REPO / "cls"))

os.environ.setdefault("XFORMERS_DISABLED", "1")

from imagenet_parquet import ImageNetParquetVal
from sc_attention_patch import patch_model


def main():
    device = torch.device("cuda")
    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    ds = Subset(ImageNetParquetVal(
        "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data",
        transform=tfm), list(range(32)))
    loader = DataLoader(ds, batch_size=32, num_workers=2)

    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitl14_lc",
        source="github", trust_repo=True,
    ).to(device).eval()

    # Intercept every nn.Linear named qkv / proj inside attention blocks
    # BEFORE patching so we see the raw FP input (which is what the SC
    # kernel would see at runtime).
    captured = defaultdict(list)  # (op, block_idx) -> list[row_max tensors]

    # DINOv2 lc wraps the ViT — find the blocks by walking modules.
    blocks = None
    for m in model.modules():
        if hasattr(m, "blocks") and isinstance(m.blocks, torch.nn.ModuleList):
            blocks = m.blocks
            break
    if blocks is None:
        for m in model.modules():
            if isinstance(m, torch.nn.ModuleList) and len(m) >= 20:
                blocks = m
                break
    assert blocks is not None, "could not locate transformer blocks"
    print(f"Found {len(blocks)} transformer blocks")

    for b_idx, blk in enumerate(blocks):
        attn = blk.attn
        q_lin = attn.qkv
        o_lin = attn.proj
        def make_hook(op, idx):
            def hook(module, inputs, output):
                x = inputs[0]  # (B, N, D) or (M, D)
                if x.dim() == 3:
                    xf = x.float().reshape(-1, x.shape[-1])
                else:
                    xf = x.float()
                rm = xf.abs().amax(dim=-1)  # per-row
                captured[(op, idx)].append(rm.detach().cpu())
            return hook
        q_lin.register_forward_hook(make_hook("qkv_proj", b_idx))
        o_lin.register_forward_hook(make_hook("out_proj", b_idx))

    # One forward pass (batch 32)
    with torch.no_grad():
        for x, _ in loader:
            _ = model(x.to(device))
            break

    # Aggregate: compute heterogeneity stats per (op, block)
    stats = []
    for (op, idx), vs in sorted(captured.items()):
        v = torch.cat(vs)  # (M,) = B*N rows
        max_v = v.max().item()
        mean_v = v.mean().item()
        median_v = v.median().item()
        p99 = torch.quantile(v, 0.99).item()
        p50 = torch.quantile(v, 0.50).item()
        p5 = torch.quantile(v, 0.05).item()
        stats.append({
            "op": op, "block": idx, "N_rows": int(v.numel()),
            "max": round(max_v, 4), "mean": round(mean_v, 4),
            "median": round(median_v, 4),
            "max_over_mean": round(max_v / max(mean_v, 1e-9), 2),
            "p99": round(p99, 4), "p50": round(p50, 4), "p5": round(p5, 4),
            "p99_over_p50": round(p99 / max(p50, 1e-9), 2),
            "p50_over_p5":  round(p50 / max(p5, 1e-9), 2),
        })

    # Summary per op
    for op in ("qkv_proj", "out_proj"):
        rows = [s for s in stats if s["op"] == op]
        mom = [r["max_over_mean"] for r in rows]
        pop = [r["p99_over_p50"] for r in rows]
        cold = [r["p50_over_p5"] for r in rows]
        print(f"\n=== {op} across {len(rows)} blocks ===")
        print(f"  max/mean:       min={min(mom):.2f}  mean={sum(mom)/len(mom):.2f}  max={max(mom):.2f}")
        print(f"  p99/p50:        min={min(pop):.2f}  mean={sum(pop)/len(pop):.2f}  max={max(pop):.2f}")
        print(f"  p50/p5 (cold):  min={min(cold):.2f}  mean={sum(cold)/len(cold):.2f}  max={max(cold):.2f}")
        # Print one sample block in detail
        sample = rows[len(rows)//2]
        print(f"  [block {sample['block']}] mean={sample['mean']} median={sample['median']} "
              f"p99={sample['p99']} max={sample['max']}  (max/mean={sample['max_over_mean']})")

    out = REPO / "results" / "e2e" / "proj_heterogeneity.json"
    out.write_text(json.dumps(stats, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
