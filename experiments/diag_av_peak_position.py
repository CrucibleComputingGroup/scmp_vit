"""Are ViT attention sinks real? For pathologically peaky rows (amax > 0.5),
where is the argmax position?

If many peak to position 0 (CLS) → sink-like behavior.
If spread across positions → semantic attention, no sink.
"""
from __future__ import annotations
import os, sys
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
                # Store full attn_sm (B, H, N, N)
                captured[idx] = attn_sm.detach().cpu().float()
                out = attn_sm @ v
                return _self.proj(out.transpose(1, 2).reshape(B, N, C))
            return new
        attn.forward = make_new(orig, b_idx).__get__(attn, type(attn))

    with torch.no_grad():
        for x, _ in loader:
            _ = model(x.to(device)); break

    # For peaky rows (amax > 0.5), where is the peak?
    all_argmax_positions = []  # (block, argmax_pos, amax_value)
    for b_idx in sorted(captured.keys()):
        attn = captured[b_idx]  # (B, H, N, N)
        row_max, row_argmax = attn.max(dim=-1)  # (B, H, N), (B, H, N)
        # Threshold: peaky = row with amax > 0.5
        mask = row_max > 0.5
        pos = row_argmax[mask]      # LongTensor
        val = row_max[mask]
        all_argmax_positions.append((b_idx, pos, val))

    # Aggregate by (block, argmax_position)
    print("Peaky rows (amax > 0.5) and where they peak:")
    print(f"{'block':>5s}  {'n_peaky':>8s}  {'% to pos 0 (CLS)':>18s}  "
          f"{'% to pos 1-4 (reg)':>20s}  {'median argmax':>15s}")
    grand_cls = 0; grand_reg = 0; grand_tot = 0
    for b_idx, pos, val in all_argmax_positions:
        if pos.numel() == 0:
            print(f"{b_idx:>5d}  (no peaky rows)")
            continue
        cls_frac = (pos == 0).float().mean().item()
        reg_frac = ((pos >= 1) & (pos <= 4)).float().mean().item()
        median_pos = int(pos.median().item())
        print(f"{b_idx:>5d}  {pos.numel():>8d}  {100*cls_frac:>16.1f}%  "
              f"{100*reg_frac:>18.1f}%  {median_pos:>15d}")
        grand_cls += (pos == 0).sum().item()
        grand_reg += ((pos >= 1) & (pos <= 4)).sum().item()
        grand_tot += pos.numel()
    print(f"\nGrand total: n_peaky={grand_tot}  "
          f"→ pos 0 (CLS): {100*grand_cls/max(grand_tot,1):.1f}%  "
          f"→ pos 1-4 (register): {100*grand_reg/max(grand_tot,1):.1f}%")

    # Additional: which heads are "always peaky"?
    print("\nPer-(block, head) peaky-row fraction (top 10 by fraction):")
    rec = []
    for b_idx in sorted(captured.keys()):
        attn = captured[b_idx]
        B, H, N, _ = attn.shape
        row_max = attn.max(dim=-1).values  # (B, H, N)
        mask = row_max > 0.5
        frac_per_bh = mask.float().mean(dim=(0, 2))  # (H,) — frac of peaky rows per head
        for h in range(H):
            rec.append((b_idx, h, frac_per_bh[h].item()))
    rec.sort(key=lambda r: -r[2])
    print(f"{'block':>5s}  {'head':>5s}  {'frac peaky':>10s}")
    for b, h, f in rec[:10]:
        print(f"{b:>5d}  {h:>5d}  {100*f:>9.1f}%")


if __name__ == "__main__":
    main()
