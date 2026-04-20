"""Diagnose per-row distribution statistics for ALL 6 SC operators on a
pure FP DINOv2 ViT-L/14 forward. Outputs:

  - qkv_proj, out_proj, mlp_fc1, mlp_fc2: per-token-row |x|.amax (Linear input)
  - qk: per-head |Q|.amax (QK BMM input)
  - av: per-row attn.amax (AV BMM input; softmax output)

Per op: min / mean / max of max/mean, p99/p50, p50/p5 across 24 blocks, plus
one sample block's raw histogram.
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


def ratio_stats(per_block_values: list[float]) -> dict:
    return {
        "min":  round(min(per_block_values), 3),
        "mean": round(sum(per_block_values) / len(per_block_values), 3),
        "max":  round(max(per_block_values), 3),
    }


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
    assert blocks is not None, "could not find blocks"
    print(f"Found {len(blocks)} transformer blocks")

    captured_linear = defaultdict(list)  # (op, block_idx) -> list[tensor]
    captured_attn = defaultdict(list)    # (op, block_idx) -> list[tensor]

    for b_idx, blk in enumerate(blocks):
        attn = blk.attn
        mlp = blk.mlp
        # Linear hooks on input
        for op_name, lin in [
            ("qkv_proj", attn.qkv),
            ("out_proj", attn.proj),
            ("mlp_fc1", mlp.fc1),
            ("mlp_fc2", mlp.fc2),
        ]:
            def make_lin_hook(op, idx):
                def hook(_m, inputs, _out):
                    x = inputs[0]
                    xf = x.float().reshape(-1, x.shape[-1])
                    captured_linear[(op, idx)].append(
                        xf.abs().amax(dim=-1).detach().cpu())
                return hook
            lin.register_forward_hook(make_lin_hook(op_name, b_idx))

        # QK + AV hooks: we need q/k before QK BMM and attn (softmax output)
        # before AV BMM. Wrap attn.qkv and intercept the forward via a pre/post
        # hook on attn itself.
        # Hook on attn.qkv output → get Q/K/V. Hook on attn → monkey-patch.
        # Easier: monkey-patch attn.forward to capture.
        orig_forward = attn.forward

        def make_attn_forward(orig_fwd, idx):
            def new_forward(_self, x, *a, **k):
                B, N, C = x.shape
                H = _self.num_heads
                D = C // H
                qkv = _self.qkv(x).reshape(B, N, 3, H, D)
                q, _k, v = torch.unbind(qkv, 2)
                q = q.transpose(1, 2)  # (B, H, N, D)
                _k = _k.transpose(1, 2)
                v = v.transpose(1, 2)
                # qk metric: per-head |q|.amax across (B, N, D)
                q_perhead = q.float().abs().amax(dim=(0, 2, 3))  # (H,)
                captured_attn[("qk", idx)].append(q_perhead.detach().cpu())
                # Now compute attention to capture softmax output
                attn_raw = (q @ _k.transpose(-2, -1)) * _self.scale
                attn_sm = attn_raw.softmax(dim=-1)
                # av metric: per-row amax (per (b, h, query))
                av_metric = attn_sm.float().amax(dim=-1).reshape(-1)  # (B*H*N,)
                captured_attn[("av", idx)].append(av_metric.detach().cpu())
                # Continue standard forward
                out = attn_sm @ v
                out = out.transpose(1, 2).reshape(B, N, C)
                out = _self.proj(out)
                out = _self.proj_drop(out) if hasattr(_self, "proj_drop") else out
                return out
            return new_forward

        attn.forward = make_attn_forward(orig_forward, b_idx).__get__(
            attn, type(attn))

    with torch.no_grad():
        for x, _ in loader:
            _ = model(x.to(device))
            break

    # Aggregate per-op stats
    summary = {}
    for op in ("qkv_proj", "out_proj", "mlp_fc1", "mlp_fc2", "qk", "av"):
        source = captured_linear if op in ("qkv_proj", "out_proj",
                                            "mlp_fc1", "mlp_fc2") else captured_attn
        per_block_stats = []
        for b_idx in range(len(blocks)):
            key = (op, b_idx)
            if key not in source:
                continue
            v = torch.cat(source[key])
            m = v.max().item()
            mn = v.mean().item()
            md = v.median().item()
            p99 = torch.quantile(v, 0.99).item()
            p50 = torch.quantile(v, 0.50).item()
            p5 = torch.quantile(v, 0.05).item()
            per_block_stats.append({
                "block": b_idx, "N": int(v.numel()),
                "max": m, "mean": mn, "median": md,
                "max_over_mean": m / max(mn, 1e-9),
                "p99_over_p50": p99 / max(p50, 1e-9),
                "p50_over_p5":  p50 / max(p5, 1e-9),
            })
        mom = [s["max_over_mean"] for s in per_block_stats]
        pop = [s["p99_over_p50"] for s in per_block_stats]
        p50p5 = [s["p50_over_p5"] for s in per_block_stats]
        mid = per_block_stats[len(per_block_stats)//2]
        summary[op] = {
            "n_blocks": len(per_block_stats),
            "rows_per_block": per_block_stats[0]["N"] if per_block_stats else 0,
            "max_over_mean": ratio_stats(mom),
            "p99_over_p50":  ratio_stats(pop),
            "p50_over_p5":   ratio_stats(p50p5),
            "sample_block_12": {
                "block": mid["block"],
                "mean": round(mid["mean"], 4),
                "median": round(mid["median"], 4),
                "p99": round(torch.quantile(torch.cat(source[(op, mid["block"])]),
                                             0.99).item(), 4),
                "max": round(mid["max"], 4),
            },
        }

    # Print report
    print()
    print(f"{'op':<11s}  rows/blk  max/mean     p99/p50      p50/p5   sample b12")
    print("-" * 85)
    for op, st in summary.items():
        mom = st["max_over_mean"]
        pop = st["p99_over_p50"]
        pp5 = st["p50_over_p5"]
        sb = st["sample_block_12"]
        print(f"{op:<11s}  {st['rows_per_block']:>8d}  "
              f"{mom['min']:.2f}/{mom['mean']:.2f}/{mom['max']:.2f}   "
              f"{pop['min']:.2f}/{pop['mean']:.2f}/{pop['max']:.2f}   "
              f"{pp5['min']:.2f}/{pp5['mean']:.2f}/{pp5['max']:.2f}   "
              f"μ={sb['mean']} p99={sb['p99']} max={sb['max']}")

    out = REPO / "results" / "e2e" / "all_ops_distribution.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
