"""Smoke test for the scmp_kernels-backed ViT SC path.

Runs without network / DINOv2 weights: builds a DINOv2-shaped toy ViT (the
patcher keys off duck-typing — ``attn.qkv`` / ``attn.proj`` / ``attn.num_heads``
and ``block.mlp.fc1/fc2``), then checks that

1. the vendored ``vit_sc/kernels`` copy is the one that got imported,
2. ``patch_model`` swaps the modules the op-selection asks for,
3. SC forward runs and tracks FP within a sane error band,
4. shrinking ``stoc_len`` degrades monotonically (the SC quality knob works),
5. the MP paths (fixed / quantile-adaptive / range) all execute,
6. the precision trace records the calls.

Usage:  python tests/test_sc_vit_smoke.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.sc_vit import (  # noqa: E402
    SCLinear, patch_model, set_sc_defaults,
    MPConfig, AdaptiveMPConfig, RangeMPConfig,
)
from model._bootstrap import kernels_provenance, KERNELS_DIR  # noqa: E402
import scmp_kernels  # noqa: E402
from scmp_kernels import trace as sc_trace  # noqa: E402


# ---------------------------------------------------------------- toy model
class Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Identity()

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        H = self.num_heads
        D = C // H
        q, k, v = torch.unbind(self.qkv(x).reshape(B, N, 3, H, D), 2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, heads, hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attn(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ToyViT(nn.Module):
    """Same duck-type surface as DINOv2 ViT-L/14, but small."""

    def __init__(self, dim=256, heads=4, depth=3, hidden=1024):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(dim, heads, hidden) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x).mean(1))


def rel_err(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        raise SystemExit("needs a GPU (Triton SC kernels are CUDA-only)")

    ok = True

    # 1 -------------------------------------------------- kernel provenance
    prov = kernels_provenance()
    local = Path(scmp_kernels.__file__).resolve().parents[1] == KERNELS_DIR.resolve()
    print(f"[1] kernels: {prov}")
    print(f"    vendored copy in use: {local}")
    ok &= local

    # 2 --------------------------------------------------------- patch stats
    x = torch.randn(2, 32, 256, device=dev)
    fp_model = ToyViT().to(dev).eval()
    with torch.no_grad():
        ref = fp_model(x)

    import copy
    m = copy.deepcopy(fp_model)
    stats = patch_model(m, sc_prec=8, sc_ops="mlp_fc1,mlp_fc2,qk,av,proj",
                        chunk_d=128)
    print(f"[2] patch stats: {stats}")
    expect = {"attn": 3, "mlp_fc1": 3, "mlp_fc2": 3, "qkv_proj": 3, "out_proj": 3}
    ok &= stats == expect
    n_sclinear = sum(isinstance(mod, SCLinear) for mod in m.modules())
    print(f"    SCLinear modules: {n_sclinear} (expect 12)")
    ok &= n_sclinear == 12

    # 3 ------------------------------------------------------- SC vs FP
    with torch.no_grad():
        sc = m(x)
    e_full = rel_err(sc, ref)
    print(f"[3] all-SC rel_err @ stoc_len=256: {e_full:.4f}")
    ok &= e_full < 0.35
    agree = (sc.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"    argmax agreement: {agree:.2f}")

    # 4 ---------------------------------------- stoc_len quality monotonicity
    print("[4] stoc_len sweep (qk+av+mlp, per-call stream length):")
    errs = {}
    for sl in (256, 128, 64, 32, 16):
        mm = copy.deepcopy(fp_model)
        patch_model(mm, sc_prec=8, sc_ops="mlp_fc1,mlp_fc2,qk,av",
                    stoc_len=sl, chunk_d=128)
        with torch.no_grad():
            errs[sl] = rel_err(mm(x), ref)
        print(f"    stoc_len={sl:4d}  rel_err={errs[sl]:.4f}")
    mono = errs[256] < errs[64] < errs[16]
    print(f"    monotone (256 < 64 < 16): {mono}")
    ok &= mono

    # 5 ------------------------------------------------------------ MP paths
    print("[5] mixed-precision paths:")
    # fixed quantile MPConfig on linears + attention
    mp = MPConfig(stoc_len_levels=[256, 64], level_fractions=[0.5, 0.5])
    mm = copy.deepcopy(fp_model)
    patch_model(mm, sc_prec=8, sc_ops="mlp_fc1,mlp_fc2,qk,av",
                linear_mp_spec={"mlp_fc1": mp, "mlp_fc2": mp},
                attn_mp_spec={"qk": mp, "av": mp}, chunk_d=128)
    with torch.no_grad():
        print(f"    fixed MPConfig      rel_err={rel_err(mm(x), ref):.4f}")

    # adaptive via quantile target_fractions (no calibration table needed)
    ad = AdaptiveMPConfig(stoc_len_levels=[256, 64],
                          target_fractions=[0.5, 0.5])
    mm = copy.deepcopy(fp_model)
    patch_model(mm, sc_prec=8, sc_ops="mlp_fc1,mlp_fc2",
                linear_mp_spec={"mlp_fc1": ad, "mlp_fc2": ad}, chunk_d=128)
    with torch.no_grad():
        print(f"    AdaptiveMPConfig    rel_err={rel_err(mm(x), ref):.4f}")

    # range MP (per-weight-group, static)
    rg = RangeMPConfig(stoc_len_levels=[256, 64], base_threshold=0.3)
    mm = copy.deepcopy(fp_model)
    patch_model(mm, sc_prec=8, sc_ops="mlp_fc1",
                linear_mp_spec={"mlp_fc1": {"range": rg,
                                            "range_group_size": 128}},
                chunk_d=128)
    with torch.no_grad():
        print(f"    RangeMPConfig       rel_err={rel_err(mm(x), ref):.4f}")

    # 6 -------------------------------------------------------------- trace
    import json
    import tempfile
    trace_path = Path(tempfile.gettempdir()) / "vit_sc_smoke_trace.json"
    sc_trace.reset()
    sc_trace.enable(str(trace_path), mode="summary")
    mm = copy.deepcopy(fp_model)
    patch_model(mm, sc_prec=8, sc_ops="mlp_fc1,mlp_fc2,qk,av", chunk_d=128)
    with torch.no_grad():
        mm(x)
    sc_trace.flush()
    sc_trace.disable()
    with open(trace_path) as f:
        payload = json.load(f)
    groups = payload.get("groups", [])
    ops = sorted({g.get("op") for g in groups})
    total_macs = sum(g.get("macs", 0) for g in groups)
    print(f"[6] trace groups={len(groups)} ops={ops}")
    print(f"    total macs={total_macs:,}  -> {trace_path}")
    ok &= len(groups) > 0 and total_macs > 0

    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
