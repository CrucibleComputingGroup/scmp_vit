"""Monkey-patch DINOv2 attention + MLP to use stochastic computing matmuls.

- Q @ K^T: per-head SC (bipolar/XNOR) via sc_matmul_qk_multihead.
- Attn @ V: per-head SC via sc_matmul (bipolar).
- MLP fc1 / fc2: SC linear via sc_matmul (bipolar, per-tensor symmetric range).

Softmax, layernorms, activations, and qkv/proj linears stay FP.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Local SC primitives (vit_sc/sc) and thin integration wrappers
# (vit_sc/sc_integration).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sc"))
sys.path.insert(0, str(HERE))

from sc_integration.sc_matmul import (  # noqa: E402
    sc_matmul_qk_multihead,
    quantize_for_sc_per_head,
    dequantize_sc_result,
)
from sc_triton import sc_matmul_grouped_enable_triton as sc_matmul_grouped_bipolar  # noqa: E402
from config_helpers import make_sobol_simple_config  # noqa: E402


_CFG_CACHE = {}


def _get_config(D, sc_prec):
    key = (D, sc_prec)
    if key not in _CFG_CACHE:
        _CFG_CACHE[key] = make_sobol_simple_config(D, D, sc_prec)
    return _CFG_CACHE[key]


def _sym_range(t: torch.Tensor) -> tuple[float, float]:
    m = float(t.detach().abs().max().item())
    m = max(m, 1e-6)
    return m, -m


def _sc_linear(x: torch.Tensor, weight: torch.Tensor, bias, sc_prec: int):
    """y = x @ weight^T + bias; SC with per-row (per-token) / per-output-channel
    quantization via sc_matmul_grouped (unipolar + signs)."""
    orig_shape = x.shape
    D_in = orig_shape[-1]
    x2 = x.reshape(-1, D_in).float()
    w = weight.float()  # (D_out, D_in)
    cfg = _get_config(D_in, sc_prec)
    y = sc_matmul_grouped_bipolar(x2, w, group_a=1, group_b=1,
                                  mode="bipolar", sc_prec=sc_prec, config=cfg)
    if bias is not None:
        y = y + bias.float()
    y = y.to(x.dtype).reshape(*orig_shape[:-1], w.shape[0])
    return y


def make_sc_attention_forward(sc_prec: int, sc_av: bool):
    """Return a forward(self, x) with SC Q@K^T and optionally SC Attn@V."""

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        H = self.num_heads
        D = C // H
        qkv = self.qkv(x).reshape(B, N, 3, H, D)
        q, k, v = torch.unbind(qkv, 2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q_int, q_scale = quantize_for_sc_per_head(q.float(), sc_prec)
        k_int, k_scale = quantize_for_sc_per_head(k.float(), sc_prec)
        cfg_qk = _get_config(D, sc_prec)
        attn_sc = sc_matmul_qk_multihead(q_int, k_int, sc_prec=sc_prec, config=cfg_qk)
        attn = dequantize_sc_result(attn_sc, q_scale, k_scale)
        attn = attn.to(x.dtype) * self.scale
        attn = attn.softmax(dim=-1)

        if sc_av:
            # Use sc_matmul_grouped (unipolar + signs) with per-row quant to
            # handle the very wide dynamic range of softmax'd attn.
            a_flat = attn.float().reshape(B * H, N, N)
            b_flat = v.float().transpose(-1, -2).reshape(B * H, D, N)
            cfg_av = _get_config(N, sc_prec)
            out = torch.empty(B * H, N, D, dtype=torch.float32, device=x.device)
            for i in range(B * H):
                out[i] = sc_matmul_grouped_bipolar(
                    a_flat[i], b_flat[i], group_a=1, group_b=1,
                    mode="bipolar", sc_prec=sc_prec, config=cfg_av,
                )
            out = out.to(x.dtype).reshape(B, H, N, D)
        else:
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    return forward


class SCLinear(nn.Module):
    """Drop-in replacement for nn.Linear that runs matmul via SC."""

    def __init__(self, linear: nn.Linear, sc_prec: int):
        super().__init__()
        self.sc_prec = sc_prec
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x):
        return _sc_linear(x, self.weight, self.bias, self.sc_prec)


def patch_model(model, sc_prec: int = 8,
                sc_qk: bool = True, sc_av: bool = False, sc_mlp: bool = False,
                sc_qkv_proj: bool = False, sc_out_proj: bool = False,
                mlp_skip_last_k: int = 0,
                mlp_skip_first_k: int = 0):
    """Apply SC patches to a DINOv2 ViT model in-place.

    Returns a dict counting patched modules.
    """
    stats = {"attn": 0, "mlp_fc1": 0, "mlp_fc2": 0, "qkv_proj": 0, "out_proj": 0}

    if sc_qk or sc_av:
        fwd = make_sc_attention_forward(sc_prec, sc_av=sc_av)
        for m in model.modules():
            if hasattr(m, "qkv") and hasattr(m, "proj") and hasattr(m, "num_heads"):
                m.forward = fwd.__get__(m, type(m))
                stats["attn"] += 1

    if sc_qkv_proj or sc_out_proj:
        for m in model.modules():
            if hasattr(m, "qkv") and hasattr(m, "proj") and hasattr(m, "num_heads"):
                if sc_qkv_proj and isinstance(m.qkv, nn.Linear):
                    m.qkv = SCLinear(m.qkv, sc_prec)
                    stats["qkv_proj"] += 1
                if sc_out_proj and isinstance(m.proj, nn.Linear):
                    m.proj = SCLinear(m.proj, sc_prec)
                    stats["out_proj"] += 1

    if sc_mlp:
        blocks = None
        if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
            blocks = list(model.backbone.blocks)
        elif hasattr(model, "blocks"):
            blocks = list(model.blocks)
        skip_mlps = set()
        if blocks is not None and mlp_skip_last_k > 0:
            for blk in blocks[-mlp_skip_last_k:]:
                if hasattr(blk, "mlp"):
                    skip_mlps.add(id(blk.mlp))
        if blocks is not None and mlp_skip_first_k > 0:
            for blk in blocks[:mlp_skip_first_k]:
                if hasattr(blk, "mlp"):
                    skip_mlps.add(id(blk.mlp))
        for m in model.modules():
            if m.__class__.__name__ == "Mlp" and hasattr(m, "fc1") and hasattr(m, "fc2"):
                if id(m) in skip_mlps:
                    continue
                if isinstance(m.fc1, nn.Linear):
                    m.fc1 = SCLinear(m.fc1, sc_prec)
                    stats["mlp_fc1"] += 1
                if isinstance(m.fc2, nn.Linear):
                    m.fc2 = SCLinear(m.fc2, sc_prec)
                    stats["mlp_fc2"] += 1

    return stats


# Back-compat alias
def patch_model_attention(model, sc_prec: int = 8):
    return patch_model(model, sc_prec=sc_prec, sc_qk=True)["attn"]
