"""INT fixed-point baseline for ViT, granularity-matched to the SC path.

Ported from scmp_llm's PTQ protocol (`benchmark/quant/ptq.py`:
`_fake_quant` / `_fake_quant_lastdim_chunks` / `QuantConfig`) so INT rows in a
vit_sc figure mean the same thing they do in the LLM tables.

The point is a comparison that isolates *what SC costs you*, so every tensor is
quantized on the same grid **and the same scale scope** the SC path uses, and
only the MAC differs:

* SC  — quantize, then estimate the dot product from a length-L bit stream.
* INT — quantize identically, then compute the dot product exactly.

Whatever is left over is stochastic-stream variance, not a quantizer gap.

Scale-scope map — each row mirrors what ``model/sc_vit.py`` actually calls:

| op | SC call | INT here |
|---|---|---|
| `nn.Linear` | `per_row` + `chunk_d=128` | per-row × per-128-wide input chunk |
| Q@K^T | `per_head` (one range per (B,H)) | one scale per (B,H) slice |
| attn@V | `per_row`, no chunking | per-row, no chunking |

The linear row is exactly scmp_llm's `chunk_size=128` rule. The **attention
rows are not**: scmp_llm quantizes QK/AV operands per-row
(`_int_attention_matmul_ab_t` reduces over the last dim) while its SC attention
runs `per_head`, which hands INT a finer scale scope than SC gets. For a ViT —
where attention is a much larger share of the SC surface than in an LLM — that
would flatter INT, so the default here matches SC instead. Pass
``attn_scope="per_row"`` to reproduce scmp_llm's literal behaviour.

SmoothQuant is *not* applied. scmp_llm's protocol uses α=0.5 for both its INT
and SC cells; vit_sc's SC path passes ``smooth_scales=None``, so enabling it on
one side only would break the comparison. Keep both off, or add both.

Level counting (the figure's x-axis):

| SC halved stream | effective levels | INT equivalent |
|---|---|---|
| 32 | 64 | int6 |
| 64 | 128 | int7 |
| 96 | 192 | *(no power-of-two INT)* |
| 128 | 256 | int8 |
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["IntConfig", "IntLinear", "patch_model_int",
           "make_int_attention_forward", "fake_quant", "fake_quant_chunks"]


def fake_quant(x: torch.Tensor, bits: int, sym: bool, dim: int = -1) -> torch.Tensor:
    """RTN quantize→dequantize, scale/zp reduced over ``dim``.

    Verbatim semantics from scmp_llm ``benchmark/quant/ptq._fake_quant``:
    symmetric spans ``[-2**(bits-1), 2**(bits-1)-1]`` (a full ``2**bits``
    levels, asymmetric clamp bounds); asymmetric spans ``[0, 2**bits-1]``
    with a zero point.
    """
    if bits >= 16:
        return x
    xf = x.float()
    if sym:
        qmax = float(2 ** (bits - 1) - 1)
        qmin = -float(2 ** (bits - 1))
        amax = xf.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8)
        scale = amax / qmax
        deq = torch.clamp(torch.round(xf / scale), qmin, qmax) * scale
    else:
        qmax = float(2 ** bits - 1)
        xmin = xf.amin(dim=dim, keepdim=True)
        xmax = xf.amax(dim=dim, keepdim=True)
        scale = (xmax - xmin).clamp_min(1e-8) / qmax
        zp = torch.round(-xmin / scale)
        q = torch.clamp(torch.round(xf / scale) + zp, 0.0, qmax)
        deq = (q - zp) * scale
    return deq.to(x.dtype)


def fake_quant_chunks(x: torch.Tensor, bits: int, sym: bool,
                      chunk_size: int) -> torch.Tensor:
    """RTN fake-quant independently over chunks of the last dimension.

    Matches the SC linear fast path's ``chunk_d`` scale scope: each row gets a
    separate scale per ``chunk_size``-wide slice of the reduction dimension.
    The trailing short slice is quantized as-is.
    """
    if bits >= 16:
        return x
    if chunk_size <= 0 or x.shape[-1] <= chunk_size:
        return fake_quant(x, bits, sym, dim=-1)
    return torch.cat(
        [fake_quant(x[..., s:s + chunk_size], bits, sym, dim=-1)
         for s in range(0, x.shape[-1], chunk_size)], dim=-1)


def _fake_quant_perhead(x: torch.Tensor, bits: int, sym: bool) -> torch.Tensor:
    """One scale per leading-dim slice of a (BH, N, D) tensor — what
    ``sc_matmul(granularity="per_head")`` does (range over the whole N×D slice).
    """
    flat = x.reshape(x.shape[0], -1)                 # (BH, N*D)
    return fake_quant(flat, bits, sym, dim=-1).reshape(x.shape)


class IntConfig:
    """INT-b knobs, mirroring scmp_llm's ``QuantConfig`` for the W=A case."""

    def __init__(self, bits: int = 8, sym: bool = True,
                 chunk_size: int = 128, quantize_attention: bool = True,
                 attn_scope: str = "per_head"):
        self.bits = int(bits)
        self.sym = bool(sym)
        self.chunk_size = int(chunk_size)
        self.quantize_attention = bool(quantize_attention)
        if attn_scope not in ("per_head", "per_row"):
            raise ValueError(f"attn_scope must be per_head|per_row, got {attn_scope!r}")
        self.attn_scope = attn_scope

    @property
    def levels(self) -> int:
        return 1 << self.bits

    def tag(self) -> str:
        return f"W{self.bits}A{self.bits}_{'symm' if self.sym else 'asymm'}"

    def describe(self) -> str:
        return (f"{self.tag()} (levels={self.levels}) RTN, per-row act + "
                f"per-output-row weight, input chunk={self.chunk_size}, "
                f"attn={'on' if self.quantize_attention else 'off'}"
                f"({self.attn_scope}), no SmoothQuant")


class IntLinear(nn.Module):
    """Drop-in INT-b replacement wrapping an existing ``nn.Linear``.

    Same wrap-a-module convention as ``model.sc_vit.SCLinear``, so both
    patchers share the op-selection code.
    """

    def __init__(self, linear: nn.Linear, cfg: IntConfig,
                 operator: str | None = None, block_idx: int | None = None):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.cfg = cfg
        self.operator = operator
        self.block_idx = block_idx
        # Weights are static: quantize once, exactly like scmp_llm's QuantLinear.
        self.register_buffer(
            "_wq",
            fake_quant_chunks(linear.weight.detach().float(),
                              cfg.bits, cfg.sym, cfg.chunk_size),
            persistent=False)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"{self.cfg.tag()}, chunk={self.cfg.chunk_size}, "
                f"op={self.operator}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.shape, x.dtype
        a = fake_quant_chunks(x.reshape(-1, shape[-1]).float(),
                              self.cfg.bits, self.cfg.sym, self.cfg.chunk_size)
        y = a @ self._wq.t()
        if self.bias is not None:
            y = y + self.bias.float()
        return y.to(dtype).reshape(*shape[:-1], self.out_features)


def make_int_attention_forward(cfg: IntConfig, int_qk: bool, int_av: bool):
    """``forward(self, x, attn_bias=None)`` with INT Q@K^T and/or INT attn@V."""

    def _q_attn(t, per_head):
        if per_head and cfg.attn_scope == "per_head":
            return _fake_quant_perhead(t, cfg.bits, cfg.sym)
        return fake_quant(t, cfg.bits, cfg.sym, dim=-1)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        H = self.num_heads
        D = C // H
        q, k, v = torch.unbind(self.qkv(x).reshape(B, N, 3, H, D), 2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if int_qk:
            qf = _q_attn(q.float().reshape(B * H, N, D), True)
            kf = _q_attn(k.float().reshape(B * H, N, D), True)
            attn = (qf @ kf.transpose(-2, -1)).reshape(B, H, N, N)
            attn = attn.to(x.dtype) * self.scale
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if int_av:
            # SC runs AV as per_row with no chunking -> plain per-row here.
            a3 = fake_quant(attn.float().reshape(B * H, N, N),
                            cfg.bits, cfg.sym, dim=-1)
            v3 = fake_quant(v.float().transpose(-1, -2).reshape(B * H, D, N),
                            cfg.bits, cfg.sym, dim=-1)
            out = (a3 @ v3.transpose(-2, -1)).reshape(B, H, N, D).to(x.dtype)
        else:
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))

    return forward


def patch_model_int(model, bits: int = 8, *, ops=None, sym: bool = True,
                    chunk_size: int = 128, attn_scope: str = "per_head") -> dict:
    """Swap INT-b ops into a DINOv2-style ViT in place. Returns patch counts.

    ``ops`` takes the same names as ``model.sc_vit.patch_model``'s ``sc_ops``
    ({mlp_fc1, mlp_fc2, qk, av, proj}); ``None`` = all of them.
    """
    from .sc_vit import parse_sc_ops, _discover_blocks, _find_attn

    sel = (parse_sc_ops(ops) if ops is not None
           else {"mlp_fc1", "mlp_fc2", "qk", "av", "proj"})
    cfg = IntConfig(bits, sym=sym, chunk_size=chunk_size,
                    attn_scope=attn_scope)
    stats = {"attn": 0, "mlp_fc1": 0, "mlp_fc2": 0,
             "qkv_proj": 0, "out_proj": 0}

    for i, blk in enumerate(_discover_blocks(model)):
        attn = _find_attn(blk)
        if attn is not None and ("qk" in sel or "av" in sel):
            fwd = make_int_attention_forward(cfg, "qk" in sel, "av" in sel)
            attn.forward = fwd.__get__(attn, type(attn))
            stats["attn"] += 1
        if attn is not None and "proj" in sel:
            if isinstance(attn.qkv, nn.Linear):
                attn.qkv = IntLinear(attn.qkv, cfg, "qkv_proj", i)
                stats["qkv_proj"] += 1
            if isinstance(attn.proj, nn.Linear):
                attn.proj = IntLinear(attn.proj, cfg, "out_proj", i)
                stats["out_proj"] += 1
        mlp = getattr(blk, "mlp", None)
        if mlp is None:
            continue
        if "mlp_fc1" in sel and isinstance(getattr(mlp, "fc1", None), nn.Linear):
            mlp.fc1 = IntLinear(mlp.fc1, cfg, "mlp_fc1", i)
            stats["mlp_fc1"] += 1
        if "mlp_fc2" in sel and isinstance(getattr(mlp, "fc2", None), nn.Linear):
            mlp.fc2 = IntLinear(mlp.fc2, cfg, "mlp_fc2", i)
            stats["mlp_fc2"] += 1
    return stats
