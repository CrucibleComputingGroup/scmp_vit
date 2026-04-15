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
from sc_integration.noise_matmul import noisy_sc_matmul_grouped, _noisy_matmul_core  # noqa: E402
from config_helpers import make_sobol_simple_config  # noqa: E402

# Module-level toggle: when True, SC matmuls use the calibrated Gaussian-noise
# surrogate (sc_integration/noise_matmul.py) instead of the real bitstream /
# enable-signal kernels. Orders of magnitude faster — for sensitivity sweeps.
USE_NOISE_MODEL = False


def set_noise_model(enabled: bool):
    global USE_NOISE_MODEL
    USE_NOISE_MODEL = bool(enabled)


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


def _sc_linear(x: torch.Tensor, weight: torch.Tensor, bias, sc_prec: int,
               mode: str = "bipolar"):
    """y = x @ weight^T + bias via SC.

    mode="bipolar": per-row symmetric quant (|max|), q_max = 2^(sc_prec-1)-1.
        Good for near-zero-mean tensors (post-LN activations, weights).
    mode="unipolar": per-row asymmetric quant (scale = range/q_max, zero-point),
        q_max = 2^sc_prec - 1 → ~2× grid resolution. Good for heavy-tailed /
        asymmetric tensors (post-GELU activations for fc2).
    """
    orig_shape = x.shape
    D_in = orig_shape[-1]
    x2 = x.reshape(-1, D_in).float()
    w = weight.float()  # (D_out, D_in)
    if USE_NOISE_MODEL:
        y = noisy_sc_matmul_grouped(x2, w, group_a=1, group_b=1,
                                    mode=mode, sc_prec=sc_prec)
    else:
        cfg = _get_config(D_in, sc_prec)
        y = sc_matmul_grouped_bipolar(x2, w, group_a=1, group_b=1,
                                      mode=mode, sc_prec=sc_prec, config=cfg)
    if bias is not None:
        y = y + bias.float()
    y = y.to(x.dtype).reshape(*orig_shape[:-1], w.shape[0])
    return y


def make_sc_attention_forward(sc_prec: int, sc_av: bool, sc_qk: bool = True):
    """Return a forward(self, x) with optional SC Q@K^T and SC Attn@V."""

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        H = self.num_heads
        D = C // H
        qkv = self.qkv(x).reshape(B, N, 3, H, D)
        q, k, v = torch.unbind(qkv, 2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if sc_qk:
            if USE_NOISE_MODEL:
                # (B,H,N,D) @ (B,H,N,D)^T via surrogate (per-head amax scale).
                attn = _noisy_matmul_core(
                    q.float(), k.float(), L=2 ** sc_prec,
                    mode="bipolar", per_row_scale=False,
                ).to(x.dtype) * self.scale
            else:
                q_int, q_scale = quantize_for_sc_per_head(q.float(), sc_prec)
                k_int, k_scale = quantize_for_sc_per_head(k.float(), sc_prec)
                cfg_qk = _get_config(D, sc_prec)
                attn_sc = sc_matmul_qk_multihead(q_int, k_int, sc_prec=sc_prec, config=cfg_qk)
                attn = dequantize_sc_result(attn_sc, q_scale, k_scale)
                attn = attn.to(x.dtype) * self.scale
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if sc_av:
            # Use sc_matmul_grouped (unipolar + signs) with per-row quant to
            # handle the very wide dynamic range of softmax'd attn.
            a_flat = attn.float().reshape(B * H, N, N)
            b_flat = v.float().transpose(-1, -2).reshape(B * H, D, N)
            if USE_NOISE_MODEL:
                out = _noisy_matmul_core(
                    a_flat, b_flat, L=2 ** sc_prec,
                    mode="bipolar", per_row_scale=True,
                ).reshape(B, H, N, D).to(x.dtype)
            else:
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
    """Drop-in replacement for nn.Linear that runs matmul via SC.

    mode selects symmetric (bipolar) or asymmetric zero-point (unipolar) quant.
    """

    def __init__(self, linear: nn.Linear, sc_prec: int, mode: str = "bipolar"):
        super().__init__()
        self.sc_prec = sc_prec
        self.mode = mode
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x):
        return _sc_linear(x, self.weight, self.bias, self.sc_prec, mode=self.mode)


# Canonical op names for the sc_ops vector API.
SC_OP_NAMES = ("mlp_fc1", "mlp_fc2", "qk", "av", "proj")


def _discover_blocks(model):
    """Return the list of transformer blocks in a DINOv2-style model."""
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        return list(model.backbone.blocks)
    if hasattr(model, "blocks"):
        return list(model.blocks)
    raise RuntimeError("Could not locate transformer blocks on model.")


def normalize_sc_ops_per_block(spec, n_blocks):
    """Normalize a per-block-per-op spec into a complete dict.

    Accepts ``spec`` as ``dict[op_name, list[int]]`` where each list has
    length ``n_blocks`` (or length 1 — broadcast — or a scalar 0/1). Missing
    op keys default to all-zero. Returns a dict with every key in
    ``SC_OP_NAMES`` mapped to a length-``n_blocks`` tuple of 0/1 ints.
    """
    if spec is None:
        return {k: tuple([0] * n_blocks) for k in SC_OP_NAMES}
    if not isinstance(spec, dict):
        raise TypeError("sc_ops_per_block must be a dict[op_name, list]")
    out = {}
    for k in SC_OP_NAMES:
        v = spec.get(k, 0)
        if isinstance(v, (int, bool)):
            out[k] = tuple([int(bool(v))] * n_blocks)
            continue
        v = list(v)
        if len(v) == 1:
            out[k] = tuple([int(bool(v[0]))] * n_blocks)
        elif len(v) == n_blocks:
            out[k] = tuple(int(bool(x)) for x in v)
        else:
            raise ValueError(
                f"sc_ops_per_block[{k!r}] length {len(v)} != n_blocks {n_blocks}")
    extra = set(spec) - set(SC_OP_NAMES)
    if extra:
        raise ValueError(f"unknown op names in sc_ops_per_block: {sorted(extra)}")
    return out


def parse_sc_ops(spec):
    """Normalize an ``sc_ops`` spec into a set of canonical op names.

    Accepts any of:
      * a set/list/tuple of names drawn from SC_OP_NAMES
      * a 5-element 0/1 vector in SC_OP_NAMES order
      * a comma-separated string of names (e.g. "mlp_fc1,av")
    Returns a set[str].
    """
    if spec is None:
        return set()
    if isinstance(spec, str):
        items = [s.strip() for s in spec.split(",") if s.strip()]
        return parse_sc_ops(items)
    items = list(spec)
    if len(items) == len(SC_OP_NAMES) and all(isinstance(x, (int, bool)) for x in items):
        return {n for n, v in zip(SC_OP_NAMES, items) if v}
    out = set()
    for x in items:
        if x not in SC_OP_NAMES:
            raise ValueError(f"unknown sc op: {x!r} (valid: {SC_OP_NAMES})")
        out.add(x)
    return out


def patch_model(model, sc_prec: int = 8,
                sc_qk: bool = True, sc_av: bool = False, sc_mlp: bool = False,
                sc_qkv_proj: bool = False, sc_out_proj: bool = False,
                mlp_skip_last_k: int = 0,
                mlp_skip_first_k: int = 0,
                sc_mlp_mode: str = "bipolar",
                sc_proj_mode: str = "bipolar",
                mlp_fc_mask: str = "both",  # "both" | "fc1" | "fc2"
                mlp_sc_spec=None,           # list[tuple[int, str]] | None
                sc_ops=None,                # set/vec/str over SC_OP_NAMES
                sc_ops_per_block=None):     # dict[op_name, list[0/1]] length n_blocks
    """Apply SC patches to a DINOv2 ViT model in-place.

    Priority: ``sc_ops_per_block`` > ``sc_ops`` > legacy per-op bools.

    ``sc_ops`` (when given) takes priority over per-op bool flags. It accepts
    any form understood by ``parse_sc_ops`` — a set of names, a 5-vector
    ``[mlp_fc1, mlp_fc2, qk, av, proj]`` of 0/1, or a comma-separated string.
    The selected ops are turned on uniformly across all 24 blocks (no block-
    level granularity); "proj" enables both qkv_proj and out_proj together.

    ``mlp_sc_spec`` (if given) is an explicit list of
    ``(block_idx, "fc1"|"fc2")`` pairs naming exactly which MLP linears go SC.
    When set, it overrides ``mlp_skip_first_k`` / ``mlp_skip_last_k`` /
    ``mlp_fc_mask`` / ``sc_ops`` for MLP selection.

    Returns a dict counting patched modules.
    """
    # sc_ops dispatches to per-op bools (unless mlp_sc_spec is set for MLP).
    if sc_ops is not None:
        ops = parse_sc_ops(sc_ops)
        sc_qk = "qk" in ops
        sc_av = "av" in ops
        sc_qkv_proj = "proj" in ops
        sc_out_proj = "proj" in ops
        if mlp_sc_spec is None:
            want_fc1 = "mlp_fc1" in ops
            want_fc2 = "mlp_fc2" in ops
            sc_mlp = want_fc1 or want_fc2
            if want_fc1 and want_fc2:
                mlp_fc_mask = "both"
            elif want_fc1:
                mlp_fc_mask = "fc1"
            elif want_fc2:
                mlp_fc_mask = "fc2"
    stats = {"attn": 0, "mlp_fc1": 0, "mlp_fc2": 0, "qkv_proj": 0, "out_proj": 0}

    # ---- highest-priority path: per-block per-op vectors ----
    if sc_ops_per_block is not None:
        blocks = _discover_blocks(model)
        pbm = normalize_sc_ops_per_block(sc_ops_per_block, len(blocks))
        for i, blk in enumerate(blocks):
            # Locate attention submodule (the one with qkv/proj/num_heads)
            attn_mod = None
            for m in blk.modules():
                if (hasattr(m, "qkv") and hasattr(m, "proj")
                        and hasattr(m, "num_heads")):
                    attn_mod = m
                    break
            q_on, a_on, p_on = pbm["qk"][i], pbm["av"][i], pbm["proj"][i]
            if attn_mod is not None and (q_on or a_on):
                fwd = make_sc_attention_forward(
                    sc_prec, sc_av=bool(a_on), sc_qk=bool(q_on))
                attn_mod.forward = fwd.__get__(attn_mod, type(attn_mod))
                stats["attn"] += 1
            if attn_mod is not None and p_on:
                if isinstance(attn_mod.qkv, nn.Linear):
                    attn_mod.qkv = SCLinear(attn_mod.qkv, sc_prec, mode=sc_proj_mode)
                    stats["qkv_proj"] += 1
                if isinstance(attn_mod.proj, nn.Linear):
                    attn_mod.proj = SCLinear(attn_mod.proj, sc_prec, mode=sc_proj_mode)
                    stats["out_proj"] += 1
            mlp = getattr(blk, "mlp", None)
            if mlp is not None:
                if pbm["mlp_fc1"][i] and isinstance(getattr(mlp, "fc1", None), nn.Linear):
                    mlp.fc1 = SCLinear(mlp.fc1, sc_prec, mode=sc_mlp_mode)
                    stats["mlp_fc1"] += 1
                if pbm["mlp_fc2"][i] and isinstance(getattr(mlp, "fc2", None), nn.Linear):
                    mlp.fc2 = SCLinear(mlp.fc2, sc_prec, mode=sc_mlp_mode)
                    stats["mlp_fc2"] += 1
        return stats

    if sc_qk or sc_av:
        fwd = make_sc_attention_forward(sc_prec, sc_av=sc_av, sc_qk=sc_qk)
        for m in model.modules():
            if hasattr(m, "qkv") and hasattr(m, "proj") and hasattr(m, "num_heads"):
                m.forward = fwd.__get__(m, type(m))
                stats["attn"] += 1

    if sc_qkv_proj or sc_out_proj:
        for m in model.modules():
            if hasattr(m, "qkv") and hasattr(m, "proj") and hasattr(m, "num_heads"):
                if sc_qkv_proj and isinstance(m.qkv, nn.Linear):
                    m.qkv = SCLinear(m.qkv, sc_prec, mode=sc_proj_mode)
                    stats["qkv_proj"] += 1
                if sc_out_proj and isinstance(m.proj, nn.Linear):
                    m.proj = SCLinear(m.proj, sc_prec, mode=sc_proj_mode)
                    stats["out_proj"] += 1

    if sc_mlp:
        blocks = None
        if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
            blocks = list(model.backbone.blocks)
        elif hasattr(model, "blocks"):
            blocks = list(model.blocks)

        if mlp_sc_spec is not None:
            # Explicit list mode. Build per-block set of linears to patch.
            selected = {}  # block_idx -> set of "fc1"/"fc2"
            for item in mlp_sc_spec:
                bi, fc = item
                assert fc in ("fc1", "fc2"), f"bad linear name: {fc}"
                selected.setdefault(int(bi), set()).add(fc)
            if blocks is None:
                raise RuntimeError("mlp_sc_spec requires discoverable blocks")
            for bi, blk in enumerate(blocks):
                if not hasattr(blk, "mlp"):
                    continue
                mlp = blk.mlp
                want = selected.get(bi, set())
                if "fc1" in want and isinstance(getattr(mlp, "fc1", None), nn.Linear):
                    mlp.fc1 = SCLinear(mlp.fc1, sc_prec, mode=sc_mlp_mode)
                    stats["mlp_fc1"] += 1
                if "fc2" in want and isinstance(getattr(mlp, "fc2", None), nn.Linear):
                    mlp.fc2 = SCLinear(mlp.fc2, sc_prec, mode=sc_mlp_mode)
                    stats["mlp_fc2"] += 1
        else:
            # Legacy first-k / last-k block skipping path.
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
                    if mlp_fc_mask in ("both", "fc1") and isinstance(m.fc1, nn.Linear):
                        m.fc1 = SCLinear(m.fc1, sc_prec, mode=sc_mlp_mode)
                        stats["mlp_fc1"] += 1
                    if mlp_fc_mask in ("both", "fc2") and isinstance(m.fc2, nn.Linear):
                        m.fc2 = SCLinear(m.fc2, sc_prec, mode=sc_mlp_mode)
                        stats["mlp_fc2"] += 1

    return stats


# Back-compat alias
def patch_model_attention(model, sc_prec: int = 8):
    return patch_model(model, sc_prec=sc_prec, sc_qk=True)["attn"]
