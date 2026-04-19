"""Module-swap SC patching for an EVA ViTDet detector.

Walks ``model.backbone.net.blocks`` (the EVA ``ViT`` transformer stack) and
swaps primitives with stochastic-computing equivalents according to a
per-(op, block) schedule.

Reusable from ``vit_sc``:
  - ``SCLinear`` wraps ``nn.Linear`` → SC linear (vit_sc.sc_attention_patch).

New here:
  - ``SCMatMul``  (./sc_matmul.py) wraps EVA's ``MatMul`` markers.

Op taxonomy mirrors ``vit_sc``'s ``SC_OP_NAMES`` but is spelled out into six
distinct names (``proj`` expanded into ``qkv_proj`` / ``out_proj``):

    mlp_fc1, mlp_fc2, qkv_proj, out_proj, qk, av
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch.nn as nn

# Make vit_sc importable so sc_integration / sc resolve.
_VIT_SC = Path(__file__).resolve().parents[2]  # vit_sc/
if str(_VIT_SC) not in sys.path:
    sys.path.insert(0, str(_VIT_SC))
if str(_VIT_SC / "sc") not in sys.path:
    sys.path.insert(0, str(_VIT_SC / "sc"))

from sc_integration.sc_linear import SCLinear  # noqa: E402

from .sc_matmul import SCMatMul


SC_OP_NAMES = ("mlp_fc1", "mlp_fc2", "qkv_proj", "out_proj", "qk", "av")


def _discover_blocks(model):
    """Return EVA ViT transformer blocks. Accepts a few common wrapping paths."""
    if hasattr(model, "backbone") and hasattr(model.backbone, "net") \
            and hasattr(model.backbone.net, "blocks"):
        return list(model.backbone.net.blocks)
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        return list(model.backbone.blocks)
    if hasattr(model, "blocks"):
        return list(model.blocks)
    raise RuntimeError("Could not locate transformer blocks on model.")


def normalize_sc_ops_per_block(spec, n_blocks: int):
    """Normalize ``spec`` (dict[op_name -> list[0/1] | int]) to full
    dict[op_name -> tuple[int] * n_blocks]. Missing keys default to all-zero.
    Also accepts "proj" as an alias expanding to qkv_proj + out_proj.
    """
    if spec is None:
        return {k: tuple([0] * n_blocks) for k in SC_OP_NAMES}
    if not isinstance(spec, dict):
        raise TypeError("sc_ops_per_block must be dict[op_name, list]")

    # Expand "proj" alias.
    spec = dict(spec)
    if "proj" in spec:
        v = spec.pop("proj")
        spec.setdefault("qkv_proj", v)
        spec.setdefault("out_proj", v)

    unknown = set(spec) - set(SC_OP_NAMES)
    if unknown:
        raise ValueError(f"unknown sc_ops_per_block keys: {sorted(unknown)}")

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
                f"sc_ops_per_block[{k!r}] length {len(v)} != n_blocks {n_blocks}"
            )
    return out


def build_skip_last_fc2_schedule(n_blocks: int, skip_last_k: int):
    """Heuristic skip_worst: SC all (op, block) except mlp_fc2 for the last K
    blocks. Transfers the vit_sc finding that late-layer fc2 is most
    sensitive. A proper sensitivity-matrix-driven schedule for EVA-on-COCO
    requires a per-(op, block) sweep (follow-up).
    """
    sched = {op: [1] * n_blocks for op in SC_OP_NAMES}
    if skip_last_k > 0:
        for i in range(n_blocks - skip_last_k, n_blocks):
            sched["mlp_fc2"][i] = 0
    return sched


def load_schedule_json(path: str, n_blocks: int) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return normalize_sc_ops_per_block(raw, n_blocks)


def count_sc_ops(sched: dict) -> dict:
    """Per-op counts (#SC, #total) for a normalized schedule."""
    return {op: (int(sum(v)), len(v)) for op, v in sched.items()}


def sc_patch_eva(model,
                 sc_prec: int = 8,
                 sc_ops_per_block=None,
                 mlp_mode: str = "bipolar",
                 proj_mode: str = "bipolar",
                 qk_mode: str = "bipolar",
                 av_mode: str = "bipolar"):
    """Swap EVA-ViT primitives with SC versions in place.

    Parameters
    ----------
    sc_prec : int
        SC precision (stoc_len = 2 ** sc_prec).
    sc_ops_per_block : dict[op_name, list[0/1]] | None
        Per-(op, block) schedule. ``None`` ⇒ no-op.
    mlp_mode / proj_mode : "bipolar" | "unipolar"
        Quant mode for the SCLinear wrappers.
    qk_mode : "bipolar" | "unipolar"
        Quant mode for matmul1 (Q @ K^T). Q/K are signed post-LN → bipolar.
    av_mode : "bipolar" | "unipolar"
        Quant mode for matmul2 (Attn @ V). Although post-softmax attn ∈
        [0, 1], V is signed — unipolar mode requires BOTH operands to be
        non-negative, so bipolar is the safe default (verified: ~4.7% rel
        err vs. 69% for unipolar with signed V).

    Returns
    -------
    stats : dict
        Count of swapped modules per op.
    """
    blocks = _discover_blocks(model)
    sched = normalize_sc_ops_per_block(sc_ops_per_block, len(blocks))
    stats = {k: 0 for k in SC_OP_NAMES}

    for i, blk in enumerate(blocks):
        attn = getattr(blk, "attn", None)
        mlp = getattr(blk, "mlp", None)

        if attn is not None:
            if sched["qkv_proj"][i] and isinstance(getattr(attn, "qkv", None), nn.Linear):
                attn.qkv = SCLinear(attn.qkv, sc_prec=sc_prec, mode=proj_mode)
                stats["qkv_proj"] += 1
            if sched["out_proj"][i] and isinstance(getattr(attn, "proj", None), nn.Linear):
                attn.proj = SCLinear(attn.proj, sc_prec=sc_prec, mode=proj_mode)
                stats["out_proj"] += 1
            # matmul1 / matmul2: swap only if current module is the FP MatMul
            # marker — don't double-wrap.
            if sched["qk"][i]:
                m1 = getattr(attn, "matmul1", None)
                if m1 is not None and not isinstance(m1, SCMatMul):
                    attn.matmul1 = SCMatMul(sc_prec=sc_prec, mode=qk_mode)
                    stats["qk"] += 1
            if sched["av"][i]:
                m2 = getattr(attn, "matmul2", None)
                if m2 is not None and not isinstance(m2, SCMatMul):
                    attn.matmul2 = SCMatMul(sc_prec=sc_prec, mode=av_mode)
                    stats["av"] += 1

        if mlp is not None:
            if sched["mlp_fc1"][i] and isinstance(getattr(mlp, "fc1", None), nn.Linear):
                mlp.fc1 = SCLinear(mlp.fc1, sc_prec=sc_prec, mode=mlp_mode)
                stats["mlp_fc1"] += 1
            if sched["mlp_fc2"][i] and isinstance(getattr(mlp, "fc2", None), nn.Linear):
                mlp.fc2 = SCLinear(mlp.fc2, sc_prec=sc_prec, mode=mlp_mode)
                stats["mlp_fc2"] += 1

    return stats
