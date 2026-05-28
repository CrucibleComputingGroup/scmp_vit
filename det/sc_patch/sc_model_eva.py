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
from sc_integration.mp_linear import (  # noqa: E402
    MPConfig, RangeMPConfig, AdaptiveMPConfig,
)

from .sc_matmul import SCMatMul


_DET_LINEAR_OPS = ("mlp_fc1", "mlp_fc2", "qkv_proj", "out_proj")


def _normalize_det_linear_mp_spec(spec):
    """Mirror of cls's ``_normalize_linear_mp_spec`` for EVA-det.

    Accepts ``None`` or a dict ``{op_name | "proj": MPConfig |
    AdaptiveMPConfig | RangeMPConfig | {"fixed": ..., "adaptive": ...,
    "range": ..., "range_group_size": ...}}``.

    Returns ``{op_name: (mp_cfg, adaptive_cfg, range_cfg, gsize)}``.
    """
    out: dict = {}
    if not spec:
        return out
    for k, v in spec.items():
        if k == "proj":
            targets = ("qkv_proj", "out_proj")
        elif k in _DET_LINEAR_OPS:
            targets = (k,)
        else:
            raise ValueError(
                f"unknown linear_mp_spec key: {k!r}; "
                f"expected one of {_DET_LINEAR_OPS + ('proj',)}")
        if isinstance(v, MPConfig):
            entry = (v, None, None, 0)
        elif isinstance(v, AdaptiveMPConfig):
            entry = (None, v, None, 0)
        elif isinstance(v, RangeMPConfig):
            entry = (None, None, v, 0)
        elif isinstance(v, dict):
            entry = (
                v.get("fixed"),
                v.get("adaptive"),
                v.get("range"),
                int(v.get("range_group_size", 0)),
            )
        else:
            raise TypeError(
                f"linear_mp_spec[{k!r}] must be MPConfig, AdaptiveMPConfig, "
                f"RangeMPConfig, or dict; got {type(v).__name__}")
        for op in targets:
            out[op] = entry
    return out


def _det_mp_kwargs(linear_mp, op_name):
    if not linear_mp or op_name not in linear_mp:
        return {}
    mp_cfg, adaptive_cfg, range_cfg, gsize = linear_mp[op_name]
    return dict(mp_cfg=mp_cfg, adaptive_mp_cfg=adaptive_cfg,
                range_mp_cfg=range_cfg, range_mp_group_size=gsize,
                operator=op_name)


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
                 av_mode: str = "bipolar",
                 linear_mp_spec=None,
                 attn_mp_spec=None,
                 stoc_len: int | None = None,
                 mlp_chunk_d: int = 0):
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
    linear_mp_spec : dict | None
        Mirror of cls's ``linear_mp_spec``. Applies MPConfig /
        AdaptiveMPConfig / RangeMPConfig to the SCLinear wrappers for
        ``mlp_fc1, mlp_fc2, qkv_proj, out_proj`` (alias ``proj`` expands to
        both proj ops). For ViT without a native timestep, adaptive MP
        reads (t, T) from ``sc_integration.sc_linear.set_vit_timestep``.

    Returns
    -------
    stats : dict
        Count of swapped modules per op.
    """
    blocks = _discover_blocks(model)
    sched = normalize_sc_ops_per_block(sc_ops_per_block, len(blocks))
    stats = {k: 0 for k in SC_OP_NAMES}
    linear_mp = _normalize_det_linear_mp_spec(linear_mp_spec)
    # Pluck QK / AV attn mp configs (MPConfig or AdaptiveMPConfig each).
    attn_mp = attn_mp_spec or {}
    qk_mp = attn_mp.get("qk")
    av_mp = attn_mp.get("av")
    _attn_ok = (MPConfig, AdaptiveMPConfig)
    if qk_mp is not None and not isinstance(qk_mp, _attn_ok):
        raise TypeError(
            f"attn_mp_spec['qk'] must be MPConfig or AdaptiveMPConfig, got "
            f"{type(qk_mp).__name__}")
    if av_mp is not None and not isinstance(av_mp, _attn_ok):
        raise TypeError(
            f"attn_mp_spec['av'] must be MPConfig or AdaptiveMPConfig, got "
            f"{type(av_mp).__name__}")

    for i, blk in enumerate(blocks):
        attn = getattr(blk, "attn", None)
        mlp = getattr(blk, "mlp", None)

        if attn is not None:
            if sched["qkv_proj"][i] and isinstance(getattr(attn, "qkv", None), nn.Linear):
                attn.qkv = SCLinear(
                    attn.qkv, sc_prec=sc_prec, mode=proj_mode,
                    stoc_len=stoc_len,
                    **_det_mp_kwargs(linear_mp, "qkv_proj"))
                stats["qkv_proj"] += 1
            if sched["out_proj"][i] and isinstance(getattr(attn, "proj", None), nn.Linear):
                attn.proj = SCLinear(
                    attn.proj, sc_prec=sc_prec, mode=proj_mode,
                    stoc_len=stoc_len,
                    **_det_mp_kwargs(linear_mp, "out_proj"))
                stats["out_proj"] += 1
            # matmul1 / matmul2: swap only if current module is the FP MatMul
            # marker — don't double-wrap.
            if sched["qk"][i]:
                m1 = getattr(attn, "matmul1", None)
                if m1 is not None and not isinstance(m1, SCMatMul):
                    attn.matmul1 = SCMatMul(
                        sc_prec=sc_prec, mode=qk_mode, op="qk",
                        stoc_len=stoc_len,
                        mp_cfg=qk_mp if isinstance(qk_mp, MPConfig) else None,
                        adaptive_mp_cfg=qk_mp if isinstance(qk_mp, AdaptiveMPConfig) else None)
                    stats["qk"] += 1
            if sched["av"][i]:
                m2 = getattr(attn, "matmul2", None)
                if m2 is not None and not isinstance(m2, SCMatMul):
                    attn.matmul2 = SCMatMul(
                        sc_prec=sc_prec, mode=av_mode, op="av",
                        stoc_len=stoc_len,
                        mp_cfg=av_mp if isinstance(av_mp, MPConfig) else None,
                        adaptive_mp_cfg=av_mp if isinstance(av_mp, AdaptiveMPConfig) else None)
                    stats["av"] += 1

        if mlp is not None:
            if sched["mlp_fc1"][i] and isinstance(getattr(mlp, "fc1", None), nn.Linear):
                mlp.fc1 = SCLinear(
                    mlp.fc1, sc_prec=sc_prec, mode=mlp_mode,
                    stoc_len=stoc_len, chunk_d=mlp_chunk_d,
                    **_det_mp_kwargs(linear_mp, "mlp_fc1"))
                stats["mlp_fc1"] += 1
            if sched["mlp_fc2"][i] and isinstance(getattr(mlp, "fc2", None), nn.Linear):
                mlp.fc2 = SCLinear(
                    mlp.fc2, sc_prec=sc_prec, mode=mlp_mode,
                    stoc_len=stoc_len, chunk_d=mlp_chunk_d,
                    **_det_mp_kwargs(linear_mp, "mlp_fc2"))
                stats["mlp_fc2"] += 1

    return stats
