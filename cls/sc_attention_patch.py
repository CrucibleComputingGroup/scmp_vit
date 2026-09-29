"""Compat shim: the ViT SC patcher now lives in ``vit_sc/model/sc_vit.py``.

The classification entry points historically did
``from sc_attention_patch import patch_model, parse_sc_ops, SCLinear``. That
still works, but the implementation moved to ``model.sc_vit`` and is backed by
``scmp_kernels`` (vendored at ``vit_sc/kernels``) instead of the retired in-tree
``sc/`` + ``sc_integration/`` stack, which now sits under ``deprecated/``.

Prefer importing from ``model.sc_vit`` directly in new code.
"""
import sys
from pathlib import Path

_VIT_SC = Path(__file__).resolve().parents[1]
if str(_VIT_SC) not in sys.path:
    sys.path.insert(0, str(_VIT_SC))

from model.sc_vit import (  # noqa: E402,F401
    SCLinear,
    patch_model,
    parse_sc_ops,
    normalize_sc_ops_per_block,
    make_sc_attention_forward,
    set_av_metric_kind,
    set_sc_defaults,
    sc_defaults,
    set_vit_timestep,
    get_vit_timestep,
    load_adaptive_mp,
    SC_OP_NAMES,
    LINEAR_OP_NAMES,
    MPConfig,
    AdaptiveMPConfig,
    RangeMPConfig,
)

__all__ = [
    "SCLinear", "patch_model", "parse_sc_ops", "normalize_sc_ops_per_block",
    "make_sc_attention_forward", "set_av_metric_kind", "set_sc_defaults",
    "sc_defaults", "set_vit_timestep", "get_vit_timestep", "load_adaptive_mp",
    "SC_OP_NAMES", "LINEAR_OP_NAMES",
    "MPConfig", "AdaptiveMPConfig", "RangeMPConfig",
]


def patch_model_attention(model, sc_prec: int = 8):
    """Back-compat alias: patch only Q@K^T."""
    return patch_model(model, sc_prec=sc_prec, sc_qk=True)["attn"]
