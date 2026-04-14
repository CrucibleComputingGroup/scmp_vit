"""Minimal sc_integration for vit_sc.

Only the SC attention-matmul helpers are re-exported. The rest of Q-DiT's
integration (SCAttention/SCMlp/SCBlock/SCController/...) depends on
qdit.quant and friends and is intentionally not imported here.
"""
from .sc_matmul import (
    sc_matmul_qk,
    sc_matmul_qk_multihead,
    quantize_for_sc,
    quantize_for_sc_per_head,
    dequantize_sc_result,
)

__all__ = [
    "sc_matmul_qk",
    "sc_matmul_qk_multihead",
    "quantize_for_sc",
    "quantize_for_sc_per_head",
    "dequantize_sc_result",
]
