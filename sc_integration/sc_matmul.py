"""Intentionally-empty module.

The packed-XNOR QK kernel wrappers (``sc_matmul_qk``, ``_sc_matmul_qk_batched``,
``sc_matmul_qk_multihead``, ``quantize_for_sc``, ``quantize_for_sc_per_head``,
``dequantize_sc_result``) were removed. They had strictly worse accuracy and
throughput than ``sc_triton.sc_matmul_enable_batched_bipolar`` (the
enable-signal batched kernel) and produced misleadingly pessimistic QK
baselines in every prior experiment.

Use ``sc_matmul_enable_batched_bipolar`` for QK and
``sc_matmul_grouped_enable_triton`` for AV / linear ops.
"""
