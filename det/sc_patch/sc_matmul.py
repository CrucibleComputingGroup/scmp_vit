"""SCMatMul — drop-in replacement for EVA's `MatMul(A, B) = A @ B`.

Applies stochastic-computing matmul via the grouped enable-signal kernel
(`vit_sc.sc.sc_triton.sc_matmul_grouped_enable_triton`).

EVA's `Attention.forward` does:

    attn = self.matmul1(q * scale, k.transpose(-2, -1))   # (BH, N, N)
    x    = self.matmul2(attn, v)                          # (BH, N, D)

Shapes for both are 3-D `(BH, M, K) @ (BH, K, N) -> (BH, M, N)`. The SC
kernel is 2-D `(M, K) @ (N, K) -> (M, N)`, so we loop over the leading batch
dim. This is slow compared to the native `@`, but kernel-level batching can
be plugged in later.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_VIT_SC = Path(__file__).resolve().parents[2]  # vit_sc/
if str(_VIT_SC / "sc") not in sys.path:
    sys.path.insert(0, str(_VIT_SC / "sc"))
if str(_VIT_SC) not in sys.path:
    sys.path.insert(0, str(_VIT_SC))

from sc_triton import sc_matmul_grouped_enable_triton  # noqa: E402
from config_helpers import make_sobol_simple_config    # noqa: E402


_CFG_CACHE: dict[tuple[int, int], dict] = {}


def _get_config(D: int, sc_prec: int) -> dict:
    key = (D, sc_prec)
    if key not in _CFG_CACHE:
        _CFG_CACHE[key] = make_sobol_simple_config(D, D, sc_prec)
    return _CFG_CACHE[key]


class SCMatMul(nn.Module):
    """Drop-in replacement for EVA `MatMul` that runs `A @ B` via SC.

    Parameters
    ----------
    sc_prec : int
        SC precision; stochastic stream length = 2 ** sc_prec.
    mode : "bipolar" | "unipolar"
        Quantization mode. `bipolar` for signed operands (Q, K), `unipolar`
        for non-negative (post-softmax attn for matmul2).
    """

    def __init__(self, sc_prec: int = 8, mode: str = "bipolar"):
        super().__init__()
        assert mode in ("bipolar", "unipolar"), mode
        self.sc_prec = sc_prec
        self.mode = mode

    def extra_repr(self) -> str:
        return f"sc_prec={self.sc_prec}, mode={self.mode}"

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A: (..., M, K);  B: (..., K, N) → out: (..., M, N)
        if A.dim() < 2 or B.dim() < 2:
            raise ValueError(f"SCMatMul expects ≥2-D, got A={A.shape} B={B.shape}")
        if A.shape[-1] != B.shape[-2]:
            raise ValueError(
                f"Inner-dim mismatch: A.shape[-1]={A.shape[-1]} B.shape[-2]={B.shape[-2]}"
            )
        M, K = A.shape[-2], A.shape[-1]
        N = B.shape[-1]

        lead = A.shape[:-2]
        # Broadcast leading dims if mismatched (rare in EVA — keep defensive)
        if B.shape[:-2] != lead:
            B = B.expand(*lead, K, N)

        A_flat = A.reshape(-1, M, K).float().contiguous()       # (B, M, K)
        B_flat = B.reshape(-1, K, N).float().contiguous()       # (B, K, N)

        cfg = _get_config(K, self.sc_prec)
        out_flat = torch.empty(A_flat.shape[0], M, N,
                               device=A.device, dtype=torch.float32)
        for i in range(A_flat.shape[0]):
            # sc_matmul_grouped_enable_triton computes a @ b.T where
            # a: (M, K), b: (N, K). So pass b = B_flat[i].T → (N, K).
            out_flat[i] = sc_matmul_grouped_enable_triton(
                A_flat[i], B_flat[i].transpose(0, 1).contiguous(),
                group_a=1, group_b=1,
                mode=self.mode, sc_prec=self.sc_prec, config=cfg,
            )
        return out_flat.reshape(*lead, M, N).to(A.dtype)
