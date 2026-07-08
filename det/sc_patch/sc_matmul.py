"""SCMatMul — drop-in replacement for EVA's `MatMul(A, B) = A @ B`.

Applies stochastic-computing matmul via the grouped enable-signal kernel
(`vit_sc.sc.sc_triton.sc_matmul_grouped_enable_triton`).

EVA's `Attention.forward` does:

    attn = self.matmul1(q * scale, k.transpose(-2, -1))   # (BH, N, N)
    x    = self.matmul2(attn, v)                          # (BH, N, D)

Shapes for both are 3-D `(BH, M, K) @ (BH, K, N) -> (BH, M, N)`. The SC
kernel is 2-D `(M, K) @ (N, K) -> (M, N)`, so we loop over the leading batch
dim.

Mixed-precision:
  * ``mp_cfg``          → MPConfig. ``op="qk"`` classifies per-head by
    metric=``|A|.amax((1,2))`` (one stoc_len per attention head).
    ``op="av"`` classifies per-attn-row by ``A.amax(-1)`` inside each head.
  * ``adaptive_mp_cfg`` → AdaptiveMPConfig, same metric but timestep-aware
    (reads (t, T) from ``sc_linear.get_vit_timestep``).
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

from scmp_kernels.sc import sc_matmul  # noqa: E402  unified dispatcher (enable path)
from scmp_kernels.sc.config_helpers import make_sobol_simple_config    # noqa: E402
from scmp_kernels.mp import (  # noqa: E402
    MPConfig, AdaptiveMPConfig,
    classify_rows_by_metric, adaptive_classify_rows,
)
from sc_integration import sc_linear as _scl  # noqa: E402
from sc_integration.mp_linear import sc_prec_for_stoc_len  # noqa: E402


_CFG_CACHE: dict[tuple[int, int], dict] = {}


def _get_config(D: int, sc_prec: int) -> dict:
    key = (D, sc_prec)
    if key not in _CFG_CACHE:
        _CFG_CACHE[key] = make_sobol_simple_config(D, D, sc_prec)
    return _CFG_CACHE[key]


def _classify(metric: torch.Tensor,
              cfg: "MPConfig | AdaptiveMPConfig",
              operator: str) -> "RowAssignment":
    if isinstance(cfg, AdaptiveMPConfig):
        t, T = _scl.get_vit_timestep()
        return adaptive_classify_rows(metric, t, T, cfg, operator=operator)
    return classify_rows_by_metric(
        metric, cfg.stoc_len_levels, cfg.level_fractions,
    )


class SCMatMul(nn.Module):
    """Drop-in replacement for EVA `MatMul` that runs `A @ B` via SC.

    Parameters
    ----------
    sc_prec : int
        SC precision; stochastic stream length = 2 ** sc_prec.
    mode : "bipolar" | "unipolar"
        Quantization mode. `bipolar` for signed operands (Q, K), `unipolar`
        for non-negative (post-softmax attn for matmul2).
    mp_cfg : MPConfig | None
        Static per-head (op="qk") / per-attn-row (op="av") MP.
    adaptive_mp_cfg : AdaptiveMPConfig | None
        Adaptive (timestep-aware) MP with the same metric. Wins over
        ``mp_cfg`` when both set.
    op : "qk" | "av" | None
        Which operator this SCMatMul is playing; drives the classifier
        metric and per-op (α, β) lookup. ``None`` ⇒ uniform only (legacy).
    """

    def __init__(self, sc_prec: int = 8, mode: str = "bipolar",
                 mp_cfg: "MPConfig | None" = None,
                 adaptive_mp_cfg: "AdaptiveMPConfig | None" = None,
                 op: "str | None" = None,
                 stoc_len: "int | None" = None):
        super().__init__()
        assert mode in ("bipolar", "unipolar"), mode
        self.sc_prec = sc_prec
        self.stoc_len = stoc_len
        self.mode = mode
        self.mp_cfg = mp_cfg
        self.adaptive_mp_cfg = adaptive_mp_cfg
        self.op = op

    def extra_repr(self) -> str:
        mp = "adaptive" if self.adaptive_mp_cfg else ("static" if self.mp_cfg else "none")
        sl = self.stoc_len if self.stoc_len is not None else (1 << self.sc_prec)
        return (f"sc_prec={self.sc_prec}, stoc_len={sl}, "
                f"mode={self.mode}, op={self.op}, mp={mp}")

    # ---- active config selection ----
    def _mp(self):
        return self.adaptive_mp_cfg or self.mp_cfg

    # ---- QK bucketing: per-head (stoc_len per BH) ----
    def _forward_qk_mp(self, A_flat, B_flat, M, K, N):
        cfg = self._mp()
        BH = A_flat.shape[0]
        # Metric per-head: |A|.amax over (M,K)  (same as cls _classify_per_head)
        metric = A_flat.abs().amax(dim=(1, 2))  # (BH,)
        assignment = _classify(metric, cfg, operator="qk")

        out_flat = torch.empty(BH, M, N, device=A_flat.device, dtype=torch.float32)
        # B is fed to the kernel in row-major (N, K) form; doing the transpose
        # once per bucket (over the head subset) is much cheaper than per-head.
        for sl, heads in assignment.level_row_indices.items():
            if int(sl) == 0 or heads.numel() == 0:
                continue
            sp = sc_prec_for_stoc_len(int(sl))
            cfg_k = _get_config(K, sp)
            A_sub = A_flat.index_select(0, heads).contiguous()         # (n_h, M, K)
            B_sub = B_flat.index_select(0, heads).transpose(1, 2).contiguous()  # (n_h, N, K)
            sub = sc_matmul(
                A_sub, B_sub, granularity="per_row",
                group_a=1, group_b=1,
                mode=self.mode, sc_prec=sp, config=cfg_k,
                stoc_len=int(sl),
            )
            out_flat.index_copy_(0, heads, sub)
        return out_flat

    # ---- AV bucketing: per-attn-row inside each head ----
    def _forward_av_mp(self, A_flat, B_flat, M, K, N):
        cfg = self._mp()
        BH = A_flat.shape[0]
        out_flat = torch.empty(BH, M, N, device=A_flat.device, dtype=torch.float32)
        for i in range(BH):
            row_max = A_flat[i].amax(dim=-1)  # (M,)
            assignment = _classify(row_max, cfg, operator="av")
            for sl, rows in assignment.level_row_indices.items():
                if int(sl) == 0 or rows.numel() == 0:
                    continue
                sp = sc_prec_for_stoc_len(int(sl))
                cfg_k = _get_config(K, sp)
                sub = sc_matmul(
                    A_flat[i].index_select(0, rows).contiguous(),
                    B_flat[i].transpose(0, 1).contiguous(),
                    granularity="per_row",
                    group_a=1, group_b=1,
                    mode=self.mode, sc_prec=sp, config=cfg_k,
                    stoc_len=int(sl),
                )
                out_flat[i, rows] = sub
        return out_flat

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
        if B.shape[:-2] != lead:
            B = B.expand(*lead, K, N)

        A_flat = A.reshape(-1, M, K).float().contiguous()       # (BH, M, K)
        B_flat = B.reshape(-1, K, N).float().contiguous()       # (BH, K, N)

        if self._mp() is not None and self.op == "qk":
            out_flat = self._forward_qk_mp(A_flat, B_flat, M, K, N)
        elif self._mp() is not None and self.op == "av":
            out_flat = self._forward_av_mp(A_flat, B_flat, M, K, N)
        else:
            # One batched launch over BH — replaces the per-head Python loop.
            # Transpose B once (BH, K, N) → (BH, N, K) instead of per-head.
            cfg = _get_config(K, self.sc_prec)
            B_rowmajor = B_flat.transpose(1, 2).contiguous()       # (BH, N, K)
            out_flat = sc_matmul(
                A_flat, B_rowmajor, granularity="per_row",
                group_a=1, group_b=1,
                mode=self.mode, sc_prec=self.sc_prec, config=cfg,
                stoc_len=self.stoc_len,
            )
        return out_flat.reshape(*lead, M, N).to(A.dtype)
