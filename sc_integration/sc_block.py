"""
SC-enabled DiT Block for Q-DiT.

This module provides SCDiTBlock, which extends the quantized DiT block
with stochastic computing support.
"""

import torch
import torch.nn as nn
from typing import Optional
from copy import deepcopy

from ..quant import Quantizer
from ..qLinearLayer import QLinearLayer
from .sc_controller import SCController
from .sc_attention import SCAttention
from .sc_mlp import SCMlp
from models.models import DiTBlock, modulate


class SCDiTBlock(nn.Module):
    """
    DiT Block with stochastic computing support.

    This block uses SCAttention for attention operations, enabling SC
    for q@k^T based on the SCController settings.

    Args:
        dit_block: Original DiTBlock from the model
        args: Quantization arguments
        block_idx: Index of this block in the model
        sc_controller: SCController instance
    """

    def __init__(
        self,
        dit_block: DiTBlock,
        args,
        block_idx: int,
        sc_controller: SCController,
    ):
        super().__init__()

        self.args = args
        self.block_idx = block_idx
        self.sc_controller = sc_controller
        self.quantize_bmm_input = args.quantize_bmm_input

        # SC-enabled attention
        self.attn = SCAttention(
            dit_block.attn,
            deepcopy(args),
            block_idx=block_idx,
            sc_controller=sc_controller,
        )

        # Normalization layers
        self.norm1 = dit_block.norm1
        self.norm2 = dit_block.norm2

        # SC-enabled MLP
        self.mlp = SCMlp(dit_block.mlp, deepcopy(args),
                         block_idx=block_idx, sc_controller=sc_controller)

        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            dit_block.adaLN_modulation[0],
            QLinearLayer(dit_block.adaLN_modulation[1], deepcopy(args))
        )

        # Optional quantizers for BMM input
        if self.quantize_bmm_input:
            self.ln1_quant = Quantizer(args=deepcopy(args))
            self.attn_quant = Quantizer(args=deepcopy(args))
            self.ln2_quant = Quantizer(args=deepcopy(args))
            self.mlp_quant = Quantizer(args=deepcopy(args))
            self.adaln_quant = Quantizer(args=deepcopy(args))

    def to(self, *args, **kwargs):
        super(SCDiTBlock, self).to(*args, **kwargs)
        self.attn = self.attn.to(*args, **kwargs)
        self.mlp = self.mlp.to(*args, **kwargs)
        self.norm1 = self.norm1.to(*args, **kwargs)
        self.norm2 = self.norm2.to(*args, **kwargs)
        self.adaLN_modulation = self.adaLN_modulation.to(*args, **kwargs)
        if self.quantize_bmm_input:
            self.ln1_quant = self.ln1_quant.to(*args, **kwargs)
            self.attn_quant = self.attn_quant.to(*args, **kwargs)
            self.ln2_quant = self.ln2_quant.to(*args, **kwargs)
            self.mlp_quant = self.mlp_quant.to(*args, **kwargs)
            self.adaln_quant = self.adaln_quant.to(*args, **kwargs)
        return self

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SC-enabled DiT block.

        Args:
            x: Input tensor, shape (B, N, C)
            c: Conditioning tensor (timestep + class embedding)

        Returns:
            Output tensor, shape (B, N, C)
        """
        if not self.quantize_bmm_input:
            # Standard path without BMM input quantization
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(c).chunk(6, dim=1)

            # Attention with optional SC for q@k^T
            x = x + gate_msa.unsqueeze(1) * self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa)
            )

            # MLP (standard quantized)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm2(x), shift_mlp, scale_mlp)
            )
        else:
            # Path with BMM input quantization
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaln_quant(self.adaLN_modulation(c)).chunk(6, dim=1)

            # Attention with optional SC for q@k^T
            x = x + gate_msa.unsqueeze(1) * self.attn_quant(
                self.attn(modulate(self.ln1_quant(self.norm1(x)), shift_msa, scale_msa))
            )

            # MLP (standard quantized)
            x = x + gate_mlp.unsqueeze(1) * self.mlp_quant(
                self.mlp(modulate(self.ln2_quant(self.norm2(x)), shift_mlp, scale_mlp))
            )

        return x

    def extra_repr(self) -> str:
        return (
            f"block_idx={self.block_idx}, "
            f"sc_prec={self.sc_controller.sc_prec}, "
            f"qklayerwise={self.sc_controller.qklayerwise}"
        )
