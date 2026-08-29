"""Minimal CCB adapter for the released Tiny Recursive Models core.

This intentionally uses the released TRM recurrence: fixed H/L initial
buffers, one shared reasoning module, ``H_cycles - 1`` detached updates and
one gradient-bearing update.  It is kept separate from the earlier
legacy prototype code, which is no longer part of the active model package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ccb.encoding import DomainCodec, TransitionBatch
from ccb.models.common import ModelOutput


class SwiGLU(nn.Module):
    def __init__(self, width: int, expansion: float) -> None:
        super().__init__()
        hidden = max(1, int(width * expansion))
        self.gate = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.output = nn.Linear(hidden, width, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.output(torch.nn.functional.silu(self.gate(value)) * self.value(value))


class TRMReasoningBlock(nn.Module):
    """Post-normalized, bidirectional attention + SwiGLU block from TRM."""

    def __init__(self, width: int, heads: int, expansion: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True, bias=False)
        self.attention_norm = nn.RMSNorm(width)
        self.mlp = SwiGLU(width, expansion)
        self.mlp_norm = nn.RMSNorm(width)

    def forward(self, hidden: Tensor, token_mask: Tensor) -> Tensor:
        attended, _ = self.attention(
            hidden, hidden, hidden, key_padding_mask=~token_mask, need_weights=False
        )
        hidden = self.attention_norm(hidden + attended)
        return self.mlp_norm(hidden + self.mlp(hidden))


@dataclass(frozen=True)
class PublishedTRMCarry:
    z_h: Tensor
    z_l: Tensor


class PublishedTRMCCB(nn.Module):
    """Upstream-derived TRM core with an explicit CCB trace-token adapter.

    Each output trace cell receives a token encoding ``(initial_cell,
    operation_at_step)``.  Labels are *only* the target trace-cell values;
    they are never fed to the model.  Padded trace cells are masked from
    attention and loss by the existing CCB batch mask.
    """

    def __init__(
        self,
        codec: DomainCodec,
        *,
        width: int = 64,
        heads: int = 4,
        h_cycles: int = 3,
        l_cycles: int = 6,
        layers: int = 2,
        expansion: float = 4.0,
        max_depth: int = 100,
        halt_max_steps: int = 4,
        halt_exploration_prob: float = 0.1,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if min(h_cycles, l_cycles, layers, max_depth, halt_max_steps) < 1:
            raise ValueError("TRM cycle, layer, and max-depth counts must be positive")
        if not 0.0 <= halt_exploration_prob <= 1.0:
            raise ValueError("halt exploration probability must lie in [0, 1]")
        self.codec = codec
        self.width = width
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        self.max_depth = max_depth
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.sequence_length = max_depth * codec.state_size
        token_vocab = codec.state_vocab_size * codec.operation_vocab_size
        self.embed_scale = math.sqrt(width)
        self.embed_tokens = nn.Embedding(token_vocab, width)
        self.embed_positions = nn.Embedding(self.sequence_length, width)
        self.l_level = nn.ModuleList(
            [TRMReasoningBlock(width, heads, expansion) for _ in range(layers)]
        )
        self.lm_head = nn.Linear(width, codec.state_vocab_size, bias=False)
        self.q_head = nn.Linear(width, 2)
        self.register_buffer("h_init", torch.randn(width))
        self.register_buffer("l_init", torch.randn(width))
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

    def _tokens_and_mask(self, batch: TransitionBatch) -> tuple[Tensor, Tensor]:
        if batch.codec != self.codec:
            raise ValueError("batch codec does not match model codec")
        batch_size, depth = batch.operations.shape
        if depth > self.max_depth:
            raise ValueError(f"depth {depth} exceeds configured max_depth {self.max_depth}")
        state = batch.initial_state[:, None, :].expand(-1, depth, -1)
        operation = batch.operations[:, :, None].expand_as(state)
        tokens = state * self.codec.operation_vocab_size + operation
        token_mask = batch.step_mask[:, :, None].expand_as(tokens)
        return tokens.reshape(batch_size, -1), token_mask.reshape(batch_size, -1)

    def _input_embeddings(self, tokens: Tensor) -> Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.embed_scale * (self.embed_tokens(tokens) + self.embed_positions(positions))

    def initial_carry(self, batch: TransitionBatch) -> PublishedTRMCarry:
        length = batch.operations.shape[1] * self.codec.state_size
        shape = (batch.initial_state.shape[0], length, self.width)
        return PublishedTRMCarry(self.h_init.expand(shape), self.l_init.expand(shape))

    def _reason(self, hidden: Tensor, injection: Tensor, token_mask: Tensor) -> Tensor:
        hidden = hidden + injection
        for block in self.l_level:
            hidden = block(hidden, token_mask)
        return hidden

    def refine(
        self, batch: TransitionBatch, carry: PublishedTRMCarry
    ) -> tuple[PublishedTRMCarry, ModelOutput, tuple[Tensor, Tensor]]:
        tokens, token_mask = self._tokens_and_mask(batch)
        input_embeddings = self._input_embeddings(tokens)
        z_h, z_l = carry.z_h, carry.z_l
        with torch.no_grad():
            for _ in range(self.h_cycles - 1):
                for _ in range(self.l_cycles):
                    z_l = self._reason(z_l, z_h + input_embeddings, token_mask)
                z_h = self._reason(z_h, z_l, token_mask)
        for _ in range(self.l_cycles):
            z_l = self._reason(z_l, z_h + input_embeddings, token_mask)
        z_h = self._reason(z_h, z_l, token_mask)
        batch_size, depth = batch.operations.shape
        logits = self.lm_head(z_h).reshape(
            batch_size, depth, self.codec.state_size, self.codec.state_vocab_size
        )
        q_logits = self.q_head(z_h[:, 0]).float()
        return PublishedTRMCarry(z_h.detach(), z_l.detach()), ModelOutput(logits), (
            q_logits[:, 0], q_logits[:, 1]
        )

    def forward(self, batch: TransitionBatch) -> ModelOutput:
        _, output, _ = self.refine(batch, self.initial_carry(batch))
        return output
