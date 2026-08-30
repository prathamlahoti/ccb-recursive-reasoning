"""CCB boundary adapter around the mechanically verified official TRM core."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ccb.encoding import DomainCodec, TransitionBatch
from ccb.models.common import ModelOutput
from ccb.models.official_trm_core import (
    OfficialTRMACTCarry,
    OfficialTRMACTWrapper,
    OfficialTRMConfig,
)


@dataclass(frozen=True)
class CCBTRMLayout:
    state_tokens: slice
    operation_tokens: slice
    query_tokens: slice
    sequence_length: int
    vocabulary_size: int


@dataclass(frozen=True)
class CCBOfficialACTCarry:
    """CCB labels paired with the official token-only ACT carry.

    The official model deliberately knows only token inputs.  CCB's supervised
    loss additionally needs the target trace for each row currently active in
    ACT, so this boundary object mirrors exactly the row replacement performed
    by the official wrapper without exposing labels to the model.
    """

    core: OfficialTRMACTCarry
    current_batch: TransitionBatch

    @property
    def steps(self) -> Tensor:
        return self.core.steps

    @property
    def halted(self) -> Tensor:
        return self.core.halted


class OfficialTRMCCBAdapter(nn.Module):
    """Use the official TRM core with a narrow, target-free CCB token adapter.

    Input sequence layout is ``initial-state | operations | query-slots``.
    Query slots contain a constant token and are selected only by position; no
    label or intermediate target enters the input. The official vocabulary is
    widened only enough to represent both state values and operation IDs. CCB
    output uses the first ``state_vocab_size`` logits at the query positions.
    """

    def __init__(
        self,
        codec: DomainCodec,
        *,
        max_depth: int,
        hidden_size: int = 512,
        num_heads: int = 8,
        l_layers: int = 2,
        h_cycles: int = 3,
        l_cycles: int = 6,
        expansion: float = 4.0,
        halt_max_steps: int = 16,
        halt_exploration_prob: float = 0.1,
        forward_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.codec = codec
        self.max_depth = max_depth
        self.layout = CCBTRMLayout(
            state_tokens=slice(0, codec.state_size),
            operation_tokens=slice(codec.state_size, codec.state_size + max_depth),
            query_tokens=slice(codec.state_size + max_depth, codec.state_size + max_depth + max_depth * codec.state_size),
            sequence_length=codec.state_size + max_depth + max_depth * codec.state_size,
            vocabulary_size=max(codec.state_vocab_size, codec.operation_vocab_size),
        )
        self.core = OfficialTRMACTWrapper(
            OfficialTRMConfig(
                batch_size=1,
                seq_len=self.layout.sequence_length,
                vocab_size=self.layout.vocabulary_size,
                h_cycles=h_cycles,
                l_cycles=l_cycles,
                l_layers=l_layers,
                hidden_size=hidden_size,
                expansion=expansion,
                num_heads=num_heads,
                halt_max_steps=halt_max_steps,
                halt_exploration_prob=halt_exploration_prob,
                pos_encodings="rope",
                forward_dtype=forward_dtype,
                no_act_continue=True,
            )
        )

    def input_tokens(self, batch: TransitionBatch) -> Tensor:
        """Serialize CCB inputs without reading `targets`."""

        if batch.codec != self.codec:
            raise ValueError("batch codec does not match adapter codec")
        if batch.operations.shape[1] > self.max_depth:
            raise ValueError("batch depth exceeds adapter max_depth")
        tokens = torch.zeros(
            (batch.initial_state.shape[0], self.layout.sequence_length),
            dtype=torch.long,
            device=batch.initial_state.device,
        )
        tokens[:, self.layout.state_tokens] = batch.initial_state
        tokens[:, self.layout.operation_tokens.start : self.layout.operation_tokens.start + batch.operations.shape[1]] = batch.operations
        return tokens

    def _output(self, logits: Tensor, depth: int) -> ModelOutput:
        query = logits[:, self.layout.query_tokens, : self.codec.state_vocab_size]
        query = query.reshape(logits.shape[0], self.max_depth, self.codec.state_size, self.codec.state_vocab_size)
        return ModelOutput(query[:, :depth])

    def initial_carry(self, batch: TransitionBatch) -> CCBOfficialACTCarry:
        return CCBOfficialACTCarry(self.core.initial_carry(self.input_tokens(batch)), batch)

    # Compatibility with the generic ACT launcher.  This is intentionally an
    # alias, not a second state implementation.
    def initial_act_carry(self, batch: TransitionBatch) -> CCBOfficialACTCarry:
        return self.initial_carry(batch)

    @staticmethod
    def _replace_halted_rows(
        previous: TransitionBatch, incoming: TransitionBatch, reset: Tensor
    ) -> TransitionBatch:
        """Mirror official ACT's row-level ``torch.where`` current-data update."""

        if previous.codec != incoming.codec or previous.domain != incoming.domain:
            raise ValueError("ACT cannot replace rows across incompatible CCB batches")
        if previous.operations.shape != incoming.operations.shape:
            raise ValueError("ACT requires a fixed CCB batch tensor shape")
        row = reset[:, None]
        row_state = reset[:, None, None]
        return TransitionBatch(
            previous.domain,
            torch.where(row, incoming.initial_state, previous.initial_state),
            torch.where(row, incoming.operations, previous.operations),
            torch.where(row_state, incoming.targets, previous.targets),
            previous.codec,
            torch.where(row, incoming.step_mask, previous.step_mask),
            torch.where(reset, incoming.depths, previous.depths),
        )

    def act_step(
        self, carry: CCBOfficialACTCarry, batch: TransitionBatch
    ) -> tuple[CCBOfficialACTCarry, ModelOutput, tuple[Tensor, Tensor]]:
        current_batch = self._replace_halted_rows(carry.current_batch, batch, carry.core.halted)
        core_carry, outputs = self.core(carry.core, self.input_tokens(batch))
        return (
            CCBOfficialACTCarry(core_carry, current_batch),
            self._output(outputs["logits"], batch.operations.shape[1]),
            (outputs["q_halt_logits"], outputs["q_continue_logits"]),
        )

    def forward(self, batch: TransitionBatch) -> ModelOutput:
        carry = self.initial_carry(batch)
        output: ModelOutput | None = None
        for _ in range(self.core.config.halt_max_steps):
            carry, output, _ = self.act_step(carry, batch)
        if output is None:
            raise RuntimeError("halt_max_steps must be positive")
        return output
