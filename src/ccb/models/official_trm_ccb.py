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
    bos_position: int
    state_tokens: slice
    ops_delimiter_position: int
    operation_tokens: slice
    output_delimiter_position: int
    query_tokens: slice
    sequence_length: int
    vocabulary_size: int
    operation_offset: int
    bos_token: int
    ops_token: int
    output_token: int
    mask_token: int
    pad_token: int


def build_ccb_trm_layout(codec: DomainCodec, max_depth: int) -> CCBTRMLayout:
    """Create the shared, collision-free CCB token layout."""

    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    operation_offset = codec.state_vocab_size
    special_offset = operation_offset + codec.operation_vocab_size
    state_tokens = slice(1, 1 + codec.state_size)
    ops_delimiter_position = state_tokens.stop
    operation_tokens = slice(ops_delimiter_position + 1, ops_delimiter_position + 1 + max_depth)
    output_delimiter_position = operation_tokens.stop
    query_tokens = slice(
        output_delimiter_position + 1,
        output_delimiter_position + 1 + max_depth * codec.state_size,
    )
    return CCBTRMLayout(
        bos_position=0,
        state_tokens=state_tokens,
        ops_delimiter_position=ops_delimiter_position,
        operation_tokens=operation_tokens,
        output_delimiter_position=output_delimiter_position,
        query_tokens=query_tokens,
        sequence_length=query_tokens.stop,
        vocabulary_size=special_offset + 5,
        operation_offset=operation_offset,
        bos_token=special_offset,
        ops_token=special_offset + 1,
        output_token=special_offset + 2,
        mask_token=special_offset + 3,
        pad_token=special_offset + 4,
    )


def encode_ccb_trm_tokens(
    batch: TransitionBatch, codec: DomainCodec, max_depth: int, layout: CCBTRMLayout
) -> Tensor:
    """Serialize CCB inputs without reading labels or target states."""

    if batch.codec != codec:
        raise ValueError("batch codec does not match adapter codec")
    if batch.operations.shape[1] > max_depth:
        raise ValueError("batch depth exceeds adapter max_depth")
    tokens = torch.full(
        (batch.initial_state.shape[0], layout.sequence_length),
        layout.pad_token,
        dtype=torch.long,
        device=batch.initial_state.device,
    )
    tokens[:, layout.bos_position] = layout.bos_token
    tokens[:, layout.state_tokens] = batch.initial_state
    tokens[:, layout.ops_delimiter_position] = layout.ops_token
    operation_slots = tokens[:, layout.operation_tokens]
    observed_depth = batch.operations.shape[1]
    encoded_operations = batch.operations + layout.operation_offset
    operation_slots[:, :observed_depth] = torch.where(
        batch.step_mask,
        encoded_operations,
        torch.full_like(encoded_operations, layout.pad_token),
    )
    tokens[:, layout.output_delimiter_position] = layout.output_token
    query_slots = tokens[:, layout.query_tokens].reshape(
        batch.initial_state.shape[0], max_depth, codec.state_size
    )
    valid_queries = torch.zeros(
        (batch.initial_state.shape[0], max_depth),
        dtype=torch.bool,
        device=batch.initial_state.device,
    )
    valid_queries[:, :observed_depth] = batch.step_mask
    query_slots.copy_(
        torch.where(
            valid_queries[:, :, None],
            torch.full_like(query_slots, layout.mask_token),
            torch.full_like(query_slots, layout.pad_token),
        )
    )
    return tokens


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

    Input sequence layout is
    ``BOS | initial-state | OPS | operations | OUTPUT | masked-output``.
    State values, operations, delimiters, MASK, and PAD occupy disjoint token
    namespaces. No label or intermediate target enters the input. CCB output
    uses state-value logits at the masked-output positions.
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
        self.layout = build_ccb_trm_layout(codec, max_depth)
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

        return encode_ccb_trm_tokens(batch, self.codec, self.max_depth, self.layout)

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
