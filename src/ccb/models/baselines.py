"""Non-recursive control model for the CCB-TRM extension."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ccb.encoding import DomainCodec, TransitionBatch
from ccb.models.common import ModelOutput, StateDecoder


def _causal_mask(length: int, device: torch.device) -> Tensor:
    return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)


def _sinusoidal_positions(length: int, width: int, device: torch.device) -> Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10_000.0, device=device)) / width)
    )
    encoding = torch.zeros(length, width, device=device)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding


class DirectTransformer(nn.Module):
    """Direct sequence-to-trace control with no recursive shared weights."""

    def __init__(
        self, codec: DomainCodec, *, width: int = 128, heads: int = 4, layers: int = 4
    ) -> None:
        super().__init__()
        self.codec = codec
        self.width = width
        self.state_embedding = nn.Embedding(codec.state_vocab_size, width)
        self.state_position = nn.Embedding(codec.state_size, width)
        self.operation_embedding = nn.Embedding(codec.operation_vocab_size, width)
        self.initial_projection = nn.Linear(codec.state_size * width, width)
        block = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=4 * width, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.decoder = StateDecoder(width, codec.state_size, codec.state_vocab_size)

    def forward(self, batch: TransitionBatch) -> ModelOutput:
        if batch.codec != self.codec:
            raise ValueError("batch codec does not match model codec")
        positions = torch.arange(self.codec.state_size, device=batch.initial_state.device)
        initial = self.state_embedding(batch.initial_state) + self.state_position(positions)
        summary = self.initial_projection(initial.flatten(1))
        steps = _sinusoidal_positions(batch.operations.shape[1], self.width, batch.operations.device)
        inputs = self.operation_embedding(batch.operations) + summary[:, None, :] + steps[None]
        hidden = self.encoder(inputs, mask=_causal_mask(inputs.shape[1], inputs.device))
        return ModelOutput(self.decoder(hidden))
