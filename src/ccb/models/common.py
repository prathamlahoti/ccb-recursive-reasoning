from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelOutput:
    logits: Tensor
    loop_logits: tuple[Tensor, ...] = ()


class StateDecoder(nn.Module):
    def __init__(self, width: int, state_size: int, vocabulary_size: int) -> None:
        super().__init__()
        self.cell = nn.Embedding(state_size, width)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, vocabulary_size),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        # hidden may be [B,T,D] or [B,T,S,D].
        if hidden.ndim == 3:
            positions = torch.arange(self.cell.num_embeddings, device=hidden.device)
            hidden = hidden.unsqueeze(-2) + self.cell(positions)
        return self.output(self.norm(hidden))


class ResidualMLP(nn.Module):
    def __init__(self, width: int, expansion: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(
            nn.Linear(width, expansion * width),
            nn.GELU(),
            nn.Linear(expansion * width, width),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.layers(self.norm(value))


class CellMixer(nn.Module):
    """Shared cross-cell communication needed for permutations and graph state."""

    def __init__(self, width: int) -> None:
        super().__init__()
        heads = 4 if width % 4 == 0 else 1
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)

    def forward(self, value: Tensor) -> Tensor:
        original_shape = value.shape
        flattened = value.reshape(-1, original_shape[-2], original_shape[-1])
        normalized = self.norm(flattened)
        mixed, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        return (flattened + mixed).reshape(original_shape)
