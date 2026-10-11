"""A D1-specific learned source-cell routing baseline (567 parameters)."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ccb.models.common import ModelOutput


class LearnedGridExecutor(nn.Module):
    """Learn seven source-position maps from one-step grid supervision.

    This strong baseline assumes D1 operations rearrange nine distinct values.
    It never calls the oracle during inference. Hard rollout makes no claim
    about learning via gradients through the entire program.
    """

    def __init__(self) -> None:
        super().__init__()
        self.routing_logits = nn.Parameter(torch.randn(7, 9, 9) * .01)

    def source_logits(self, operation_ids: Tensor) -> Tensor:
        return self.routing_logits[operation_ids]

    def forward(self, batch) -> ModelOutput:
        if batch.codec.domain != 'alien_grid':
            raise ValueError('LearnedGridExecutor supports D1 only')
        mappings = self.routing_logits.argmax(-1)
        state = batch.initial_state
        trace = []
        for t in range(batch.operations.shape[1]):
            candidate = state.gather(1, mappings[batch.operations[:, t]])
            state = torch.where(batch.step_mask[:, t, None], candidate, state)
            trace.append(torch.nn.functional.one_hot(state, 9).float() * 20)
        return ModelOutput(torch.stack(trace, dim=1))
