from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch import Tensor

from ccb.encoding import DomainCodec


def _maximum_unique_assignment(scores: Tensor) -> Tensor:
    """Exact maximum-score row-to-distinct-column assignment for small states."""

    if scores.ndim != 2 or scores.shape[0] > scores.shape[1]:
        raise ValueError("scores must be [cells, values] with cells <= values")
    cells, values = scores.shape
    score_rows = scores.detach().cpu().tolist()

    @lru_cache(maxsize=None)
    def solve(row: int, used: int) -> tuple[float, tuple[int, ...]]:
        if row == cells:
            return 0.0, ()
        best_score = float("-inf")
        best_assignment: tuple[int, ...] = ()
        for value in range(values):
            if used & (1 << value):
                continue
            suffix_score, suffix = solve(row + 1, used | (1 << value))
            candidate = score_rows[row][value] + suffix_score
            if candidate > best_score:
                best_score = candidate
                best_assignment = (value, *suffix)
        return best_score, best_assignment

    return torch.tensor(solve(0, 0)[1], dtype=torch.long, device=scores.device)


def constrained_argmax(logits: Tensor, codec: DomainCodec) -> Tensor:
    """Decode logits while enforcing the domain's hard state constraints."""

    if logits.shape[-2:] != (codec.state_size, codec.state_vocab_size):
        raise ValueError("logit shape does not match codec")
    leading = logits.shape[:-2]
    flat = logits.reshape(-1, codec.state_size, codec.state_vocab_size)
    decoded = []
    if codec.domain == "alien_grid":
        decoded = [_maximum_unique_assignment(example) for example in flat]
    elif codec.domain == "symbolic_pointers":
        decoded = [example.argmax(dim=-1) for example in flat]
    elif codec.domain == "social_logic":
        agents = codec.number_of_agents
        if agents is None:
            raise ValueError("social codec lacks number_of_agents")
        for example in flat:
            matrix = example.reshape(agents, agents, 3)
            result = torch.empty(agents, agents, dtype=torch.long, device=logits.device)
            diagonal = torch.arange(agents, device=logits.device)
            result[diagonal, diagonal] = 2
            for left in range(agents):
                for right in range(left + 1, agents):
                    relation = (matrix[left, right] + matrix[right, left]).argmax()
                    result[left, right] = relation
                    result[right, left] = relation
            decoded.append(result.flatten())
    else:
        raise ValueError(f"unknown domain: {codec.domain}")
    return torch.stack(decoded).reshape(*leading, codec.state_size)


def decode_state(tokens: Tensor, codec: DomainCodec) -> Any:
    values = tokens.detach().cpu().tolist()
    if codec.domain == "alien_grid":
        restored = [value + 1 for value in values]
        return tuple(tuple(restored[index : index + 3]) for index in range(0, 9, 3))
    if codec.domain == "symbolic_pointers":
        return tuple(values)
    if codec.domain == "social_logic":
        agents = codec.number_of_agents
        if agents is None:
            raise ValueError("social codec lacks number_of_agents")
        restored = [value - 1 for value in values]
        return tuple(
            tuple(restored[index : index + agents])
            for index in range(0, agents * agents, agents)
        )
    raise ValueError(f"unknown domain: {codec.domain}")
