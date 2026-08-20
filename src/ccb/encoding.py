from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from ccb.domains.alien_grid import GridOperation
from ccb.domains.social_logic import SocialOperation
from ccb.domains.symbolic_pointers import PointerOperation, VARIABLES
from ccb.records import Episode


@dataclass(frozen=True)
class DomainCodec:
    domain: str
    state_size: int
    state_vocab_size: int
    operation_vocab_size: int
    number_of_agents: int | None = None


@dataclass(frozen=True)
class TransitionBatch:
    domain: str
    initial_state: Tensor
    operations: Tensor
    targets: Tensor
    codec: DomainCodec
    step_mask: Tensor
    depths: Tensor

    def to(self, device: torch.device | str) -> "TransitionBatch":
        return TransitionBatch(
            self.domain,
            self.initial_state.to(device),
            self.operations.to(device),
            self.targets.to(device),
            self.codec,
            self.step_mask.to(device),
            self.depths.to(device),
        )


def codec_for(episode: Episode[Any, Any]) -> DomainCodec:
    if episode.domain == "alien_grid":
        return DomainCodec(episode.domain, 9, 9, len(GridOperation))
    if episode.domain == "symbolic_pointers":
        return DomainCodec(episode.domain, len(VARIABLES), 10, len(PointerOperation))
    if episode.domain == "social_logic":
        agents = int(episode.metadata["number_of_agents"])
        return DomainCodec(episode.domain, agents * agents, 3, 2 * agents * agents, agents)
    raise ValueError(f"unknown domain: {episode.domain}")


def _flatten_state(domain: str, state: Any) -> tuple[int, ...]:
    if domain == "alien_grid":
        return tuple(value - 1 for row in state for value in row)
    if domain == "symbolic_pointers":
        return tuple(state)
    if domain == "social_logic":
        return tuple(value + 1 for row in state for value in row)
    raise ValueError(f"unknown domain: {domain}")


def _operation_id(operation: Any, codec: DomainCodec) -> int:
    if codec.domain == "alien_grid":
        return list(GridOperation).index(operation)
    if codec.domain == "symbolic_pointers":
        return list(PointerOperation).index(operation)
    if codec.domain == "social_logic":
        social: SocialOperation = operation
        agents = codec.number_of_agents
        if agents is None:
            raise ValueError("social codec is missing number_of_agents")
        relation = 1 if int(social.relation) == 1 else 0
        return (social.source * agents + social.target) * 2 + relation
    raise ValueError(f"unknown domain: {codec.domain}")


def collate_episodes(episodes: Sequence[Episode[Any, Any]]) -> TransitionBatch:
    if not episodes:
        raise ValueError("cannot collate an empty batch")
    codec = codec_for(episodes[0])
    if any(codec_for(episode) != codec for episode in episodes):
        raise ValueError("all episodes in a batch must share a codec")
    max_depth = max(episode.depth for episode in episodes)
    initial = torch.tensor(
        [_flatten_state(episode.domain, episode.initial_state) for episode in episodes],
        dtype=torch.long,
    )
    operations = torch.zeros((len(episodes), max_depth), dtype=torch.long)
    targets = torch.zeros(
        (len(episodes), max_depth, codec.state_size), dtype=torch.long
    )
    step_mask = torch.zeros((len(episodes), max_depth), dtype=torch.bool)
    depths = torch.tensor([episode.depth for episode in episodes], dtype=torch.long)
    for row, episode in enumerate(episodes):
        depth = episode.depth
        operations[row, :depth] = torch.tensor(
            [_operation_id(operation, codec) for operation in episode.operations],
            dtype=torch.long,
        )
        targets[row, :depth] = torch.tensor(
            [
                _flatten_state(episode.domain, transition.after)
                for transition in episode.transitions
            ],
            dtype=torch.long,
        )
        step_mask[row, :depth] = True
    return TransitionBatch(
        episodes[0].domain,
        initial,
        operations,
        targets,
        codec,
        step_mask,
        depths,
    )
