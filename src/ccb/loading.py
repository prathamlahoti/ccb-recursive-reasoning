from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ccb.encoding import TransitionBatch, codec_for, collate_episodes
from ccb.records import Episode


class EpisodeDataset(Dataset[Episode[Any, Any]]):
    def __init__(self, episodes: Sequence[Episode[Any, Any]]) -> None:
        if not episodes:
            raise ValueError("an episode dataset cannot be empty")
        self.episodes = tuple(episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> Episode[Any, Any]:
        return self.episodes[index]


class CodecBatchSampler(Sampler[list[int]]):
    """Deterministic shuffled batches that never mix incompatible D3 sizes."""

    def __init__(
        self,
        episodes: Sequence[Episode[Any, Any]],
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        groups: dict[object, list[int]] = {}
        for index, episode in enumerate(episodes):
            groups.setdefault(codec_for(episode), []).append(index)
        self.groups = tuple(groups[key] for key in sorted(groups, key=repr))

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed)
        batches: list[list[int]] = []
        for group in self.groups:
            if self.shuffle:
                order = torch.randperm(len(group), generator=generator).tolist()
                indices = [group[index] for index in order]
            else:
                indices = list(group)
            batches.extend(
                indices[start : start + self.batch_size]
                for start in range(0, len(indices), self.batch_size)
            )
        if self.shuffle and len(batches) > 1:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in order]
        yield from batches

    def __len__(self) -> int:
        return sum(
            (len(group) + self.batch_size - 1) // self.batch_size
            for group in self.groups
        )


def make_dataloader(
    episodes: Sequence[Episode[Any, Any]],
    *,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader[TransitionBatch]:
    dataset = EpisodeDataset(episodes)
    sampler = CodecBatchSampler(
        dataset.episodes, batch_size, shuffle=shuffle, seed=seed
    )
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_episodes)
