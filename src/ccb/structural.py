from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable, Sequence

from ccb.dataset import SplitConfig
from ccb.records import Episode
from ccb.serialization import stable_hash, to_primitive


def operation_ngrams(episode: Episode[Any, Any], n: int = 2) -> frozenset[str]:
    if n < 1:
        raise ValueError("n must be positive")
    operations = tuple(to_primitive(operation) for operation in episode.operations)
    return frozenset(
        stable_hash(operations[index : index + n])
        for index in range(max(0, len(operations) - n + 1))
    )


def structural_feature_audit(
    train: Sequence[Episode[Any, Any]],
    test: Sequence[Episode[Any, Any]],
    feature: Callable[[Episode[Any, Any]], Iterable[str]],
) -> dict[str, Any]:
    train_counts = Counter(item for episode in train for item in feature(episode))
    test_counts = Counter(item for episode in test for item in feature(episode))
    shared = set(train_counts) & set(test_counts)
    return {
        "train_features": len(train_counts),
        "test_features": len(test_counts),
        "shared_features": len(shared),
        "test_only_features": len(set(test_counts) - set(train_counts)),
        "feature_disjoint": not shared,
    }


def generate_filtered_split(
    generator: Callable[..., Episode[Any, Any]],
    config: SplitConfig,
    accept: Callable[[Episode[Any, Any]], bool],
    *,
    maximum_attempt_multiplier: int = 1_000,
) -> tuple[Episode[Any, Any], ...]:
    """Deterministic rejection sampling for explicit structural regimes."""

    if maximum_attempt_multiplier < 1:
        raise ValueError("maximum_attempt_multiplier must be positive")
    episodes: list[Episode[Any, Any]] = []
    for depth_index, depth in enumerate(config.depths):
        selected = 0
        attempt = 0
        while selected < config.seeds_per_depth:
            if attempt >= config.seeds_per_depth * maximum_attempt_multiplier:
                raise RuntimeError(f"could not fill {config.name} at depth {depth}")
            seed = (
                config.base_seed
                + depth_index * config.seeds_per_depth * maximum_attempt_multiplier
                + attempt
            )
            episode = generator(depth=depth, seed=seed)
            attempt += 1
            if accept(episode):
                episodes.append(episode)
                selected += 1
    return tuple(episodes)


def d3_size_feature(episode: Episode[Any, Any]) -> frozenset[str]:
    return frozenset({str(episode.metadata["number_of_agents"])})


def contains_operation_pair(episode: Episode[Any, Any], left: Any, right: Any) -> bool:
    return any(
        first == left and second == right
        for first, second in zip(episode.operations, episode.operations[1:])
    )
