from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable, Sequence

from ccb.dataset import OfficialEvaluationFirewall, SplitConfig
from ccb.domains.alien_grid import (
    OFFICIAL_INITIAL_GRID,
    GridOperation,
    GridState,
    apply_grid_operation,
)
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


def d1_transformation_signature(operations: Sequence[GridOperation]) -> tuple[int, ...]:
    """Canonical position permutation induced by a D1 program."""

    state = OFFICIAL_INITIAL_GRID
    for operation in operations:
        state = apply_grid_operation(state, operation)
    return tuple(value for row in state for value in row)


def d1_semantic_subprogram_signatures(
    episode: Episode[GridState, GridOperation],
) -> frozenset[tuple[int, ...]]:
    """Transformations of every nonempty contiguous operation segment."""

    signatures: set[tuple[int, ...]] = set()
    for start in range(episode.depth):
        state = OFFICIAL_INITIAL_GRID
        for operation in episode.operations[start:]:
            state = apply_grid_operation(state, operation)
            signatures.add(tuple(value for row in state for value in row))
    return frozenset(signatures)


def contains_d1_semantic_subprogram(
    episode: Episode[GridState, GridOperation], signature: tuple[int, ...]
) -> bool:
    return signature in d1_semantic_subprogram_signatures(episode)


def episode_transition_fingerprints(episode: Episode[Any, Any]) -> frozenset[str]:
    """Exact `(before, operation, after)` oracle transitions in an episode."""

    return frozenset(
        stable_hash(
            {
                "domain": episode.domain,
                "before": transition.before,
                "operation": transition.operation,
                "after": transition.after,
            }
        )
        for transition in episode.transitions
    )


def transition_overlap_audit(
    splits: dict[str, Sequence[Episode[Any, Any]]],
) -> dict[str, Any]:
    """Report concrete oracle-transition reuse across named splits."""

    signatures = {
        name: set().union(*(episode_transition_fingerprints(item) for item in episodes))
        if episodes
        else set()
        for name, episodes in splits.items()
    }
    names = sorted(signatures)
    overlaps = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlaps.append(
                {
                    "left": left,
                    "right": right,
                    "transition_overlap": len(signatures[left] & signatures[right]),
                }
            )
    return {
        "unique_transitions": {name: len(values) for name, values in signatures.items()},
        "overlaps": overlaps,
        "transition_disjoint": all(item["transition_overlap"] == 0 for item in overlaps),
    }


def generate_filtered_firewalled_split(
    generator: Callable[..., Episode[Any, Any]],
    config: SplitConfig,
    accept: Callable[[Episode[Any, Any]], bool],
    *,
    firewall: OfficialEvaluationFirewall,
    forbidden_transition_fingerprints: set[str] | None = None,
    maximum_attempt_multiplier: int = 10_000,
) -> tuple[Episode[Any, Any], ...]:
    """Sample deterministically while excluding official and held transitions."""

    if maximum_attempt_multiplier < 1:
        raise ValueError("maximum_attempt_multiplier must be positive")
    forbidden = set() if forbidden_transition_fingerprints is None else forbidden_transition_fingerprints
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
            attempt += 1
            episode = generator(depth=depth, seed=seed)
            transitions = episode_transition_fingerprints(episode)
            if firewall.blocks(episode) or not accept(episode) or transitions & forbidden:
                continue
            episodes.append(episode)
            forbidden.update(transitions)
            selected += 1
    result = tuple(episodes)
    firewall.assert_safe(result)
    return result
