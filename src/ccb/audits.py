from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ccb.dataset import instance_fingerprint, program_fingerprint
from ccb.domains.alien_grid import GridOperation, GridState, apply_grid_operation
from ccb.records import Episode
from ccb.serialization import stable_hash


@dataclass(frozen=True)
class ReachabilityLevel:
    depth: int
    total_programs: int
    distinct_endpoints: int
    new_endpoints: int

    @property
    def endpoint_compression(self) -> float:
        return self.total_programs / self.distinct_endpoints


def alien_grid_reachability(
    *,
    maximum_depth: int,
    operations: Sequence[GridOperation],
    initial_state: GridState = ((1, 2, 3), (4, 5, 6), (7, 8, 9)),
) -> tuple[ReachabilityLevel, ...]:
    """Enumerate exact-depth endpoints for the finite D1 state space.

    This exposes a central shortcut risk: exponentially many operation strings
    collapse onto a small number of grid permutations.
    """

    if maximum_depth < 0:
        raise ValueError("maximum_depth must be non-negative")
    if not operations:
        raise ValueError("at least one operation is required")

    frontier = {initial_state}
    seen = {initial_state}
    levels = [ReachabilityLevel(0, 1, 1, 1)]
    for depth in range(1, maximum_depth + 1):
        frontier = {
            apply_grid_operation(state, operation)
            for state in frontier
            for operation in operations
        }
        new = frontier - seen
        seen.update(frontier)
        levels.append(
            ReachabilityLevel(
                depth=depth,
                total_programs=len(operations) ** depth,
                distinct_endpoints=len(frontier),
                new_endpoints=len(new),
            )
        )
    return tuple(levels)


def _entropy(counts: Iterable[int]) -> float:
    values = tuple(counts)
    total = sum(values)
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in values)


def dataset_shortcut_audit(episodes: Sequence[Episode[Any, Any]]) -> dict[str, Any]:
    """Measure duplicate programs, endpoints, and label concentration."""

    program_counts = Counter(program_fingerprint(episode) for episode in episodes)
    instance_counts = Counter(instance_fingerprint(episode) for episode in episodes)
    endpoint_counts = Counter(stable_hash(episode.final_state) for episode in episodes)
    depth_counts = Counter(episode.depth for episode in episodes)
    identity_count = sum(
        episode.final_state == episode.initial_state for episode in episodes
    )
    size = len(episodes)
    return {
        "examples": size,
        "unique_instances": len(instance_counts),
        "unique_programs": len(program_counts),
        "unique_endpoints": len(endpoint_counts),
        "duplicate_instances": size - len(instance_counts),
        "duplicate_programs": size - len(program_counts),
        "endpoint_entropy_bits": _entropy(endpoint_counts.values()),
        "maximum_endpoint_frequency": max(endpoint_counts.values(), default=0),
        "identity_rate": identity_count / size if size else 0.0,
        "depth_histogram": dict(sorted(depth_counts.items())),
    }

