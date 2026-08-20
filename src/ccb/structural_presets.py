from __future__ import annotations

from typing import Any

from ccb.dataset import SplitConfig, generate_split
from ccb.domains.alien_grid import AlienGridDomain, GridOperation
from ccb.domains.social_logic import SocialLogicDomain
from ccb.domains.symbolic_pointers import PointerOperation, SymbolicPointersDomain
from ccb.records import Episode
from ccb.structural import contains_operation_pair, generate_filtered_split


def build_structural_splits(
    domain: str, *, seeds_per_depth: int = 50
) -> tuple[dict[str, tuple[Episode[Any, Any], ...]], dict[str, Any]]:
    if seeds_per_depth < 1:
        raise ValueError("seeds_per_depth must be positive")

    if domain == "d1":
        generator = AlienGridDomain().generate
        reserved = (GridOperation.ROTATE_90_CW, GridOperation.SHIFT_ROW_2_LEFT)
        train_config = SplitConfig("train", tuple(range(5, 21)), seeds_per_depth, 10_000_000)
        test_config = SplitConfig("test_composition", (5, 10, 15, 20, 25), seeds_per_depth, 11_000_000)
        has_reserved = lambda episode: contains_operation_pair(episode, *reserved)
        splits = {
            "train": generate_filtered_split(generator, train_config, lambda episode: not has_reserved(episode)),
            "test_composition": generate_filtered_split(generator, test_config, has_reserved),
        }
        audit = {
            "holdout": "ordered_operation_bigram",
            "reserved": [operation.value for operation in reserved],
            "train_exposure": sum(has_reserved(episode) for episode in splits["train"]),
            "test_exposure": sum(has_reserved(episode) for episode in splits["test_composition"]),
        }
        return splits, audit

    if domain == "d2":
        generator = SymbolicPointersDomain().generate
        train_config = SplitConfig("train", tuple(range(5, 21)), seeds_per_depth, 12_000_000)
        test_config = SplitConfig("test_composition", (5, 10, 15, 20, 25), seeds_per_depth, 13_000_000)
        reserved = (
            PointerOperation.SET_D_TO_A_PLUS_B,
            PointerOperation.SET_C_TO_G_MINUS_E,
        )
        has_reserved = lambda episode: contains_operation_pair(episode, *reserved)
        splits = {
            "train": generate_filtered_split(
                generator, train_config, lambda episode: not has_reserved(episode)
            ),
            "test_composition": generate_filtered_split(
                generator, test_config, has_reserved
            ),
        }
        audit = {
            "holdout": "ordered_operation_bigram",
            "reserved": [operation.value for operation in reserved],
            "train_exposure": sum(has_reserved(episode) for episode in splits["train"]),
            "test_exposure": sum(
                has_reserved(episode) for episode in splits["test_composition"]
            ),
        }
        return splits, audit

    if domain == "d3":
        train: list[Episode[Any, Any]] = []
        test: list[Episode[Any, Any]] = []
        for index, agents in enumerate((6, 8)):
            config = SplitConfig(f"train_n{agents}", tuple(range(5, 21)), seeds_per_depth, 14_000_000 + index * 100_000)
            train.extend(generate_split(SocialLogicDomain(agents, "chain").generate, config))
        for index, agents in enumerate((10, 12)):
            config = SplitConfig(f"test_n{agents}", (25, 50), seeds_per_depth, 15_000_000 + index * 100_000)
            test.extend(generate_split(SocialLogicDomain(agents, "star").generate, config))
        splits = {"train": tuple(train), "test_size_topology": tuple(test)}
        audit = {
            "holdout": "graph_size_and_topology",
            "train_sizes": [6, 8],
            "test_sizes": [10, 12],
            "train_topology": "chain",
            "test_topology": "star",
            "feature_disjoint": True,
        }
        return splits, audit

    raise ValueError(f"unknown structural domain: {domain}")
