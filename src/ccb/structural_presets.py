from __future__ import annotations

from typing import Any

from ccb.dataset import OfficialEvaluationFirewall, SplitConfig, generate_split
from ccb.domains.alien_grid import AlienGridDomain, GridOperation
from ccb.domains.social_logic import SocialLogicDomain
from ccb.domains.symbolic_pointers import PointerOperation, SymbolicPointersDomain
from ccb.records import Episode
from ccb.structural import (
    contains_d1_semantic_subprogram,
    contains_operation_pair,
    d1_semantic_subprogram_signatures,
    d1_transformation_signature,
    generate_filtered_firewalled_split,
    generate_filtered_split,
    transition_overlap_audit,
)


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


def build_d1_semantic_structural_splits(
    *, seeds_per_depth: int = 50
) -> tuple[dict[str, tuple[Episode[Any, Any], ...]], dict[str, Any]]:
    """Build a semantically screened, randomized-initial-state D1 suite.

    It excludes every training or validation subprogram with the same net grid
    permutation as the reserved pair, not only that pair's literal tokens.
    """

    if seeds_per_depth < 1:
        raise ValueError("seeds_per_depth must be positive")
    domain = AlienGridDomain(randomize_initial_state=True)
    firewall = OfficialEvaluationFirewall.from_official_records("d1")
    reserved = (GridOperation.ROTATE_90_CW, GridOperation.SHIFT_ROW_2_LEFT)
    reserved_signature = d1_transformation_signature(reserved)
    requires_direct_pair = lambda episode: contains_operation_pair(episode, *reserved)
    excludes_reserved_semantics = lambda episode: not contains_d1_semantic_subprogram(
        episode, reserved_signature
    )

    # The test is generated first. Its concrete state-transition triples are
    # then protected from validation and training reuse.
    forbidden_transitions: set[str] = set()
    test_depth = generate_filtered_firewalled_split(
        domain.generate,
        SplitConfig(
            "test_semantic_pair_depth",
            (25, 30, 35, 40, 45, 50),
            seeds_per_depth,
            41_000_000,
        ),
        requires_direct_pair,
        firewall=firewall,
        forbidden_transition_fingerprints=forbidden_transitions,
    )
    test_matched = generate_filtered_firewalled_split(
        domain.generate,
        SplitConfig(
            "test_semantic_pair_matched", (5, 10, 15, 20), seeds_per_depth, 42_000_000
        ),
        requires_direct_pair,
        firewall=firewall,
        forbidden_transition_fingerprints=forbidden_transitions,
    )
    validation = generate_filtered_firewalled_split(
        domain.generate,
        SplitConfig(
            "validation_semantic",
            tuple(range(1, 21)),
            max(1, seeds_per_depth // 4),
            43_000_000,
        ),
        excludes_reserved_semantics,
        firewall=firewall,
        forbidden_transition_fingerprints=forbidden_transitions,
    )
    train = generate_filtered_firewalled_split(
        domain.generate,
        SplitConfig("train_semantic", tuple(range(1, 21)), seeds_per_depth, 44_000_000),
        excludes_reserved_semantics,
        firewall=firewall,
        forbidden_transition_fingerprints=forbidden_transitions,
    )
    splits = {
        "train": train,
        "validation": validation,
        "test_semantic_pair_matched": test_matched,
        "test_semantic_pair_depth": test_depth,
    }
    transition_audit = transition_overlap_audit(splits)
    semantic_exposure = {
        name: sum(
            reserved_signature in d1_semantic_subprogram_signatures(episode)
            for episode in episodes
        )
        for name, episodes in splits.items()
    }
    if not transition_audit["transition_disjoint"]:
        raise RuntimeError("semantic D1 split has cross-split transition overlap")
    if semantic_exposure["train"] or semantic_exposure["validation"]:
        raise RuntimeError("reserved semantic transformation leaked into training")
    return splits, {
        "schema": "ccb_d1_semantic_structural_v1",
        "randomized_initial_states": True,
        "reserved_tokens": [operation.value for operation in reserved],
        "reserved_transformation": list(reserved_signature),
        "semantic_segment_exposure": semantic_exposure,
        "transition_audit": transition_audit,
    }
