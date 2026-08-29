from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from ccb.dataset import (
    OfficialEvaluationFirewall,
    SplitConfig,
    generate_firewalled_split,
)
from ccb.domains.alien_grid import AlienGridDomain
from ccb.domains.social_logic import SocialLogicDomain
from ccb.domains.symbolic_pointers import SymbolicPointersDomain
from ccb.records import Episode


PRIMARY_SPLITS = (
    SplitConfig("train", (5, 10, 15, 20), 100, 10_000_000),
    SplitConfig("validation", (5, 10, 15, 20), 25, 20_000_000),
    SplitConfig("test_depth", (25, 30, 35, 40, 45, 50), 100, 30_000_000),
    SplitConfig("test_strong", (60, 80, 100), 100, 40_000_000),
)


def domain_generator(name: str) -> Callable[..., Episode[Any, Any]]:
    domains = {
        "d1": AlienGridDomain().generate,
        "d2": SymbolicPointersDomain().generate,
        "d3": SocialLogicDomain().generate,
    }
    try:
        return domains[name]
    except KeyError as error:
        raise ValueError(f"unknown domain preset: {name}") from error


def primary_config(domain: str) -> dict[str, Any]:
    return {
        "benchmark": "ccb_learn_v1",
        "domain": domain,
        "splits": [asdict(config) for config in PRIMARY_SPLITS],
    }


def build_primary_splits(
    domain: str,
) -> tuple[
    dict[str, tuple[Episode[Any, Any], ...]], OfficialEvaluationFirewall
]:
    """Build the definitive official-semantics CCB-Learn split suite."""

    generator = domain_generator(domain)
    firewall = OfficialEvaluationFirewall.from_official_records(domain)
    splits = {
        config.name: generate_firewalled_split(
            generator,
            config,
            firewall,
        )
        for config in PRIMARY_SPLITS
    }
    return splits, firewall
