from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ccb.records import Episode
from ccb.serialization import canonical_json, serialize_episode, stable_hash, to_primitive


@dataclass(frozen=True)
class SplitConfig:
    name: str
    depths: tuple[int, ...]
    seeds_per_depth: int
    base_seed: int


@dataclass(frozen=True)
class FirewallAudit:
    """Overlap counts against the immutable official evaluation records."""

    domain: str
    episodes: int
    seed_depth_overlap: int
    program_overlap: int
    instance_overlap: int

    @property
    def safe(self) -> bool:
        return not (
            self.seed_depth_overlap or self.program_overlap or self.instance_overlap
        )


@dataclass(frozen=True)
class OfficialEvaluationFirewall:
    """Fingerprint index that prevents official CCB records entering learn splits."""

    domain: str
    seed_depth_keys: frozenset[tuple[int, int]]
    program_hashes: frozenset[str]
    instance_hashes: frozenset[str]

    @classmethod
    def from_official_records(cls, domain: str) -> "OfficialEvaluationFirewall":
        # Imports stay local so ordinary dataset utilities do not require the
        # official files unless the firewall is explicitly requested.
        from ccb.official import load_official_records
        from ccb.presets import domain_generator

        records = load_official_records(domain)
        generator = domain_generator(domain)
        episodes = tuple(
            generator(depth=record.depth, seed=record.seed) for record in records
        )
        return cls(
            domain=domain,
            seed_depth_keys=frozenset((record.depth, record.seed) for record in records),
            program_hashes=frozenset(program_fingerprint(item) for item in episodes),
            instance_hashes=frozenset(instance_fingerprint(item) for item in episodes),
        )

    def blocks(self, episode: Episode[Any, Any]) -> bool:
        return (
            (episode.depth, episode.seed) in self.seed_depth_keys
            or program_fingerprint(episode) in self.program_hashes
            or instance_fingerprint(episode) in self.instance_hashes
        )

    def audit(self, episodes: Sequence[Episode[Any, Any]]) -> FirewallAudit:
        seed_overlap = sum(
            (episode.depth, episode.seed) in self.seed_depth_keys for episode in episodes
        )
        program_overlap = sum(
            program_fingerprint(episode) in self.program_hashes for episode in episodes
        )
        instance_overlap = sum(
            instance_fingerprint(episode) in self.instance_hashes for episode in episodes
        )
        return FirewallAudit(
            domain=self.domain,
            episodes=len(episodes),
            seed_depth_overlap=seed_overlap,
            program_overlap=program_overlap,
            instance_overlap=instance_overlap,
        )

    def assert_safe(self, episodes: Sequence[Episode[Any, Any]]) -> FirewallAudit:
        audit = self.audit(episodes)
        if not audit.safe:
            raise ValueError(
                "official evaluation contamination detected: "
                f"seed/depth={audit.seed_depth_overlap}, "
                f"program={audit.program_overlap}, instance={audit.instance_overlap}"
            )
        return audit


def generate_split(
    generator: Callable[..., Episode[Any, Any]], config: SplitConfig
) -> tuple[Episode[Any, Any], ...]:
    episodes: list[Episode[Any, Any]] = []
    for depth_index, depth in enumerate(config.depths):
        for offset in range(config.seeds_per_depth):
            seed = config.base_seed + depth_index * config.seeds_per_depth + offset
            episodes.append(generator(depth=depth, seed=seed))
    return tuple(episodes)


def generate_firewalled_split(
    generator: Callable[..., Episode[Any, Any]],
    config: SplitConfig,
    firewall: OfficialEvaluationFirewall,
    *,
    forbidden_instances: set[str] | None = None,
    max_candidates_per_example: int = 10_000,
) -> tuple[Episode[Any, Any], ...]:
    """Generate a deterministic split while rejecting official/existing instances.

    Candidate seeds occupy a large, deterministic range per depth. Rejection does
    not change any other depth's seed stream, making rebuilds reproducible.
    """

    seen = forbidden_instances
    episodes: list[Episode[Any, Any]] = []
    stride = config.seeds_per_depth * max_candidates_per_example
    for depth_index, depth in enumerate(config.depths):
        accepted = 0
        candidate_offset = 0
        while accepted < config.seeds_per_depth:
            if candidate_offset >= stride:
                raise RuntimeError(
                    f"could not generate {config.name} depth={depth} without overlap"
                )
            seed = config.base_seed + depth_index * stride + candidate_offset
            candidate_offset += 1
            episode = generator(depth=depth, seed=seed)
            fingerprint = instance_fingerprint(episode)
            if firewall.blocks(episode) or (seen is not None and fingerprint in seen):
                continue
            episodes.append(episode)
            if seen is not None:
                seen.add(fingerprint)
            accepted += 1
    result = tuple(episodes)
    firewall.assert_safe(result)
    return result


def program_fingerprint(episode: Episode[Any, Any]) -> str:
    return stable_hash(
        {
            "domain": episode.domain,
            "operations": episode.operations,
            "structural_metadata": {
                key: value
                for key, value in episode.metadata.items()
                if key not in {"query", "query_answer"}
            },
        }
    )


def instance_fingerprint(episode: Episode[Any, Any]) -> str:
    return stable_hash(
        {
            "domain": episode.domain,
            "initial_state": episode.initial_state,
            "operations": episode.operations,
            "metadata": episode.metadata,
        }
    )


def audit_disjoint_splits(
    splits: Mapping[str, Sequence[Episode[Any, Any]]]
) -> dict[str, Any]:
    instance_sets = {
        name: {instance_fingerprint(episode) for episode in episodes}
        for name, episodes in splits.items()
    }
    program_sets = {
        name: {program_fingerprint(episode) for episode in episodes}
        for name, episodes in splits.items()
    }
    overlaps: list[dict[str, Any]] = []
    names = sorted(splits)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlaps.append(
                {
                    "left": left,
                    "right": right,
                    "instance_overlap": len(instance_sets[left] & instance_sets[right]),
                    "program_overlap": len(program_sets[left] & program_sets[right]),
                }
            )
    return {
        "split_sizes": {name: len(episodes) for name, episodes in splits.items()},
        "overlaps": overlaps,
        "instances_disjoint": all(item["instance_overlap"] == 0 for item in overlaps),
        "programs_disjoint": all(item["program_overlap"] == 0 for item in overlaps),
    }


def build_manifest(
    splits: Mapping[str, Sequence[Episode[Any, Any]]],
    *,
    config: Mapping[str, Any],
    official_firewall: OfficialEvaluationFirewall | None = None,
) -> dict[str, Any]:
    records = {
        name: [
            {
                "seed": episode.seed,
                "depth": episode.depth,
                "instance_hash": instance_fingerprint(episode),
                "program_hash": program_fingerprint(episode),
            }
            for episode in episodes
        ]
        for name, episodes in splits.items()
    }
    manifest = {
        "schema": "ccb_manifest_v2",
        "config": to_primitive(config),
        "splits": records,
        "audit": audit_disjoint_splits(splits),
    }
    if official_firewall is not None:
        manifest["official_firewall"] = {
            name: to_primitive(official_firewall.assert_safe(episodes))
            for name, episodes in splits.items()
        }
    manifest["manifest_hash"] = sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
    return manifest


def rehash_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy whose hash covers every field except the hash itself."""

    result = dict(manifest)
    result.pop("manifest_hash", None)
    result["manifest_hash"] = sha256(canonical_json(result).encode("utf-8")).hexdigest()
    return result


def write_jsonl(path: Path, episodes: Iterable[Episode[Any, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for episode in episodes:
            stream.write(serialize_episode(episode))
            stream.write("\n")


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
