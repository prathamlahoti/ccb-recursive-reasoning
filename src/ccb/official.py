from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable

from ccb.records import Episode

from ccb.domains.alien_grid import AlienGridDomain
from ccb.domains.social_logic import SocialLogicDomain
from ccb.domains.symbolic_pointers import SymbolicPointersDomain

OFFICIAL_COMMIT = "e1e953a7a33a8c2406f936a54f4900f740dea8dd"
OFFICIAL_DATA_HASHES = {
    "d1": "53de6530dd03be60c3ebc49e354c11dc99b30e47a4b250ee190779dac6dfbf90",
    "d2": "2886d19465d7ee250d5a0e8a4de3917ee61734353318d633be06b1df17a6c11f",
    "d3": "e1b43b4e160ef8928bc02875448d0eaa6c25797a606a60a5c5d3194b356c622a",
}


@dataclass(frozen=True)
class OfficialRecord:
    domain: str
    depth: int
    seed: int
    prompt: str
    trace: tuple[str, ...]
    answer: str


@dataclass(frozen=True)
class CompatibilityReport:
    domain: str
    records: int
    exact_matches: int
    mismatches: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.mismatches and self.records == self.exact_matches


def official_data_directory() -> Path:
    directory = Path(__file__).resolve().parents[2] / "data" / "ccb_official"
    if not directory.is_dir():
        raise FileNotFoundError(f"Official CCB data directory is missing: {directory}")
    return directory


def _data_path(domain: str) -> Path:
    if domain not in OFFICIAL_DATA_HASHES:
        raise ValueError(f"unknown official domain: {domain}")
    return official_data_directory() / f"{domain}.json"


def verify_data_hash(domain: str) -> bool:
    path = _data_path(domain)
    return sha256(path.read_bytes()).hexdigest() == OFFICIAL_DATA_HASHES[domain]


def load_official_records(domain: str) -> tuple[OfficialRecord, ...]:
    path = _data_path(domain)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"official {domain} data must be a JSON list")
    records = tuple(
        OfficialRecord(
            domain=domain,
            depth=int(item["depth"]),
            seed=int(item["seed"]),
            prompt=str(item["prompt"]),
            trace=tuple(str(step) for step in item["gt_trace"]),
            answer=str(item["gt_ans"]),
        )
        for item in payload
    )
    if len(records) != 400:
        raise ValueError(f"official {domain} data has {len(records)} records, expected 400")
    return records


def load_official_episodes(domain: str) -> tuple[Episode[Any, Any], ...]:
    """Regenerate structured episodes in the immutable official record order."""

    records = load_official_records(domain)
    generators = {
        "d1": AlienGridDomain().generate,
        "d2": SymbolicPointersDomain().generate,
        "d3": SocialLogicDomain().generate,
    }
    try:
        generate = generators[domain]
    except KeyError as error:
        raise ValueError(f"unknown official domain: {domain}") from error
    return tuple(generate(depth=item.depth, seed=item.seed) for item in records)


def _solver(domain: str) -> Callable[..., tuple[str, list[str], str]]:
    if domain == "d1":
        return AlienGridDomain().solve
    if domain == "d2":
        return SymbolicPointersDomain().solve
    if domain == "d3":
        return SocialLogicDomain().solve
    raise ValueError(f"unknown official domain: {domain}")


def verify_official_records(
    domain: str, records: Iterable[OfficialRecord] | None = None
) -> CompatibilityReport:
    official_records = tuple(records) if records is not None else load_official_records(domain)
    solve = _solver(domain)
    mismatches: list[str] = []
    exact_matches = 0
    for record in official_records:
        prompt, trace, answer = solve(depth=record.depth, seed=record.seed)
        differences = []
        if prompt != record.prompt:
            differences.append("prompt")
        if tuple(trace) != record.trace:
            differences.append("trace")
        if answer != record.answer:
            differences.append("answer")
        if differences:
            mismatches.append(
                f"{domain} depth={record.depth} seed={record.seed}: {','.join(differences)}"
            )
        else:
            exact_matches += 1
    return CompatibilityReport(
        domain=domain,
        records=len(official_records),
        exact_matches=exact_matches,
        mismatches=tuple(mismatches),
    )
