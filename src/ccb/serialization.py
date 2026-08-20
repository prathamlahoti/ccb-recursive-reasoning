from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping, Sequence

from ccb.records import Episode


class SerializationError(ValueError):
    pass


def to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize value of type {type(value)!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_primitive(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def stable_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def serialize_episode(episode: Episode[Any, Any]) -> str:
    payload = {
        "schema": "ccb_episode_v1",
        "specification": episode.specification,
        "domain": episode.domain,
        "seed": episode.seed,
        "depth": episode.depth,
        "initial_state": episode.initial_state,
        "operations": episode.operations,
        "trace": episode.trace,
        "final_state": episode.final_state,
        "metadata": episode.metadata,
    }
    return canonical_json(payload)


def parse_prediction(text: str, *, expected_depth: int) -> dict[str, Any]:
    """Strictly parse the canonical model-output JSON schema.

    The parser intentionally performs no repair. Natural-language CCB parsing
    will be a separate compatibility layer because the paper's original regex
    implementation is not publicly specified.
    """

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise SerializationError(f"Invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise SerializationError("Prediction must be a JSON object")
    if set(payload) != {"trace", "answer"}:
        raise SerializationError("Prediction must contain exactly 'trace' and 'answer'")
    trace = payload["trace"]
    if not isinstance(trace, list):
        raise SerializationError("trace must be a list")
    if len(trace) != expected_depth:
        raise SerializationError(
            f"trace length {len(trace)} does not match expected depth {expected_depth}"
        )
    return payload


def serialize_prediction(trace: Sequence[Any], answer: Any) -> str:
    return canonical_json({"trace": trace, "answer": answer})

