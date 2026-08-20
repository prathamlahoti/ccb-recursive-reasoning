from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Generic, Mapping, Sequence, TypeVar

StateT = TypeVar("StateT")
OperationT = TypeVar("OperationT")


@dataclass(frozen=True)
class Transition(Generic[StateT, OperationT]):
    """One deterministic state transition in an episode."""

    index: int
    operation: OperationT
    before: StateT
    after: StateT


@dataclass(frozen=True)
class Episode(Generic[StateT, OperationT]):
    """A generated problem with its complete oracle trace."""

    domain: str
    seed: int
    initial_state: StateT
    operations: Sequence[OperationT]
    transitions: Sequence[Transition[StateT, OperationT]]
    final_state: StateT
    metadata: Mapping[str, Any]
    specification: str = "ccb_learn_v1"

    @property
    def depth(self) -> int:
        return len(self.operations)

    @property
    def trace(self) -> tuple[StateT, ...]:
        return tuple(transition.after for transition in self.transitions)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
