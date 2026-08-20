from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from ccb.records import Episode, Transition

VARIABLES = tuple("ABCDEFG")
PointerState = tuple[int, int, int, int, int, int, int]
OFFICIAL_INITIAL_POINTERS: PointerState = (1, 2, 3, 4, 5, 6, 7)


class PointerOperation(str, Enum):
    SHIFT_RIGHT = "SHIFT_RIGHT"
    SHIFT_LEFT = "SHIFT_LEFT"
    SWAP_A_G = "SWAP_A_G"
    SET_D_TO_A_PLUS_B = "SET_D_TO_A_PLUS_B"
    SET_C_TO_G_MINUS_E = "SET_C_TO_G_MINUS_E"


OFFICIAL_POINTER_OPERATIONS = tuple(PointerOperation)


def _validate_state(state: PointerState) -> None:
    if len(state) != len(VARIABLES):
        raise ValueError("Symbolic Pointers requires seven registers")
    if any(value < 0 or value > 9 for value in state):
        raise ValueError("Register values must be digits")


def apply_pointer_operation(state: PointerState, operation: PointerOperation) -> PointerState:
    _validate_state(state)
    values = list(state)
    if operation is PointerOperation.SHIFT_RIGHT:
        result = state[-1:] + state[:-1]
    elif operation is PointerOperation.SHIFT_LEFT:
        result = state[1:] + state[:1]
    elif operation is PointerOperation.SWAP_A_G:
        values[0], values[6] = state[6], state[0]
        result = tuple(values)
    elif operation is PointerOperation.SET_D_TO_A_PLUS_B:
        values[3] = (state[0] + state[1]) % 10
        result = tuple(values)
    elif operation is PointerOperation.SET_C_TO_G_MINUS_E:
        values[2] = (state[6] - state[4] + 10) % 10
        result = tuple(values)
    else:
        raise ValueError(f"Unsupported pointer operation: {operation}")
    typed: PointerState = result  # type: ignore[assignment]
    _validate_state(typed)
    return typed


def format_pointer_state(state: PointerState) -> str:
    return "[" + ", ".join(f"{name}={value}" for name, value in zip(VARIABLES, state)) + "]"


@dataclass(frozen=True)
class SymbolicPointersDomain:
    """Official CCB D2 semantics, with optional randomized-start extension."""

    randomize_initial_state: bool = False

    def generate(self, *, depth: int, seed: int) -> Episode[PointerState, PointerOperation]:
        if depth < 1:
            raise ValueError("depth must be positive")
        rng = random.Random(seed)
        if self.randomize_initial_state:
            initial: PointerState = tuple(rng.sample(range(10), 7))  # type: ignore[assignment]
        else:
            initial = OFFICIAL_INITIAL_POINTERS
        state = initial
        operations: list[PointerOperation] = []
        transitions: list[Transition[PointerState, PointerOperation]] = []
        for index in range(1, depth + 1):
            operation = rng.choice(OFFICIAL_POINTER_OPERATIONS)
            next_state = apply_pointer_operation(state, operation)
            operations.append(operation)
            transitions.append(Transition(index, operation, state, next_state))
            state = next_state
        provenance = "ccb_official" if not self.randomize_initial_state else "ccb_learn"
        return Episode(
            domain="symbolic_pointers",
            seed=seed,
            initial_state=initial,
            operations=tuple(operations),
            transitions=tuple(transitions),
            final_state=state,
            metadata={
                "variables": VARIABLES,
                "randomized_initial_state": self.randomize_initial_state,
                "provenance": provenance,
            },
            specification=f"{provenance}_v1",
        )

    def solve(self, *, depth: int, seed: int) -> tuple[str, list[str], str]:
        episode = self.generate(depth=depth, seed=seed)
        prompt = "\n".join(
            f"Step {index}: {operation.value}"
            for index, operation in enumerate(episode.operations, start=1)
        )
        trace = [
            f"Step {transition.index}: {format_pointer_state(transition.after)}"
            for transition in episode.transitions
        ]
        return prompt, trace, format_pointer_state(episode.final_state)

