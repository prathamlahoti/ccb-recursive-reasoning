from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from ccb.records import Episode, Transition

GridState = tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
OFFICIAL_INITIAL_GRID: GridState = ((1, 2, 3), (4, 5, 6), (7, 8, 9))


class GridOperation(str, Enum):
    ROTATE_90_CW = "ROTATE_90_CW"
    SHIFT_ROW_2_LEFT = "SHIFT_ROW_2_LEFT"
    SWAP_CORNERS = "SWAP_CORNERS"
    REVERSE_GRID = "REVERSE_GRID"
    FLIP_HORIZONTAL = "FLIP_HORIZONTAL"
    TRANSPOSE_GRID = "TRANSPOSE_GRID"
    SHIFT_COL_1_UP = "SHIFT_COL_1_UP"


OFFICIAL_OPERATIONS = tuple(GridOperation)


def _validate_grid(state: GridState) -> None:
    if len(state) != 3 or any(len(row) != 3 for row in state):
        raise ValueError("Alien Grid state must be 3x3")
    values = [value for row in state for value in row]
    if len(set(values)) != 9:
        raise ValueError("Alien Grid requires nine distinct entities")


def _as_grid(rows: Iterable[Iterable[int]]) -> GridState:
    materialized = tuple(tuple(row) for row in rows)
    state: GridState = materialized  # type: ignore[assignment]
    _validate_grid(state)
    return state


def apply_grid_operation(state: GridState, operation: GridOperation) -> GridState:
    _validate_grid(state)
    rows = [list(row) for row in state]
    if operation is GridOperation.ROTATE_90_CW:
        return _as_grid(zip(*rows[::-1]))
    if operation is GridOperation.SHIFT_ROW_2_LEFT:
        rows[1] = rows[1][1:] + rows[1][:1]
    elif operation is GridOperation.SWAP_CORNERS:
        old_tl, old_tr = rows[0][0], rows[0][2]
        old_bl, old_br = rows[2][0], rows[2][2]
        rows[0][0], rows[0][2], rows[2][0], rows[2][2] = (
            old_br,
            old_bl,
            old_tr,
            old_tl,
        )
    elif operation is GridOperation.REVERSE_GRID:
        flattened = [value for row in rows for value in row][::-1]
        rows = [flattened[index : index + 3] for index in range(0, 9, 3)]
    elif operation is GridOperation.FLIP_HORIZONTAL:
        rows = [row[::-1] for row in rows]
    elif operation is GridOperation.TRANSPOSE_GRID:
        return _as_grid(zip(*rows))
    elif operation is GridOperation.SHIFT_COL_1_UP:
        column = [rows[index][0] for index in range(3)]
        column = column[1:] + column[:1]
        for index, value in enumerate(column):
            rows[index][0] = value
    else:
        raise ValueError(f"Unsupported grid operation: {operation}")
    return _as_grid(rows)


def format_grid(state: GridState) -> str:
    return str([list(row) for row in state])


@dataclass(frozen=True)
class AlienGridDomain:
    """Official CCB D1 semantics, with optional randomized-start extension."""

    randomize_initial_state: bool = False

    def generate(self, *, depth: int, seed: int) -> Episode[GridState, GridOperation]:
        if depth < 1:
            raise ValueError("depth must be positive")
        rng = random.Random(seed)
        if self.randomize_initial_state:
            values = list(range(1, 10))
            rng.shuffle(values)
            initial = _as_grid(values[index : index + 3] for index in range(0, 9, 3))
        else:
            initial = OFFICIAL_INITIAL_GRID
        operations = tuple(rng.choice(OFFICIAL_OPERATIONS) for _ in range(depth))
        state = initial
        transitions: list[Transition[GridState, GridOperation]] = []
        for index, operation in enumerate(operations, start=1):
            next_state = apply_grid_operation(state, operation)
            transitions.append(Transition(index, operation, state, next_state))
            state = next_state
        provenance = "ccb_official" if not self.randomize_initial_state else "ccb_learn"
        return Episode(
            domain="alien_grid",
            seed=seed,
            initial_state=initial,
            operations=operations,
            transitions=tuple(transitions),
            final_state=state,
            metadata={
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
            f"Step {transition.index}: {format_grid(transition.after)}"
            for transition in episode.transitions
        ]
        return prompt, trace, format_grid(episode.final_state)

