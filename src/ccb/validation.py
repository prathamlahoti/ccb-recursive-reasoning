from __future__ import annotations

from typing import Any

from ccb.domains.alien_grid import apply_grid_operation
from ccb.domains.social_logic import signed_closure
from ccb.domains.symbolic_pointers import apply_pointer_operation
from ccb.records import Episode


class EpisodeValidationError(ValueError):
    pass


def validate_episode(episode: Episode[Any, Any]) -> None:
    """Replay an episode and reject any inconsistent oracle trace."""

    if episode.depth < 1:
        raise EpisodeValidationError("episodes must contain at least one operation")
    if len(episode.transitions) != episode.depth:
        raise EpisodeValidationError("transition count does not match depth")

    state = episode.initial_state
    accumulated_operations: list[Any] = []
    for expected_index, (operation, transition) in enumerate(
        zip(episode.operations, episode.transitions), start=1
    ):
        if transition.index != expected_index:
            raise EpisodeValidationError("transition indices must be consecutive")
        if transition.operation != operation or transition.before != state:
            raise EpisodeValidationError("transition input does not match episode")
        accumulated_operations.append(operation)
        if episode.domain == "alien_grid":
            expected = apply_grid_operation(state, operation)
        elif episode.domain == "symbolic_pointers":
            expected = apply_pointer_operation(state, operation)
        elif episode.domain == "social_logic":
            expected = signed_closure(len(state), tuple(accumulated_operations))
        else:
            raise EpisodeValidationError(f"unknown domain: {episode.domain}")
        if transition.after != expected:
            raise EpisodeValidationError(f"incorrect oracle state at step {expected_index}")
        state = expected

    if state != episode.final_state:
        raise EpisodeValidationError("final state does not match replayed trace")

