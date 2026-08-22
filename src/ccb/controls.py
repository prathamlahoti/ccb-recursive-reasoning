"""Leakage-resistant input-ablation controls for CCB evaluations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn

from ccb.encoding import TransitionBatch
from ccb.evaluation import evaluate_model


def permute_operation_order(batch: TransitionBatch, *, seed: int) -> TransitionBatch:
    """Randomly reorder each program while preserving its operation multiset.

    Targets intentionally remain unchanged. Accuracy should consequently
    collapse for a model that truly consumes ordered program instructions.
    Padding positions are left untouched.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    operations = batch.operations.clone()
    for row, depth in enumerate(batch.depths.tolist()):
        if depth > 1:
            order = torch.randperm(depth, generator=generator, device=operations.device)
            operations[row, :depth] = operations[row, order]
    return TransitionBatch(
        batch.domain,
        batch.initial_state,
        operations,
        batch.targets,
        batch.codec,
        batch.step_mask,
        batch.depths,
    )


def corrupt_initial_states(batch: TransitionBatch) -> TransitionBatch:
    """Replace each initial state with another valid state while keeping targets.

    A cyclic row shift preserves valid states when the batch contains multiple
    examples. The final singleton batch is corrupted by a cyclic cell shift.
    """

    initial = batch.initial_state.roll(shifts=1, dims=0)
    if torch.equal(initial, batch.initial_state):
        initial = batch.initial_state.roll(shifts=1, dims=1)
    return TransitionBatch(
        batch.domain,
        initial,
        batch.operations,
        batch.targets,
        batch.codec,
        batch.step_mask,
        batch.depths,
    )


def evaluate_causality_controls(
    model: nn.Module,
    batches: Callable[[], Iterable[TransitionBatch]],
    *,
    device: torch.device | str,
    bootstrap_resamples: int = 2_000,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Evaluate normal and deliberately invalid causal-input conditions."""

    def transformed(
        transform: Callable[[TransitionBatch], TransitionBatch],
    ) -> Iterable[TransitionBatch]:
        return (transform(batch) for batch in batches())

    return {
        "normal": evaluate_model(
            model,
            batches(),
            device=device,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=seed,
        ),
        "operation_order_permuted": evaluate_model(
            model,
            transformed(lambda batch: permute_operation_order(batch, seed=seed)),
            device=device,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=seed,
        ),
        "initial_state_corrupted": evaluate_model(
            model,
            transformed(corrupt_initial_states),
            device=device,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=seed,
        ),
    }
