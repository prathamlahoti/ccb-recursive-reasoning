from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from torch import nn

from ccb.decoding import constrained_argmax
from ccb.encoding import TransitionBatch
from ccb.metrics import (
    bootstrap_retention_interval,
    clopper_pearson_interval,
    fit_step_retention,
    normalized_accuracy_depth_auc,
    success_horizon,
)
from ccb.models.common import ModelOutput


@dataclass
class DepthCounts:
    examples: int = 0
    final_correct: int = 0
    trace_correct: int = 0
    tfbc: int = 0
    transitions: int = 0
    transitions_correct: int = 0
    elements: int = 0
    elements_correct: int = 0
    valid_states: int = 0
    first_divergence_sum: int = 0
    first_divergence_count: int = 0


def _valid_states(predicted: torch.Tensor, batch: TransitionBatch) -> torch.Tensor:
    if batch.domain == "alien_grid":
        ordered = predicted.sort(dim=-1).values
        expected = torch.arange(9, device=predicted.device).expand_as(ordered)
        return (ordered == expected).all(dim=-1)
    if batch.domain == "symbolic_pointers":
        return torch.ones(predicted.shape[:2], dtype=torch.bool, device=predicted.device)
    if batch.domain == "social_logic":
        agents = batch.codec.number_of_agents
        if agents is None:
            raise ValueError("social codec lacks number_of_agents")
        matrices = predicted.reshape(*predicted.shape[:2], agents, agents)
        symmetric = (matrices == matrices.transpose(-1, -2)).all(dim=(-1, -2))
        diagonal = matrices.diagonal(dim1=-2, dim2=-1)
        return symmetric & (diagonal == 2).all(dim=-1)
    raise ValueError(f"unknown domain: {batch.domain}")


def _summarize(counts: DepthCounts) -> dict[str, Any]:
    final_interval = clopper_pearson_interval(counts.final_correct, counts.examples)
    return {
        **asdict(counts),
        "final_exact_accuracy": counts.final_correct / counts.examples,
        "final_exact_ci95": list(final_interval),
        "trace_exact_accuracy": counts.trace_correct / counts.examples,
        "transition_exact_accuracy": counts.transitions_correct / counts.transitions,
        "element_accuracy": counts.elements_correct / counts.elements,
        "valid_state_rate": counts.valid_states / counts.transitions,
        "tfbc_rate_all": counts.tfbc / counts.examples,
        "tfbc_rate_given_final_correct": (
            counts.tfbc / counts.final_correct if counts.final_correct else 0.0
        ),
        "mean_first_divergence_k_star": (
            counts.first_divergence_sum / counts.first_divergence_count
            if counts.first_divergence_count
            else None
        ),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    batches: Iterable[TransitionBatch],
    *,
    device: torch.device | str = "cpu",
    constrained: bool = False,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Evaluate all examples and aggregate exact, trace, depth, and loop metrics."""

    model.eval()
    depth_counts: dict[int, DepthCounts] = defaultdict(DepthCounts)
    overall = DepthCounts()
    loop_final: list[list[int]] = []  # [correct, examples] per recursive loop

    for original_batch in batches:
        batch = original_batch.to(device)
        output: ModelOutput = model(batch)  # type: ignore[assignment]
        predicted = (
            constrained_argmax(output.logits, batch.codec)
            if constrained
            else output.logits.argmax(dim=-1)
        )
        matches = predicted == batch.targets
        transition_matches = matches.all(dim=-1)
        valid_states = _valid_states(predicted, batch)
        rows = torch.arange(predicted.shape[0], device=predicted.device)
        final_indices = batch.depths - 1
        final_matches = transition_matches[rows, final_indices]
        trace_matches = (transition_matches | ~batch.step_mask).all(dim=1)

        for loop_index, logits in enumerate(output.loop_logits):
            while len(loop_final) <= loop_index:
                loop_final.append([0, 0])
            loop_predicted = logits.argmax(dim=-1)
            loop_correct = (
                loop_predicted[rows, final_indices]
                == batch.targets[rows, final_indices]
            ).all(dim=-1)
            loop_final[loop_index][0] += int(loop_correct.sum())
            loop_final[loop_index][1] += len(loop_correct)

        for row in range(predicted.shape[0]):
            depth = int(batch.depths[row])
            transition_row = transition_matches[row, :depth]
            final_correct = bool(final_matches[row])
            trace_correct = bool(trace_matches[row])
            divergence_positions = (~transition_row).nonzero(as_tuple=False)
            first_divergence = (
                int(divergence_positions[0, 0]) + 1
                if len(divergence_positions)
                else None
            )
            elements_correct = int(matches[row, :depth].sum())
            for cell in (depth_counts[depth], overall):
                cell.examples += 1
                cell.final_correct += int(final_correct)
                cell.trace_correct += int(trace_correct)
                cell.tfbc += int(final_correct and not trace_correct)
                cell.transitions += depth
                cell.transitions_correct += int(transition_row.sum())
                cell.elements += depth * batch.codec.state_size
                cell.elements_correct += elements_correct
                cell.valid_states += int(valid_states[row, :depth].sum())
                if first_divergence is not None:
                    cell.first_divergence_sum += first_divergence
                    cell.first_divergence_count += 1

    if overall.examples == 0:
        raise ValueError("evaluation received no examples")
    per_depth = {str(depth): _summarize(depth_counts[depth]) for depth in sorted(depth_counts)}
    retention_counts = {
        depth: (counts.final_correct, counts.examples)
        for depth, counts in depth_counts.items()
    }
    retention = fit_step_retention(retention_counts)
    retention_interval = bootstrap_retention_interval(
        retention_counts,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    accuracy_by_depth = {
        depth: counts.final_correct / counts.examples
        for depth, counts in depth_counts.items()
    }
    result: dict[str, Any] = {
        "schema": "ccb_evaluation_v1",
        "decoding": "constrained" if constrained else "raw_argmax",
        "overall": _summarize(overall),
        "per_depth": per_depth,
        "per_step_retention_p_d": retention,
        "per_step_retention_ci95": list(retention_interval),
        "success_horizon_50": success_horizon(retention, 0.5),
        "success_horizon_90": success_horizon(retention, 0.9),
        "loop_final_exact_accuracy": [
            correct / examples for correct, examples in loop_final
        ],
    }
    result["normalized_accuracy_depth_auc"] = (
        normalized_accuracy_depth_auc(accuracy_by_depth)
        if len(accuracy_by_depth) > 1
        else math.nan
    )
    return result
