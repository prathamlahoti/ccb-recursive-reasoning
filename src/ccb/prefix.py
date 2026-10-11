"""Training-only prefix weighting and target-based refinement diagnostics.

These functions never supply labels to the model. Prefix weighting is a
supervised ablation, not a guarantee of monotonic or correct execution.
"""
from __future__ import annotations

import torch
from torch import Tensor


def correct_prefix(predictions: Tensor, targets: Tensor, step_mask: Tensor) -> Tensor:
    matches = (predictions == targets).all(dim=-1) | ~step_mask
    return matches.long().cumprod(dim=1).bool() & step_mask


def prefix_weights(
    predictions: Tensor, targets: Tensor, step_mask: Tensor, *,
    frontier_weight: float = 0.0, preservation_weight: float = 0.0,
    previous_predictions: Tensor | None = None, continuing_rows: Tensor | None = None,
) -> Tensor:
    if frontier_weight < 0 or preservation_weight < 0:
        raise ValueError("prefix weights must be nonnegative")
    prefix = correct_prefix(predictions.detach(), targets, step_mask)
    length = prefix.sum(dim=1)
    positions = torch.arange(step_mask.shape[1], device=step_mask.device)[None]
    frontier = (positions == length[:, None]) & step_mask
    weights = step_mask.to(torch.float64) * (1 + frontier_weight * frontier)
    if previous_predictions is not None:
        if continuing_rows is None:
            raise ValueError("previous predictions require continuing_rows")
        preserved = correct_prefix(previous_predictions.detach(), targets, step_mask)
        preserved &= continuing_rows[:, None]
        weights += preservation_weight * preserved
    return weights


def refinement_metrics(
    predictions: Tensor, targets: Tensor, step_mask: Tensor, *,
    previous_predictions: Tensor | None = None, continuing_rows: Tensor | None = None,
) -> dict[str, float | int]:
    prefix = correct_prefix(predictions, targets, step_mask)
    metrics: dict[str, float | int] = {"mean_correct_prefix": float(prefix.sum(1).float().mean())}
    if previous_predictions is not None:
        if continuing_rows is None:
            raise ValueError("previous predictions require continuing_rows")
        prior = correct_prefix(previous_predictions, targets, step_mask)
        eligible = prior & continuing_rows[:, None]
        correct_now = (predictions == targets).all(-1)
        metrics["previous_prefix_states"] = int(eligible.sum())
        metrics["regressed_prefix_states"] = int((eligible & ~correct_now).sum())
        metrics["prefix_change_sum"] = int(
            ((prefix.sum(1) - prior.sum(1)) * continuing_rows).sum()
        )
        metrics["continuing_examples"] = int(continuing_rows.sum())
    return metrics
