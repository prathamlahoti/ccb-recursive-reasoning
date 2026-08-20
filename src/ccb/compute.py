from __future__ import annotations

from typing import Any

from torch import nn

from ccb.models import (
    DirectTransformer,
    FastSlowRecurrentModel,
    LoopedTransformer,
    RecurrentBaseline,
    SocialMessagePassingGNN,
    StateTransitionRecursiveModel,
    VanillaTRM,
)


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def compute_signature(model: nn.Module, *, depth: int) -> dict[str, Any]:
    """Architecture-independent accounting unit for matched-compute tables."""

    if depth < 1:
        raise ValueError("depth must be positive")
    if isinstance(model, DirectTransformer):
        updates = len(model.encoder.layers)
        unit = "full_sequence_transformer_block"
    elif isinstance(model, RecurrentBaseline):
        updates = depth * model.recurrent.num_layers
        unit = "recurrent_cell_layer"
    elif isinstance(model, LoopedTransformer):
        updates = model.loops
        unit = "full_sequence_shared_transformer_block"
    elif isinstance(model, VanillaTRM):
        updates = depth * model.loops * (model.low_cycles + 1)
        unit = "prefix_state_reasoning_block"
    elif isinstance(model, FastSlowRecurrentModel):
        updates = depth * (model.fast_loops + 1)
        unit = "state_cell_recurrent_update"
    elif isinstance(model, StateTransitionRecursiveModel):
        updates = depth * (model.inner_loops + 1)
        unit = "state_cell_recurrent_update"
    elif isinstance(model, SocialMessagePassingGNN):
        updates = depth * model.message_steps
        unit = "graph_message_passing_step"
    else:
        raise ValueError(f"unsupported model type: {type(model).__name__}")
    return {
        "trainable_parameters": trainable_parameters(model),
        "depth": depth,
        "block_evaluations": updates,
        "block_unit": unit,
    }

