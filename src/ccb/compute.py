from __future__ import annotations

from typing import Any

from torch import nn

from ccb.models import DirectTransformer, PublishedTRMCCB


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def compute_signature(model: nn.Module, *, depth: int) -> dict[str, Any]:
    """Architecture-independent accounting unit for matched-compute tables."""

    if depth < 1:
        raise ValueError("depth must be positive")
    if isinstance(model, DirectTransformer):
        updates = len(model.encoder.layers)
        unit = "full_sequence_transformer_block"
    elif isinstance(model, PublishedTRMCCB):
        updates = len(model.l_level) * model.h_cycles * (model.l_cycles + 1)
        unit = "shared_trm_reasoning_block"
    else:
        raise ValueError(f"unsupported model type: {type(model).__name__}")
    return {
        "trainable_parameters": trainable_parameters(model),
        "depth": depth,
        "block_evaluations": updates,
        "block_unit": unit,
    }
