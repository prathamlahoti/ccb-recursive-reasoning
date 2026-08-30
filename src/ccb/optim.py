"""Reference, unfused implementations of optimizers used by released TRM."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor


class AdamATan2(torch.optim.Optimizer):
    """Mathematical PyTorch implementation of the released AdamATan2 update.

    The upstream project uses a fused CUDA extension.  This deliberately small
    implementation follows its published kernel update exactly, but is not a
    performance replacement for that extension.  It lets CCB use the same
    optimizer rule on CPU and in a self-contained Kaggle environment.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
    ) -> None:
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        if not all(0 <= beta < 1 for beta in betas):
            raise ValueError("betas must lie in [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[no-untyped-def]
        if closure is not None:
            with torch.enable_grad():
                closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            weight_decay = float(group["weight_decay"])
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("AdamATan2 does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = int(state["step"])
                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]

                # Matches adam_atan2.cu: decoupled multiplicative decay,
                # lerp moment updates, then bias-corrected atan2 direction.
                parameter.mul_(1.0 - lr * weight_decay)
                exp_avg.lerp_(gradient, 1.0 - beta1)
                exp_avg_sq.lerp_(gradient.square(), 1.0 - beta2)
                denominator = exp_avg_sq.sqrt().div_(math.sqrt(1.0 - beta2**step))
                parameter.add_(torch.atan2(exp_avg, denominator), alpha=-lr / (1.0 - beta1**step))
        return None


def official_trm_learning_rate(
    step: int,
    *,
    base_learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.0,
) -> float:
    """Released TRM linear-warmup / cosine-decay scalar schedule.

    ``step`` is zero-based, matching the scheduler position before an optimizer
    update.  This is the released ``cosine_schedule_with_warmup_lr_lambda``.
    """

    if step < 0 or warmup_steps < 0 or total_steps < 1 or not 0 <= min_ratio <= 1:
        raise ValueError("invalid official TRM learning-rate schedule arguments")
    if step < warmup_steps:
        return base_learning_rate * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_learning_rate * (
        min_ratio + max(0.0, (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
    )
