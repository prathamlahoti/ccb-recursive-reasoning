from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ccb.encoding import DomainCodec, TransitionBatch
from ccb.models import (
    DirectTransformer,
    FaithfulCCBTRM,
    FastSlowRecurrentModel,
    LoopedTransformer,
    RecurrentBaseline,
    SocialMessagePassingGNN,
    StateTransitionRecursiveModel,
    VanillaTRM,
)
from ccb.models.common import ModelOutput


@dataclass(frozen=True)
class TrainConfig:
    model: str
    width: int = 64
    layers_or_loops: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    steps: int = 200
    seed: int = 0
    loop_supervision_weight: float = 0.25
    supervision: str = "final"
    trm_latent_steps: int = 6
    trm_refinement_steps: int = 3
    trm_supervision_steps: int = 16
    ema_decay: float = 0.999


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def build_model(config: TrainConfig, codec: DomainCodec) -> nn.Module:
    width = config.width
    count = config.layers_or_loops
    if config.model == "transformer":
        heads = 4 if width % 4 == 0 else 1
        return DirectTransformer(codec, width=width, heads=heads, layers=count)
    if config.model in {"gru", "lstm"}:
        return RecurrentBaseline(codec, width=width, layers=count, cell=config.model)
    if config.model == "looped_transformer":
        heads = 4 if width % 4 == 0 else 1
        return LoopedTransformer(codec, width=width, heads=heads, loops=count)
    if config.model == "trm":
        return VanillaTRM(codec, width=width, loops=count)
    if config.model == "trm_faithful":
        return FaithfulCCBTRM(
            codec,
            width=width,
            latent_steps=config.trm_latent_steps,
            refinement_steps=config.trm_refinement_steps,
        )
    if config.model == "dis_trm":
        return VanillaTRM(
            codec, width=width, loops=count, detach_warmup=False
        )
    if config.model == "fast_slow":
        return FastSlowRecurrentModel(codec, width=width, fast_loops=count)
    if config.model == "strm":
        return StateTransitionRecursiveModel(codec, width=width, inner_loops=count)
    if config.model == "gnn":
        return SocialMessagePassingGNN(codec, width=width, message_steps=count)
    raise ValueError(f"unknown model: {config.model}")


def supervised_loss(
    output: ModelOutput,
    targets: Tensor,
    *,
    loop_supervision_weight: float,
    step_mask: Tensor | None = None,
    supervision: str = "final",
    initial_state: Tensor | None = None,
) -> Tensor:
    if supervision not in {"final", "deep", "dis"}:
        raise ValueError("supervision must be 'final', 'deep', or 'dis'")
    mask = (
        torch.ones(targets.shape[:2], dtype=torch.bool, device=targets.device)
        if step_mask is None
        else step_mask
    )

    def masked_ce(logits: Tensor, labels: Tensor) -> Tensor:
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
        ).reshape_as(labels)
        expanded_mask = mask.unsqueeze(-1).expand_as(labels)
        return losses[expanded_mask].mean()

    if supervision == "dis" and output.loop_logits:
        if initial_state is None:
            raise ValueError("DIS requires initial_state")
        improvement_targets = deep_improvement_targets(
            initial_state, targets, mask, len(output.loop_logits)
        )
        return torch.stack(
            [
                masked_ce(logits, loop_target)
                for logits, loop_target in zip(output.loop_logits, improvement_targets)
            ]
        ).mean()

    loss = masked_ce(output.logits, targets)
    if supervision == "deep" and output.loop_logits and loop_supervision_weight > 0:
        loop_loss = torch.stack(
            [masked_ce(logits, targets) for logits in output.loop_logits]
        ).mean()
        loss = loss + loop_supervision_weight * loop_loss
    return loss


def deep_improvement_targets(
    initial_state: Tensor, targets: Tensor, step_mask: Tensor, loops: int
) -> tuple[Tensor, ...]:
    """Construct DIS discrete-diffusion targets from input toward oracle trace.

    The original DIS method reveals a random subset of incorrect target cells at
    each loop. Here the input frame is the initial symbolic state repeated across
    transitions; the last loop is exactly the complete oracle transition trace.
    """

    if loops < 1:
        raise ValueError("loops must be positive")
    start = initial_state[:, None, :].expand_as(targets)
    valid = step_mask[:, :, None].expand_as(targets)
    differing = (start != targets) & valid
    batch_size, depth, state_size = targets.shape
    flat_size = depth * state_size
    random_scores = torch.rand((batch_size, flat_size), device=targets.device)
    random_scores = random_scores.masked_fill(~differing.reshape(batch_size, -1), float("inf"))
    permutation = random_scores.argsort(dim=1)
    rank = permutation.argsort(dim=1)
    differing_count = differing.reshape(batch_size, -1).sum(dim=1)
    frames: list[Tensor] = []
    for loop in range(loops):
        reveal_count = (
            differing_count
            if loops == 1
            else (loop * differing_count) // (loops - 1)
        )
        reveal = rank < reveal_count[:, None]
        frame = torch.where(
            reveal.reshape(batch_size, depth, state_size), targets, start
        )
        # Padded labels are irrelevant but kept within the vocabulary.
        frames.append(torch.where(valid, frame, targets))
    return tuple(frames)


@torch.no_grad()
def evaluate_batch(model: nn.Module, batch: TransitionBatch) -> dict[str, float]:
    model.eval()
    output: ModelOutput = model(batch)  # type: ignore[assignment]
    predicted = output.logits.argmax(dim=-1)
    cell_mask = batch.step_mask.unsqueeze(-1).expand_as(batch.targets)
    matches = predicted == batch.targets
    element_accuracy = matches[cell_mask].float().mean().item()
    transition_matches = matches.all(dim=-1)
    transition_exact = transition_matches[batch.step_mask].float().mean().item()
    rows = torch.arange(predicted.shape[0], device=predicted.device)
    final_indices = batch.depths - 1
    final_exact = matches[rows, final_indices].all(dim=-1).float().mean().item()
    trace_exact = (
        (transition_matches | ~batch.step_mask).all(dim=-1).float().mean().item()
    )
    return {
        "element_accuracy": element_accuracy,
        "transition_exact_accuracy": transition_exact,
        "final_exact_accuracy": final_exact,
        "trace_exact_accuracy": trace_exact,
    }


def train_fixed_batch(
    model: nn.Module,
    batch: TransitionBatch,
    config: TrainConfig,
    *,
    log_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    optimizer = build_optimizer(model, config)
    history: list[dict[str, Any]] = []
    for step in range(1, config.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output: ModelOutput = model(batch)  # type: ignore[assignment]
        loss = supervised_loss(
            output,
            batch.targets,
            loop_supervision_weight=config.loop_supervision_weight,
            step_mask=batch.step_mask,
            supervision=config.supervision,
            initial_state=batch.initial_state,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step == config.steps or step % max(1, config.steps // 10) == 0:
            record: dict[str, Any] = {"step": step, "loss": float(loss.detach())}
            record.update(evaluate_batch(model, batch))
            history.append(record)
            if log_callback is not None:
                log_callback(record)
    return optimizer, history


def train_batches(
    model: nn.Module,
    batches: Iterable[TransitionBatch],
    config: TrainConfig,
    *,
    device: torch.device | str = "cpu",
    log_callback: Callable[[Mapping[str, Any]], None] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    start_step: int = 0,
    checkpoint_callback: Callable[[int, nn.Module, torch.optim.Optimizer], None] | None = None,
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    """Train through ``config.steps``, optionally resuming a deterministic stream."""

    if start_step < 0 or start_step > config.steps:
        raise ValueError("start_step must be within [0, config.steps]")
    if optimizer is None:
        optimizer = build_optimizer(model, config)
    history: list[dict[str, Any]] = []
    model.to(device)
    for step, original_batch in enumerate(islice(batches, start_step, None), start=start_step + 1):
        if step > config.steps:
            break
        batch = original_batch.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output: ModelOutput = model(batch)  # type: ignore[assignment]
        loss = supervised_loss(
            output,
            batch.targets,
            loop_supervision_weight=config.loop_supervision_weight,
            step_mask=batch.step_mask,
            supervision=config.supervision,
            initial_state=batch.initial_state,
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm.detach()),
        }
        history.append(record)
        if log_callback is not None:
            log_callback(record)
        if checkpoint_callback is not None:
            checkpoint_callback(step, model, optimizer)
    if start_step + len(history) < config.steps:
        raise ValueError(
            f"batch stream ended after {start_step + len(history)} steps; expected {config.steps}"
        )
    return optimizer, history


class ExponentialMovingAverage:
    """Minimal EMA used by the CCB adaptation of published TRM training."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie strictly between zero and one")
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                parameter.copy_(self.shadow[name])


def train_faithful_trm_batches(
    model: FaithfulCCBTRM,
    batches: Iterable[TransitionBatch],
    config: TrainConfig,
    *,
    device: torch.device | str = "cpu",
    log_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    """TRM's detached deep-supervision training, counted by optimizer update."""

    if config.trm_supervision_steps < 1:
        raise ValueError("trm_supervision_steps must be positive")
    model.to(device)
    optimizer = build_optimizer(model, config)
    # Construct EMA after device placement; its shadow tensors must share the
    # model's device for in-place updates during GPU training.
    ema = ExponentialMovingAverage(model, config.ema_decay)
    history: list[dict[str, Any]] = []
    batch_iterator = iter(batches)
    update = 0
    while update < config.steps:
        batch = next(batch_iterator).to(device)
        answer, latent = model.initial_states(batch)
        for deep_step in range(1, config.trm_supervision_steps + 1):
            if update >= config.steps:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            answer, latent, output, halt_logits = model.refine(batch, answer, latent)
            transition_matches = (output.logits.argmax(dim=-1) == batch.targets).all(dim=-1)
            answer_target = (transition_matches | ~batch.step_mask).all(dim=1)
            halt_loss = F.binary_cross_entropy_with_logits(
                halt_logits, answer_target.to(dtype=halt_logits.dtype)
            )
            loss = supervised_loss(
                output,
                batch.targets,
                loop_supervision_weight=0.0,
                step_mask=batch.step_mask,
                supervision="final",
            ) + 0.5 * halt_loss
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            update += 1
            record = {
                "step": update,
                "loss": float(loss.detach()),
                "halt_loss": float(halt_loss.detach()),
                "deep_supervision_step": deep_step,
                "gradient_norm": float(gradient_norm.detach()),
            }
            history.append(record)
            if log_callback is not None:
                log_callback(record)
    # Evaluation and final checkpoint use the stabilizing EMA weights.
    ema.copy_to(model)
    return optimizer, history


def jsonl_logger(path: Path) -> Callable[[Mapping[str, Any]], None]:
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(record: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(dict(record), sort_keys=True) + "\n")

    return log


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    codec: DomainCodec,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "ccb_checkpoint_v1",
            "step": step,
            "config": asdict(config),
            "codec": asdict(codec),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "python_random_state": random.getstate(),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
) -> int:
    """Restore model, optimizer, and RNG state; return the completed step."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "ccb_checkpoint_v1":
        raise ValueError(f"unsupported checkpoint schema in {path}")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_states") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
    if payload.get("python_random_state") is not None:
        random.setstate(payload["python_random_state"])
    return int(payload["step"])
