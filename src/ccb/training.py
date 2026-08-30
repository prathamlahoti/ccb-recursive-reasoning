from __future__ import annotations

import json
import os
import random
import copy
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ccb.encoding import DomainCodec, TransitionBatch
from ccb.models import (
    CCBTokenTransformer,
    DirectTransformer,
    OfficialTRMCCBAdapter,
    PublishedTRMCCB,
)
from ccb.models.common import ModelOutput
from ccb.optim import AdamATan2, official_trm_learning_rate


def stablemax_log_probs(logits: Tensor) -> Tensor:
    """Released TRM stablemax transform, evaluated in float64 for stability."""

    values = logits.to(torch.float64)
    transformed = torch.where(values < 0, 1.0 / (1.0 - values + 1e-30), values + 1.0)
    return torch.log(transformed / transformed.sum(dim=-1, keepdim=True))


def trm_sequence_loss(
    output: ModelOutput, q_halt: Tensor, batch: TransitionBatch
) -> tuple[Tensor, Tensor, Tensor]:
    """TRM token loss normalized per example plus the released halt BCE term."""

    labels = batch.targets.reshape(batch.targets.shape[0], -1)
    valid = batch.step_mask[:, :, None].expand_as(batch.targets).reshape_as(labels)
    logits = output.logits.reshape(labels.shape[0], labels.shape[1], -1)
    log_probs = stablemax_log_probs(logits)
    token_loss = -torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    lm_loss = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    with torch.no_grad():
        correct = ((logits.argmax(dim=-1) == labels) | ~valid).all(dim=1)
    halt_loss = F.binary_cross_entropy_with_logits(q_halt, correct.to(q_halt.dtype), reduction="none")
    return (lm_loss + 0.5 * halt_loss).mean(), lm_loss.mean(), halt_loss.mean()


@dataclass(frozen=True)
class TrainConfig:
    model: str
    width: int = 64
    layers_or_loops: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    steps: int = 200
    seed: int = 0
    loop_supervision_weight: float = 0.0
    supervision: str = "final"
    ema_decay: float = 0.999
    trm_h_cycles: int = 3
    trm_l_cycles: int = 6
    trm_max_depth: int = 100
    trm_halt_max_steps: int = 4
    trm_halt_exploration_prob: float = 0.1
    optimizer: str = "adamw"
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    lr_warmup_steps: int = 0
    lr_min_ratio: float = 0.0
    official_trm_forward_dtype: str = "float32"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
            betas=config.optimizer_betas,
        )
    if config.optimizer == "adam_atan2":
        return AdamATan2(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
            betas=config.optimizer_betas,
        )
    raise ValueError(f"unknown optimizer: {config.optimizer}")


def apply_training_schedule(optimizer: torch.optim.Optimizer, config: TrainConfig, step: int) -> None:
    """Apply the released TRM scheduler only when explicitly requested."""

    if config.optimizer != "adam_atan2":
        return
    learning_rate = official_trm_learning_rate(
        step - 1,
        base_learning_rate=config.learning_rate,
        warmup_steps=config.lr_warmup_steps,
        total_steps=config.steps,
        min_ratio=config.lr_min_ratio,
    )
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def build_model(config: TrainConfig, codec: DomainCodec) -> nn.Module:
    width = config.width
    count = config.layers_or_loops
    if config.model == "transformer":
        heads = 4 if width % 4 == 0 else 1
        return DirectTransformer(codec, width=width, heads=heads, layers=count)
    if config.model == "ccb_token_transformer":
        if config.official_trm_forward_dtype == "float32":
            forward_dtype = torch.float32
        elif config.official_trm_forward_dtype == "bfloat16":
            forward_dtype = torch.bfloat16
        else:
            raise ValueError("official_trm_forward_dtype must be float32 or bfloat16")
        heads = 8 if width % 8 == 0 else (4 if width % 4 == 0 else 1)
        return CCBTokenTransformer(
            codec,
            max_depth=config.trm_max_depth,
            width=width,
            heads=heads,
            layers=count,
            forward_dtype=forward_dtype,
        )
    if config.model == "trm_upstream_core":
        heads = 4 if width % 4 == 0 else 1
        return PublishedTRMCCB(
            codec,
            width=width,
            heads=heads,
            layers=count,
            h_cycles=config.trm_h_cycles,
            l_cycles=config.trm_l_cycles,
            max_depth=config.trm_max_depth,
            halt_max_steps=config.trm_halt_max_steps,
            halt_exploration_prob=config.trm_halt_exploration_prob,
        )
    if config.model == "official_trm_ccb":
        if config.official_trm_forward_dtype == "float32":
            forward_dtype = torch.float32
        elif config.official_trm_forward_dtype == "bfloat16":
            forward_dtype = torch.bfloat16
        else:
            raise ValueError("official_trm_forward_dtype must be float32 or bfloat16")
        heads = 8 if width % 8 == 0 else (4 if width % 4 == 0 else 1)
        return OfficialTRMCCBAdapter(
            codec,
            hidden_size=width,
            num_heads=heads,
            l_layers=count,
            h_cycles=config.trm_h_cycles,
            l_cycles=config.trm_l_cycles,
            max_depth=config.trm_max_depth,
            halt_max_steps=config.trm_halt_max_steps,
            halt_exploration_prob=config.trm_halt_exploration_prob,
            forward_dtype=forward_dtype,
        )
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
        apply_training_schedule(optimizer, config, step)
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
        apply_training_schedule(optimizer, config, step)
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
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
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


def train_trm_act_batches(
    model: PublishedTRMCCB | OfficialTRMCCBAdapter,
    batches: Iterable[TransitionBatch],
    config: TrainConfig,
    *,
    device: torch.device | str = "cpu",
    log_callback: Callable[[Mapping[str, Any]], None] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    ema: "ExponentialMovingAverage" | None = None,
    act_state: tuple[Any, TransitionBatch] | None = None,
    start_step: int = 0,
    checkpoint_callback: Callable[
        [int, nn.Module, torch.optim.Optimizer, "ExponentialMovingAverage", Any, TransitionBatch], None
    ]
    | None = None,
) -> tuple[torch.optim.Optimizer, "ExponentialMovingAverage", tuple[Any, TransitionBatch], list[dict[str, Any]]]:
    """Train the upstream-derived TRM through its ACT state machine.

    The incoming stream must remain depth-bucketed until every active row has
    halted. This is enforced by ``PublishedTRMCCB.act_step`` rather than
    silently resetting recursive state on incompatible sequences.
    """

    if start_step < 0 or start_step > config.steps:
        raise ValueError("start_step must be within [0, config.steps]")
    model.to(device)
    if optimizer is None:
        optimizer = build_optimizer(model, config)
    if ema is None:
        ema = ExponentialMovingAverage(model, config.ema_decay)
    iterator = islice(iter(batches), 0 if act_state is None else start_step + 1, None)
    if act_state is None:
        first = next(iterator).to(device)
        carry = model.initial_act_carry(first)
        pending = first
    else:
        carry, pending = act_state
    history: list[dict[str, Any]] = []
    for step in range(start_step + 1, config.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        apply_training_schedule(optimizer, config, step)
        carry, output, (q_halt, _) = model.act_step(carry, pending)
        loss, lm_loss, halt_loss = trm_sequence_loss(output, q_halt, carry.current_batch)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ema.update(model)
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "lm_loss": float(lm_loss.detach()),
            "halt_loss": float(halt_loss.detach()),
            "gradient_norm": float(gradient_norm.detach()),
            "act_mean_steps": float(carry.steps.float().mean()),
            "act_halted_fraction": float(carry.halted.float().mean()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        if log_callback is not None:
            log_callback(record)
        pending = next(iterator).to(device)
        if checkpoint_callback is not None:
            checkpoint_callback(step, model, optimizer, ema, carry, pending)
    return optimizer, ema, (carry, pending), history


class ExponentialMovingAverage:
    """Copied EMA evaluation model; live optimisation weights are untouched."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie strictly between zero and one")
        self.decay = decay
        self.evaluation_model = copy.deepcopy(model).eval()
        for parameter in self.evaluation_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        live = model.state_dict()
        averaged = self.evaluation_model.state_dict()
        for name, value in averaged.items():
            source = live[name].detach()
            if torch.is_floating_point(value):
                value.lerp_(source, 1.0 - self.decay)
            else:
                value.copy_(source)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.evaluation_model.state_dict())

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "model_state": self.evaluation_model.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if float(state["decay"]) != self.decay:
            raise ValueError("EMA decay does not match checkpoint")
        self.evaluation_model.load_state_dict(state["model_state"])


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
    ema: ExponentialMovingAverage | None = None,
    act_carry: Any | None = None,
    dataset_manifest_hash: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "ccb_checkpoint_v2",
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
            "ema_state": None if ema is None else ema.state_dict(),
            "act_carry": act_carry,
            "dataset_manifest_hash": dataset_manifest_hash,
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
    ema: ExponentialMovingAverage | None = None,
    return_payload: bool = False,
) -> int:
    """Restore model, optimizer, and RNG state; return the completed step."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") not in {"ccb_checkpoint_v1", "ccb_checkpoint_v2"}:
        raise ValueError(f"unsupported checkpoint schema in {path}")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_states") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
    if payload.get("python_random_state") is not None:
        random.setstate(payload["python_random_state"])
    if ema is not None and payload.get("ema_state") is not None:
        ema.load_state_dict(payload["ema_state"])
    if return_payload:
        return int(payload["step"]), payload  # type: ignore[return-value]
    return int(payload["step"])
