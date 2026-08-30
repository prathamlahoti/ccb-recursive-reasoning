"""Small fixed-data fit gates for validating a training path before benchmarking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import repeat
from pathlib import Path
from typing import Any, Mapping

import torch

from ccb.dataset import OfficialEvaluationFirewall, SplitConfig, generate_firewalled_split
from ccb.encoding import collate_episodes
from ccb.presets import domain_generator
from ccb.results import write_result
from ccb.training import (
    ExponentialMovingAverage,
    TrainConfig,
    build_model,
    evaluate_batch,
    jsonl_logger,
    save_checkpoint,
    seed_everything,
    train_batches,
    train_trm_act_batches,
)


@dataclass(frozen=True)
class FitGateConfig:
    domain: str
    model: str
    output_directory: str
    examples: int = 64
    depth: int = 5
    seed: int = 17
    data_seed: int = 90_000_000
    width: int = 32
    layers_or_loops: int = 1
    steps: int = 2_000
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    checkpoint_interval_steps: int = 100
    device: str = "cpu"
    ema_decay: float = 0.999
    trm_h_cycles: int = 3
    trm_l_cycles: int = 6
    trm_max_depth: int = 5
    trm_halt_max_steps: int = 3
    trm_halt_exploration_prob: float = 0.1

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FitGateConfig":
        return cls(**dict(payload))


def load_fit_gate_config(path: Path) -> FitGateConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fit-gate config must be a JSON object")
    return FitGateConfig.from_mapping(payload)


def run_fit_gate(config: FitGateConfig) -> dict[str, Any]:
    """Fit one fixed, firewalled CCB set and record live/EMA training metrics.

    This is deliberately an in-distribution memorisation test.  It makes no
    claim about held-out depth generalisation; its only question is whether the
    specified training path can learn an unchanging small set.
    """

    if config.model not in {"transformer", "trm_upstream_core"}:
        raise ValueError("fit gate supports transformer or trm_upstream_core")
    if min(config.examples, config.depth, config.steps, config.width) < 1:
        raise ValueError("examples, depth, steps, and width must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.model == "trm_upstream_core" and config.trm_max_depth < config.depth:
        raise ValueError("trm_max_depth must be at least the fit-gate depth")

    output = Path(config.output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    seed_everything(config.seed)
    firewall = OfficialEvaluationFirewall.from_official_records(config.domain)
    episodes = generate_firewalled_split(
        domain_generator(config.domain),
        SplitConfig("fit_gate", (config.depth,), config.examples, config.data_seed),
        firewall,
    )
    if len(episodes) != config.examples:
        raise RuntimeError("fit-gate generator did not return the requested example count")
    batch = collate_episodes(episodes).to(config.device)
    train_config = TrainConfig(
        model=config.model,
        width=config.width,
        layers_or_loops=config.layers_or_loops,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        steps=config.steps,
        seed=config.seed,
        ema_decay=config.ema_decay,
        trm_h_cycles=config.trm_h_cycles,
        trm_l_cycles=config.trm_l_cycles,
        trm_max_depth=config.trm_max_depth,
        trm_halt_max_steps=config.trm_halt_max_steps,
        trm_halt_exploration_prob=config.trm_halt_exploration_prob,
    )
    model = build_model(train_config, batch.codec).to(config.device)
    initial_live = evaluate_batch(model, batch)
    checkpoint = output / "checkpoint.pt"

    if config.model == "trm_upstream_core":
        ema = ExponentialMovingAverage(model, config.ema_decay)

        def checkpoint_if_due(step, current_model, optimizer, current_ema, carry, pending) -> None:
            if config.checkpoint_interval_steps and step % config.checkpoint_interval_steps == 0:
                save_checkpoint(
                    checkpoint,
                    model=current_model,
                    optimizer=optimizer,
                    config=train_config,
                    codec=batch.codec,
                    step=step,
                    ema=current_ema,
                    act_carry=(carry, pending),
                )

        optimizer, ema, _, history = train_trm_act_batches(
            model,
            repeat(batch),
            train_config,
            device=config.device,
            log_callback=jsonl_logger(output / "train.jsonl"),
            ema=ema,
            checkpoint_callback=checkpoint_if_due,
        )
        final_live = evaluate_batch(model, batch)
        final_ema = evaluate_batch(ema.evaluation_model, batch)
    else:
        def checkpoint_if_due(step, current_model, optimizer) -> None:
            if config.checkpoint_interval_steps and step % config.checkpoint_interval_steps == 0:
                save_checkpoint(
                    checkpoint,
                    model=current_model,
                    optimizer=optimizer,
                    config=train_config,
                    codec=batch.codec,
                    step=step,
                )

        optimizer, history = train_batches(
            model,
            repeat(batch),
            train_config,
            device=config.device,
            log_callback=jsonl_logger(output / "train.jsonl"),
            checkpoint_callback=checkpoint_if_due,
        )
        final_live = evaluate_batch(model, batch)
        final_ema = None

    result = {
        "schema": "ccb_fit_gate_v1",
        "purpose": "fixed-data training-path diagnostic; not a benchmark result",
        "config": asdict(config),
        "firewall_safe": firewall.audit(episodes).safe,
        "initial_live": initial_live,
        "final_live": final_live,
        "final_ema": final_ema,
        "first_training_record": history[0],
        "last_training_record": history[-1],
    }
    write_result(output / "result.json", result)
    return result
