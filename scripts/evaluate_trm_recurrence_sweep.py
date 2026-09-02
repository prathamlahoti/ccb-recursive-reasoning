"""Evaluate a fixed-data TRM checkpoint after each recurrent ACT step."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from ccb.dataset import OfficialEvaluationFirewall, SplitConfig, generate_firewalled_split
from ccb.encoding import collate_episodes
from ccb.fit_gate import load_fit_gate_config
from ccb.presets import domain_generator
from ccb.training import ExponentialMovingAverage, TrainConfig, build_model


def metrics(output, batch) -> dict[str, float]:
    predicted = output.logits.argmax(dim=-1)
    matches = predicted == batch.targets
    cell_mask = batch.step_mask.unsqueeze(-1).expand_as(batch.targets)
    transitions = matches.all(dim=-1)
    rows = torch.arange(predicted.shape[0], device=predicted.device)
    return {
        "element_accuracy": matches[cell_mask].float().mean().item(),
        "transition_exact_accuracy": transitions[batch.step_mask].float().mean().item(),
        "final_exact_accuracy": matches[rows, batch.depths - 1].all(dim=-1).float().mean().item(),
        "trace_exact_accuracy": (transitions | ~batch.step_mask).all(dim=-1).float().mean().item(),
    }


def train_config(config) -> TrainConfig:
    return TrainConfig(
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
        optimizer=config.optimizer,
        optimizer_betas=config.optimizer_betas,
        lr_warmup_steps=config.lr_warmup_steps,
        lr_min_ratio=config.lr_min_ratio,
        official_trm_forward_dtype=config.official_trm_forward_dtype,
    )


@torch.no_grad()
def sweep(model, batch, steps: int) -> list[dict[str, float]]:
    model.eval()
    carry = model.initial_carry(batch)
    records = []
    for step in range(1, steps + 1):
        carry, output, _ = model.act_step(carry, batch)
        records.append({"recurrence_step": step, **metrics(output, batch)})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_fit_gate_config(args.config)
    firewall = OfficialEvaluationFirewall.from_official_records(config.domain)
    episodes = generate_firewalled_split(
        domain_generator(config.domain),
        SplitConfig("fit_gate", (config.depth,), config.examples, config.data_seed),
        firewall,
    )
    batch = collate_episodes(episodes).to(args.device)
    model = build_model(train_config(config), batch.codec).to(args.device)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    result = {
        "schema": "ccb_trm_recurrence_sweep_v1",
        "config": asdict(config),
        "checkpoint_step": int(payload["step"]),
        "live": sweep(model, batch, config.trm_halt_max_steps),
        "ema": None,
    }
    if payload.get("ema_state") is not None:
        ema = ExponentialMovingAverage(model, config.ema_decay)
        ema.load_state_dict(payload["ema_state"])
        result["ema"] = sweep(ema.evaluation_model, batch, config.trm_halt_max_steps)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
