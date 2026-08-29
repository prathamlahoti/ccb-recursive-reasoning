from __future__ import annotations

import json
import platform
import statistics
import sys
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Mapping

import torch

from ccb.compute import trainable_parameters
from ccb.audits import dataset_shortcut_audit
from ccb.dataset import build_manifest, rehash_manifest, write_manifest
from ccb.encoding import TransitionBatch, codec_for, collate_episodes
from ccb.evaluation import evaluate_model
from ccb.loading import make_dataloader
from ccb.official import OFFICIAL_COMMIT, load_official_episodes
from ccb.presets import build_primary_splits, primary_config
from ccb.results import write_result
from ccb.serialization import stable_hash
from ccb.training import (
    TrainConfig,
    build_optimizer,
    build_model,
    ExponentialMovingAverage,
    jsonl_logger,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    train_batches,
    train_trm_act_batches,
)


@dataclass(frozen=True)
class ExperimentConfig:
    domain: str
    models: tuple[str, ...]
    seeds: tuple[int, ...] = (0, 1, 2)
    output_directory: str = "runs"
    width: int = 64
    layers_or_loops: int = 4
    steps: int = 1_000
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    loop_supervision_weight: float = 0.0
    include_official_evaluation: bool = False
    bootstrap_resamples: int = 2_000
    device: str = "cpu"
    checkpoint_interval_steps: int = 0
    ema_decay: float = 0.999
    trm_h_cycles: int = 3
    trm_l_cycles: int = 6
    trm_max_depth: int = 100
    trm_halt_max_steps: int = 4
    trm_halt_exploration_prob: float = 0.1

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExperimentConfig":
        normalized = dict(payload)
        normalized["models"] = tuple(normalized["models"])
        normalized["seeds"] = tuple(normalized.get("seeds", (0, 1, 2)))
        return cls(**normalized)


def load_experiment_config(path: Path) -> ExperimentConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON object")
    return ExperimentConfig.from_mapping(payload)


def _batch_stream(episodes: tuple[Any, ...], config: ExperimentConfig, seed: int):
    epochs = (
        make_dataloader(
            episodes,
            batch_size=config.batch_size,
            shuffle=True,
            seed=seed + epoch,
        )
        for epoch in range(config.steps)
    )
    return chain.from_iterable(epochs)


def _act_batch_stream(episodes: tuple[Any, ...], config: ExperimentConfig, seed: int):
    """Provide ACT with a perpetually fixed-shape stream of fresh episodes.

    ACT carries state per row across optimiser steps.  Consequently, it must
    never encounter a final short batch or a depth-dependent tensor shape.  We
    sample complete batches with replacement between epochs and pad every trace
    to the largest *training* depth.  Individual valid lengths remain in
    ``step_mask`` and are the only positions included by attention/loss.
    """

    if len(episodes) < config.batch_size:
        raise ValueError("ACT training requires at least one full batch of episodes")
    fixed_depth = max(episode.depth for episode in episodes)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(episodes), generator=generator).tolist()
    cursor = 0

    def padded(indices: list[int]) -> TransitionBatch:
        batch = collate_episodes([episodes[index] for index in indices])
        padding = fixed_depth - batch.operations.shape[1]
        if padding == 0:
            return batch
        return TransitionBatch(
            batch.domain,
            batch.initial_state,
            torch.nn.functional.pad(batch.operations, (0, padding)),
            torch.nn.functional.pad(batch.targets, (0, 0, 0, padding)),
            batch.codec,
            torch.nn.functional.pad(batch.step_mask, (0, padding)),
            batch.depths,
        )

    while True:
        if cursor + config.batch_size > len(order):
            order = torch.randperm(len(episodes), generator=generator).tolist()
            cursor = 0
        indices = order[cursor : cursor + config.batch_size]
        cursor += config.batch_size
        yield padded(indices)


def _validate_trm_evaluation_depth(
    config: ExperimentConfig, splits: Mapping[str, tuple[Any, ...]]
) -> None:
    """Reject a TRM configuration that cannot encode every requested split."""

    if "trm_upstream_core" not in config.models:
        return
    required_depth = max(
        episode.depth
        for split_name in ("validation", "test_depth", "test_strong")
        for episode in splits[split_name]
    )
    if config.trm_max_depth < required_depth:
        raise ValueError(
            "trm_max_depth must cover every evaluation trace: "
            f"configured {config.trm_max_depth}, required {required_depth}"
        )


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def _mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "runs": len(values),
    }


def aggregate_experiment_results(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate headline and per-depth measures across training seeds."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result["model"]), []).append(result)
    models: dict[str, Any] = {}
    for model_name, model_results in sorted(grouped.items()):
        split_names = sorted(
            set.intersection(
                *(set(item["evaluations"]) for item in model_results)
            )
        )
        split_summaries: dict[str, Any] = {}
        for split_name in split_names:
            evaluations = [item["evaluations"][split_name] for item in model_results]
            metrics = {
                name: _mean_std([float(item[name]) for item in evaluations])
                for name in (
                    "per_step_retention_p_d",
                    "success_horizon_50",
                    "success_horizon_90",
                    "normalized_accuracy_depth_auc",
                )
            }
            for name in (
                "final_exact_accuracy",
                "trace_exact_accuracy",
                "transition_exact_accuracy",
                "element_accuracy",
                "valid_state_rate",
                "tfbc_rate_all",
                "tfbc_rate_given_final_correct",
            ):
                metrics[name] = _mean_std(
                    [float(item["overall"][name]) for item in evaluations]
                )
            shared_depths = sorted(
                set.intersection(*(set(item["per_depth"]) for item in evaluations)),
                key=int,
            )
            per_depth = {
                depth: _mean_std(
                    [
                        float(item["per_depth"][depth]["final_exact_accuracy"])
                        for item in evaluations
                    ]
                )
                for depth in shared_depths
            }
            split_summaries[split_name] = {
                "metrics": metrics,
                "final_exact_accuracy_by_depth": per_depth,
            }
        models[model_name] = {
            "seeds": sorted(int(item["seed"]) for item in model_results),
            "splits": split_summaries,
        }
    return {"schema": "ccb_experiment_summary_v1", "models": models}


def run_experiment_matrix(
    config: ExperimentConfig, *, dry_run: bool = False
) -> list[dict[str, Any]]:
    """Run or enumerate a deterministic model × seed experiment matrix."""

    if config.domain not in {"d1", "d2", "d3"}:
        raise ValueError("domain must be d1, d2, or d3")
    if not config.models or not config.seeds:
        raise ValueError("models and seeds cannot be empty")
    plans = [
        {"domain": config.domain, "model": model, "seed": seed}
        for model in config.models
        for seed in config.seeds
    ]
    if dry_run:
        return plans

    splits, firewall = build_primary_splits(config.domain)
    _validate_trm_evaluation_depth(config, splits)
    output_root = Path(config.output_directory).resolve()
    manifest = build_manifest(
        splits,
        config=primary_config(config.domain),
        official_firewall=firewall,
    )
    manifest["shortcut_audits"] = {
        name: dataset_shortcut_audit(episodes) for name, episodes in splits.items()
    }
    manifest = rehash_manifest(manifest)
    write_manifest(output_root / "dataset_manifest.json", manifest)
    completed: list[dict[str, Any]] = []
    codec = codec_for(splits["train"][0])

    for plan in plans:
        model_name = str(plan["model"])
        seed = int(plan["seed"])
        train_config = TrainConfig(
            model=model_name,
            width=config.width,
            layers_or_loops=config.layers_or_loops,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            steps=config.steps,
            seed=seed,
            loop_supervision_weight=config.loop_supervision_weight,
            ema_decay=config.ema_decay,
            trm_h_cycles=config.trm_h_cycles,
            trm_l_cycles=config.trm_l_cycles,
            trm_max_depth=config.trm_max_depth,
            trm_halt_max_steps=config.trm_halt_max_steps,
            trm_halt_exploration_prob=config.trm_halt_exploration_prob,
        )
        run_identity = {
            "experiment": asdict(config),
            "training": asdict(train_config),
            "dataset_manifest_hash": manifest["manifest_hash"],
        }
        run_hash = stable_hash(run_identity)[:12]
        run_directory = output_root / config.domain / model_name / f"seed-{seed}-{run_hash}"
        result_path = run_directory / "result.json"
        if result_path.exists():
            completed.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue

        run_directory.mkdir(parents=True, exist_ok=True)
        (run_directory / "resolved_config.json").write_text(
            json.dumps(run_identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        seed_everything(seed)
        model = build_model(train_config, codec)
        checkpoint_path = run_directory / "checkpoint.pt"
        optimizer = build_optimizer(model, train_config)
        start_step = 0
        ema: ExponentialMovingAverage | None = None
        act_state = None
        if checkpoint_path.exists():
            if model_name == "trm_upstream_core":
                ema = ExponentialMovingAverage(model, config.ema_decay)
                start_step, checkpoint_payload = load_checkpoint(
                    checkpoint_path, model=model, optimizer=optimizer, device=config.device,
                    ema=ema, return_payload=True,
                )
                act_state = checkpoint_payload.get("act_carry")
            else:
                start_step = load_checkpoint(
                    checkpoint_path, model=model, optimizer=optimizer, device=config.device
                )

        def checkpoint_if_due(
            step: int,
            current_model: torch.nn.Module,
            current_optimizer: torch.optim.Optimizer,
        ) -> None:
            if (
                config.checkpoint_interval_steps > 0
                and step < config.steps
                and step % config.checkpoint_interval_steps == 0
            ):
                save_checkpoint(
                    checkpoint_path,
                    model=current_model,
                    optimizer=current_optimizer,
                    config=train_config,
                    codec=codec,
                    step=step,
                )

        if model_name == "trm_upstream_core":
            def act_checkpoint_if_due(step, current_model, current_optimizer, current_ema, carry, pending):
                if config.checkpoint_interval_steps and step % config.checkpoint_interval_steps == 0:
                    save_checkpoint(
                        checkpoint_path, model=current_model, optimizer=current_optimizer,
                        config=train_config, codec=codec, step=step, ema=current_ema,
                        act_carry=(carry, pending), dataset_manifest_hash=manifest["manifest_hash"],
                    )

            optimizer, ema, _, history = train_trm_act_batches(
                model, _act_batch_stream(splits["train"], config, seed), train_config,
                device=config.device, log_callback=jsonl_logger(run_directory / "train.jsonl"),
                optimizer=optimizer, ema=ema, act_state=act_state, start_step=start_step,
                checkpoint_callback=act_checkpoint_if_due,
            )
            evaluation_model = ema.evaluation_model
        else:
            optimizer, history = train_batches(
                model,
                _batch_stream(splits["train"], config, seed),
                train_config,
                device=config.device,
                log_callback=jsonl_logger(run_directory / "train.jsonl"),
                optimizer=optimizer,
                start_step=start_step,
                checkpoint_callback=checkpoint_if_due,
            )
            evaluation_model = model
        evaluations = {}
        for split_name in ("validation", "test_depth", "test_strong"):
            evaluations[split_name] = evaluate_model(
                evaluation_model,
                make_dataloader(
                    splits[split_name], batch_size=config.batch_size, shuffle=False
                ),
                device=config.device,
                bootstrap_resamples=config.bootstrap_resamples,
                bootstrap_seed=seed,
            )
        if config.include_official_evaluation:
            evaluations["official_test"] = evaluate_model(
                    evaluation_model,
                make_dataloader(
                    load_official_episodes(config.domain),
                    batch_size=config.batch_size,
                    shuffle=False,
                ),
                device=config.device,
                bootstrap_resamples=config.bootstrap_resamples,
                bootstrap_seed=seed,
            )
        result = {
            "run_hash": run_hash,
            "domain": config.domain,
            "model": model_name,
            "seed": seed,
            "parameters": trainable_parameters(model),
            "training_steps": config.steps,
            "last_training_loss": history[-1]["loss"],
            "evaluations": evaluations,
            "environment": _environment(),
            "official_data_commit": OFFICIAL_COMMIT,
            "official_evaluation_used": config.include_official_evaluation,
        }
        write_result(result_path, result)
        save_checkpoint(
            run_directory / "checkpoint.pt",
            model=model,
            optimizer=optimizer,
            config=train_config,
            codec=codec,
            step=config.steps,
        )
        completed.append(result)
    summary = aggregate_experiment_results(completed)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return completed
