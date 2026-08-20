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
from ccb.encoding import codec_for
from ccb.evaluation import evaluate_model
from ccb.loading import make_dataloader
from ccb.official import OFFICIAL_COMMIT, load_official_episodes
from ccb.presets import build_primary_splits, primary_config
from ccb.results import write_result
from ccb.serialization import stable_hash
from ccb.training import (
    TrainConfig,
    build_model,
    jsonl_logger,
    save_checkpoint,
    seed_everything,
    train_batches,
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
    loop_supervision_weight: float = 0.25
    include_official_evaluation: bool = False
    bootstrap_resamples: int = 2_000
    device: str = "cpu"

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
        supervision = "dis" if model_name == "dis_trm" else "final"
        train_config = TrainConfig(
            model=model_name,
            width=config.width,
            layers_or_loops=config.layers_or_loops,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            steps=config.steps,
            seed=seed,
            loop_supervision_weight=config.loop_supervision_weight,
            supervision=supervision,
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
        optimizer, history = train_batches(
            model,
            _batch_stream(splits["train"], config, seed),
            train_config,
            device=config.device,
            log_callback=jsonl_logger(run_directory / "train.jsonl"),
        )
        evaluations = {}
        for split_name in ("validation", "test_depth", "test_strong"):
            evaluations[split_name] = evaluate_model(
                model,
                make_dataloader(
                    splits[split_name], batch_size=config.batch_size, shuffle=False
                ),
                device=config.device,
                bootstrap_resamples=config.bootstrap_resamples,
                bootstrap_seed=seed,
            )
        if config.include_official_evaluation:
            evaluations["official_test"] = evaluate_model(
                model,
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
            "training_steps": len(history),
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
