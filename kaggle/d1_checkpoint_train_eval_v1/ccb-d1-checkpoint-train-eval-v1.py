"""Evaluate saved seed-17 live/EMA checkpoints on their exact train split."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

SOURCE_COMMIT = "25f2c3579d726d912e787242ec01997f65d94f81"
SOURCE_KERNEL = "prathamlahoti2/ccb-d1-depth-generalization-seed17-float32-v1"
SOURCE_KERNEL_VERSION = 2
WORKING = Path("/kaggle/working")
OUTPUT = WORKING / "d1-checkpoint-train-eval-v1"
OUTPUT.mkdir(parents=True, exist_ok=True)
MODELS = ("official_trm_ccb", "ccb_token_transformer")


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_project() -> Path:
    project = WORKING / "ccb-recursive-reasoning"
    if project.exists():
        shutil.rmtree(project)
    source_trees = list(Path("/kaggle/input").rglob("pyproject.toml"))
    if len(source_trees) != 1:
        raise RuntimeError(f"expected one source tree from the source Version output: {source_trees}")
    shutil.copytree(source_trees[0].parent, project)
    return project


def find_checkpoint(model_name: str) -> Path:
    matches = [
        path for path in Path("/kaggle/input").rglob("checkpoint.pt")
        if model_name in path.parts
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {model_name} checkpoint, found {matches}")
    return matches[0]


def worker(project: Path, model_name: str) -> None:
    sys.path.insert(0, str(project / "src"))
    import torch
    from ccb.dataset import instance_fingerprint
    from ccb.encoding import codec_for
    from ccb.evaluation import evaluate_model
    from ccb.loading import make_dataloader
    from ccb.presets import build_primary_splits
    from ccb.training import (
        ExponentialMovingAverage,
        TrainConfig,
        build_model,
    )

    device = torch.device("cuda")
    checkpoint = find_checkpoint(model_name)
    manifest_path = checkpoint.parents[3] / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_header = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_header["config"]["model"] != model_name:
        raise RuntimeError("checkpoint model identity mismatch")
    if checkpoint_header.get("dataset_manifest_hash") != manifest["manifest_hash"]:
        raise RuntimeError("checkpoint and saved manifest hash mismatch")

    splits, _ = build_primary_splits("d1")
    train = splits["train"]
    regenerated_hashes = [instance_fingerprint(episode) for episode in train]
    saved_hashes = [record["instance_hash"] for record in manifest["splits"]["train"]]
    if regenerated_hashes != saved_hashes:
        raise RuntimeError("regenerated training episodes do not match the saved manifest")

    config = TrainConfig(**checkpoint_header["config"])
    codec = codec_for(train[0])
    model = build_model(config, codec).to(device)
    ema = ExponentialMovingAverage(model, config.ema_decay)
    model.load_state_dict(checkpoint_header["model_state"])
    if checkpoint_header.get("ema_state") is None:
        raise RuntimeError("checkpoint has no EMA state")
    ema.load_state_dict(checkpoint_header["ema_state"])
    step = int(checkpoint_header["step"])
    if step != 10_000:
        raise RuntimeError(f"incomplete checkpoint: step={step}")

    started = time.perf_counter()
    live = evaluate_model(
        model, make_dataloader(train, batch_size=8, shuffle=False), device=device,
        bootstrap_resamples=2_000, bootstrap_seed=17,
    )
    live_seconds = time.perf_counter() - started
    print("LIVE_EVALUATED", model_name, live["overall"]["final_exact_accuracy"], flush=True)

    started = time.perf_counter()
    averaged = evaluate_model(
        ema.evaluation_model, make_dataloader(train, batch_size=8, shuffle=False),
        device=device, bootstrap_resamples=2_000, bootstrap_seed=17,
    )
    ema_seconds = time.perf_counter() - started
    result = {
        "schema": "ccb_d1_checkpoint_train_eval_v1",
        "purpose": "checkpoint-only train-distribution diagnostic; no optimization",
        "source_commit": SOURCE_COMMIT,
        "source_kernel": SOURCE_KERNEL,
        "source_kernel_version": SOURCE_KERNEL_VERSION,
        "model": model_name,
        "checkpoint_step": step,
        "dataset_manifest_hash": manifest["manifest_hash"],
        "training_examples": len(train),
        "official_evaluation_used": False,
        "live": live,
        "ema": averaged,
        "live_evaluation_seconds": live_seconds,
        "ema_evaluation_seconds": ema_seconds,
    }
    atomic_json(OUTPUT / f"{model_name}.json", result)
    print("EMA_EVALUATED", model_name, averaged["overall"]["final_exact_accuracy"], flush=True)


def controller() -> None:
    project = prepare_project()
    atomic_json(OUTPUT / "run_started.json", {
        "schema": "ccb_d1_checkpoint_train_eval_v1",
        "purpose": "checkpoint-only train-distribution diagnostic; no optimization",
        "source_commit": SOURCE_COMMIT,
        "source_kernel": SOURCE_KERNEL,
        "source_kernel_version": SOURCE_KERNEL_VERSION,
        "models": MODELS,
        "started_unix": time.time(),
    })
    processes = {}
    logs = {}
    try:
        for gpu, model_name in enumerate(MODELS):
            log = (OUTPUT / f"{model_name}.log").open("w", encoding="utf-8", buffering=1)
            logs[model_name] = log
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            processes[model_name] = subprocess.Popen(
                [sys.executable, __file__, "--worker", str(project), model_name],
                env=environment, stdout=log, stderr=subprocess.STDOUT, text=True,
            )
            print("STARTED", model_name, "GPU", gpu, flush=True)
        return_codes = {name: process.wait() for name, process in processes.items()}
        for log in logs.values():
            log.close()
        if any(code != 0 for code in return_codes.values()):
            raise RuntimeError(f"checkpoint-evaluation worker failure: {return_codes}")
        results = {
            name: json.loads((OUTPUT / f"{name}.json").read_text(encoding="utf-8"))
            for name in MODELS
        }
        atomic_json(OUTPUT / "final_summary.json", {
            "schema": "ccb_d1_checkpoint_train_eval_v1",
            "purpose": "checkpoint-only train-distribution diagnostic; no optimization",
            "source_commit": SOURCE_COMMIT,
            "source_kernel": SOURCE_KERNEL,
            "source_kernel_version": SOURCE_KERNEL_VERSION,
            "models": results,
        })
        print("TRAIN_CHECKPOINT_EVALUATION_DURABLY_SAVED", {
            name: {
                "live": result["live"]["overall"]["final_exact_accuracy"],
                "ema": result["ema"]["overall"]["final_exact_accuracy"],
            }
            for name, result in results.items()
        }, flush=True)
    except Exception:
        for log in logs.values():
            if not log.closed:
                log.close()
        atomic_json(OUTPUT / "run_failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]), sys.argv[3])
    else:
        controller()
