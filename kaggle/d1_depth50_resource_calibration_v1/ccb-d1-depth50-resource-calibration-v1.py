"""Measure depth-50 train/eval memory and time without optimizing a model."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORKING = Path("/kaggle/working")
OUTPUT = WORKING / "d1-depth50-resource-calibration-v1"
OUTPUT.mkdir(parents=True, exist_ok=True)
CANDIDATE_BATCH_SIZES = (1, 2, 4, 8)
# Version 1 durably saved the complete TRM calibration.  Version 2 is an
# intentionally bounded retry of only the failed Transformer worker.
CALIBRATION_MODELS = ("ccb_token_transformer",)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_project() -> Path:
    project = WORKING / "ccb-recursive-reasoning"
    if project.exists():
        shutil.rmtree(project)
    archives = list(Path("/kaggle/input").rglob("ccb-recursive-reasoning.tar"))
    if len(archives) == 1:
        project.mkdir(parents=True)
        shutil.unpack_archive(archives[0], project, format="tar")
        return project
    source_trees = list(Path("/kaggle/input").rglob("pyproject.toml"))
    if len(source_trees) != 1:
        raise RuntimeError(f"expected one source archive/tree; archives={archives}, trees={source_trees}")
    shutil.copytree(source_trees[0].parent, project)
    return project


def worker(project: Path, model_name: str) -> None:
    sys.path.insert(0, str(project / "src"))
    import torch
    from ccb.dataset import OfficialEvaluationFirewall, SplitConfig, generate_firewalled_split
    from ccb.encoding import collate_episodes
    from ccb.presets import domain_generator
    from ccb.training import (
        TrainConfig, build_model, evaluate_batch, supervised_loss, trm_sequence_loss,
    )

    device = torch.device("cuda")
    episodes = generate_firewalled_split(
        domain_generator("d1"), SplitConfig("resource_calibration", (50,), 8, 50_000_000),
        OfficialEvaluationFirewall.from_official_records("d1"),
    )
    reference = collate_episodes(episodes[:1])
    config = TrainConfig(
        model=model_name, width=512, layers_or_loops=2, steps=1,
        learning_rate=1e-4, weight_decay=0.1, optimizer="adam_atan2",
        optimizer_betas=(0.9, 0.95), lr_warmup_steps=2000, lr_min_ratio=1.0,
        trm_h_cycles=3, trm_l_cycles=6, trm_max_depth=50,
        trm_halt_max_steps=16, trm_halt_exploration_prob=0.1,
        official_trm_forward_dtype="float32",
    )
    model = build_model(config, reference.codec).to(device)
    records = []
    result_path = OUTPUT / f"{model_name}.json"
    for batch_size in CANDIDATE_BATCH_SIZES:
        batch = collate_episodes(episodes[:batch_size]).to(device)
        record = {"batch_size": batch_size}
        try:
            model.train()
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            if model_name == "official_trm_ccb":
                carry = model.initial_carry(batch)
                _, output, (q_halt, _) = model.act_step(carry, batch)
                loss, _, _ = trm_sequence_loss(output, q_halt, batch)
            else:
                output = model(batch)
                loss = supervised_loss(
                    output,
                    batch.targets,
                    loop_supervision_weight=0.0,
                    step_mask=batch.step_mask,
                )
            loss.backward()
            torch.cuda.synchronize(device)
            record["train_step_seconds"] = time.perf_counter() - started
            record["train_peak_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
            record["train_ok"] = True

            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            evaluation = evaluate_batch(model, batch)
            torch.cuda.synchronize(device)
            record["eval_seconds"] = time.perf_counter() - started
            record["eval_peak_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
            record["eval_ok"] = True
            record["evaluation_smoke"] = evaluation
        except torch.OutOfMemoryError:
            record.update(train_ok=False, eval_ok=False, error="cuda_out_of_memory")
            records.append(record)
            atomic_json(result_path, {"model": model_name, "records": records})
            break
        records.append(record)
        atomic_json(result_path, {
            "schema": "ccb_depth50_resource_calibration_v1", "model": model_name,
            "token_canvas_length": model.layout.sequence_length,
            "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "records": records,
        })
        print("CALIBRATED", model_name, batch_size, record["train_peak_gib"],
              record["eval_peak_gib"], flush=True)


def controller() -> None:
    project = prepare_project()
    atomic_json(OUTPUT / "run_started.json", {
        "schema": "ccb_depth50_resource_calibration_v1",
        "purpose": "resource calibration only; no optimization or research accuracy claim",
        "candidate_batch_sizes": CANDIDATE_BATCH_SIZES,
        "started_unix": time.time(),
    })
    processes, logs = {}, {}
    try:
        for gpu, model_name in enumerate(CALIBRATION_MODELS):
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
            raise RuntimeError(f"calibration worker failure: {return_codes}")
        summary = {
            "schema": "ccb_depth50_resource_calibration_v1",
            "purpose": "resource calibration only; no optimization or research accuracy claim",
            "models": {
                name: json.loads((OUTPUT / f"{name}.json").read_text(encoding="utf-8"))
                for name in processes
            },
        }
        atomic_json(OUTPUT / "final_summary.json", summary)
        print("DEPTH50_RESOURCE_CALIBRATION_DURABLY_SAVED", flush=True)
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
