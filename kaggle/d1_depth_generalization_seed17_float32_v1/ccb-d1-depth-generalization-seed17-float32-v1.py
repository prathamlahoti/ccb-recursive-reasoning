"""Run the preregistered D1 seed-17 depth-generalization calibration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

SOURCE_COMMIT = "__SOURCE_COMMIT__"
WORKING = Path("/kaggle/working")
OUTPUT = WORKING / "d1-depth-generalization-v1"
OUTPUT.mkdir(parents=True, exist_ok=True)
PLANS = {
    "trm": "configs/d1_depth_generalization_trm_seed17_float32_v1.json",
    "transformer": "configs/d1_depth_generalization_transformer_seed17_float32_v1.json",
}


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


def latest_step(root: Path) -> int | None:
    ledgers = list(root.rglob("train.jsonl"))
    if not ledgers:
        return None
    lines = ledgers[0].read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        try:
            return int(json.loads(line)["step"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return None


project = prepare_project()
environment_base = os.environ.copy()
environment_base["PYTHONPATH"] = str(project / "src")

verification = subprocess.run(
    [sys.executable, "-m", "ccb", "verify-official"],
    cwd=project,
    env=environment_base,
    check=True,
    capture_output=True,
    text=True,
)
atomic_json(OUTPUT / "official_compatibility.json", json.loads(verification.stdout))

resolved_plans = {name: project / relative for name, relative in PLANS.items()}
for config_path in resolved_plans.values():
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

atomic_json(OUTPUT / "run_started.json", {
    "schema": "ccb_d1_depth_generalization_seed17_float32_v1",
    "source_commit": SOURCE_COMMIT,
    "purpose": "single-seed generated-data calibration; official evaluation remains sealed",
    "plans": {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in resolved_plans.items()
    },
    "started_unix": time.time(),
})

processes: dict[str, subprocess.Popen[str]] = {}
logs = {}
started = time.time()
try:
    for gpu, (name, config_path) in enumerate(resolved_plans.items()):
        log = (OUTPUT / f"{name}.log").open("w", encoding="utf-8", buffering=1)
        logs[name] = log
        environment = environment_base.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        processes[name] = subprocess.Popen(
            [sys.executable, "-m", "ccb", "launch", "--config", str(config_path)],
            cwd=project,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print("STARTED", name, "GPU", gpu, flush=True)

    while any(process.poll() is None for process in processes.values()):
        time.sleep(60)
        progress = {
            name: latest_step(OUTPUT / name)
            for name in processes
        }
        print("PROGRESS", progress, flush=True)

    return_codes = {name: process.wait() for name, process in processes.items()}
    for log in logs.values():
        log.close()
    if any(code != 0 for code in return_codes.values()):
        raise RuntimeError(f"generalization worker failure: {return_codes}")

    results = {}
    for name in processes:
        matches = list((OUTPUT / name).glob("d1/*/seed-*/result.json"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one result for {name}, found {matches}")
        results[name] = json.loads(matches[0].read_text(encoding="utf-8"))
    summary = {
        "schema": "ccb_d1_depth_generalization_seed17_float32_v1",
        "source_commit": SOURCE_COMMIT,
        "purpose": "single-seed generated-data calibration; not a final multi-seed claim",
        "wall_seconds": time.time() - started,
        "models": {
            name: {
                "model": result["model"],
                "evaluation_weight_source": result["evaluation_weight_source"],
                "wall_seconds": result["wall_seconds"],
                "peak_cuda_memory_gib": result["peak_cuda_memory_gib"],
                "validation": result["evaluations"]["validation"],
                "test_depth": result["evaluations"]["test_depth"],
                "live_validation": result["live_evaluations"]["validation"],
                "live_test_depth": result["live_evaluations"]["test_depth"],
            }
            for name, result in results.items()
        },
    }
    atomic_json(OUTPUT / "final_summary.json", summary)
    print("D1_DEPTH_GENERALIZATION_DURABLY_SAVED", {
        name: result["evaluations"]["test_depth"]["overall"]["final_exact_accuracy"]
        for name, result in results.items()
    }, flush=True)
except Exception:
    for log in logs.values():
        if not log.closed:
            log.close()
    atomic_json(OUTPUT / "run_failed.json", {
        "source_commit": SOURCE_COMMIT,
        "traceback": traceback.format_exc(),
    })
    raise
