"""Run paired 64-example Gate C concurrently on two Kaggle T4 GPUs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

SOURCE_CORE_COMMIT = "6fbcf8453940e9d5b27833a00f5c4f8503ef468d"
WORKING = Path("/kaggle/working")
OUTPUT = WORKING / "d1-gate-c-t4-float32"
OUTPUT.mkdir(parents=True, exist_ok=True)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


project = WORKING / "ccb-recursive-reasoning"
if project.exists():
    shutil.rmtree(project)
archives = list(Path("/kaggle/input").rglob("ccb-recursive-reasoning.tar"))
if len(archives) == 1:
    project.mkdir(parents=True)
    shutil.unpack_archive(archives[0], project, format="tar")
else:
    source_trees = list(Path("/kaggle/input").rglob("pyproject.toml"))
    if len(source_trees) != 1:
        raise RuntimeError(f"expected one source archive/tree; archives={archives}, trees={source_trees}")
    shutil.copytree(source_trees[0].parent, project)

plans = {
    "trm": project / "configs" / "d1_official_trm_fit_64x5_t4_float32_v3.json",
    "transformer": project / "configs" / "d1_token_transformer_fit_64x5_t4_float32_v3.json",
}
for path in plans.values():
    if not path.is_file():
        raise FileNotFoundError(path)

atomic_json(OUTPUT / "run_started.json", {
    "schema": "ccb_gate_c_t4_float32_v3", "source_core_commit": SOURCE_CORE_COMMIT,
    "purpose": "T4 float32 64-example fit diagnostic; not released-precision evidence",
    "plans": {name: json.loads(path.read_text(encoding="utf-8")) for name, path in plans.items()},
    "started_unix": time.time(),
})

processes, logs = {}, {}
started = time.time()
try:
    for gpu, (name, config_path) in enumerate(plans.items()):
        log = (OUTPUT / f"{name}.log").open("w", encoding="utf-8", buffering=1)
        logs[name] = log
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PYTHONPATH"] = str(project / "src")
        processes[name] = subprocess.Popen(
            [sys.executable, "-m", "ccb", "fit-gate", "--config", str(config_path)],
            cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        print("STARTED", name, "GPU", gpu, flush=True)
    return_codes = {name: process.wait() for name, process in processes.items()}
    for log in logs.values():
        log.close()
    if any(code != 0 for code in return_codes.values()):
        raise RuntimeError(f"one or more workers failed: {return_codes}")
    results = {
        name: json.loads(Path(json.loads(path.read_text())["output_directory"], "result.json").read_text())
        for name, path in plans.items()
    }
    summary = {
        "schema": "ccb_gate_c_t4_float32_v3", "source_core_commit": SOURCE_CORE_COMMIT,
        "purpose": "T4 float32 64-example fit diagnostic; not released-precision evidence",
        "wall_seconds": time.time() - started, "results": results,
        "gate_passed": all(result["passed"] for result in results.values()),
    }
    atomic_json(OUTPUT / "final_summary.json", summary)
    print("GATE_C_T4_FLOAT32_DURABLY_SAVED", summary["gate_passed"],
          {name: result["gate_metrics"]["trace_exact_accuracy"] for name, result in results.items()},
          flush=True)
except Exception:
    for log in logs.values():
        if not log.closed:
            log.close()
    atomic_json(OUTPUT / "run_failed.json", {"source_core_commit": SOURCE_CORE_COMMIT,
                                               "traceback": traceback.format_exc()})
    raise
