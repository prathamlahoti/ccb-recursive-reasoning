"""Run the bounded 10K-update TRM-only Gate-B diagnostic."""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path


SOURCE_CORE_COMMIT = "6fbcf8453940e9d5b27833a00f5c4f8503ef468d"
WORKING = Path("/kaggle/working")
OUTPUT = WORKING / "d1-gate-b-trm-10k-t4-float32"
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
sys.path.insert(0, str(project / "src"))

from ccb.fit_gate import load_fit_gate_config, run_fit_gate  # noqa: E402

config_path = project / "configs" / "d1_official_trm_fit_16x5_t4_float32_10k_v3.json"
if not config_path.is_file():
    raise FileNotFoundError(config_path)
config = load_fit_gate_config(config_path)
atomic_json(
    OUTPUT / "run_started.json",
    {
        "schema": "ccb_gate_b_trm_10k_t4_float32_v3",
        "source_core_commit": SOURCE_CORE_COMMIT,
        "purpose": "T4 float32 extended fit diagnostic; not released-precision evidence",
        "config": json.loads(config_path.read_text(encoding="utf-8")),
        "started_unix": time.time(),
    },
)

started = time.time()
try:
    result = run_fit_gate(config)
    summary = {
        "schema": "ccb_gate_b_trm_10k_t4_float32_v3",
        "source_core_commit": SOURCE_CORE_COMMIT,
        "purpose": "T4 float32 extended fit diagnostic; not released-precision evidence",
        "wall_seconds": time.time() - started,
        "result": result,
    }
    atomic_json(OUTPUT / "final_summary.json", summary)
    print(
        "GATE_B_TRM_10K_T4_FLOAT32_DURABLY_SAVED",
        result["passed"],
        result["gate_weights"],
        result["gate_metrics"]["trace_exact_accuracy"],
        flush=True,
    )
except Exception:
    atomic_json(
        OUTPUT / "run_failed.json",
        {"source_core_commit": SOURCE_CORE_COMMIT, "traceback": traceback.format_exc()},
    )
    raise
