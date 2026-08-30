"""Private, server-side D1 fit gate for the verified official TRM CCB adapter."""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path


SOURCE_COMMIT = "f64fce4b"
WORKING = Path("/kaggle/working")
OUTPUT = WORKING / "d1-official-trm-fit-gate-v1"
OUTPUT.mkdir(parents=True, exist_ok=True)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


project = WORKING / "ccb-recursive-reasoning"
if project.exists():
    shutil.rmtree(project)
source_trees = list(Path("/kaggle/input").rglob("pyproject.toml"))
source_archives = list(Path("/kaggle/input").rglob("ccb-recursive-reasoning.tar"))
if len(source_trees) == 1:
    shutil.copytree(source_trees[0].parent, project)
elif len(source_archives) == 1:
    project.mkdir(parents=True)
    shutil.unpack_archive(source_archives[0], project, format="tar")
else:
    raise RuntimeError(
        "expected exactly one CCB source tree or source archive; "
        f"trees={source_trees}, archives={source_archives}"
    )
sys.path.insert(0, str(project / "src"))

from ccb.fit_gate import load_fit_gate_config, run_fit_gate  # noqa: E402
from ccb.official import verify_official_records  # noqa: E402

config_path = project / "configs" / "d1_official_trm_fit_gate_v1.json"
config = load_fit_gate_config(config_path)
atomic_json(
    OUTPUT / "run_started.json",
    {
        "schema": "ccb_kaggle_fit_gate_v1",
        "source_commit": SOURCE_COMMIT,
        "config": json.loads(config_path.read_text(encoding="utf-8")),
        "official_d1_check": verify_official_records("d1"),
    },
)

try:
    result = run_fit_gate(config)
    atomic_json(OUTPUT / "completed_result.json", result)
    atomic_json(OUTPUT / "run_complete.json", {"source_commit": SOURCE_COMMIT, "status": "complete"})
    print("OFFICIAL_TRM_FIT_GATE_DURABLY_SAVED", OUTPUT / "completed_result.json")
except Exception:
    atomic_json(
        OUTPUT / "run_failed.json",
        {"source_commit": SOURCE_COMMIT, "status": "failed", "traceback": traceback.format_exc()},
    )
    raise
