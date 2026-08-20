# Implementation and Verification

## Implemented components

- Exact official D1-D3 operations, generators, solvers, formatting, and fixed
  record loader.
- Exhaustive 1,200-record compatibility verification.
- Deterministic learning and structural-holdout generators.
- Canonical JSON serialization, strict prediction parsing, manifests, and
  SHA-256 hashes.
- Oracle replay validation and shortcut/leakage audits.
- Metrics and confidence intervals specified in `BENCHMARK.md`.
- Structured codecs and batches for D1-D3.
- An immutable official-test fingerprint firewall covering seed/depth,
  canonical programs, and full instances.
- Variable-depth right padding, boolean transition masks, original depth
  tensors, and deterministic codec-safe data loaders.
- Full dataset evaluation aggregated overall and per depth, including recursive
  loop trajectories and deterministic confidence intervals.
- A resumable model-by-seed experiment matrix runner with resolved configs,
  environment capture, manifests, logs, checkpoints, and result JSON.
- Direct Transformer, GRU, LSTM, looped Transformer, CCB-adapted vanilla TRM,
  DIS-TRM, fast-slow recurrence, STRM, and a permutation-equivariant D3 GNN.
- Forward/backward tests, recursive-loop trajectory export, tiny-set overfit,
  structured decoding, parameter counting, block-evaluation accounting,
  JSONL logging, checkpoints, and result artifacts.

## Official compatibility implementation

`src/ccb/official.py` loads the fixed records and pins both upstream commit and
file hashes. Domain `solve` methods reproduce the exact upstream text format.
The exhaustive test is `tests/test_official_compatibility.py`.

Run:

```powershell
$env:PYTHONPATH='src'
python -m ccb verify-official
python -m unittest discover -s tests -v
```

## Model interface

An encoded batch contains initial state cells, padded operation IDs, the
complete target state after every operation, a boolean transition mask, and
each example's unpadded depth. Every model returns logits with shape
`[batch, depth, state_cells, state_vocabulary]`. Recursive models additionally
return a prediction after every inner loop. Padding occurs only after the real
program, and losses and metrics ignore it. Causal models cannot attend from a
real step to a later padded step.

Cross-cell communication is mandatory for structured recurrence: independent
cell updates cannot perform grid permutations or graph propagation. Sequence
models use deterministic sinusoidal step encodings rather than finite learned
position tables, avoiding an artificial maximum training depth.

## CCB-adapted TRM fidelity

The official TRM source was audited at commit
`c01103738605ba39d1430519b1ee0c62f4c707f8`. The local adaptation preserves:

- shared reasoning weights for high/answer and low/latent state updates;
- repeated low cycles inside each high cycle;
- no-gradient early high cycles;
- final-cycle gradient flow;
- independent prefix-wise prediction, preventing future-operation leakage.

Puzzle-identifier embeddings are intentionally removed. Q/ACT halting,
stablemax, EMA, upstream RMSNorm/SwiGLU kernels, and the distributed optimizer
recipe remain optional fidelity ablations and are not claimed as reproduced.

## Deep Improvement Supervision

The DIS source was audited at commit
`5de79e3376f27767d456d6b80df0973b73dfe2e8`. Its defining mechanism is a
discrete-diffusion target path: a random ordering of cells that differ between
the input and oracle is progressively replaced until the final loop's target
is the complete oracle. The CCB adaptation repeats the initial structured state
as the starting trace and contracts it toward the full oracle transition
trace. Every loop receives its corresponding intermediate target.

`trm` retains the original no-gradient warm-up schedule and final-target loss.
`dis_trm` enables gradient flow through all refinement loops and uses the DIS
target path. Separate names prevent a DIS result from being reported as
vanilla TRM. Fixed seeds make the random reveal order reproducible.

## Evaluation and experiment isolation

`ccb launch` builds the dataset once, writes a hashed manifest, expands the
model-by-seed matrix, and uses a stable run hash. Existing completed result
files are reused. Validation and both generated extrapolation tests run by
default. Official evaluation runs only when
`include_official_evaluation=true`; it is never used for training, stopping,
or automatic selection.

## Validation status

- Official compatibility: 1,200/1,200 exact.
- Python tests: 56 passing on native Windows.
- The same 56-test suite passes under Ubuntu 22.04 WSL2.
- All model families pass CPU forward/backward checks.
- All model families have passed a tiny exact-overfit debug gate; this verifies
  implementation capacity only and is not a scientific result.
- A native-Windows integration run built the definitive D1 suite, trained one
  step, evaluated validation/test-depth/test-strong, and wrote a manifest,
  checkpoint, training log, resolved configuration, and result artifact. The
  official evaluation flag remained false.

## Generated artifacts

`artifacts/`, `runs/`, `checkpoints/`, caches, and `tmp/` are disposable and
git-ignored. They can be recreated from commands in the README. The only data
stored as source are the pinned official fixed records under
`data/ccb_official`.

## Kaggle free-GPU execution runbook

The pinned source repository is private:
`https://github.com/prathamlahoti/ccb-recursive-reasoning` on branch `master`.
For Kaggle, the private Dataset route below is preferred because it avoids
placing a GitHub credential in the notebook. If cloning instead, use a
read-only fine-grained GitHub token stored as a Kaggle Secret; never paste a
token into notebook source or output.

To transfer as a private Kaggle Dataset:

1. Create a clean ZIP containing `.gitignore`, `README.md`, `pyproject.toml`,
   `src`, `tests`, `docs`, `paper`, and `data`. Do not include `.git`, `runs`,
   `tmp`, caches, or previous checkpoints.
2. On Kaggle, create a private Dataset from the ZIP.
3. Create a new Notebook, add that Dataset as an input, enable Internet, and
   select a GPU accelerator. A single P100 is preferable to T4 x2 because the
   current runner intentionally uses one GPU.
4. Locate and copy the read-only input project into `/kaggle/working`:

```python
from pathlib import Path
import shutil

matches = list(Path("/kaggle/input").rglob("pyproject.toml"))
assert len(matches) == 1, matches
source = matches[0].parent
project = Path("/kaggle/working/ccb-recursion")
if project.exists():
    shutil.rmtree(project)
shutil.copytree(source, project)
print(project)
```

5. Verify and install without replacing Kaggle's CUDA-enabled PyTorch:

```python
import subprocess, sys, torch

subprocess.run(["nvidia-smi"], check=True)
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0), torch.__version__)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", str(project), "--no-deps"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
    cwd=project,
    check=True,
)
sys.path.insert(0, str(project / "src"))
```

6. Use one model and one seed per first pilot. This limits loss if a free
   session terminates. For D1 Transformer seed 0:

```python
import json

config = {
    "domain": "d1",
    "models": ["transformer"],
    "seeds": [0],
    "output_directory": "/kaggle/working/runs/d1-transformer-seed0",
    "width": 64,
    "layers_or_loops": 4,
    "steps": 1000,
    "batch_size": 64,
    "learning_rate": 0.001,
    "bootstrap_resamples": 200,
    "include_official_evaluation": False,
    "device": "cuda"
}
config_path = Path("/kaggle/working/d1-transformer-seed0.json")
config_path.write_text(json.dumps(config, indent=2) + "\n")
```

7. Run in the notebook process so peak CUDA memory and wall time are captured:

```python
import json, time, torch
from ccb.experiment import load_experiment_config, run_experiment_matrix

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
started = time.perf_counter()
results = run_experiment_matrix(load_experiment_config(config_path))
torch.cuda.synchronize()
pilot_metrics = {
    "wall_seconds": time.perf_counter() - started,
    "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    "gpu": torch.cuda.get_device_name(0),
    "official_evaluation_used": results[0]["official_evaluation_used"],
}
assert not pilot_metrics["official_evaluation_used"]
Path("/kaggle/working/pilot_metrics.json").write_text(
    json.dumps(pilot_metrics, indent=2) + "\n"
)
print(pilot_metrics)
```

8. Archive the configuration, manifest, logs, checkpoint, result, summary, and
   timing record, then use Kaggle's **Save Version / Save & Run All** and
   download the output ZIP:

```python
import shutil

bundle = Path("/kaggle/working/d1-transformer-seed0")
bundle.mkdir(exist_ok=True)
shutil.copy2(config_path, bundle / config_path.name)
shutil.copy2("/kaggle/working/pilot_metrics.json", bundle / "pilot_metrics.json")
shutil.copytree(
    "/kaggle/working/runs/d1-transformer-seed0",
    bundle / "runs",
    dirs_exist_ok=True,
)
shutil.make_archive(str(bundle), "zip", bundle)
```

Before consuming a full session, run the notebook interactively with `steps: 2`
as a smoke test. Then change it to `steps: 1000` and use **Save & Run All** so
the full job runs once from a clean environment and its outputs persist; do not
run the 1,000-step job interactively and then rerun it during version saving.

## Recorded CUDA viability gate

On 2026-08-20, the private Kaggle notebook `CCB D1 Transformer GPU Pilot`
completed the bounded D1 direct-Transformer pilot on a Tesla T4.  The source
was the pinned project commit `54bd91d`; the notebook first verified all 1,200
official records exactly (400/400 in each of D1, D2, and D3).  Training used
one seed, width 64, four layers, batch size 64, and 1,000 steps.  It completed
in 25.125 seconds with a peak CUDA allocation of 0.110 GiB.  Official
evaluation was disabled (`official_evaluation_used: false`).

This is a technical viability result only: the CUDA path, dataset mount,
official-record firewall, and bounded training pipeline work end-to-end.  It
is not yet a scientific comparison or evidence that TRM/STRM improves over the
baseline.  The next evidence-producing jobs are the same budgeted pilot for
TRM, DIS-TRM, and STRM, followed by multi-seed repeats after a pilot comparison
shows a signal.

Repeat step 6 with `trm`, `dis_trm`, and `strm`, then repeat those four jobs for
D2. D3 should use `batch_size: 16` initially. Do not enable official evaluation
until model selection and hyperparameters are frozen. The current launcher
skips completed runs but checkpoints only after a model finishes; therefore a
first pilot should not combine several unmeasured models into one 12-hour job.
