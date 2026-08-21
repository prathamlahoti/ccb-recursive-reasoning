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
   select a GPU accelerator. Use a Tesla T4 in the current Kaggle image: its
   PyTorch 2.10.0+cu128 build cannot execute kernels on Kaggle's P100
   (`cudaErrorNoKernelImageForDevice`). The runner intentionally uses one GPU,
   even if Kaggle displays a T4 x2 allocation.
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

## Recorded D1 vanilla-TRM pilot

The matching D1 vanilla-TRM pilot completed on 2026-08-20 using the same seed,
width (64), loops (4), batch size (64), and 1,000-step budget. It used a Tesla
T4, contained 123,657 parameters, ran in 106.920 seconds, and peaked at
0.428 GiB CUDA allocation. The final training loss was 1.16457; the result
artifact is `ccb_result_v1`, run hash `99840ca8583d`, and did not use official
evaluation.

This pilot **does not demonstrate depth generalization**: validation final
exact accuracy was 8.4% (42/500), while generated test-depth (25--50) and
test-strong (60/80/100) final exact accuracy were both 0.0%. Its mean first
divergence was about 2.84 transitions and its extrapolation depth-AUC was 0.
That is a useful negative control at this tiny training budget, not a verdict
on TRM. It means we must not claim a SOTA result or advance to D2 based on it.
The appropriate next compute gate is a matched DIS-TRM and STRM D1 pilot,
followed only by a longer, validation-selected and multi-seed D1 experiment if
one produces a clear depth/horizon improvement.

## Scaled-run readiness gate

The launcher now writes atomic periodic checkpoints and can resume exactly from
them: model parameters, optimizer, Python RNG, CPU Torch RNG, and CUDA RNG are
restored; the deterministic batch stream skips precisely the already-completed
updates. A CPU test confirms that an interrupted three-step run resumed to six
steps produces bit-identical parameters to an uninterrupted six-step run.

`configs/d1-scale-finding-seed0.json` is the next approved-scale configuration:
the four matched D1 model families (Transformer, vanilla TRM, DIS-TRM, STRM),
one prespecified seed, width 64, four layers/loops, 10,000 updates, and an
atomic checkpoint every 500 updates. It remains a **scale-finding** experiment,
not a final paper run: test-depth and test-strong are reported for all models,
not used to tune one after observing the results. A clear result requires a
model to improve both in-distribution validation and held-out depth/horizon
measures over the matched Transformer and vanilla TRM. Only then should that
model receive a longer three-seed D1 run.

## D1 three-seed confirmation plan

`configs/d1-confirmation-3seeds.json` fixes the next comparison before it is
run: the direct Transformer, vanilla TRM, DIS-TRM, and STRM at width 64 and
four layers/loops, with seeds 0/1/2 and 10,000 updates each. This is a 12-run
matrix. It retains the sealed-official-test policy and uses generated
validation, depth-extension, and strong-depth suites only.

For Kaggle, each model/seed must be launched as its own one-plan job and, after
each completes, its result, checkpoint, logs, resolved configuration, and the
cumulative manifest must be copied into a durable output bundle. The bundle is
rewritten atomically after every completed plan and zipped before the next plan
begins. If the session stops, rerunning the cell skips `result.json` files that
already exist and resumes the remaining plans. A private Quick Save is required
after the matrix completes to persist the output bundle outside the session.

## Lost-output incident: 2026-08-21 D1 scale-finding attempt

A one-seed D1 comparison of Transformer, TRM, DIS-TRM, and STRM at 10,000
steps completed in a private Kaggle draft session on a Tesla T4. The notebook
reported 6,597.189 seconds of measured workload time and 9.718 GiB peak CUDA
allocation, with official evaluation disabled for all four models. However,
the notebook printed only those aggregate runtime fields; its per-model
`result.json` artifacts remained solely under `/kaggle/working` and were gone
after the Kaggle session reset. Consequently, no accuracy, horizon, or model
comparison can be recovered from this attempt and it must not be cited as an
experimental result. The GPU was released after the audit.

All future Kaggle runs must copy `runs/`, the resolved configuration, and a
compact metrics JSON to a notebook output bundle before ending the session, and
use **Save Version** so those output artifacts persist.

## Persisted D1 scale-finding rerun: 2026-08-21

The rerun used the same one-seed, width-64, four-layer/loop, 10,000-update D1
configuration on a Tesla T4. It wrote `d1-scale-finding-output.zip` before
completion and was Quick Saved successfully as private Kaggle notebook version
1. The draft GPU session was then off. Measured workload time was 6,838.750
seconds; peak allocation was 9.718 GiB; official evaluation was disabled.

| Model | Validation exact | Test-depth exact | Test-depth p_d | Test-depth h50 | Test-depth AUC | Strong-depth exact |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 22.4% | 0.0% | 0.500 | 1.00 | 0.000 | 0.0% |
| Vanilla TRM | 18.8% | 0.0% | 0.500 | 1.00 | 0.000 | 0.0% |
| DIS-TRM | 55.6% | 0.33% | 0.843 | 4.06 | 0.004 | 0.0% |
| STRM | 81.4% | 5.17% | 0.916 | 7.88 | 0.045 | 0.0% |

This is a strong **single-seed D1 scale-finding signal**: STRM substantially
outperforms matched Transformer and vanilla TRM on validation and generated
depth extrapolation; DIS-TRM is the second-best recursive baseline. It is not
a SOTA or paper claim: the strong-depth suite remains at zero exact accuracy,
there is only one seed, the Kaggle source snapshot predates the resumable
launcher, and a structural holdout has not been run. The next justified spend
is a persisted three-seed STRM versus DIS-TRM versus matched Transformer D1
experiment, then the D1 structural split before D2/D3.

Do not enable official evaluation until model selection and hyperparameters are
frozen. Do not advance to D2 or D3 until the D1 scale-finding result identifies
a model with credible held-out-depth signal.
