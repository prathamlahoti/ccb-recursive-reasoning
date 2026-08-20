# TRM on the Complexity Ceiling Benchmark

Research code for evaluating Tiny Recursive Models and a persistent
State-Transition Recursive Model (STRM) on the official Complexity Ceiling
Benchmark (CCB).

## Current status

- Official CCB repository pinned at commit
  `e1e953a7a33a8c2406f936a54f4900f740dea8dd`.
- All 1,200 official records are included with upstream hashes.
- Local D1, D2, and D3 solvers reproduce every official prompt, intermediate
  trace, and final answer exactly: 1,200/1,200.
- 56 tests pass on Windows and Ubuntu 22.04 under WSL2.
- Direct Transformer, GRU/LSTM, looped Transformer, CCB-adapted TRM,
  DIS-TRM, fast-slow recurrence, STRM, and D3 GNN share one training interface.
- The CPU engineering foundation is complete: official-test firewall,
  definitive splits, variable-depth batching, full depth evaluation,
  multi-seed launch, and Deep Improvement Supervision.
- No scientific model-comparison claim has been made yet. Multi-seed training
  is the next phase and should use CUDA.

## Setup

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
python -m ccb verify-official
```

WSL2 verification:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Generate learning or structural-holdout data without touching the fixed
official evaluation records:

```powershell
python -m ccb generate --domain d1 --output artifacts\ccb_learn_v1\d1
python -m ccb generate-structural --domain d2 `
  --output artifacts\ccb_learn_structural_v1\d2
```

Run a reproducible experiment matrix from a JSON configuration:

```json
{
  "domain": "d1",
  "models": ["transformer", "trm", "dis_trm", "strm"],
  "seeds": [0, 1, 2],
  "output_directory": "runs/d1-pilot",
  "width": 64,
  "layers_or_loops": 4,
  "steps": 1000,
  "batch_size": 32,
  "checkpoint_interval_steps": 500,
  "include_official_evaluation": false,
  "device": "cpu"
}
```

```powershell
python -m ccb launch --config experiment.json --dry-run
python -m ccb launch --config experiment.json
```

Official evaluation is opt-in and must remain false during architecture and
hyperparameter selection.

## Source-of-truth documents

- [Official benchmark and extension specification](docs/BENCHMARK.md)
- [Implementation and verification](docs/IMPLEMENTATION.md)
- [Research plan and literature](docs/RESEARCH.md)
- [Provisional paper draft](paper/draft.md)
- [Official data provenance](data/ccb_official/SOURCE.md)

## Repository layout

```text
data/ccb_official/   pinned official fixed records and license
docs/                current project documentation
paper/               provisional paper draft
src/ccb/             benchmark, models, training, metrics, and CLI
tests/               unit, integration, and exhaustive compatibility tests
```

Generated datasets, checkpoints, logs, caches, upstream clones, and rendered
paper intermediates are intentionally excluded from version control.
