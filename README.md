# Applying TRM to the Complexity Ceiling Benchmark

This repository implements a **supervised CCB-TRM extension**, not an
official CCB LLM leaderboard submission. Its objective is to test whether an
a faithfully ported Tiny Recursive Model retains sequential state accuracy under
CCB-style depth scaling better than a compute-matched Transformer.

The full scope, non-claims, and required implementation gates are in
[docs/PROJECT_METHOD.md](docs/PROJECT_METHOD.md). Previous pilot models and
results have been removed because they do not support this objective.

## Verified foundation

- The official CCB source is pinned at `e1e953a7a33a8c2406f936a54f4900f740dea8dd`.
- The included D1/D2/D3 task adapters reproduce all 1,200 official records
  exactly.
- The fixed official records are evaluation-only. Generated data is required
  for supervised TRM training and must be labelled as an extension.
- The no-puzzle official TRM core and ACT wrapper have exact float32
  forward/gradient equivalence against the pinned upstream checkout. The next
  gate is the fixed-data run in
  [`configs/d1_official_trm_fit_gate_v1.json`](configs/d1_official_trm_fit_gate_v1.json).

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

## Source-of-truth documents

- [Project method](docs/PROJECT_METHOD.md)
- [TRM fidelity audit](docs/TRM_FIDELITY_AUDIT.md)
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
