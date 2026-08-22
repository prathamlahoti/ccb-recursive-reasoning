# D1 STRM Seed-Stability Results

Source: private Kaggle notebook `CCB D1 STRM Seed-2 Replication`, completed
Version 5 (script version `344084766`). The immutable Kaggle output contains
the raw `seed3-result.json`, `seed4-result.json`, `seed5-result.json`, each
configuration, and `completed_records.json`.

## Protocol

- Domain: D1
- Model: STRM, 153,033 parameters; width 64; 4 loops
- Optimizer configuration: learning rate 0.001; batch size 64
- Training: 10,000 steps per seed
- Seeds 3 and 4 ran concurrently on independently pinned T4 GPUs; seed 5
  started once GPU 1 became free. Each seed wrote an isolated JSON result.
- Evaluation uses the CCB-Learn held-out-depth and strong-depth splits.
  `official_evaluation_used` is false, so these are not claims about an
  external official leaderboard.

## Held-out depth (D1, depths 25--50)

| Seed | Train loss | Final exact | Trace exact | Transition exact | Element accuracy | Valid state rate | TFBC | \(p_d\) | \(h_{50}\) | Depth AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 3.23e-7 | 89.00% | 82.00% | 96.70% | 99.54% | 96.91% | 7.00% | 0.996873 | 221.29 | 88.80% |
| 4 | 2.91e-7 | 98.67% | 96.17% | 99.61% | 99.95% | 99.63% | 2.50% | 0.999642 | 1934.06 | 98.80% |
| 5 | 1.38e-7 | 97.67% | 97.00% | 99.54% | 99.94% | 99.55% | 0.67% | 0.999370 | 1099.07 | 97.50% |

## Strong-depth holdout

| Seed | Final exact | Trace exact | Transition exact | Element accuracy | Valid state rate | TFBC | \(p_d\) | \(h_{50}\) | Depth AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 66.33% | 44.00% | 88.59% | 97.76% | 91.10% | 22.33% | 0.994778 | 132.40 | 65.75% |
| 4 | 90.67% | 83.00% | 97.18% | 99.49% | 98.50% | 7.67% | 0.998769 | 562.86 | 90.00% |
| 5 | 86.33% | 82.00% | 94.45% | 99.07% | 95.52% | 4.33% | 0.998159 | 376.07 | 86.75% |

## Interpretation

The high-generalization regime is now reproducible across three new random
seeds. Strong-depth results remain lower than standard held-out-depth results,
which is expected for a stricter extrapolation test, but remain substantial:
66.33--90.67% final exact accuracy. This materially weakens the hypothesis
that seed 2 was only an artifact or a single lucky checkpoint.

However, seeds 0 and 1 from the original six-seed study were low-performing.
The responsible claim is therefore that STRM exhibits a reproducible but
seed-sensitive high-performing regime on this D1 configuration. Structural
holdout and negative-control experiments are still required before any
algorithmic or SOTA claim.
