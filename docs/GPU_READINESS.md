# GPU Readiness: CCB-TRM D1 Calibration v1

Status: ready for one server-side Kaggle Version run per configuration.

## Preconditions verified locally

- The CCB task adapters reproduce all 1,200 pinned official records exactly.
- Generated CCB-TRM data begins at depth 5, matching CCB's published minimum.
- ACT batches are grouped by codec and depth. A new depth is accepted only
  after all active rows halt; recursive state is never silently reused with a
  different sequence shape.
- The TRM port uses fixed H/L buffers, shared reasoning blocks, the released
  detached H-cycle schedule, stablemax per-example token loss, halt BCE,
  copied EMA evaluation, and fixed-max-step evaluation.
- ACT checkpoint/resume restores optimizer, copied EMA, ACT carry/pending
  batch, Python/Torch/CUDA RNG state, and manifest identity. The deterministic
  interrupted/resumed CPU test matches uninterrupted parameters exactly.
- A three-update full launcher run completed on CPU with checkpointing and
  EMA evaluation. The official-evaluation flag was false.

## Frozen calibration

Run these configurations separately, with no tuning between them:

1. `configs/d1_trm_act_calibration_v1.json`
2. `configs/d1_transformer_compute_matched_v1.json`

Both use generated CCB-TRM D1 data, seed 17, width 32, 1,000 optimizer
updates, batch size 64, learning rate 0.003, 50-update durable checkpoints,
and 200 bootstrap resamples. The official D1 records remain sealed.

The TRM has one shared block executed 21 times per optimizer update
(`H_cycles=3`, `L_cycles=6`). The Transformer has 21 unshared encoder blocks.
This is compute-matched by block execution, not parameter count; that is the
appropriate first comparison for a weight-sharing method.

## What this run can establish

It is a single-seed calibration: it can establish whether the corrected TRM
path trains, fits generated CCB semantics, and merits multi-seed work. It
cannot establish SOTA, a general reasoning claim, or an original CCB
leaderboard result.
