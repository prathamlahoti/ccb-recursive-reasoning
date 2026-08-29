# Corrected TRM Implementation Checklist

Status: **do not launch the D1 comparison until every Required item is
complete and tested.** This checklist replaces the earlier plan to use the
`trm_upstream_core` fixed-refinement model directly in a benchmark.

Upstream reference: `SamsungSAILMontreal/TinyRecursiveModels`, commit
`c01103738605ba39d1430519b1ee0c62f4c707f8`, principally
`models/recursive_reasoning/trm.py`, `models/losses.py`, and `models/ema.py`.
The architectural audit is in `TRM_UPSTREAM_AUDIT.md`.

## Already complete

| Item | Evidence | Status |
|---|---|---|
| Freeze old STRM and `trm_faithful` claims | `TRM_UPSTREAM_AUDIT.md` | Complete |
| Upstream source audit | pinned commit and file-level notes | Complete |
| Upstream-style inner recurrence | `PublishedTRMCCB`: fixed `z_H/z_L`, shared blocks, 3x6 detached schedule | Complete |
| Explicit target-safe CCB token layout | trace-cell input is `(initial cell, operation)`; oracle trace is label-only | Complete |
| Core-only fit check | 8 fixed D1 episodes reached 100% trace exact by update 80 | Complete |
| Baseline comparison configs | frozen but explicitly **not launched** | Complete |

The core-only fit result validates tensor shapes, gradients, recursion, and
basic optimization. It does **not** validate the published TRM training
algorithm or generalization.

## Required fixes before the first corrected comparison

### 1. Implement the ACT wrapper, not only its Q-head

Current state: `PublishedTRMCCB` returns two Q logits but always starts from
fresh initial carry and performs exactly one outer update.

Implement an `ACTCarry` holding:

- detached `z_H/z_L` state;
- per-example step counters and halted flags; and
- the current CCB episode tensors for active examples.

For every outer update, reset only halted examples to fixed initial buffers,
replace their current episode, retain non-halted examples and their carry, and
then run the inner recurrence. During training, reproduce the upstream
halt/continue rule and exploration policy. During evaluation, run all examples
for `halt_max_steps`; this matches upstream batch-consistent evaluation.

**Acceptance tests:** reset only affects halted rows; active rows retain their
carry; evaluation takes exactly the configured number of outer updates; a
fixed random seed makes halting decisions reproducible.

### 2. Use the published loss structure

Current state: generic CCB training uses global softmax cross-entropy.

Implement TRM's stablemax token loss, normalize it independently by each
example's number of valid trace tokens, and add the halt BCE term at the
published half-weight. Preserve the upstream optional continue-Q bootstrap as
a named configuration; use the upstream default `no_ACT_continue=true` unless
we pre-register a justified CCB deviation.

**Acceptance tests:** compare stablemax loss against a direct reference
calculation; prove padded tokens contribute zero; prove changing one example's
trace length cannot reweight another example's loss.

### 3. Replace in-place EMA with a copied evaluation model

Current state: the old prototype has a shadow-tensor EMA; the new core has no
EMA evaluation path.

Create a copied EMA model after device placement, update it after each
optimizer step, and evaluate only the copied model. Keep live training weights
unchanged. Save both live and EMA state in checkpoints.

**Acceptance tests:** optimizer parameters do not change during EMA
evaluation; checkpoint reload exactly restores both models; raw-versus-EMA is
reported only as a diagnostic, never silently substituted.

### 4. Make resumable training include algorithmic state

Current state: generic checkpoints save model and optimizer but not ACT carry,
active examples, RNG state, or EMA model.

Checkpoint atomically at a fixed update interval and include model, EMA,
optimizer, update count, ACT carry/current batch, Python/Torch/CUDA RNG state,
resolved configuration, and dataset-manifest hash. The Kaggle controller must
archive result JSON, checkpoint, train JSONL, and manifest after every plan.

**Acceptance test:** interrupted-then-resumed training is bitwise-equivalent
to an uninterrupted CPU run for a short deterministic schedule.

### 5. Validate architectural fidelity and declare deliberate adapters

The CCB adapter necessarily differs from Sudoku/ARC TRM: CCB has no puzzle-ID
embedding; its trace token layout is new; native PyTorch replaces the released
casted layers; and this implementation currently uses learned positions rather
than the selected upstream positional option.

For each difference, either implement the upstream option or record it as an
explicit CCB adapter choice with an ablation plan. Also check fixed-buffer
initialization, Q-head initialization, no-gradient boundaries, attention
causality (must remain bidirectional), and shared-weight identity.

**Acceptance tests:** parameter identity confirms the same `L_level` is used
for H and L; gradient checks prove only the final H cycle is tracked; target
mutation cannot alter logits; a reference forward comparison covers the parts
that can be matched without the upstream data format.

### 6. Run an ACT-enabled tiny-fit gate

Do not reuse the earlier core-only gate as this test. Use a fixed, deterministic
tiny D1 batch and the full ACT loss/training path. Require 100% trace exact
accuracy, sensible finite halt losses, and a saved/reloaded checkpoint that
retains the result.

This is a CPU test first. GPU is unnecessary until it passes.

### 7. Freeze and run one calibration comparison

Only after steps 1-6 pass, freeze:

- CCB-Learn D1 primary manifest and firewall hash;
- one seed, optimizer-update budget, learning-rate schedule, batch size, and
  precision policy;
- the definition of compute matching; and
- primary metrics: final exact, transition exact, trace exact, retention,
  horizon, and confidence intervals.

Run `trm_upstream_core` versus Direct Transformer on generated CCB-Learn D1
with the official 400 D1 records still excluded. The present 21-block
Transformer configuration is a candidate compute-matched calibration, but it
must be regenerated after the ACT implementation fixes the actual per-update
TRM cost. It is **not** yet the final experiment.

## Explicitly out of scope for this correction

- New STRM architectures or more STRM seeds.
- D2/D3 scaling.
- Structural-holdout or SOTA claims.
- Evaluation on official records.
- Gaussian-splatting work.

Those become meaningful only after the corrected D1 calibration is complete.

## Definition of success at this stage

Success is not a high accuracy number. It is a reproducible, audited,
ACT-enabled CCB adaptation whose tiny-fit gate and one frozen comparison can
be trusted. A positive comparison then motivates multi-seed and D2/D3 work;
a negative comparison is still a valid answer to the original direction.
