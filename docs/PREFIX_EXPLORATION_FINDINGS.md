# Prefix-refinement investigation and decision

Date: 2026-10-12. All new experiments used CPU. No GPU job submitted.

## Findings

The prefix hypothesis is implemented as an opt-in training ablation, but no
successful generalisation result has been established. The important new
finding is that a D1-specific learned executor reaches perfect performance on
the generated length tests with 567 parameters. D1 alone offers little room for
a TRM accuracy advantage over a properly structured baseline.

## Concrete foundation corrections

- The ACT and generic trainers now log accepted example presentations by
  depth and supervised state-cell presentations. Pending examples are not
  automatically accepted by ACT. The result records these counts for each
  invocation, with the resume interval explicitly identified.
- Fixed padding does not remove PAD tokens from attention; the ACT-stream
  docstring incorrectly said that it did. This description is corrected.
  The upstream full-attention operator itself is retained.
- The official adapter carries detached previous predictions for loss-only
  diagnostics. Row-reset masks exclude comparisons between different
  episodes. Checkpoints preserve this state across resumed training.
- Prefix length and regression counts are now measured. Existing losses and
  core parameters are unchanged at the default zero prefix weights.

These improve measurement and prevent misleading comparisons. They do not
resolve the original model's poor learning or prove a new method.

Verification: 84 tests passed, including prefix masks, ACT row replacements,
zero-weight loss identity, checkpoint resume, and learned-executor label
independence. A two-update launcher smoke run exercised both modified training
paths with checkpoint writes and generated validation: ACT accepted eight
example presentations versus sixteen for the Transformer. The CPU exploration
script now atomically saves each completed condition before the final summary.

## Loss and its limitations

Let t index execution states and r index neural refinement iterations. Let
k_r be the number of consecutive exact execution states predicted at iteration
r. The prototype weights valid trace state t by:

`w(r,t) = 1 + alpha * [t = k_r + 1] + beta * [t belongs to the previous correct prefix]`.

The preservation term is active only for rows continuing the same ACT episode.
Weights are detached and the state-token NLL is normalised by their sum per
example. The full-trace halt BCE is retained. Targets enter the loss and
diagnostics only. The model input remains the initial state and operations.

This is supervised reweighting of existing labels. It can focus learning on the
first error, but preservation may spend gradient budget on already solved
states. A smaller NLL does not guarantee a longer exact prefix. No new learned
progress head or hard-freezing rule was added before this basic test.

When the prefix is empty, the preservation term has no signal. When the entire
trace is correct, uniformly increasing preservation weights cancels in the
normalisation. Hard-freezing confidence could lock in incorrect states; that
mechanism is therefore not justified by the present results.

The prototype overlaps with existing supervision and refinement work:
[CLRS](https://arxiv.org/abs/2205.15659),
[causal regularisation](https://arxiv.org/abs/2302.10258), and
[Denoising Recursion Models](https://proceedings.mlr.press/v306/cameron26a.html).
There is no defensible novelty claim for this loss alone.

## Eight paired CPU runs

16 fixed generated depth-3 training examples, 32 fresh depth-3 validation
examples; seeds 7 and 17; width 32, one layer, H=1, L=1, ACT horizon 4, 400
updates, AdamW learning rate 0.003, no weight decay or warm-up, EMA 0.99.
The weighted runs used alpha=beta=1. This is a reduced-model diagnostic,
not the released-width TRM training recipe.

| Seed | Canvas depth | Weight | EMA train final | EMA validation transition | Training prefix-regression rate |
|---|---:|---:|---:|---:|---:|
| 7 | 3 | 0 | 12.50% | 0.00% | 83/247 = 33.60% |
| 7 | 3 | 1 | 0.00% | 1.04% | 235/860 = 27.33% |
| 7 | 20 | 0 | 0.00% | 0.00% | 74/203 = 36.45% |
| 7 | 20 | 1 | 0.00% | 1.04% | 208/668 = 31.14% |
| 17 | 3 | 0 | 0.00% | 0.00% | 38/70 = 54.29% |
| 17 | 3 | 1 | 0.00% | 3.13% | 210/447 = 46.98% |
| 17 | 20 | 0 | 0.00% | 2.08% | 52/90 = 57.78% |
| 17 | 20 | 1 | 0.00% | 3.13% | 180/361 = 49.86% |

All eight had zero EMA training trace accuracy, validation trace accuracy, and
validation final accuracy. All accepted 1,600 example presentations during
400 updates (6,400 pending row presentations). Thus pending-row counts would
overstate ACT example exposure by four times in this controlled case.

Regression denominators differ between methods. These are pooled online
training comparisons across iterations, during changing model weights; they
are not independent trials or frozen-model inference-stability estimates.
The consistent direction is a small signal worth recording, not proof of a
mechanism or a statistically established advantage. The short budget and small
architecture cannot rule out the idea at released scale.

Canvas differences are also inconclusive: most runs underfit, so these tests
cannot establish padding as the original failure's cause.

## Strong learned D1 baseline

`LearnedGridExecutor` learns a 7 x 9 x 9 tensor of source-cell routing logits.
Each operation chooses which previous grid cell supplies each new grid cell.
The operation maps are learned from 512 random-start one-step examples for
200 Adam updates at learning rate 0.1. Training source indices are derived
from before/after state pairs, exploiting D1's unique cell values.

At inference, the model repeatedly applies its learned maps to its own
predicted state. Neither the oracle nor target states participate in rollout.
A target-corruption test confirms predictions do not depend on test labels.

| Generated test depth | Examples | Final exact | Full trace exact |
|---:|---:|---:|---:|
| 5 | 100 | 100% | 100% |
| 20 | 100 | 100% | 100% |
| 50 | 100 | 100% | 100% |
| 100 | 100 | 100% | 100% |

This baseline has a substantial task-specific inductive bias: D1 operations
are permutations of nine cells. It is not an apples-to-apples architectural
comparison against whole-trace TRM, a new general reasoner, an official CCB
evaluation, or SOTA evidence. Its role is to show that D1's length challenge
can disappear once transitions are represented appropriately.

If every learned source-cell map equals the actual operation map, induction
ensures every subsequent predicted state is correct for any program length.
This explains these scores without invoking a mysterious reasoning ability.

## Numbers constraining future work

- D1 canvases contain `12 + 10 * max_depth` tokens: 62 at depth 5, 212 at
  depth 20, and 512 at depth 50.
- Training at depths up to 20 directly supervises at most 180 output cells;
  depth-50 evaluation asks for 450. The remaining 270 cells are 60% of the
  requested output and become MASK slots after being PAD during training.
  This is a representation-distribution shift, not proof of a coding bug.
- Under an independent constant per-transition success approximation, 90%
  full-trace success at depth 50 needs p=0.997895; at depth 100 it needs
  p=0.998947. At p=0.99, depth-50 success is only 60.50%; at p=0.95 it is
  7.69%. These are illustrations of compounded error, not fitted estimates.
- With independent uniform predictions over nine values, one exact state
  has probability 9^-9 = 2.58e-9. A permutation-constrained random state has
  probability 1/9! = 2.76e-6. Both illustrate why full-trace halt positives
  can be sparse initially; actual model predictions are neither distribution.

An attempt to retrieve the old private Kaggle training ledger failed because
the local environment denied the network socket. Therefore the historical
10,000-step run's exact accepted-example counts were not recovered. The new
instrumentation and measured CPU counts must not be retroactively assigned
to that older experiment.

## Decision

Do not launch a large prefix-loss GPU matrix from these results. Keep the
implementation as an explicit ablation and the routing model as a required D1
control. The gap demonstrated here is in the experimental foundation:
transition structure and ACT exposure were inadequately represented/measured.

A research continuation should first ask whether recursion improves learning
or repair on tasks where the transition rule is context-dependent or where
global constraints require refinement, compared with a trained local executor
and looped Transformer under equivalent exposure and compute. D2 or a suitable
CLRS task may provide this setting; their difficulty must be established by
controls rather than assumed. Changing the task remains an explicit research
extension, not a reinterpretation of the original CCB protocol.

No confirmed publishable TRM improvement has been found in this investigation.
The result is useful because it gives a cheap strong baseline, reliable
instrumentation, and a concrete reason to avoid another unsupported GPU sweep.

Reproduce: `PYTHONPATH=src python scripts/explore_prefix_cpu.py --steps 400`.
Results: `results/prefix_cpu_exploration_v1/summary.json`.
