# Corrected CCB–TRM Fit Ladder v2

Status: implemented and CPU-verified. Commit
`6fbcf8453940e9d5b27833a00f5c4f8503ef468d` was uploaded to the private
Kaggle source dataset on 2026-09-02. No v2 GPU gate has been launched.

## Why v1 is invalid as a TRM fit verdict

The v1 adapter used overlapping integer IDs for state values and operations,
filled output query positions with the valid state token `0`, and used `0` for
operation padding. Its fit configuration also set `lr_min_ratio` to `0.0`,
while the released TRM training configuration uses `1.0`. The v1 negative
result remains a useful adapter diagnostic, but it cannot decide whether a
correct CCB–TRM interface fits D1.

## Collision-free representation

Every model in the v2 comparison consumes the same sequence:

```text
BOS | state-value tokens | OPS | operation tokens | OUTPUT | output canvas
```

The output canvas contains `MASK` at valid state cells and `PAD` beyond the
instance depth. Token ranges are disjoint:

| Namespace | IDs |
| --- | --- |
| State values | `0 .. state_vocab_size-1` |
| Operations | offset by `state_vocab_size` |
| `BOS`, `OPS`, `OUTPUT`, `MASK`, `PAD` | five unique IDs above both ranges |

Only output-canvas positions receive labels. Context and padded output cells
are excluded by the existing step mask. Unit tests prove that changing target
states cannot alter serialized model inputs, and that the TRM and matched
Transformer receive identical token tensors.

## Released TRM training contract

The v2 TRM gates use width 512, 8 heads, 2 shared L layers, H/L cycles 3/6,
ACT horizon 16, bfloat16 forward precision, AdamATan2 with betas `(0.9, 0.95)`,
weight decay `0.1`, learning rate `1e-4`, 2,000 warm-up updates, and
`lr_min_ratio=1.0`. The released repository defaults EMA off in its generic
configuration, but all documented task-result commands enable `ema=True` and
switch to the EMA copy for evaluation. Therefore an upstream-style
reported-result gate must use the EMA copy, while also retaining live metrics
for diagnosis. Evaluation runs the full 16 ACT steps, matching the upstream
batching rule; selecting a better post-hoc recurrence step is not allowed.

Puzzle-ID embeddings remain disabled. CCB generated instances do not have a
non-leaking persistent puzzle identity; assigning one ID per instance would
turn the fit/generalization study into an embedding lookup.

The bfloat16 configurations require hardware with native bfloat16 support.
Tesla T4 is not the target for the released-precision run; a T4 float32 run
must be labelled a precision ablation rather than silently substituted.

The paired T4 diagnostic uses
`d1_official_trm_fit_8x1_t4_float32_v2.json` and
`d1_token_transformer_fit_8x1_t4_float32_v2.json`. It changes only forward
precision from bfloat16 to float32, runs the two models on separate T4 GPUs,
and persists independent checkpoints, logs, results, plus a final summary.
Its purpose is to clear the implementation gate while native-bfloat16
hardware is unavailable; it is not a publishable released-precision result.
After Gate A passes, its depth-5 continuation uses the corresponding
`d1_official_trm_fit_16x5_t4_float32_v2.json` and
`d1_token_transformer_fit_16x5_t4_float32_v2.json` pair.

After that paired Gate B showed that the Transformer passed while TRM's EMA
copy reached 87.5% trace exactness at the required 16-step horizon, the only
authorized follow-up is the TRM-only
`d1_official_trm_fit_16x5_t4_float32_10k_v3.json` diagnostic. It restarts the
same 16 examples from the same seed, increases the budget to 10,000 updates,
and gates explicitly on EMA/max-16 metrics. Gate C remains blocked unless this
run reaches the original 99% final- and trace-exact thresholds.

The 10,000-update Gate B diagnostic subsequently passed with 100% EMA/max-16
trace exactness; its recurrence sweep confirmed 100% EMA accuracy at every
step. Gate C is therefore eligible. Its paired T4 configs are
`d1_official_trm_fit_64x5_t4_float32_v3.json` and
`d1_token_transformer_fit_64x5_t4_float32_v3.json`. TRM gates on EMA and the
Transformer control gates on live weights, following their respective
training paths.

## Sequential gates

| Gate | TRM config | Matched token-Transformer control | Maximum updates |
| --- | --- | --- | ---: |
| A | `d1_official_trm_fit_8x1_v2.json` | `d1_token_transformer_fit_8x1_v2.json` | 2,000 |
| B | `d1_official_trm_fit_16x5_v2.json` | `d1_token_transformer_fit_16x5_v2.json` | 4,000 |
| C | `d1_official_trm_fit_64x5_v2.json` | `d1_token_transformer_fit_64x5_v2.json` | 10,000 |

Each pair uses the same examples, generator seed, model width, layer count,
optimizer, schedule, precision, and update budget. A gate passes only when
the predeclared weight source reaches at least 99% final-state and
complete-trace exact accuracy: EMA for the upstream-style TRM result and live
weights for the current Transformer control. Gate B is not launched unless
Gate A passes; Gate C is not launched unless Gate B passes. No held-out-depth
experiment is permitted until Gate C passes.

These are memorization and pipeline-correctness gates, not publishable
generalization results.

## Completion status

The T4/float32 Gate C Version run completed in 6,867.13 seconds. Both TRM
(EMA/max-16) and the matched token-Transformer (live) reached 100% final- and
trace-exact accuracy on 64 fixed depth-5 examples. The diagnostic ladder is
complete. No further fixed-set fit run is needed; subsequent GPU work must use
a predeclared train/validation/held-out-depth protocol.
