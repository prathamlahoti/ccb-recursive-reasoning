# Corrected CCB–TRM Fit Ladder v2

Status: implemented and CPU-verified on 2026-08-30. No v2 GPU gate has been
launched.

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
`lr_min_ratio=1.0`. Live weights determine gate passage. EMA evaluation is
disabled and may be added later only as an explicit ablation.

Puzzle-ID embeddings remain disabled. CCB generated instances do not have a
non-leaking persistent puzzle identity; assigning one ID per instance would
turn the fit/generalization study into an embedding lookup.

The bfloat16 configurations require hardware with native bfloat16 support.
Tesla T4 is not the target for the released-precision run; a T4 float32 run
must be labelled a precision ablation rather than silently substituted.

## Sequential gates

| Gate | TRM config | Matched token-Transformer control | Maximum updates |
| --- | --- | --- | ---: |
| A | `d1_official_trm_fit_8x1_v2.json` | `d1_token_transformer_fit_8x1_v2.json` | 2,000 |
| B | `d1_official_trm_fit_16x5_v2.json` | `d1_token_transformer_fit_16x5_v2.json` | 4,000 |
| C | `d1_official_trm_fit_64x5_v2.json` | `d1_token_transformer_fit_64x5_v2.json` | 10,000 |

Each pair uses the same examples, generator seed, model width, layer count,
optimizer, schedule, precision, and update budget. A gate passes only when
both live final-state exact accuracy and complete-trace exact accuracy reach
at least 99%. Gate B is not launched unless Gate A passes; Gate C is not
launched unless Gate B passes. No held-out-depth experiment is permitted until
Gate C passes.

These are memorization and pipeline-correctness gates, not publishable
generalization results.
