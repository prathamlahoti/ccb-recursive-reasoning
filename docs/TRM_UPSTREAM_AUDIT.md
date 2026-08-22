# TRM Upstream Audit and Reset

Status: **binding correction before any paper comparison**.

Upstream reviewed: `SamsungSAILMontreal/TinyRecursiveModels` at commit
`c01103738605ba39d1430519b1ee0c62f4c707f8`, specifically
`models/recursive_reasoning/trm.py`, `models/losses.py`, and `models/ema.py`.

## What published TRM actually implements

- Two detached recurrent sequence states, `z_H` and `z_L`, initialized from
  fixed buffers rather than learnable per-cell answer/latent tensors.
- One shared `L_level` containing attention (or its configured MLP-over-token
  alternative), RMS normalization, and SwiGLU.
- For each reasoning update: `H_cycles - 1` no-gradient recursions, each with
  `L_cycles` updates of `z_L` followed by one `z_H` update; then one final
  gradient-bearing recursion of the same form.
- An ACT wrapper that carries state, tracks per-example steps, resets halted
  examples, and trains a two-logit halt/continue head. Evaluation uses the
  configured maximum number of steps for batch consistency.
- A sequence-token output head and per-example normalized token loss, plus
  the ACT loss. EMA is a copied evaluation model, not an in-place replacement
  of the live training model.

## Consequence for this repository

`trm_faithful` is **not** an official TRM reproduction and must no longer be
called faithful in reports. It is a previous CCB-specific recursive prototype.
Its 1,000-update calibration and fixed-data tests are implementation
diagnostics only.

The previous STRM results are also retained only as superseded internal pilot
evidence: their central structural split has a semantic-equivalence flaw and
does not support a publishable comparison.

## Five-step reset protocol

1. Freeze prior STRM and prototype-TRM results as non-paper diagnostics.
2. Implement a CCB adapter of the upstream `z_H/z_L` TRM core, preserving its
   recurrence, shared block structure, detached carries, loss normalization,
   ACT interface, and copied EMA evaluation.
3. Define one explicit CCB token layout: every output trace cell receives an
   input token encoding its initial-state cell and its operation; labels are
   the corresponding oracle trace cell, with padded positions ignored.
4. Pass a deterministic tiny fixed-batch train-fit gate using that core.
5. Freeze the configuration and run exactly one compute-matched
   published-TRM-core versus direct-Transformer comparison on the original
   CCB protocol. The sealed official records are evaluated only afterward.

No STRM result or current `trm_faithful` result may be reported as a paper
number during this protocol.

## Implemented core: `trm_upstream_core`

The implementation in `src/ccb/models/published_trm.py` is the new, separate
candidate used for this reset. It preserves the upstream computational core:

- fixed, non-trainable `h_init` and `l_init` buffers;
- one shared stack of bidirectional-attention, RMSNorm, SwiGLU blocks used for
  both `z_H` and `z_L` updates;
- exactly `H_cycles - 1` no-gradient cycles followed by one gradient-bearing
  cycle; and
- detached outgoing carry plus a two-logit Q-head interface.

The explicit CCB adapter flattens `(transition step, state-cell)` tokens. A
token is `initial_state[cell] * operation_vocab_size + operation[step]`; its
label is only the oracle value of that same trace cell. Thus no target value
can enter the forward input. Padding is excluded through the CCB step mask.

This is an **upstream-derived core**, not yet an exact end-to-end reproduction:
it uses native PyTorch attention/embedding layers rather than Samsung's casted
layers/RoPE implementation, has no puzzle-ID embedding because CCB provides no
puzzle IDs, and does not yet activate the full ACT sampling/loss loop. The
tiny-batch gate is deliberately run with fixed refinement before ACT is added;
otherwise it would conflate architecture correctness with a halting-policy
failure. These deviations must remain explicit in any report.

## Frozen comparison manifest (not yet launched)

`configs/d1-trm-upstream-core-v1.json` and
`configs/d1-transformer-compute-matched-v1.json` define the single-seed,
generated-data calibration comparison. Both use seed 17, width 32, 1,000
optimizer updates, batch size 64, learning rate 0.003, the same primary D1
splits, and leave the official records sealed. The TRM has one shared block
executed 21 times per forward call (`3 * (6 + 1)`); the Direct Transformer has
21 unshared encoder layers. This matches block executions rather than
parameters, which is the meaningful initial comparison for a weight-sharing
recursive architecture. It is a calibration, not a paper result and is not to
be launched until the ACT loss/halting implementation is included or the
omission is deliberately registered as an ablation.
