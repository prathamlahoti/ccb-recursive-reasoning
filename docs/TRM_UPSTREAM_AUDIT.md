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
