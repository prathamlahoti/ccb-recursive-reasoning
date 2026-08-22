# Faithful TRM Algorithmic Adaptation for CCB

This document defines `trm_faithful`, the primary model for the instruction to
apply TRM to CCB. It replaces neither the original TRM repository nor its
ARC/Sudoku results; it is a controlled adaptation of TRM's algorithm to CCB's
structured transition traces.

## Preserved TRM algorithm

For one CCB program, the model embeds its question tensor `x` (initial state
and ordered operation sequence), answer tensor `y`, and latent tensor `z`.
One latent recursion applies the same shared network six times:

```text
z <- net(x, y, z)    # repeated n=6 times
y <- net(0, y, z)
```

One deep refinement performs `T-1` latent recursions without gradients and one
final recursion with gradients, with default `T=3`. Training carries detached
`y,z` into the next deep-supervision update, applies cross-entropy to the full
oracle trace plus a halt BCE objective, and maintains an EMA of trainable
weights (default decay 0.999). Training budget is counted in optimizer updates,
not outer data-loader passes.

These are the central algorithmic parts of TRM described in the paper:
shared `y/z` refinement, no-gradient warm-up refinements, a final
gradient-bearing refinement, deep supervision, and EMA.

The initial implementation trains the halt head but does **not** yet use it to
stop recursion adaptively. Adaptive computation time is therefore an explicit
follow-up ablation, not something claimed by the first CCB calibration.

## CCB-specific choices

- `x` contains the D1/D2/D3 initial state and ordered program.
- `y` is a latent representation decoded to a full state after every program
  operation, so CCB trace metrics remain available.
- The refinement network uses a CCB cell mixer plus MLP; it is not the exact
  original TRM MLP-Mixer architecture.
- The initial calibration uses smaller width/batch values than the paper,
  because CCB has different state shapes and free-tier GPU constraints.

Therefore call this a **faithful TRM algorithmic adaptation**, never an exact
reproduction of the published Sudoku/ARC implementation.

## Primary protocol

1. Train on generated official-semantics CCB-Learn only.
2. Freeze configuration before touching official CCB records.
3. Evaluate direct Transformer, recurrent baseline, and `trm_faithful` on the
   same serialized primary splits and fixed optimizer-update budget.
4. Evaluate the selected frozen configuration once on official CCB records.
5. Run STRM only as an extension/ablation. Run randomized semantic D1 only as
   a diagnostic CCB-Learn stress test.

## First GPU calibration

`configs/d1-trm-faithful-calibration-v1.json` is deliberately one seed, no
official evaluation, and a 1,000-update budget. It measures runtime, memory,
loss dynamics, and whether the implementation learns at all before any
multi-seed comparison. It uses four (rather than sixteen) deep-supervision
updates per loader batch so it is a calibration, not a paper result.

## Fixed-data overfit gate

Before tuning or scaling, run one fixed-data D1 overfit gate: 32 deterministic
depth-4 episodes, one seed, and no held-out evaluation. The acceptance
criterion is near-perfect final and trace exact accuracy on those same 32
examples. Failure means the implementation or optimization recipe is not yet
fit for a benchmark run; success only establishes basic learnability, not
generalization.

When an overfit gate fails, compare the live (raw) weights with EMA weights
under an identical seed and training budget before changing architectural or
optimization settings. EMA is the paper-style evaluation default; raw weights
are a diagnostic only and must never be silently substituted into a benchmark.

## Implementation checks

- `FaithfulCCBTRM.refine` enforces the TRM detach schedule.
- `train_faithful_trm_batches` carries detached states across deep-supervision
  updates and uses EMA weights for final evaluation/checkpointing.
- CPU tests cover forward/backward behavior, target-independence, and a small
  deep-supervision/EMA training run.
