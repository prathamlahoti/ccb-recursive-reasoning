# TRM on CCB: Research Go/No-Go Decision

Date: 2026-09-03

## Current verdict

The exact public combination of Tiny Recursive Models (TRM) with the Complexity
Ceiling Benchmark (CCB) was not found in searches of arXiv, OpenReview, GitHub,
and Semantic Scholar, and neither official repository references the other.
This is evidence that the crossover is not publicly documented, not proof that
no private or unindexed attempt exists.

Novelty by combination is weak. A paper requires a scientific result about
depth generalization, recurrence, adaptive computation, or training—not merely
running TRM on three new generators.

The present seed-17 result is a valid negative calibration:

- TRM and the matched token Transformer both have 0% final exact accuracy at
  held-out depths 25--50.
- Transformer exceeds TRM in aggregate validation transition accuracy
  (22.48% versus 12.40%) and held-out transition accuracy (5.78% versus 3.85%).
- Neither model generalizes reliably to fresh examples even at depths included
  in training.
- TRM costs approximately 13.2 times more wall time in this run and applies 42
  transformer blocks per inner forward versus two for the control.

These results do not support a SOTA claim or a TRM advantage. They also do not
yet establish that TRM fundamentally cannot help: training-set accuracy was not
recorded, so optimization failure and memorization remain unresolved.

## Why absence of prior work is unsurprising

TRM and CCB are recent and address different experimental settings. TRM is a
small supervised, task-specific recursive solver demonstrated on Sudoku,
mazes, and ARC. Its released Sudoku recipe uses roughly 1,000 base examples,
1,000 augmentations, and 50,000 epochs. CCB is released as an inference-time
diagnostic for prompted large language models, with fixed textual records and
depth-controlled traces. A collision-free fixed-canvas supervised interface is
therefore an adaptation, not a plug-in evaluation supported by either project.

## Bounded decision gate

Run exactly one checkpoint-only diagnostic on the already trained models:

1. Evaluate live and EMA checkpoints on the exact 400 training episodes.
2. Do not optimize, tune, expose official records, or repeat seeds.
3. If training exact accuracy is low, stop the current configuration: it did
   not learn its own training distribution.
4. If training exact accuracy is high but validation remains low, stop simple
   scaling: it memorized 400 programs and needs a redesigned data curriculum or
   local transition objective.

The private Kaggle execution harness for this gate is
`prathamlahoti2/ccb-d1-checkpoint-train-eval-v1`. It consumes Version 2 of the
completed generalization kernel as a read-only input, restores the two step-10k
live/EMA checkpoints, regenerates and hash-verifies the exact 400 training
episodes, and evaluates both weight sources without performing an optimizer
step. The official records are not evaluated.

## Conditions for continuing toward a paper

Continue only if a clearly motivated change produces all of the following:

- high training and same-depth validation accuracy;
- a reproducible held-out-depth advantage over Transformer;
- comparison at both parameter-matched and compute-matched budgets;
- at least three seeds after the single-seed method gate;
- transfer beyond D1, preferably to D2;
- ablations identifying why recursion helps rather than merely showing a score.

A plausible paper direction is not “TRM on CCB,” but a controlled study of
whether recursive refinement can overcome depth-induced error accumulation,
with a method change such as step-local state-transition supervision or a
curriculum that explicitly stabilizes recurrent rollout. That is a new research
hypothesis and must outperform strong recurrent and Transformer controls. If it
does not, the honest outcome is a negative workshop study, not a new SOTA
method.

## Source links

- TRM paper: https://arxiv.org/abs/2510.04871
- Official TRM repository: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- Official CCB repository: https://github.com/Shubh-Chapra/Complexity_Ceiling_Benchmark
- CCB paper/forum: https://openreview.net/forum?id=08mMAMjyLB
