# TRM on CCB: Research Go/No-Go Decision

Date: 2026-09-04

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

These results do not support a SOTA claim or a TRM advantage. The subsequent
checkpoint-only training evaluation resolves the immediate ambiguity: the
current TRM configuration did not learn its own 400-example training
distribution. This rejects the present recipe, not the broader hypothesis that
a redesigned recurrent objective or curriculum could help.

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
`prathamlahoti2/ccb-d1-checkpoint-train-evaluation-v1`. It consumes Version 2 of the
completed generalization kernel as a read-only input, restores the two step-10k
live/EMA checkpoints, regenerates and hash-verifies the exact 400 training
episodes, and evaluates both weight sources without performing an optimizer
step. The official records are not evaluated.

Version 1 failed before evaluation because the generic checkpoint loader mapped
the saved CPU RNG-state tensor to CUDA and then passed it to PyTorch's CPU RNG
API. Checkpoint discovery, identity, and manifest validation had already
passed. Version 1 therefore produced no metric. The evaluation-only retry loads
only live and EMA model weights and does not restore optimizer or RNG state.
The generic resume loader is separately corrected to move saved CPU/CUDA RNG
state byte tensors to CPU before calling PyTorch's RNG restoration APIs.

Version 2 completed successfully in 872.2 seconds on two T4 GPUs. It used the
saved step-10,000 checkpoints and the exact regenerated training split with
manifest hash
`f589979a22625a7d0ba68c05194d663fac5013d27c7e732f6c92537593e49167`.
No optimization or official-record evaluation occurred.

### Training-distribution results

| Model | Weights | Final exact | Trace exact | Transition exact | Element accuracy | Valid-state rate | Mean first divergence |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRM | live | 0.75% | 0.00% | 10.88% | 42.96% | 13.72% | 2.20 |
| TRM | EMA | 0.50% | 0.00% | 16.34% | 51.33% | 19.92% | 2.94 |
| Transformer | live | 57.25% | 30.75% | 58.56% | 91.80% | 59.26% | 5.77 |
| Transformer | EMA | 53.75% | 32.00% | 58.04% | 90.98% | 59.20% | 5.98 |

Final-exact accuracy by training depth shows that the Transformer essentially
learned depth 5 but not the whole mixed-depth set; TRM failed at every depth.

| Model/weights | Depth 5 | Depth 10 | Depth 15 | Depth 20 |
|---|---:|---:|---:|---:|
| TRM live | 2% | 1% | 0% | 0% |
| TRM EMA | 1% | 1% | 0% | 0% |
| Transformer live | 99% | 75% | 32% | 23% |
| Transformer EMA | 100% | 77% | 27% | 11% |

This is not merely an unseen-depth generalization failure. TRM has a training
or optimization failure on the scaled mixed-depth dataset despite passing the
small fixed-data fit gates. The Transformer shows a curriculum/capacity issue:
performance deteriorates sharply as training depth increases. Consequently,
the current configuration is stopped. More seeds, D2/D3 runs, large-scale
training, and evaluation on the sealed official records are not justified for
this recipe.

The durable compact record is
`results/d1_checkpoint_train_evaluation_v1/summary.json`. The immutable Kaggle
source is Version 2 of
`prathamlahoti2/ccb-d1-checkpoint-train-evaluation-v1`.

## Conditions for continuing toward a paper

Continue only if a clearly motivated change produces all of the following:

- high training and same-depth validation accuracy;
- a reproducible held-out-depth advantage over Transformer;
- comparison at both parameter-matched and compute-matched budgets;
- at least three seeds after the single-seed method gate;
- transfer beyond D1, preferably to D2;
- ablations identifying why recursion helps rather than merely showing a score.

A plausible paper direction is not “TRM on CCB,” but a controlled study of
whether recursive refinement can overcome depth-induced error accumulation.
The evidence now prioritizes a staged depth curriculum and/or step-local
transition objective: first require near-perfect training and same-depth
validation through depth 20, then test depth extrapolation. That is a new
research hypothesis and must outperform strong recurrent and Transformer
controls. If it does not, the honest outcome is a negative workshop study, not
a new SOTA method.

## Source links

- TRM paper: https://arxiv.org/abs/2510.04871
- Official TRM repository: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- Official CCB repository: https://github.com/Shubh-Chapra/Complexity_Ceiling_Benchmark
- CCB paper/forum: https://openreview.net/forum?id=08mMAMjyLB
