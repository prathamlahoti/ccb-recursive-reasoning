# D1 Methodology Foundation Audit

Status: **the implementation executes the intended recurrent prediction task,
but the current D1 structural holdout is not a semantically disjoint
compositional-generalization test.** This audit supersedes any stronger
interpretation of the seed-4 structural result.

## What is mechanically correct

1. The D1 generator reproduces the imported official records exactly. The
   oracle applies each official grid operation, records every resulting grid,
   and the encoder supplies the model only the initial grid and operation IDs.
2. `StateTransitionRecursiveModel` maintains a persistent `y` state and latent
   `z` state across program positions. For each operation it performs four
   shared inner `z` updates, then updates `y`, and decodes a predicted grid.
   Targets are used only in the loss after the forward pass; there is no
   teacher-forced ground-truth state input at training or evaluation time.
3. The structural baselines were evaluated on one identical baseline manifest
   (`b69a16...d4140`) and use the same seed-4 core budget. This supports the
   narrow statement that the three baseline numbers are mutually comparable.

Thus STRM is doing a meaningful form of **recurrent trace prediction** on the
defined D1 generator. Its advantage is not explained by targets being passed
into its forward method.

## Critical limitation: token holdout is not semantic holdout

The structural split excludes the adjacent token pair

```text
ROTATE_90_CW -> SHIFT_ROW_2_LEFT
```

from training and requires it at test time. However, exact enumeration of the
official D1 operations establishes

```text
TRANSPOSE_GRID -> FLIP_HORIZONTAL == ROTATE_90_CW
TRANSPOSE_GRID -> FLIP_HORIZONTAL -> SHIFT_ROW_2_LEFT
    == ROTATE_90_CW -> SHIFT_ROW_2_LEFT
```

for the official initial grid (indeed, as grid permutations). These three
tokens do not contain the held bigram, so they are admissible in training.
The model can therefore encounter the same state before `SHIFT_ROW_2_LEFT`
through an alternative spelling. The current test is a **token-sequence
holdout**, not evidence that the underlying composed transformation was unseen.

This does not make the 93.5% STRM result invalid as an execution result. It
does invalidate the previous phrasing that it proves semantic compositional
generalization from an unseen operation pair.

## Additional limits of D1 as currently configured

- The official D1 initial grid is fixed. Training and test therefore probe
  operation-program execution along a single starting grid, not generalization
  over arbitrary grid inputs. Corrupting the initial state at evaluation is a
  useful leakage control but not proof of generalization, because the model was
  never trained on varied valid initial grids.
- STRM is a new CCB-specific architecture, while `VanillaTRM` is a
  prefix-wise adaptation of the published TRM update schedule rather than an
  end-to-end reproduction of the original ARC/Sudoku setup. The comparison is
  useful as an architectural baseline, not an official TRM benchmark result.
- The current structural comparison is one architecture seed per model. It
  cannot establish stable superiority.

## Consequence for the reported baseline gap

The Version-12 result remains correctly recorded as:

| Model | Pair + depth 25--50 final exact |
|---|---:|
| Transformer | 0.00% |
| Vanilla TRM | 0.00% |
| DIS-TRM | 0.50% |
| STRM seed 4 | 93.50% |

The defensible conclusion is now narrower: **STRM has a strong advantage on
this token-level D1 split under this fixed budget.** It is not yet evidence of
an unseen-semantic-pair advantage, SOTA, or a method contribution by itself.

## Required repair before further method claims

1. Introduce randomized valid initial grids during both training and testing.
2. Construct a group-aware semantic split: canonicalize the grid permutation
   enacted by a program/prefix and reject training examples that realize the
   held test transformation or held state--operation transition through an
   alternative token spelling.
3. Audit and report overlap of `(current state, operation, next state)` triples
   across train and every test split, not merely raw program strings.
4. Rebuild STRM and all baseline splits from one saved manifest; independently
   load/evaluate all checkpoints.
5. Only after the repaired split passes, run seeds 3 and 5 for STRM and
   multi-seed matched baselines.

Until then, do not use the existing structural result as the headline research
claim. It is a useful diagnostic and a reason to improve the benchmark.
