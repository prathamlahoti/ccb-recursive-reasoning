# D1 Validation Results After the Initial Seed Study

This document records the validation experiments performed after the original
D1 four-model, seeds 0--2 matrix. Raw records remain in private Kaggle Version
outputs; this file preserves the experiment intent and headline results in the
private Git history.

## Fixed model and evaluation settings

Unless explicitly noted, STRM uses width 64, four recursive loops, learning
rate 0.001, batch size 64, 10,000 optimizer steps, raw-argmax decoding, and the
CCB-Learn D1 generator with the official-record firewall. The successful
reference configuration is 153,033 parameters.

Baseline structural runs use the *same* D1 split, seed 4, width 64,
four layers/loops, learning rate 0.001, batch size 64, and 10,000 steps. Their
parameter counts differ naturally by architecture; this is a fixed-budget,
not parameter-matched, comparison.

## Causality controls (STRM seed 4)

Private Kaggle Version `344107631` retrained seed 4 and evaluated held-out
depth under deliberately invalid inputs. Targets were never provided as model
inputs.

| Evaluation condition | Final exact | Transition exact |
|---|---:|---:|
| Normal inputs | 98.67% | 99.61% |
| Operation order independently permuted within each program | 0.00% | 0.72% |
| Initial state corrupted while program and targets are fixed | 0.00% | 0.00% |

Interpretation: the high score depends on both the ordered operation program
and the initial state. These controls rule out the most direct input-shortcut
explanations, but are not a substitute for structural generalization tests.

## Structural holdout (STRM seed 4)

Private Kaggle Version `344194025` uses an explicit ordered-pair holdout:

- Train/validation programs exclude
  `ROTATE_90_CW -> SHIFT_ROW_2_LEFT`.
- Structural test programs require that exact pair.
- Structural matched-depth test uses depths 5, 10, 15, and 20.
- Structural-plus-depth test uses depths 25--50.

Duplicates are allowed inside an individual D1 split. This is necessary:
D1 has only seven possible depth-1 programs. The structural condition itself
makes the train and structural-test program classes disjoint. The official
firewall remains active.

| Evaluation split | Final exact | Transition exact |
|---|---:|---:|
| Validation, no held pair | 100.00% | 100.00% |
| Matched-depth structural pair holdout | 99.50% | 99.92% |
| Structural pair + depth holdout | 93.50% | 97.89% |

The successful run completed in 2,919 seconds and wrote progress/checkpoints
every 1,000 steps. An earlier structural run was cancelled because it
incorrectly demanded globally unique programs and could never fill depth 1;
it is excluded from all analysis.

## Fair structural baselines (seed 4)

Private Kaggle Version `344204584` (Version 12, completed successfully in
2,394.9 seconds on two T4 GPUs) ran Transformer, vanilla TRM, and DIS-TRM
against the exact same generated splits. Every result has the same manifest
hash, `b69a16a994c18381485204bb075df721551854d068e96ff27cb8c87b4a6d4140`,
which verifies split identity across the three baselines. STRM row is the
previously completed seed-4 run (`344194025`) and is shown only for direct
comparison.

| Model | Parameters | Final train loss | Validation no-pair: final / transition | Matched-depth pair: final / transition | Pair + depth (25--50): final / transition |
|---|---:|---:|---:|---:|---:|
| Transformer | 248,649 | 0.690 | 22.20% / 39.24% | 7.00% / 32.50% | 0.00% / 11.78% |
| Vanilla TRM | 123,657 | 1.013 | 13.20% / 18.78% | 1.25% / 15.30% | 0.00% / 5.49% |
| DIS-TRM | 123,657 | 0.339 | 55.40% / 72.46% | 50.00% / 69.40% | 0.50% / 29.29% |
| STRM | 153,033 | n/a* | 100.00% / 100.00% | 99.50% / 99.92% | 93.50% / 97.89% |

\*The STRM structural runner reported the durable evaluation record but not a
directly comparable final-loss field in the short result summary; its earlier
seed-4 controls and its structural validation both reached essentially exact
fit.

The critical result is the combined **unseen operation-pair plus extrapolated
depth** split. At the same 10,000-step budget, STRM reaches 93.50% final-exact
accuracy, whereas the best baseline, DIS-TRM, reaches 0.50%. This is a large
and qualitatively meaningful separation, not a minor improvement. It also
shows why matched-depth structural accuracy alone is insufficient: DIS-TRM
retains 50.0% final exact there but fails when sequence depth rises from
5--20 to 25--50.

This result is a strong fixed-budget advantage on the current **token-level**
split, but it must not be called semantic compositional generalization. A
post-run algebra audit found that the held pair has an allowed alternative
spelling: `TRANSPOSE_GRID -> FLIP_HORIZONTAL -> SHIFT_ROW_2_LEFT` implements
the same net transformation as the held pair. See
`D1_FOUNDATION_AUDIT.md`. The benchmark is CCB-Learn rather than the
unavailable original CCB learning generator, and the comparison is one seed
per architecture.

## Current evidence and remaining bounded D1 work

- Baseline depth-only matrix: Transformer, TRM, DIS-TRM, and STRM all have
  seeds 0--2.
- STRM has additional depth-only seeds 3--5, recorded in
  `D1_SEED_STABILITY_RESULTS.md`.
- The seed-4 structural baseline comparison is complete but has exposed a
  weakness in the token-level split. A semantic, state-transition-disjoint
  replacement must be implemented before further structural seeds are useful.
