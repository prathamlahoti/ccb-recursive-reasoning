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

Baseline structural runs will use the *same* D1 split, seed 4, width 64,
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

## Current evidence and remaining bounded D1 work

- Baseline depth-only matrix: Transformer, TRM, DIS-TRM, and STRM all have
  seeds 0--2.
- STRM has additional depth-only seeds 3--5, recorded in
  `D1_SEED_STABILITY_RESULTS.md`.
- The next run is the exact structural split above for Transformer, TRM, and
  DIS-TRM at seed 4. It answers comparative advantage before additional STRM
  structural replications.
