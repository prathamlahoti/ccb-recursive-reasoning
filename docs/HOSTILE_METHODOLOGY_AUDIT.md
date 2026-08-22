# Hostile Methodology Audit (2026-08-23)

This audit assumes the previous large D1 gap is wrong until supported by a
specific check. It separates verified mechanics from invalidated claims and
unverified artifacts. No new GPU experiment should be interpreted before the
requirements at the end of this document are met.

## Verdict

The project has a sound oracle/encoding/training *foundation*, but the prior
D1 results do **not** establish the intended research claim. They remain
useful as implementation diagnostics and as evidence that STRM can fit and
execute the old generator, but they are not evidence of semantic
compositional-generalization superiority.

The audit does **not** recommend replacing CCB with randomized D1. Original
CCB remains the primary TRM application; the randomized semantic suite is a
separate diagnostic extension that protects against overclaiming.

## Verified

| Check | Result | Evidence |
|---|---|---|
| Official oracle compatibility | Pass | `ccb verify-official` regenerated all 400 records in each of D1, D2, and D3 with zero mismatches. |
| Target leakage into D1 forward pass | Pass | Automated target-mutation test shows unchanged logits for the direct Transformer, vanilla-TRM adaptation, and STRM. |
| Trace supervision | Pass | Targets are full oracle states after each operation; padding is masked in the loss and evaluator. |
| Raw output evaluation | Pass | Reported D1 scores use raw argmax rather than a constraint solver repairing predictions. |
| Repaired suite firewall | Pass | The new semantic D1 generator rejects official-seed, program, and instance overlaps. |
| Repaired semantic exclusion | Pass | Train/validation exclude every contiguous program segment whose net D1 permutation equals the reserved composition. |
| Repaired transition overlap | Pass | The new suite rejects exact `(before, operation, after)` transition reuse across every split. |

## Invalidated or insufficient claims

| Previous claim or assumption | Audit finding | Consequence |
|---|---|---|
| “Unseen operation pair” means unseen transformation | False | `TRANSPOSE_GRID -> FLIP_HORIZONTAL -> SHIFT_ROW_2_LEFT` has the same net transformation as `ROTATE_90_CW -> SHIFT_ROW_2_LEFT`. The old split is token-level only. |
| Fixed initial grid tests general algorithm execution | False | Original D1 always starts from the same grid. It tests program execution on one orbit, not input-grid generalization. |
| Old STRM-vs-baseline gap is a paper result | Insufficient | It is one seed per architecture on a weak split and does not use a compute-matched protocol. |
| `VanillaTRM` is an official TRM reproduction | False | It is a CCB prefix-wise adaptation of the recursive update schedule. It must be described only as an adaptation baseline. |
| Old Kaggle artifact proves the current codebase result | Insufficient | The notebook used the private source dataset snapshot `v54bd91d`; repaired split code is only in Git commit `553db17` and has not yet been run on Kaggle. |

## Why the old D1 task can yield implausibly high numbers

D1 has seven primitive operations acting on a finite 3x3 permutation state.
Operation strings rapidly collapse to the same endpoint transformations:

| Program depth | Strings | Distinct endpoints | Strings per endpoint |
|---:|---:|---:|---:|
| 5 | 16,807 | 2,117 | 7.94 |
| 8 | 5,764,801 | 70,316 | 81.98 |
| 10 | 282,475,249 | 314,588 | 897.92 |

This algebraic compression makes token-level train/test separation much weaker
than it appears. A recurrent model with a state-like inductive bias can exploit
the one-step transition structure much more directly than a prefix predictor.
That outcome may be real, but it is not surprising enough to establish a new
reasoning method without the repaired controls.

## Remaining unknowns

1. Whether STRM succeeds on the repaired randomized, semantic suite.
2. Whether the repaired result is stable across seeds.
3. Whether baselines remain weak under parameter- and compute-accounted
   tuning, rather than only one shared hyperparameter setting.
4. Whether the effect extends beyond D1 to D2 and D3.
5. Whether an independent checkpoint evaluator reproduces every new Kaggle
   result from a saved commit, manifest, and checkpoint.

## Non-negotiable protocol for the next result

Before the semantic-suite protocol below, implement and validate a faithful
CCB-TRM training loop: outer deep answer/latent refinement with the published
detach schedule and EMA. Its official-CCB result is the primary answer to the
assignment; the semantic suite remains a stress test.

1. Upload only commit `553db17` or later as a new private Kaggle source
   Dataset version; record its Git commit in every result.
2. Generate the semantic suite once with
   `ccb generate-structural --domain d1 --semantic-d1`; save its manifest and
   train/evaluate every model from that exact serialized data, not regenerated
   splits inside each worker.
3. Run STRM seed 4 as a calibration only. Save code commit, manifest hash,
   optimizer/configuration, log, checkpoint, and result before interpreting it.
4. Independently load that checkpoint in a clean process and reproduce metrics.
5. If calibration passes, run STRM seeds 3 and 5 and the baselines on the same
   manifest. Report every seed, including failures.
6. Treat D2/D3 as independent replications, not as follow-up polish.

Until this protocol completes, the correct label for the prior D1 results is
**debugging evidence from a superseded token-level split**.
