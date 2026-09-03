# D1 Held-Out-Depth Protocol

Status: preregistration draft after completion of the corrected fixed-set fit
ladder. No held-out-depth GPU run has been launched.

## Question

Given an initial D1 state and the complete ordered operation sequence, does the
verified TRM port retain exact transition accuracy at unseen greater depths
better than a nonrecursive Transformer receiving exactly the same tokens?

This is a CCB-TRM supervised extension experiment, not an original CCB
leaderboard result. The official 400 D1 records remain sealed.

## Frozen generated splits

| Split | Depths | Examples per depth | Seed base | Role |
| --- | --- | ---: | ---: | --- |
| Train | 5, 10, 15, 20 | 100 | 10,000,000 | optimization |
| Validation | 5, 10, 15, 20 | 25 | 20,000,000 | model selection only |
| Held-out depth | 25, 30, 35, 40, 45, 50 | 100 | 30,000,000 | primary generalization result |

Depths 60, 80, and 100 are excluded from the first experiment. They are a
secondary stress test and would force a much longer fixed token canvas. The
official D1 records are excluded from training, validation, and initial test
iteration by the existing firewall.

## Models and fairness

- `official_trm_ccb`: width 512, two shared L layers, H/L cycles 3/6, ACT
  horizon 16, AdamATan2, and the verified collision-free CCB boundary.
- `ccb_token_transformer`: width 512 and two nonrecursive layers, consuming
  exactly the same token tensor and output positions.
- At D1 maximum depth 50 these have 6,838,274 and 6,326,272 trainable
  parameters respectively, so this is a parameter-matched comparison within
  8.1%, not a compute-matched comparison.

TRM executes 42 transformer-block applications in each inner forward step
(including the no-gradient cycles), while the control executes two. Runtime,
peak memory, parameters, optimizer updates, and token counts must therefore be
reported. A future compute-matched control is a separate ablation; the current
two-layer control must never be called compute-matched.

Both models must maintain EMA weights with the same decay and report both live
and EMA results. EMA is the predeclared primary weight source for both models,
preventing TRM from receiving an evaluation advantage that the control lacks.

## Optimization contract

- Training seed: 17 for the single-seed calibration.
- Width 512, eight heads, and two layers.
- 10,000 optimizer updates initially.
- AdamATan2, betas `(0.9, 0.95)`, weight decay `0.1`.
- Learning rate `1e-4`, 2,000-update warm-up, `lr_min_ratio=1.0`.
- ACT H/L cycles 3/6, horizon 16, exploration probability 0.1.
- T4 runs are explicitly float32 diagnostics. A native-bfloat16 confirmation
  is required before a final released-contract result.

The first run is a seed-17 calibration. Multi-seed work is authorized only if
the pipeline completes and at least one model has nontrivial held-out-depth
accuracy. Test results cannot be used to tune the completed run.

## Primary outcomes

For validation and held-out depth, report final-state exact accuracy,
complete-trace exact accuracy, transition exact accuracy, element accuracy,
per-depth exact accuracy, per-step retention, success horizons, normalized
depth AUC, valid-state rate, and TFBC with confidence intervals.

## Required launcher corrections before execution

1. Add identical copied-EMA training and evaluation to the Transformer path.
2. Persist final live model, optimizer, EMA, ACT carry where applicable, RNG
   state, and dataset-manifest hash without overwriting EMA at completion.
3. Separate training and evaluation batch sizes; depth-50 evaluation needs a
   smaller batch on T4.
4. Exclude `test_strong` and official records from this first run.
5. Record peak CUDA memory, wall time, token-canvas length, and model
   parameters.
6. Run a short memory/runtime calibration before choosing a full-run batch
   size. Do not reduce the scientific split or silently change precision to
   make the job fit.

The launcher corrections are implemented and locally verified. The private
`ccb-d1-depth-50-resource-calibration-v1` Version 1 run on 2026-09-03 tested
batch sizes 1, 2, 4, and 8 and saved partial results after every successful
batch size. The TRM worker completed all four sizes. Its batch-8 training step
used 2.517 GiB and 0.784 seconds; its full max-16 evaluation used 0.232 GiB
and 6.793 seconds. The Transformer worker failed before its first measurement
because the calibration wrapper omitted the required explicit
`loop_supervision_weight=0.0` argument to `supervised_loss`. This was a wrapper
failure, not an out-of-memory or model failure. Version 2 is limited to the
corrected Transformer worker; the successful Version 1 TRM data are retained.
Version 2 completed successfully and supplied all Transformer measurements.
The complete batch-8 calibration is:

| Model | Train step | Train peak | Evaluation batch | Evaluation time | Evaluation peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official TRM CCB | 0.784 s | 2.517 GiB | 8 | 6.793 s | 0.232 GiB |
| Token Transformer | 0.096 s | 0.396 GiB | 8 | 0.014 s | 0.112 GiB |

Batch-size-1 timings are discarded for throughput estimation because they
include CUDA warm-up. No accuracy conclusion is drawn from calibration because
the models were untrained. The preregistered seed-17 run will use training
batch size 8 and evaluation batch size 8 for both models. This holds batch size
constant across models, fits conservatively on each T4, and should finish in
roughly 2.5--3 hours when the two workers run concurrently. The resource gate
is passed.

## Execution plan

The first held-out-depth run uses two isolated workers, with TRM pinned to GPU
0 and the token Transformer pinned to GPU 1. Each worker receives its own
output directory, atomically replaced checkpoint every 250 updates, training
ledger, dataset manifest, final EMA checkpoint, primary EMA evaluation, and
secondary live-weight evaluation. The controller reports progress periodically
and writes a compact final summary. It does not create a duplicate ZIP archive,
which avoids the earlier Kaggle disk-exhaustion failure. Official evaluation is
disabled and `test_strong` is absent from the run manifest and evaluations.

Version 1 of the private Kaggle kernel
`prathamlahoti2/ccb-d1-depth-generalization-seed17-float32-v1` was submitted on
2026-09-03 from pinned source commit
`9726ab7f57883cf04570e5a50e30b10b7955c4bc`. Kaggle reported the Version run
as running after submission. Its outputs are calibration evidence until the
predeclared multi-seed gate is met; they are not final paper numbers.

The first submission failed before optimizer update 1 because the general
launcher created both EMA copies on CPU and only then moved the live models to
CUDA inside the trainers. Both workers therefore rejected their first EMA
update. This is an execution-order bug, not an experimental outcome. The fix
moves each live model to its configured device before optimizer and EMA
construction and adds an explicit cross-device EMA invariant plus regression
test. Version 1 contributes no accuracy or runtime result.

Version 2 was submitted on 2026-09-03 from corrected source commit
`25f2c3579d726d912e787242ec01997f65d94f81`. Before submission, all 78 local
tests passed, including both training paths and a regression check that rejects
cross-device EMA updates before tensor arithmetic. Kaggle reported Version 2
as running after submission.

## Seed-17 outcome

Version 2 completed after 9,781 seconds (2.72 hours). Both workers reached all
10,000 updates, both final checkpoints and ledgers were retained, and the two
manifests have the identical hash
`f589979a22625a7d0ba68c05194d663fac5013d27c7e732f6c92537593e49167`.
The manifests contain only train, validation, and `test_depth`; all generated instances and
program fingerprints are disjoint, all official-firewall checks pass, and
official evaluation was not run.

EMA is the predeclared primary result:

| Split / metric | Official TRM CCB | Token Transformer |
| --- | ---: | ---: |
| Validation final exact | 0.00% | 1.00% |
| Validation trace exact | 0.00% | 0.00% |
| Validation transition exact | 12.40% | 22.48% |
| Validation element accuracy | 36.92% | 45.43% |
| Validation mean first divergence | 2.53 | 3.74 |
| Depth 25--50 final exact | 0.00% | 0.00% |
| Depth 25--50 trace exact | 0.00% | 0.00% |
| Depth 25--50 transition exact | 3.85% | 5.78% |
| Depth 25--50 element accuracy | 21.57% | 21.92% |
| Depth 25--50 valid-state rate | 4.94% | 6.15% |
| Depth 25--50 mean first divergence | 2.41 | 3.13 |

Held-out EMA transition exact accuracy by depth:

| Depth | Official TRM CCB | Token Transformer |
| ---: | ---: | ---: |
| 25 | 5.92% | 10.84% |
| 30 | 5.13% | 9.40% |
| 35 | 4.40% | 7.74% |
| 40 | 3.62% | 6.12% |
| 45 | 3.33% | 4.69% |
| 50 | 2.30% | 0.40% |

The Transformer finished its worker in 743 seconds using 0.469 GiB peak CUDA
memory. TRM required 9,764 seconds and 2.596 GiB, consistent with 42 versus two
transformer-block applications per inner forward. EMA modestly improved both
models, so weight-source choice does not explain the negative result.

This is a valid negative calibration, not a TRM advantage. Both models fail at
new examples even within the training depth range; the Transformer is stronger
than TRM on aggregate validation and held-out transition accuracy, while both
score zero held-out final/trace exact. The final training losses (TRM 1.162;
Transformer 0.544) are also far from the fixed-set fit-gate regime. Therefore
multi-seed replication, D2/D3 transfer, and official evaluation are blocked.

The next bounded diagnostic is evaluation of the saved live and EMA checkpoints
on the exact 400 training episodes, without additional optimization. High train
accuracy with low validation indicates memorization/data insufficiency; low
train accuracy indicates optimization or training-contract failure. That
diagnostic must precede any new training run.
