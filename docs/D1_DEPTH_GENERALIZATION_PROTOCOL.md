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

Until all six checks pass locally, the held-out-depth job is not GPU-ready.
