# Calibration Log

This file records completed, reproducible engineering calibrations. It is not
a CCB leaderboard, a SOTA claim, or a claim about the original CCB LLM
evaluation protocol.

## D1 upstream-derived TRM ACT calibration v1

- Status: completed successfully.
- Execution: private Kaggle Version 5 on Tesla T4 ×2.
- Source revision: `879be79630c374367f960b32baf001a89899a0ff`.
- Result artifact: `final_result.json`, SHA-256
  `eed3f02eb7df2184034a7cdeed6cfc0d7ae4f8aa68d544c89a0aa1dbcf28acb0`.
- Config: [`configs/d1_trm_act_calibration_v1.json`](../configs/d1_trm_act_calibration_v1.json).
- Run URL: <https://www.kaggle.com/code/prathamlahoti2/ccb-d1-trm-act-calibration-v1>.

### Integrity and execution

All 400 sealed official records in each of D1, D2, and D3 regenerated exactly
and their pinned data hashes validated before the run. The run used generated
CCB-TRM D1 data only; `include_official_evaluation` was false. It ran 1,000
optimizer updates for seed 17 with 47,618 trainable parameters and took 64.91
seconds of measured workload time.

### Result

| Split | Final exact | Transition exact | Element accuracy |
| --- | ---: | ---: | ---: |
| Validation, depths 5–20 | 0.00% | 5.12% | 29.32% |
| Held-out depth, 25–50 | 0.00% | 1.48% | 18.69% |
| Strong depth, 60–100 | 0.00% | 0.68% | 14.62% |

The strong-depth transition accuracies were 1.07% at depth 60, 0.63% at depth
80, and 0.50% at depth 100. The final training loss was 1.3452.

### Interpretation

This validates the complete execution path—official-record verification,
fixed-shape ACT training, copied-EMA evaluation, checkpointing, and all
requested evaluation depths. It **does not validate learning performance**:
zero validation final-exact accuracy means this configuration must not be used
for a TRM-versus-baseline claim. The next work is a bounded optimisation/
faithfulness diagnosis on tiny fixed data, before any multi-seed benchmark.

## D1 upstream-derived TRM fixed-data fit gate v1

- Status: completed successfully; **fit gate failed**.
- Execution: private Kaggle Version 1 on Tesla T4 ×2.
- Source revision: `bef670272f95b779cf8e79ce29e6c6f1d0eaa7fb`.
- Result artifact SHA-256:
  `e564b41ed39c502aeb78c80d548b25865c4524507754745efd7e1a6a37850845`.
- Config: [`configs/d1_trm_fit_gate_v1.json`](../configs/d1_trm_fit_gate_v1.json).

This trained on exactly 64 firewall-checked generated D1 examples at depth 5,
then evaluated those same examples. After 2,000 updates, live/EMA final-exact
accuracy was 4.69%/7.81% (3/64 and 5/64); EMA element accuracy was 80.38% and
transition-exact accuracy was 28.44%. The final ACT training record retained
zero halted rows.

This is not a generalisation result. It is a negative training-path diagnostic:
the present upstream-derived TRM/ACT configuration cannot memorise the frozen
set, so it must not be scaled or compared as a candidate method. The immediate
control is a fixed-data direct-Transformer fit gate using the same generated
episodes, seed, update count, batch size, and optimizer settings.
