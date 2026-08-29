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
