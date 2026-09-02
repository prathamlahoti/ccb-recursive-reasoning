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

## D1 direct-Transformer fixed-data fit gate v1

- Status: completed successfully; **control passed**.
- Execution: private Kaggle Version 1 on Tesla T4.
- Source revision: `6a3cfea37bff17209126528d5bd212e8e812805e`.
- Result artifact SHA-256:
  `8001c585b7033d9aa36def85d4ef09641b5a8418e07649e6b630944e8148300d`.
- Config: [`configs/d1_transformer_fit_gate_v1.json`](../configs/d1_transformer_fit_gate_v1.json).

On the exact same 64 firewall-checked D1 depth-5 examples, with the same data
seed, training seed, batch size, 2,000-update budget, and AdamW settings, the
four-block direct Transformer reached 100% element, transition-exact,
final-exact, and trace-exact training-set accuracy. Its final loss was 0.00858.

This is a pipeline control, not a benchmark comparison. It localizes the
failed TRM fit gate to the current TRM/ACT adaptation rather than generated
D1 data, CCB serialization, target construction, or the general fit-gate
trainer. The next isolated test forces ACT to halt after one outer step.

## D1 upstream-derived TRM forced-halt fit gate v1

- Status: completed successfully; **fit gate failed**.
- Execution: private Kaggle Version 1 on Tesla T4.
- Source revision: `dfebe7e4c35672ee1eca3c057aa4b2ed651f3bac`.
- Result artifact SHA-256:
  `b47b9e4fd9c5064bf1acc7f3068599593f9a9cc59ead00b684fb64b3018922bb`.
- Config: [`configs/d1_trm_forced_halt_fit_gate_v1.json`](../configs/d1_trm_forced_halt_fit_gate_v1.json).

This held the same frozen 64 depth-5 examples, seed, core, and training budget
constant but forced every ACT row to halt after one outer step. It did halt all
rows (`act_halted_fraction = 1.0`), yet final-exact fit was only 6.25% live and
3.13% EMA (4/64 and 2/64); EMA transition-exact fit was 26.56% and element
accuracy 76.28%.

Therefore persistent ACT carry is not the explanation for the failed original
fit gate. The next bounded diagnostic should test the core recurrence with a
single-step, non-ACT supervised loss and matching one-step evaluation. If that
fits, the defect is in the ACT/stablemax/halting adaptation; if not, it is in
the current core/representation adaptation. No benchmark-scale run is justified
until this boundary is resolved.

## D1 upstream-derived TRM one-step supervised fit gate v1

- Status: completed successfully; **fit gate failed**.
- Execution: private Kaggle Version 1 on Tesla T4.
- Source revision: `33fa60f8abb0f4f3ceafef70e5373d175b506b46`.
- Result artifact SHA-256:
  `63b3f3f87441fadc537ba2ef2bca4b181354aa703d2547e7fcd3976dab341ee3`.
- Config: [`configs/d1_trm_one_step_supervised_fit_gate_v1.json`](../configs/d1_trm_one_step_supervised_fit_gate_v1.json).

This used the same core and frozen 64 depth-5 examples, but removed ACT carry,
halt BCE, and stablemax from training: each update used one fresh recursive
step with ordinary supervised token loss, and evaluation also used one step.
It reached only 3.13% final-exact fit (2/64), 20.63% transition-exact fit, and
75.38% element accuracy after 2,000 updates.

Together with the passed direct-Transformer control, this localizes failure to
the present CCB adaptation of the TRM core or its basic optimization setup—not
to CCB data, target serialization, ordinary supervised training, ACT carry,
halting, or stablemax. GPU experiments stop here. The required next step is a
line-by-line fidelity audit against the pinned official TRM implementation
before any new tuning or benchmark run.

## D1 official-primitive TRM fixed-data fit gate v1

- Status: completed successfully; **fit gate failed**.
- Execution: private Kaggle Version 2 on two Tesla T4 GPUs; 134.4 seconds to
  durable result save.
- Source revision: `f64fce4b` (the verified official-core integration).
- Result artifact SHA-256:
  `54583884851bf11f03306308d6db072b2c952b76a025016b7e1f652e3317295e`.
- Config: [`configs/d1_official_trm_fit_gate_v1.json`](../configs/d1_official_trm_fit_gate_v1.json).

This is the first gate using `official_trm_ccb`, not the rejected
`trm_upstream_core` adapter. It used the verified no-puzzle official primitive
layers and ACT wrapper, target-free CCB token boundary, released stablemax
loss equation, mathematically matched unfused AdamATan2 with `(0.9, 0.95)`,
weight decay `0.1`, a 200-step warm-up/cosine schedule, and copied EMA.

After 2,000 updates on the same frozen 64 firewalled depth-5 D1 examples, the
live model achieved 29.93% element accuracy but 0% transition, final-exact,
and trace-exact accuracy. EMA reached 22.81% element accuracy and likewise 0%
on all exact metrics. Loss decreased from 2.369 to 1.826, while the ACT state
reached the configured 16-step horizon.

This is a valid negative result for this **CCB adaptation and configuration**.
It does not refute released TRM on its native puzzle data, nor establish a CCB
benchmark. No depth-generalization, official-record, or multi-seed run is
authorized. The next work must be a narrow diagnosis of the CCB output/readout
representation and training objective, with a small baseline-preserving test
before spending more GPU time.

Post-run audit: this v1 gate used overlapping state/operation token IDs, used
valid token `0` for both query and padding roles, and set `lr_min_ratio=0.0`
instead of the released `1.0`. It is therefore retained as a negative result
for the v1 adapter only. It is not evidence that the corrected TRM port cannot
fit CCB. The replacement protocol is documented in
[`CORRECTED_FIT_LADDER.md`](CORRECTED_FIT_LADDER.md).

## D1 corrected Gate A — paired T4 float32 diagnostic v2

- Status: completed successfully; **both fit controls passed**.
- Execution: private Kaggle Version 1, one process per Tesla T4 GPU.
- Verified model-core commit: `6fbcf8453940e9d5b27833a00f5c4f8503ef468d`.
- Wall time: 151.80 seconds.
- Compact result: [`results/d1_gate_a_t4_float32_v2/final_summary.json`](../results/d1_gate_a_t4_float32_v2/final_summary.json).

Both models trained on the same eight firewalled D1 examples at depth 1, with
the collision-free token representation, seed 17, data seed 90000000, width
512, two layers, AdamATan2, 2,000-step warm-up/update budget, and live-weight
evaluation. Both began at 0% transition-, final-, and trace-exact accuracy and
finished at 100% element, transition-exact, final-exact, and trace-exact
accuracy. TRM loss fell from 2.41781 to 0.01685; matched token-Transformer loss
fell from 2.57729 to 0.000143. The TRM finished with all rows halted and 1.125
mean ACT steps.

This clears the basic corrected I/O, optimizer, loss, and training-path gate.
It does not measure held-out generalization or establish a TRM advantage. In
particular, depth 1 does not substantially exercise recursive computation, and
this run substituted float32 for the released bfloat16 precision because T4
lacks native bfloat16. Gate B (16 fixed examples at depth 5) is now eligible as
the next diagnostic; a released-precision result still requires suitable
native-bfloat16 hardware.

## D1 corrected Gate B — paired T4 float32 diagnostic v2

- Status: completed successfully; **Transformer passed, TRM failed the
  predeclared live/max-16 rule**.
- Execution: private Kaggle Version 1, one process per Tesla T4 GPU.
- Verified model-core commit: `6fbcf8453940e9d5b27833a00f5c4f8503ef468d`.
- Wall time: 857.43 seconds.
- Compact result: [`results/d1_gate_b_t4_float32_v2/final_summary.json`](../results/d1_gate_b_t4_float32_v2/final_summary.json).
- Recurrence sweep: [`results/d1_gate_b_t4_float32_v2/trm_recurrence_sweep.json`](../results/d1_gate_b_t4_float32_v2/trm_recurrence_sweep.json).

The matched token-Transformer memorized all 16 depth-5 traces exactly. At the
predeclared 16-step live-weight evaluation point, TRM reached 94.03% element,
70.00% transition-exact, 68.75% final-exact, and 31.25% trace-exact accuracy.
Therefore Gate B did not pass and Gate C must not run.

A post-hoc recurrence sweep, reported separately rather than replacing the
predeclared result, showed that repeated inference degraded the live model:
step 1 had 100% final and 93.75% trace exactness, while step 16 had 68.75% and
31.25%. The saved EMA model reached 100% final/trace exactness at step 1 and
100% final/87.5% trace exactness at step 16. Thus the corrected TRM learned
almost all of the tiny fixed set, but it had not learned a stable fixed point
under the released 16-step evaluation horizon.

The upstream repository's reported experiment commands set `ema=True`, and
its evaluator explicitly runs all rows for `halt_max_steps=16`. Consequently,
the faithful upstream-style reading of this checkpoint is the EMA step-16
result—not the perfect post-hoc EMA step-1 result. This still fails the 99%
trace gate (14/16 traces). The earlier decision to make live weights the sole
faithful gate was too strict relative to the reported upstream commands and
must be corrected before another TRM fit run. This correction does not turn
Gate B into a pass and does not authorize Gate C.

## D1 corrected Gate B — TRM-only 10K EMA diagnostic v3

- Status: completed successfully; **Gate B passed**.
- Execution: private Kaggle Version 1 on Tesla T4.
- Verified model-core commit: `6fbcf8453940e9d5b27833a00f5c4f8503ef468d`.
- Wall time: 1,934.37 seconds (32.24 minutes).
- Compact result: [`results/d1_gate_b_trm_10k_t4_float32_v3/final_summary.json`](../results/d1_gate_b_trm_10k_t4_float32_v3/final_summary.json).
- Recurrence sweep: [`results/d1_gate_b_trm_10k_t4_float32_v3/trm_recurrence_sweep.json`](../results/d1_gate_b_trm_10k_t4_float32_v3/trm_recurrence_sweep.json).

The predeclared upstream-style EMA/max-16 gate reached 100% element,
transition-exact, final-exact, and trace-exact accuracy on all 16 fixed
depth-5 examples. A post-run sweep confirmed that EMA retained 100% on all
four metrics at every recurrence step from 1 through 16, so this is a stable
fit rather than a lucky evaluation step. Live weights remained less stable
(87.5% final and 62.5% trace exactness in the recorded gate result), confirming
that the pass specifically relies on the EMA evaluation used by the upstream
reported experiments.

This establishes only fixed-set optimization correctness. It is neither a
generalization result nor evidence that TRM beats the matched Transformer.
Gate C—the paired 64-example depth-5 fit check—is now eligible.
