# Research Plan and Literature

## Research question

Can a small weight-shared recursive model preserve correct structured state
over longer unseen operation sequences than parameter- and compute-accounted
non-recursive models, especially on official CCB D3 social logic?

## Scope correction: faithful TRM first, extensions second

The originating direction is to **apply TRM to CCB**. The primary experiment
must preserve the original CCB task definition, including fixed D1/D2 initial
states and sealed official records. Randomized initial grids and semantic
structural holdouts are valuable CCB-Learn stress tests, but they are not
replacements for the CCB task.

The current `VanillaTRM` is a useful *prefix-wise CCB adaptation* of the
published recursive update schedule, not yet a faithful TRM training
implementation. Published TRM repeatedly refines answer state `y` and latent
state `z` through an outer deep-supervision loop, detaches earlier refinements,
backpropagates through the final refinement, and uses EMA. Our current model
has shared `y/z` updates and detached warm-up cycles but lacks that outer
procedure and EMA. It must not be called a reproduction.

Correct hierarchy:

1. **Primary:** faithful CCB-TRM on the original fixed-input CCB task, trained
   only on generated official-semantics data and evaluated once on sealed
   official records.
2. **Baselines:** direct Transformer, recurrent sequence model, and
   compute-accounted CCB-TRM with the same serialized CCB input/output.
3. **Extension:** STRM as an explicit persistent-transition-state ablation of
   faithful CCB-TRM.
4. **Diagnostic extension:** randomized semantic D1 tests, labelled CCB-Learn.

## Proposed contribution

Applying vanilla TRM alone is not sufficient novelty. The paper-worthy study is
a controlled comparison of recursive-depth utility plus a persistent
State-Transition Recursive Model (STRM), trained on official-semantics generated
data and evaluated first on the untouched official CCB records, then on depth
and structural extensions.

STRM carries explicit answer state `y` and latent state `z` across operations.
At each operation it performs shared inner updates with initial-state and
operation recall before emitting the next full structured state. The decisive
comparison isolates persistence, weight sharing, recall, cross-cell mixing,
process supervision, and recurrent test-time compute.

## Required baselines

1. Exact oracle.
2. GRU and LSTM.
3. Direct and parameter/compute-matched Transformers.
4. Universal/looped Transformer.
5. CCB-adapted vanilla TRM.
6. Deep Improvement Supervision variant.
7. Fast-slow recurrence.
8. STRM and its ablations.
9. D3 message-passing GNN.

## Experiment order

1. Train only on generated official-semantics data; keep all 1,200 official
   records untouched.
2. Debug D1/D2 learning and generalization before interpreting D3.
3. Compare direct Transformer, TRM, STRM, and GNN in a small three-seed pilot.
4. Match parameter counts and report block evaluations, FLOPs, latency, and
   memory.
5. Test official depths 5-50, then extension depths 60/80/100.
6. Run structural holdouts and recurrence sweeps over
   `K ∈ {1,2,4,8,16,32}`.
7. Measure every loop's marginal improvement, overthinking, oscillation, and
   last useful update.
8. Run required ablations and only then scale final seeds/configurations.

## CPU foundation completed

The following prerequisites are implemented and tested before GPU spending:

1. Official evaluation firewall and zero-overlap assertions.
2. Definitive official-semantics train, validation, depth, and strong-depth
   splits with deterministic seeds and manifests.
3. Variable-depth padded data loading with exact masks.
4. Full overall/per-depth evaluation, `k*`, TFBC, retention, horizons,
   confidence intervals, AUC, validity, and loop diagnostics.
5. Reproducible model-by-seed experiment launching with resumable artifacts.
6. A faithful CCB adaptation of DIS discrete-diffusion targets and a separate
   all-gradient `dis_trm` baseline.

The next action is a CPU debug pilot on D1 and D2 with small configurations,
followed by a timed single-GPU pilot. The official test remains sealed until
the configuration and selection rule are frozen.

## Paper-worthiness gate

STRM must improve at least one primary depth measure—per-step retention, first
divergence, 50% success horizon, or depth AUC—against parameter- and
compute-accounted baselines, and the improvement must survive a structural
shift. Otherwise the defensible outcome is a controlled negative or diagnostic
study.

## Core literature

- CCB paper: https://arxiv.org/abs/2606.29278
- CCB code/data: https://github.com/Shubh-Chapra/Complexity_Ceiling_Benchmark
- Original TRM: https://arxiv.org/abs/2510.04871
- TRM code: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- TRM critical analysis: https://arxiv.org/abs/2512.11847
- Deep Improvement Supervision: https://arxiv.org/abs/2511.16886
- Deep Improvement Supervision code:
  https://github.com/machinestein/Deep-Improvement-Supervision
- Fast-slow latent recurrence: https://arxiv.org/abs/2604.01577
- Looped Transformers: https://arxiv.org/abs/2409.15647
- Stability in Looped Transformers: https://arxiv.org/abs/2604.15259
- Tiny Autoregressive Recursive Models: https://arxiv.org/abs/2603.08082
- Probabilistic TRM: https://arxiv.org/abs/2605.19943
- Energy-guided Recursive Model: https://arxiv.org/abs/2607.10128
- Universal Transformer: https://arxiv.org/abs/1807.03819

## Compute boundary

The exact generators, audits, metrics, tests, tiny overfit checks, and small
debug pilots run on the current 16-core/22-thread CPU with 32 GB RAM. Meaningful
multi-model, multi-seed training should use an NVIDIA CUDA GPU. The current
machine has CPU-only PyTorch in both Windows and WSL2, so GPU execution will
require a separate CUDA machine or cloud instance.

The upstream TRM repository reports roughly 18 hours for its smallest Sudoku
run on one L40S and multiple days for ARC on four H100s; those are reference
runtimes, not CCB estimates. CCB pilot runtime must be measured from our own
configs before purchasing a full allocation.

## Free-tier GPU pilot plan

Colab and Kaggle free GPUs are sufficient for the initial feasibility and
runtime pilot because the CCB models are small and the datasets are synthetic.
They are not dependable infrastructure for the final paper experiment matrix:
free GPU availability, accelerator type, and quota can change, and notebook
sessions can terminate before a long run finishes.

Kaggle is preferred for the first reproducible run because a saved notebook
version can execute from a clean environment for up to 12 hours and preserve
its output. Colab is useful for short interactive debugging, but its free GPU
limits and assigned hardware are dynamic and unpublished.

The free-tier sequence is:

1. Run D1 with Transformer, vanilla TRM, DIS-TRM, and STRM, one seed, width 64,
   four layers/loops, and 1,000 training steps.
2. Repeat the same pilot on D2.
3. Run D3 only after D1/D2 losses, masking, checkpoint restoration, and
   per-depth evaluation are confirmed.
4. Keep `include_official_evaluation=false`; use validation and generated
   depth-extension suites only.
5. Record assigned GPU, peak memory, examples/second, steps/second, and total
   runtime. Save periodic checkpoints and all resolved configuration files.
6. Use these measurements to size a stable A100/L40S allocation for the final
   three-to-five-seed baselines, ablations, and recurrence sweeps.

If each pilot finishes comfortably within one session, some three-seed pilot
runs can also use the free tier. Final claims should use consistent dedicated
hardware so runtime and compute comparisons are meaningful.

## Novelty audit (August 2026)

### What is already established

The broad ingredients in this project are not new in isolation:

- TRM uses a small weight-shared recursive network for reasoning
  ([Jolicoeur-Martineau, 2025](https://arxiv.org/abs/2510.04871)).
- Looped, weight-shared Transformer computation has already been studied for
  length generalization
  ([Giannou et al., 2024](https://arxiv.org/abs/2409.15647)).
- Persistent fast/slow latent recurrence, including carrying latent state across
  observations and performing several shared inner updates, is already directly
  studied for train-short/test-long generalization
  ([Fast-Slow Latent Recurrence, 2026](https://arxiv.org/abs/2604.01577)).

Therefore we must **not** claim that STRM is the first recurrent or
persistent-latent architecture, nor that weight sharing or iterative reasoning
is new.

### Plausible contribution, conditional on experiments

The potentially publishable unit is a controlled study of stateful recurrence
for *compositional state-transition reasoning*:

1. a leakage-audited CCB-Learn implementation with exact official-record
   compatibility and depth/structural holdouts;
2. trace-level evaluation (full answer exactness, transition accuracy,
   retention probability, and inferred reasoning horizon), rather than only a
   final task score;
3. a matched comparison among direct Transformer, vanilla TRM,
   intermediate-supervised recurrence, fast/slow recurrence, and STRM;
4. evidence that an explicitly decoded persistent answer state plus a latent
   state improves extrapolation on CCB, including on D2 and D3.

This is a hypothesis, not a completed novelty claim. We have not yet
established that no prior paper evaluated an equivalent stateful recurrence on
CCB, and a systematic related-work and repository audit is required before
using words such as “first”.

### Current evidence and its limit

The single-seed D1 scale run is a feasibility signal: STRM achieved 5.17%
final exact accuracy on held-out depth 25--50, versus 0% for the direct
Transformer and vanilla TRM, while the strong-depth 60--100 split remained
0%. It does not establish a stable result, SOTA, or a paper contribution. The
three-seed D1 confirmation, matched-compute ablations, structural holdout,
D2/D3, and comparison with the closest published fast/slow formulation remain
necessary.

### Seed-stability interpretation

The primary CCB-Learn train, validation, depth, and strong-depth splits are
built once before the model-by-seed loop, so all training seeds are evaluated
on the same deterministic instances. A difference between seeds therefore
reflects optimisation randomness (initial weights and minibatch order), not a
different held-out test set. Full-trajectory exactness compounds local errors:
a model can have useful transition accuracy but receive zero final exact credit
after one early wrong transition. Consequently, seed stability must be reported
using all seeds and confidence intervals; a single unusually high STRM seed is
not evidence of a reliable extrapolation gain.

### STRM seed-2 anomaly: current evidence and audit protocol

The D1 STRM seed-2 record reports 99.17% final exact accuracy on depths
25--50 and 92.33% on depths 60--100, whereas seeds 0 and 1 report 5.17% and
0.83% respectively. This is an anomaly to audit, not a result to claim.

Static code inspection establishes two useful safeguards:

1. The primary splits are built once before the model-by-seed loop, so the
   generated suites and manifest are independent of the optimization seed.
2. The STRM forward pass consumes only initial state and operations; targets
   are used only after the forward pass by the supervised loss and evaluator.

These checks rule out the most direct train/test or target-input leakage path,
but they do not establish that a new training run will reproduce the outcome.
The persisted-artifact audit did pass: all 12 result records and 12 matching
checkpoints exist under one manifest hash,
e5437b07e6aaa10d19b5c211e97ab0dcf7db21dfeb657c5d42eac87fbf561893.
The saved STRM seed-2 checkpoint reports step 10,000 with the intended
width-64/four-loop configuration, and an independent fresh evaluation of that
checkpoint exactly reproduced its stored held-out-depth final exact accuracy
(99.17%) and transition exact accuracy (99.69%).

Before reporting a method result, rerun seed 2 into a new output directory.
The rerun should use deterministic-algorithm settings where feasible and must
also include negative controls (permuted operations and labels) plus structural
holdout evaluation.

## Execution and storage policy after the D1 incident

### Immediate D1 sequence

1. Preserve the completed twelve-run records, configurations, manifest,
   ledger, and checkpoints as the D1 artifact set.
2. Run one fresh STRM seed-2 training replication in a new output directory.
   This is a training replication, not a checkpoint re-evaluation; the latter
   has already passed exactly.
3. If the fresh run again reaches the high-generalization regime, run at least
   two additional new STRM seeds to estimate the frequency of that regime.
   If it does not, report a bimodal or unstable optimization finding rather
   than a mean-only performance claim.
4. Run operation/label permutation controls and D1 structural holdout before
   claiming a method effect.
5. Only then start the D2 calibration seed. D3 follows only after D2 confirms
   masking, memory, runtime, and persistence behavior.

### Kaggle design

Use Kaggle only as compute and a short-lived durable workspace:

- For long unattended matrices, use a private saved Version/batch run rather
  than depending on an interactive draft session.
- For D2/D3, use two independent GPU-pinned worker processes, one job per T4,
  not distributed training of a small single model. This reduces wall-clock
  time but not total GPU quota.
- Each worker receives a unique run directory and writes its own resolved
  config, manifest reference, train log, checkpoint, and result JSON.
- A small append-only JSONL ledger and a compact completed-records JSON are
  updated atomically after each seed. These are the durability mechanism;
  zipping is optional.
- A ZIP, when desired for download, must be written outside the source
  directory and should exclude unnecessary source copies and prior ZIPs.

### Result retention

Keep compact scientific records in the Git repository: configuration,
manifest/hash, seed, hardware, metrics, and report tables. Keep large
checkpoints in a private Kaggle Output/Dataset or another private artifact
store, referenced by immutable run identifiers and checksums. Never rely on a
browser transcript as the sole copy of a result.

### Venue positioning

This work sits in the relatively specialised intersection of algorithmic
generalisation, recurrent/iterative reasoning, and mechanistic evaluation of
neural state transitions. A well-supported result is plausibly suitable for a
NeurIPS/ICLR workshop or a focused benchmark/method paper. A main-track claim
would require substantially broader and more robust evidence. It is not a
CVPR-shaped project after the Gaussian-splatting direction was removed.
