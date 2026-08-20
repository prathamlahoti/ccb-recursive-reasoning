# Research Plan and Literature

## Research question

Can a small weight-shared recursive model preserve correct structured state
over longer unseen operation sequences than parameter- and compute-accounted
non-recursive models, especially on official CCB D3 social logic?

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
