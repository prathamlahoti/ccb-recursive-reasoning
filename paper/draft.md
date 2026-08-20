# Beyond the Complexity Ceiling: Persistent State-Transition Recurrence for Depth-Generalizable Reasoning

## Abstract (provisional, no result claims)

Sequential reasoning systems can answer short state-tracking problems while
failing abruptly as the number of required transitions grows. We construct
CCB-Learn, a reproducible derivative of the Complexity Ceiling Benchmark with
deterministic generators, exact intermediate states, depth and structural
holdouts, and leakage audits. We use it to compare direct, recurrent,
weight-shared, and graph-structured models under parameter- and
compute-accounted evaluation. We introduce the State-Transition Recursive
Model (STRM), which carries an explicit structured answer state and persistent
latent state across operations while performing shared inner refinement with
input recall. The study is designed to test whether persistent recurrence
improves per-transition retention and unseen-depth success horizons, rather
than merely increasing test-time compute. Numerical claims will be inserted
only after multi-seed GPU experiments are complete.

## 1. Benchmark

### 1.1 Provenance

CCB-Learn is derived from the D1 Alien Grid, D2 Symbolic Pointers, and D3 Social
Logic task families introduced by the Complexity Ceiling Benchmark (CCB). The
paper alone does not specify every generator detail. The authors' official
repository provides D1-D3 implementations and fixed evaluation records. We
therefore evaluate official CCB compatibility separately from deterministic
CCB-Learn extensions and do not describe results on generated extension data as
reproduction of original CCB model scores.

### 1.2 Deterministic transition systems

D1 represents a 3 by 3 permutation and applies the seven official whole-grid,
row, column, and corner transformations. D2 represents seven digit-valued
registers and applies the five official shift, swap, modular-sum, and
modular-difference operations; SET operations may create duplicate values. D3
represents signed undirected relations; alliance has sign +1, rivalry -1, and
consistent transitive closure follows sign multiplication.
Each episode stores its initial state, ordered operations, exact state after
every operation, final state, seed, semantics identifier, and specification
version. Every serialized episode is replayed through its oracle before being
written.

### 1.3 Depth regimes

The primary training set spans depths 1-20. Primary extrapolation tests use
depths 25, 30, 35, 40, 45, and 50; strong tests use 60, 80, and 100. An
official-evaluation firewall rejects any candidate sharing an official
seed/depth, canonical program, or complete instance. Generated train/validation
overlap is reported rather than claimed absent: fixed-start D1 and D2 have only
seven and five possible depth-1 programs, making shallow in-distribution
separation impossible at useful sample counts. Depth-extrapolation suites
occupy distinct depths.

### 1.4 Structural regimes

Separate held-out suites isolate structural generalization. D1 withholds the
ordered composition `ROTATE_90_CW` followed by `SHIFT_ROW_2_LEFT`; both
primitives remain available individually. D2 withholds the ordered composition
`SET_D_TO_A_PLUS_B` followed by `SET_C_TO_G_MINUS_E`. D3 trains on six- and
eight-agent chain-topology updates and tests
ten- and twelve-agent star-topology updates. Exposure counters are stored in
the manifest and required to be zero for training and positive for testing.

### 1.5 Leakage and shortcut audits

We report cross-split instance and program overlap, operation n-gram overlap,
endpoint entropy, maximum endpoint frequency, identity rate, and structural
feature exposure. D1 additionally receives an exact reachability audit because
exponentially many operation strings collapse onto a finite set of grid
permutations. Models receive no puzzle or example identifier embeddings.

### 1.6 Evaluation

Principal measures are final exact accuracy, full-trace exact accuracy,
per-transition exact accuracy, state-element accuracy, validity rate, first
divergence, transition failure before correctness, geometric per-step
retention, 50% and 90% success horizons, and normalized accuracy-depth AUC.
Accuracy intervals use the exact Clopper-Pearson procedure used by CCB;
retention uncertainty uses deterministic parametric bootstrap intervals.
Models are compared on paired examples and multiple training seeds.

## 2. Method

### 2.1 Problem formulation

Let `s_0` be an initial structured state and `o_1, ..., o_T` an ordered program.
The target transition system produces `s_t = f(s_(t-1), o_t)`. A learned model
predicts every `s_t`, not only `s_T`, enabling direct process supervision and
identification of the first incorrect transition.

### 2.2 State-Transition Recursive Model

STRM maintains an explicit answer representation `y_t` and latent state `z_t`.
At outer step `t`, an operation embedding and initial-state recall are combined
with the previous answer. A shared inner update runs `K` times:

```text
y_0 = Embed(s_0),  z_0 = 0
r_t = Recall(MixCells(y_(t-1)), Embed(o_t), Embed(s_0))
z_t^0 = z_(t-1)
z_t^(k+1) = Norm(MixCells(GRU_z(r_t, z_t^k)))
y_t = Norm(MixCells(GRU_y(z_t^K, y_(t-1))))
prediction_t = Decode(y_t)
```

Cross-cell mixing is necessary: cell-independent updates cannot represent grid
rotations, register permutations, or relational propagation. All inner updates
share parameters. Initial-state and operation recall are injected at every
outer transition, while `y` and `z` persist across the entire program.

### 2.3 Training objective

The portable objectives include final/process cross-entropy and conventional
deep supervision of inner-loop readouts:

```text
L = L_process(final inner readout)
  + lambda_loop * mean_k L_process(inner readout k)
```

We separately adapt Deep Improvement Supervision (DIS). Starting from the
initial state repeated across the target trace, a discrete-diffusion schedule
reveals a nondecreasing subset of oracle cells at each recursive loop; the last
target is the complete oracle trace. Unlike vanilla TRM, DIS-TRM allows
gradients through every refinement loop. It is reported as a distinct baseline.

Validity-constrained decoding enforces a permutation for D1 and symmetric
relations with valid diagonals for D3; D2 uses unconstrained digit decoding as
required by the official task. Validity loss and improvement-specific
objectives are ablations and will not be included in the headline method
unless they improve held-out results.

### 2.4 Baselines

The shared interface includes a direct causal Transformer, GRU, LSTM, looped
Transformer, fast-slow recurrence, D3 message-passing GNN, and CCB-adapted
vanilla TRM. The TRM baseline preserves the official shared high/low reasoning
block, repeated low cycles, no-gradient early high cycles, and final-cycle
gradient. It removes puzzle identifiers and predicts each causal prefix
independently, so it does not inherit STRM's persistent transition state.

### 2.5 Fair comparison

Every result reports trainable parameters and architecture-specific block
evaluations. Primary comparisons include parameter-matched and update-matched
variants. Fixed single-trajectory inference is primary; stochastic voting is a
separate breadth-versus-depth experiment. Recursive models export predictions
after every loop to measure marginal improvement, overthinking, and the last
useful recurrent update.

## 3. Current validation and result embargo

All implementations pass deterministic oracle, official-firewall, metric,
serialization, structural-split, forward/backward, and tiny-set overfit tests
on Windows and WSL2 CPU environments (56 tests in each). The complete CPU
runner also builds the definitive suite, trains, evaluates all generated depth
regimes, and writes reproducible artifacts. This establishes implementation correctness but is not
scientific evidence for depth generalization. Benchmark comparison tables,
confidence intervals, and claims remain intentionally blank until controlled
multi-seed experiments are run on CUDA hardware.

## Limitations (provisional)

CCB-Learn is distinct from the now-public official CCB generator. Its D3
semantics are an explicit research choice. Synthetic state spaces can contain algebraic
shortcuts even after auditing. Structural holdouts test selected forms of
generalization rather than all distribution shifts. Compute units across
attention, recurrence, and message passing are not equivalent to measured
FLOPs; both must be reported in final experiments. Finally, tiny-set
memorization only detects broken implementations and says nothing about model
ranking.
