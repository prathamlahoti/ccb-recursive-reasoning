# Fixed-Data Fit Gate

The fit gate is the first prerequisite for a CCB-TRM benchmark. It trains on
one frozen, firewall-checked set of 64 generated D1 examples at depth 5 and
evaluates on those exact same examples. It is **not** a generalisation or CCB
leaderboard experiment.

The current eligible gates are the sequential v2 ladder in
[`CORRECTED_FIT_LADDER.md`](CORRECTED_FIT_LADDER.md). They use
`official_trm_ccb`: a narrow CCB I/O boundary over
the verified official no-puzzle TRM core, released stablemax/ACT loss equation,
mathematically matched unfused AdamATan2, released warmup/cosine scheduler and
copied EMA evaluator. It saves a checkpoint every 100 updates. Passing means
near-perfect final-exact accuracy on the fixed set; failure means diagnosis,
not a larger benchmark.

Run locally or on GPU with:

```text
python -m ccb fit-gate --config configs/d1_official_trm_fit_8x1_v2.json
```

The direct-Transformer control uses
[`configs/d1_transformer_fit_gate_v1.json`](../configs/d1_transformer_fit_gate_v1.json):
the same fixed examples, seed, optimizer settings, batch size, and update
budget. It uses four unshared encoder blocks so that it is a sufficiently
capable pipeline control; it is not presented as a parameter- or
compute-matched benchmark baseline.

The following are completed historical diagnostics of the rejected
`trm_upstream_core` adapter, retained only to explain the fidelity audit. The
next diagnostic was
[`configs/d1_trm_forced_halt_fit_gate_v1.json`](../configs/d1_trm_forced_halt_fit_gate_v1.json).
It keeps the same TRM core and frozen data but forces one outer ACT step, so
every update starts from a fresh recurrent state. It tests carry/halting
semantics, not generalisation.

The core-isolation diagnostic is
[`configs/d1_trm_one_step_supervised_fit_gate_v1.json`](../configs/d1_trm_one_step_supervised_fit_gate_v1.json).
It preserves the same recursive core and fixed data, but uses one fresh
recurrence step, ordinary supervised token loss, and one-step evaluation. It
does not claim to be the released TRM training procedure; it localizes whether
the core/adapter can learn independently of the ACT/stablemax/halting path.
