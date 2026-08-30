# Fixed-Data Fit Gate

The fit gate is the first prerequisite for a CCB-TRM benchmark. It trains on
one frozen, firewall-checked set of 64 generated D1 examples at depth 5 and
evaluates on those exact same examples. It is **not** a generalisation or CCB
leaderboard experiment.

The TRM gate uses the active upstream-derived ACT/stablemax/EMA path. It saves
a checkpoint every 100 updates and records both the live model and its copied
EMA evaluator. Passing means near-perfect final-exact accuracy on the fixed
set; failure means the next action is port/optimisation diagnosis, not a
larger benchmark.

Run locally or on GPU with:

```text
python -m ccb fit-gate --config configs/d1_trm_fit_gate_v1.json
```

The direct-Transformer control uses
[`configs/d1_transformer_fit_gate_v1.json`](../configs/d1_transformer_fit_gate_v1.json):
the same fixed examples, seed, optimizer settings, batch size, and update
budget. It uses four unshared encoder blocks so that it is a sufficiently
capable pipeline control; it is not presented as a parameter- or
compute-matched benchmark baseline.
