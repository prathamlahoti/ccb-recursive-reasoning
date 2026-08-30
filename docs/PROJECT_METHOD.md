# Applying TRM to CCB: Correct Project Definition

## Original objective

Apply Tiny Recursive Models (TRM) to the three sequential-reasoning tasks in
the Complexity Ceiling Benchmark (CCB) in a scientifically meaningful way.

Primary sources:

- TRM paper: https://arxiv.org/abs/2510.04871
- TRM code: https://github.com/SamsungSAILMontreal/TinyRecursiveModels at
  `c01103738605ba39d1430519b1ee0c62f4c707f8`
- CCB paper: https://arxiv.org/abs/2606.29278
- CCB code/data: https://github.com/Shubh-Chapra/Complexity_Ceiling_Benchmark
  at `e1e953a7a33a8c2406f936a54f4900f740dea8dd`

## What CCB is, and is not

CCB is an **evaluation suite for prompted LLMs**. It provides 400 fixed,
fully labelled instances per domain at depths 5, 10, ..., 50, and evaluates
model-generated reasoning traces. It is not a train/test dataset for neural
models and it publishes no TRM baseline.

Therefore there are two distinct, valid projects. They must never be merged in
the wording, data flow, or results table.

| Project | Valid claim | Training data | Evaluation |
|---|---|---|---|
| A. CCB evaluation adapter | “We implement CCB exactly as an evaluator for a TRM-style solver.” | None from the 1,200 fixed CCB records | Exact reproduction of CCB parsing/metrics on the fixed records, if a solver can accept each full task instance |
| B. CCB-TRM supervised extension | “We study depth generalization of TRM on generators that reproduce CCB semantics.” | Newly generated instances only | Held-out generated depths; the fixed CCB records are one final transfer set, not validation or tuning data |

Project B is the feasible research direction. It is an adaptation experiment,
not an original CCB leaderboard result. The project must say this plainly.

## Correct technical question

> Given the complete initial state and full ordered operation/event sequence,
> can a faithfully ported TRM learn to emit the complete sequence of CCB
> states and retain transition accuracy at unseen greater depths better than a
> compute-matched nonrecursive Transformer?

This tests exactly the failure mode CCB was designed to expose: sequential
state retention under increasing depth. It does **not** test language
understanding, prompt following, or compare a small supervised model directly
with the frontier LLMs in the CCB paper.

## Mandatory implementation sequence

1. **Freeze the task adapter.** Import or precisely port each official CCB
   generator and formatting rule. Keep the existing 1,200/1,200 exact
   compatibility check. Inputs are initial state plus the full ordered program;
   labels are full intermediate-state traces. No target token may enter input.
2. **Make the training extension explicit.** Generate new instances only from
   the verified generators. Train depths must begin at 5, matching CCB; choose
   held-out depths from the official range before any >50 stress test. Record
   generator commit, seed lists, split hashes, and duplicate/semantic-overlap
   audits.
3. **Port the actual TRM algorithm.** Mechanically preserve the released
   primitive layers, initialization, RoPE, fixed H/L buffers, shared
   `L_level`, no-gradient H cycles, detached carries, ACT state/reset/halting,
   stablemax loss, optimizer schedule, and EMA behavior. The CCB I/O adapter
   is the only permitted adaptation. See
   [the fidelity audit](TRM_FIDELITY_AUDIT.md): `PublishedTRMCCB` does not meet
   this gate; `official_trm_ccb` is the verified replacement pending its fit
   gate.
4. **Pass deterministic gates.** Unit tests for no target leakage, exact
   upstream recurrence semantics, loss masking/normalization, ACT reset,
   copied EMA, and checkpoint-resume; then an ACT-enabled fixed-batch fit
   gate.
5. **Run one preregistered calibration.** TRM and a compute-matched direct
   Transformer train on exactly the same generated data, seeds, updates,
   precision, and stopping rule. Model selection uses generated validation
   only. Evaluate the untouched 400 official records per domain once,
   afterwards, as transfer.

## Boundaries and non-claims

- Do not call generated-data results “CCB benchmark results”; call them
  **CCB-TRM extension results**.
- Do not call any previous adapter, including `PublishedTRMCCB`, a TRM
  reproduction or upstream-faithful model.
- Do not report any prior D1 pilot as a result. It used an invalid structural
  premise and a non-upstream model.
- Do not train on or tune against the fixed official CCB records.
- D1 is a low-entropy finite transformation system. It is useful as a smoke
  test, but cannot by itself substantiate a general reasoning claim. D2 and
  especially D3 are required before making a method-level claim.

## Clean repository scope

Keep only: official CCB data/provenance, exact verified task adapters,
serialization/metrics/evaluation, the direct Transformer baseline, the new
TRM port, deterministic tests, and current project documentation. Remove old
pilot configurations, results, reports, paper drafts, and obsolete recursive
models rather than presenting them as historical research.
