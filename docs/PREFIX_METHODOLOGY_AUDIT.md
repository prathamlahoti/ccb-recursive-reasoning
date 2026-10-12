# Audit of the prefix-loss CPU interpretation

Date: 2026-10-12. Review prompted by the user's question about an obvious
oversight. No new training or GPU job was run for this audit.

## Confirmed errors and limitations

1. The CPU split generator rejected official instances, but the caller did
   not pass a shared `forbidden_instances` set. Neither within-split
   duplicates nor train/validation duplicates were excluded. Regenerating the
   exact historical seeds confirms 16 training records contain 15 unique
   instances, 32 validation records contain 27 unique instances, and one
   instance occurs in both. Calling validation "fresh" was inaccurate.
2. All eight models had zero full-trace training accuracy. Thus the tiny
   width-32, H=1/L=1, 400-update experiment never established that its baseline
   could learn the task. It cannot diagnose extrapolation or establish a
   useful repair advantage. Passing software tests does not establish that
   experimental prerequisite. Prior released-width fit gates do not establish
   it for this substantially changed small configuration.
3. The regression denominator is the model's own previously correct prefix
   states. The weighted model produces a different collection of eligible
   events. Fractions across these risk sets are not a controlled test of
   stability. For seed 7, short canvas, the reported fraction drops from
   83/247 (33.60%) to 235/860 (27.33%), while absolute regressions increase
   from 83 to 235. Neither absolute counts alone nor fractions alone isolate
   a causal improvement. Repeated states and iterations are correlated.
4. These are online comparisons across changing model weights, not pure
   recurrent dynamics of a frozen checkpoint. We did not save the eight CPU
   model checkpoints, so their frozen inference trajectories cannot now be
   reconstructed without retraining. No retrospective stability score is
   claimed.
5. Alpha and beta were changed together. We did not run frontier-only or
   preservation-only controls. Any observed change cannot be attributed to
   preservation specifically. There are only two model seeds; the two canvas
   settings do not supply four independent seed replications.
6. The loss is normalised reweighting of existing labels. It supplies no new
   transition information, constrains no explicit execution dependency, and
   guarantees no monotonic progress. In particular, it can prioritise easy
   early states while failing to learn the rest of the trace.

The prior sentence "reduced measured training regression rates" was
arithmetically correct about these particular fractions but too suggestive as
evidence of a useful mechanism. The defensible conclusion is: the experiment
did not produce an interpretable test of prefix-loss effectiveness.

## Scope check on the older D1 experiment

Separately regenerating the primary D1 splits found:

| Split | Records | Unique instances |
|---|---:|---:|
| Train | 400 | 400 |
| Validation | 100 | 100 |
| Held-out depth | 600 | 600 |
| Strong depth | 300 | 300 |

Every pair has zero shared instances. The CPU diagnostic overlap error does
not contaminate the older seed-17 primary split. Its checkpoint training
evaluation remains evidence that that particular recipe failed training fit.
Representation and exposure concerns remain separate questions.

The learned D1 routing baseline is not invalidated by this overlap finding:
it used its own random-start, one-step data and independent test seed range.
However, its source-position labels and permutation inductive bias differ
substantially from whole-trace TRM. Its 100% score demonstrates D1 structure,
not an architectural superiority result for equal training interfaces.

## Corrections made

- The CPU diagnostic now passes a shared exclusion set, checks uniqueness,
  and asserts both instance and program separation before training.
- Its default output is a new v2 directory. Historical scores remain intact
  and are labelled unsuitable for interpreting a prefix effect.
- If an unweighted baseline fails 99% EMA training trace accuracy in its actual
  configuration, the script stops before loss comparisons. This is a practical
  experimental gate, not a universal requirement for all methods research.
- Interpretations claiming even a tentative stability signal are withdrawn.

## Requirements for a meaningful next comparison

First fit the actual baseline configuration. Then evaluate frozen models on
the same disjoint probes, logging prefix progress, exact trace/final accuracy,
and complete refinement trajectories. Report regression alongside the amount
of correct work produced, compare matched stages of competence, and identify
frontier versus preservation effects separately. If using shared corrupted
states as probes, corruption and how it enters the model must be specified;
the current adapter cannot simply accept arbitrary discrete answer states.

Do not infer effectiveness from a lower fraction over a changing denominator,
and do not infer impossibility from an underfitting baseline. This audit
justifies improving the experiment before spending more GPU time.
