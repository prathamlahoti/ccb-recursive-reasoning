# TRM improvement candidates grounded in our CCB experiments

Date: 2026-10-12. Status: literature screening and proposed experiments;
no new method result or GPU run.

Follow-up: [CPU exploration findings](PREFIX_EXPLORATION_FINDINGS.md).
The implemented prefix ablation has not established a generalisation benefit;
a 567-parameter D1-specific learned executor solves all tested generated lengths.
This raises the baseline requirement and weakens the case for a D1-only method.
The [subsequent methodology audit](PREFIX_METHODOLOGY_AUDIT.md) found duplicate
and shared CPU split instances and non-comparable regression risk sets. Those
eight reduced-model runs cannot establish a prefix-loss effect.

## Recommendation

Investigate dependency-aware refinement of execution traces, with explicit
measurement of progress and regression across TRM iterations. The question is:
can a shared recursive model extend a correct execution prefix without damaging
earlier states, and thereby improve accuracy at longer program lengths for the
same compute budget?

This is a candidate research gap, not established novelty. Intermediate-state
supervision, curricula, causal regularisation, and iterative refinement all
have substantial prior work. A contribution must isolate an execution-specific
mechanism and show an advantage over those ingredients in strong controls.

## What our results actually establish

The verified adapter and upstream core fit small fixed datasets. On the 400
training examples spanning depths 5/10/15/20, however, step-10,000 TRM final
accuracy was 0.75% live and 0.50% EMA, with zero full-trace accuracy. Transformer
final accuracy was 57.25% live and 53.75% EMA. Both had zero final accuracy on
held-out depths 25--50. See `RESEARCH_GO_NO_GO.md` and the compact result JSONs.

These observations reject the current recipe. They do not isolate a mechanism,
prove an intrinsic TRM limitation, or establish that additional supervision
will fix it. Likewise, Transformer decline across training depths does not
by itself establish a capacity problem or a need for a curriculum.

The adapter already supervises every valid execution-state token. Thus simply
"adding intermediate trace supervision" would misdescribe our current code.
Execution time t and neural refinement iteration r are different axes: the
existing loss asks every iteration to predict the entire finished trace. The
proposed extension changes how refinement is trained and allocated, rather
than merely adding trace labels.

## Prior work that constrains novelty

| Work | Established direction | Consequence for our proposal |
|---|---|---|
| [Original TRM](https://arxiv.org/abs/2510.04871) and [official repository](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) | Shared latent/answer refinement, deep supervision, training halting, EMA | These are baseline components, not our contribution. |
| [CGAR](https://arxiv.org/abs/2511.08653) | Progressive architectural recursion depth and supervision weighting | "Use curriculum to accelerate TRM" is already explored. Program-length curriculum is distinct but conventional. |
| [Probabilistic TRM](https://arxiv.org/abs/2605.19943) | Noise-driven parallel trajectories and selection through the Q head | Adding randomness or multiple attempts has direct TRM prior work. |
| [TRM-Planner](https://aclanthology.org/2026.findings-acl.1350/) | Offline teacher trajectory caching and distillation | Improved refinement targets alone have close prior art. |
| [Denoising Recursion Models](https://proceedings.mlr.press/v306/cameron26a.html) | Train recursive denoising paths through intermediate corruption states | A generic "teach a refinement trajectory" claim is insufficient. |
| [When Recursive Models Finish Computing](https://arxiv.org/abs/2609.26487) | Completion dynamics and anisotropic local stability; cumulative solves beyond nominal budget | More iterations and stability diagnosis are active areas, not untouched gaps. Cumulative solve rate must not be confused with deployable final accuracy without an answer-selection rule. |
| [Tiny Autoregressive Recursive Models](https://arxiv.org/abs/2603.08082) | Controlled comparisons on algorithmic tasks; no reliable benefit from the full autoregressive TRM in their setting | Merely making TRM sequential is not a defensible new direction. |
| [CLRS](https://arxiv.org/abs/2205.15659), [CLRS-Text](https://arxiv.org/abs/2406.04229), [Hint-ReLIC](https://arxiv.org/abs/2302.10258) | Execution hints, length generalisation, causal invariance | Learned execution and causal regularisation long predate TRM. These supply controls and related work. |

This screening used primary paper/repository pages and abstracts, plus HTML
for original TRM and CGAR. It is not a full implementation-level comparison
of every related method. Targeted searches did not establish an exact existing
match for the proposed execution-prefix mechanism; search absence is not proof
of novelty. Full close-paper/code comparison remains necessary before claiming
a new algorithm.

## Foundational confounds to resolve first

1. **Output-position extrapolation.** `build_ccb_trm_layout` reserves operation
   and trace slots to max depth 50, but training labels stop at depth 20.
   Test depths activate output slots never directly supervised, and change PAD
   tokens to MASK. This mixes program-depth generalisation with representation
   extrapolation. Shared output weights may generalise, but that is not assured.
2. **Canvas size.** Small fit gates use short canvases; the depth-50 canvas has
   512 tokens for D1. Re-run the same fixed training examples under identical
   short and padded canvases before attributing their difference to reasoning.
3. **ACT exposure accounting.** The trainer supplies a fresh pending batch
   each update but accepts new examples only in halted rows. Equal optimiser
   updates do not imply equal unique-example exposure or compute relative to
   Transformer. Count accepted rows by depth, recurrence steps, supervised
   tokens, and measured GPU time/FLOPs. Long episodes may occupy rows longer.
   This is a measurement concern, not evidence that carry replacement is wrong.
4. **Sparse exact-halt signal.** `trm_sequence_loss` gives a positive halt
   target only when the whole valid trace is exact. Inspect positives by depth
   and iteration. The hypothesis that this harms learning remains untested.
5. **Inference selection.** Keep last-iterate, learned-halt, and oracle-best
   iteration scores separate. Oracle-best is a diagnostic upper bound and must
   never become the reported deployment score.

## Candidate mechanism: prefix-aware refinement

Retain the shared TRM core. For each neural iteration r, predict the execution
trace and an auxiliary estimate of the length of its correct prefix. During
training, oracle traces can supply prefix-length labels and identify states
that were correct before a refinement update. Train for advancing the prefix
and penalise corruption of previously correct states, alongside the full trace
loss. Begin with this loss-only experiment before designing a larger module.

At inference, no oracle prefix is available. Any adaptive scheduling must use
the learned progress estimate, whose calibration must be measured on generated
validation data. Avoid hard-freezing confident tokens in the first prototype:
a wrongly trusted prefix can permanently block repair. A later soft gate may
focus refinement near uncertain dependencies while retaining a repair path.

There is no guarantee that every iteration should advance one program step.
Test the proposal against plain deep supervision, ordinary program-length
curriculum, and generic target-planning/denoising approaches; stop if the gain
comes entirely from a conventional training improvement.

## Bounded experimental decision tree

1. CPU: audit input/label boundaries, output-slot coverage, ACT accepted-example
   counts, prefix/regression metrics, and deterministic resumability. Preserve
   official records for the predeclared final transfer evaluation.
2. Small paired GPU diagnostic: same fixed depth-5 data, short versus padded
   canvases, TRM and Transformer. Define budgets and stopping rules before
   launching. This determines whether the representation blocks learning.
3. Single-seed generated-data experiment: baseline TRM, TRM plus program-length
   curriculum, and TRM plus prefix loss; match exposure and report compute.
   Require high train and fresh same-depth validation accuracy through depth
   20 before considering any held-out-depth advantage.
4. Compare held-out depths 25--50 using full-trace and final accuracy, per-depth
   transition accuracy, first divergence, iteration regression rate, and
   accuracy versus compute. Add matched Transformer and looped Transformer.
   A learned one-step executor rolled out repeatedly is a necessary strong
   baseline; its predictions, not oracle states, drive test rollout.
5. Only on passing the gate: three seeds, a compositional split, and D2
   transfer. A broad TRM claim also needs evidence outside the very simple D1
   permutation domain, ideally a suitable CLRS task or original TRM task.

Do not require near-perfect same-depth validation as a universal theorem for
all research; it is our practical gate for these deterministic simple tasks.

## What would justify a paper

A defensible result would show that execution-aware refinement reduces
regression, increases depth horizon, and improves exact trace accuracy under
matched compute across seeds and tasks. Ablations must show the prefix
mechanism matters beyond curriculum, additional labels, a better adapter, or
ordinary sequential rollout. A gain only on D1 or only against our failed
baseline would be insufficient for a broad new-method claim.

If the strong one-step executor already solves the tasks and TRM adds neither
accuracy nor efficiency, this direction should stop. The feasibility of the
implementation is substantially higher than the probability of a publishable
advantage; no empirical odds or guaranteed venue can be assigned yet.

## Other candidates

- Execution-aware calibrated halting: potentially useful after the model
  learns; measure predicted completion against correct-prefix progress. Close
  to existing halting/stability work and currently lower priority.
- Repair training on model-generated erroneous traces: possible recovery from
  rollout error, but denoising and teacher-trajectory work overlap strongly.
  Requires careful matched corrupted-data controls.
- Generic warm-up, gradient clipping, more loops, or curriculum: sensible
  engineering controls; alone they are not a compelling new TRM method.
