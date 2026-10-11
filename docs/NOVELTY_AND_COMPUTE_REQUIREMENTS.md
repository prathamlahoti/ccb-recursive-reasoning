# Novelty boundary and conditional compute requirements

Date: 2026-10-12. This is a research/resource assessment, not an experiment launch.

## Correction to the proposed research question

"Does TRM help when transitions depend on context or global constraints?" is
not an undiscovered question. The original TRM studies globally constrained
puzzles, including Sudoku. More recent work already studies refinement,
repair, algorithmic tasks, and combinatorial optimisation. Our prior sentence
identified a direction for testing, not a confirmed literature gap.

Relevant primary sources:

- [Original TRM](https://arxiv.org/abs/2510.04871): recursive puzzle solving.
- [Denoising Recursion Models](https://proceedings.mlr.press/v306/cameron26a.html):
  supervised multi-step denoising/refinement and comparisons with TRM.
- [Tiny Autoregressive Recursive Models](https://arxiv.org/abs/2603.08082):
  controlled, compute-matched algorithmic-task comparisons. Their full
  autoregressive TRM did not yield reliable gains in the studied setting.
- [Recursive Inference Machines](https://arxiv.org/abs/2603.05234): extending
  TRM with a reweighting component for reasoning benchmarks. This must be
  checked before making a broad claim about new TRM reweighting.
- [TRM for combinatorial optimisation](https://arxiv.org/abs/2609.34585):
  graph-aware recurrence for TSP and maximum independent set.
- [Causal regularisation for neural algorithmic reasoning](https://arxiv.org/abs/2302.10258):
  execution-state invariance and size generalisation outside TRM.

These sources establish significant overlap. They do not prove that every
possible execution-prefix loss exists already. A full equation/code comparison
with the closest methods remains necessary before claiming novelty.

## What we would have to demonstrate

1. Identify an explicit mechanism absent from the closest methods and a
   predicted failure mode it addresses. "Use TRM on harder tasks" is inadequate.
2. Establish competent vanilla TRM and non-TRM baselines on the selected tasks.
   A win against our failed D1 recipe is not useful evidence of improvement.
3. Show an accuracy/compute improvement at matched data and label access,
   counting ACT accepted presentations. Report both compute-matched and
   parameter-matched comparisons where appropriate.
4. Show that the proposed mechanism matters: remove it, substitute ordinary
   curriculum or weighting, and test the relevant close published alternative.
5. Replicate with at least three training seeds across at least two task
   settings; test held-out lengths/structures and model-generated corruption.
6. Demonstrate repair without test oracle states, test labels, or oracle-best
   selection. Report end-to-end trace accuracy and deployable answer selection.

These are practical criteria for a convincing empirical study, not a universal
publication rule. Neither a venue nor an improvement can be guaranteed.

## Known measurements

The saved D1 run used 6,838,274 TRM parameters and 6,326,272 Transformer
parameters. At the existing 512-token canvas and training batch 8:

| Worker | Training plus configured evaluation wall time | Peak allocated CUDA memory |
|---|---:|---:|
| TRM, 10,000 updates | 9,763.9 s = 2.71 h | 2.596 GiB |
| Transformer, 10,000 updates | 742.5 s = 0.206 h | 0.469 GiB |

Source: `results/d1_depth_generalization_seed17_float32_v1/summary.json`.
These timings include evaluation and apply to this particular adapter, float32
T4 environment, and schedule. They are not a general training-speed estimate
for a new task, dataset, sequence length, or model.

The paired Kaggle run held two T4s for about 2.72 wall hours. Allocated device
time was therefore approximately 5.43 GPU-hours, although the Transformer
worker finished much earlier. Independent two-GPU workers can save wall time;
an idle second GPU can waste allocation. Kaggle quota accounting should be
checked separately rather than assumed to equal physical device-hours.

## Hardware and planning allowances

One NVIDIA GPU with about 16 GiB VRAM is a reasonable starting point for a
similarly sized small-model pilot; our existing workload already ran on a T4.
This is not a guarantee for a larger graph or longer trace. Start with a
100--200-update calibration at the largest intended input and measure memory,
step time, ACT exposure, and evaluation time.

Use 16--32 GB host RAM and roughly 20--50 GB free artifact storage as practical
initial allowances. Keep only required checkpoints and separate archives from
their input directories. Persist configs, hashes, counters, results, and final
weights through server-side Version outputs and local/private backup.

| Stage | Compute planning allowance | What it decides |
|---|---|---|
| Close-method comparison, generators, oracle and leakage tests | CPU; zero GPU hours required | Whether an experiment has a distinct, testable claim |
| One-task training-path and feasibility pilot | Initially cap at 6--12 T4 device-hours | Whether baselines learn and the new mechanism has a useful signal |
| Small multi-task/seed/ablation study, only after a positive pilot | Provisional 50--150 T4 device-hours, potentially more | Whether the effect survives fair controls and replication |

The allowances are proposed spending limits, not measured requirements or
promises of convergence. Even the first pilot can be inadequate; it is a
decision gate. As a scale illustration, four model variants x two tasks x
three seeds gives 24 jobs. At 2.71 hours per job that would be about 65
device-hours before extra ablations. Some controls are much faster; difficult
tasks may take much longer. Re-estimate from task-specific calibration.

No frontier language-model training or multi-node cluster is inherently needed
for a small TRM experiment. A second T4 can execute independent jobs in parallel
if their runtimes and storage fit; do not assume two devices act as one larger
memory pool.

For procedurally generated CCB-like tasks, a starting data budget of several
thousand base examples and fresh generated validation is practical, with
on-demand sampling or augmentation. This is a proposed scale, not evidence
that that quantity suffices. Original TRM benchmark reproduction may require
the released augmentation and much longer training contract.

## Current recommendation

GPU access is not the immediate blocker. We have not established a distinct
method gap, and the prefix prototype has no successful validation result.
First compare its mechanism to the close repair/reweighting papers and define
one concrete hypothesis, task, and control set. Only then commit a 6--12-hour
pilot allowance. Extra compute alone cannot establish novelty.
