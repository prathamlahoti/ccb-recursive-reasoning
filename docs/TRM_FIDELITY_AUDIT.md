# TRM Fidelity Audit

Status: completed on 2026-08-30. No new GPU experiment is authorized from the
audited adapter.

## Scope and source

The audited reference is the official Tiny Recursive Models repository at
commit `c01103738605ba39d1430519b1ee0c62f4c707f8`, specifically:

- `models/recursive_reasoning/trm.py`
- `models/layers.py`
- `models/losses.py`
- `models/ema.py`
- `pretrain.py`
- `config/arch/trm.yaml` and `config/cfg_pretrain.yaml`

The local target was `src/ccb/models/published_trm.py` plus the associated
training path. The result is unambiguous: it is an **experimental
upstream-inspired CCB adapter**, not an upstream-faithful TRM port.

## Findings

| Component | Official implementation | Audited CCB adapter | Status |
| --- | --- | --- | --- |
| Outer recurrence | Fixed H/L buffers; `H_cycles-1` no-grad iterations plus one grad-bearing iteration | Same broad schedule | Partial match |
| L-level sharing | Shared `L_level` module | Shared module list | Partial match |
| Attention | Custom QKV projection, RoPE, scale-dot-product attention, no padding path | `nn.MultiheadAttention`, learned positions, padding mask | Material divergence |
| RMS normalization | Functional, scale-less RMS norm | `nn.RMSNorm` with learned affine scale | Material divergence |
| SwiGLU | Official rounded/intermediate width and custom CastedLinear | Different intermediate width; standard `nn.Linear` | Material divergence |
| Initialization | Truncated LeCun for embeddings/linears; truncated-normal fixed H/L buffers | PyTorch defaults; `torch.randn` H/L buffers | Material divergence |
| Precision | Configured bfloat16 casted modules | Standard float32 modules | Material divergence |
| Position encoding | RoPE by default | Learned position embeddings | Material divergence |
| Default model scale | 512 hidden, 8 heads, 2 L layers, 16 ACT steps | 32 hidden, 4 heads, 1 L layer, 1–3 ACT steps in gates | Different model regime |
| Input representation | One fixed token sequence, optional puzzle-ID embedding | CCB-specific `(initial-cell, operation)` composite tokens and no puzzle embedding | Necessary CCB adaptation; unvalidated |
| Output representation | Token logits over the source sequence vocabulary | Per-state-cell trace logits | Necessary CCB adaptation; unvalidated |
| Stablemax loss | Per-example token normalization, summed over batch then driver divides by global batch | Equivalent mean scaling for the no-continue path | Close match |
| Q-continue loss | Available when continue ACT is enabled | Omitted | Benign only while `no_ACT_continue=True` |
| ACT state semantics | Per-row reset and current-data replacement | Same intended semantics with extra fixed-shape safeguards | Requires equivalence test |
| Optimizer/schedule | AdamATan2, β=(0.9,0.95), weight decay 0.1, warm-up schedule | AdamW defaults, β=(0.9,0.999), zero decay, fixed 0.003 LR | Material divergence |
| EMA | Parameter-only moving average copy | State-dict moving average copied to evaluator | Close, but not byte-identical |

## Consequence

The completed fit gates demonstrate only that this experimental adapter did
not fit the frozen D1 set. They do **not** test whether the released TRM
algorithm can fit CCB semantics. In particular, the successful Transformer
control isolates the failure to this adapter/training recipe, not to CCB data.

## Required replacement path

1. Vendor or faithfully port the official primitive layers: custom
   `CastedLinear`, `CastedEmbedding`, scale-less RMS norm, official SwiGLU,
   RoPE, and initialization.
2. Preserve the official TRM inner and ACT wrappers mechanically, with only a
   narrow CCB I/O adapter at the boundary.
3. Reproduce a reference forward/gradient trace from the upstream code on a
   synthetic fixed-token batch before attaching CCB data.
4. Port the upstream optimizer schedule or explicitly label any alternative
   optimizer as an ablation.
5. Re-run the 64-example fit gate. Only a passing faithful port may proceed to
   the generated-depth calibration.

Until these steps are complete, do not call `PublishedTRMCCB` a reproduction,
an upstream-faithful model, or TRM evidence.

## Implementation status

The first replacement milestone is complete:

- [`official_trm_core.py`](../src/ccb/models/official_trm_core.py) is a
  mechanical local port of the pinned official no-puzzle-embedding core and
  ACT wrapper.
- [`verify_trm_fidelity.py`](../scripts/verify_trm_fidelity.py) loads the
  official state dict into that core and verifies exact logits, recurrent
  carry, Q logits, and parameter gradients on a fixed float32 token batch.
- [`official_trm_ccb.py`](../src/ccb/models/official_trm_ccb.py) supplies the
  only CCB-specific boundary: target-free initial-state tokens, operation
  tokens, and positioned query slots whose logits are read as CCB state cells.

The core-equivalence check and the CCB target-leakage/output-shape unit test
pass locally. The adapter has not yet been connected to a faithful CCB loss,
ACT loss head, official optimizer schedule, or experiment launcher. Therefore
it remains **CPU-verified integration work**, not a GPU-ready model.
