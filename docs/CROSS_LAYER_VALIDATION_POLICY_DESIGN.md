# Cross-Layer Validation Policy Design

## Classification

`cross_layer_validation_policy_design_recorded`

## Scope

This document records final-token ordered validation evidence only. It covers
layers 2, 3, 4, and 5, plus focused layer11 MLP evidence. It is not a
production runtime design, not a default routing proposal, not a CUDA kernel
proposal, and not a final-logit, all-layer, server, or 4097-token claim.

The purpose is to prevent the current evidence from being collapsed into one
global policy switch. The observed policies are operator-specific and
layer-sensitive.

## Evidence Matrix

| Layer | Evidence | Operator policies | Rejected/collateral policies | Output emitted? | Caveats |
| --- | --- | --- | --- | --- | --- |
| 2 | Source-complete final-token ordered attention audit cleared. | Raw-QK, weighted-V, and o-proj are current/default exact. MLP down clears under `deterministic_f32_abs_ascending_sum_then_bf16_output`. | BF16-product rejected with broad collateral mismatches. | false | Final-token ordered validation only. |
| 3 | Source-complete final-token ordered attention audit cleared. | Raw-QK requires explicit `pairwise_f32_scale_after_sum_bf16_output`; reverse also clears. Weighted-V and o-proj are current/default exact. MLP down baseline current sequential is exact; deterministic abs-ascending is exact but not required. | BF16-product rejected with broad collateral mismatches. | false | Clears only under explicit validation-only raw-QK policy, not default/runtime parity. |
| 4 | Source-complete final-token ordered attention audit cleared. | Raw-QK and weighted-V are current/default exact. o-proj requires `reverse_f32_accum_f32_bias_bf16_output`. MLP down baseline current sequential is exact. | Deterministic abs-ascending regresses layer4 MLP down; BF16-product rejected with broad collateral mismatches. | false | Clears only under explicit validation-only o-proj reverse policy. |
| 5 | Source-complete final-token ordered attention audit clears only with explicit weighted-V policy. | Raw-QK is current/default exact. Weighted-V clears under `pairwise_f32_bf16_output`; reverse also clears. o-proj requires `reverse_f32_accum_f32_bias_bf16_output` after weighted-V policy. MLP down clears under deterministic abs-ascending. | BF16-product rejected with broad collateral mismatches. | false | Clears only under explicit validation-only weighted-V plus o-proj policies. |
| 11 | Focused ordered MLP evidence only. | Attention source is a coarse official attention residual seam, not source-complete attention. MLP norm needs pairwise in coarse mode. Selected MLP down chain clears under validation-only candidates including deterministic abs-ascending. | BF16-product rejected with broad collateral mismatches. | false | Not a full ordered attention plus MLP surface. |

## Operator-Specific Conclusions

### Raw-QK

Layer3 needs pairwise or reverse raw-QK accumulation to clear the full raw-QK
and masked-logit matrices. Layers4 and 5 are current/default exact for raw-QK.
No global raw-QK policy is justified. Any raw-QK policy must remain explicit
and validation-only until additional layers prove it is needed and
non-regressive.

### Weighted-V

Layer5 needs pairwise or reverse weighted-V accumulation. Layers2, 3, and 4
clear with current/default weighted-V behavior. The layer5 debug localized the
single weighted-V mismatch to accumulation/output rounding; sink handling,
GQA mapping, and source layout were not the cause. No global weighted-V policy
is justified.

### Attention o-proj

Layers4 and 5 need reverse o-proj accumulation to clear their ordered o-proj,
attention residual, and bridge checks. Layers2 and 3 are current/default exact.
Layer0 had earlier chunked-pairwise o-proj discriminator evidence, so even
o-proj is not uniform yet. No global o-proj policy is justified.

### Selected MLP down

Deterministic abs-ascending MLP down clears layer1, layer2, layer5, and
layer11 evidence. Layer4 baseline current sequential is exact, and
deterministic abs-ascending regresses layer4 with collateral mismatches.
Therefore deterministic abs-ascending is not globally safe. BF16-product
remains rejected.

### RMSNorm

Pairwise/f64 reductions cleared prior layer2 attention norm and layer11 MLP
norm diagnostics. Norm policy remains validation-only. No production norm
change is authorized.

## Rejected Interpretations

- "Just switch everything to pairwise."
- "Just switch everything to reverse."
- "Just use deterministic abs-ascending everywhere."
- "BF16-product is a correction."
- "These exact layers authorize runtime parity."
- "Layer outputs can now be emitted or promoted."
- "This proves final logits or all-layer parity."

## Authorized Uses

The current evidence authorizes only validation-only replay/status modes,
seam localization, policy sweeps, docs/provenance updates, bounded additional
ordered surfaces, and future disabled-by-default experiments after separate
design approval.

## Not Authorized

- Production runtime behavior changes.
- Default model-runner routing changes.
- CUDA kernel changes.
- Correction metadata application.
- Tolerance-based pass criteria.
- Layer ladder continuation from these surfaces.
- Output emission or promotion.
- Final-logit, all-layer, server, or 4097-token claims.

## Proof Gates Before Any Implementation Experiment

- At least one more ordered surface, or a written rationale for stopping
  surface collection.
- Release/performance characterization for any expensive policy.
- Collateral-mismatch sweeps over full vectors and matrices.
- Separation of attention policies from MLP policies.
- Disabled-by-default validation feature flag design.
- Rollback and no-default-routing guarantee.
- Clear statement of which operator each policy applies to.
- No raw `/tmp` or `.live` artifact commitment.

## Recommended Next Options

### Option A - layer6 ordered surface

Generate a layer6 ordered surface if the goal is more evidence. It should
follow the same audit-first pattern and must not emit or promote a layer
output.

### Option B - validation-only implementation design

Create a docs/design-only branch first. It may propose disabled-by-default
knobs for raw-QK, weighted-V, o-proj, and selected MLP down validation, but it
must not touch production defaults.

Recommended: Option B first, because the matrix already shows enough
variation to make policy design necessary before more layer collection.

## Caveats

- Final-token only.
- One prompt/case.
- Ordered bundle/audit surfaces, not the full server path.
- Some surfaces use explicit oracle/audit seams.
- No final logits.
- No all-layer parity.
- No 4097 claim.
- No production/server parity.
