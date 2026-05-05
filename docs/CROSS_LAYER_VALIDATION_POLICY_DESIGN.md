# Cross-Layer Validation Policy Design

## Classification

`cross_layer_validation_policy_design_recorded`

Layer6 update: `cross_layer_validation_policy_design_layer6_updated`

Official-linear discriminator follow-up:

- Branch: `design/official-linear-backend-discriminator`
- Doc: `docs/OFFICIAL_LINEAR_BACKEND_DISCRIMINATOR_DESIGN.md`
- Classification: `official_linear_backend_discriminator_design_recorded`
- Purpose: scope a validation-only BF16 `attn.out` / `F.linear`
  discriminator before any layer6 o-proj implementation or runtime-policy
  discussion.

## Scope

This document records final-token ordered validation evidence only. It covers
layers 2, 3, 4, 5, and 6, plus focused layer11 MLP evidence. It is not a
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
| 6 | Source-complete final-token ordered attention audit cleared. | Raw-QK and weighted-V are current/default exact. Attention o-proj remains blocked under current/default: no bounded consumer o-proj sweep policy clears the full vector without collateral mismatches. Producer-side dtype probe confirms official live `attn.out` and `torch.nn.functional.linear` match the o-proj artifact; simpler matmul/einsum full-vector replay has 826 mismatches. MLP down baseline current sequential is exact; deterministic abs-ascending is exact but not required. | BF16-product rejected with broad MLP collateral mismatches; lane-local o-proj focus fixes are rejected because they introduce other o-proj mismatches. | false | Layer6 remains blocked pending an official-linear-backend discriminator; not a runtime/default policy. |
| 11 | Focused ordered MLP evidence only. | Attention source is a coarse official attention residual seam, not source-complete attention. MLP norm needs pairwise in coarse mode. Selected MLP down chain clears under validation-only candidates including deterministic abs-ascending. | BF16-product rejected with broad collateral mismatches. | false | Not a full ordered attention plus MLP surface. |

## Operator-Specific Conclusions

### Raw-QK

Layer3 needs pairwise or reverse raw-QK accumulation to clear the full raw-QK
and masked-logit matrices. Layers4, 5, and 6 are current/default exact for
raw-QK. No global raw-QK policy is justified. Any raw-QK policy must remain
explicit and validation-only until additional layers prove it is needed and
non-regressive.

### Weighted-V

Layer5 needs pairwise or reverse weighted-V accumulation. Layers2, 3, 4, and
6 clear with current/default weighted-V behavior. The layer5 debug localized
the single weighted-V mismatch to accumulation/output rounding; sink handling,
GQA mapping, and source layout were not the cause. No global weighted-V policy
is justified.

### Attention o-proj

Layers4 and 5 need reverse o-proj accumulation to clear their ordered o-proj,
attention residual, and bridge checks. Layers2 and 3 are current/default exact.
Layer6 is different: reverse, pairwise, chunked-pairwise, f64 diagnostic, and
bias variants do not clear the full layer6 o-proj vector without collateral
mismatches.

The layer6 producer-side dtype probe
(`/tmp/layer6_attention_oproj_lane22_dtype_probe_status.json`) confirms that
official live `attn.out(weighted_flat)` and `torch.nn.functional.linear` match
the prior o-proj artifact exactly, while simpler matmul/einsum full-vector
replay has 826 mismatches. The layer6 focus mismatch is lane `22`: consumer
local `9.125`, official `9.0625`, diff `0.0625`; weighted-V live versus prior
artifact has zero mismatches and the official output dtype is BF16.

Therefore o-proj is not merely a reverse-accumulation policy problem. It may
require an official BF16 linear backend discriminator for validation. Layer0
had earlier chunked-pairwise o-proj discriminator evidence, so even o-proj is
not uniform yet. No global o-proj policy is justified.

### Selected MLP down

Deterministic abs-ascending MLP down clears layer1, layer2, layer5, layer6,
and layer11 evidence. Layer4 baseline current sequential is exact, and
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
- "Layer4/layer5 reverse o-proj should be applied globally."
- "Layer6 can be fixed by lane-local focus correction."
- "Matmul/einsum equivalence is sufficient for official F.linear parity."
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
- A validation-only official-linear-backend discriminator before any o-proj
  policy implementation discussion.
- Layer6 producer-side F.linear evidence consumed as source proof before any
  future o-proj backend experiment.
- No raw `/tmp` or `.live` artifact commitment.

## Recommended Next Options

### Option A - layer7 ordered surface

Generate a layer7 ordered surface only if the goal is more evidence. It should
follow the same audit-first pattern and must not emit or promote a layer
output. This is not preferred until the layer6 o-proj official-backend
boundary is documented.

### Option B - validation-only official-linear-backend design

Create a docs/design-only branch first to scope a BF16 `F.linear` /
`attn.out` discriminator for attention o-proj. It must not touch production
defaults, must not change runtime/default routing/CUDA behavior, and must not
add a Torch runtime dependency in Rust. A later code branch, if separately
authorized, may compare existing cuBLAS BF16 tensor-op validation helpers
against the official F.linear boundary.

Recommended: Option B first, because layer6 showed the existing o-proj policy
family is insufficient before more layer collection or implementation
scaffolding.

## Disabled-by-Default Validation Feature Design

Classification: `cross_layer_validation_policy_feature_design_recorded`

### Design Intent

This feature design is for validation replay only. Policies must be explicit
per operator, and defaults must remain current behavior. It authorizes no
production runtime change, no default model-runner routing change, no CUDA
kernel change, no output emission, no ladder continuation, and no
tolerance-based pass criteria.

### Proposed Validation Knobs

These knobs are conceptual only and are not implemented by this document.

Raw-QK:

```text
--raw-qk-accum-policy current|pairwise|reverse|f64-diagnostic
```

Weighted-V:

```text
--weighted-v-accum-policy current|pairwise|reverse|f64-diagnostic
```

Attention o-proj:

```text
--attention-oproj-policy current|reverse|pairwise|chunked-pairwise|f64-diagnostic
```

Selected MLP down:

```text
--selected-mlp-down-policy current|deterministic-abs-ascending|pairwise-f32|pairwise-f64|f64-diagnostic
```

RMSNorm:

```text
--norm-reduction-policy current|pairwise|f64
```

`f64` variants are diagnostic only. BF16-product is intentionally excluded as
a candidate knob; if it is exposed for evidence, it must be marked
evidence-only/rejected. Any unsupported policy/layer combination should fail
closed rather than silently falling back.

### Policy Scope Rules

- Every policy result must name the operator and layer.
- Every status JSON must record `runtime_behavior_changed = false`,
  `production_routing_changed = false`, and `cuda_kernels_changed = false`.
- Every explicit policy must record the source proof status that justified it.
- Pairwise, reverse, deterministic abs-ascending, and related variants must
  never be inferred globally.
- Exact full-vector or full-matrix metrics remain authoritative.
- Focus-lane checks are diagnostic only.

### Status JSON Contract

Future status JSONs should include at least:

```text
classification
validation_only
runtime_behavior_changed
production_routing_changed
cuda_kernels_changed
layer_index
operator
selected_policy
default_policy_result
explicit_policy_result
source_statuses
full_vector_or_matrix_metrics
collateral_mismatches
output_emitted
ladder_continued
final_logit_claim
all_layer_claim
server_claim
context_length_claim
next_bounded_step
```

### Runtime Guardrails

- Compile-time or CLI-only validation path.
- No default route change.
- No production CUDA replacement.
- No correction metadata.
- No tolerance pass.
- No raw `/tmp` or `.live` commits.
- No Torch runtime dependency in Rust.
- No layer output emission unless a separate output-emission design is
  approved.

### Proof Gates Before Code Implementation

1. Accepted design doc.
2. One additional ordered surface, or an explicit rationale not to collect one.
3. Release/performance characterization for costly policies.
4. Collateral sweeps over full matrices/vectors.
5. Explicit unsupported-policy behavior.
6. Rollback story.
7. Exact command examples for each operator.
8. Status JSON schema agreed before code.
9. Validation-only official-linear-backend discriminator scoped before any
   attention o-proj policy implementation discussion.
10. Layer6 producer-side F.linear evidence consumed as source proof for any
    future o-proj backend experiment.

### Suggested Implementation Order, if later authorized

1. Status-schema helper only.
2. Parser-only CLI knobs that still execute current behavior.
3. Raw-QK policy hook in validation bundle path.
4. Weighted-V policy hook.
5. o-proj policy hook.
6. Selected MLP down policy hook.
7. Matrix summary mode.

Implementation is not authorized by this document alone.

### Open Questions

- Does layer6 o-proj require a validation-only BF16 F.linear backend rather
  than accumulation-order variants?
- Should the producer-side F.linear probe be repeated for layer4/layer5 to
  distinguish reverse policy from official backend coincidence?
- Can existing cuBLAS BF16 tensor-op validation helpers approximate official
  F.linear on o-proj, or is CPU BF16 F.linear fundamentally different?
- Should layer7 be generated before code, or only after the layer6 o-proj
  backend boundary is documented?
- Should layer4/layer5 reverse o-proj get a PyTorch dtype probe?
- Should selected MLP down abs-ascending be demoted to proof-oracle only
  because of cost and layer4 regression?
- Should attention policies be split into separate docs from MLP policies?
- What release/perf budget would be acceptable for validation-only replay?

### Recommendation

The recommended next step after this document is either:

- Generate layer7 ordered surface for pure evidence collection.
- Create a tiny status-schema-only branch if the team wants implementation
  scaffolding.
- Create a docs-only official-linear-backend discriminator design for layer6
  attention o-proj.

Preferred: docs-only official-linear-backend discriminator design before code
or layer7. Do not generate layer7 yet unless the goal is pure evidence
collection.

## Caveats

- Final-token only.
- One prompt/case.
- Ordered bundle/audit surfaces, not the full server path.
- Some surfaces use explicit oracle/audit seams.
- No final logits.
- No all-layer parity.
- No 4097 claim.
- No production/server parity.
