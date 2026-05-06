# Fused Linear/AddMM Official API Seam Synthesis

## Classification

```text
fused_linear_addmm_official_api_seam_synthesis_recorded
```

## Scope

This is a docs-only synthesis and decision record for the official
fused-linear/addmm attention o-proj API seam. It covers the
`developer-message-user-smoke` final-token ordered-surface evidence for sampled
layers 6, 10, 13, 16, 18, and 21.

This record does not authorize implementation, backend selection, consumer
revalidation, runtime/default routing changes, CUDA kernel changes, output
emission, or ladder continuation.

## Evidence Chain

The evidence chain now includes:

- Producer/API final matrix:
  `docs/O_PROJ_PRODUCER_API_FINAL_MATRIX.md`
- Backend/helper candidate exhaustion:
  `docs/FUSED_LINEAR_ADDMM_BACKEND_DISCRIMINATOR_DESIGN.md`
- cuBLASLt fused-bias prototype failure:
  `docs/FUSED_LINEAR_ADDMM_VALIDATION_PLAN.md`
- CPU producer attribution:
  `/tmp/fused_linear_addmm_cpu_producer_attribution_status.json`
- AddMM boundary localization:
  `/tmp/fused_linear_addmm_addmm_boundary_localization_status.json`
- Fused-bias arithmetic contract:
  `/tmp/fused_linear_addmm_fused_bias_arithmetic_contract_status.json`

The latest arithmetic-contract classification is:

```text
fused_linear_addmm_fused_bias_arithmetic_contract_inconclusive
```

## Strongest Current Conclusion

The strongest current conclusion is that the official reference for the sampled
attention o-proj seam is the CPU Torch API seam:

- module call
- `torch.nn.functional.linear`
- `torch._C._nn.linear`
- `torch.addmm(bias, input, weight.T)`

The observed reference uses BF16 input, BF16 weight, and BF16 bias. The strongest
arithmetic signal is fused bias before the final observable BF16 output.

Full-vector exactness is required. Focus-lane clears are diagnostic only.

The following remain negative controls:

- explicit matmul plus bias
- explicit einsum plus bias
- zero-bias addmm plus separate bias
- explicit unfused BF16 bias

## Unresolved

The following remain unresolved:

- concrete CPU backend identity
- one global accumulation/product policy
- a Rust/CUDA helper candidate that clears layers 6, 10, 13, 16, 18, and 21
- runtime promotion path

The arithmetic-contract probe supports bias-before-output-rounding, but no
explicit arithmetic variant localizes the exact accumulation/product policy
across the full sampled set.

## Why Blind Sweeps Should Stop

Further blind sweeps are lower value because:

- existing local helper families are exhausted for this evidence set
- cuBLAS/cuBLASLt candidates did not clear the sampled producer/API reference
- arithmetic variants are partial and layer-sensitive
- focus-lane clears remain diagnostic only and cannot select a policy
- explicit matmul/einsum/unfused-bias forms are negative controls, not
  alternative official references

## Decision

Preserve Workstream A as an official Torch API seam for now.

Do not implement runtime o-proj changes from this evidence. Do not run consumer
revalidation from this evidence. If future validation needs this boundary, use
producer/API artifacts as oracle seams rather than promoting local arithmetic
policies.

## Future Options

### Option A — Preserve And Move On

Preserve this synthesis and move to the next blocker or workstream.

### Option B — Producer/API Artifact Reuse Path

Design a validation-only path that reuses producer/API artifacts as oracle seam
inputs. This would be an artifact-consumption design, not runtime behavior.

### Option C — Custom Validation Helper Later

Only later design a custom validation helper if a global arithmetic policy is
discovered and proves full-vector safe across blocked layers and controls.

### Option D — GPU/Sharded Torch Oracle Later

Revisit GPU or sharded Torch oracle generation later if needed. For single-GPU
work, use GPU1 because displays are on GPU0, and consult the multi-GPU sharding
lane before any sharded run.

## Rust/CUDA Policy Feasibility Plan

Follow-up plan:

```text
docs/FUSED_LINEAR_ADDMM_RUST_CUDA_POLICY_FEASIBILITY_PLAN.md
```

Classification:

```text
fused_linear_addmm_rust_cuda_policy_feasibility_plan_recorded
```

The plan defines three gates before any policy implementation discussion:

- Gate A: CPU Torch dispatch-stability.
- Gate B: bounded Rust CPU policy synthesis, only if Gate A is stable.
- Gate C: CUDA mirror, only if one Rust CPU policy clears the sampled set.

The plan keeps the official API seam as the reference and explicitly rejects
per-layer policy selection, per-lane policy selection, focus-lane promotion,
tolerance passes, f64 diagnostic promotion, and runtime backend promotion from
this evidence.

## Gate A Dispatch-Stability Result

Status:

```text
/tmp/fused_linear_addmm_cpu_dispatch_stability_status.json
```

Classification:

```text
fused_linear_addmm_cpu_dispatch_stability_stable
```

The official CPU Torch addmm seam is stable across the tested fresh-process CPU
thread/backend settings for layers 6, 10, 13, 16, 18, and 21. No configuration
changed the full-vector addmm output relative to baseline, and every
configuration matched the official o-proj artifacts exactly. Baseline
matmul/einsum/unfused-bias controls remained negative.

This strengthens the official API seam as a stable oracle reference, but it
does not select a backend or authorize Rust CPU policy synthesis, consumer
revalidation, runtime/default/CUDA changes, output emission, or ladder
continuation.

## Gate B Rust CPU Policy Synthesis Result

Status:

```text
/tmp/fused_linear_addmm_rust_cpu_policy_synthesis_status.json
```

Classification:

```text
fused_linear_addmm_rust_cpu_policy_synthesis_partial_only
```

The bounded Rust CPU search did not find one selectable arithmetic policy that
clears all sampled o-proj full vectors for layers 6, 10, 13, 16, 18, and 21.
Some per-layer candidates clear layers 10, 13, or 21, but the sampled-set gate
fails because layers 6, 16, and 18 retain full-vector mismatches. Focus-lane
clears, diagnostic-only candidates, and evidence-only candidates remain
non-promotable.

This preserves the official CPU Torch module/F.linear/_C/addmm seam as the
current Workstream A oracle boundary. It does not authorize Gate C CUDA mirror
work, consumer revalidation, runtime/default/CUDA behavior changes, output
emission, or ladder continuation.

## Gate B Closure Audit Result

Status:

```text
/tmp/fused_linear_addmm_rust_cpu_policy_closure_audit_status.json
```

Classification:

```text
fused_linear_addmm_rust_cpu_policy_closure_no_global_policy
```

The validation-only closure audit replayed all 238 previously bounded
focus-clearing selectable candidates full-vector. No single candidate cleared
the sampled layers 6, 10, 13, 16, 18, and 21 as a global policy after closure.

The top near-global candidates still leave residual mismatches on the sampled
set. Residual samples in layers 6, 16, and 18 are within one BF16 ULP in the
simple check, but the audit did not localize a shared rounding/tie rule that
justifies another narrow policy branch.

The recommended state is `stop_policy_lane_preserve_official_api_seam`. This
reinforces the synthesis decision: preserve Workstream A as an official CPU
Torch API seam, not a Rust/CUDA backend identity. No backend is selected, no
implementation is authorized, no consumer revalidation is authorized, and no
runtime/default/CUDA behavior change follows.

## Producer/API Artifact Reuse Plan

Plan:

```text
docs/FUSED_LINEAR_ADDMM_PRODUCER_API_ARTIFACT_REUSE_PLAN.md
```

Classification:

```text
fused_linear_addmm_producer_api_artifact_reuse_plan_recorded
```

The reuse plan records the post-closure decision: Workstream A remains an
official CPU Torch module/F.linear/_C/addmm artifact seam. Future validation
may consume producer/API o-proj artifacts as oracle references, but this is
artifact consumption only. It does not select a Rust/CUDA backend, authorize
consumer revalidation, authorize CUDA mirror work, emit outputs, or change
runtime/default/CUDA behavior.

## Guardrails

- No backend selected.
- No implementation authorized.
- No consumer revalidation authorized.
- No runtime/default/CUDA behavior change.
- No output emission.
- No ladder continuation.
- No correction metadata promotion.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
- No Torch runtime dependency in Rust.
