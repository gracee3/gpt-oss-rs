# Fused Linear/AddMM Rust/CUDA Policy Feasibility Plan

## Classification

```text
fused_linear_addmm_rust_cuda_policy_feasibility_plan_recorded
```

## Scope

This is a docs-only feasibility plan for determining whether the official
fused-linear/addmm attention o-proj seam can ever become a Rust/CUDA
validation policy.

This plan does not implement probes, select a backend, authorize consumer
revalidation, authorize runtime/default/CUDA behavior changes, emit outputs, or
continue the ladder.

## Current Official Seam Contract

Current contract:

- operator: `attention_o_proj`
- sampled layers: 6, 10, 13, 16, 18, 21
- official reference: CPU Torch module/F.linear/_C._nn.linear/addmm
- input dtype: BF16 weighted-V
- weight dtype: BF16 o-proj weight
- bias dtype: BF16 o-proj bias
- bias behavior: fused into addmm before final observable BF16 output
- output dtype: BF16
- full-vector exactness required
- focus-lane-only clears rejected

The current synthesis is recorded in:

```text
docs/FUSED_LINEAR_ADDMM_OFFICIAL_API_SEAM_SYNTHESIS.md
```

## Research Objective

Primary question:

```text
Can one Rust-replayable fused-bias arithmetic policy reproduce the official sampled-set full vectors?
```

Not acceptable:

- per-layer policy selection
- per-lane policy selection
- focus-lane-only promotion
- tolerance pass
- f64 diagnostic promotion
- producer/API seam treated as runtime backend

## Gate A — CPU Torch Dispatch-Stability

Next implementation branch:

```text
oracle/fused-linear-addmm-cpu-dispatch-stability
```

Purpose:

Check whether `torch.addmm(bias, input, weight.T)` is stable under CPU
thread/backend settings before trying to replay it in Rust.

Required checks:

- `torch.set_num_threads(1)`
- `torch.set_num_threads(8)`
- `torch.set_num_interop_threads`, if feasible
- MKLDNN enabled true/false, if safe and supported
- fresh-process environment variants where feasible:
  - `OMP_NUM_THREADS=1`
  - `OMP_NUM_THREADS=8`
  - `MKL_NUM_THREADS=1`
  - `MKL_NUM_THREADS=8`
  - `ONEDNN_VERBOSE` or `DNNL_VERBOSE`, if useful

Rules:

- Always CPU-only.
- `cuda_available` may be recorded.
- `cuda_used` must remain false.

Success:

- Official addmm full vectors are identical across tested CPU settings for all
  sampled layers.

Failure:

- Official addmm full vectors change under tested settings.
- If this fails, Rust/CUDA policy feasibility becomes much weaker and must be
  recorded before more implementation work.

## Gate B — Rust CPU Policy Synthesis

Future branch, only after Gate A review:

```text
validation/fused-linear-addmm-rust-cpu-policy-synthesis
```

Purpose:

Search a bounded finite Rust-replayable arithmetic policy space.

Candidate policy dimensions:

- product:
  - BF16 inputs multiplied into f32
  - f64 diagnostic only
  - BF16-rounded product evidence-only
- accumulation:
  - forward f32
  - reverse f32
  - pairwise f32
  - chunked pairwise f32
  - fixed tile sizes: 4, 8, 16, 32, 64, 128, 256, 512
  - tile reduction order variants
- bias placement:
  - bias added before final cast
  - bias as initial accumulator
  - bias as term in reduction tree
- output:
  - final BF16 cast once
  - no intermediate BF16 core unless explicitly evidence-only

Required sampled set:

```text
6, 10, 13, 16, 18, 21
```

Success:

- One single named policy clears every sampled layer full-vector exactly:
  - `full_vector_mismatches = 0`
  - `max_abs_diff = 0`

Failure:

- no global policy clears
- only per-layer policies clear
- only focus lanes clear
- only diagnostics clear

## Gate C — CUDA Mirror Only After CPU Policy Clear

Future branch, only if Gate B succeeds:

```text
validation/fused-linear-addmm-cuda-policy-mirror
```

Purpose:

Implement a narrow validation-only CUDA mirror of the exact CPU-clearing policy.

Rules:

- no cuBLAS/cuBLASLt guessing
- no runtime/default routing
- correctness first, performance later
- deterministic kernel is acceptable even if slow
- GPU1 default for single-GPU work because displays are on GPU0
- no full Torch GPU model loading unless separately authorized and sharding
  readiness is reviewed

Success:

- CUDA mirror clears the sampled set full-vector exactly against producer/API
  references.

Failure:

- CUDA cannot reproduce the CPU-clearing policy exactly.

## Promotion Proof Gate

Even if a CUDA mirror clears, runtime behavior is still not promoted.

Future required doc:

```text
docs/FUSED_LINEAR_ADDMM_POLICY_PROMOTION_PROOF_PLAN.md
```

That plan must require:

- regression guards
- negative controls
- consumer revalidation plan
- server/final-logit claim separation
- performance implications
- fallback behavior
- clear separation between validation helper and runtime default

## Stop Conditions

Stop this lane if any of the following occur:

- Torch CPU output is not stable across backend/thread settings.
- No single Rust CPU policy clears the sampled set.
- Only per-layer policy choices clear.
- CUDA mirror cannot reproduce the CPU policy.
- Candidate requires tolerance, correction metadata, or f64 diagnostic
  promotion.
- Candidate requires production routing changes to validate.

If any stop condition occurs, preserve Workstream A as an official Torch API
seam and do not continue implementation in this lane without a new design
review.

## Future Scaling And GPU Note

CPU policy synthesis remains the immediate path.

GPU Torch oracle generation is future work only. Single-GPU work defaults to
GPU1 because displays are on GPU0. Full Torch model loading on one 24 GB GPU is
expected to be fragile or OOM.

Before GPU Torch attribution or large GPU oracle generation, inspect:

```text
/home/emmy/openai/worktrees/runtime-multi-gpu-layer-sharding/docs/
```

Do not modify that worktree from this branch.

## Guardrails

- Docs-only.
- No implementation authorized.
- No backend selected.
- No consumer revalidation authorized.
- No runtime/default/CUDA behavior change.
- No output emission.
- No ladder continuation.
- No correction metadata promotion.
- No tolerance pass.
- No final-logit claim.
- No all-layer claim.
- No server claim.
- No 4097/context-length claim.
- No Torch runtime dependency in Rust.

## Gate A Result — CPU Torch Dispatch-Stability

Implementation branch:

```text
oracle/fused-linear-addmm-cpu-dispatch-stability
```

Status:

```text
/tmp/fused_linear_addmm_cpu_dispatch_stability_status.json
```

Classification:

```text
fused_linear_addmm_cpu_dispatch_stability_stable
```

Result:

- All required fresh-process configurations executed.
- Tested configurations included default, `torch.set_num_threads(1)`,
  `torch.set_num_threads(8)`, interop thread variants, MKLDNN enabled/disabled,
  `OMP_NUM_THREADS` variants, `MKL_NUM_THREADS` variants, and combined
  OMP/MKL variants.
- For layers 6, 10, 13, 16, 18, and 21, every tested configuration reproduced
  the baseline `torch.addmm(bias, input_2d, weight_t_2d)` full vector exactly.
- Every tested configuration also matched the official o-proj artifact
  full-vector exactly.
- Baseline explicit matmul/einsum/unfused-bias sanity controls remained
  negative controls.

Gate A passed for this sampled seam. This does not authorize Gate B by itself;
Rust CPU policy synthesis still requires explicit approval. No backend is
selected, no implementation is authorized, no consumer revalidation is
authorized, and no runtime/default/CUDA behavior change follows from this
result.

## Gate B Result — Rust CPU Policy Synthesis

Implementation branch:

```text
validation/fused-linear-addmm-rust-cpu-policy-synthesis
```

Status:

```text
/tmp/fused_linear_addmm_rust_cpu_policy_synthesis_status.json
```

Classification:

```text
fused_linear_addmm_rust_cpu_policy_synthesis_partial_only
```

Result:

- The validation-only Rust CPU mode enumerated 350 named policy records across
  BF16-input/f32-product, BF16-rounded-product evidence, f64 diagnostic,
  forward/reverse/pairwise/chunked/tiled accumulation, bias-placement, and
  output-cast dimensions.
- Full-vector replay was only run after a candidate cleared the layer focus
  lane, because full-vector exactness implies focus-lane exactness.
- No single selectable Rust CPU policy cleared the sampled set
  6/10/13/16/18/21 full-vector exactly.
- Per-layer full-vector clears appeared for layers 10, 13, and 21, but not for
  layers 6, 16, or 18.
- The best selectable full-vector replays still had residual mismatches on
  layer6 (1 mismatch), layer16 (2 mismatches), and layer18 (1 mismatch).
- Focus-lane-only, diagnostic-only, and evidence-only candidates remain
  rejected.

Gate B did not pass. Gate C CUDA mirror work is not authorized from this
result. No backend is selected, no implementation is authorized, no consumer
revalidation is authorized, and no runtime/default/CUDA behavior change follows.

## Gate B Closure Audit Result

Implementation branch:

```text
validation/fused-linear-addmm-rust-cpu-policy-closure-audit
```

Status:

```text
/tmp/fused_linear_addmm_rust_cpu_policy_closure_audit_status.json
```

Classification:

```text
fused_linear_addmm_rust_cpu_policy_closure_no_global_policy
```

Result:

- The closure audit replayed all 238 selectable focus-clearing candidates that
  Gate B had rejected only because bounded full-vector replay had not executed.
- No replay was blocked.
- After closure, no single selectable Rust CPU policy cleared layers
  6, 10, 13, 16, 18, and 21 full-vector exactly.
- The best near-global candidate cleared 3 of 6 sampled layers and still left
  5 total mismatches across the sampled set.
- Residual samples for layers 6, 16, and 18 were one BF16 ULP or less in the
  simple residual analysis, but no shared tie/rounding rule was localized.
- The recommended next state is
  `stop_policy_lane_preserve_official_api_seam`.

This closes the current bounded Gate B policy space. Gate C CUDA mirror work is
not authorized. The official CPU Torch module/F.linear/_C/addmm seam remains
the Workstream A oracle boundary. No backend is selected, no implementation is
authorized, no consumer revalidation is authorized, and no runtime/default/CUDA
behavior change follows.

## PyTorch Source Attribution Plan

Plan:

```text
docs/FUSED_LINEAR_ADDMM_PYTORCH_SOURCE_ATTRIBUTION_PLAN.md
```

Classification:

```text
fused_linear_addmm_pytorch_source_attribution_plan_recorded
```

Because Gate B closure found no global Rust CPU policy, any future attempt to
reopen Rust/CUDA policy synthesis should first inspect the PyTorch/ATen source
and dispatch path behind the stable CPU Torch addmm seam. The plan defines an
external research workspace under `/home/emmy/openai/pytorch*` with isolated
virtual environments. This docs branch performs no PyTorch clone, venv setup,
source build, consumer revalidation, CUDA mirror work, or runtime/default/CUDA
behavior change.
