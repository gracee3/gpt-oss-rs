# CUDA BF16 Layer0 Downstream Equivalence Summary

This is a concise status summary for the validation-only custom/decomposed BF16 projection-policy
branch. The detailed audit trail is in
`docs/CUDA_BF16_PROJECTION_POLICY_IMPLEMENTATION_PLAN.md`.

## Scope

- Exact case: `developer-message-user-smoke`.
- Source proof branch/commit:
  `feature/runtime-forward` at `5bcba1d2edcb9c15b1ed567700976dad03e12300`.
- Validation branch:
  `projection/cuda-bf16-biased-linear-kernel`.
- Validation mode:
  scratch/harness artifact generation and comparison only.
- Runtime status:
  no production runtime routing, no production CUDA kernel replacement, and no raw scratch artifacts
  committed.

## Validated Policy

- Custom/decomposed BF16 projection policy.
- BF16 inputs, weights, and biases.
- FP32 accumulation where relevant.
- BF16 output boundaries.
- Official/model RoPE call for Q/K RoPE pinning.
- Torch/model calls were used only for oracle/scratch artifact generation, not as a runtime
  dependency.

## Seam Table

| Seam | Result | max abs diff | mismatches | Notes |
| --- | --- | ---: | ---: | --- |
| Q/K raw QK | exact | `0.0` | `0` | Custom/decomposed Q/K with official/model RoPE. |
| V weighted sum | exact | `0.0` | `0` | V projection-boundary differences disappeared downstream. |
| Attention o-proj before residual | exact | `0.0` | `0` | BF16 output projection over downstream-equivalent weighted-V. |
| Attention residual before MLP | exact | `0.0` | `0` | BF16 residual add/output. |
| MLP norm | exact | `0.0` | `0` | BF16 input, FP32 RMS reduction, BF16 output. |
| Router logits | exact | `0.0` | `0` | BF16 router linear output. |
| Top-k/routing | exact | `0.0` | `0` | Experts `[3, 30, 11, 27]`; weights `[0.4453125, 0.2275390625, 0.189453125, 0.13671875]`. |
| Selected expert outputs | exact after fresh expert30 reconstruction | `0.0` | `0` | Prior expert30 mismatch was a stale/generated scratch artifact issue. |
| Weighted expert sum | exact | `0.0` | `0` | Fresh reconstructed expert30 selected output. |
| MLP residual / layer0 final-token output | exact | `0.0` | `0` | Full layer0 final-token output boundary clears. |

## Key Corrections

- The earlier large Q/K raw-QK mismatch came from the scratch RoPE generator, not the projection
  policy.
- The official/model RoPE call reproduced official K pre/post RoPE exactly.
- The V projection-boundary mismatch disappeared at the weighted-V BF16 boundary.
- The expert30 selected-output mismatch was a stale/generated scratch artifact issue; fresh local
  replay clears selected-output and weighted-sum parity.

## Decision

The custom/decomposed BF16 chain reaches exact layer0 final-token output in scratch validation for
the exact `developer-message-user-smoke` case. This supports the policy as a strong validation
candidate, not a production runtime route.

## Caveats

- Not production runtime routing.
- Not performance-optimized.
- Not all layers.
- Not all prompts.
- Not 4097.
- Not final logits/default server parity.
- Not a reason to commit raw scratch artifacts.
- Not permission to import the runtime-forward proof harness wholesale.

## Next Steps

1. Decide whether to build a validation-only runtime path that reproduces this chain inside
   Rust/CUDA without scratch Python artifact generation.
2. Add performance measurements for any custom kernel path.
3. Extend to another layer or another exact case before production routing.
4. Add a final-token/logit smoke once an integration-safe path exists.
5. Keep 4097 boundary work deferred until a shorter exact-case path is stable.
