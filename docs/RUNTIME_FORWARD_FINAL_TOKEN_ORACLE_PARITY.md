# Runtime-Forward Final-Token Oracle Parity

This records the first promotion breadcrumb for the runtime-forward final-token
oracle parity milestone. It does not promote runtime code or claim default
runtime parity.

## Milestone

- Source branch: `feature/runtime-forward`
- Milestone commit: `5bcba1d2edcb9c15b1ed567700976dad03e12300`
- Exact case: `developer-message-user-smoke`
- Source artifact: `.live/runtime-forward-final-readout-20260423/developer-message.runner-final-readout-direct-module-rerun-status.json`
- Classification: `final_readout_direct_module_logits_cleared`

The proof path matched official for the final token through the final
transformer stack, final norm, and LM-head logits. The regenerated direct-module
LM-head logits digest was:

```text
67f31845dd24db26cc91954607cfae8ae7ff7b9c8954cb9d3b1610ca9c635209
```

The prior PPP logits digest
`5a7d47edfab63d59c17825b8d7b7668cc7a15ad2d107f902ca2caa05488ecd44`
was stale and mismatched on 37 tail logits. The runtime-forward local logits
matched the regenerated direct-module PPP logits.

## Validation

Validation recorded on `feature/runtime-forward`:

- `cargo fmt`
- `cargo check -p gpt-oss-bench --bin runtime_forward_layer0_qkv_bf16_candidate_status --features cuda`
- `final-readout-direct-module-rerun-status`
- `jq` validation
- `git diff --check`

## Promotion Boundary

This milestone is full final-token oracle parity for the proof path only.
Proof and candidate mechanisms remain bench/proof-only unless separately
reviewed and promoted, including:

- oneDNN Q/K/V projection candidate paths
- selected expert output readout correction
- broad debug capture plumbing
- cuBLAS BF16/pedantic helpers
- large raw PPP and full-value artifacts

Promotion should proceed in small commits:

1. Docs/status breadcrumb first.
2. Harness extraction next, only if it can be done without runtime-affecting
   changes.
3. Narrow runtime candidates later, each with focused review and validation.
