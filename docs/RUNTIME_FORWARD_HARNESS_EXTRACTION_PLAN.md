# Runtime-Forward Harness Extraction Plan

Classification: `harness_extraction_ready_for_docs_and_schema_tools`

This audit covers whether proof/harness material from `feature/runtime-forward`
can be extracted onto `promotion/runtime-forward-final-token-oracle-parity`
without promoting runtime-affecting code.

## Source

- Current branch: `promotion/runtime-forward-final-token-oracle-parity`
- Source branch: `feature/runtime-forward`
- Source worktree: `/home/emmy/openai/worktrees/runtime-forward`
- Source milestone commit: `5bcba1d2edcb9c15b1ed567700976dad03e12300`
- Source classification: `final_readout_direct_module_logits_cleared`

## Compatibility Finding

The full Rust bench binary
`crates/gpt-oss-bench/src/bin/runtime_forward_layer0_qkv_bf16_candidate_status.rs`
is not safe to extract directly. It is about 45k lines and depends on
runtime-forward-only CUDA/model-runner APIs, including `GpuWorker`,
`Layer0QkvTrace`, `Layer0QGemmTrace`, BF16 GEMM invocation records, `cudarc`,
and many `.live` or `/tmp` artifact paths.

The current integration branch has `gpt-oss-bench` bins and Python tools, but
does not have `runtime_forward_layer0_qkv_bf16_candidate_status.rs`. A direct
bin import would require `Cargo.toml` dependency changes and runtime API
alignment, so it is deferred.

## Candidate Table

| File or group | Purpose | Can land without runtime code? | Dependencies | Risk | Recommended action |
| --- | --- | --- | --- | --- | --- |
| `runtime_forward_layer0_qkv_bf16_candidate_status.rs` | Monolithic layer0 through final-readout status driver | No | CUDA feature, `cudarc`, `gpt-oss-model-runner::gpu_runner` debug types, `GpuWorker`, `.live`, `/tmp` artifacts | High | Do not import. Split later only after runtime API alignment. |
| `final_readout_direct_module_rerun.py` | Recompute final norm and LM-head direct-module proof | Not as first extraction | Torch, model checkpoint, ladder/readout helper modules, full PPP direct-module bundle | Medium | Defer. Candidate for a later Torch-tool extraction if artifacts and model-root assumptions are documented. |
| `final_readout_norm_and_lm_head_compare.py` and `lm_head_weight_layout_arithmetic_policy.py` | Final norm and LM-head comparison/discriminator | Not as first extraction | Torch, checkpoint modules, ladder helper, full readout artifacts | Medium | Defer behind schema/manifest tooling. |
| `layer1_attention_ordered_bundle_compare.py`, `layer1_mlp_ordered_bundle_compare.py`, `layer2_to_final_coarse_layer_ladder_compare.py` | Ordered bundle and coarse ladder comparison | Not as first extraction | Torch, checkpoint loading, source bundle artifacts, large PPP bundles | Medium | Defer. Useful later as proof tools, but not minimal. |
| `layer0_attention_*` and `layer0_mlp_*` compare scripts | Layer0 attention/MLP proof localization | Mostly no | Torch, runtime-forward status artifacts, model checkpoint, candidate paths | Medium to high | Defer; import only selected pure comparison helpers later if isolated from checkpoint replay. |
| `layer0_*onednn*`, `layer0_q_projection_*`, `layer0_v_projection_*` | oneDNN Q/K/V candidate proof | No | Torch/MKLDNN policy, model checkpoint, candidate artifacts, verbose logs | High | Do not extract in harness-first slice. Keep proof-only. |
| Selected expert output readout scripts | Selected expert output capture/readout correction proof | No | Torch, source mismatch artifacts, candidate correction logic | High | Do not extract before an approved proof-only design. |
| Manifest/checksum validation helper | Validate manifest/checksum references and small status JSON presence | Yes | Python stdlib or jq, manifest paths | Low | Best first harness extraction candidate. |
| Status JSON schema checker | Validate final readout artifact classification, digest, and stale PPP note | Yes | Python stdlib or jq, local/source artifact path | Low | Best first harness extraction candidate. |
| Docs-only workflow note | Preserve exact commands and artifact expectations | Yes | None | Low | Safe immediately. |

## Minimal Extraction Candidate

The smallest safe next harness commit is a docs/schema-tools extraction, not a
GPU bench import:

1. Add a tiny Python stdlib-only validator such as
   `crates/gpt-oss-bench/tools/runtime_forward_final_readout_status_check.py`.
2. Make it consume an explicit JSON path and check:
   - `classification == final_readout_direct_module_logits_cleared`
   - exact case `developer-message-user-smoke`
   - final block, final norm, and LM-head metrics are matched with zero diff
   - direct-module LM-head digest
     `67f31845dd24db26cc91954607cfae8ae7ff7b9c8954cb9d3b1610ca9c635209`
   - stale PPP digest note
     `5a7d47edfab63d59c17825b8d7b7668cc7a15ad2d107f902ca2caa05488ecd44`
3. Optionally add a second stdlib-only manifest checker for
   `LARGE_ARTIFACTS_MANIFEST.json` and `SHA256SUMS`, but keep it independent of
   `.live` being tracked.
4. Validate scripts with `python -m py_compile`.

This avoids CUDA execution, runtime model execution, model checkpoints,
Torch/MKLDNN policy assumptions, and ignored `.live` tracking policy.

## Do Not Extract

- Full GPU/CUDA bench modes.
- oneDNN Q/K/V candidate code.
- selected expert output readout correction code.
- broad debug capture plumbing.
- runtime-affecting helper APIs.
- raw large PPP/full-value artifacts.
- `.live` artifacts unless a separate tracked artifact policy is approved.

## Runtime Candidates Deferred

| Candidate | Promote now? | Reason | Required proof artifact | Future branch or commit name |
| --- | --- | --- | --- | --- |
| RoPE half-split pairing fix | Defer | Runtime kernel behavior change, not harness infrastructure | K RoPE/grouped compare status and runtime smoke | `runtime: promote gpt-oss rope half-split pairing` |
| Layer0 BF16 RMSNorm policy | Defer | Default runtime path change for layer0 f16 path | RMSNorm scalar/runtime-fix status and final-token proof | `runtime: promote layer0 bf16 rmsnorm policy` |
| cuBLAS BF16/pedantic helpers | Defer | GPU API/helper behavior needs isolated review | all-token K pedantic/runtime-fix status | `runtime: add scoped bf16 cublas helpers` |
| Q/K/V projection-policy candidates | No | Candidate/proof-only oneDNN policy paths, not default runtime | Q/K/V oneDNN scoped candidate statuses | `proof: retain qkv projection candidates` |
| Selected expert output readout correction | No | Proof readout correction only; not runtime behavior | selected expert readout fix/localization statuses | `proof: retain selected expert readout correction` |
| Debug capture plumbing | Defer | Broad public/debug API surface and runtime coupling | final readout and ordered bundle status artifacts | `harness: expose scoped debug capture APIs` |

## Validation Summary

Prior validation on this promotion branch:

- `git diff --check`: passed.
- `cargo metadata --no-deps`: confirmed `gpt-oss-bench` exists.
- `find crates -maxdepth 3 -type f -name 'runtime_forward_layer0_qkv_bf16_candidate_status.rs'`: target absent.
- `cargo check -p gpt-oss-bench`: failed on existing integration CUDA feature-gating in `restricted_prefill_topk`.
- `cargo check -p gpt-oss-bench --lib`: passed with pre-existing warnings.

This plan is docs-only; no additional cargo check is required beyond
`git diff --check`.

## Recommended Next Commit

Add a stdlib-only final-readout status validator and optionally a
manifest/checksum reference validator. Do not import the monolithic Rust bench
binary or Torch replay tools until a separate extraction can isolate them from
runtime-forward-only APIs and large local artifacts.
