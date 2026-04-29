# Ordered Bundle Generation Plan

## Purpose

This lane is a docs-first staging lane for generating ordered attention and MLP
oracle bundles beyond layer1. Its output should be reference artifact contracts
and capture plans for the validation-runtime handoff branch, not production
runtime behavior.

The immediate target is a bounded layer2 pilot. Layers 2..23 must not be
generated in bulk until the layer2 ordered bundle schema is validated by the
consumer.

## Consumer Need

The consumer branch, `projection/layer0-validation-runtime-handoff`, now has
generic bundle-driven ladder scaffolding:

- `crates/gpt-oss-bench/src/bin/layer0_validation_runtime_path.rs`
  - `--mode layer-bundle-discover`
  - `--mode layer-bundle-validate`
  - `--mode layer-ladder`
- Default artifact root:
  `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424`

That scaffold stops at layer2 because it looks for filenames containing
`layer2`, `attention` or `mlp`, `ordered`, `boundary`, and `bundle`, and no
ordered layer2 attention or MLP bundle is present. The existing layer2-to-final
artifact is coarse ladder data only and cannot satisfy the ordered bundle
contract.

## Current Artifact Provenance

The pinned artifacts inspected were:

- `developer-message.ppp-layer1-final-token-attention-ordered-boundary-bundle-status.json`
- `developer-message.ppp-layer1-final-token-mlp-ordered-boundary-bundle-status.json`
- `developer-message.ppp-layer2-to-final-final-token-coarse-layer-ladder-bundle-status.json`

All three use schema
`pinned-prompt-official-intermediate-capture-output/v2` and carry
`backend = official_torch`. They do not embed the generating command or a
producer path in the JSON itself.

The producing code was found outside this lane in the sibling
`pinned-prompt-parity` worktree:

- `/home/emmy/openai/worktrees/pinned-prompt-parity/crates/gpt-oss-bench/tools/pinned_prompt_official_capture.py`
- Dispatch function:
  `capture_intermediate(...)`
- Runner path:
  `run_distributed_gpu_capture(...)` or `run_cpu_capture(...)`, both loading
  `Transformer.from_checkpoint(...)` and then calling `capture_intermediate`.

This lane currently contains consumer/validator code for ordered bundles, but
not the original official capture producer script.

## Layer1 Attention Ordered Path

The layer1 attention ordered bundle was produced by the official PyTorch capture
helper path:

1. `pinned_prompt_parity capture-official-intermediate`
2. helper script `pinned_prompt_official_capture.py`
3. `capture_intermediate(...)`
4. boundary selector
   `layer1_final_token_attention_ordered_boundary_bundle`
5. function
   `capture_layer1_final_token_attention_ordered_boundary_bundle(...)`

The function is hard-coded to layer1. It computes:

- embedding and layer0 full block output as the layer1 input
- layer1 attention norm and Q/K/V projections
- Q/K RoPE, grouped K history, raw scaled QK, masked logits, softmax
- weighted V, o_proj, and attention residual add

The captured boundaries are:

- `layer1_final_token_q_projection_output_before_rope`
- `layer1_final_token_k_projection_output_before_rope`
- `layer1_final_token_v_projection_output_before_attention`
- `layer1_final_token_q_post_rope_before_attention`
- `layer1_grouped_k_post_rope_before_attention`
- `layer1_final_token_raw_scaled_qk_logits_pre_mask`
- `layer1_final_token_masked_scaled_qk_logits_pre_softmax`
- `layer1_final_token_attention_probs_post_softmax`
- `layer1_final_token_attention_weighted_value_sum_before_output_projection`
- `layer1_final_token_attention_output_after_o_proj_before_residual`
- `layer1_final_token_hidden_state_after_attention_residual_add_before_mlp`

The consumer-side validation path is
`execute_layer_attention_bundle_validation(...)` in
`crates/gpt-oss-bench/src/bin/layer0_validation_runtime_path.rs`. That path can
accept a generic `layer` index, but it only consumes an already-generated
ordered bundle.

## Layer1 MLP Ordered Path

The layer1 MLP ordered bundle was produced by the official PyTorch capture
helper path:

1. `pinned_prompt_parity capture-official-intermediate`
2. helper script `pinned_prompt_official_capture.py`
3. `capture_intermediate(...)`
4. boundary selector `layer1_final_token_mlp_ordered_boundary_bundle`
5. function `capture_layer1_final_token_mlp_ordered_boundary_bundle(...)`

The function is hard-coded to layer1. It computes:

- embedding, layer0 full block, and layer1 attention output
- layer1 MLP norm
- router logits and sorted top-k routing
- selected expert MLP1, SwiGLU, MLP2, and selected expert outputs
- weighted expert sum and final MLP residual add

The captured boundaries are:

- `layer1_final_token_mlp_norm_output_before_mlp_projections`
- `layer1_final_token_mlp_router_logits_before_routing`
- `layer1_final_token_mlp_topk_expert_indices_and_routing_weights`
- `layer1_final_token_selected_expert_outputs_before_routing_weighted_sum`
- `layer1_final_token_mlp_output_after_routing_weighted_sum_before_residual`
- `layer1_final_token_hidden_state_after_mlp_residual_add`

The pinned layer1 selected expert order is `[28, 6, 1, 18]`.

The consumer-side validation path is
`execute_layer_mlp_bundle_validation(...)` in
`crates/gpt-oss-bench/src/bin/layer0_validation_runtime_path.rs`. It already
uses the generic boundary names for a requested layer and replays the local MLP
backend from the ordered bundle.

## Layer2-To-Final Coarse Path

The layer2-to-final coarse ladder bundle was produced by the official PyTorch
capture helper path:

1. `pinned_prompt_parity capture-official-intermediate`
2. helper script `pinned_prompt_official_capture.py`
3. `capture_intermediate(...)`
4. boundary selector `layer2_to_final_final_token_coarse_layer_ladder_bundle`
5. function
   `capture_layer2_to_final_final_token_coarse_layer_ladder_bundle(...)`

The function runs embedding, layer0, and layer1 first, then iterates layers
2..final. For each layer it records only final-token coarse ladder boundaries:

- `layerN_final_token_layer_input_before_attention_norm`
- `layerN_final_token_attention_norm_output_before_qkv`
- `layerN_final_token_hidden_state_after_attention_residual_add_before_mlp`
- `layerN_final_token_mlp_norm_output_before_mlp_projections`
- `layerN_final_token_hidden_state_after_mlp_residual_add`

The bundle explicitly does not capture Q/K/V, RoPE internals, attention scores,
attention probabilities, selected expert internals, logits, or LM head data.
Consumer-side coarse inspection already confirmed
`can_support_ordered_bundle_adapter = false`.

## Layer2 Ordered Entrypoints

No layer2 ordered attention or MLP producer entrypoint was found in this lane.
The original official producer script has only these ordered selectors:

- `layer1_final_token_attention_ordered_boundary_bundle`
- `layer1_final_token_mlp_ordered_boundary_bundle`
- `layer2_to_final_final_token_coarse_layer_ladder_bundle`

There is no generic
`layerN_final_token_attention_ordered_boundary_bundle` or
`layerN_final_token_mlp_ordered_boundary_bundle` selector in the current
captured producer evidence. The consumer schema is already generic enough to
load layer2 ordered bundles if their boundary names match the layer-indexed
contract; the missing piece is the official capture entrypoint.

Classification for this reconnaissance:
`ordered_bundle_generation_blocked_by_missing_capture_entrypoint`.

## Low-Memory Constraints

The risky point is loading and running the official PyTorch checkpoint through
`Transformer.from_checkpoint(...)` while MXFP4 expert weights are dequantized to
BF16. Existing docs record that full official model forward dequantized MXFP4
weights to BF16 and exceeded available 24 GB CUDA memory before all-token
layer1 QKV capture.

A sibling `official-capture-lowmem` worktree contains an operational wrapper:

- `/home/emmy/openai/worktrees/official-capture-lowmem/scripts/pinned_prompt_official_capture_lowmem.sh`
- serialized execution via `flock`
- isolated temp/cache directories
- CPU or two-GPU distributed mode
- preflight disk, memory, and `nvidia-smi` capture

However, that worktree's helper only supports the older small intermediate
boundaries (`final_token_post_final_norm_pre_unembedding` and
`transformer_layer_output`) and does not contain the layer1 ordered bundle or
layer2 coarse bundle producer functions. It is useful as wrapper design, not as
the complete ordered producer.

## Safe Reuse

Safe pieces for a layer2 pilot:

- The layer1 ordered schema and boundary naming pattern.
- The consumer-side generic validators in
  `layer0_validation_runtime_path.rs`.
- The low-memory wrapper ideas: serialized runs, explicit temp/cache roots,
  CPU mode where feasible, two-GPU distributed mode where required, and run
  status/preflight metadata.
- The layer2-to-final coarse bundle as a source of layer2 coarse boundary names
  and shape expectations only.

## Keep Out Of Consumer Branch

These parts should remain outside the consumer branch:

- raw official capture helper plumbing
- runtime-forward proof binaries
- `.live` artifacts and `/tmp` outputs
- host-specific low-memory wrapper state and logs
- CUDA debug/proof capture code
- Torch dependencies in Rust runtime code

Consumer-facing code should only see stable bundle contracts, compact fixture or
status references when needed, and validation-only readers.

## Stage 1 Plan

Stage 1 should be a layer2-only pilot:

1. Port or add a generic official capture entrypoint in the oracle generation
   lane, not in the consumer branch.
2. Start with `--boundary layerN_final_token_attention_ordered_boundary_bundle`
   and `--layer-idx 2`, or an equivalent explicit
   `layer2_final_token_attention_ordered_boundary_bundle` selector.
3. Reuse the layer1 attention function shape, but parameterize the layer index
   and advance hidden through preceding blocks before capturing layer2.
4. Emit only the layer2 attention bundle first and run
   `layer-bundle-discover`/attention validation against it.
5. Add the layer2 MLP bundle only after attention schema validation.
6. Validate filenames against the consumer discovery pattern:
   `*layer2*attention*ordered*boundary*bundle*status.json` and
   `*layer2*mlp*ordered*boundary*bundle*status.json`.
7. Keep run artifacts outside git until the schema and consumer validation are
   confirmed.
8. Stop after layer2. Do not generate layers 3..23 until layer2 validates.

## Non-Goals And Caveats

- No runtime production changes.
- No CUDA changes.
- No Torch runtime dependency in Rust.
- No committed raw tensor artifacts.
- No committed `.live` artifacts.
- No committed `/tmp` artifacts.
- No runtime-forward proof binaries imported into the consumer branch.
- No parity or final-logit claim.
- No 4097 work.
- No broad all-layer bundles.
- No bulk layers 2..23 until the layer2 pilot is schema-validated.
