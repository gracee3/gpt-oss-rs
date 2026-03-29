# Deferred Frontier Shape Audit

This note is a reconnaissance artifact only.
It does not widen trusted support claims or change current runtime behavior.

## Scope

This audit compares three sources:

- local checkpoint/config artifacts under `/data/models/openai`
- upstream reference material in `~/openai/gpt-oss`
- current `gpt-oss-rs` semantics/runtime/reference/conformance code

The goal is to clarify actual model shapes and naming surfaces for deferred frontiers:

- graph
- sliding/local attention
- sink attention

## Artifact Reality

### Checkpoint / config facts

From `/data/models/openai/gpt-oss-20b/config.json`:

- `model_type = "gpt_oss"`
- `num_hidden_layers = 24`
- `hidden_size = 2880`
- `head_dim = 64`
- `num_attention_heads = 64`
- `num_key_value_heads = 8`
- `sliding_window = 128`
- `initial_context_length = 4096`
- `max_position_embeddings = 131072`
- `rope_theta = 150000`
- `rope_scaling.rope_type = "yarn"`
- `attention_bias = true`

Most important shape fact:

- `layer_types` is explicitly present and alternates:
  - even layers: `sliding_attention`
  - odd layers: `full_attention`

From `/data/models/openai/gpt-oss-20b/model.safetensors.index.json`:

- every layer has `model.layers.<i>.self_attn.sinks`
- sinks are not limited to sliding layers
- HF-export attention weights are split into:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`

From `/data/models/openai/gpt-oss-20b/original/dtypes.json`:

- original checkpoint naming is `block.<i>.attn.*`
- original attention uses fused `attn.qkv.weight` / `attn.qkv.bias`
- original sink name is `block.<i>.attn.sinks`

Practical implication:

- checkpoint artifacts already encode the true alternating attention schedule
- sink tensors are learned per-layer weights, not just a runtime visibility flag
- the loader/reference surface must translate between original `block.*` names and HF-export `model.layers.*` names

### Upstream `gpt-oss` facts

Upstream uses:

- `sliding_window` as the config field
- parity-by-layer-index for alternating sliding/full behavior
- `sinks` as learned per-head logits applied in attention
- `graph` only for CUDA graph replay in decode runtime

Upstream does **not** use:

- `layer_types`
- `local_attention`
- `global_attention`
- `sink_tokens`

Practical implication:

- `layer_types` is a local normalization layer, not an upstream concept
- `local_attention` is a local alias only
- `sink_tokens` is a local modeling term and should not be confused with upstream learned sink logits

## Naming Drift

### Attention naming

Current repo:

- semantics/runtime-plan/reference accept both:
  - `sliding_attention`
  - `local_attention`
- semantics/reference also accept:
  - `full_attention`
  - `global_attention`

But current runner fast paths mostly branch on canonical strings only.

Drift:

- artifact truth: `sliding_attention` and `full_attention`
- local aliasing: `local_attention` and `global_attention`

Risk:

- configs can validate in semantics/planner/reference and miss the intended GPU/model-runner path

### Sink naming

Current repo uses several incompatible sink concepts:

- semantics: `SinkBehavior::{Disabled, Available}`
- reference/cache: `sink_tokens`
- model-runner/checkpoint: `sinks` tensor
- GPU code: “sink attention” sometimes means sink logits plus sliding-window path together

Artifact truth:

- upstream/checkpoint `sinks` are learned per-head logits
- they are not the same thing as a visible-prefix token-count policy

Risk:

- future sink/sliding work will mix incompatible schemas unless the vocabulary is normalized first

### Phase naming

Current repo splits the same concept across:

- `StepKind`
- `RequestKind`
- `ReferencePhase`
- raw `is_prefill: bool`

Risk:

- as frontier-specific cases grow, serializer/trace/comparison seams will multiply

## Concrete Drift Findings

### 1. `layer_types` is now artifact-backed

Earlier recon treated `layer_types` partly as a local planning abstraction.
The shipped HF config already contains the explicit alternating list.

Revised conclusion:

- for HF-export checkpoints, `layer_types` is real artifact truth
- for upstream/original-format checkpoints, parity may still need to be synthesized from layer index

### 2. `local_attention` remains a repo-local alias

Artifacts and upstream reference material use `sliding_window` / `sliding_attention` concepts, not `local_attention`.

Revised conclusion:

- smallest honest first case should use `sliding_attention`, not a generic `local_attention` label

### 3. `sink_tokens` and `sinks` are not the same frontier knob

Artifacts show learned `sinks` tensors on every layer.
Reference/cache code models `sink_tokens` as a prefix-visibility policy.

Revised conclusion:

- sink frontier work must keep these separate:
  - learned sink logits
  - visible-token retention policy

### 4. Graph has no checkpoint schema

I did not find graph topology/config metadata in checkpoint artifacts.
Readable vLLM cache JSONs only exposed compile/cache environment flags, not model graph schema.

Revised conclusion:

- graph remains a runtime-execution frontier, not a checkpoint-shape frontier

## Revised Smallest Honest First Cases

### Sliding / local attention

Revised smallest honest first case:

- one concrete `sliding_attention` decode slice
- single sequence
- single layer
- `sliding_window = 128`
- no sinks enabled yet as the primary assertion
- no MoE
- no trusted-mode claim

Why narrower than prior recon:

- artifact truth already tells us the real window is `128`
- the concrete exported term is `sliding_attention`
- this is a more honest first case than a generic “local attention family” slice

### Sink attention

Revised smallest honest first case:

- the same concrete `sliding_attention` decode slice
- with nonzero learned `sinks` enabled on that same attention row
- still single sequence
- still single layer
- still no MoE
- still experimental / non-trusted

Why this is tighter than prior recon:

- artifact truth shows sinks are per-layer learned logits
- upstream applies them inside the same attention row, not as a separate cache subsystem
- sink should be treated as the next increment on the same decode slice, not a separate broad frontier

### Graph

Revised smallest honest first case:

- decode-only graph replay
- fixed padded batch bucket
- dense full-attention or already-proven attention row semantics
- no prefill capture claim
- no trusted-mode claim

Why unchanged in principle:

- graph is still a runtime replay concern
- artifacts did not reveal a smaller checkpoint-driven graph surface

## Revised Ordering

The earlier ordering needs one refinement.

Old summary:

1. sliding/local
2. sink
3. graph

Refined summary:

1. one concrete `sliding_attention` decode case
2. the same case with learned `sinks` enabled
3. graph replay for an already-proven decode shape

So the broad order still stands:

- sliding/local first
- sink second
- graph last

But the more honest interpretation is:

- sink is not a separate large memory frontier
- sink is the next increment on top of the same sliding decode shape

## Highest-Risk Drift Seams

- `local_attention` / `global_attention` alias support is inconsistent across planner/semantics/reference vs runner/GPU code.
- `sink_tokens` vs learned `sinks` is the most important schema mismatch.
- conformance observed traces are still too synthetic for sliding/sink shape claims.
- selective MoE metadata is richer in reference/conformance than in canonical runtime/semantics config types.

## Notes

- One potentially useful but unreadable artifact on this host is:
  - `/data/models/openai/.cache/vllm/vllm/modelinfos/vllm-model_executor-models-gpt_oss-GptOssForCausalLM.json`
- It is root-readable only here, so it was not used as evidence in this note.
