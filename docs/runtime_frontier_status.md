# Runtime Frontier Status

Branch/worktree:

- `feature/runtime-forward`
- `~/openai/worktrees/runtime-forward`

Current frontier split:

- `838d3f8` is closed/deferred on the current runtime seam
- `bd49d35` is the only live semantic frontier

## `838d3f8` status

Decision:

- keep `838d3f8` closed/deferred

Why:

- same-input CUDA prefill boundary observation on an active sliding-sink case showed identical KV/cache visibility boundary metadata between:
  - the real nonzero-sink model
  - a zero-sink control clone
- the observed boundary values stayed the same:
  - `context_len0`
  - `seq_start_pos0`
  - derived sliding-window start
  - block-table preview
- only the sink tensor payload changed
- that is direct runtime evidence for extra sink-logit terms, not extra visible KV token offsets

What would be required to reopen it honestly:

- a new runtime implementation seam that explicitly carries extra visible KV offsets, not just sink logits
- or direct same-input runtime observation at a seam closer to the actual attention read set that shows extra prefix KV slots being consumed when sinks are active

Until one of those exists:

- do not treat `WorkerConfig::semantic_cache_layout_with_flavor(ExperimentalSinkAware { .. })` as promotable
- do not use nonzero learned sink tensors alone as proof of extra visible KV offsets

## `bd49d35` status

Current positive evidence:

- YaRN table-construction math matches the local HF/vLLM-style reference on the active GPT-OSS config
- a no-YaRN control case did not perturb runtime traces
- completed same-input sink-free `>4096` runtime evidence exists for:
  - post-RoPE `q/k`
  - post-attention context
  - post-attention residual
- retained-state `restricted_logit_diff` seam evidence now confirms the safe-side retained prefill reaches all of the following boundaries on the active `4096 + 1` case:
  - retained layer-0 late-token router traversal through `mlp_done`
  - retained layer-1 attention completion
  - retained layer-1 post-attention residual handoff
  - retained layer-1 `mlp_begin`
  - retained layer-1 `router_input_ready`

Current blocker:

- the retained-state seam still does not emit a continuation-token artifact in bounded time
- the retained-state seam still does not reach `decode1`
- there is still no safe/variant retained continuation comparison for the continuation token

Current retained-seam limitation:

- none of the retained-state results above should be treated as proof of full runtime correctness
- the retained-state findings are preserved here so future work can build on the confirmed boundaries even if the workflow pivots to a bounded live-smoke lane

Guardrail:

- none of the safe extraction commits or current proof seams should be treated as proof of full GPT-OSS runtime correctness
