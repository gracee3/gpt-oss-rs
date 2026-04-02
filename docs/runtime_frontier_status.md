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

Current blocker:

- completed post-4096 same-input runtime observation still does not emit a finished artifact in bounded time

Next honest proof target:

- the smallest post-4096 runtime seam that captures one localized last-token value and exits immediately
- preferred order:
  - post-RoPE `q`
  - post-RoPE `k`
  - post-attention context
  - post-attention residual

Guardrail:

- none of the safe extraction commits or current proof seams should be treated as proof of full GPT-OSS runtime correctness
