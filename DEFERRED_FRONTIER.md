# Deferred Frontier

This note records durable deferred-frontier conclusions.
It does not widen trusted support claims, change runtime behavior, or serve as an implementation plan.

## Stable Conclusions

- Graph remains a runtime replay frontier, not a checkpoint-shape frontier.
- `sliding_attention` is the artifact-backed term for the first deferred case.
- `layer_types` is present in the shipped Hugging Face config.
- Learned `sinks` are not the same thing as repo-local `sink_tokens`.

## Deferred Order

The deferred order remains:

1. one concrete `sliding_attention` case
2. the same case with learned `sinks`
3. graph replay for an already-proven decode shape

## Guardrails

- Presence of code paths, config fields, or experimental plumbing does not imply trusted support.
- Learned `sinks` and repo-local `sink_tokens` should remain distinct.
- Deferred-frontier notes should stay scoped to conclusions worth carrying on `main`.
