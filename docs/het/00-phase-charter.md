# Phase charter

## Direction, not a design

**Verified (assignment direction):** continue `gpt-oss-rs` as a clean, slim,
fast, correct, open Rust runtime. Make exact static heterogeneous GPT-OSS-120B
execution across CPU, GPU0, and GPU1 the next major proving target. Use
GPT-OSS-20B as the smaller correctness and integration control. Freeze Tiger
Lake Xe as an active optimization target. Defer Qwen and upstream integration.

The workstation is the first proving ground. Host facts may bound experiments,
but must not become implicit runtime identity checks or universal policy.

## Four-stage boundary

| Stage | Authorized outcome | Explicit stop |
|---|---|---|
| 1. Pre-research / pre-planning (this record) | Establish repository, host, checkpoint, execution, validation, and evidence baselines; expose conflicts and research questions. | No external prior-art conclusions, design selection, implementation plan, or code change. |
| 2. Research | Inspect queued prior art and controlled local checkouts; run targeted topology, memory, transfer, concurrency, and model experiments; compare candidate seams. | No implementation before research review. |
| 3. Planning | Select one design and define invariants, work packages, tests, benchmarks, evidence, rollback, and promotion gates. | No unreviewed implementation. |
| 4. Implementation | Implement only the approved plan and promote only after exactness and evidence gates pass. | No adaptive or approximate expansion before the static exact proof. |

**Deferred:** online popularity adaptation, expert migration, placement
prediction/prefetch, cache prediction, approximate expert deferral, automatic
dispatch promotion, Qwen, renewed Xe optimization, generic-server redesign,
upstream PR work, and protocol expansion.

## Provisional proof target

The following is **provisional until research and planning complete**:

> GPT-OSS-120B produces a correct retained continuation while at least one
> model layer executes selected experts across CPU, GPU0, and GPU1 within one
> inference operation, with exact routing and weighted reduction, bounded
> weight/scratch/staging memory, deterministic failure behavior, and complete
> per-layer/per-device/transfer timing. GPT-OSS-20B supplies the smaller control
> and integration path.

Static placement is the expected first-design bias, not a selected design.
Research must determine both the smallest honest proof case and the useful
steady-state policy. CPU and GPU work need not overlap in every layer or token
unless evidence shows that requirement is necessary.

## Promotion invariants to preserve for later planning

These are constraints, not an implementation plan:

1. Routing logits, stable top-k/tie behavior, selected-weight normalization,
   expert arithmetic, rank-ordered weighted reduction, BF16 boundaries, and
   residual/KV commit semantics must remain exact against the accepted oracle.
2. Prefill and decode evidence must be separate. A CUDA decode kernel does not
   prove CUDA prefill, and `sm_86` compilation does not prove model inference.
3. Every persistent representation and temporary workspace must have an owner,
   device, lifetime, byte bound, cleanup path, and failure disposition.
4. CPU work, GPU-local work, H2D, D2H, GPU-to-GPU transfer, and synchronization
   must be measured separately.
5. “Both GPUs work in tests” is not evidence that one inference spans them.
6. The installed CPU's runtime capabilities, not a product-generation name,
   must select CPU kernels. Feature-gated compilation remains separate evidence.
7. A request may publish tokens or commit mutable state only after all required
   device work for that step has succeeded or a defined recovery has completed.

## Evidence and decision discipline

- Source-derived facts cite repository-relative files and symbols.
- Local evidence is summarized with a timestamp and sanitized command; raw
  hardware identifiers are intentionally omitted.
- Calculations state dimensions, stored widths, units, and alignment assumptions.
- Existing seams are classified as candidates, never as authorization to
  genericize or refactor them.
- Historical evidence is usable only within the authority declared by current
  repository policy. Retired oracle captures cannot support new parity claims.
- A conflict remains visible until a named check resolves it.

## Completion and handoff gate

Stage 1 is complete when the linked records map the present runtime and evidence
to concrete source, capture sanitized host/checkpoint metadata, distinguish
implemented/validated/scaffolded/absent/unknown capabilities, and prioritize
the questions blocking design selection. The handoff then stops for review.
