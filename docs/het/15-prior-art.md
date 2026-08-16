# Prior-art findings

This comparison is question-led. It records techniques and constraints, not
project popularity, implementation selection, or novelty claims. Source
identities, licenses, and inspected locations are in the
[source ledger](11-source-ledger.md).

| Blocking question | Primary source evidence | Applies here | Does not establish |
|---|---|---|---|
| Can official GPT-OSS bytes be used without a destructive full conversion? | OpenAI's Torch reader maps native names and decodes native MXFP4 blocks/scales directly; its parameter map matches the local complete comparison. The Triton path performs backend-specific expansion/transposition/requantization after loading. | Native shard views plus backend-specific owner construction are valid families. Checkpoint identity and execution representation should be separate. | Bounded Rust view lifetimes, this host's residency peak, or bit-exact current CPU/CUDA parity. |
| Should hybrid inference move activations or weights? | Fiddler measures and models activation movement versus PCIe weight movement, then statically assigns popular experts to GPU and cold experts to CPU. | A selected route moves 5.76 KiB at decode while one packed expert is 13.24 MB locally; measurements confirm the order-of-magnitude argument. | GPT-OSS top-4 exactness, two GPUs without P2P, prefill balance, or lifecycle rollback. |
| How can CPU work overlap GPU work? | KTransformers uses persistent pinned regions and an explicit submit/sync boundary; its paper/code overlap routed CPU expert work with GPU work. | Pool pinned staging, issue independent CPU/GPU jobs, and join on explicit completion events. | Suitability of its AMX kernels on the installed non-AMX Xeon, deterministic routing-rank reduction, or a three-device commit. |
| How should selected experts be represented to a backend? | llama.cpp `mul_mat_id` and mistral.rs MXFP4 gather paths pass selected IDs/rows to packed expert operations instead of scanning every expert. llama.cpp then performs explicit routing-rank weighting/sum. | Selected-expert kernels and rank-bearing results are feasible boundaries; packed 17-byte records are independently corroborated. | That either API exactly matches this repository's BF16 round points, supports native safetensors directly, or assigns individual experts to CPU/GPU0/GPU1. |
| Is static placement enough? | Fiddler supports static hot/cold placement. HybriMoE reports phase- and request-varying popularity/load balance and proposes adaptive cache/scheduling. | Static placement is a valid first proof, but measurements and admission must account for bucket imbalance and prefill/decode differences. | That adaptive placement is needed before exact static execution. Migration, prediction, caching, and deferral remain deferred. |
| Can NCCL hide the no-P2P topology? | NVIDIA documents SHM fallback; pinned NCCL source plus local diagnostics prove SHM/direct on both channels. | NCCL worker/stream lifecycle concepts may inform error handling. | Peer access, low-latency arbitrary expert jobs, or correctness of the current tensor-parallel model. Direct pinned relay is simpler for the proof contract. |
| What Rust ownership patterns are relevant? | mistral.rs keeps model routing and selected-expert MXFP4 operations explicit and returns typed errors through device-aware modules. | Preserve model semantics outside a narrow backend job boundary; make device ownership and errors explicit. | An upstream-integration target, a ready API to copy, or transactional multi-device KV state. |

## Reusable patterns, with provenance limits

1. **Static single ownership and activation movement.** Fiddler is the clearest
   published precedent; local byte/latency evidence independently justifies the
   pattern. Its partition constants and Mixtral code must not be copied.
2. **Persistent pinned staging plus submit/sync.** KTransformers demonstrates the
   operational pattern. A Rust implementation must be independently designed
   around this repository's allocator and cancellation model.
3. **Selected-expert packed operations.** llama.cpp and mistral.rs demonstrate
   that a backend need not scan every expert. Exact kernel semantics must still
   be derived from `CpuModel::moe_batch` and validated by the local oracle.
4. **Phase-aware cost models.** Fiddler and HybriMoE both separate prefill and
   decode. The local occupancy profile supplies this project's dimensions and
   prevents importing their thresholds or hardware assumptions.
5. **Backend-specific representations after an authoritative checkpoint view.**
   OpenAI's Torch/Triton split supports keeping storage identity independent
   from a CUDA execution layout. It does not justify full eager duplication.

## Sources deliberately not expanded

**Deferred:** PowerInfer-style prediction, MoE-Infinity caching/prefetch, and
HybriMoE dynamic scheduling were not inspected further because their adaptive
mechanisms do not answer a remaining static-exact design blocker. HybriMoE was
kept only for its evidence that occupancy changes by phase.

**Deferred:** a cluster-oriented GPU expert-parallel runtime was not cloned.
The local CUDA peer API and NCCL source/diagnostics already closed the transport
question: there is no direct peer path, and collectives use host SHM. P2P/RDMA
packing designs would add cluster assumptions without resolving this runtime's
exact selected-expert kernel or commit boundary.

**Deferred:** xInfer was not added after mistral.rs supplied the bounded Rust
ownership/API comparison requested by the backlog. There is no upstream
integration objective.

## Research implication

**Inferred:** prior art supports the thesis's components—static ownership,
activation movement, pinned overlap, selected-expert execution—but no inspected
source provides their exact composition for GPT-OSS top-4 across CPU and two
non-peer GPUs with this repository's transactional state. The remaining work is
integration-specific. This is not a claim that the combination is novel.
