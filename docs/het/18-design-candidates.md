# Architecture-family comparison

This is a research disposition, not an implementation plan. “Survives” means
the evidence does not rule the family out; it does not freeze APIs, placement
counts, schedules, or work packages.

| Family | Exactness and ownership | 120B memory / transfers | Concurrency and failure | Current reuse / new mechanism | Disposition |
|---|---|---|---|---|---|
| 1. GPU layer owner + static expert workers | Router, attention, residual, and rank-order reduction stay on one layer GPU. Each expert has exactly one CPU/GPU0/GPU1 owner. | Fits the owner-selective envelopes. Same-owner routes stay local; CPU and other-GPU routes move compact activations/results through pooled pinned memory. No weight streaming. | Independent expert jobs can overlap; one owner barrier validates all ranks before publish. GPU-to-GPU remote work needs two host relay legs. | Reuses CPU semantic path, stable route records, pinned allocator, device/stream wrappers, evidence, and prepared-state discipline after adaptation. Needs exact selected-expert GPU execution and provisional cache visibility. | **Surviving finalist for the proof target.** Narrowest family that actually exercises all three devices without moving dense state every layer. |
| 2. Host-owned MoE coordination | Router IDs/weights and reduction are made host-visible; CPU and GPUs are resident workers; output returns to the layer device. | Owner-selective weights still fit, but every layer crosses D2H/H2D even when all selected work is local to its layer GPU. Host f32/BF16 ordering is easy to audit. | Host is a simple join point but a synchronization bottleneck; cancellation ownership is explicit. CPU arithmetic and DMA contend measurably. | Reuses more CPU reduction code and pinned staging; requires GPU router synchronization and host coordinator. | **Survives as an oracle/control family, not the performance finalist.** Useful if owner-side exact reduction proves intractable. |
| 3. Alternating layer owners + local reductions | Dense/layer state alternates GPUs while expert owners remain static. Reduction is local to the current layer owner. | Adds a full `[M,H]` host-relayed activation at every layer transition and complicates KV/dense placement. Does not reduce total expert bytes. | May balance dense compute but inserts mandatory inter-layer barriers on the no-P2P topology. Failure state spans alternating cache owners. | Reuses device workers but needs multi-owner dense/KV lifecycle and transfer telemetry. | **Rejected for the first proof.** Added movement and lifecycle complexity do not solve an established blocker or create memory necessary for fit. |
| 4. Existing tensor-parallel/NCCL direction | Dimension-split Q/O/expert projections followed by collectives. | Intended to split weights, but current global/local dimension mismatch invalidates operations. NCCL traverses host SHM; all-reduce moves whole tensors rather than only selected expert payloads. | Worker launch exists; current standalone NCCL pointer API is unsafe and no model transaction exists. | Low-level worker/channel concepts only. Current sharding, NCCL wrapper, and model route cannot be reused. | **Rejected as a model path/control for correctness.** It remains useful only as evidence of what compilation/collective allocation does not prove. |
| 5. CPU MoE + GPU attention control | Exact CPU expert work, layer dense/attention on one GPU, all MoE activations/results cross host. | Fits if CPU owns all experts (`56.879 GiB` owner form plus state), but leaves GPU expert capacity unused. Current prefill host MoE is f32 and not exact, so a control must use the CPU contract rather than current CUDA `forward`. | Simple ownership; GPU and CPU overlap opportunity is limited by layer dependency and host contention. | Reuses CPU oracle and supplies a performance/correctness baseline. Still needs a safe GPU/CPU cache barrier. | **Retain as control, not target.** It provides the lower-complexity reference against which three-way placement must justify itself. |

## Why family 1 is the narrowest surviving finalist

**Verified:** its persistent memory requirement is exactly the same set of
single-owner expert bytes as any viable static family. It avoids always moving
the full layer activation to host and avoids mandatory inter-layer GPU relays.
P2P absence is not fatal because routed activation/result payloads are small and
pinned host relay is measured. The CPU path already defines a route/result
contract whose model-specific surface ends at expert execution.

**Verified:** neither current CUDA MoE nor current tensor parallel can implement
the family as-is. Survival depends on the design space, not an attractive
existing type name.

**Unknown:** exact resident GPU expert latency, destination packing cost, event
overhead, cache publication mechanism, and owner allocation peak. These
unknowns prevent a settled implementation design or static placement ratio.

## Family-1 policy limits

The research supports static single ownership and a designated layer owner. It
does not establish that:

- CPU and both GPUs must execute in every layer/token;
- all layers use the same placement ratio;
- GPU0 is always the layer owner;
- route activations must be duplicated per rank rather than packed per
  destination/source row;
- weighted contributions or unweighted expert outputs are the final wire form;
- one of the two commit barriers is already selected;
- decode and prefill use the same scheduling threshold.

Those are planning decisions gated by the exact selected-expert oracle and
measured construction/kernel costs. Adaptive placement, migration, cache
prediction, prefetch policy, and approximate deferral remain **Deferred** until
static exact execution is proven and profiled.
