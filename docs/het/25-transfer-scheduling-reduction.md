# Transfer, scheduling, and reduction plan

## Measured baseline, not an acceptance threshold

Phase 1 measured these completion distributions on this host:

- conservative top-4 decode payload: 23,040 bytes;
- pinned H2D: approximately 11–12 us;
- pinned D2H: approximately 6–7 us;
- serialized 23,040-byte GPU→host→other-GPU leg: approximately 18.4–18.5 us;
- one 13,236,480-byte expert H2D: approximately 1.66–1.69 ms; and
- pinned allocation: roughly 0.7–0.8 ms for small buffers and 4–4.6 ms for an
  expert-sized buffer.

These values justify pooling and activation movement. They are not proposed
latency budgets, speedup gates, or selected-expert times. Exact GPU
selected-expert, packing, event, and interference costs remain unmeasured until
H2/H6/H10.

## Decode pipeline (`M=1`)

The coordinator performs one all-or-nothing reservation, then this event flow:

```text
GPU0 owner compute:
  attention/KV-private-write -> post-norm -> exact BF16 router projection
  -> BF16 logits + stable top-4 -> routes_ready
       |                                              |
       |                                              +-> GPU0 local expert(s)
       |                                                    -> local_ready
       v
GPU0 relay stream waits routes_ready
  one activation + route descriptor D2H -> host_remote_ready
       |                                  |
       |                                  +-> CPU queue -> cpu_ready
       +-> stable GPU1 pack -> GPU1 H2D -> selected expert(s)
                                      -> GPU1 D2H -> gpu1_host_ready

host observes cpu_ready/gpu1_host_ready (in either order)
  -> GPU0 relay stream H2D each result into its canonical route slots
  -> remote_results_ready

GPU0 owner compute waits remote_results_ready; local_ready is stream-ordered
  -> validate route slots -> rank 0,1,2,3 weighting/reduction -> residual
  -> layer_ready
```

### One download feeds both remote owners

Yes. If any selected route is nonlocal, GPU0 downloads the BF16 source
activation once into an immutable pinned activation lease. The CPU worker reads
that lease directly after `host_remote_ready`. Host packing copies the same row
into the GPU1 outbound pinned lease once for every GPU1 route that requires it.
Concurrent CPU reads and GPU1 packing/H2D reads are permitted; no writer can
touch the source lease until both consumers release it.

The H4 router reads the native BF16 router weight/bias resident in GPU0's
non-expert allocation, matches the CPU projection/rounding trace, supports both
E=32 and E=128, and then performs stable selection and BF16 selected softmax.
The current f32/cuBLAS projection and E≤64 CUDA selector are not fallback paths.

The canonical four route descriptors are downloaded with the activation for
dispatch validation. The authoritative BF16 weight bits and rank slots remain
on GPU0 for reduction; host metadata cannot rewrite them.

### Compact destination packing

Packing starts from canonical row/rank order, looks up the immutable placement,
then stable-groups by `(destination, expert_id)` while retaining:

```text
source_row, route_rank, expert_id, weight_bf16_bits,
canonical_result_slot, source_activation_slot
```

For decode, CPU work may reference the single source row rather than duplicate
it. GPU1 receives a contiguous route-major BF16 input of at most four rows.
Outputs carry the same descriptors. GPU0-local, CPU, and GPU1 writers target
disjoint canonical result slots. No pack/unpack operation infers rank from
expert order.

### Result return

CPU produces unweighted BF16 outputs in a pinned CPU-result lease. GPU1 D2Hs
unweighted BF16 outputs into its own pinned result lease. As each host result
becomes terminal, the coordinator enqueues an H2D on GPU0's relay stream into
the descriptor's fixed route slots and records a ready event. Separate leases
avoid a host merge copy; the relay stream serializes writes while GPU0 compute
can finish local work. GPU0 reduction waits every required upload event.

No transfer carries expert weights. No transfer or reduction uses NCCL or peer
access.

## Streams and dependencies

| Device | Stream | Purpose | Dependencies |
|---|---|---|---|
| GPU0 | owner compute | dense/attention, exact router, local selected experts, rank-order reduction, residual | Records `routes_ready`; waits remote-result events before reduction |
| GPU0 | relay | activation/descriptor D2H and CPU/GPU1 result H2D | Waits `routes_ready`; records source-D2H and each result-upload event |
| GPU1 | expert work | packed H2D, selected-expert kernel sequence, result D2H | Host submits only after source D2H/packing; one stream gives explicit order; records terminal event |
| CPU | bounded worker | exact x8 selected expert buckets | Begins after source D2H; returns join handle and output lease |

GPU0 events are used with stream-wait operations. GPU1 completion crosses
device contexts through a host join before GPU0 H2D; no unsupported cross-device
event assumption is made. Every CUDA event is explicit, timing-capable where
evidence needs it, and owned by the prepared step until drained.

CUDA graphs are disabled for the first heterogeneous proof because current
graph capture assumes one runner stream and stable all-model allocations. Graph
promotion is deferred until event/lease semantics are proven.

## Bounded buffers and backpressure

Decode worst-case raw pinned capacity for one active layer is:

```text
one source activation D2H     5,760 B
four route descriptors           64 B  (versioned/aligned layout maximum)
GPU1 outbound (four routes)  23,040 B
GPU1 inbound results         23,040 B
CPU result (four routes)     23,040 B
raw total                    74,944 B
```

The decode pool uses hard-capped size classes with at most 128 KiB total raw
payload per active dispatch, plus separately accounted allocator headers/events.
GPU0's BF16 contribution arena is 23,040 B; GPU selected-expert scratch is
17,280 B per concurrently executing route and is reused serially on each device
in the first proof.

The proof coordinator admits one active layer dispatch and queue depth one per
CPU/GPU1 worker. It reserves all required host/device leases and queue slots
before dispatch. If any reservation is unavailable, it waits under bounded
backpressure or cancels cleanly without enqueueing. `PinnedPool::acquire`'s
current allocate-on-empty behavior must be wrapped/replaced; pool exhaustion
cannot allocate opportunistically.

Leases carry step generation and outstanding-event counts. A pool accepts a
lease only after CPU joins and all referencing CUDA events are terminal. A
failed stream quarantines its leases until context teardown or an explicit
health check proves quiescence.

## Deterministic owner reduction

The GPU0 contribution arena is indexed `[source_row][route_rank][hidden]`, not
by owner/expert completion order. Before arithmetic, the reducer validates four
unique populated ranks and exact descriptor/placement agreement. It then, for
each hidden element:

```text
acc = 0.0f32
for rank in 0..4:
    contribution = bf16_output[rank].to_f32()
                 * bf16_weight[rank].to_f32()
    acc = acc + contribution
moe_bf16 = bf16(acc)
```

The kernel forbids atomic accumulation, expert-order accumulation, tree
reduction, and reassociation. Residual is applied only at the existing model
boundary after the BF16 MoE result. Weighting/contribution/reduction trace
points support first divergence. A missing result fails; zero-fill or owner
substitution is forbidden.

## Prefill pipeline

Prefill uses bounded chunks of `C≤64` source rows. For a chunk:

- GPU0 produces exactly `4C` canonical route descriptors;
- the host D2H downloads at most one 5,760-byte activation per source row that
  has any nonlocal route, plus descriptors;
- each destination pack maps unique source activation slots to stable
  expert-grouped route records;
- conservative GPU1/CPU route input or result is `C×4×5,760`; and
- GPU0 result arena remains `[C,4,2880]`.

At `C=64`, the deliberately conservative raw pinned sum used for admission is:

```text
source activations         64×5,760       =   368,640 B
route descriptors          64×4×16        =     4,096 B
GPU1 outbound + inbound    2×64×4×5,760   = 2,949,120 B
CPU results                64×4×5,760     = 1,474,560 B
total                                         4,796,416 B
```

The pool cap is 8 MiB for one active chunk. Chunks are sequential in the first
proof; no queue can grow with prompt length. CUDA buckets use the promoted
`M=1` primitive serially until a separate grouped kernel passes its oracle.
CPU buckets may use the existing exact stable batch path. The route/result
contract and GPU0 reduction do not change. A larger prompt creates more
chunks, not larger buffers.

Double buffering, two GPU1 streams, grouped prefill kernels, and cross-layer
pipelining are H10 candidates, not assumptions in H6–H9.

## Cancellation and legal overlap

Legal overlap after `routes_ready`:

- GPU0 local experts with GPU0 D2H;
- CPU expert work with GPU1 pack/H2D/kernel/D2H;
- GPU0 local work with CPU and GPU1 work; and
- CPU/GPU1 result readiness in either order.

Not assumed or allowed in the first proof:

- another step for the same sequence while one visibility epoch is prepared;
- buffer reuse while a DMA/kernel event is outstanding;
- reduction before all canonical slots validate;
- P2P, NCCL, expert weight transfer, or current CUDA all-expert fallback; and
- CPU/GPU concurrency in every layer/token. A route with no work for an owner
  legitimately leaves that worker idle.

Cancellation stops unissued work. Already launched GPU kernels/copies and
started CPU jobs drain. Results may be copied only if required to drain safely;
no reduction/publication is required after cancellation unless already
enqueued, in which case it drains and is discarded.

## Evidence that proves overlap

Each step trace records route/owner counts, buffer high water, queue reserve and
wait, bytes by direction, host pack/copy intervals, CPU start/end, CUDA event
elapsed times, host enqueue/event-observation brackets, dependency reason, idle
wait, reduction, drain, and terminal outcome.

CUDA elapsed clocks from different contexts are not naively overlaid. H6 must
also retain one bounded globally correlated timeline (Nsight Systems or an
equivalent CUPTI/host-callback trace) showing CPU, GPU0, GPU1, H2D, and D2H
intervals. If no correlated trace is available, the result may prove correct
multi-owner execution but **not concurrency**. Wall time shorter than summed
components is supporting evidence only, never the proof.

No latency threshold is set before H2/H6 measure the exact selected-expert and
packing/event terms. H10 derives performance placement only from those results.
