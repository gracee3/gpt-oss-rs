# Heterogeneous execution retrospective

## Scope and preserved boundaries

The heterogeneous campaign tested static expert ownership across CPU, GPU0,
and GPU1. Its successful bounded milestone is H7 at
[`a9ab97a`](https://github.com/gracee3/gpt-oss-rs/tree/a9ab97aef349e7f05b79dd6a1aa6eed1853dd7b4),
preserved by the annotated tag `research/het-h7-20b`. The complete subsequent
history ends at
[`7bb4593`](https://github.com/gracee3/gpt-oss-rs/tree/7bb459361c68b00eed45f56a622c061bb4b135ff),
preserved by `archive/het-2026-08` and the
`archive/heterogeneous-research-2026-08` branch.

The v0.1.0 publication tree does not import that runtime source. It records the
successful 20B boundary, the later incomplete extensions, and the reusable
ownership lessons without turning any of them into a production claim.

## H7 three-owner architecture and exact result

GPU0 held the layer-owner role: it ran the native BF16 router, owned its static
share of experts, and reduced unweighted BF16 expert results in canonical route
rank order. CPU and GPU1 were expert workers, and every expert had exactly one
static owner. GPU0-local inputs used device-to-device movement; nonlocal work
used a fixed pinned-host relay. Peer access, NCCL dispatch, tensor parallelism,
the existing all-expert CUDA MoE path, and decode-time weight streaming were
outside the accepted design.

H7 executed the 63-token retained prompt as bounded serial `M=1` steps and
produced this exact eight-token continuation twice, once after a cold-process
load with an identity-valid CPU x8 cache and once after a same-process warm
reload:

```text
[200005, 35644, 200008, 976, 1825, 5003, 25, 392]
```

Both repetitions committed 70 input steps, retained the real
`[GPU0, CPU, GPU1, GPU0]` selected-owner route at layer zero, demonstrated
three-way compute overlap, and returned to the bounded idle resource state.
The immutable [H7 gate
record](https://github.com/gracee3/gpt-oss-rs/blob/a9ab97aef349e7f05b79dd6a1aa6eed1853dd7b4/docs/het/evidence/implementation-2026-08/h7/README.md)
is the detailed authority.

## Transaction, bounds, retry, and quarantine

Each step prepared private token and K/V state that remained invisible until
the coordinator's exclusive commit callback. The shell state committed first;
one visibility epoch advanced last. Cancellation or failure before publication
therefore discarded private state instead of exposing a partial generation.

Execution used fixed result slots, a fixed five-lease pinned pool, one CPU job
at high water, and bounded relay storage. A real post-enqueue remote fault
proved the recoverable path: all submitted siblings drained, all leases
returned, committed state remained unchanged, and the clean retry succeeded.
The destructive companion fault made the fallback drain unprovable. It
suppressed publication and reuse, poisoned the runtime components, and retained
all five possibly referenced leases and their device/host ownership until
process exit. The uncertainty was quarantined rather than converted into a
fallback success.

## Later loader and capacity-one work

After H7, the archive returned to construction ownership. It added
construction-memory instrumentation, a bounded SafeTensors shard catalog, a
payload-free native-to-runtime consumer plan, device-resident router handoff,
scoped one-shard transactions with terminal-or-quarantine semantics, and
runtime checkpoint-payload retirement. Later commits added capacity-one shard
planning, durable incremental CPU x8 record publication, and a production
capacity-one constructor.

Those changes tightened payload lifetime and retry rules, but they did not
retroactively turn the H7 control into a 120B result. Several intermediate
milestones were source, synthetic-fixture, or construction evidence only. The
complete archive retains those distinctions and their original gate records.

## Exact 120B and R4 stopping boundaries

The H8 120B construction extension never passed. An initial construction-only
attempt stopped when the host-wide no-new-swap gate failed; its one authorized
retry stopped at the GPU-expert construction observation on another global
swap increase. The target process itself reported zero swap in both records,
but the frozen global gate still failed. A later final admission stopped during
its exact-byte-stability preflight, before launching a constructor or loading a
model. No successful 120B construction artifact was produced, and 120B
execution never began.

R4 then compared the retained 20B monolithic and capacity-one constructors
under a fixed supervised matrix. After earlier admission and measurement
boundary stops, the final authorized attempt passed the corrected cold
monolithic source-release and cleanup cell. The next cold capacity-one cell
rejected the single 13,761,300,984-byte native shard before mapping because it
exceeded the frozen 10,544,040,680-byte mapping window by 3,217,260,304 bytes.
No warm, repeat, R4 H7, H8, or 120B work followed. The attempt was consumed and
R4 remained incomplete. The immutable [R4 closeout
record](https://github.com/gracee3/gpt-oss-rs/blob/7bb459361c68b00eed45f56a622c061bb4b135ff/docs/het/36-r4-retained-20b-comparison.md)
preserves the full sequence.

## Claim boundary and reuse

H7 is an exact end-to-end 20B architecture and lifecycle proof. It is not a
controlled performance result: the gate did not define a CPU-only comparator,
randomized benchmark order, repeated timing distribution, or statistical
promotion rule. No HET latency, throughput, memory-efficiency, or scaling claim
should be inferred from it.

The reusable results are narrower: stable single-owner placement, absolute
route identity, bounded relay and pools, private-state publication, terminal
drain before reuse, atomic incremental cache records, and process-lifetime
quarantine when external ownership cannot be proved terminal. Applying those
contracts elsewhere still requires new model, topology, correctness, resource,
and performance gates.
