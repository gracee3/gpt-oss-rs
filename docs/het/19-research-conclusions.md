# Research conclusions

**Recommendation:** `conditionally_ready` for the planning phase after review.
The physical/format/transport thesis survives, and the narrow executor seam is
clear enough to plan. Exact selected-expert CUDA semantics, owner-selective load
peak, and provisional KV publication remain explicit proof gates; the research
does not pretend they already exist.

## Required answers

### 1. Can 120B fit safely?

**Verified, conditionally:** 4,608 experts require 56.8790 GiB in the
conservative single-owner representation; non-expert payload is 3.9629 GiB.
With 4 GiB reserved on each GPU and dense payload on GPU0, a concrete envelope
owns 1,299 experts on GPU0, 1,620 on GPU1, and 1,689 (20.8482 GiB) on CPU.
At 32K context, exact KV bytes are 2.25 GiB. This fits the aggregate physical
memories with useful host headroom.

**Rejected:** simultaneously resident native 120B plus full CPU repack is
117.647 GiB before execution state, above host RAM. Safe fit requires bounded
shard access, reclaimable page cache, single-owner construction, no unused
alternate expert form, and context-aware reserves. Future peak is not yet
measured.

### 2. Which loading family?

**Research preference:** a hybrid native loader: keep the official native
package authoritative, expose Q/K/V as contiguous views and all other tensors
as aliases, persist x8 records only for CPU-owned experts, and upload native
packed bytes directly for GPU-owned experts. Direct-native without persistent
repack and bounded offline transformed snapshots are both technically viable.

The offline snapshot is not required for tensor correctness: all 20B payload
bytes prove it is renaming plus Q/K/V slicing, and the full 120B metadata map is
complete. It remains a fallback for a simpler current-loader boundary, at about
65.28 GB extra disk including the official root assets.

### 3. What expert contract is exact?

Post-attention BF16 `[M,2880]` activation; f32 router accumulation rounded to
BF16 logits; stable descending top-4 with lower expert ID on ties; selected-only
f32 softmax with each weight BF16-rounded; stable `(row,rank,expert,weight)`
grouping; MXFP4 gate/up and down with the CPU projection round points; GPT
SwiGLU alpha/clamps/`up+1`; BF16 unweighted down output; f32 weighting and
rank-0→3 reduction; BF16 MoE result; residual afterward; no visible state until
commit.

### 4. What is the narrowest seam?

A model-owned router emits a BF16 activation plus rank-bearing route records;
backend jobs execute independently owned resident experts and return BF16
per-rank unweighted outputs; the layer owner validates, weights, reduces in rank
order, applies residual, and participates in the prepared-step barrier. The seam
does not import attention, KV layout, Harmony, tokenizer, or general model
policy. GPT-OSS MXFP4/SwiGLU remains explicitly model/backend-specific.

### 5. Which current components are trustworthy inputs?

- CPU scalar semantics, real-weight trace, retained-continuation path, and
  `PreparedCpuStep` discipline;
- stable top-k/group/reduce semantics;
- exact CPU x8/AVX2/VNNI operations within their current policy gates;
- pinned allocation, basic CUDA devices/streams/modules/errors, and separately
  tested cuBLAS primitives;
- worker/channel structure only after lifecycle audit;
- memory reservation/accounting and trace/manifest machinery after adding
  owner/transfer/commit identities.

### 6. Which current paths should be bypassed or replaced?

Current CUDA prefill host f32 MoE; CUDA decode's E≤64 top-k, all-expert scan,
full per-call FP16 expansions, ordinary SiLU, and expert-order accumulation;
tensor-parallel sharding/global-dimension route; and the standalone NCCL wrapper
that passes host slices as device pointers. Compilation and CUDA unit tests do
not change those dispositions.

### 7. What communication and overlap exist?

CUDA peer access is zero and peer enable fails in both directions with
`cudaErrorPeerAccessUnsupported`. NCCL v2.31.2-1 uses SHM/direct host transport;
NVLink is not operational. Pinned 23,040-byte top-4 decode transfer medians are
about 11–12 µs H2D and 6–7 µs D2H; a serialized GPU→host→GPU relay is about
18.4–18.5 µs one way. Dual-GPU independent pinned transfers overlap, and a
cuBLAS control overlaps a transfer on separate streams. Pinned allocation costs
0.7 ms small and about 4–4.6 ms at 13.2 MB, so pooling is mandatory.

### 8. When is each execution location favored?

**Same GPU:** favored whenever an exact resident expert is selected because it
avoids relay; exact kernel latency is still unknown.

**Other GPU:** plausible even at decode because two compact relay legs are tens
of microseconds, far below the current approved CPU M=1 projection core at
about 4.57 ms. The exact GPU kernel/event/packing term must be measured before a
threshold is chosen.

**CPU:** required as a capacity owner and valuable for parallel selected work;
current exact decode cost is known. Prefill occupancy is highly skewed and the
current auto multirow path is scalar, so CPU-owned high-occupancy experts can
dominate. Forced AVX results are exact controls but the closed policy lane
forbids promoting them from this research.

**Weight streaming:** rejected as the default. One 13,236,480-byte H2D costs
1.66–1.69 ms before expansion and arithmetic, versus kilobyte-scale activation
movement.

### 9. What gates implementation correctness?

The layer-0 real-weight 20B oracle records activation, router logits, exact IDs
and rank, BF16 weights, per-rank gate/up/SwiGLU/down, unweighted output, weighted
contribution, rank-ordered reduction, residual, placement, bytes/events, and
commit outcome. It covers decode `M=1` and observed prefill occupancies
`M=7,24,33,61`. IDs/order and BF16 boundaries remain bit-exact; f32 tolerance
may only use an existing authoritative gate.

The host control already passed the pinned eight-token continuation and emitted
a four-rank layer-zero trace with no swap. This must remain the control before
any 120B execution or heterogeneous promotion.

### 10. What failure/commit model is viable?

All expert contributions and KV/output deltas remain provisional until every
CPU/GPU/copy completion is drained, exact rank completeness is checked,
reduction succeeds, and request revision/cancellation is revalidated. Two
models survive: fully staged KV/output, or private append slots hidden behind a
visibility epoch. The latter is likely slimmer; the former is easier to prove.
Neither is selected here.

Fallback is safe only before dispatch or after complete drain/discard with no
visible revision change. Partial reduction, zero-filled contribution, owner
substitution, and buffer reuse during DMA are forbidden.

### 11. Which architecture families survive?

- **Finalist:** GPU layer owner plus static single-owner CPU/GPU0/GPU1 expert
  workers and pinned relay.
- **Diagnostic survivor:** host-owned MoE coordination.
- **Controls:** exact CPU MoE/GPU-attention-style boundary and the current full
  CPU retained path.
- **Rejected for first proof:** alternating layer owners and existing
  tensor-parallel/NCCL model execution.

### 12. Is planning ready?

`conditionally_ready`. Format viability, aggregate memory, exact CPU contract,
20B control, no-P2P transport, model-sized copy distributions, current component
disposition, failure states, and a narrow finalist are all established.

The conditions that planning must preserve as promotion gates are:

1. measured cold construction and steady residency for owner-selective native/
   hybrid loading remain within a selected context-aware envelope;
2. a selected-expert CUDA primitive passes the one-layer bit-exact/first-
   divergence oracle without scanning or expanding every expert per call;
3. the chosen provisional KV/output visibility barrier proves cancellation,
   drain, cleanup, and revision behavior;
4. exact GPU execution, packing, event, and interference measurements replace
   the current cuBLAS control before placement thresholds are frozen.

These are design acceptance conditions, not an implementation task breakdown.

## Commands and deliberate omissions

**Run:** read-only Git/source/model metadata inspection; streamed 20B native↔
runtime byte comparison; metadata-only 120B mapping against the pinned official
index; exact small-asset hashes; clean locked release builds of the internal
`cpu_parity` and existing matrix benchmark; three bounded internal 20B CPU
controls; standalone real-weight CPU expert probe; CUDA peer, transfer, relay,
overlap, and cuBLAS controls; source-built pinned NCCL transport/all-reduce
probe; sanitized host/GPU/storage checks; and primary-source/PDF inspection.

**Not run:** full 120B load/generation, any 120B transformation or payload
download, official Python/PyTorch oracle image, HTTP semantic checks, Docker
builds, broad workspace/CUDA validation already covered by the sanity report,
throughput benchmarking, privileged tuning, model copying/hashing, Tiger Lake,
Qwen, upstream work, or production implementation.

Nothing was committed or pushed. External checkouts and harnesses remain under
`~/src`; no reference source was modified except ignored build products. The
final Git and protected-device confirmation are recorded in the evidence index.
