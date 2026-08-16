# Owner-selective loading and memory plan

## Primary loading route

The primary route is the Phase 1 **hybrid native representation**:

1. local native shards and their revision/index/manifests are authoritative;
2. a validated, read-only `GptOssCheckpointView` applies the complete
   native-to-runtime name map;
3. Q/K/V are contiguous row slices of native QKV tensors and retain the shard
   mapping lifetime; every other runtime name is an alias;
4. GPU-owned experts are uploaded once in native U8 blocks/scales plus BF16
   biases and remain resident on their single device;
5. only CPU-owned experts are persisted as x8 records; and
6. no owner retains an unused expert representation.

Direct-native CPU decode without persistent x8 and a streamed full runtime
snapshot remain documented recovery routes, not the implementation critical
path. Native plus a full CPU x8 repack is rejected.

## File/view contract

`model_loader/gpt_oss_native.rs` will own shard mappings, parsed tensor entries,
the 543-to-687 map, and Q/K/V slice descriptors. A view contains shard identity,
absolute byte range, dtype, logical shape, and mapping owner; it never owns a
copied tensor. `model_loader/owner_selective.rs` consumes views only after the
placement manifest validates.

The map is accepted only when:

- source revision, config, index, shard names/sizes, and small-asset identities
  match the manifest;
- all 543 native tensors and all 687 runtime names are accounted for;
- Q/K/V slices exactly partition native `[5120,2880]` weight rows and `[5120]`
  bias rows into Q `[4096,...]`, K `[512,...]`, and V `[512,...]` without gap,
  overlap, or copy;
- every expert tensor has the proven shape/dtype and expert-major contiguous
  byte ranges; and
- tokenizer/protocol assets are reused only when their full byte identities
  match the pinned mapping evidence.

The mappings outlive all tensor and expert handles. A model cannot unmap shards
until construction is terminal and every worker/step/weight handle is drained.

## CPU x8 persistence

`cpu_repack.rs::CpuRepackCache::open_or_create` currently creates one record for
an entire expert tensor. H3 adds an owner-filtered, layer-scoped format:

```text
header: source revision and slice identities, layout/version, layer,
        sorted CPU-owned expert IDs, per-expert offsets and byte counts
payload: only those experts, each gate/up then down x8 records
publication: temporary file -> fsync -> atomic rename -> directory fsync
```

Each layer file is independently complete or absent. An interrupted temporary
is never loadable. Restart validates the exact CPU expert set and source
identity before mapping it. A changed placement gets a different cache key; old
records are inert and are not mixed. Repack is streamed one expert at a time
with bounded output batches; it never creates a full-layer or full-model x8
buffer. BF16 biases may be retained as BF16 or widened once to f32 in the
CPU-owner record; the conservative envelope counts the f32 form.

## Byte-exact representation ledger

Binary GiB are bytes divided by `2^30`.

| Term | 20B | 120B | Memory meaning |
|---|---:|---:|---|
| Native payload | 13,761,264,768 B (12.8162 GiB) | 65,248,815,744 B (60.7677 GiB) | Read-only file-backed address-space mapping; resident pages contribute RSS/page cache and are reclaimable, but not free |
| Expert checkpoint payload | 10,165,616,640 B (9.4675 GiB) | 60,993,699,840 B (56.8048 GiB) | Authoritative blocks/scales/BF16 biases |
| Non-expert payload | 3,595,648,128 B (3.3487 GiB) | 4,255,115,904 B (3.9629 GiB) | GPU0 persistent dense/router/embedding/output payload assumption |
| Conservative owner expert | 13,253,760 B each | 13,253,760 B each | 13,219,200 B x8/native-equivalent payload plus 34,560 B f32-bias allowance |
| All conservative owner experts | 10,178,887,680 B (9.4798 GiB) | 61,073,326,080 B (56.8790 GiB) | Sum across physical owners, never an additional host-wide copy |
| Full x8 payload without bias | 10,152,345,600 B | 60,914,073,600 B | Rejected as a second all-model representation for 120B |
| Native + full conservative owner form | 23,940,152,448 B | 126,322,141,824 B (117.647 GiB) | **Rejected physical-residency interpretation** for 120B before runtime state |

One checkpoint expert remains exactly:

```text
gate/up blocks  5,760×90×16 = 8,294,400 B
gate/up scales  5,760×90    =   518,400 B
gate/up bias    5,760×2     =    11,520 B
down blocks     2,880×90×16 = 4,147,200 B
down scales     2,880×90    =   259,200 B
down bias       2,880×2     =     5,760 B
checkpoint total             = 13,236,480 B
x8 blocks+scales             = 13,219,200 B
x8 + f32 bias allowance      = 13,253,760 B
```

GPU execution reads the 13,236,480-byte native form. The 13,253,760-byte value
is retained for quota safety; an implementation may not use the difference to
admit more experts until actual allocation/alignment evidence exists.

## Concrete proof envelopes

### 20B proof placement

The known layer-0 retained route permits an intentionally sparse three-owner
proof: 766 experts on GPU0 and one each on CPU and GPU1. Conservative expert
bytes are 10,152,380,160 B, 13,253,760 B, and 13,253,760 B respectively. GPU0
expert + dense + 4-GiB reserve is:

```text
10,152,380,160 + 3,595,648,128 + 4,294,967,296
= 18,042,995,584 B = 16.8038 GiB
```

This is proof placement, not a useful final balance. H3 measures the 20B cold
construction and then unloads/reloads it once before H8.

### 120B existence envelope

Each 3090 reports 25,769,803,776 bytes. With all non-expert payload on GPU0 and
a 4,294,967,296-byte inclusive execution reserve on each device:

```text
GPU0 expert budget = 25,769,803,776 - 4,255,115,904 - 4,294,967,296
                   = 17,219,720,576 B
floor(budget / 13,253,760) = 1,299 experts
owner bytes                    = 17,216,634,240 B

GPU1 expert budget = 25,769,803,776 - 4,294,967,296
                   = 21,474,836,480 B
floor(budget / 13,253,760) = 1,620 experts
owner bytes                    = 21,471,091,200 B

CPU experts = 4,608 - 1,299 - 1,620 = 1,689
CPU conservative owner bytes            = 22,385,600,640 B
CPU persisted x8 payload                 = 22,327,228,800 B
```

These `1,299 / 1,620 / 1,689` counts assume, explicitly:

- all 4,255,115,904 non-expert bytes are resident on GPU0, with no GPU1 copy;
- each expert is charged the conservative 13,253,760 bytes including f32-bias
  allowance, even though CUDA intends BF16 bias;
- no owner retains source `Vec`, FP16 matrix, second CUDA copy, or second x8
  form;
- each GPU's 4 GiB **inclusive** reserve covers CUDA context/modules, cuBLAS,
  allocator retention/fragmentation, configured K/V, attention scratch,
  selected-expert scratch, staging, outputs, and a safety remainder;
- the proof context cap is configured before admission; 120B BF16 K/V is
  73,728 B/token (0.28125 GiB at 4,096, 2.25 GiB at 32,768); and
- mapped native/x8 pages are demand-resident and reclaimable. Their virtual
  size is not added as anonymous RAM, but their actual RSS/PSS and
  `MemAvailable` effect are measured and can reject the envelope.

The reserve is not “4 GiB still free after context initialization.” H8 measures
context/module/allocator use inside it and recomputes expert quotas before
materialization. If configured K/V plus measured runtime/safety exceeds 4 GiB,
the solver reduces GPU owner counts and moves the difference to CPU in a new
manifest. It never erodes the reserve silently. Maximum 131K context needs 9
GiB K/V and is outside this example.

## Construction stages and measurement points

| Stage | Action and permitted temporary memory | Measurement and abort point | Rollback |
|---|---|---|---|
| L0 identity | Read small configs/indexes/manifests; build complete map; resolve stable devices; validate placement/quota arithmetic | Source hashes/revision, map cardinality, `MemAvailable`, swap, per-GPU total/free | Drop metadata only |
| L1 runtime baseline | Create GPU0/GPU1 contexts, streams, events, modules, cuBLAS; allocate no model tensors | Record RSS/PSS, file/anonymous split, pinned bytes, allocator state, per-GPU used/free; charge observed runtime baseline to reserves | Drain streams, destroy contexts in reverse order |
| L2 mappings | Map native shards read-only and create alias/slice table; do not scan payload | Record virtual map bytes separately from RSS/PSS/page cache; validate no writable model map | Unmap all views |
| L3 GPU0 non-experts | Stream each view into final GPU0 allocation in bounded chunks; Q/K/V slices address native rows | Maximum 16 MiB pinned upload stage; after every tensor record allocation and predicted reserve; largest dense payload is not materialized as a host `Vec` | Drop current tensor, then all registered dense handles |
| L4 GPU experts | Iterate manifest order `(layer,expert)`; upload only that owner's blocks/scales/bias into final device allocation; register handle only after all six surfaces validate | One expert working set plus ≤16 MiB reusable upload stage; record source pages, actual aligned VRAM, owner counts | Drop incomplete expert; reverse-drop registered experts on either GPU |
| L5 CPU experts | Open a valid owner-filtered layer x8 file or stream-create it one expert at a time, atomically publish, map read-only, then register handles | Bound anonymous repack buffer to 2 MiB and total construction staging to 256 MiB; sample RSS/PSS/page cache per layer | Delete only incomplete task-created temp; drop maps/handles; published identity-valid cache remains reusable |
| L6 execution reserves | Allocate/configure K/V for the proof context, selected-expert scratch, route/result arenas, and capped pinned pools | Verify ledger plus actual `cudaMemGetInfo`, host high-water, zero swap, and required safety remainder | Drain, release pools/KV, then weights/contexts |
| L7 publish model | Validate materialized key set equals manifest once, freeze owner pools, produce construction evidence, then atomically register model | Final RSS/PSS, mapped/file/anonymous/pinned split, per-GPU categories, high waters and reserve remainder | No model is visible until this succeeds; reverse-drop all prior stages |
| L8 repeatability | Unload with mandatory drain; verify allocations return to baseline tolerance; repeat one cold/warm construction cycle | No unowned handles, active events, retained pinned leases, process swap, or unexplained GPU allocation | Quarantine/leak is a failed H3/H8 gate |

Dense BF16→execution conversion, where required by existing CUDA dense paths,
must be streamed directly into final device storage. Current loaders that create
a whole host `Vec<f32>`/`Vec<f16>` or whole U8 layer `Vec` are bypassed for this
route. The selected-expert GPU path performs no persistent FP16 expansion.

## Admission and hard stops

Budgets are proposed acceptance thresholds, not Phase 1 measurements:

- At every stage, projected next allocation must fit the category ledger before
  it is attempted.
- For the first 120B proof, host `MemAvailable` must remain at least 12 GiB,
  process `VmSwap` must remain zero, and system swap used may not grow above the
  captured idle baseline. Construction stops before a violated threshold.
- A configurable process-RSS guard defaults to 72 GiB for the first H8 run.
  File-backed RSS is still counted by this guard; PSS/page-cache labels explain
  it but do not waive it. Changing the guard requires review and new evidence.
- Anonymous construction staging, excluding final resident owner state, is
  capped at 256 MiB; pinned upload/construction staging is capped at 16 MiB.
- Each GPU uses a category ledger. Actual allocation plus all remaining
  configured reserve categories and safety must be ≤ reported total. A
  material discrepancy between the ledger and `cudaMemGetInfo`, an allocator
  OOM, or a reserve deficit aborts before execution.
- The initial prefill chunk is 64. Raw pinned transfer/result payload is
  4,796,416 B worst-case; the one-active-chunk pool cap is 8 MiB. Decode raw
  payload is 74,944 B; its size-class leases are capped at 128 KiB. Pools cannot
  allocate beyond those caps.

No code may call global cache-drop controls or confuse page cache with free
memory. Measurement records payload bytes, address-space mappings, file-backed
RSS/PSS, anonymous resident allocations, pinned memory, device allocations,
temporary conversion, allocator-retained bytes, K/V, scratch, output, and
safety reserve as separate fields.

## Construction gates

H3 passes on 20B only after exact ownership, byte accounting, bounded
construction, partial-failure cleanup, and a repeat load/unload. H8 repeats the
same measured proof for 120B without executing it. Any full alternate 120B
representation, any swap, an owner count mismatch, reserve erosion, or retained
unowned allocation stops the package and forbids H9.
