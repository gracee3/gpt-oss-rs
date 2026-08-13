# Tiger Lake forced Xe expert residency

Status: implemented experiment; forced-only; live promotion evidence pending.

## Boundary

`--xe-expert-cache-mib N` enables a bounded in-memory OpenCL cache only when
the requested device is explicitly `xe`. Its default is zero. Zero uses the
existing production streaming path, and automatic device selection cannot
enable the cache. The existing M>=4, prefill-only projection policy remains
unchanged; routing, SwiGLU, activation/output streaming, decode, attention,
KV, sampling, and state remain CPU-owned.

Each entry owns immutable OpenCL weight and bias buffers in the validated v2
layout. Its key combines model and tensor source identities, layer, expert,
projection role, logical dimensions, layout version, kernel source and ABI,
build options, PCI identity, and driver/runtime library identities. A changed
field is a miss, never a reinterpretation.

## Ownership and failure

The OpenCL runtime owns the deterministic LRU and its device objects under the
same serialized queue mutex as projection. Capacity is reserved and victims
are drained, evicted, and released before a miss allocation, so even transient
cache residency cannot exceed the configured bound. A projection larger than
the cache bypasses residency and follows the existing streaming path.

On a hit, neither the CPU x8-v2 repack traversal nor the weight/bias upload is
performed. On a miss, repack is invoked lazily exactly once. The runtime
records hits, misses, bypasses, evictions, current/high-water resident bytes,
repacks and upload bytes avoided, all streamed or cache-insert upload bytes,
and faults. Profiling records
the per-projection residency state and current resident bytes.

Any allocation, upload, submit, wait, readback, or release failure drains the
queue, discards the uncommitted output, opens the existing process-wide Xe
circuit breaker, and permits exactly one CPU recomputation in the model
runner. Shutdown drains first and releases every cache object once; repeated
shutdown remains safe.

## Validation and evidence

Unit tests cover disabled and undersized capacities, strict accounting,
deterministic eviction, identity drift, hit/miss statistics, and explicit-only
configuration. The opt-in live suite additionally checks cold miss, warm hit,
exact BF16-boundary output, repack avoidance, capacity eviction, resident-byte
high-water, and repeated shutdown on the real OpenCL stack.

Live 0/128/256/512 MiB corpus measurements and the final disposition are
recorded in the candidate-specific Tiger Lake closure. Regardless of their
performance result, this sprint does not authorize automatic Xe promotion.
The representative-corpus driver accepts `--xe`, `--xe-max-resident-mib`, and
`--xe-expert-cache-mib`, so resident and zero-cache controls use the same
scenario order, profiling schema, repetition policy, and artifact indexing as
the CPU corpus.

`xe_projection_gate` also compares `xe_streaming` and `xe_resident` against
scalar, CPU Auto, and AVX2 on real layer-0 gate/up and down checkpoint tensors.
Its `--rows` control accepts every production-legal M>=4 bucket, including
non-multiples of four that exercise the OpenCL dispatch-padding contract.
Its resident repack closure is lazy, so warm samples include activation
preparation, submission, wait, and readback but genuinely exclude weight
repacking and weight/bias upload. The output records the exact runtime
descriptor, cache statistics, source/binary identity, per-method samples, and
BF16-boundary correctness result.
