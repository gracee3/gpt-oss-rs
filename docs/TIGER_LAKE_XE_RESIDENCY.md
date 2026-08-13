# Tiger Lake forced Xe expert residency

Status: implemented experiment; forced-only; live evidence complete; negative
full-model residency result.

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
`--xe-expert-cache-mib`. `--xe-warmup-prefills N` primes that cache on the
same attached runner before the measured request. The capture records the
pre-request counter snapshot and profiler sequence boundary; the deterministic
summarizer subtracts cumulative counters and excludes warmup profile rows.
Resident and zero-cache controls therefore use the same
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

## Live result

All 22 opt-in live OpenCL tests pass on Intel NEO `26.05.037020`. This includes
startup qualification, real projection shapes and tails, cold/warm residency,
identity drift, deterministic eviction, corruption recovery, fault handling,
and idempotent shutdown. Startup self-test traffic is intentionally reset
before workload telemetry so upload counters describe only caller work.

The isolated real layer-0 projection gate at 128 MiB produced 482 hits from
484 resident requests, avoided 3,194,156,160 upload bytes, held 13,253,760
bytes, and had no evictions or faults. At M=4, resident gate/up fell from a
10.868 ms streaming median to 1.797 ms; down fell from 7.336 ms to 1.607 ms.
Its artifact root is
`/home/emmy/gpt-oss-rs-artifacts/tiger-lake-optimization/5da8afa2c1ac3899ab1bfbbc220eb5ad4bed4e8a/xe-residency/`,
whose `SHA256SUMS` hashes to
`e360ab26f721549ea92f92d3a58088c7a506cab3812b4e200afb3ac266492911`.

The representative full-model harmony_63 control gives the opposite result.
With zero cache, exact official tokens completed in 20.114 s (17.028 s
prefill). After one cache-priming prefill, 128, 256, and 512 MiB each recorded
zero hits in the measured prefill: the second layer/expert traversal evicted
every reusable entry before it was reached. Each pass had 846 misses and
uploaded 5,606,340,480 bytes. Measured full-request times were 21.086,
21.228, and 21.138 s respectively, a roughly 4.8-5.5% regression from the
zero-cache control. All four runs produced exact official tokens and no
faults. The indexed root is
`/home/emmy/gpt-oss-rs-artifacts/tiger-lake-optimization/8c40301051d1518e878bbb8f657c7ce8274dcfbb/xe-full-model/`;
its `SHA256SUMS` hashes to
`2476b97c21bc8bc6701d4749ad90154397eba94c160e6d6242ffb8ac4ef9a9c5`.

Consequently no capacity is selected, the default remains zero, the feature
remains explicit-Xe-only, and automatic Xe remains disabled. A future cache
would need a different traversal or a capacity large enough for the working
set; neither is in this sprint.
