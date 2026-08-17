# R4 release-boundary retry evidence

Verdict: `failed_second_cell_capacity_one_mapping_window`

The advised-release correction is commit
`2e8820b14cae2497d1be787d06b772b3536243a7` on
`agent/r4-retained-20b-comparison`. It was pushed before the three release
binaries were built. No PR or merge was created.

## Validation and admission

Formatting, release-handshake and supervisor tests, protected-NVMe topology
fixtures, watchdog tests, touched-binary Clippy, the locked workspace
check/test/doc-test suite, CUDA 13.3 `sm_86` kernel compilation, both RTX 3090
PCI identities, and the six non-model CUDA router tests passed. Historical
model-runner and engine CUDA warnings were unchanged.

The one authorized preflight ran from 16:57:33 through 16:59:33 EDT. Its six
samples span 120,019 ms and pass every frozen R2 predicate: global, target-tree,
and comparison-cgroup swap stayed zero; PSI `some/full avg10` stayed zero;
process attribution was complete; minimum `MemAvailable` was 96,152,887,296
bytes; both GPUs stayed at 1 MiB used; and `nvme0n1` stayed read-only and
unmounted. [`preflight-summary.json`](preflight-summary.json) binds the raw
15,527-byte record by SHA-256.

## Terminal matrix result

`cold-monolithic-control` passed. Its checkpoint mapping received successful
`MADV_DONTNEED` and `POSIX_FADV_DONTNEED`, was unmapped and closed, and reported
zero source mappings, PSS, descriptors, process swap, and cgroup swap. The
nonce-bound release proof passed five samples over 34,503 ms. The maximum clean
file delta was 2,299,572,224 bytes against the unchanged 11,488,417,896-byte
allowance; maximum dirty/writeback delta was 2,166,784 bytes; both post-exit
drifts were zero. The v7 terminal construction record and its eleven-event
journal are hash-bound in [`run-summary.json`](run-summary.json).

The next fixed cell, `cold-capacity-one`, failed before mapping or release
publication with `capacity-one source shard exceeds the frozen mapping window`.
The local native 20B package contains one 13,761,300,984-byte SafeTensors shard;
the unchanged `RETAINED_MAX_SOURCE_MAPPING_BYTES` is 10,544,040,680 bytes. The
3,217,260,304-byte excess is a hard constructor rejection. The child exited
with code 1; no terminal output, source-release marker, or capacity-one cache
was published. The supervisor completed its post-exit settle and stopped.

Warm, repeat, both H7 parity cells, and cache identity comparison did not start.
No retry, threshold change, alternate constructor, shard rewrite, or cache
clearing was attempted. No H8 or 120B process, input, or access began.

## Cleanup and boundary

No comparison process remained, both GPUs returned to 1 MiB, global and cgroup
swap remained zero, PSI avg10 was zero, and the protected namespace remained
read-only and unmounted. The documented Wi-Fi default route and local-only
direct-link route remained present. The pre-existing owner cache had the same
before/after payload inventory SHA-256,
`c69bcdffe20ddff32ced3f6acc422a6c696d24b5e5f8888597b071d6cc57a09a`.

The task-unique monolithic cache and bounded raw evidence remain under
`/home/emmy/workspace/gpt-oss-rs-het-r4-20260817-egYmqJ`. Git retains only the
summaries and hashes here. [`validation.tsv`](validation.tsv) lists each gate;
`SHA256SUMS` authenticates this directory except itself.
