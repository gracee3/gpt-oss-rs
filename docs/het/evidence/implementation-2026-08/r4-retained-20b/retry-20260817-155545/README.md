# R4 retained-20B retry evidence

Verdict: `failed_first_cell_clean_file_allowance`

The topology guard correction is commit
`248c724a95277a4734a89eddd0cec1c74971fde5` on
`agent/r4-retained-20b-comparison`. It was pushed before the three release
binaries were built. No PR or merge was created.

The shared resolver enumerates exactly the two expected Samsung 990 PRO
namespaces from sysfs model data, traces the root device through sysfs slave
ancestry, and treats the other namespace as protected. Mount checks use the
major:minor ancestry from mountinfo. No serial, UUID, or block payload is read.
The retry resolved `nvme0n1` as the protected namespace and kept that nonempty
identity stable through every retained observation.

## Validation and admission

Formatting, targeted watchdog/supervisor tests, touched-binary Clippy, the
locked workspace check/test suite, CUDA kernel compilation, two-GPU PCI
identity, and the non-model CUDA router tests passed. The only CUDA build
warnings were the unchanged baseline warnings in upstream model-runner and
engine code.

The one authorized preflight ran from 15:56:07 through 15:58:07 EDT. Its six
samples span 120,020 ms and pass every frozen R2 predicate: global, target-tree,
and comparison-cgroup swap stayed zero; PSI `some/full avg10` stayed zero;
process attribution was complete; minimum `MemAvailable` was 96,518,619,136
bytes; both GPUs stayed at 1 MiB used; and `nvme0n1` stayed read-only and
unmounted. [`preflight-summary.json`](preflight-summary.json) binds the raw
15,513-byte record by SHA-256.

## Terminal matrix result

Fresh admission passed and the supervisor started only the first fixed cell,
`cold-monolithic-control`. At 169 observations, the comparison cgroup's
clean-file delta reached 11,510,542,336 bytes. The frozen allowance is
11,488,417,896 bytes, so the 22,124,440-byte excess was a hard failure. The
supervisor sent `SIGINT`, completed the fixed settle observation, wrote the
immutable run record, and stopped.

The child did not write terminal construction evidence. Six bounded memory
events show progress only through the layer-owner dense stage. No comparison
cache directory became visible. The capacity-one cold cell, both warm cells,
both repeat cells, both H7 parity cells, and cache identity comparison did not
start. [`run-summary.json`](run-summary.json) binds the raw 34,492-byte run
record and the six-event external journal by hash.

## Cleanup and boundary

Global and cgroup swap remained zero, PSI avg10 returned/stayed zero, both GPUs
were at 1 MiB used, no comparison/constructor/control process remained, the
protected namespace was still read-only and unmounted, and the documented
network route roles remained present. The pre-existing owner-cache inventory
remained exactly
`06cbcaf72a2986ad918c8b64ba818e718d292ac0b7f3b89ee9aa9ae5a43255d6`.

The cgroup retained the clean file-cache charge after child exit; its maximum
post-exit `memory.current` drift was 11,515,166,720 bytes, so that frozen cleanup
gate also did not pass. No cache clearing, cgroup/swap change, retry, threshold
change, or fallback was attempted. No H8 or 120B process, input, or access
began. A further R4 attempt requires new authorization.

[`validation.tsv`](validation.tsv) lists the individual gates. `SHA256SUMS`
authenticates the bounded files in this directory except itself. Raw evidence
and the task-unique incomplete run root remain under `/home/emmy/workspace`.
