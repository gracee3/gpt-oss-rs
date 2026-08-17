# R4 retained-20B constructor comparison

R4 adds a commit-bound supervisor for the separately authorized comparison of
the monolithic-control and capacity-one constructors. It implements the frozen
R2 admission policy, a fixed cold/warm/repeat construction matrix, an exact
two-run H7 continuation check for both constructors, immutable cache identity
comparison, and process-group cleanup. The runner contains no 120B input and
records `h8_or_120b_started: false`.

## Implemented boundary

The release supervisor binds the repository commit, its own executable, both
child executable hashes, the R2 policy identity, the retained 20B mapping and
placement, the original CPU trace identity, the task-unique cache roots, and
the protected-device state. The corrected guard resolves the protected Samsung
namespace from root-device sysfs ancestry rather than a kernel enumeration
number, records that kernel name, and requires one stable nonempty identity.
Cold and warm 20B construction no longer require or open a 120B checkpoint.
The capacity-one H7 control constructs from the bounded catalog and native
metadata map before entering the unchanged retained runtime path.

Runtime supervision enforces target and cgroup swap zero, global swap
nonincrease, the 12-GiB `MemAvailable` floor, PSI `some/full avg10` zero,
complete process attribution, the clean-file and dirty/writeback allowances,
the 30-second plus five-sample settle protocol, 64-MiB post-exit memory and file
drift, GPU cleanup, and the protected NVMe guard. Retained samples include
cgroup `anon`, `file`, `file_mapped`, `file_dirty`, `file_writeback`, and swap.

## Initial authorized attempt

The implementation was committed as `eb1d816` on
`agent/r4-retained-20b-comparison` before the release binaries were built. A
fresh 120-second preflight began on 2026-08-17 at 01:16:55 EDT and retained six
samples over 120,024 ms. It failed before child launch for two frozen gates:

- the comparison cgroup held 1,146,880 bytes of pre-existing swap instead of
  exactly zero; and
- memory PSI `some/full avg10` reached 0.07 at the final retained sample.

Global swap did not grow, target-tree swap stayed zero, process attribution was
complete, `MemAvailable` stayed at or above 96,249,933,824 bytes, both GPUs
retained their PCI identities and free-memory reserves, and the protected NVMe
remained read-only and unmounted.

The supervisor therefore stopped with status `blocked_pre_model`. No
construction cell, 20B model load, H7 repeat, H8, or 120B access began. The
existing owner cache was byte-for-byte unchanged. The exact bounded record is
indexed in [the R4 evidence directory](evidence/implementation-2026-08/r4-retained-20b/README.md).

## Authorized retry

Commit `248c724` corrected the hard-coded protected-device name and was pushed
before rebuilding the release binaries. Synthetic tests cover both namespace
orders, root-through-device-mapper ancestry, writable/mounted/ambiguous
topologies, identity changes, and fixtures without serial data. The full
formatting, targeted, Clippy, locked workspace, and CUDA validation ladder
passed.

The retry's fresh six-sample preflight spanned 120,020 ms and passed every R2
gate with zero global/cgroup/target swap, zero PSI avg10, stable GPU identity,
and stable protected namespace `nvme0n1`. Fresh admission then started the
first cell only. `cold-monolithic-control` was interrupted when its clean-file
delta reached 11,510,542,336 bytes, exceeding the frozen 11,488,417,896-byte
allowance by 22,124,440 bytes. No terminal construction output or comparison
cache was published.

The supervisor stopped without starting any capacity-one, warm, repeat, H7,
H8, or 120B work. Swap and GPU cleanup passed, the old owner cache remained
byte-identical, and no target process remained. The cgroup retained
11,515,166,720 bytes of maximum post-exit current drift, so its frozen cleanup
gate also remained failed; no cache-clearing or threshold change was made. The
[bounded retry record](evidence/implementation-2026-08/r4-retained-20b/retry-20260817-155545/README.md)
binds the raw external evidence.

## Authorized release-boundary correction

The failed first cell exposed a supervisor phase error rather than a reason to
change the frozen allowance. Clean file cache was being compared with the
allowance while checkpoint-backed source mappings were still intentionally
live. The corrected construction paths now retain the original read-only file
descriptors, apply `MADV_DONTNEED`, unmap the source, apply
`POSIX_FADV_DONTNEED`, close the descriptor, and prove zero source mappings,
PSS, and descriptors before publishing a task-unique release-ready marker.
Filenames in this telemetry preserve their raw Unix bytes and do not add serial,
UUID, or block-payload reads.

The supervisor validates the nonce-, cell-, constructor-, ordinal-, and
R2-policy-bound release proof, then performs the unchanged 30-second plus five
one-second-sample clean-file gate. Only a passing, hash-bound continuation
marker lets the child proceed. H7 performs that handshake twice. Cgroup swap,
dirty, and writeback remain continuous stop gates; the exact 64-MiB
`memory.current` and file-drift checks remain post-exit gates. No threshold,
matrix cell, cache rule, process-group rule, or cleanup rule changed.

Current construction evidence is schema v7 for monolithic, v3 for
capacity-one, and v4 for H7; the comparison run record is v3. Historical
records remain untouched. Synthetic release-proof, stale-marker, duplicate
publication, exact-boundary, phase-order, output-schema, and fixed-matrix tests
pass alongside the protected-NVMe topology fixtures.

## Final authorized R4 attempt

Commit `2e8820b` was pushed and the three release binaries were built from that
clean identity. The fresh six-sample preflight passed every frozen R2 gate over
120,019 ms. `cold-monolithic-control` then passed the corrected boundary: its
advice calls succeeded, source mapping/PSS/descriptors reached zero, five
clean-file samples passed over 34,503 ms, terminal v7 output validated, and
post-exit memory/file drift was zero.

The fixed matrix stopped in `cold-capacity-one` before source mapping. The
local native 20B package is one 13,761,300,984-byte shard, which exceeds the
unchanged 10,544,040,680-byte capacity-one source-mapping window by
3,217,260,304 bytes. The constructor exited with code 1 without a release
marker, terminal output, or capacity-one cache. No warm, repeat, H7, H8, or
120B work followed. The
[bounded final record](evidence/implementation-2026-08/r4-retained-20b/release-retry-20260817-165733/README.md)
binds the raw external evidence and final cleanup audit.

## Next decision

R4 is not complete, so the phase is not ready for H8. The single authorized
attempt has been consumed and may not be retried. Any change to the frozen
source-mapping window, checkpoint layout, or R4 procedure requires a new,
separate decision. H8, H9, and H10 remain stopped.
