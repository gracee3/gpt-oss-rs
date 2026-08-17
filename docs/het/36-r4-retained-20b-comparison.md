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
the protected-device state. Cold and warm 20B construction no longer require
or open a 120B checkpoint. The capacity-one H7 control constructs from the
bounded catalog and native metadata map before entering the unchanged retained
runtime path.

Runtime supervision enforces target and cgroup swap zero, global swap
nonincrease, the 12-GiB `MemAvailable` floor, PSI `some/full avg10` zero,
complete process attribution, the clean-file and dirty/writeback allowances,
the 30-second plus five-sample settle protocol, 64-MiB post-exit memory and file
drift, GPU cleanup, and the protected NVMe guard. Retained samples include
cgroup `anon`, `file`, `file_mapped`, `file_dirty`, `file_writeback`, and swap.

## Authorized attempt

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

## Next decision

A future comparison attempt requires separate authorization and a genuinely
fresh preflight in a cgroup with zero `memory.swap.current` while PSI avg10
remains zero for the complete window. The frozen thresholds must not be changed
to convert this result into a pass. H8, H9, and H10 remain stopped.
