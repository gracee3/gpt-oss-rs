# Final authorized H8 attempt

**Status:** blocked before construction. No H8 child or model load was started,
and the single construction launch remained unused because the required final
preflight did not pass.

## Final committed-identity admission

The fail-closed process-identity repair is commit `2082c41`; the pre-fix
diagnostic record is commit `4de9dad`. The branch was clean and synchronized at
`4de9dadb295b6e7f633a9adb7b710b6dc9fbd391` before rebuilding both binaries.
The exact final identities are:

- watchdog SHA-256:
  `916146f92a30409d6f4c49951b82f962a76ea81187f27831f3c492e89b460b90`;
- unexecuted fault-enabled construction SHA-256:
  `b5925c5e69f4908491832772be847bf47b495d125ff6339621718849917afb7b`.

`preflight-final.json` retains five samples spanning exactly 120,000 ms. It
passed the memory floor, zero global swap growth, zero target-tree `VmSwap`,
zero memory PSI, complete process scan, no-active-H8, and protected-NVMe gates.
It failed the unchanged exact-byte-stability gate at the final sample:

- `SwapFree` increased by 12,288 bytes, from 107,349,860,352 to
  107,349,872,640 bytes;
- global swap used decreased by the same 12,288 bytes, from 24,317,952 to
  24,305,664 bytes;
- `SwapCached` increased by 20,480 bytes, from 1,843,200 to 1,863,680 bytes.

This is a swap release/cache change, not new swap allocation, but the frozen
preflight requires both counters to remain byte-identical. The watchdog wrote
the immutable failure and exited without accepting a `run` action. No
`120b-construction.json`, watchdog-run record, or new placement artifact exists.
Both GPUs remained at the idle 9 MiB / 1 MiB allocation state, and the
project-scoped cache remained the same two 20B records totaling 26,508,424
bytes with no partial or symlink artifact. The protected NVMe remained
read-only and unmounted. H8 is unpassed; H9/H10 remain prohibited.

## Pre-fix admission diagnostic

`preflight.json` was produced from clean synchronized HEAD
`b4b7206692814233973b30e6ff690d5351b0a34d` by watchdog executable SHA-256
`80f516abb737b1e3421825cf63ad2727b38b062900c9a3b3ebb0ff7877f34b98`.
Its five samples span exactly 120,000 ms and pass every frozen preflight gate.
The subsequent `run` command rejected before spawning a child because the
process scanner confused the watchdog's own trailing child arguments with an
active constructor. No model was opened and no construction attempt occurred.

Commit `2082c41` narrows the readable-process identity fallback to the exact
Linux-truncated constructor name and adds the real watchdog-command regression.
The final preflight was generated from the clean committed identity above. This
diagnostic remains immutable and was not used to admit a construction run.
