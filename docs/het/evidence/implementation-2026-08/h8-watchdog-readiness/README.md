# Pre-H8 watchdog readiness

**Status:** watchdog contract implemented and validated; H8 remains unpassed and
was not run. A future H8 retry still requires separate authorization/review and
a fresh passing preflight.

This follow-up adds a standalone, non-CUDA admission and supervision boundary
for the construction-only H8 probe. It does not weaken the frozen global
no-new-swap gate, load model weights, initialize the 120B checkpoint, or start
H9/H10.

## Contract

- `heterogeneous_h8_watchdog preflight` retains at least four samples spanning
  120,000 ms from the first retained sample. `SwapFree` and `SwapCached` must be
  byte-identical throughout; global swap use may not grow; target-tree `VmSwap`
  and memory PSI must remain zero.
- Process attribution reads `cmdline` first. An ordinary process never makes the
  scan incomplete solely because `/proc/<pid>/exe` is inaccessible. Only a
  command with exactly one `--mode h8` requires executable verification; an
  unreadable executable for that candidate fails closed.
- Admission recomputes the analysis and sample-chain hash instead of trusting a
  serialized `passed` bit. It requires a fresh preflight from the exact running
  watchdog inode and binds one opened `heterogeneous_construct` inode, one exact
  `--mode h8`, one new child-evidence path, repository HEAD, and swap baseline.
- The child runs in a separate process group with Linux parent-death signaling.
  Ordinary errors, watchdog drop, SIGINT/SIGTERM/SIGHUP, and abrupt watchdog
  death terminate and reap the complete child group. Runtime observation is
  fail-closed for swap, target-tree `VmSwap`, PSI, memory floor, process-scan
  completeness, protected-NVMe state, and the mandatory post-exit sample.
- Child evidence must bind the watchdog run, preflight, executable, source HEAD,
  and passing H8 schema before the watchdog can report success.

## Retained observations

All three records are preflight-only and non-passing. None launched
`heterogeneous_construct`.

| Record | Identity/status | Result |
|---|---|---|
| `preflight-observation.json` | superseded diagnostic | Five samples were retained, but the original timing anchored the window before the first sample and observed only 119,967 ms. The corrected analyzer rejected it. |
| `preflight-observation-v2.json` | strict non-passing diagnostic | The corrected window spanned exactly 120,000 ms. `SwapFree` released 4 KiB and the then-current scan treated unrelated root-owned `exe` permission failures as incomplete. The swap failure remains valid; the scan behavior was repaired rather than waived. |
| `preflight-observation-v3.json` | final-source live observation | Exactly 120,000 ms and five samples; `proc_scan_complete=true` throughout, no active H8, zero target-tree swap/PSI, and safe protected-NVMe state. It remains non-passing because `SwapFree` released 16 KiB and `SwapCached` released 4 KiB during the window. Global swap growth was zero. |

The final-source observation proves the `/proc` permission fix on this host. It
does not authorize H8 because exact byte stability did not hold.

## Identity

- Source parent: `07cb220a243eee63dd0dbd71e302d50f7b1fba73`.
- Reviewed source-set fingerprint:
  `c64f85e8324e56c3f87a172323bda40bece9436e8de1377e1d9fe3b508354a94`.
- Release watchdog SHA-256:
  `ec90806ded39a1de8ce9ffca9cdef3aca7df2682d2ea6b974b25295612258b7b`.
- Unexecuted fault-enabled construction binary SHA-256:
  `8798f8b1c4bd7dd355542ef5d0a414cd925a16b694a614ead1f03aad6fb6fd6d`.

The source-set fingerprint hashes, in listed order, the package manifest,
package library root, watchdog library, watchdog binary, construction binary,
and lifecycle integration test. `watchdog-readiness.json` retains their
individual hashes and the exact command inventory.

## Validation

Passed from the final source:

- locked watchdog library tests: 8/8;
- locked watchdog binary tests: 10/10;
- real subprocess lifecycle tests: 3/3, including SIGTERM, watchdog SIGKILL
  parent-death behavior, and fd-executed H8 detection;
- exact quota-balanced H8 manifest property test: 1/1;
- formatting and locked workspace/all-target check;
- strict package Clippy for the watchdog, plain `cuda` construction binary, and
  `heterogeneous-test-faults` construction binary;
- both release binaries built; the H8 construction binary was not executed.

Pre-existing warnings from dependency crates remain visible in CUDA builds and
were not changed or suppressed by this follow-up.

## Future admission

A separately authorized retry must first create a new output path with a fresh
120-second preflight from the reviewed release watchdog. The supervised `run`
action may be used only if that evidence passes unchanged validation. A
preflight that releases swap, allocates swap, observes target-process swap,
reports PSI, cannot complete attribution, or finds the protected NVMe unsafe is
not admissible. Prior H8 evidence under `../h8/` remains immutable and failed.
