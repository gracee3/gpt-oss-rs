# H8 attempt record — stopped on swap growth

**Status:** failed gate; H8 is not complete. The initial construction-only
probe was stopped with SIGINT after host-wide swap allocation grew by 7,409,664
bytes from its clean preflight baseline. Read-only attribution found those
swapped pages in non-target idle supervisor Codex, Docker daemon, and shell/
session processes; `heterogeneous_construct` remained at `VmSwap: 0`.

With no H8 process, SwapFree and SwapCached were then unchanged for 138 seconds.
The one authorized retry captured that new baseline and stopped itself at the
real `GpuExperts` resource observation when global swap grew by another 704,512
bytes. The target process again remained at `VmSwap: 0`. No second retry is
authorized or planned, and `120b-construction.json` was not produced.

The corrected placement manifest is retained only as diagnostic evidence. It
uses the deterministic quota-balanced SHA-256 policy, contains all 4,608 keys
exactly once, matches the live measured global quotas `1,820 / 1,233 / 1,555`
in CPU/layer-owner/remote order, gives every one of the 36 layers all three
owners, and has per-layer ranges `50–51 / 34–35 / 43–44`. This establishes the
manifest repair, not the H8 construction gate.

After both attempts, the process exited, both GPUs returned to 9 MiB and 1 MiB
used, the authorized cache remained exactly two files and 26,508,424 bytes,
and no temporary, lock, symlink, or 120B owner record remained. The protected
NVMe remained read-only and unmounted.

Files:

- `placement-120b-h8.json`: atomic corrected manifest from admission.
- `failed-attempt-swap-growth.json`: bounded preflight, stop, manifest, and
  cleanup facts. This is explicitly a failed-attempt record.

H8 remains unpassed. Resume only after review of the global-growth requirement
or a separately authorized new run; do not treat the manifest-only result as a
construction or memory-envelope pass.

## Post-attempt source checkpoint

The failed attempts ran with source fingerprint
`3b80197a32e18a0509df7661e965e6312fb5f8cb86cb4238f46896e8683cb00a`
and executable
`55c645b1637a3ac966871d013a07554112df00a063f929de98f3eff1d622600b`.
Afterward, H8-only imports, constants, and the disk helper were placed behind
the existing `heterogeneous-test-faults` feature boundary. This does not change
the H8 runtime path, but fixes the plain-CUDA build surface without allowances.

The resulting checkpoint source fingerprint is
`bb1dea2c0fd34d1d4879447eda8ad9c4b4dff2b1e80ca66ed223d956d2c38035`;
its unexecuted fault-enabled release binary is
`cf1834e6c20f01a45b96d63f4a1fee634b0bda784877cda5f2161437d9d4bbd5`.
Format, locked workspace/all-target check, fault-feature check, the exact quota
property test, and strict Clippy under both `cuda` and
`heterogeneous-test-faults` passed. H8 was deliberately not rerun.
