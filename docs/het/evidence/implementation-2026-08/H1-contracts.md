# H1 gate record — contracts, placement, identity, and evidence

**Status:** passed locally on 2026-08-15. The completion commit is the commit
containing this record; the package's start commit was
`19096865398a0515c5a9c78d16a0e9c5f71d837b` on
`agent/het-implementation`.

## Scope and invariants

- Added durable CUDA identity as normalized PCI domain/bus/device/function,
  with process-local ordinal resolution and name, compute-capability, and
  minimum-memory admission checks.
- Added complete, hashed GPT-OSS placement manifests for the 20B `24×32` and
  120B `36×128` expert rectangles. Every assignment is checked once, GPU roles
  name their declared stable device, and count quotas are checked before any
  future materialization.
- Added canonical BF16-bit route descriptors carrying source row, expert ID,
  route rank, activation slot, and canonical result slot through stable
  owner/expert grouping. Rank is never reconstructed.
- Added representation, result, completion, prepared-state, and deterministic
  error-precedence descriptors. No loader, execution, K/V, CLI, or service path
  consumes them in H1.
- Added `gpt-oss-rs.heterogeneous-step-trace/v1` terminal evidence, including
  stable devices, identities, epochs, canonical routes, bounded resources,
  intervals, errors, and committed/discarded visibility rules.
- `serde` was added to `gpt-oss-gpu` from the already pinned workspace
  dependency. The only lockfile change is that local package's dependency-list
  entry; no registry package or version changed.

## Starting resource snapshot

The host had approximately 89.4 GiB available memory, zero swap in use, and
both GPUs idle (single-digit MiB driver use) before H1. CUDA enumeration
resolved two RTX 3090 devices by their stable PCI identities. The protected
secondary NVMe reported `RO=1` and had no mount.

## Gate commands

All commands ran from `/home/emmy/gpt-oss-rs` with the repository lockfile.

| Command | Result |
|---|---|
| `cargo test --locked -p gpt-oss-evidence` | 14 passed; schema golden and negative visibility/rank cases passed |
| `cargo test --locked -p gpt-oss-model-runner heterogeneous::` | 13 passed; 20B/120B coverage, owner errors, PCI mismatch, ordinal permutation, BF16, grouping, result identity, state and error ordering passed |
| `cargo test --locked -p gpt-oss-gpu device::tests::` | 5 mock/default identity tests passed |
| `cargo test --locked -p gpt-oss-gpu --features cuda cuda_devices_publish_resolvable_stable_pci_identities` | passed against both locally enumerated CUDA devices |
| `cargo check --workspace --locked` | passed |
| `cargo test --workspace --locked` | passed; all workspace and doc tests green |
| `cargo check --locked -p gpt-oss-model-runner --features cuda` | passed with the pre-existing experimental CUDA warning inventory |
| `cargo clippy --locked -p gpt-oss-evidence -p gpt-oss-gpu --all-targets --no-deps -- -D warnings` | passed |
| `cargo clippy --locked -p gpt-oss-gpu --all-targets --no-default-features --features cuda --no-deps -- -D warnings` | passed |
| `cargo clippy -p gpt-oss-evidence -p gpt-oss-bench --all-targets --no-deps --locked -- -D warnings` | passed (configured CI lane) |
| `cargo fmt --all -- --check` and `git diff --check` | passed |
| Python benchmark/oracle unit discovery and `tools/check_markdown_links.py` | 35 + 10 tests passed; 110 Markdown files validated before this record was added |

An exploratory `cargo test --locked -p gpt-oss-model-runner --features cuda
heterogeneous::` did not reach the filtered H1 tests because pre-existing
K/V-cache unit modules unconditionally import `MockGpuAllocator`, which is
intentionally absent when CUDA is enabled. This is not a configured H1 or
baseline CUDA test lane. The CUDA library check above and the real-device
stable-identity test both passed. H1 did not alter those K/V-cache tests or use
their failure as a substitute for a package gate.

## Deliberately omitted

No model was loaded, transformed, copied, or downloaded. No kernel behavior,
runtime dispatch, K/V state, service surface, Docker image, system setting,
source checkout, or remote Git state was changed. H2 CUDA expert work did not
begin in this package.
