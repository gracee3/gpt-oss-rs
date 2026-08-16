# Phase 1 bounded evidence (`research-2026-08`)

This directory contains sanitized, review-sized evidence for the heterogeneous
research captured on 2026-08-15. Interpretation lives in documents `10` through
`19`; JSON `interpretation` fields are deliberately separate from raw result
fields. `SHA256SUMS` authenticates every retained evidence file except itself.

## Common identity and sources

- [`common-identity.json`](common-identity.json) records repository HEAD,
  pre-existing dirty-state fingerprint, sanitized hardware, and toolchains.
- [`source-pins.json`](source-pins.json) records official repository commits,
  licenses, papers, and the rejected paper identifier.
- The working tree's Phase 1 final state and protected-device check are recorded
  in `final-state.json` after validation.

## Checkpoint evidence

- [`checkpoint-summary.json`](checkpoint-summary.json) records commands,
  harness hashes, parameters, result hashes, units, timestamps, and the bounded
  interpretation.
- [`checkpoint-map-20b-full.json`](checkpoint-map-20b-full.json) is the complete
  459-entry native-to-runtime map and retains each tensor's SHA-256 after a
  streamed byte comparison.
- [`checkpoint-map-120b-metadata.json`](checkpoint-map-120b-metadata.json) is the
  complete 687-entry local-native-to-pinned-runtime metadata map. It explicitly
  says payload comparison was not performed.
- [`120b-runtime-config.json`](120b-runtime-config.json) and
  [`120b-runtime-index.json`](120b-runtime-index.json) are small official files
  pinned to revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`; no weight
  payload was fetched.
- [`hf-120b-api-metadata.json`](hf-120b-api-metadata.json) retains the official
  revision, root/LFS file identities, and exact disk sizes used by the offline
  snapshot envelope.
- [`120b-asset-hash-comparison.txt`](120b-asset-hash-comparison.txt) records the
  exact local-20B versus pinned-120B tokenizer/protocol asset identities.

## Measurement evidence

- [`control-20b-summary.json`](control-20b-summary.json) records the bounded
  internal load+prefill, retained-continuation/trace, and occupancy-profile runs.
- [`cuda-transfer-summary.json`](cuda-transfer-summary.json) records peer API
  results, steady transfer and relay distributions, first-copy controls, harness
  hashes, completion units, and caveats.
- [`nccl-summary.json`](nccl-summary.json) records the pinned build, sanitized
  SHM transport finding, validated all-reduce distributions, and raw-result hash.
- [`cpu-expert-summary.json`](cpu-expert-summary.json) records real-weight exact
  reference/repack measurements, existing matrix-kernel controls, output
  identity, policy boundary, and exploratory CPU/CUDA interference.

Raw per-sample JSONL, one-second monitor samples, and multi-megabyte trace/profile
records remain in `~/src/het-research/results` with the hashes recorded here.
They were not copied into the repository because the summary distributions,
complete checkpoint maps, and exact artifact hashes are sufficient for review.
The only raw diagnostic deliberately excluded from identity is verbose NCCL
debug text, whose prefixes contain machine/process identifiers; its transport
facts were sanitized into `nccl-summary.json`.

## Safety and scope

All measurement commands were bounded and interruptible. No 120B payload was
loaded, copied, transformed, or downloaded. No model shard was modified. No
Docker image or Python oracle environment was built. No hostname, GPU UUID,
serial number, filesystem UUID, token, IP/MAC address, or protected-device
content appears in this evidence.
