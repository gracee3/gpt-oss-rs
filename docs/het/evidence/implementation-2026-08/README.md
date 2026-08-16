# Heterogeneous implementation evidence — 2026-08

This directory contains bounded, sanitized gate records for the H0–H10
implementation campaign. Large logs, binaries, model-derived tensors, host
names, GPU UUIDs/serials, and other machine identifiers are intentionally not
retained.

| Package | Record | Status |
|---|---|---|
| H0 | Phase boundaries are represented by commits `f4d4f2b` and `1909686` | passed |
| H1 | [contracts, placement, identity, and evidence schemas](H1-contracts.md) | passed; the record's introducing commit is the completion commit |
| H2 | [selected-expert CUDA gate](h2/README.md) | passed; exact synthetic and four-expert/two-GPU real oracle, post-enqueue drain, bounded high-water, and preserved failed attempt; the record's introducing commit is the completion commit |
| H3 | [owner-selective construction gate](h3/README.md) | passed; exact single-owner construction, bounded pinned/x8 staging, cold/warm cleanup, 120B metadata envelope, and real eight-stage rollback campaign; the record's introducing commit is the completion commit |
