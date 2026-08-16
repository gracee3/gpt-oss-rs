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
| H4 | [exact router, bounded packing, and pinned relay](h4/README.md) | passed; native E=32/E=128 GPU router, canonical descriptors, collision-free bounded packing, real x8/native three-owner selected work, correlated overlap, and post-enqueue drain; the record's introducing commit is the completion commit |
| H5 | [deterministic reduction and atomic transaction](h5/README.md) | passed; exact GPU0 canonical-arena rank reduction, explicit relay-generation lifecycle, private K/V visibility epoch, full failure/cancellation matrices, and clean second-run proof; the record's introducing commit is the completion commit |
| H6 | [real one-layer owner shell and three-owner oracle](h6/README.md) | passed; exact real CPU/GPU0/GPU1 selected work, full route/completion identities, strict three-way overlap, commit/drained-discard/clean-repeat, fail-closed teardown, and bounded process/global resource evidence |
| H7 | [20B end-to-end retained continuation](h7/README.md) | passed; exact eight-token cold/warm continuation, real three-owner routes, atomic shell/coordinator publication, bounded memory, recoverable retry, and unproven-drain quarantine |
| H7 follow-up | [all top-4 owner cardinalities](h7-cardinality-followup/README.md) | passed; all 15 CPU/GPU0/GPU1 count triples, exact retained continuation, multi-result partial-submit recovery/quarantine, and preserved non-passing attempts 01/02 |
| Pre-H8 readiness | [exact execution-reserve admission contract](execution-reserve-readiness/README.md) | passed; checked 20B/120B metadata ledgers at context 4096, honest post-executor admission boundary, real eight-stage rollback, final H4–H7 regressions, and watched no-new-swap evidence; H8 remains stopped |
| Pre-H8 watchdog | [fail-closed admission and supervision](h8-watchdog-readiness/README.md) | validated without running H8; exact 120-second preflight, process/inode binding, process-group and parent-death cleanup, continuous fail-closed resource observation, and immutable non-passing live preflights; H8 remains stopped |
