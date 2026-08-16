# H3 loading-research instrumentation

**Status:** source and fixture gates passed; no model construction evidence was
generated. H8 remains unpassed.

This checkpoint adds an identity-bound, bounded construction-memory event
journal. It does not alter prior H3/H8 evidence and did not open, map, load, or
execute the 120B checkpoint.

## Retained contract

- event schema: `gpt-oss-rs.construction-memory-event/v1`;
- construction record schema: `gpt-oss-rs.heterogeneous-construction/v5`;
- maximum 64 events, 64 KiB per event, 4 MiB total;
- create-new event directory and create-new, synchronized publication;
- SHA-256/byte/sequence index in the final construction record;
- mandatory watchdog revalidation for a future successful H8 child;
- no change to process-zero-swap, global-no-growth, memory floor, GPU reserve,
  protected-NVMe, or model gates.

## Validation scope

The final source was checked with:

- construction-memory fixture/live sampler unit tests;
- the complete watchdog unit suite, including journal tamper rejection;
- the construction binary's quota-balanced manifest property test;
- locked normal-CUDA and fault-feature checks;
- strict no-dependency Clippy for the normal-CUDA construction binary, the
  fault-feature construction binary, and the watchdog;
- repository formatting and diff checks.

No `heterogeneous_construct` model mode, H8/H9/H10 command, 20B construction,
or 120B access was run. The proven 20B path was deliberately not repeated
because this instrumentation has complete synthetic parsing/publication
coverage and the user prohibited any 120B access; the current combined harness
performs preliminary 120B identity validation even for its legacy 20B modes.

The source audit and per-shard release candidate are recorded in
[`../../../30-h3-loading-research.md`](../../../30-h3-loading-research.md).
The bounded final command and source-identity record is
[`validation.json`](validation.json).
