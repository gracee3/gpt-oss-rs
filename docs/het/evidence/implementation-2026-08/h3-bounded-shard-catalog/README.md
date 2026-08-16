# H3 bounded shard-catalog evidence

**Status:** source and tiny synthetic fixture gates passed. No model or
construction command was run. H8 remains unpassed.

## Retained proof

- bounded index/header-only catalog with deterministic order;
- exact index/header tensor-set and shard-assignment validation;
- checked dtype/shape/relative/absolute range validation;
- duplicate, path traversal, symlink, malformed/oversized metadata, overflow,
  overlap/gap, out-of-file, and replacement rejection;
- structural header reader that stops exactly at SafeTensors data start;
- deterministic fixture catalog/index hashes;
- capacity-one scoped shard mapping with success/error/panic cleanup; and
- serialized catalog descriptors exclude device, inode, and internal path;
- no integration into the production owner-selective constructor.

The full contract, identity boundary, TOCTOU assumptions, and next integration
seam are in
[`../../../31-h3-bounded-shard-catalog.md`](../../../31-h3-bounded-shard-catalog.md).

[`validation.json`](validation.json) retains the final source identities, the
visible 10-test `--lib` gate, workspace/Python/docs results, and the zero-
diagnostic touched-path warning audit. The broader model-runner strict Clippy
baseline remains red on 54 pre-existing diagnostics outside the touched files
and is recorded only as a non-gate diagnostic. `SHA256SUMS` covers this bounded
evidence directory; neither file contains fixture payloads or host identifiers.
