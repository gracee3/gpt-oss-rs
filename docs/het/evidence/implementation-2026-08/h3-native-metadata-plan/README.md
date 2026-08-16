# H3 payload-free native-plan evidence

**Status:** source and synthetic-fixture gates passed. No real model path,
construction command, or inference command was used. H8 remains unpassed.

## Retained proof

- caller-supplied config plus bounded catalog metadata validates exact native
  tensor sets and the existing 363-to-459 / 543-to-687 runtime maps;
- a tiny `CpuTensorStore` with a non-UTF8 filename proves its refactored hash
  equals the verbatim legacy algorithm and retains the prior fallback;
- stable-device placement is fully validated without CUDA ordinal discovery;
- each mapping slice equals its declared byte length before expansion;
- all six surfaces of every expert retain exact single-owner identity;
- ordered actions exactly partition every synthetic shard payload;
- 20B/120B action counts and compatibility, mapping, and framed-plan hashes are
  pinned;
- a real bounded-catalog adapter is exercised with a three-byte temporary
  fixture; and
- serialized plans and hashes exclude process-local filesystem device/inode
  guards and host paths while retaining durable CUDA placement identity.

The full boundary and next integration seam are in
[`../../../32-h3-native-metadata-plan.md`](../../../32-h3-native-metadata-plan.md).

[`validation.json`](validation.json) records source identities, exact commands,
visible focused tests, workspace checks, and the touched-path warning audit.
The broad model-runner Clippy inventory remains pre-existing and is retained as
a non-gate diagnostic. No fixture payload or host identifier is stored here.
