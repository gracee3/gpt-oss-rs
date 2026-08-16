# H3 scoped shard transaction evidence

**Status:** source and tiny-synthetic-fixture prerequisite validated for
supervisor review. This record does not represent constructor integration,
model construction, CUDA publication, or an H8/H9/H10 gate.

The retained [validation manifest](validation.json) binds the exact four-file
loader source identity and the final default-feature model-runner test
executable. Focused tests prove:

- one mapping and its exact file length are the count/byte high-water;
- nested admission and a channel-coordinated true cross-thread attempt reject
  through the same catalog state before a second file open, then a clean retry
  succeeds after release;
- only checked plan actions expose bytes;
- plan, catalog, shard, action, opened-file, and header mismatches fail closed;
- pre-handoff error and panic release normally and permit retry;
- unproven post-handoff normal return, error, and panic retain the mmap and
  permanently quarantine that catalog instance;
- a successful terminal callback permits release; and
- a failed terminal callback quarantines and cannot be rehabilitated.

Stale opened-file and header identities fail before a successful mmap, so
their mapped-byte high-water remains exactly zero. A quarantined mmap remains
live for process life; the external checkpoint-immutability obligation lasts
equally long. Keeping a `File` open would not prevent external mutation or
truncation, and the process-local catalog quarantine is not artifact-level
revocation.

The synthetic 20B/120B metadata-plan fixtures retain their pinned identities;
they contain shape/range metadata only and no model payload. The public
`GptOssShardConsumerPlan` adapter is separately exercised with a real tiny
three-byte SafeTensors catalog and a fully framed synthetic plan.

Every validation command explicitly removes all established model paths and
opt-in run gates. No real model/checkpoint path was opened, statted, mapped,
hashed, constructed, or executed. H8/H9/H10 were not started. No host, swap,
watchdog, admission, reserve, storage, network, or system state was changed.

The design boundary and remaining real-20B constructor gate are documented in
[`../../../34-h3-scoped-shard-transaction.md`](../../../34-h3-scoped-shard-transaction.md).

Verify this bounded record with:

```bash
cd docs/het/evidence/implementation-2026-08/h3-scoped-shard-transaction
jq -e . validation.json >/dev/null
sha256sum -c SHA256SUMS
```
