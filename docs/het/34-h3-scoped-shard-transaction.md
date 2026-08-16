# H3 scoped shard consumption transaction

**Status:** source and tiny-synthetic-fixture prerequisite implemented for
review. It is not integrated into `OwnerSelectiveConstructor`, does not access
a real checkpoint, and does not pass or start H8/H9/H10.

## Purpose and boundary

Documents 31 and 32 established two separate authorities:

- `SafeTensorShardCatalog` validates immutable shard/header/tensor ranges and
  admits at most one callback-scoped mapping; and
- `GptOssShardConsumerPlan` binds every payload byte to its exact runtime
  consumer, owner, placement epoch, and deterministic plan identity.

The former previously released its mapping on every callback unwind, while the
latter did not participate in a mapping lifetime. That was safe only while no
external asynchronous consumer could retain a source address. This milestone
adds the narrow join:

```text
validated plan + matching catalog + shard index
    -> recompute framed plan identity and totals
    -> validate exact catalog/shard/action ranges
    -> reserve the catalog's sole mapping slot
    -> revalidate opened file and header identity
    -> synchronous action use OR explicit external handoff
    -> terminal proof, normal release, or irreversible quarantine
```

The existing payload-bearing checkpoint view and production constructor remain
unchanged. No CUDA stream, CPU x8 publication, resident router, or construction
ledger is connected here.

## Identity and range admission

Before any payload mapping, `with_scoped_shard_transaction` requires:

1. `GptOssShardConsumerPlan::validate_identity` recomputes checked action and
   payload totals plus the schema-framed v1 plan hash;
2. plan and catalog metadata identities match;
3. the indexed plan shard equals the cataloged filename, length, data start,
   payload length, header hash, and process-local inode/device guards;
4. every action is nonempty, ordered, gap-free, and nonoverlapping;
5. action and native-slice lengths match;
6. the named catalog tensor belongs to this shard and the absolute action
   range equals the checked tensor start plus its native slice; and
7. action bytes exactly cover the complete shard payload and equal the
   recorded per-shard payload total.

Only then does the catalog open, revalidate, and map the shard. A transaction
can expose only a plan action's checked slice, never the whole mmap. The
callback return type cannot borrow that slice.

## Lifecycle contract

| State | Mapping disposition | Retry |
|---|---|---|
| `PreHandoff` | success, error, or panic unmaps and releases capacity | allowed |
| `ExternalHandoffPending` | normal return, error, or panic retains the mmap for process life and quarantines the catalog | forbidden |
| `ExternalOwnershipTerminal` | the caller-supplied terminal operation returned success; mapping may unmap | allowed |
| `Quarantined` | retained mapping and poisoned catalog state remain | forbidden |

`with_synchronous_action` is explicitly limited to work that creates no
lifetime outside its callback. Any CUDA DMA, worker task, or other external use
must begin an external handoff first. The handoff owns the mutable transaction
borrow until `prove_terminal_with` runs the future integration's real drain.
A terminal callback error or panic cannot be retried or rehabilitated.

The source-only seam cannot determine that an arbitrary callback truly drained
CUDA. It therefore does not claim CUDA correctness: later constructor wiring
must supply the proven stream/event drain and retain destination ownership.

## Capacity and quarantine

The catalog now uses one shared `idle -> active -> quarantined` admission state
for both the earlier scoped mapping API and this transaction. Activity evidence
retains count and byte high-water values. Normal drop unmaps before returning
the slot to idle.

An unproven external lifetime uses safe `mem::forget` retention of the current
`Mmap`, leaves its exact byte count active, and moves the catalog instance to a
permanent quarantined state. Nested/concurrent admission and all later mapping
attempts reject. The quarantine is process-local and catalog-instance-local;
it is not an artifact-level revocation system.

Because the quarantined mmap remains live until process exit, the catalog's
existing external checkpoint-immutability obligation also lasts until process
exit. Retaining an open `File` would not enforce that obligation: another
actor could still write or truncate the inode. The future constructor must
therefore preserve the same externally immutable artifact contract even after
an unproven drain forces quarantine.

## Synthetic proof

Tiny generated SafeTensors fixtures cover:

- exact action bytes, out-of-range rejection, nested rejection, deterministic
  cross-thread concurrent rejection, and count/byte high-water equal to one
  mapping and its exact file length;
- framed plan, catalog, shard, and action-range mismatches before mapping;
- replaced-file and same-inode changed-header rejection with mapped-byte
  high-water remaining zero because neither reaches a successful mmap;
- pre-handoff returned error and panic cleanup followed by a clean retry;
- post-handoff normal return, returned error, and panic quarantine with
  permanent no-reuse;
- successful terminal proof and normal release; and
- failed terminal proof and retained quarantine.

The existing metadata-only synthetic 20B/120B plan tests also recompute their
pinned plan identities and reject a tampered framed total. They allocate no
model payload and access no model path.

## Remaining integration gate

A later separately reviewed H3 change must make the owner-selective constructor
consume this transaction while proving, for each shard:

- all GPU destinations own their copied bytes and every source-referencing
  stream/event is terminal;
- CPU x8 records are atomically published or remain rollback-owned;
- resident router and other dense consumers no longer borrow checkpoint data;
- every partial construction error drains or quarantines all external owners;
  and
- construction/evidence high-water remains bounded on the proven 20B control.

That production integration requires a real 20B gate and separate authority.
No swap, watchdog, admission, reserve, placement, protected-storage, network,
or host policy changed in this milestone.

Bounded evidence is indexed at
[`evidence/implementation-2026-08/h3-scoped-shard-transaction/README.md`](evidence/implementation-2026-08/h3-scoped-shard-transaction/README.md).
