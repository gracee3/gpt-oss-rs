# H3 bounded SafeTensors shard catalog

**Status:** source/fixture milestone implemented; not integrated into owner-
selective construction. H8 remains unpassed and H9/H10 remain prohibited.
No real 20B or 120B checkpoint path was opened, queried with `stat`, mapped,
hashed, or otherwise accessed during this work.

## Purpose and boundary

Document 30 found that `CpuTensorStore` maps every shard before it can validate
the native tensor catalog, then retains all mappings for the checkpoint-view
lifetime. This milestone establishes the first bounded prerequisite for the
per-shard release candidate: `model_loader::shard_catalog` can validate a
SafeTensors artifact using only its index and shard headers. It does not replace
`CpuTensorStore`, `GptOssCheckpointView`, or `OwnerSelectiveConstructor`.

The only compatibility refactor keeps the production store's original unsafe
mapping at its explicit immutable-snapshot boundary and adds the narrowly named
`cpu_tensor_store::map_cataloged_immutable_shard` for the scoped catalog API.
That helper revalidates the cataloged inode and length, and its documentation
requires external immutability for the mapping lifetime; merely opening a file
read-only is explicitly insufficient. No production constructor calls the new
scoped API.

## Verified metadata contract

`SafeTensorShardCatalog::open` performs these operations in deterministic shard
filename order:

1. require a real, non-symlink root directory;
2. discover at most 4,096 regular, non-symlink `.safetensors` leaf files;
3. read at most 16 MiB of `model.safetensors.index.json`, when present;
4. require an index for a multi-shard artifact and exact agreement between its
   shard set, weight map, header tensor names, and per-tensor shard assignment;
5. read exactly the eight-byte SafeTensors header length and at most 16 MiB of
   JSON header per shard, with a 128-MiB total shard-header cap;
6. validate at most 1,000,000 tensors, 4,096 UTF-8 bytes per tensor name, and
   rank at most 64;
7. derive element and byte counts with checked arithmetic, then require each
   nonempty tensor range to be in-file and the sorted ranges to partition the
   shard payload with neither overlap nor gap; and
8. retain shard filename, file length, data start, payload length, header hash,
   process-local device/inode guard, tensor dtype/shape, relative offsets, and
   checked absolute ranges.

Index and header JSON maps reject duplicate keys. Index shard names must be one
UTF-8 `.safetensors` leaf: absolute names, parent traversal, forward or backward
separators, missing indexed shards, extra shard files, wrong-shard mappings,
and symlink substitution are rejected. Bounded index reads validate the
non-symlink path and opened inode/length, read only the declared length plus at
most one byte, then revalidate path, inode, and length. A growing file therefore
cannot turn a metadata read into an unbounded allocation.

## Identity semantics

Catalog identity uses schema
`gpt-oss-rs.safetensors-shard-catalog/v1` and SHA-256 over:

- the raw index SHA-256, if an index exists;
- ordered shard leaf names, file lengths, data starts, payload lengths, and
  header SHA-256 values; and
- ordered tensor names, shard indexes, dtypes, rank-prefixed shapes, and
  absolute ranges.

The process-local device/inode values are retained for open-time replacement
checks but excluded from serialization and the deterministic hash. Payload
contents and page residency are intentionally excluded: this is an immutable
*metadata artifact identity*, not a checkpoint content hash or a memory-fit
claim. The tiny two-shard fixture pins these exact identities:

```text
catalog 08650b3d5ccf811e80d040b84b75c03a3c09ee364ce033a3054a95e6b5375e08
index   fcd62bc8c750aa36660232e02d85d166f5fa56080d2c6b0161421c7984d1e89f
```

Reversing fixture creation order produces the same identities. Changing only
payload bytes also leaves catalog identity unchanged by design.

## Scoped one-shard mapping

`with_mapped_shard` is a separate, callback-scoped research API. Before mapping
it atomically reserves the catalog's single mapping slot, opens the cataloged
path, verifies that lstat/open refer to the retained inode and length, and
re-hashes the header from that opened file. The callback can borrow checked
tensor bytes only from that shard. Its return value cannot retain the mapping
borrow. Nested or concurrent mapping is rejected before another file opens.

RAII releases the mapping and active slot on success, returned error, and Rust
panic. Synthetic tests prove high-water `1`, current `0` after every outcome,
path-replacement and same-inode header-change rejection, and a clean mapping
after recoverable error/panic cleanup.

This API does **not** provide CUDA-drain, CPU-publication, or router-lifetime
proof. Consequently it is not connected to construction and is not evidence
that a real shard may yet be unmapped after upload.

## Payload-read and TOCTOU limits

**Verified:** catalog parsing requests only bytes `[0, 8 + header_length)`. A
synthetic reader advertises additional payload bytes but supplies only the
prefix; catalog header parsing succeeds and records its maximum read position
at the exact data start. No production catalog code seeks into payload.

**Assumption:** checkpoint directories and opened shard inodes remain immutable
while cataloged or mapped, as required by the existing loader. Revalidating the
opened inode, length, and header closes path replacement and header mutation,
but a same-inode payload write after catalog creation is deliberately not
detected because payload hashing would cross this milestone's boundary. A
future integration must bind the catalog to the existing authoritative model
manifest/shard integrity policy before construction.

**Assumption:** directory contents are not concurrently renamed during catalog
creation. Exact index/header set validation fails closed for observed changes,
but this is not a transactional filesystem snapshot.

## Next integration boundary

The next source milestone, still before another H8 attempt, is:

1. make native GPT-OSS mapping/placement validation consume this catalog rather
   than a payload-bearing `CpuTensorStore`;
2. derive and hash a complete deterministic per-shard consumer plan only after
   the native-to-runtime and owner manifests both validate;
3. remove the runtime router's post-construction borrow of checkpoint payload;
4. use `with_mapped_shard` for one planned shard only after GPU copies have a
   terminal event and CPU x8 publication owns its bytes; and
5. retain/quarantine a mapping on any uncertain CUDA drain rather than allowing
   callback return and unmap.

No `madvise`/`posix_fadvise` behavior is added here. No swap, watchdog, reserve,
placement, identity, or protected-storage gate changes. H8 remains unpassed.
