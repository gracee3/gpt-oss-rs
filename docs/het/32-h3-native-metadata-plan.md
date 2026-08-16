# H3 payload-free native metadata and shard-consumer plan

**Status:** source and synthetic-fixture milestone validated; not integrated
into owner-selective construction. H8 remains unpassed and H9/H10 remain
prohibited. No real 20B or 120B model path was opened, mapped, hashed,
constructed, or executed during this work.

## Purpose and boundary

Document 31 established a bounded SafeTensors header catalog but deliberately
stopped before native GPT-OSS mapping and placement. This milestone closes that
metadata-only seam:

```text
caller config bytes + bounded shard catalog
    -> exact native tensor-set validation
    -> deterministic native-to-runtime view map
    -> validated stable-device placement
    -> complete ordered per-shard consumer plan
```

None of these steps maps a payload. `GptOssCheckpointView` still owns the
existing `CpuTensorStore` in production, and `OwnerSelectiveConstructor` is
unchanged. The scoped mapping API from document 31 remains disconnected from
construction.

## Native mapping contract

`GptOssNativeCatalogMap::from_config_bytes` accepts caller-supplied config
bytes, a nonempty checkpoint revision, and `SafeTensorShardCatalog`. It:

1. accepts only the existing exact GPT-OSS 20B or 120B dimensions;
2. requires the complete native tensor set with no missing or extra name;
3. validates dtype, shape, checked byte count, and shard identity for every
   native tensor;
4. produces the existing aliases plus contiguous Q/K/V row slices; and
5. retains config, catalog, compatibility-metadata, and v1 runtime-mapping
   identities without retaining tensor bytes.

The 20B shape maps 363 native tensors to 459 runtime views. The 120B shape maps
543 native tensors to 687 runtime views. Every mapping slice is checked to have
exactly `mapping.bytes` before it can be expanded into owner actions.

### Production compatibility

The existing mapping functions now consume a private metadata-source trait.
The `CpuTensorStore` adapter preserves its prior filename behavior exactly,
including the historical `"shard"` fallback for a non-UTF8 filename, and uses
the unchanged `gpt-oss-rs-native-metadata-v1` and
`gpt-oss-rs.native-checkpoint-map/v1` identity algorithms. Strict UTF-8 shard
validation remains a property of the new bounded catalog only.

A focused executable compatibility fixture opens a tiny `CpuTensorStore` with
a non-UTF8 shard filename on Unix and compares the refactored hash against a
verbatim test copy of the pre-refactor algorithm. The hashes and fallback name
match exactly.

`GptOssExpertPlacementManifestV1::validate_static` reuses the same header,
assignment, duplicate, budget, and identity checks as the existing resolved
validation, but does not resolve CUDA ordinals. Existing resolved placement
still validates in the same order and returns the same assignment/count/hash
results.

## Per-shard consumer plan

`GptOssShardConsumerPlan` binds four immutable authorities before any payload
access:

- catalog identity;
- native compatibility metadata and runtime mapping identities;
- stable-device placement identity and epoch; and
- the exact per-tensor checked absolute ranges in each shard.

Dense aliases and Q/K/V slices become layer-owner actions. Each of the six
expert surfaces becomes one action per expert with the exact `(layer, expert)`
key and its single static CPU, layer-owner GPU, or remote-GPU owner. Actions are
sorted by absolute range and must cover every shard payload byte exactly once,
without a gap, overlap, reversed range, missing shard, wrong shard, or
arithmetic overflow. Action count is capped at 100,000.

The serialized record and its schema-framed hash input use
`gpt-oss-rs.shard-consumer-plan/v1`. The hash input is an ordered typed JSON
object containing explicit fields, sequences, and enum records. Serialization
and hash input exclude catalog filesystem paths and
process-local filesystem device/inode guards. They intentionally retain stable
CUDA PCI/device ownership because that is part of placement meaning.

## Deterministic synthetic identities

The exhaustive fixtures allocate metadata only. They represent model-scale
shapes and checked byte ranges but contain no model payload. Creation order is
reversed to prove placement/plan determinism.

| Shape | Native/runtime views | Shards | Actions | Compatibility metadata SHA-256 | Mapping SHA-256 | Plan SHA-256 |
|---|---:|---:|---:|---|---|---|
| 20B (`24 x 32`) | `363 / 459` | 3 | 4,923 | `6205a6c690ef4168328d61fcc0896998778df81d367ba33357e692718ed2b04d` | `bd6b537ca72ade7c71c37a4a9447820d2bcbde66590a03196f709c8e38c0c79c` | `7ae911f978d42957d0f35740c23c04fcd31b5bfde627b44f996a8fc16b8859a3` |
| 120B (`36 x 128`) | `543 / 687` | 7 | 28,119 | `ff89ee14ea3c70b1114d86c9a3a38366c73dccdcb9d1e5c58eed772cd377ef00` | `7fccaf4f0bb43abd321111f301ce30265d8f4fc2fbb309626b7885aa1b739240` | `085f20acca3343358c3d5a5105d853e7696f49a3f3f193c4b4998a759c76afc6` |

A separate real `SafeTensorShardCatalog` adapter test uses one temporary
three-byte U8 tensor. It proves the production adapter consumes catalog
metadata without requiring the exhaustive fake; it is not a model fixture.

Fail-closed tests cover missing/extra tensor sets, bad shapes, mismatched
catalog/mapping/placement identities, duplicate placement assignments, missing
or wrong shards, reversed/overflowing ranges, range gaps, all six surfaces per
expert, exact owner retention, deterministic ordering, and host-identity
exclusion.

## Limits and next boundary

**Verified:** a full, owner-specific shard schedule can be validated and hashed
without mapping payload. Existing production checkpoint and constructor
behavior is unchanged.

**Unknown:** this source-only milestone does not prove that the local model
artifacts produce these synthetic identities, that a payload range can be
released after upload/repack, or that router and CUDA lifetimes permit a shard
transaction to close.

**Next integration boundary:** a later reviewed H3 change may make the
constructor consume a validated plan one shard at a time only after it supplies
terminal CUDA-copy ownership, atomic CPU-record publication, resident router
handles, and quarantine on uncertain drain. That integration must first pass a
20B construction/control gate and still requires separate authorization before
another H8 attempt.

No swap, watchdog, reserve, placement, protected-NVMe, or identity gate is
weakened. H8 remains unpassed.
