# H3 gate record — owner-selective construction

**Status:** passed on 2026-08-16. H3 constructs the 20B proof placement once,
validates the 120B metadata envelope without loading 120B weights, and remains
detached from model execution. The commit introducing this record is the H3
completion commit.

The only persistent generated artifact is the project-scoped CPU x8 cache at
`/home/emmy/workspace/gpt-oss-rs-het-cache`. It contains two immutable,
identity-valid one-expert records totaling 26,508,424 bytes: the cold/warm
proof placement and the distinct lifecycle-fault placement. Native model files
under `/data` were opened read-only. No model snapshot was copied or
transformed, and no 120B weight was loaded or executed.

## Final probe identity and records

All four final records were produced in order by the same release probe,
SHA-256
`2e50c9d703e0556e8194becdb2d2748ab29fa6b986e2a3c7edebf92721ed18eb`.
Earlier captures were overwritten after feature-shape lint repairs changed the
binary; they are not H3 evidence.

| Record | Result |
|---|---|
| [`metadata-validation.json`](metadata-validation.json) | 20B 363→459 and 120B 543→687 native/runtime mappings exactly match the Phase 1 ledgers; both proof manifests cover every expert once |
| [`mapping-20b.generated.json`](mapping-20b.generated.json) | Complete generated 20B native-to-runtime map |
| [`mapping-120b.generated.json`](mapping-120b.generated.json) | Complete generated 120B native-to-runtime map |
| [`placement-20b.json`](placement-20b.json) | 766 GPU0 / 1 GPU1 / 1 CPU; manifest `cd72f92f…a0fe` |
| [`placement-120b-existence.json`](placement-120b-existence.json) | 1,299 GPU0 / 1,620 GPU1 / 1,689 CPU arithmetic envelope; manifest `ea7707d2…1b9e` |
| [`20b-cold-construction.json`](20b-cold-construction.json) | Passed from an absent cache root; atomically created one 13,254,212-byte CPU record |
| [`20b-warm-construction.json`](20b-warm-construction.json) | Passed; reused the record with no byte growth |
| [`20b-construction-faults.json`](20b-construction-faults.json) | Passed all eight real-constructor fault points and a subsequent clean construction |
| [`SHA256SUMS`](SHA256SUMS) | SHA-256 identities for the bounded JSON evidence |

## Construction and ownership result

The final 20B ledger is exact in both cold and warm runs:

| Category | Count / bytes |
|---|---:|
| Read-only native address mappings | 13,761,264,768 B |
| GPU0 non-expert dense tensors | 315 tensors / 3,595,648,128 B |
| GPU0 native-packed experts | 766 / 10,139,143,680 B |
| GPU1 native-packed experts | 1 / 13,236,480 B |
| CPU owner-filtered x8 experts | 1 / 13,253,760 B payload |
| Reusable pinned construction lease | 16,777,216 B high-water |
| CPU conversion temporary | 1,114,112 B high-water |

All six blocks/scales/bias surfaces of every GPU-owned expert pass through the
single reusable pinned staging lease into their final allocation. The lease is
destroyed before construction continues to CPU records. GPU handles retain
only native packed MXFP4; CPU records contain only CPU-owned x8 payloads. The
manifest-to-materialized key comparison proves exactly one representation per
expert before publication.

Cold construction took 48,939 ms and peaked at 13,995,933,696 B RSS and
13,991,967,744 B PSS. Peak anonymous PSS was 70,172,672 B; peak file PSS was
13,905,357,824 B. Warm construction took 48,765 ms and did not change the
13,254,212-byte cache. Swap remained zero. Both runs restored CUDA free bytes
exactly to 25,005,785,088 B on the layer owner and 25,017,319,424 B on the
remote worker after teardown.

The 120B metadata envelope remains an existence proof, not H8 admission. It
accounts for 65,248,815,744 mapped native bytes, 4,255,115,904 non-expert
bytes, 17,194,187,520 B of GPU0 native expert payload, 21,443,097,600 B of
GPU1 native expert payload, 22,385,600,640 B of CPU x8 records, and a 4 GiB
inclusive reserve per GPU. H8 must recompute owner counts from then-current
driver-usable memory and measured reserve categories before allocating 120B;
H3 did not treat the example counts as an executable placement.

## Real-constructor failure gate

`construct_with_fault` exists only under the explicit
`heterogeneous-test-faults` feature. Each fault runs through
`OwnerSelectiveConstructor` and fires after the named stage's real action and
observer callback. The earlier standalone mock rollback test was removed and
is not claimed as evidence.

| Injected stage | Real state reached | CUDA cleanup | Pinned current after | Cache effect |
|---|---|---|---:|---|
| identity | source/manifest identity validated | exact on both GPUs | 0 | none |
| runtime baseline | both CUDA executors created | exact | 0 | none |
| mappings | complete mapped-view ledger observed | exact | 0 | none |
| layer-owner dense | first real dense allocation/upload | exact | 0 | none |
| GPU experts | first real native expert allocation/upload | exact | 0 | none |
| CPU experts | full GPU topology plus atomically published CPU record | exact | 0 | one valid 13,254,212-byte record |
| execution reserve | all owners materialized and verified once | exact | 0 | published record reused |
| publish | final byte/ownership ledger observed | exact | 0 | published record reused |

Every case retained zero partial files/locks, zero swap growth, and a pinned
high-water no greater than 16 MiB. CUDA free bytes before and after every case
were identical. After all injections, a clean 48,728 ms construction reused
both identity-valid records without cache growth, passed exact ownership and
byte checks, and unloaded exactly.

The cache writer charges existing regular files plus the exact aligned header
and upcoming payload before creating a temporary file. Publication uses
`create_new`, file `fsync`, atomic rename, and directory `fsync`; errors remove
only task-created temporary files. Recursive accounting and partial-artifact
inspection use `symlink_metadata` and refuse symlinks rather than following
them.

## Gate commands

All commands ran from `/home/emmy/gpt-oss-rs` with the lockfile, Rust 1.97.1,
CUDA 13.3 targeting `sm_86`, and NVIDIA driver 610.43.02.

| Command or check | Result |
|---|---|
| Final release `heterogeneous_construct` build with `heterogeneous-test-faults` | Passed; all four final records share its SHA-256 above |
| Final validate → cold → warm → faults sequence | Passed; cold began with absent cache, warm reused it, faults added only the distinct valid record |
| `cargo test --locked -p gpt-oss-model-runner cpu_repack` | 9 passed, including exact upcoming-capacity charging and symlink refusal |
| `cargo test --locked -p gpt-oss-model-runner --features heterogeneous-test-faults --test owner_selective_contract --release` | Passed conservative representation/temp bounds |
| H2 synthetic selected-expert CUDA regression on both GPUs | Passed, including exact special values and lifecycle faults |
| H2 real four-expert/two-GPU oracle regression | Passed all eight routes exactly |
| `cargo check --workspace --locked` and `cargo test --workspace --locked` | Passed |
| Three configured strict Clippy lanes | Passed |
| Strict H3 probe Clippy with `cuda` and with `heterogeneous-test-faults` | Passed; no H3 warning in either feature shape |
| `CUDA_ARCH=sm_86 cargo build --release --locked --features cuda` | Passed; pre-existing experimental model-runner/engine warnings remain inventoried |
| Python unit discovery | 35 benchmark-tool and 10 oracle tests passed |
| `cargo fmt --all -- --check`, `git diff --check`, Markdown links, and scope audit | Passed at package close |

## Scope and safety

H3 did not connect the constructor to the model forward path, execute a model,
load 120B weights, move expert weights during decode, use NCCL/P2P, or touch the
rejected all-expert CUDA path. No dependency or lockfile change was introduced.
No remote state changed. At close, no construction process remained, system
swap use was zero, both GPUs were idle, and `/dev/nvme1n1` remained read-only
and unmounted.
