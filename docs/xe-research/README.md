# Iris Xe X0–X8 Research

- Status: X0–X7 complete; X8 forced-only optimization complete
- Host: Lenovo T14, Tiger Lake-LP Iris Xe `8086:9a49`
- Validated stack: Compute Runtime 26.05.37020.3, Level Zero 1.28.2
- X0–X7 raw evidence: `/home/emmy/src/xe-research/results/20260811-xe-one-sweep/`
- X8 raw evidence: `/home/emmy/src/xe-research/results/20260812-xe-optimization-sprint/`
- X7 manifest SHA-256: `1378bd9ab319254d19ae95c91fc601e888ec56b34c301f2ffc2dbfe564a81430`

X7's `fail` is its predeclared promotion-gate result, not an overall research
failure: numerical, artifact, memory, checkpoint, and native-code evidence all
passed. X8 preserves that checkpoint and adds correct, faster forced OpenCL
research variants. It selects separate M=1, M=2, and M>=4 winners with
conservative intervals above canonical Xe parity. No production backend,
automatic dispatch, serving integration, dependency, or public runtime API was
added.

## Reports

| Work package | Report | Result |
| --- | --- | --- |
| X0 | [Environment, provenance, and dependencies](00-environment-provenance-and-dependencies.md) | pass |
| X1 | [Host, model, and execution lifecycles](01-host-model-execution-lifecycles.md) | decision-complete |
| X2 | [Capabilities, diagnostics, and timing](02-capabilities-diagnostics-and-timing.md) | pass |
| X3 | [Artifact pipeline, cache, and kernel ABI](03-artifact-pipeline-cache-and-kernel-abi.md) | pass |
| X4 | [Checkpoint ingestion and integrated memory](04-checkpoint-ingestion-and-integrated-memory.md) | pass |
| X5 | [MXFP4 exactness and Xe-LP code generation](05-mxfp4-exactness-and-xelp-codegen.md) | pass |
| X6 | [Real-tensor prefill vertical slice](06-real-tensor-prefill-vertical-slice.md) | useful-win fail |
| X7 | [Decision and forced implementation pre-plan](07-decision-and-forced-implementation-preplan.md) | negative closeout |
| X8 | [Performance diagnosis and forced-only optimization](08-performance-diagnosis-and-optimization.md) | selected forced decode/prefill winners |

## Claim-to-evidence index

All paths below are relative to the raw evidence root. Manifest hashes cover
their command, repository revision, timestamps, host/device identity, loaded
libraries and hashes, artifacts, exit status, and detailed result.

| Evidence ID | Claim | Raw manifest | SHA-256 |
| --- | --- | --- | --- |
| X0-OCL | current OpenCL environment and mixed-generation guard | `opencl/x0-opencl.manifest.json` | `60ef6bb4805f9d7deb264add62e8a09263f75b47134ca68f4d839e1ef42cfddb` |
| X0-L0 | current Level Zero environment and mixed-generation guard | `level-zero-regular/x0-level-zero.manifest.json` | `5d42149ec1184119b3e87d417785413f4ae9f1bad0f8b27e32006365e5768e24` |
| X1-X2-OCL | OpenCL capabilities and safe negatives | `opencl/x1-x2-opencl.manifest.json` | `7ffb8c17ee16632064f037abfe53c3e41c1285ab16578587a9ee6e11c6beb6d7` |
| X1-X2-L0 | Level Zero capabilities and isolated negatives | `level-zero-regular/x1-x2-level-zero.manifest.json` | `c79ab93e9413493824a6bd2c88ae788297070495c1c454d47b9a284e17efd3d9` |
| X3-OCL | source, binary reload, and same-SPIR-V OpenCL paths | `opencl/x3-opencl.manifest.json` | `f5d5180ab4d907069491eeff4f8dd96b3dfabb0daf32e2313b71b03012ac8cd4` |
| X3-L0 | same-SPIR-V and native-reload Level Zero paths | `level-zero-regular/x3-level-zero.manifest.json` | `37ced41fee78c00c7c3be9a2b6eebc953b15edc9fbd59ebce1a89f655f0700c7` |
| X4-OCL | allocation classes, contention, and checkpoint slice | `opencl/x4-memory-opencl.manifest.json` | `09b656628b650fe2456fcedf1d16a4125ff37c08b28d29f91e9387464dd8af4e` |
| X4-L0 | allocation classes, contention, and checkpoint slice | `level-zero-regular/x4-memory-level-zero.manifest.json` | `5e9b6c9081958367276da6ea22db0aa95b544e8f2e90a2cd31ed0bd9436d411d` |
| X5-OCL | exhaustive/random exactness and OpenCL-native DP4A | `opencl/x5-opencl.manifest.json` | `51cc124d2c3e2a5a635dde1b54de2bfd57afcef30574f881ad908b0f294352bf` |
| X5-L0 | exhaustive/random exactness and Level Zero-native DP4A | `level-zero-regular/x5-level-zero.manifest.json` | `4ce3e32bd3c17cb47b2a54be571e0a20bc17e7e172bd7ea3bfa9b2f20e50af8c` |
| X6-OCL | all-shape OpenCL real-tensor performance | `opencl/x6-opencl.manifest.json` | `79cdaf5dea2b8d8bed784cfc2c6deaad17a12101abc93e046aafe752234e4b6a` |
| X6-L0-R | all-shape recycled regular-list performance | `level-zero-regular/x6-level-zero.manifest.json` | `6f903c4b9b224bdbeb4fff4b9a1204c3679d63d6f66332e6a1f4439331616639` |
| X6-L0-I | all-shape immediate-list performance | `level-zero-immediate/x6-level-zero.manifest.json` | `0e9d556a5f64531a6bbee4f475b5aa4089ea578e1ad367f07e2c6c74b446a5af` |
| X7 | performance-gated negative closeout and decisions | `closeout/x7-opencl.manifest.json` | `1378bd9ab319254d19ae95c91fc601e888ec56b34c301f2ffc2dbfe564a81430` |

The X0–X7 table above remains an immutable index into the original evidence
root. X8's independent claim and hash table is in the X8 report; its full raw
artifact index has SHA-256
`ac1f5a016ddb4fad2312123de44854cebbe82a82ed81787851d68bdf07e3f1a1`.

Supplemental X6 subgroup/work-group manifest hashes are listed in the X6
report. The final AC/display/frequency/thermal capture is
`x6-system-state.txt`, SHA-256
`c1d9e6f58626e10ab89fa42dbd3c6c9c9a473591fe3135d6d59438f5823e01f5`.
Generated SPIR-V, native modules, disassembly, logs, and raw benchmark JSON stay
outside Git; their individual hashes are embedded in the manifests.

## Reproduction boundary

The standalone research workspace has its own lockfile and does not belong to
the production Cargo workspace. Build and test it offline:

```console
RUSTC_WRAPPER= CARGO_NET_OFFLINE=true cargo build \
  --manifest-path tools/xe-research/Cargo.toml --locked --offline --release
RUSTC_WRAPPER= CARGO_NET_OFFLINE=true cargo test \
  --manifest-path tools/xe-research/Cargo.toml --locked --offline
```

Every hardware subcommand requires both an exact API and device:

```console
tools/xe-research/target/release/gpt-oss-xe-research capabilities \
  --backend opencl --device 8086:9a49 --results RESULTS
tools/xe-research/target/release/gpt-oss-xe-research benchmark \
  --backend opencl --device 8086:9a49 --results RESULTS \
  --variant tile32-m4-v2 --shapes 4,8,16,32,64,128 --local-size 32
```

OpenCL runs also set `OCL_ICD_VENDORS=/etc/OpenCL/vendors/intel.icd`. Level
Zero runs use `--backend level-zero`; immediate-list experiments additionally
use `--immediate`. The closeout command refuses to synthesize a negative result
if any primary X6 manifest reports a useful win.

These results are bounded to this T14, device, checkpoint slice, and captured
driver stack. They do not establish behavior on another Xe generation or
driver and do not justify full-model or serving claims.

## Validation

- standalone `cargo test --locked --offline`: 7 passed after X8;
- standalone warnings-denied Clippy: passed;
- production `gpt-oss-cpu-kernels`: 36 unit tests and doc tests passed;
- all hardware environment/capability/artifact/memory/MXFP4 commands completed;
- all primary and exploratory X6 paths completed with structured manifests;
- `git diff --check`: required before each checkpoint and final merge.

The full production `cargo test --workspace --locked --offline` build is
unavailable on this host because the existing `openssl-sys` dependency cannot
find OpenSSL development headers or `openssl.pc`. No package change was made.
This limitation is outside the standalone research graph and does not replace
the passing focused CPU-kernel validation.
