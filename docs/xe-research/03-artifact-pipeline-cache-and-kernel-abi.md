# X3 — Artifact Pipeline, Cache, and Kernel ABI

- Result: pass
- Preferred portable artifact: reproducibly generated, validated SPIR-V
- Native binaries: runtime cache artifacts only
- Canonical kernel: exact integer `xe_i32_add`, 4,096 elements

## Same-SPIR-V result

`tools/xe-research/scripts/build-spirv.sh` compiles the checked-in source with
Clang 18 to LLVM bitcode, translates it with `llvm-spirv-18`, validates it with
`spirv-val --target-env opencl2.2`, and retains `spirv-dis` output. No fast-math
or unexplained optimization pass is used. The canonical bytes are:

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `elementwise.cl` | 455 | `decb68ffbb2ab27f95c138ef32084b59e08daaecae829d03cad3d156c7b9cd7f` |
| `elementwise.spv` | 1,452 | `caee7f071e7eeb4124ebe749db294367c839b23a5445af3356060edc633a4120` |
| ABI v1 JSON | 6,394 | `f22481e9c4920e85903f3be658b107ac16e6fc02763eae18c680d1cec2f451c2` |

The exact SPIR-V hash and identical inputs produced all 4,096 exact `i32`
outputs through OpenCL IL and Level Zero. The native module extracted after
those two same-SPIR-V loads was also byte-identical: 6,816 bytes,
`b8feb15cc7bc856eae8b48762b1c6a27ad17a0a4b31c0b057b8c7b6659fcab8a`.
That is a fact about this driver stack, not a portability promise.

OpenCL online source compilation produced a distinct 6,936-byte program binary
(`aba733946122d3deae7b582d56d05233b6d6720dd03ba1e75b6267f39e7db457`),
and reload validated exactly. Level Zero native extraction and reload also
validated exactly.

## Creation and warm execution

Each variant used ten process-cold creations, ten warmups, and thirty warm
samples. Medians from the final X3 manifests are:

| Variant | Cold create | Warm host | Warm device |
| --- | ---: | ---: | ---: |
| OpenCL source | 18.00 ms | 31.65 us | 5.00 us |
| OpenCL program-binary reload | 17.88 ms | 32.63 us | 5.36 us |
| OpenCL same SPIR-V | 17.67 ms | 32.09 us | 5.21 us |
| Level Zero same SPIR-V, regular list | 27.17 ms | 26.82 us | 6.76 us |
| Level Zero native reload | 27.01 ms | 28.64 us | 7.38 us |

The native caches did not materially reduce process-cold creation here. These
micro-timings are artifact evidence only; they do not select a model API.

## Cache and ABI policy

The versioned `gpt-oss-rs.xe-kernel-abi/v1` manifest enumerates every entry
point and argument index, name, type, byte size, address space, alignment,
mutability, and buffer extent, plus local/subgroup assumptions, K/tail rules,
aliasing, and exact cache identity. A cache key includes backend format,
source/SPIR-V/compiler hashes and versions, options, vendor/device, resolved
driver path/hash and API version, entry point, and ABI schema.

Corrupted SPIR-V and stale entry-point identity both failed safely in isolated
children. OpenCL returned normal creation/kernel errors. The current Level
Zero driver rejected stale identity normally but terminated the malformed
SPIR-V child with exit 10; therefore untrusted cache validation cannot be
treated as safely recoverable in-process.

Native modules are never checked in as portable artifacts. A future runtime
may cache them atomically after a successful SPIR-V load and validation, but a
hash, ABI, device, driver, compiler, option, or backend-format mismatch is a
hard miss followed by SPIR-V rebuild, never an attempted stale load.

## `ocloc` scope

The only preserved `ocloc` reports `23.43.027642`; it cannot be a canonical
26.05 compiler input. A controlled attempt to compile the same source was
recorded and failed before compilation because the old isolated FCL dependency
set is incomplete. The attempt log hash is
`a09b9d1fe6440eae2c67bcd686e7ec3ea04984d18432c13fe55c9d928f2875a4`;
the version-query output hash is
`0634a72b1bee6cdc266c97ac45bdf6f480d3b32ee8c7670237147039ab4fd41a`.

The old executable is useful only as an offline native-container decoder in
X5. It is never mapped into a hardware-process run, never contributes code to
the canonical SPIR-V, and never authorizes reuse of 23.43 binaries.

## Evidence records

| ID | Manifest SHA-256 | Raw comparison SHA-256 |
| --- | --- | --- |
| X3-OCL | `f5d5180ab4d907069491eeff4f8dd96b3dfabb0daf32e2313b71b03012ac8cd4` | `8729e4eb6e97ec423b30a38119435349550682510c329017e52ee3522cd14293` |
| X3-L0 | `37ced41fee78c00c7c3be9a2b6eebc953b15edc9fbd59ebce1a89f655f0700c7` | `a0299d7da5276075a303e7624efce22b59630c449f3b6ee141baf5ca163d7e93` |

Both records are `pass` and report repository revision
`4545cb24d923a2330d8ff68fa9ba7e3377e73359`.
