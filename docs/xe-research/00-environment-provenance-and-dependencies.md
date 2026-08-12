# X0 — Environment, Provenance, and Dependencies

- Scope: one Lenovo T14, Tiger Lake-LP Iris Xe `8086:9a49`
- Evidence schema: `gpt-oss-rs.xe-research/v1`
- Raw evidence root: `/home/emmy/src/xe-research/results/20260811-xe-one-sweep/`
- Result: pass for a forced-only research harness; no production dependency or backend

## Captured stack

The X0 commands select either `/etc/OpenCL/vendors/intel.icd` or the system
Level Zero loader and then require exactly Intel `8086:9a49`. They capture the
kernel, `i915`, firmware packages, PCI identity, render nodes, ACLs and groups,
ICDs, package versions and candidates, headers, loader cache, compiler tools,
full `clinfo` or the locally rebuilt minimal `ze_info`, and `/proc/<pid>/maps`
while a selected session is live.

The tested current-generation boundary is:

| Component | Version or identity |
| --- | --- |
| Kernel | `7.0.0-29-generic`, `i915` |
| Device | Tiger Lake-LP GT2 Iris Xe, PCI `8086:9a49`, renderD128 |
| Compute Runtime | packages `26.05.37020.3-1`; OpenCL driver `26.05.037020` |
| Level Zero loader | package and ABI `1.28.2` |
| Level Zero driver API | 1.14, reported raw as `65550` |
| IGC | package `libigc1 1.0.17791.18+1-3`; mapped `libigc.so.2.28.4+0` |
| SPIR-V translator | cached LLVM 18 translator, offline |

Installed legacy compatibility packages are inventory, not selected inputs.
Every live session resolves its loader with `dladdr`, enumerates mapped files,
resolves package ownership, and hashes the actual files. The guard rejects a
mapped path containing `23.43`, `compute-runtime-23.43`,
`level-zero-v1.16.1`, or the preserved toolchain sysroot library directory. It
also fails unless the expected current Intel backend library is mapped. This
prevents an old `LD_LIBRARY_PATH` from silently pairing a current loader with a
23.43 driver.

## Source correspondence

The source corpus is evidence, not a build dependency. The exact clean pins
used for source inspection are:

| Corpus | Revision |
| --- | --- |
| Compute Runtime 26.05.37020.3 | `a5e0dd79db5ff7b3ed6c5cd3d11064ab7cbb9aa5` |
| Level Zero v1.28.2 | `6369d8d642e9c7625e67f38664267f171b8e42dc` |
| llama.cpp | `0b1bad14ff204627636aeb1de22ddcd5acb859d4` |
| Level Zero tests | `d373228d721184255597790310c3d13e8216a43d` |
| PTI GPU | `c71e8316e19bb5316157b9046d877b5eff0e262c` |

Ubuntu source archive hashes agree with signed APT indexes. The locally
extracted `.dsc` maintainer signature was not authenticated with an installed
maintainer keyring. That caveat limits claims about authenticated maintainer
authorship; it does not alter the observed package bytes, API behavior, or
black-box results. Driver-internal claims require exact source-to-binary proof
that this sprint does not claim.

## Raw FFI and Rust-reference comparison

The research tool path-loads only the required OpenCL and Level Zero entry
points in `native/xe_probe.c`. Rust owns a deliberately narrow `Session` and
borrowed `Buffer` boundary in `src/runtime.rs`; all `unsafe` calls are confined
there and in the C shim. The declarations were checked against the cached
Khronos OpenCL headers at revision
`c9c8ccfab584f9f7610057c4633dbd3df7e012cc` and the exact Level Zero 1.28.2
headers.

For an independent ownership comparison, the local references are `cl3`
`9e2cdd8f34f09abfe49a8c2718ac58f1f762ae61`, `opencl3`
`072410552fecfc1e3f5395856735cb8684501f74`, and `oneapi-rs`
`1581663fdd0fd73e79df2900a2576d6cca8ff2a1`. The harness follows their basic
RAII shape without importing their broad surfaces. Both ordinary loader
resolution and runtime symbol loading are represented: system tools exercise
normal dynamic linking, while the shim exercises exact `dlopen`/`dlsym` and
`dladdr` provenance.

## Dependency budget

| Budget | Allowed in this sprint | Production change |
| --- | --- | --- |
| System runtime | installed OpenCL ICD, Compute Runtime, Level Zero loader/driver, IGC | none |
| Contributor tools | Clang 18, LLVM SPIR-V translator, SPIRV-Tools, C compiler, cached headers | none |
| Research Cargo | `anyhow`, `half`, `libc`, `memmap2`, `rand`, `rand_chacha`, `safetensors`, `serde`, `serde_json`, `sha2`, `cc`, path CPU kernels | isolated lockfile only |
| Production Cargo | existing graph | none |
| Checked-in artifacts | source kernels, script, ABI manifest, Rust/C harness | no generated binary |

The standalone workspace rebuild command is:

```console
RUSTC_WRAPPER= CARGO_NET_OFFLINE=true cargo build \
  --manifest-path tools/xe-research/Cargo.toml --locked --offline --release
```

Kernel and harness sources are original project work under the repository's
Apache-2.0 license. Generated SPIR-V, native modules, logs, and benchmark data
remain outside Git and carry source/tool hashes in their manifests. No
third-party implementation source is copied into the repository.

No rollback was needed because no package or system-policy change was made.
If restoration is separately authorized, the captured `dpkg-query` and
`apt-cache policy` records are the package inventory to resolve first; this
document does not prescribe an unaudited downgrade command.
