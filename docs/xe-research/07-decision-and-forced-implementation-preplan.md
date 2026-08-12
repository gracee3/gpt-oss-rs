# X7 — Decision and Forced Implementation Pre-Plan

- Terminal result: negative closeout
- Production Xe backend: do not implement
- Automatic selection, serving, full-model inference: not authorized
- CPU behavior: unchanged

## Decision

The one-sweep lane closes because no Xe path meets the predeclared useful-win
gate. Correctness, exact MXFP4, native DP4A lowering, allocation classes,
checkpoint ingestion, and compact-memory policy all passed. Performance did
not: all plausible M=4–64 OpenCL confidence intervals are below AVX2 parity,
and both Level Zero submission modes are below parity at every shape. The lane
is closed without reducing the 1.25x floor, changing plausible shapes, ignoring
request-path work, retaining duplicate model weights, or weakening numerical
requirements.

## Independent terminal decisions

| Axis | Decision |
| --- | --- |
| Host API | none selected for implementation |
| Kernel delivery | reproducible validated SPIR-V is the only preferred portable artifact if the lane is reopened |
| Native cache | optional atomic runtime cache only; never checked in or considered portable |
| Memory/residency | one compact persistent selected region or fixed bounded slab, plus reusable activation/output scratch |
| Submission | none implemented; counterfactual OpenCL uses one serialized in-order queue and completion event |
| Integration | no backend; archival boundary is one forced experimental model attachment below commit |
| CPU fallback | discard all failed/unvalidated Xe output and recompute only before model-state commit |

Neither API was 10% faster at a plausible shape with non-overlapping intervals,
so the speed rule does not choose an API. The counterfactual forced-only
tie-breaker is OpenCL: the research boundary resolves 33 OpenCL symbols versus
43 Level Zero symbols, owns fewer queue/list/event objects, supports the same
preferred SPIR-V and native bytes, and rejected malformed modules without the
Level Zero invalid-SPIR-V child termination. This is an archival choice for a
future re-opened goal, not implementation authorization.

## Exact prospective Rust boundary

The following names and responsibilities are decision-complete but are not
implemented:

```rust
enum ForcedXeApi { OpenCl }

struct ForcedXeConfig {
    api: ForcedXeApi,
    vendor_id: u16,          // exactly 0x8086
    device_id: u16,          // exactly 0x9a49
    module_dir: PathBuf,     // validated SPIR-V plus writable native-cache dir
    max_resident_bytes: u64, // hard bounded slab/compact-region budget
    queue_count: NonZeroUsize,
}

struct XeApi;                // exact loader, driver/device identity, capabilities
struct XeContext;            // one context per forced model attachment
struct XeModule;             // ABI-validated SPIR-V plus optional native cache
struct XeModelResources;     // one compact region/slab, reusable scratch pool
struct XeQueueLease<'a>;     // exclusive serialized in-order queue lease
struct XeCompletion<'a>;     // event and all in-flight borrows
struct XeExecution<'a>;      // prepared inputs, output, commit eligibility
```

`ForcedXeConfig` would be constructed only by an explicit experimental CLI
flag that names `opencl` and `8086:9a49`; `auto`, default dispatch, public
runtime selection APIs, and silent fallback are absent. Configuration is
rejected unless the resolved loader/driver hashes pass the mixed-generation
guard and every ABI capability is queried.

The prospective FFI lives in a private `backend::xe::ffi::opencl` module and
contains only the 33 audited symbols represented by the research shim. Raw
handles never escape. `XeContext`, `XeModule`, buffers, queue, and events are
non-`Clone`; buffers borrow their context, queue leases are exclusive, and
completion owns every resource needed through terminal wait.

## Cache and ABI

```rust
struct XeCacheKey {
    schema: &'static str, // gpt-oss-rs.xe-kernel-abi/v1
    source_sha256: [u8; 32],
    spirv_sha256: [u8; 32],
    compiler_sha256: [u8; 32],
    build_options_sha256: [u8; 32],
    vendor_id: u16,
    device_id: u16,
    driver_sha256: [u8; 32],
    driver_api_version: u32,
    entry_point: String,
    backend_format: XeBinaryFormat,
}
```

Every kernel argument is checked against the committed v1 ABI before FFI.
Extent multiplication is checked, K must be divisible by 32, buffers cannot
alias contrary to the manifest, local size must satisfy the queried limit, and
the subgroup requirement must match. SPIR-V is validated before driver load.
A native cache is written to a temporary file, hashed, reloaded and validated,
then atomically published. Any identity mismatch, corruption, stale entry
point, or driver rejection is a cache miss; it never relaxes the ABI.

## Ownership, shutdown, errors, and commit

`XeModelResources::attach` is transactional: validate identity/ABI/tensor,
allocate within the hard budget, stage compact bytes, validate visibility, and
publish only after success. It owns a fixed scratch/queue pool; there is no
allocator, LRU, model-wide expanded representation, or background migration.

`shutdown()` is explicit and idempotent: reject new leases, drain terminal
completions, invalidate the attachment on context/device error, release
execution scratch, model allocations, events, queue, kernel/program, context,
and loader in reverse order. `Drop` invokes the same path but cannot promise
successful driver recovery. In-flight resources are never freed on mere host
cancellation.

The error enum remains diagnostic rather than collapsing to fallback:

```rust
enum XeError {
    UnavailableDevice,
    MixedGenerationLibrary,
    UnsupportedCapability,
    InvalidArtifact,
    InvalidAbi,
    InvalidShape,
    AllocationFailed,
    BuildFailed { log: String },
    LaunchFailed,
    Timeout,
    ContextInvalidated,
    DeviceLost,
    VisibilityMismatch,
    NumericalMismatch,
    ShutdownFailed,
}
```

Fallback is controlled by an explicit pre-commit guard. CPU-owned inputs are
prepared first. Xe output remains private until completion and numerical/status
validation. On any Xe error before commit, the bytes are discarded and the
operation is recomputed through `ResidualQ8 + Avx2 + InterleavedSplitX8V2`.
Exactly one validated result then commits model-visible state. After commit,
the operation is never replayed and fallback cannot hide a partial external
effect.

## Rejected alternatives

- Production or automatic Xe dispatch: useful-win gate failed.
- Level Zero: no 10% plausible-shape advantage, larger audited lifecycle
  surface, and unsafe malformed-module behavior in the current driver.
- OpenCL online source as primary delivery: reproducible same-SPIR-V passed
  both APIs and produced identified DP4A lowering.
- Checked-in native binaries: driver/compiler/device scoped, not portable.
- Immediate Level Zero lists: no material benefit over recycled regular lists.
- Explicit subgroup DP4A kernels: all slower than the compiler-lowered scalar
  source at M=4.
- Model-scale derived weights, whole-model residency, general allocator/LRU,
  direct mmap zero-copy, and overlapping CPU/GPU execution: unproven or failed
  memory/performance gates.
- Full-model inference, decode, KV migration, serving, and heterogeneous
  scheduling: outside the slice and unjustified after the stop condition.

## Evidence needed to reopen

A new goal must identify a material change: a newer validated driver/compiler,
a different kernel organization that retains compact weights, or a different
target device. It must rerun X0 mixed-generation provenance, X3 cache/ABI,
X4 contention/residency, X5 exact/random/codegen evidence, and the complete X6
protocol. Reopening still requires 1.25x at a predeclared plausible shape with
the 95% interval above parity, zero correctness regressions, and no model-scale
duplicate representation. Automatic selection additionally requires
full-model inference, sustained mixed display load, cancellation/device-loss
recovery, multi-request behavior, and serving/API evidence that this sprint
does not provide.

## Closeout record

The X7 `gpt-oss-rs.xe-research/v1` manifest has status `fail`, terminal result
`negative_closeout`, repository revision
`21ad5be042a1be517f8def522c47f467c6f197c7`, and SHA-256
`1378bd9ab319254d19ae95c91fc601e888ec56b34c301f2ffc2dbfe564a81430`.
Here, `fail` is the performance gate result; the record explicitly preserves
passing correctness and memory results.
