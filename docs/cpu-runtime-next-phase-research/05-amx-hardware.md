# C5: AMX Hardware Closure Runbook

- Outcome: **deferred**
- Scope: native AMX-INT8 validation and lifecycle runbook only
- Local hardware result: `C5-NEG-001: unsupported`
- Source budget used: current repository, official Linux XSTATE documentation,
  official Intel AMX documentation, oneDNN (NX-SRC-006), and the pinned Rust
  toolchain source

## Objective, questions, and non-questions

C5 refreshes the exact gates between a portable AMX prototype and native
eligibility, defines process permission and per-thread tile lifecycles, and
provides the future hardware evidence matrix. It asks what must be observed on
a real host before forced or automatic execution can be trusted.

It does not run AMX here, tune tiles, select AMX automatically, choose
persistent panels, investigate AMX-BF16, or treat compilation/emulation as
native validation. The local Tiger Lake/Xe platform is an unavailable test
host, not an AMX result.

## Current repository and host baseline

- **C5-E-001 / CURRENT-REPO FACT:** the `amx-int8` feature propagates through
  the server, engine, model runner, and CPU-kernel crates. CPU CI performs a
  locked feature check, kernel tests, and warnings-denied Clippy.
- **C5-E-002 / CURRENT-REPO FACT:**
  `gpt-oss-cpu-kernels/src/amx.rs::{AmxRuntimeStatus,
  initialize_amx_int8}` distinguishes build, Linux x86-64 target, raw CPUID
  AMX-TILE/AMX-INT8, kernel XSTATE query, and process TILE_DATA permission.
  A `OnceLock` retains the initialization result.
- **C5-E-003 / CURRENT-REPO FACT:** model construction explicitly initializes
  AMX only for forced AMX selection before model mapping and worker-pool use.
  Automatic selection neither requests permission nor chooses AMX.
- **C5-E-004 / CURRENT-REPO FACT:**
  `native/amx_int8.cpp::gpt_oss_amx_int8_tile` validates pointers/row count,
  loads a 64-byte palette-one configuration, zeroes C, loads A/B, executes
  signed `TDPBSSD`, stores C, and calls `_tile_release`. All status returns
  occur before tile configuration; the configured path has one normal exit.
- **C5-E-005 / CURRENT-REPO FACT:** `matmul.rs::amx_matmul_with_tile` and its
  tests cover portable A/B panel ordering, scalar tile emulation, primary and
  residual scaling, scratch/canaries, extrema, full tiles, and scalar M=1/N-tail
  fallback. These are portable semantics, not instruction execution.
- **C5-NEG-001 / READ-ONLY HOST PROBE:** `lscpu` on 2026-08-11 reports an
  i7-1185G7 with AVX-512/VNNI and no AMX flags. No AMX instruction or permission
  request was attempted. Status is `unsupported`, not fail.

## Source evidence cards

### C5-E-006 / PRIMARY-SOURCE FACT

- Question: how does Linux expose dynamic AMX state?
- Source: Linux kernel, “Using XSTATE features in user space applications,”
  accessed 2026-08-11
- Document/path: `docs.kernel.org/arch/x86/xstate.html`, sections 36.1-36.3
- Observation: TILE_DATA is XSTATE component 18; applications query support and
  permission separately and request the highest required component with
  `ARCH_REQ_XCOMP_PERM`. Permission is per process, inherited on fork, cleared
  on exec, and constrained by alternate signal-stack size. First instruction
  use traps so the kernel can allocate per-task extended state; lack of
  permission produces SIGILL and allocation failure can produce SIGSEGV.
- Implication: successful `arch_prctl` is necessary but not proof that every
  worker has executed first use or that signal/error behavior is safe.
- Limitation: the document does not certify this repository's shim or host.
- Confidence: high.

### C5-E-007 / PRIMARY-SOURCE FACT

- Question: what is the application tile lifecycle?
- Source: Intel, “Advanced Matrix Extensions Intrinsics Functions,” document ID
  766088, version current, updated 2024-02-01, accessed 2026-08-11
- Section: tile/TMUL architecture and INT8 walkthrough
- Observation: software loads a 64-byte tile configuration; palette one
  exposes eight tiles with at most 16 rows by 64 bytes; signed `TDPBSSD`
  accumulates four signed byte products into INT32; software owns configuration
  changes and release. The example requests Linux TILE_DATA permission and
  releases the configuration after storing results.
- Implication: configuration and release belong to each calling thread's
  scoped native call, while permission belongs to process initialization.
- Limitation: Intel labels the sample demonstrative rather than production
  code; it does not prove this panel layout or scheduling policy.
- Confidence: high for ISA/lifecycle, low for production transfer.

### C5-E-008 / LOCAL-SOURCE OBSERVATION

- Question: can a library assume one permanent tile configuration?
- Source: NX-SRC-006, oneDNN
- Pin/path: `7a640690...`; `src/cpu/x64/amx_tile_configure.cpp::{
  amx_tile_configure,amx_tile_lazy_configure,amx_tile_release}`
- Observation: oneDNN provides explicit configure/release functions. Its lazy
  path stores and compares the current configuration on each calling thread;
  it does not assume a process-global permanent palette.
- Implication: this repository's load/execute/store/release call is the safer
  initial ownership boundary. A future cached configuration must be thread
  local and coexistence-tested.
- Limitation/conflict: oneDNN's JIT and broader runtime are not adopted.
- Confidence: high.

### C5-E-009 / PRIMARY-SOURCE FACT

- Question: can stable Rust replace the native shim now?
- Source: Rust 1.97.1 toolchain source at the recorded baseline
- Pin/path: `library/stdarch/crates/core_arch/src/x86_64/amx.rs` and
  `library/std_detect/src/detect/arch/x86.rs`
- Observation: AMX target features and intrinsics remain unstable under
  `x86_amx_intrinsics`, issue 126622.
- Implication: the small feature-gated C++17 boundary remains justified for
  this baseline; compiler acceptance is not runtime evidence.
- Limitation: later toolchains may change this decision and require a new
  evidence tuple.
- Confidence: high.

## Capability and initialization runbook

The future operator records every gate separately in the E1 runtime snapshot:

1. **Build:** confirm the `amx-int8` feature and exact compiler/shim artifact.
   Absence is `unsupported`; no native symbol is callable.
2. **Target:** require Linux x86-64. Other targets reject forced AMX before
   model preparation.
3. **Hardware:** read raw CPUID AMX-TILE and AMX-INT8. Do not use the general
   `CpuFeatures` XCR0-combined fields as the sole hardware record.
4. **Kernel support:** query `ARCH_GET_XCOMP_SUPP` and require XTILECFG (17)
   plus XTILEDATA (18). A syscall failure is distinct from an absent bit.
5. **Existing permission:** query `ARCH_GET_XCOMP_PERM`. If TILE_DATA is absent,
   request component 18 with `ARCH_REQ_XCOMP_PERM`, retain the exact errno in
   diagnostics, and query again.
6. **Signal-stack precondition:** record process/thread alternate-stack policy
   before permission. Permission rejection due to an inadequate installed
   stack is not a hardware absence; later `sigaltstack` failure after permission
   is a service-initialization failure.
7. **Process ordering:** resolve permission before readiness and before the
   project-owned CPU worker pool begins accepting work. Record fork/exec and
   worker creation ordering for any launcher; never infer post-exec permission.
8. **Per-thread first use:** on every worker eligible for AMX, execute a bounded
   known-answer tile probe under controlled error/signal observation before
   declaring the worker ready. This separately exercises kernel XSTATE
   allocation for each task.
9. **Per-call scope:** validate pointers, extents, alignment, and scratch before
   `_tile_loadconfig`; load the known palette, execute, store, and release on
   every normal return. No Rust unwind crosses the ABI. Fatal signals remain
   process failures and must be reported as such rather than converted to a
   passing fallback.
10. **Eligibility:** publish AMX only after every worker probe and the native
    matrix assertions below pass for the exact host/kernel/build tuple. Forced
    AMX rejects any missing gate. Automatic AMX remains disabled until a later
    separately authorized performance study and C3 fallback coverage.

Permission mutation is startup state. A request must not trigger it. Tile
contents/configuration are calling-thread state. They must not be cached as a
process fact, and cancellation cannot abandon a call between load and release.

## Future hardware evidence matrix

Every row requires an E1 manifest, effective runtime snapshot, raw output,
repetitions where applicable, and artifact hashes. A virtual machine is named
as such and cannot substitute for bare-metal lifecycle coverage.

| Axis | Required cases | Acceptance evidence |
| --- | --- | --- |
| Build/target | feature absent; enabled Linux x86-64; unsupported target compile | Absent/unsupported rejects before instruction; enabled artifact and flags are hashed |
| CPUID/XSTATE | TILE absent; INT8 absent; kernel components absent; all present | Each gate yields its stable reason; injected cases remain advisory beside one real all-present host |
| Permission | already granted; request succeeds; request denied; query fails; exec boundary | Status distinguishes each; post-exec process re-requests before readiness |
| Signal stacks | default stack; adequate altstack before request; intentionally inadequate isolated child | Adequate cases run; inadequate case rejects safely with retained errno and no AMX execution |
| Worker lifecycle | one worker; every pool worker; pool recreation; repeated calls; concurrent calls | Each task completes known-answer first use; no SIGILL/SIGSEGV; stable results and release behavior |
| Tile coexistence | repository palette repeated; intervening different valid palette; non-AMX work between calls | Repository reloads its palette, outputs match, and released/foreign state is not assumed |
| Native equality | Q8 and residual-Q8; M 2/15/16/17; N 16/full multiples/tails; multiple K blocks; extrema/bias | Native equals the portable emulator's INT32 tiles and accepted scalar FP32 output assertions for every effective main/fallback path |
| Failure/cancellation | invalid bounds before load; injected pre-call failure; owner cancellation while queued/running; process signal in isolated child | No partial model commit; validation never executes AMX; terminal outcome and cleanup follow C1/C2; signal result is retained, never waived |
| Resource behavior | cold first use and warm repeats on every worker | Process/thread memory deltas, first-use faults, scratch, and timing inclusions are reported without a performance claim |
| Dispatch | forced full tile; forced M=1; forced N tail; automatic | Forced records native plus covered scalar tails and never silently changes requested mode; automatic does not select AMX in current policy |

Native equality uses the exact repository feature build and a fixed test-only
problem set, not a fresh 20B run. Hardware performance and crossover work is a
separate future experiment requiring owner authorization.

## Alternatives and decision

| Alternative | Finding |
| --- | --- |
| Treat portable emulation and feature CI as closure | Rejected. Neither executes kernel permission, first-use XSTATE allocation, tiles, context switching, or release. |
| Configure a worker once and retain tile state | Deferred/rejected for initial closure. Foreign code and signals may observe/change state; current scoped reload/release is auditable. |
| Keep scoped native calls and require worker/hardware certification | Retained. It is compatible with current ABI and makes lifecycle failures visible. |

**C5-D-001 / PROVISIONAL DECISION:** the current feature remains
forced/experimental and outside trusted/automatic eligibility. No persistent
panel, AMX-BF16 path, tile caching, or crossover is considered by this runbook.

## Failure modes and focused tests

- CPUID is conflated with XSTATE/permission: assert each status field and
  stable failure reason independently.
- `OnceLock` retains an environment-specific denied result: initialization is
  process-immutable in the current design; tests requiring alternative states
  use isolated child processes, not in-process mutation.
- Permission exists in a parent but disappears on exec: launch a fresh binary
  and verify it reinitializes before workers/readiness.
- Only the initiating thread receives a first-use allocation: execute the
  known-answer tile on every stable worker identity and under concurrency.
- Inadequate altstack turns initialization or later signal delivery unsafe:
  isolate the negative test and retain errno/signal status.
- Foreign tile configuration invalidates a cached assumption: interleave a
  second valid palette and require the repository call to reload and release.
- Forced full tiles pass while scalar M=1/N tails are uncovered: record both
  effective backends and require C3 evidence for the fallback cells.
- A native/emulator mismatch is hidden by FP32 scaling: compare raw INT32 C
  tile first, then block-scaled output and final bias separately.

## Risks, conclusion, and gate

Hardware availability is the blocking risk, followed by false confidence from
portable evidence and incomplete signal/worker coverage. Native validation
also changes the effective C3 tuple: host ISA, kernel, permission, compiler,
thread pool, and fallback shapes must all be pinned.

C5 is **deferred** because this host cannot execute AMX. It becomes
planning-ready only after the required native matrix completes on an
AMX-INT8-capable host with recoverable evidence. A hardware pass would close
forced-path feasibility only; automatic selection and performance tuning would
still require a separate owner-authorized study.
