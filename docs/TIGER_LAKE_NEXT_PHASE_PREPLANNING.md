# Tiger Lake SoC Next-Phase Optimization — Pre-Planning

> Historical planning input. The repository and host facts below describe the
> pre-upgrade `6caf274` capture. Implementation began from this document only
> after the oracle reconciliation recorded in
> [`CPU_ORACLE_RECONCILIATION_LEDGER.md`](CPU_ORACLE_RECONCILIATION_LEDGER.md).
> Current decisions and evidence are tracked by
> [`TIGER_LAKE_IMPLEMENTATION_PLAN.md`](TIGER_LAKE_IMPLEMENTATION_PLAN.md).

- Status: pre-planning and research charter only
- Recorded: 2026-08-12
- Host: Dell Latitude 5320, Intel Core i7-1185G7, 32 GiB RAM
- Repository branch: `main`
- Repository revision: `6caf27423744148dafb1fb2670a03f29452311f3`
- `origin/main`: `6caf27423744148dafb1fb2670a03f29452311f3`
- Raw host/runtime evidence:
  `/home/emmy/gpt-oss-rs-artifacts/tiger-lake-preplanning/6caf27423744148dafb1fb2670a03f29452311f3/`
- Raw-file checksum-index SHA-256:
  `ee307fd20fdac5b17302a6b7bdbe98392a8ee3b9140d37105fddbddff052f5f2`
- Implementation decision: none; no kernel, dispatch, runtime, or system-policy
  change is authorized by this document

## Repository reconciliation

The checkout began clean on `agent/cpu-validation-fresh-oracle` at
`72ed57f1ce3fd0bc1b3ee1b48748fd8e53475149`. Local `main` was 21 commits
behind the fetched `origin/main`. Because the working tree was clean and the
local branch was a direct ancestor, `main` was fast-forwarded to
`6caf27423744148dafb1fb2670a03f29452311f3`. At the pre-document checkpoint,
`HEAD` and `origin/main` matched and the working tree was clean.

The native CPU and Xe production histories are unified in this `main`:

| History checkpoint | Revision | Ancestor of current `main` |
| --- | --- | --- |
| Native GPT-OSS CPU serving and parity baseline | `68a73e2` | yes |
| Promoted CPU-first AVX2 x8 runtime | `3600fa4` | yes |
| Layer-major CPU prefill closeout | `898e016` | yes |
| Final CPU feature-set verification | `edee07e` | yes |
| CPU next-phase research closeout | `33379d6` | yes |
| Standalone Xe research harness | `a9de8f8` | yes |
| Xe optimization evidence seal | `3aefcf9` | yes |
| Runtime-loaded Xe foundation | `14b2c38` | yes |
| Bounded CPU/Xe serving fallback | `c9b9f14` | yes |
| Disabled automatic Xe promotion gate | `8d3c634` | yes |
| Explicit Xe integration closeout | `6caf274` | yes |

One lineage is not unified: fresh immutable CPU oracle candidate `af6c0a2`
and closure `72ed57f` are not ancestors of `main`. Current `main` contains the
earlier official CPU oracle tool introduced with `68a73e2`, but not the later
container lock, fresh campaign harness, or closure report. Planning must not
describe the later fresh campaign as landed on `main` until it is actually
integrated. This pass does not merge it.

The Xe evidence documents name a Lenovo T14. This new host is a Dell Latitude
5320 with the same Tiger Lake CPU class and PCI GPU identity. Prior T14
measurements remain useful research inputs, but they are not silently promoted
to Dell host measurements. New tuning evidence needs this host's own identity.

## A. Objective

Treat Tiger Lake as a heterogeneous SoC containing multiple useful execution
resources rather than treating CPU and Iris Xe as mutually exclusive
whole-model backends. Preserve capability-safe generic behavior while using
evidence-backed hardware profiles to select and compose the fastest proven
implementation by phase, operation, shape, and runtime state.

The eventual conceptual decision is:

```text
hardware capabilities
+ validated hardware/runtime profile
+ phase
+ operation
+ M/N/K or context shape
+ preparation/residency state
    ->
effective implementation
```

Hardware identity and legal ISA capability are separate inputs. Production
dispatch must not string-match `i7-1185G7`; a hardware profile may key measured
evidence to that identity while capability checks continue to decide which
instructions are legal.

## B. Scope boundary

Aggressive exploration may include:

- scalar reference paths;
- AVX2;
- AVX-512/VNNI;
- wider CPU matrix microkernels;
- CPU operator fusion;
- OpenCL;
- Level Zero;
- reproducible SPIR-V;
- shared or unified memory;
- persistent bounded Xe residency;
- fused Xe expert execution;
- CPU/Xe cooperative execution;
- optimized CPU attention;
- possible later Xe attention;
- fused LM-head/argmax;
- topology and affinity experiments;
- model-shape-specialized kernels; and
- hardware-profile-based autotuning.

The following remain out of scope for this phase:

- SYCL;
- oneAPI-rs as a dependency;
- Vulkan;
- general Intel GPU support;
- a generalized heterogeneous framework;
- broad model-family support;
- large external math or runtime dependencies merely for convenience;
- system-wide governor, power, or daemon changes; and
- performance claims without named, retained evidence.

The project remains narrow: fast, understandable GPT-OSS execution on the
hardware actually available, with generic validated fallbacks retained.

## Host fingerprint

### Machine, CPU, memory, and topology

| Field | Observed value |
| --- | --- |
| System | Dell Latitude 5320, SKU `0A42`, board `0KRH0R` |
| BIOS | Dell `1.38.0`, 2024-06-05 |
| OS | Ubuntu 24.04.4 LTS |
| Kernel | `7.0.0-28-generic`, x86-64, `PREEMPT_DYNAMIC` |
| CPU | 11th Gen Intel Core i7-1185G7 at 3.00 GHz |
| Identity | GenuineIntel family 6, model 140 (`0x8c`), stepping 1 |
| Microcode | `0xbe` |
| Topology | 1 socket, 4 physical cores, 8 logical CPUs, SMT2, one NUMA node |
| L1 data | 48 KiB per core, shared by each SMT pair, 64-byte lines |
| L1 instruction | 32 KiB per core, shared by each SMT pair, 64-byte lines |
| L2 | 1,280 KiB per core, shared by each SMT pair, 5 MiB total |
| L3 | 12 MiB shared by CPUs 0-7 |
| RAM | 31 GiB visible (`numactl`: 31,820 MiB), one NUMA node |
| Swap at capture | 16 GiB configured, about 1.9 GiB used system-wide |

CPU 0 and CPU 4 share the private cache hierarchy, confirming the SMT sibling
numbering visible in sysfs. Full `lscpu -e` and `/proc/cpuinfo` are retained in
the raw evidence. `dmidecode -t memory` was attempted without sudo and could
not read SMBIOS or `/dev/mem`; no privilege escalation was used.

The frequency driver is `intel_pstate`. Every visible policy reported governor
`powersave`, energy-performance preference `performance`, and a 4.8 GHz
maximum. These values are captured identity/runtime state, not a request to
change policy. Readable thermal zones were also captured as a point-in-time
observation and must not be treated as an idle or benchmark baseline.

`cpuid` is not installed. It was not installed for this pass.

### CPU capabilities and OS state

`crates/gpt-oss-cpu-kernels/src/features.rs` performs CPUID discovery and
combines it with `OSXSAVE`/XCR0 checks. AVX requires XMM/YMM state in XCR0;
AVX-512 additionally requires opmask and ZMM state. This is capability-safe
and independent of the CPU marketing name.

The host exposes:

| Capability | Result |
| --- | --- |
| AVX2 | yes |
| FMA | yes |
| AVX-VNNI (non-AVX-512) | not exposed |
| AVX-512F | yes |
| AVX-512DQ | yes |
| AVX-512IFMA | yes |
| AVX-512CD | yes |
| AVX-512BW | yes |
| AVX-512VL | yes |
| AVX-512VBMI/VBMI2 | yes |
| AVX-512VNNI | yes |
| AVX-512BITALG/VPOPCNTDQ | yes |
| AVX-512BF16 | not exposed |
| AVX-512FP16 | not exposed |
| AMX | not exposed |
| AVX10 | not exposed |

Linux reports the AVX and AVX-512 flags to this process, and a focused run with
`GPT_OSS_CPU_KERNEL=avx512-vnni` successfully constructed the repository's
forced path. This verifies the detector's full AVX2 + AVX-512F/BW/VL/VNNI
legality requirement, including OS-enabled extended state.

The repository has no inexpensive command that prints the complete
`CpuFeatures::detect()` structure. Full-model startup logs print the resolved
dispatch plan, and tests verify detection indirectly. A model-free
`cpu-features` diagnostic that prints CPUID facts, XCR0 state, legal
capabilities, and the resolved per-operation plan is a useful observability
candidate; it is not added in this pass.

### Iris Xe and installed runtimes

| Field | Observed value |
| --- | --- |
| PCI device | Intel TigerLake-LP GT2 Iris Xe `8086:9a49`, revision 1 |
| Subsystem | Dell `1028:0a42` |
| Kernel driver | `i915`; kernel also lists the `xe` module, but it is not in use |
| DRM nodes | `/dev/dri/card1`, `/dev/dri/renderD128` |
| Access | user ACL grants read/write access to both nodes |
| GT frequency at capture | 400 MHz current, 100 MHz minimum, 1,350 MHz maximum |

`clinfo` successfully selected Intel Iris Xe through the Intel ICD and
reported:

- OpenCL 3.0 NEO, driver `23.43.027642`;
- 96 compute units and 1,350 MHz maximum clock;
- subgroup sizes 8, 16, and 32;
- `cl_khr_integer_dot_product` plus Intel subgroup extensions;
- SPIR-V 1.2 IL support;
- 64 KiB local memory;
- host/device unified memory;
- core coarse-grained SVM; and
- 28.77 GiB reported global memory, which is shared system memory rather than
  discrete VRAM.

The installed stack is:

| Component | Current host |
| --- | --- |
| Intel OpenCL ICD | `23.43.27642.40-1ubuntu3` |
| Intel Level Zero GPU driver | `23.43.27642.40-1ubuntu3` |
| IGC | `1.0.15468.25-2ubuntu0.1` |
| Level Zero loader | `1.16.1-1build1` |
| OpenCL ICD loader | `2.3.2-1build1` |

This does not match the landed X8/X9 evidence or promotion record, which pins
OpenCL driver `26.05.037020`, newer IGC, and Level Zero loader `1.28.2`.
Current library hashes are retained in `raw/runtime-library-hashes.txt`.

The production crate's opt-in explicit OpenCL attachment smoke test passed on
this current stack. It exercised capability validation, kernel compilation,
the numerical startup self-test, native-cache reopen/corruption recovery, and
shutdown. By code policy this changed stack is `unvalidated_explicit`; it is
not an automatic-promotion match.

No Level Zero information utility is installed. The research harness's
`--help` build was attempted as requested but stopped because its exact cached
header corpus is absent at `/home/emmy/src/xe-research/OpenCL-Headers`; system
OpenCL and Level Zero development headers are also absent. No package or
source was installed. The Level Zero loader and Intel driver libraries are
present, but current device enumeration, queue groups, shared allocation, and
immediate-command-list execution are therefore **not established by this
pass**. Prior same-SPIR-V Level Zero results belong to the earlier 26.05/1.28.2
T14 evidence, not this current Dell/23.43 stack.

## Current merged execution architecture

### CPU findings

| Expected assumption | Current result |
| --- | --- |
| Capability detection is CPUID/XCR0 based | confirmed |
| `Auto` is per-operation rather than one uniform ISA | confirmed |
| BF16 matvec, quantization, and RMSNorm can favor AVX-512 | confirmed; they resolve to AVX-512/VNNI on this host |
| MXFP4 GEMV can favor AVX2 x8 despite AVX-512 | confirmed; automatic MXFP4/Q8 dot and GEMV use AVX2/x8 |
| True layer-major multi-row prefill exists | confirmed in transactional scheduled batch execution |
| Explicit AVX2 MXFP4 matrix execution exists | confirmed; 4-row by 8-output panel path with tails and caller scratch |
| `Auto(M>1)` remains scalar for MXFP4 matrix work | confirmed in `Mxfp4MatmulBackend::resolve` |
| Dense BF16 multi-row remains repeated matvec | confirmed in `project_bf16_batch` |

`KernelPath::Auto` resolves its compatibility path to AVX-512/VNNI here, but
the immutable operation plan is hybrid:

```text
bf16_matvec       = avx512-vnni
quantize_q8       = avx512-vnni
mxfp4_q8_dot      = avx2
mxfp4_gemv        = avx2-x8
mxfp4_layout      = InterleavedSplitX8V2
rms_norm          = avx512-vnni
```

Forced modes remain available for correctness and benchmarking. Explicit
AVX-512 selects the genuine AVX-512/VNNI x8 MXFP4 kernel, but automatic policy
retains AVX2 x8 for that operation based on existing host evidence.

The production batch path advances a row collection through each layer,
stages sequence-local KV transactionally, performs stable route/group/unroute,
and computes logits only for requested rows. The compatibility
`CpuModelRunner::prefill` wrapper still loops tokens, but this does not negate
the layer-major scheduled serving path.

Multi-row dense Q/K/V, attention output, router, and LM-head calls are only a
model-level batching surface. They parallelize input rows and call the same
dispatched BF16 matvec for every weight row. There is no true dense BF16
matrix microkernel, prepared dense layout, or dense scratch contract.

### Xe findings

| Expected assumption | Current result |
| --- | --- |
| Production Xe is an internal runtime-loaded OpenCL backend | confirmed |
| Explicit `--device xe` is accepted | confirmed, when built with the default Xe feature |
| Automatic Xe promotion is disabled | confirmed by immutable promotion record |
| CPU owns model/KV/routing/sampling/commit state | confirmed |
| Xe accelerates only bounded expert projections | confirmed |
| Decode remains on CPU | confirmed; Xe is allowed only for an all-prefill batch |
| M>=4 prefill uses tile32 M4 | confirmed for both gate/up and down |
| Fallback/circuit-breaker behavior exists | confirmed |
| A projection-runtime seam exists | confirmed as private `ProjectionRuntime` |
| Research Level Zero and same-SPIR-V paths exist | confirmed in the standalone research harness and retained records |

Production Xe receives only residual-Q8 expert projection work. The CPU still
owns tensor mapping, attention, KV state, routing, SwiGLU, expert weighting,
sampling, and transactional commit. M=1-3 and every decode projection remain
on CPU. For M>=4 all-prefill expert buckets, the backend uses
`mxfp4_tile32_m4_v2` with workgroup 32.

The runtime owns one serialized in-order OpenCL queue and fixed bounded
buffers. The fixed slab is residency of capacity, not resident expert content:
the selected expert is repacked on the CPU, then its weights and bias are
written for each projection. Activations are written, the kernel is submitted
and waited synchronously, and results are read back. This makes Xe residency
and fused expert execution particularly relevant candidates.

On a runtime failure, partial output is discarded, the queue is drained, CPU
recomputes the projection once, and a process-wide Xe circuit breaker opens
until restart. This fits the transactional CPU commit boundary.

`ProjectionRuntime` is a useful narrow seam: descriptor, synchronous project,
drain, and shutdown. It is currently private and attachment directly
constructs `OpenClRuntime`; adding a forced Level Zero candidate would require
an explicit runtime choice and another implementation behind the trait. It
does not require a second model runtime.

### Architectural tensions to resolve before optimization

1. The current OpenCL/Level Zero userspace stack is older than the stack that
   produced X8/X9. Re-establish a named, coherent stack and restore the exact
   research header/tool corpus before new Xe comparisons.
2. The fresh immutable CPU oracle closure is not on `main`, while trusted CPU
   policy still says it is blocked pending the final i7 conformance gate.
   Decide how that evidence is integrated before changing trust claims.
3. The coarse `gpt-oss-runtime-plan` crate knows CPU/CUDA/mock, while server
   policy separately refines CPU versus CPU+Xe. This is acceptable at the
   present boundary, but future operation planning must not further overload
   either request-level planner.
4. CPU and Xe have no direct ownership conflict. Their main unresolved shared
   concern is one thread/memory/package budget, especially when both execute
   concurrently on an integrated-memory SoC.

## Candidate optimization tracks

These are candidate investigations, not implementation commitments.

### Track T1 — Tiger Lake profiling corpus

Instrument representative requests by:

```text
phase
layer/operation class
M/N/K
actual MoE expert bucket M
attention context length
requested backend
effective backend
thread policy
preparation/cache state
timing
scratch/resident memory
fallback
```

The current C4 records deferred MoE orchestration, dense BF16, and attention
decisions because this owner corpus did not exist. This is the first
implementation prerequisite. Production metrics must remain bounded-cardinality;
exact shapes, expert IDs, and detailed timing belong in opt-in traces and
offline manifests.

The corpus also needs host profile identity, CPU affinity, active SMT policy,
CPU/GPU frequency snapshots, runtime/driver identity, repetitions, timer
inclusions, preparation hits/misses, and memory high-water. Crossover
thresholds cannot be responsibly chosen from isolated kernels alone.

### Track T2 — MXFP4 CPU matrix expansion

The current policy is:

```text
M=1: established optimized GEMV
M>1 Auto: scalar matrix reference
```

Evaluate the existing explicit AVX2 4x8 matrix path, a wider
AVX-512/VNNI matrix candidate, all M/N tails, gate/up and down separately,
residual-Q8 preparation cost, and the real routed expert-bucket distribution.
Do not infer M>1 behavior from the M=1 result where AVX2 currently beats the
forced AVX-512 expert kernel.

### Track T3 — True dense BF16 matrix path

Investigate a sibling `Bf16MatrixProblem` for multi-row Q, K, V, attention
output, router, and LM-head projections. Preserve explicit BF16 input/output
boundaries, FP32 accumulation order, bias placement, preparation identity,
caller-owned output/scratch, and forced/reference modes.

Candidate implementations may include blocked AVX2 and AVX-512 paths. Decode
M=1 and prefill M>1 remain separately tunable. The existing
`project_bf16_batch` stays the correctness baseline until a candidate has
shape-complete evidence.

### Track T4 — CPU fusion

Evidence-driven candidates include residual + RMSNorm, bias + output boundary
where semantics allow it, quantization + panel packing, expert weighting +
final accumulation, final norm + LM-head preparation, and bounded Q/K/RoPE
preparation.

Every candidate must preserve the official BF16 and accumulation boundaries
established by the oracle work. Algebraic equivalence is not sufficient.

### Track T5 — Level Zero production prototype

Do not replace OpenCL. Add Level Zero only as a forced experimental runtime
candidate behind the Xe projection seam, preferably using the same kernel
semantics, ABI, and SPIR-V with a different host runtime.

Compare OpenCL, Level Zero regular lists, Level Zero immediate lists,
event/fence synchronization, device versus shared allocation, native module
cache/reload, submission/wait overhead, and long lifecycle behavior. Restore a
coherent runtime and exact header corpus first. Existing research suggests API
overhead alone is unlikely to dominate the full expert pipeline.

### Track T6 — Xe residency

Investigate bounded resident expert caching at evidence-selected capacities,
with 128, 256, and 512 MiB as experimental points rather than defaults.
Measure actual expert reuse and eviction behavior.

Cache identity must include model/tensor identity, layout/version, kernel ABI,
GPU PCI identity, and driver/runtime identity. Resident bytes, host staging,
and CPU fallback resources must remain separately accounted.

### Track T7 — Fused Xe expert pipeline

Evaluate:

```text
input activation
    -> MXFP4 gate/up
    -> official BF16 boundary
    -> SwiGLU
    -> activation preparation
    -> MXFP4 down
    -> official BF16 expert output
```

Keep intermediate data on Xe where possible and compare the current separate
projection path, fused OpenCL, fused Level Zero, and resident fused paths.
Avoiding repeated weight traffic and intermediate readback is likely more
important than shaving host launch overhead alone.

### Track T8 — Reopen Xe decode experimentally

Compare CPU AVX2 expert decode, Xe gate/up + readback + Xe down, fused Xe
expert decode, and resident fused Xe expert decode. X8's isolated M=1 win did
not become an X9 request-level win; only new end-to-end evidence can change
the production policy.

### Track T9 — CPU attention

Investigate AVX-512 QK dot, blocked KV traversal, GQA-aware layout/traversal,
stable vectorized softmax, V accumulation, sliding-window specialization,
online softmax, prefetch/cache behavior, and bounded scratch.

Short and long contexts may require different implementations. Preserve the
storage-neutral, absolute-position, sequence-isolated KV seam and all learned
sink/BF16 boundaries described by C4-C.

### Track T10 — Potential later Xe attention

Do not schedule this first. Preserve the option for sufficiently long-context
attention if KV can live in a sensible shared or resident representation.
Tiger Lake's integrated memory makes the question legitimate, not answered.

### Track T11 — Greedy LM-head fusion

For greedy requests that do not require full logits or logprobs, evaluate a
tiled LM-head projection with a running maximum/index that emits the token
without materializing the full FP32 vocabulary vector. Keep the full-logit
path for APIs that request it. CPU comes first.

### Track T12 — Thread/topology tuning

Measure one, two, and four physical cores; SMT/eight logical threads; and safe
process-local pinning versus ordinary scheduling. Repeat while Xe is active.
The correct policy may differ for AVX2, AVX-512, dense prefill, MXFP4,
attention, and concurrent CPU/Xe work. Do not alter global machine policy.

### Track T13 — CPU/Xe package contention

Measure CPU alone, Xe alone, and CPU + Xe for short bursts and steady state.
Capture wall throughput, readable memory-bandwidth indicators, CPU/GPU
frequency, temperature, and long-run stability. Prior T14 work observed shared
bandwidth contention; same-device identity does not make the result portable
to this Dell configuration.

### Track T14 — Model-shape specialization

Allow GPT-OSS-specific fixed N/K specialization while retaining generic
validated paths. Specialization may remove tails, variable loop bounds,
dimension branches, and unnecessary strides. Every artifact needs a clear
ABI/version/evidence identity and must remain capability-safe rather than
CPU-name-dispatched.

### Track T15 — Hardware-profile autotuning

A later `gpt-oss-rs tune` or equivalent internal calibration should choose
only among prevalidated candidates. A persisted profile may eventually choose
dense decode/prefill, MXFP4 decode/prefill, attention context regions,
greedy/full-logit LM head, Xe API/residency/expert mode, and thread policy.

Invalidation covers CPU capability and identity, topology, GPU identity,
driver/runtime, kernel ABI, repack layout/version, model identity, and
code/evidence version. Calibration is not implemented here.

## Proposed architectural direction

### Coarse runtime planner

Retain the existing request-level role:

```text
model family
runtime/trust mode
CPU/CUDA/etc.
request-level legality
```

Do not put every operation, shape, or kernel crossover into this planner.

### Hardware profile

Introduce conceptually immutable, evidence-backed machine/runtime knowledge:

```text
capabilities
hardware identity
runtime and driver identity
validated candidates
measured crossover regions
profile/evidence version
```

A profile cannot legalize an instruction or backend that capability checks
reject. A profile miss selects generic validated behavior.

### Operation planner

A future seam may accept:

```text
OperationContext {
    phase,
    operation,
    m,
    n,
    k,
    context_len,
    preparation_state,
    residency_state,
    concurrency_state
}
```

and resolve an effective implementation such as:

```text
cpu-avx2
cpu-avx512
cpu-avx512-matrix
xe-opencl
xe-level-zero
xe-fused-expert
reference
```

This remains conceptual pre-planning. Forced backend controls stay available
for correctness, regression localization, and benchmarking.

## Provisional answers to the sanity questions

1. **Is `main` a usable unified CPU + Xe baseline?** Yes for the landed native
   CPU and explicit Xe production histories. The fresh immutable CPU oracle
   campaign is not merged, and the present Intel userspace stack differs from
   the Xe evidence stack.
2. **Does the i7 expose expected Tiger Lake AVX-512/VNNI?** Yes. AVX2, FMA,
   AVX-512F/BW/VL/VNNI and OS state satisfy the forced repository path.
3. **Exact CPU identity?** GenuineIntel family 6, model 140, stepping 1,
   microcode `0xbe`.
4. **Exact Iris Xe?** `8086:9a49` revision 1, Dell subsystem `1028:0a42`,
   driven by `i915` at `/dev/dri/renderD128`.
5. **Are OpenCL and Level Zero both currently usable?** OpenCL is usable and
   passed the explicit startup numerical smoke. Level Zero libraries are
   installed, but current usability is not established because the harness
   cannot build and no info utility is installed.
6. **Does the current Level Zero research path build/probe?** No; it is blocked
   before CLI help by the missing cached OpenCL/Level Zero header corpus.
7. **Is `Auto(M>1) -> scalar` still true?** Yes for MXFP4 matrix work.
8. **Is dense BF16 multi-row repeated matvec?** Yes.
9. **Can production Xe host another Level Zero runtime?** The projection trait
   and ownership boundary are suitable, but attachment needs an explicit host
   API selector and a new runtime implementation.
10. **CPU/Xe architectural conflicts?** None in model-state ownership. Resolve
    evidence/stack identity, one shared resource budget, and planner-layer
    responsibilities before optimization.
11. **Highest-leverage tracks?** T1 profiling, T2 MXFP4 matrix expansion, T3
    dense BF16 matrix work, combined T6/T7 Xe residency/fusion, and T9 CPU
    attention. T12/T13 are required measurement dimensions across them.
12. **What must profiling capture first?** Phase, operation/layer class,
    M/N/K, real expert-bucket sizes, context and staged-KV shape, requested and
    effective backend, thread/affinity policy, preparation/repack/residency
    state, inclusive and component timings, scratch/resident high-water,
    fallback/reason, repetitions, and the exact host/runtime identity.

## Recommended first implementation-planning focus

Plan T1 first as bounded opt-in profiling that can answer T2, T3, T7, and T9
without changing dispatch. In parallel with that plan, make reconciliation of
the intended CPU oracle lineage and the Intel runtime/header corpus an entry
condition, not an optimization patch.

After representative profiles exist, rank one CPU prefill candidate and one
Xe data-movement candidate. The likely first comparison is explicit AVX2
MXFP4 matrix versus scalar `Auto(M>1)` over real expert buckets, followed by a
resident/fused Xe expert experiment. A Level Zero host-API port by itself is
not the first performance bet.

## Quick sanity validation

The pass intentionally omitted the overnight/full-model matrix.

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` | pass after document |
| `git diff --check` | pass after document; the untracked document also has no whitespace errors |
| `cargo check --workspace --locked` | pass |
| `cargo test -p gpt-oss-cpu-kernels --locked` | pass, 36 tests |
| Forced repository AVX-512/VNNI detector check | pass, 1 focused test |
| `cargo test -p gpt-oss-model-runner --lib --locked` | pass, 358 tests |
| `cargo test -p gpt-oss-xe --locked` | pass, 17 tests |
| Opt-in live explicit OpenCL attachment | pass, 1 hardware test |
| Xe research CLI `--help` | blocked: exact cached headers missing |
| `cpuid -1` | unavailable: utility not installed |
| `dmidecode -t memory` | unavailable without privilege; no sudo used |

No implementation was performed. No automatic dispatch was changed. No
Level Zero production runtime was added. No commit or push was performed.
