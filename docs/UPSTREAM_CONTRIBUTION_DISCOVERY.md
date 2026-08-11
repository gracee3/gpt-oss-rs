# Upstream Contribution Discovery

- Status: living opportunity ledger
- Started: 2026-08-11
- Current preference: Rust-first contributions
- Non-Rust policy: reference by default; contribute only for a clear,
  reproducible gap that this project can fix responsibly

This document raises upstream contribution discovery from an eventual idea to
a recurring output of research and implementation. It does not require
`gpt-oss-rs` to become a compatibility framework or to shape local code around
hypothetical upstream consumers.

The intended sequence is:

```text
learn locally -> reproduce -> validate -> isolate reusable result
              -> confirm upstream fit -> discuss -> contribute
```

Local correctness and understanding come first. A small fixture, diagnostic,
documentation correction, or focused kernel can be a better contribution than
a large subsystem.

## Scope and priority policy

Prefer projects where:

- Rust is the implementation or primary user-facing language;
- the gap overlaps a result already exercised in `gpt-oss-rs`;
- older/common CPUs or modest systems are meaningful targets;
- a scalar oracle, fixture, benchmark, or reproducer can accompany the work;
- the change can remain narrow and maintainable.

Python- and C/C++-centered projects remain valuable research references. They
are not active contribution destinations unless a concrete defect or missing
test is discovered and the smallest correct fix belongs there. We are not
planning general ports to Python or C++.

Upstream work must not delay a blocking local correctness fix, force a broad
dependency into this repository, or weaken attribution and license tracking.

## Candidate lifecycle

Use one of these states:

- **WATCH**: plausible overlap, not source-audited for contribution fit;
- **OBSERVED**: a specific gap or reusable result has been seen;
- **QUALIFIED**: reproduced and supported by local correctness/performance
  evidence;
- **DISCUSS**: ready for an upstream issue or maintainer-direction question;
- **READY**: scope and destination agreed; patch can be prepared;
- **SUBMITTED**: issue or patch is upstream;
- **CLOSED**: merged, rejected, obsolete, or deliberately retained locally.

A candidate is not **QUALIFIED** merely because our implementation exists.
Record the upstream contract, difference, hardware, tests, license, and why the
change belongs upstream.

## Rust-first opportunity map

| Project | Current overlap | Possible contribution | State | Priority / next gate |
| --- | --- | --- | --- | --- |
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) | Strong: Rust GPT-OSS runtime, MXFP4 semantics, MoE, attention, scheduling | Portable MXFP4 edge fixtures; older-Intel scalar/SIMD findings; repack-cache or matrix-kernel work if its contracts align; diagnostics/docs | OBSERVED | **High.** Re-audit current upstream after local CPU certification, identify an unfilled issue, and discuss API fit before porting code. |
| [`cl3`](https://github.com/kenba/cl3) / [`opencl3`](https://github.com/kenba/opencl3) | Direct Rust overlap with the T14 OpenCL host lifecycle and capability/error surface | Missing-platform/build-log diagnostics, optional-feature queries, asynchronous lifetime tests, and an evidence-backed Intel integrated-GPU example | WATCH | **Medium research.** Complete the T14 lifecycle/source map and hardware capability capture before claiming a gap. |
| [oneAPI-rs](https://github.com/oneapi-src/oneapi-rs) | Medium: young Rust/SYCL ownership layer for Intel heterogeneous compute | Runtime/device diagnostics, query coverage, USM safety tests, older-iGPU examples; possibly reusable typed-resource fixes | OBSERVED | **Medium.** Run or reproduce one bounded issue on an equipped host; ask before proposing direct Level Zero scope. |
| [Candle](https://github.com/huggingface/candle) | Potential: Rust tensor and model ecosystem with CPU kernels | A reusable numerical fixture or CPU primitive only if Candle has the matching datatype/operation and accepts the maintenance cost | WATCH | **Medium research.** Inspect current quantized CPU/MX support and open issues; do not design against an assumed API. |
| [RTen](https://github.com/robertknight/rten) | Potential: Rust CPU inference and low-level tensor operations | Quantized-matmul fixtures, packing/dispatch methodology, or older-x86 correctness/performance fixes when contracts overlap | WATCH | **Medium-low research.** Establish whether its formats and target CPUs overlap before proposing anything. |
| [Burn](https://github.com/tracel-ai/burn) | Broad Rust ML framework; limited current GPT-OSS-specific overlap | Small backend-independent tests or Rust kernel techniques only if an exact gap is found | WATCH | **Low.** Framework breadth is not a reason to generalize this runtime. |
| [rust-gpu](https://github.com/Rust-GPU/rust-gpu) | Future Rust-to-SPIR-V path for integrated-GPU experiments | Minimal compute-kernel compatibility reproducer, documentation, or compiler issue discovered by a bounded Level Zero compute experiment | WATCH | **Later.** Requires an actual SPIR-V experiment and toolchain evidence. |

The reviewed mistral.rs revision used by the CPU semantic audit is
`8010b6a0578e416120b590ed72fd46ed5f24ee85`. The reviewed oneAPI-rs revision is
`1581663fdd0fd73e79df2900a2576d6cca8ff2a1`; the initial T14 `cl3` and
`opencl3` research pins are `9e2cdd8f34f09abfe49a8c2718ac58f1f762ae61`
and `072410552fecfc1e3f5395856735cb8684501f74`. Refresh the relevant source
before preparing an upstream proposal.

## Current mistral.rs candidate areas

These are questions to verify, not promised patches:

1. Does current mistral.rs already cover synthetic E8M0 special values,
   nibble order, scale boundaries, K/N tails, and residual-Q8 equivalence?
2. Is there an accepted CPU-native MXFP4 extension point where our scalar,
   AVX2, AVX-512/VNNI, repack-cache, or matrix-prefill evidence would fit?
3. Would reusable fixtures be preferable to introducing another persistent
   packed layout?
4. Can Tiger Lake and older Xeon measurements expose a dispatch or diagnostic
   gap without claiming universal performance?
5. Which code is independently expressed here and which portions carry MIT
   adaptation obligations already recorded in `THIRD_PARTY_NOTICES.md`?

The first likely contribution should be small: a fixture, edge-case fix,
targeted portable kernel improvement, or documented benchmark result. A full
CPU backend transplant would be difficult to review and would couple two
runtimes unnecessarily.

## Current oneAPI-rs candidate areas

The project is pre-0.1 and explicitly experimental. Its direction is safe Rust
access to SYCL, not a pure-Rust Level Zero runtime. Plausible contributions are:

- distinguish missing compiler, missing SYCL runtime, no platform, no device,
  unsupported feature, module-build failure, and launch failure;
- expand platform/device query coverage with typed results and tests;
- exercise USM ownership, host accessibility, event completion, and failure
  cleanup;
- document a reproducible integrated-GPU setup and failure matrix;
- reduce an actual safety or lifetime problem to a focused test and fix.

A standalone/direct Level Zero binding is a possible separate project idea,
not an assumed oneAPI-rs patch. Ask the maintainers whether it fits before
writing it.

## Reference-first and exceptional non-Rust destinations

| Project | What we learn from it | Contribution posture |
| --- | --- | --- |
| [vLLM](https://github.com/vllm-project/vllm) | Iteration/token-budget scheduling and post-execution progress/output commit | Reference only by default. Python/C++/CUDA work requires a clear reproduced correctness or documentation gap. |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Deployment across CPUs and accelerators, especially constrained/edge systems | Low direct overlap with GPT-OSS LLM kernels; no planned C++ port. Revisit only for a clearly shared low-level issue. |
| [oneDNN](https://github.com/uxlfoundation/oneDNN) | Packing, scratch ownership, BRGeMM, ISA dispatch, and AMX lifecycle | Mature C++ reference. Contribute only a concrete reproducible defect or documentation/test gap. |
| [Level Zero](https://github.com/oneapi-src/level-zero) | Explicit device/memory/module/queue/event lifecycle and loader diagnostics | Reference and possible issue destination. A Rust wrapper does not belong automatically in this C/C++ loader repository. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Q8 activation blocks, MXFP4 packing/dot organization, dispatch, contexts, server slots | Continue source/provenance audit; C++ contribution only for a clear defect we can prove. |
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | Row-interleaved MXFP4 layouts and multi-row x86 kernels | Reference/provenance source; no planned C++ port. |
| [OpenAI gpt-oss](https://github.com/openai/gpt-oss) | Canonical model and numerical behavior | Correctness authority/reference; contribute only a model-format or fixture issue with exact evidence. |
| [MegaBlocks](https://github.com/databricks/megablocks) and [Sarathi-Serve](https://github.com/microsoft/sarathi-serve) | Route/group/unroute and bounded chunked-prefill scheduling | Research references; GPU/Python contribution is outside current preference. |

This table deliberately records why a project is not currently a destination.
That prevents repeatedly reopening broad porting discussions without new
evidence.

## Contribution quality gate

Before contacting an upstream with a proposed change, record:

1. exact repository revision, path, symbol, and applicable license;
2. upstream issue, missing behavior, or independently reproduced gap;
3. smallest input that demonstrates it;
4. reference result and why that reference is trustworthy;
5. host CPU/GPU, OS, compiler flags, runtime versions, and dispatch path;
6. before/after correctness results;
7. before/after performance only when performance motivates the change;
8. behavior on unsupported hardware and the fallback path;
9. why the change belongs in that project rather than locally;
10. maintainer guidance if the API or project direction would expand.

Do not lead with percentage claims from one machine. Supply the named hardware,
shape/workload, repetitions, variance, build configuration, and raw-enough
results for another developer to reproduce the observation.

## Opportunity record template

Copy this section for each concrete discovery:

```text
ID:
State:
Discovered:
Project and canonical URL:
Revision / path / symbol:
Language and license:
Observed gap or reusable result:
Why it overlaps gpt-oss-rs:
Minimal reproducer or fixture:
Local implementation/evidence:
Named hardware and toolchain:
Proposed upstream scope:
Maintainer-direction question:
Risks or reasons not to contribute:
Next action:
Issue/PR URL and outcome:
```

## Near-term discovery work

1. Complete the deferred CPU certification and tuning phase on the available
   Intel hosts.
2. Compare the resulting edge fixtures and measured kernels with current
   mistral.rs rather than the older audit pin.
3. Select at most one small mistral.rs candidate for an upstream-fit
   discussion.
4. During the T14 Xe research, record `cl3`, `opencl3`, oneAPI-rs, OpenCL, and
   Level Zero diagnostic or lifecycle gaps instead of immediately building a
   wrapper.
5. Triage Candle and RTen source only against a concrete reusable primitive;
   close them as non-overlapping if their formats or policies differ.

The concepts already learned from external work, including those that are not
contribution candidates, are tracked separately in
[`BORROWED_CONCEPTS.md`](BORROWED_CONCEPTS.md).
