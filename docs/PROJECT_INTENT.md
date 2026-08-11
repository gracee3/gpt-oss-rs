# Project Intent and Hardware Focus

`gpt-oss-rs` is intentionally a narrow, educational, evidence-producing
project. Its purpose is to make GPT-OSS work well on hardware available to the
project owner, to make the implementation understandable, and to publish the
code, research, measurements, and failures that may help other people. It is
not an attempt to maximize framework compatibility, model coverage, or
datacenter-scale feature breadth.

## Current motivation

CPU inference is the present priority. Capable Intel laptop and desktop CPUs
are widespread, including among people who do not own a supported discrete
GPU. A useful native CPU path can therefore make local GPT-OSS experimentation
available on ordinary machines while exposing important systems questions:

- how compact MXFP4 weights should be mapped and repacked;
- how scalar, AVX2, and AVX-512 implementations preserve the same semantics;
- how memory bandwidth, cache behavior, and instruction throughput interact;
- how prompt prefill differs from single-token decode;
- how model state, sequence state, scratch, and scheduling should be owned;
- how optimization claims can be tested against a simple reference and a full
  checkpoint.

The constraints are part of the project rather than an inconvenience to hide.
They create concrete engineering problems that can be studied, measured, and
explained.

## Present hardware envelope

Development and validation should prioritize hardware that is actually
available:

- an Intel Tiger Lake laptop for scalar, AVX2, and available AVX-512/VNNI CPU
  work;
- DDR4 systems where memory capacity and bandwidth are first-class limits;
- a second-generation Intel Xeon Scalable system when it is available again;
- RTX 3090 systems when they are available again for deliberate CUDA work.

The current absence of the Xeon server and RTX 3090 systems is a reason to
focus on CPU work that can be executed and measured now. Documentation may
prepare future experiments, and portable code may be compile-tested, but a
backend must not be presented as tuned or validated on hardware the project
does not possess or otherwise have responsible access to.

AMX is the clearest current example. It is worthwhile to research its numerical
mapping, operating-system lifecycle, API seam, portable packing, and scalar
emulation. Performance tuning and automatic selection should wait for an
AMX-capable host. The same rule applies to future hardware targets.

## Scope discipline

The project should prefer a small implementation whose behavior can be
understood and defended over a broad abstraction designed for hypothetical
consumers. In particular:

- GPT-OSS remains the model focus.
- Native CPU execution remains the active backend focus.
- Scalar code remains the correctness oracle for optimized CPU kernels.
- Optimized paths are selected only with relevant correctness and measurement
  evidence.
- Canonical checkpoints remain unchanged; generated packed caches are
  disposable, versioned derivatives.
- CUDA remains an explicit path to revisit on the available NVIDIA systems.
- OpenCL/Level Zero on available Intel integrated graphics and Apple Silicon
  are interesting later investigations, not current implementation
  commitments. Vulkan is not an active research target.
- General framework compatibility is not a design goal unless it directly
  improves this runtime or makes a specific result meaningfully reusable.

This focus does not forbid clean interfaces. Separating kernels, persistent
weights, scratch, sequence state, and scheduling makes the code easier to
reason about here and may later make parts reusable elsewhere. Reusability is
a useful consequence, not the controlling objective.

## Role of external projects

Projects such as oneDNN, llama.cpp, ik_llama.cpp, mistral.rs, vLLM, and the
official GPT-OSS implementation are references and comparison points. They can
teach this project about:

- numerical contracts and edge cases;
- packing and scratch ownership;
- CPU feature detection and dispatch;
- GEMV, GEMM, and AMX dataflows;
- hardware-context and thread lifecycle;
- batching, cancellation, and state ownership;
- benchmark design and failure modes.

oneDNN is especially valuable as a mature example of primitive descriptors,
caller-visible packing and scratch requirements, runtime ISA selection, and
scoped AMX hardware context. That does not require adopting oneDNN as a runtime
dependency or reshaping `gpt-oss-rs` around its API. An integration experiment
is justified only when it answers a concrete question, such as whether its
INT8 BRGeMM can efficiently execute the tile multiplication inside an exact
MXFP4/AMX prototype.

External implementations are evidence, not authority over this repository's
architecture. Observations must remain pinned and attributed, copied or adapted
code must retain its license obligations, and performance conclusions must be
measured on named hardware.

## Intended public contribution

The project aims to share more than a finished binary. Useful outputs include:

- readable scalar and optimized kernels;
- numerical fixtures and full-checkpoint comparisons;
- persistent-layout and cache designs;
- benchmark methods with host and workload context;
- documented false starts, regressions, and negative performance results;
- source-grounded research and implementation plans;
- narrowly reusable components when their contracts become stable.

Potential upstream contributions should follow learning and validation. Once a
kernel, algorithm, fixture, or lifecycle design is understood and supported by
evidence, it can be evaluated for contribution to a compatible project. The
project does not need to predict or satisfy every upstream interface in
advance.

## Decision rule for new targets

A new backend or hardware target should be adopted deliberately. Before it
becomes an active target, record:

1. the hardware and access available for validation;
2. the user or research value of the target;
3. the smallest honest implementation boundary;
4. the reference and correctness strategy;
5. the measurements required before automatic selection or performance
   claims;
6. the effect on build complexity and maintenance.

For now, the answer is CPU first: make GPT-OSS understandable, correct, and
useful on the Intel systems at hand, and let that constrained work produce the
knowledge needed for later choices.
