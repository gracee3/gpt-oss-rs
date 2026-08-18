# v0.1.0 benchmark protocol

## Purpose

The release measurement answers two bounded questions:

1. At real GPT-OSS 20B expert shapes, which checked CPU MXFP4 paths are
   scalar-bit-exact and measurably lower latency under a fixed host policy?
2. For one short, exact-token workload, what are the same-host startup,
   prefill/TTFT, request, decode, and memory observations for gpt-oss-rs Auto,
   gpt-oss-rs scalar, and two pinned llama.cpp CPU lanes?

It is not a production load test or a universal project ranking.

## Identity and admission

The standard-library
[`publication_benchmark.py`](../../crates/gpt-oss-bench/tools/publication_benchmark.py)
fails closed unless all of the following hold:

- repository `HEAD` equals the supplied clean publication-tooling commit;
- llama.cpp is clean at
  `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a`;
- GPT-OSS 20B config, index, and tokenizer match the pinned fixture hashes;
- the locally converted GGUF exists and is hashed;
- CPUs 0-7 are online, distinct physical cores;
- core temperatures are at or below 65 C before every run;
- CPU thermal-throttle counters do not increase and the post-run temperature
  remains at or below 95 C; and
- the output directory is new and outside the source repository.

GPU backends are disabled through both build configuration and the run
environment. Runs use eight threads under `taskset -c 0-7`. Prompt-cache reuse
is disabled. Each lane is a fresh process. No privileged page-cache drop is
performed, so repeated work observes a warm host page cache.

## MXFP4 matrix experiment

Both experiments use residual-Q8 activations, bias, scalar, AVX2,
AVX-512/VNNI, and Auto:

| Projection | M | N | K |
| --- | --- | ---: | ---: |
| Gate/up | 1, 3, 8 | 5760 | 2880 |
| Down | 1, 3, 8 | 2880 | 2880 |

Every path must produce exactly the scalar output bits. The protocol uses
three warmups, seven trials, five samples per trial, rotated method order, and
thermal/throttle rejection. The existing paired analyzer performs 10,000
bootstrap iterations. A candidate qualifies only when the upper bound of the
paired 95% interval for latency difference is below zero against every other
legal explicit candidate. Ties, uncertainty, gaps, and unobserved shapes fall
back to scalar.

## Full-model experiment

The workload is the pinned `harmony_63` fixture: 63 prompt tokens and eight
greedy output tokens. Four lanes run once as warmup and five times as measured
trials:

- gpt-oss-rs Auto with layer-major prefill;
- gpt-oss-rs scalar with the same layer-major execution shape;
- pinned llama.cpp with normal CPU prefill; and
- pinned llama.cpp with `ubatch=1` as the parity-control lane.

Lane order rotates one position each round. Published samples contain exact
prompt and generated token IDs, binary and source identities, startup,
prompt/TTFT, full-request time, decode throughput, peak RSS, thermal state, and
throttle counters. Local paths, hostnames, serial numbers, and raw process
logs are excluded.

For gpt-oss-rs, decode throughput is the seven post-first-token intervals
divided into seven tokens. llama.cpp retains its server-reported decode rate;
its prompt-processing time is the bounded TTFT proxy. These fields are useful
as same-host context but are not asserted to be identically instrumented.

## Analysis and reporting

Measured medians are reported for every lane, including negative or slower
results. Internal Auto-versus-scalar speedup is reportable only when a paired
10,000-iteration bootstrap 95% interval lies wholly above 1.0 in the favorable
direction. llama.cpp observations are contextual and do not produce a
cross-project speedup claim.

If any measured lane differs from the pinned official token sequence, the
driver retains each sequence, marks divergence, and omits decode/full-request
ratios. A core identity, correctness, thermal, execution, parse, checksum, or
schema failure makes the evidence incomplete and blocks release.

## Artifact contract

The bundle conforms to
[`publication-benchmark-v1.schema.json`](evidence/v0.1.0/schema/publication-benchmark-v1.schema.json)
and contains raw matrix samples, matrix analysis, normalized full-model
samples, aggregate analysis, conversion provenance, environment identity, and
`SHA256SUMS`. Unit tests cover parsing, aggregation, divergence, order drift,
invalid metrics, and checksum tampering.
