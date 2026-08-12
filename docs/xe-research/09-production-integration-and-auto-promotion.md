# X9: Production Integration and Automatic-Promotion Gate

## Decision

The production integration is accepted for explicit use as `--device xe` with
effective backend `cpu_xe`. Automatic Xe selection is **not promoted**. The
checked-in promotion record remains disabled, so `--device auto` selects CPU
without probing OpenCL.

This is a split decision, not a relaxation of the gate. Numerical, bounded
memory, fallback, and lifecycle evidence support the explicit hybrid path. The
predeclared full-model performance gate failed: the paired 95% lower bounds for
CPU/Xe time to first token and full-request duration did not establish an Xe
win above parity. A future automatic-selection attempt needs a new immutable
evidence record and a complete rerun.

## Production boundary

`gpt-oss-xe` is an internal, non-published crate included by the server's
default feature set. It loads the audited OpenCL functions through `libloading`;
the server has no OpenCL link-time dependency. Non-Linux builds use a portable
unsupported stub, and OpenCL-absent hosts retain CPU serving.

Attachment is deliberately narrow:

- exactly one OpenCL GPU and exact PCI identity `8086:9a49`;
- subgroup size 32, integer dot-product support, integrated memory, and checked
  allocation/workgroup limits;
- one coherent loader/driver/IGC generation;
- the compiler or a valid native-program cache;
- source SHA-256
  `dd467aa3fed7a4a4f5ef5c811bf478bf392eb943c633c2480fce0ac101dfedf6`, ABI v2
  SHA-256
  `62f13d73158f6b136993f5031b75f14cce2484cddb8482d2f0972119949bfcbe`, and build
  options `-cl-std=CL3.0 -DXE_ENABLE_DP4A=1`.

Automatic attachment additionally requires the exact checked-in driver and
core-library hashes. Explicit attachment on the same PCI device may accept a
changed stack only after capability and numerical startup tests and reports
`unvalidated_explicit`.

CPU owns model mapping, routing, attention, KV state, sampling, commit, and the
authoritative scalar/AVX2 repacks. Xe receives only selected expert projections
during prefill. The first production policy is:

- M=1–3 and every decode projection stay on CPU;
- M>=4 gate/up and down projections use `tile32-m4-v2`, workgroup 32;
- incomplete four-row groups receive zero activation records and their padded
  outputs are discarded;
- each operation repacks and uploads only the selected expert, then reuses it
  across checked row chunks;
- a 128 MiB default device slab holds one largest expert plus reusable
  activation/output buffers; host staging is accounted independently.

No decode cache, LRU, background migration, CPU/GPU overlap, split-K serving,
Level Zero serving, model-scale Xe residency, or full-GPU inference is included.

## Ownership and failure behavior

The backend owns one non-cloneable context and one serialized in-order queue.
All submissions terminate in an event, and shutdown drains before releasing
buffers, kernels, the program, queue, and context in reverse order. Shutdown is
idempotent. Unsafe code is confined to `gpt-oss-xe/src/opencl.rs`.

Native programs are cached atomically below the runtime cache root. The key
covers source, ABI, build options, PCI identity, driver version, and the
loader/driver/IGC hashes. Corruption or identity drift is a miss; source is
rebuilt and numerically tested when a compiler is available.

On a runtime OpenCL fault, the queue is drained, partial output is discarded,
and the projection falls through exactly once to the configured CPU path. The
process-wide Xe circuit breaker then remains open until restart. Since model
execution is prepared before commit, no Xe result is replayed and CPU fallback
must succeed before state can commit. Cancellation and shutdown likewise wait
for synchronous terminal completion before discard.

## Evidence protocol

New raw evidence is isolated under:

`/home/emmy/src/xe-research/results/20260812-xe-production-integration/`

X0–X8 artifacts were not modified. PMU data is absent because it requires a
separately authorized sudo command; this remains `insufficient_evidence` and is
not used as a production claim.

The paired full-model runner supplied to the summary is pinned by SHA-256 in
the combined summary. Each of ten CPU/Xe pairs per scenario has a passing
sidecar that hashes its raw capture. Samples alternate method order after one
warmup per method. The summary tool rejects missing sample IDs, token/oracle
disagreement, descriptor drift, sidecar/hash failure, and executable-source
drift. These negative-promotion captures predate the capture-level executable
hash field and their sidecars expose a dirty `c9b9f14` candidate plus an
early-to-late Cargo.lock hash change. They are therefore sufficient only to
reject automatic promotion, never to enable it. The later seven-scenario
correctness matrix was captured from clean `8d3c634` and embeds the measured
runner hash.

## Correctness and performance results

All seven pinned Harmony scenarios completed on CPU and explicit Xe with the
same native greedy tokens and the immutable official-oracle tokens. Successful
model loading also validates every checkpoint expert tensor's layout and
extent; the scenarios exercise the pinned routed-expert set. The exhaustive
projection evidence separately covers all 4,096 E2M1/E8M0 cases and 10,000
randomized residual-Q8 cases with exact integer intermediates, the four-ULP or
`1e-6` float bound, and identical BF16 boundaries.

Clean full-model correctness summary SHA-256:
`10b24225bb07d716b3949f401b16652a56f9347ccd8c052dea51f802462409af`.
Exhaustive projection manifest SHA-256:
`dc37aee155d581372a187d7135f4d46053bf1252851b84d0dadac3ecf0b87f96`.

The transfer-inclusive projection gate uses real layer-0 expert-0 gate/up and
down tensors. Timings include residual-Q8 preparation, expert repack, weight and
bias staging, activation packing, argument setup, submission, terminal wait,
readback, and BF16 conversion. Scalar, current CPU auto, explicit AVX2, and Xe
run in three rotated trials with ten warmups and thirty measured samples per
method and shape.

Both projection roles pass independently at every M in
`{4,8,16,32,64,128}`. At M=4, the conservative CPU-auto/Xe lower bounds are
8.9912 for gate/up and 8.8549 for down, so both explicit thresholds remain 4.
Transfer-inclusive projection summary SHA-256:
`2a6d59891de0a29653e1af5075a2b6cef9767c6deba87ecc16f2f767fdcaea38`.

The automatic full-model performance decision is based on `harmony_122` and
`harmony_444`. It requires both TTFT and full-request paired-bootstrap 95% lower
bounds above 1.0, Xe/CPU decode-throughput lower bound at least 0.98, p95
inter-token Xe/CPU upper bound at most 1.02, and the declared memory/no-swap
conditions. The combined evidence summary is authoritative; because any
scenario failure rejects promotion, the observed `harmony_122` intervals alone
are sufficient to keep automatic dispatch on CPU.

| Scenario | CPU/Xe TTFT, 95% CI | CPU/Xe full request, 95% CI | Xe/CPU decode, 95% CI | Xe/CPU p95 ITL, 95% CI | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `harmony_122` | 0.9993 [0.9938, 1.0050] | 0.9991 [0.9937, 1.0046] | 0.9939 [0.9880, 0.9992] | 1.0097 [0.9982, 1.0218] | fail |
| `harmony_444` | 0.9994 [0.9974, 1.0011] | 0.9995 [0.9975, 1.0013] | 1.0057 [0.9924, 1.0247] | 0.9863 [0.9560, 1.0082] | fail |

Both scenarios stayed within the declared 268,336,448-byte combined
device-plus-host bound; maximum paired Xe-minus-CPU RSS was 101,330,944 bytes
and 102,744,064 bytes respectively. Every measured process reported zero swap.
The first `harmony_122` pair incurred cold mapped-model faults and all later
pairs incurred none. `harmony_444` recorded at most four later major faults, so
the conservative unexplained-fault check also remains non-passing. None of
these results authorizes an automatic threshold.

Combined paired summary SHA-256:
`d4705c9857ac83caec2f31cd4f591bd6bc35a977ffb4ba41483d27a3c4d7170e`.
Pinned runner SHA-256:
`4ee60d453ed0102adfa40ba141641eb79619e1dd277f845d1a231fdf307c6e27`.

## Lifecycle and API gate

The lifecycle gate exercises chat completions and Responses in streaming and
non-streaming modes, two concurrent requests, cancellation before and after Xe
submission, readiness after cancellation, graceful draining, native-cache
reopen/corruption recovery, and repeated requests for 30 minutes while the
normal Wayland session remains active. It records server/process swap, major
faults, peak RSS, readiness wire keys, structured logs, and hashes of every
material artifact. A second startup supplies a deliberately invalid OpenCL
loader path under `--device auto` and proves the disabled record selects CPU
without probing OpenCL. `/ready` retains its existing wire shape.

The gate passed with 83 repeated requests over 1,812 seconds, one active
`kwin_wayland`, a peak server RSS of 11,825,856 KiB, zero server-process swap
growth, and 57 server major faults. System-wide swap use increased by 81,616
KiB during the run; this is retained rather than attributed to the server. The
corrupt native binary rebuilt to the original SHA-256, both explicit servers
shut down gracefully, and the automatic server selected CPU without loading
or probing OpenCL. Lifecycle summary SHA-256:
`3f2c09a2c22bd2774d86506fb37400d3d13c8b290cfe6a56be1c252a3aea1e96`.
Lifecycle artifact-index SHA-256:
`d4cc67ff98c077f369ef3fb7018ad90d0a67a50e1d94e3c438bd51ac2c1b411c`.

## Repository validation

The final validation covers locked/offline workspace checks and tests, default
and no-default server builds, warnings-denied Clippy for `gpt-oss-xe`, the Xe
projection gate, and CPU kernels, AMX feature compatibility, Rustls-backed HTTPS
fetch, Python evidence-tool tests, formatting, diff checks, embedded artifact
hashes, dynamic-link inspection, remote SHA verification, and GitHub Actions.

## Immutable result

The promotion record contains `automatic_enabled: false`,
`production_gate: fail_performance`, and `explicit_integration: pass`.
Correctness or lifecycle failure would have blocked the production path
entirely; none is waived. The explicit path is published because those gates
pass, while the failed statistical speed gate is preserved exactly and
prevents automatic selection.
