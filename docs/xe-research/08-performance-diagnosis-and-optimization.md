# X8 — Performance Diagnosis and Forced-Only Optimization

- Result: correct experimental winners selected for decode and prefill
- Scope: standalone research tool only
- Device: T14 Tiger Lake-LP Iris Xe `8086:9a49`
- API: forced OpenCL only
- Raw evidence: `/home/emmy/src/xe-research/results/20260812-xe-optimization-sprint/`
- Automatic or production dispatch: unchanged and disabled

## Result and claim boundary

X8 found a material kernel-organization improvement without weakening the X7
correctness, memory, or measurement rules. The selected forced research paths
are:

| Role | Variant | Workgroup | Applicable M |
| --- | --- | ---: | --- |
| Decode | `tile32-m1-v2` | 32 | 1 |
| Decode batch | `tile32-m2-v2` | 32 | 2 |
| Prefill | `tile32-m4-v2` | 32 | 4, 8, 16, 32, 64, 128 |

Every selected candidate beats the canonical Xe path with the conservative
bootstrap 95% speedup bound above parity. The M=2 and prefill candidates also
beat AVX2 by more than 2x. M=1 beats AVX2 by 1.56x. These are bounded
real-projection results for layer 0, expert 0 on this T14 and driver stack.
They do not enable automatic selection, production dispatch, serving, or
full-model inference.

Correct but slower `splitk-v2` remains available only through explicit
`benchmark --variant splitk-v2`. Its best workgroup is 64. It slightly beats
canonical Xe at M=1, loses at M=2, and loses to the selected tile candidates
at both shapes.

## Forced CLI and timing

`benchmark --variant` accepts `canonical-xe`, `tile32-m1-v2`,
`tile32-m2-v2`, `tile32-m4-v2`, or `splitk-v2`. New variants reject every API
except `--backend opencl` and still require `--device 8086:9a49`. Legacy
`--entry` remains reproducible, but combining it with `--variant` is rejected
as ambiguous. `diagnose` also requires an explicit variant and exact OpenCL
device selection.

The benchmark retains the legacy total-request measurement and separately
records residual-Q8 quantization, activation packing, upload, argument setup,
host submission, host wait, device event time, readback, and BF16 conversion.
CPU and GPU paths both reuse preallocated output/scratch storage. Median phase
times for selected cases were:

| M | Quantize | Pack | Upload | Args | Submit | Wait | Device | Read | BF16 | Total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36.5 us | 0.5 us | 1.0 us | 0.2 us | 19.3 us | 1.063 ms | 1.046 ms | 1.4 us | 2.0 us | 1.129 ms |
| 2 | 60.2 us | 0.9 us | 1.2 us | 0.2 us | 18.6 us | 1.160 ms | 1.143 ms | 1.8 us | 3.7 us | 1.251 ms |
| 4 | 104.8 us | 2.0 us | 2.0 us | 0.2 us | 20.3 us | 1.514 ms | 1.497 ms | 2.9 us | 7.3 us | 1.656 ms |
| 128 | 3.220 ms | 0.201 ms | 42.1 us | 1.5 us | 0.208 ms | 38.794 ms | 38.747 ms | 0.394 ms | 0.640 ms | 43.571 ms |

Host submission and wait are components of the OpenCL run interval and must
not be added to total as if disjoint from it. Device event time is reported by
OpenCL profiling. CPU `host_submission_ns` denotes CPU projection execution;
CPU device/wait phases are absent.

## ABI v2, layout, and residency

The immutable v1 ABI remains byte-for-byte unchanged. X8 adds
`gpt-oss-rs.xe-kernel-abi/v2` for new entries only.

Weights use `[output-tile][K-block][17 planes][32 lanes]`: plane 0 is the E8M0
scale and planes 1–16 are canonical low-nibble-first packed bytes. Adjacent
subgroup lanes therefore read adjacent bytes. The representation has the same
8,812,800 weight bytes as canonical blocks plus scales. A v2 session allocates
only this representation and the 23,040-byte FP32 bias, for 8,835,840 resident
bytes; it does not also allocate canonical device weights.

Each activation K block is one aligned 72-byte record containing primary
`i8[32]`, residual `i8[32]`, and both FP32 scales. One reusable activation
buffer replaces four independent writes.

- `tile32-m1-v2` launches in two dimensions and avoids linear-index row
  division/modulo.
- `tile32-m2-v2` decodes each weight once for two rows.
- `tile32-m4-v2` decodes each weight once for four rows.
- `splitk-v2` emits separate primary/residual FP32 terms for every K block and
  reduces primary then residual in canonical block order, without atomics. It
  is limited to M=1–2; M=2 scratch is exactly 8,294,400 bytes.

Checked shape validation rejects zero dimensions, K tails, non-32-column v2
tiles, inapplicable M values, stale/missing ABI entries, and a corrupted schema
before launch.

## Correctness

The v1 and v2 exact-block kernels both passed all 4,096 exhaustive E2M1/E8M0
combinations plus 10,000 fixed-seed randomized residual-Q8 blocks. Both
integer dot intermediates matched the scalar oracle exactly. V2 reported zero
finite-bit, NaN-behavior, and BF16-boundary mismatches across 14,096 cases.

Every applicable real-tensor shape for canonical Xe and all four candidate
families passed against the scalar oracle. All selected results happened to be
zero ULP with identical BF16 boundaries, which is stronger than the required
four-ULP/`1e-6` bound.

Unit coverage exhaustively round-trips every plane and lane of the 5,760 x 90
v2 layout and verifies 72-byte activation records. It also covers ambiguous
CLI flags, malformed dimensions/tails, inapplicable M, and ABI corruption.

## Performance and selection

Each full run used three rotating method orders, ten warmups per method and
trial, and 30 samples per method and trial: 90 retained samples for scalar,
AVX2, and the selected Xe variant at every shape.

| M | Winner median | Canonical Xe | Canonical/winner | Conservative lower | AVX2/winner | AVX2 lower |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.129 ms | 1.469 ms | 1.301x | 1.290x | 1.564x | 1.552x |
| 2 | 1.251 ms | 2.709 ms | 2.165x | 2.146x | 2.262x | 2.246x |
| 4 | 1.656 ms | 5.194 ms | 3.136x | 3.119x | 2.972x | 2.950x |
| 8 | 2.620 ms | 10.337 ms | 3.945x | 3.907x | 3.782x | 3.742x |
| 16 | 5.992 ms | 20.789 ms | 3.469x | 3.448x | 3.292x | 3.270x |
| 32 | 11.386 ms | 41.516 ms | 3.646x | 3.604x | 3.467x | 3.437x |
| 64 | 21.584 ms | 83.067 ms | 3.849x | 3.832x | 3.664x | 3.647x |
| 128 | 43.571 ms | 167.371 ms | 3.841x | 3.792x | 3.634x | 3.606x |

The replacement rule compares each candidate with current canonical Xe and
requires the conservative 95% lower bound above 1.0. The historical 1.25x
AVX2 gate is still reported separately. All prefill winners pass that older
gate; M=1 also exceeds 1.25x, although M=1 was not one of X7's predeclared
promotion shapes. No passing number changes the forced-only boundary.

Workgroups 32, 64, and 128 were screened for every family. An initial M=4
screen suggested width 128 for M4, so the entire M=4–128 protocol was rerun at
128. That result did not reproduce the screen: width 32 was faster at every
shape and remains selected. Split-K's width-64 screen did reproduce under a
complete M=1–2 rerun, so 64 is retained for that non-winner.

## Bandwidth, clocks, and PMU

The corrected GT path is
`/sys/bus/pci/devices/0000:00:02.0/drm/card1/gt/gt0/`. The diagnostic captured
under-load `rps_act_freq_mhz`, `rps_cur_freq_mhz`, and
`punit_req_freq_mhz`; observed values reached the configured 1,300 MHz
maximum. This is clock evidence, not EU-occupancy evidence.

Bandwidth cases used the exact 8,835,840-byte resident expert size and
`min(256 MiB, 5% MemAvailable)`, which selected 256 MiB. Effective median read
rates at workgroup 32 were:

| Size/path | Effective rate |
| --- | ---: |
| Exact coalesced | 14.07 GB/s |
| Exact canonical-strided | 5.39 GB/s |
| Exact v2-repacked/coalesced | 14.34 GB/s |
| 256 MiB coalesced | 11.25 GB/s |
| 256 MiB canonical-strided | 0.41 GB/s |
| 256 MiB repacked/coalesced | 11.16 GB/s |

CPU copy medians were 14.41 GB/s exact and 9.98 GB/s at 256 MiB. Concurrent
CPU/GPU traffic reduced the CPU side to 6.83 GB/s and 2.32 GB/s respectively.
This is contention evidence; it is not an overlap-throughput guarantee.

Noninteractive sudo authentication was unavailable. The PMU status is
therefore `insufficient_evidence`, and the diagnostic emits one exact user-run
`sudo perf stat` command for `i915/rcs0-busy/` and
`i915/actual-frequency/`. If supplied later through `--pmu-capture`, these
counters support render/compute engine-utilization and frequency claims only;
they do not prove that every one of 80 EUs was active.

## Native code generation

The current 26.05 driver produced the 231,008-byte OpenCL native module. The
preserved 23.43 `ocloc` was used only as an offline container/ISA decoder and
was not a compiler input or mapped into the hardware process.

| Entry | SIMD | GRFs | EU threads | DP4A instructions | Private-base payload | Scratch/spill fields |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `mxfp4_tile32_m1_v2` | 32 | 128 | 7 | 32 | present | 0 |
| `mxfp4_tile32_m2_v2` | 32 | 128 | 7 | 64 | present | 0 |
| `mxfp4_tile32_m4_v2` | 32 | 128 | 7 | 16 | present | 0 |
| `mxfp4_splitk_terms_v2` | 32 | 128 | 7 | 32 | present | 0 |
| `mxfp4_splitk_reduce_v2` | 32 | 128 | 7 | 0 | absent | 0 |

The metadata contains a stateless private-base payload for decode kernels but
does not state an allocated private-memory size. It contains no explicit
scratch, spill, or `private_memory` allocation field. That is the complete
available evidence; it is not upgraded to a claim of zero private memory or
zero register spills.

## Evidence and hashes

Paths are relative to the X8 raw evidence root.

| Claim | Artifact | SHA-256 |
| --- | --- | --- |
| Exact v1/v2 block correctness | `correctness-opencl/x5-opencl.manifest.json` | `2fc4b7e5fd09deee45c10f3e9c0bdee5d16e78c461b5a6daca478c25b297f47b` |
| Canonical Xe baseline | `benchmarks/canonical-xe-wg32/x6-opencl.manifest.json` | `d653e5d75ccf066fc0bc183b9957fbaf079358e3c1f1ca356ad7dc6bac0d01a4` |
| M1 selected | `benchmarks/tile32-m1-v2-wg32/x6-opencl.manifest.json` | `2f9651557fe8df26daa4fc4d79da985f7d3d720d07709670324d987897780d6b` |
| M2 selected | `benchmarks/tile32-m2-v2-wg32/x6-opencl.manifest.json` | `aa73adfb84661fcda3964e94c6a401a1fab5bfcb6175abe80fc43b2e178f1bf6` |
| M4 selected | `benchmarks/tile32-m4-v2-wg32/x6-opencl.manifest.json` | `e1a1a841c06ba1361cccec9a36c36065e38147aaa72db0983625b6ceb8c330cc` |
| M4 width-128 full rerun | `benchmarks/tile32-m4-v2-wg128/x6-opencl.manifest.json` | `1c7f2464d65424497d8be15bbb35093006cad898e8a6a6e58557af9d08fa8e00` |
| Split-K retained non-winner | `benchmarks/splitk-v2-wg64/x6-opencl.manifest.json` | `79e45c632fafb8d55c53e09864717b8ab745780d08c6bf12248bdde587bb6135` |
| Selection summary | `selection-summary.json` | `ff2fdd40244a7e2dae18ca5e7e3c500425b0e8701b9df3affa0023de85254428` |
| Code-generation summary | `codegen-summary.tsv` | `0c39230d616fb1e4336a60214ff8ce9ac725ca5330c276c6e0739eb5a1863443` |
| Diagnostics, clocks, bandwidth, PMU helper | `diagnose-tile32-m1-v2/x8-diagnose-opencl.manifest.json` | `f306b936b8c0fe3de766980ce16065fb896e561ba98026afa24e31defc8b7f32` |
| Full new-artifact index | `ARTIFACT_SHA256SUMS.txt` | `ac1f5a016ddb4fad2312123de44854cebbe82a82ed81787851d68bdf07e3f1a1` |

The earlier X0–X7 manifests and documented hashes were not modified. X7's
`fail` remains the honest result of its original promotion gate, not a claim
that its correctness, memory, artifact, or research work failed. X8 preserves
that checkpoint and publishes a new bounded optimization result beside it.
