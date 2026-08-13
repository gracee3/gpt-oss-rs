# Tiger Lake Optimization Foundation Implementation Plan

- Branch: `agent/tiger-lake-optimization-foundation`
- Baseline: `6caf27423744148dafb1fb2670a03f29452311f3`
- Host profile: Intel family 6, model 140, stepping 1; 4C/8T; microcode
  `0xbe`; four-thread policy

## Ordered work

1. Reconcile the immutable CPU oracle lineage without merging its divergent
   history, preserving the published inputs and lock.
2. Add structured model-free CPU identity and dispatch diagnostics plus an
   opt-in, bounded execution profiler.
3. Capture and hash a representative Tiger Lake operation/shape corpus.
4. Benchmark forced scalar, AVX2, and AVX-512/VNNI MXFP4 matrix candidates and
   promote only statistically proven regions for the exact hardware profile.
5. Add a forced-only bounded OpenCL expert-residency experiment.
6. Freeze a candidate, run the fresh correctness campaign and release checks,
   and publish closure records keyed to the implementation candidate.

Correctness gates every performance decision. Unknown hardware, uncertain
measurements, unsupported ISA state, and unobserved shapes retain scalar
multi-row Auto dispatch. Xe residency remains explicit-device-only.

## Non-goals

Dense BF16 matrix kernels, attention kernels, LM-head fusion, Xe fusion or
decode, Level Zero production, general autotuning, trusted-mode changes, broad
Intel GPU support, and automatic Xe promotion are excluded.
