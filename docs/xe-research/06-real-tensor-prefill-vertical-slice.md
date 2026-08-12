# X6 — Real-Tensor Prefill Vertical Slice

- Terminal gate: fail
- Correctness: pass at every shape and for every path
- Useful-win result: no Xe path reached 1.25x at M=4, 8, 16, 32, or 64
- Consequence: close the Xe implementation lane

## Method

The benchmark uses the real layer-0/expert-0 `gate_up` bundle documented in
X4, deterministic seeded BF16 activation rows, and
`M={1,2,4,8,16,32,64,128}`. M=4–64 were predeclared as plausible bounded
interactive routed-row shapes.

Every request-path sample includes activation residual-Q8 quantization, host
packing, required staging, submission, synchronization, visibility/readback,
and BF16 output conversion. Module creation, the 8,835,840-byte persistent
compact weight allocation/staging, and reusable scratch allocation are
reported separately. Every shape uses three independent interleaved method
orders, ten warmups per method per trial, and thirty samples per method per
trial: 90 measured samples for the scalar oracle, AVX2, and Xe.

The fixed-seed bootstrap reports a 95% interval for median speedup. Each raw
distribution also records median, p95, MAD, min, max, and its own median
interval. All values in the tables below are milliseconds except speedup.
Cells are `median / p95 / MAD`.

## OpenCL scalar projection

| M | Scalar oracle | AVX2 | Xe end-to-end | AVX2/Xe and 95% interval |
| ---: | ---: | ---: | ---: | --- |
| 1 | 17.021 / 17.484 / 0.081 | 1.767 / 1.875 / 0.018 | 1.424 / 1.612 / 0.045 | 1.242x, 1.228–1.270 |
| 2 | 34.161 / 34.529 / 0.175 | 2.814 / 2.927 / 0.007 | 2.711 / 2.805 / 0.029 | 1.038x, 1.035–1.043 |
| 4 | 68.404 / 69.962 / 0.302 | 4.907 / 5.124 / 0.045 | 5.275 / 5.688 / 0.037 | 0.930x, 0.924–0.940 |
| 8 | 136.591 / 139.115 / 0.391 | 9.796 / 9.991 / 0.047 | 10.485 / 10.895 / 0.055 | 0.934x, 0.930–0.939 |
| 16 | 273.344 / 277.933 / 0.957 | 19.606 / 20.016 / 0.092 | 21.611 / 22.280 / 0.195 | 0.907x, 0.904–0.915 |
| 32 | 549.423 / 555.344 / 2.634 | 39.644 / 40.432 / 0.262 | 42.875 / 44.187 / 0.322 | 0.925x, 0.920–0.930 |
| 64 | 1100.387 / 1107.810 / 3.445 | 79.136 / 80.962 / 0.474 | 84.771 / 86.604 / 0.609 | 0.934x, 0.928–0.939 |
| 128 | 2197.390 / 2204.993 / 3.799 | 158.565 / 161.605 / 0.659 | 169.875 / 172.779 / 0.771 | 0.933x, 0.931–0.937 |

M=1 was the only result near the threshold, but 1.242x is below 1.25x and M=1
was not a predeclared plausible gate shape. Its one-time residency break-even
estimate was 54.5 requests. M=2 estimated 181.9 requests but was only 1.038x.
No shape at which Xe was slower has a meaningful positive break-even count.

OpenCL module creation was 16.74 ms. Compact weight staging was 1.99 ms,
including 1.988 ms of reported writes. The benchmark raw JSON hash is
`68792ce511b07cfe6b9730ec8eaa704f2e5029c2a344f2c5e300659302297d88`.

## Level Zero regular recycled lists

| M | Scalar oracle | AVX2 | Xe end-to-end | AVX2/Xe and 95% interval |
| ---: | ---: | ---: | ---: | --- |
| 1 | 17.078 / 17.424 / 0.087 | 1.779 / 1.997 / 0.022 | 2.033 / 2.150 / 0.047 | 0.875x, 0.864–0.890 |
| 2 | 34.059 / 34.732 / 0.076 | 2.830 / 3.026 / 0.017 | 3.278 / 3.400 / 0.040 | 0.864x, 0.860–0.871 |
| 4 | 68.353 / 70.020 / 0.236 | 4.939 / 5.364 / 0.042 | 5.699 / 6.077 / 0.059 | 0.867x, 0.860–0.874 |
| 8 | 136.597 / 138.666 / 0.417 | 9.879 / 10.132 / 0.063 | 11.009 / 11.311 / 0.096 | 0.897x, 0.892–0.902 |
| 16 | 273.845 / 276.228 / 1.087 | 19.820 / 20.258 / 0.147 | 21.482 / 21.961 / 0.233 | 0.923x, 0.916–0.931 |
| 32 | 549.180 / 554.755 / 1.314 | 40.162 / 40.935 / 0.232 | 42.769 / 43.823 / 0.353 | 0.939x, 0.933–0.943 |
| 64 | 1100.362 / 1107.286 / 2.189 | 79.495 / 80.881 / 0.339 | 84.149 / 85.659 / 0.566 | 0.945x, 0.942–0.950 |
| 128 | 2200.239 / 2208.416 / 3.508 | 159.431 / 163.363 / 0.657 | 167.291 / 168.984 / 0.843 | 0.953x, 0.950–0.956 |

Module creation was 25.92 ms and compact weight staging was 3.51 ms. The raw
JSON hash is
`ff375ef64a80dda7c83560753898f2e6c39225201d784f743fe1ad2b078c2b42`.

## Level Zero immediate lists

| M | Scalar oracle | AVX2 | Xe end-to-end | AVX2/Xe and 95% interval |
| ---: | ---: | ---: | ---: | --- |
| 1 | 17.075 / 17.787 / 0.080 | 1.768 / 1.923 / 0.011 | 1.995 / 2.082 / 0.034 | 0.886x, 0.880–0.893 |
| 2 | 34.034 / 34.676 / 0.077 | 2.826 / 2.949 / 0.013 | 3.248 / 3.378 / 0.049 | 0.870x, 0.866–0.878 |
| 4 | 68.439 / 69.473 / 0.260 | 4.929 / 5.120 / 0.025 | 5.702 / 5.983 / 0.042 | 0.864x, 0.860–0.870 |
| 8 | 136.909 / 139.182 / 0.570 | 9.925 / 10.158 / 0.070 | 10.974 / 11.423 / 0.107 | 0.904x, 0.898–0.909 |
| 16 | 274.084 / 278.176 / 1.117 | 19.950 / 20.257 / 0.133 | 21.639 / 22.067 / 0.217 | 0.922x, 0.916–0.930 |
| 32 | 550.057 / 556.495 / 2.028 | 39.979 / 40.864 / 0.163 | 42.600 / 43.846 / 0.372 | 0.938x, 0.934–0.944 |
| 64 | 1100.021 / 1109.073 / 2.466 | 80.086 / 82.626 / 0.429 | 84.033 / 85.354 / 0.675 | 0.953x, 0.948–0.958 |
| 128 | 2201.228 / 2211.716 / 3.685 | 160.052 / 163.525 / 0.497 | 167.178 / 169.454 / 1.027 | 0.957x, 0.954–0.963 |

Module creation was 25.50 ms and compact weight staging was 3.72 ms. The raw
JSON hash is
`99aaf0caad0c841fd7d80a6888e1ab8b328df95cd5abc58e3e632b75b999a240`.
Immediate lists did not materially beat recycled regular lists; their median
intervals overlap at every plausible shape.

## Correctness and environment

All scalar-vs-AVX2 and scalar-vs-Xe comparisons at all eight shapes had zero
non-finite mismatches, zero tolerance mismatches, zero maximum ULP distance,
and zero BF16-boundary mismatches. Performance was therefore never continued
past an unexplained numerical discrepancy.

Per-run captures record an active local Wayland session, `performance` power
profile, CPU frequencies, and thermals without changing policy. The final
supplement records AC online, the active user/display sessions, CPU frequency,
and all available thermal zones; its SHA-256 is
`c1d9e6f58626e10ab89fa42dbd3c6c9c9a473591fe3135d6d59438f5823e01f5`.
No GT frequency sysfs file was exposed by the current card path, so GPU
frequency is unavailable rather than estimated. Observed package thermal
captures during long runs were approximately 72–78°C.

## Gate and API comparison

Every plausible OpenCL interval is below parity. Both Level Zero modes are
below parity at every shape. OpenCL was about 7.4% faster than Level Zero
regular at M=4 and about 4.8% faster at M=8; Level Zero became slightly faster
at larger M. No plausible shape gives either API a 10% advantage with
non-overlapping confidence intervals. The API-speed selection rule therefore
does not fire.

The explicit subgroup and work-group alternatives also fail:

| Evidence | Manifest SHA-256 | Result at M=4 |
| --- | --- | --- |
| DP4A subgroup 8, WG64 | `c5e59b7774813440b80cd3cf5c1843a28a92b4cd69ecb32143dcf5d25a1949a2` | 0.568x |
| DP4A subgroup 16, WG64 | `074b9eb7faf5de7595321a6638ef717b0c2d7b8a7b050d287a2fb883bce2f354` | 0.869x |
| DP4A subgroup 32, WG64 | `30239d866ec11548562502db71a40444a1eca72416fd6b723c404c9ea566bb19` | 0.931x |
| subgroup 32, WG32 | `43b5945466bee4ec38b3e83cc52571701f7c5d3fe8371c9eaaa4b3125edead29` | 0.917x |
| subgroup 32, WG64 repeat | `e73bd8b6b1f5523f1ffe52011bad8f636194d55cd205952f26a7eb5e6719a915` | 0.909x |
| subgroup 32, WG128 | `ddd984bb43154fbe2d47319b27898b5ff2f11368072eb96e9c5ce2d5c54887a5` | 0.879x |

## Primary evidence records

| ID | Manifest SHA-256 | Status |
| --- | --- | --- |
| X6-OCL-scalar | `79cdaf5dea2b8d8bed784cfc2c6deaad17a12101abc93e046aafe752234e4b6a` | fail |
| X6-L0-regular | `6f903c4b9b224bdbeb4fff4b9a1204c3679d63d6f66332e6a1f4439331616639` | fail |
| X6-L0-immediate | `0e9d556a5f64531a6bbee4f475b5aa4089ea578e1ad367f07e2c6c74b446a5af` | fail |

`fail` means the useful-win gate failed; correctness and memory gates remain
passing. All three primary records report revision
`2e24505ac605567b01f64222eba0cca4159c9199`.
