# X2 — Capabilities, Diagnostics, and Timing

- Device selection: exact `8086:9a49`; no ordinal-only fallback
- APIs: explicit `opencl` or `level-zero`; no automatic fallback
- Result: later experiments may use only the capabilities below

## Queried capabilities

| Capability | OpenCL 26.05 | Level Zero 1.28.2 / driver API 1.14 |
| --- | --- | --- |
| SPIR-V ingestion | `cl_khr_il_program`, IL versions 1.0–1.5 | module reports SPIR-V 1.5 |
| Native extraction/reload | program binary query/create | native module query/create |
| Subgroups | queried sizes 8, 16, 32 | queried sizes 8, 16, 32 |
| Integer dot | `cl_khr_integer_dot_product` and Intel subgroup extensions | no core capability bit; unsupported until compiler/native evidence |
| Max work-group | 512 | 512 |
| Local memory | 64 KiB | 64 KiB |
| Reported shared global memory | 14,994,825,216 bytes | 14,994,825,216 bytes |
| Max single allocation | 4,294,959,104 bytes | 4,294,959,104 bytes |
| Memory classes used | device/ordinary buffer, host-backed mapped buffer, coarse SVM | device, host, shared |
| Device timing | event profiling; timer resolution queried | timestamp event; 52 ns resolution, 36 global and 32 kernel valid bits |
| Host/device clocks | `clGetDeviceAndHostTimer` call must succeed | `zeDeviceGetGlobalTimestamps` call must succeed |

OpenCL does not expose the Level Zero timestamp-valid-bit fields; zeros in
those manifest fields mean not applicable, not an inferred clock width. Device
timestamps are recorded separately from end-to-end host time and are never
used alone for the useful-win decision.

Only one current Intel GPU matched. Selection still enumerates all platforms
or drivers and devices and compares vendor and PCI device identity rather than
accepting the first ordinal. A missing or wrong selector fails. The active
display/compositor state is captured at benchmark time; the sprint does not
disable it or change power policy.

## Negative and diagnostic matrix

| Case | Result and classification |
| --- | --- |
| Missing/wrong device | pass: deterministic `unavailable`-class selection error |
| Invalid OpenCL source/build | pass: nonzero build result and preserved compiler log |
| Invalid SPIR-V | pass: isolated child rejects it; Level Zero driver may terminate that child with exit 10 |
| Invalid native/OpenCL binary | pass: isolated child rejects it |
| Stale ABI entry point | pass: module/program creation or kernel lookup rejects it |
| Bad shape or K tail | pass: Rust validation rejects before FFI |
| Bad argument index/size | pass: checked setter returns an ABI error |
| Allocation over queried maximum | pass when the memory command receives the driver failure; host OOM is not induced |
| Launch failure | bad dimensions are rejected before submission; driver submission status is preserved |
| Timeout/cancellation | Level Zero bounded wait exists; generic OpenCL post-submit timeout is unavailable |
| Cleanup/partial initialization | pass: reverse-order null-safe destroy path and Rust `Drop` |
| Device loss | unavailable: unsafe injection on the display GPU was not fabricated |

Malformed Level Zero artifacts are always tried in a child process because
the current driver can terminate the caller instead of returning a fully
recoverable error. This is safe negative evidence but is also a lifecycle cost
against an in-process production loader. The proposed error boundary keeps
`unsupported`, `unavailable`, `invalid`, allocation failure, build failure,
launch failure, timeout, and context/device invalidation distinct.

## Evidence interface

The forced commands are:

```console
gpt-oss-xe-research environment  --backend <opencl|level-zero> --device 8086:9a49 --results DIR
gpt-oss-xe-research capabilities --backend <opencl|level-zero> --device 8086:9a49 --results DIR
```

Each command writes a `gpt-oss-rs.xe-research/v1` manifest with host/device,
repository revision and branch, exact command, timestamps, exit classification,
session capabilities, live mapped-library paths and SHA-256 hashes, and raw
artifact hashes. Allowed top-level statuses are `pass`, `fail`, `unsupported`,
`unavailable`, `invalid`, `incomplete`, and `insufficient_evidence`.

The structured negative results are evidence about this T14 and validated
stack only. They are not general OpenCL or Level Zero conformance claims.

## Evidence records

| ID | Manifest SHA-256 | Capability/negative raw SHA-256 |
| --- | --- | --- |
| X1-X2-OCL | `7ffb8c17ee16632064f037abfe53c3e41c1285ab16578587a9ee6e11c6beb6d7` | `b55a48ed080d2ee6d195e734246d0fed261a79662dfe54f0dd0c7452bfbc87f2` |
| X1-X2-L0 | `c79ab93e9413493824a6bd2c88ae788297070495c1c454d47b9a284e17efd3d9` | `d78d495b02d17dd48e7ab36bb46bf7cf1828ebb6c053463941e607e26c01bded` |

Both manifests report repository revision
`a9de8f8653eda75acce838899ac816fbf32735c7` and status `pass`.
