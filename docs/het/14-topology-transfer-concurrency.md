# Topology, transfers, concurrency, and crossover evidence

## Communication matrix

The external CUDA probe was compiled with `nvcc 13.3.73` and synchronized every
timed operation. Its final source SHA-256 is
`cfa0a584abe5eb0b137115cdbdd29e6f26f51faf65f048ff1ecd78cedc7ddc74`;
the earlier repeatable distribution run used the semantically identical probe
before the added first-copy records and is indexed in the evidence directory.

| Ordered path | Direct CUDA peer | Enable result | NCCL transport | Status |
|---|---:|---|---|---|
| GPU0 → GPU1 | `cudaDeviceCanAccessPeer = 0` | error 217, `cudaErrorPeerAccessUnsupported` | SHM/direct host transport | **Verified:** no CUDA P2P |
| GPU1 → GPU0 | `cudaDeviceCanAccessPeer = 0` | error 217, `cudaErrorPeerAccessUnsupported` | SHM/direct host transport | **Verified:** no CUDA P2P |
| CPU ↔ either GPU | n/a | pageable and pinned CUDA copies succeed | host memory | **Verified** |

Both RTX 3090 devices report compute capability 8.6, two asynchronous engines,
concurrent-kernel capability, and independent PCI addresses on separate root
branches. `nvidia-smi nvlink --status` reported no active link.

**Unknown:** software evidence cannot distinguish a physically absent bridge
from an unbridged or disabled link. The operational fact needed by this phase is
closed: CUDA and NCCL provide no direct GPU peer path on the current system.
No driver modification or workaround was attempted.

The unmodified NCCL v2.31.2-1 checkout was built for SM86 without installation.
Sanitized diagnostics reported `isAllCudaP2p 0` and both channels `via
SHM/direct`. A two-rank f32 sum produced exactly 3.0 at both ends. Median/p95
completion latency was 53.331/63.474 µs for 5,760 bytes, 60.279/62.395 µs for
23,040, 109.401/117.536 µs for 184,320, 186.176/193.046 µs for 737,280,
512.835/519.917 µs for 2,949,120, and 2,086.889/2,107.748 µs for 13,236,480.
NCCL is therefore a proven host-transport collective, not evidence for peer
memory or a natural irregular-expert dispatch API.

## Model-derived payloads

`H=2,880`, BF16 output, and top-k 4 yield:

```text
one route activation or result = M × 2,880 × 2 B = M × 5,760 B
duplicated top-4 payload        = M × 4 × 5,760 B = M × 23,040 B
full send + returned result     = 2 × the chosen packed payload
```

| Occupancy | One expert/destination | Conservative top-4 duplicate |
|---:|---:|---:|
| decode `M=1` | 5,760 B | 23,040 B |
| observed median `M=7` | 40,320 B | 161,280 B |
| observed p90 `M=24` | 138,240 B | 552,960 B |
| observed p95 `M=33` | 190,080 B | 760,320 B |
| observed max `M=61` | 351,360 B | 1,405,440 B |

IDs and f32 weights add at most 32 bytes/token in the conservative record.
Destination packing can avoid duplicating an activation when multiple ranks on
the same device share a source row, but that optimization is **Hypothesis**, not
counted in the safe payload envelope.

## Steady pinned transfers

The table reports median/p95 microseconds from the second isolated run. Sample
count is 100 except 13,236,480 bytes, where `n=40`. Each value is completion
time after a 10-iteration warm-up, not enqueue time.

| Bytes | GPU0 H2D | GPU0 D2H | GPU1 H2D | GPU1 D2H |
|---:|---:|---:|---:|---:|
| 5,760 | 6.870 / 7.537 | 5.507 / 5.864 | 6.113 / 6.793 | 4.754 / 5.009 |
| 23,040 | 11.900 / 12.683 | 6.825 / 7.195 | 10.972 / 11.886 | 5.965 / 6.271 |
| 184,320 | 24.458 / 25.172 | 22.234 / 22.746 | 21.250 / 21.768 | 22.091 / 22.728 |
| 737,280 | 98.935 / 99.904 | 82.069 / 82.813 | 95.444 / 96.816 | 83.125 / 83.749 |
| 2,949,120 | 372.265 / 382.882 | 311.828 / 320.868 | 366.729 / 376.715 | 304.911 / 317.840 |
| 13,236,480 | 1,689.627 / 1,706.406 | 1,522.200 / 1,534.402 | 1,658.070 / 1,673.498 | 1,516.527 / 1,522.046 |

Pageable controls were consistently less useful at model-sized payloads: at
737,280 bytes GPU0 pageable H2D/D2H was 144.695/162.174 µs median versus pinned
98.935/82.069; at 13,236,480 bytes it was 1,732.206/1,674.189 versus
1,689.627/1,522.200.

Fresh-buffer first-copy controls were added after synthesis exposed that the
original harness discarded its warm-ups. They are single observations after
new host/device allocation and stream creation, with the process CUDA context
already initialized—not distributions and not cold-process startup. For pinned
5,760/737,280/13,236,480-byte copies, GPU0 H2D was
12.598/105.133/1,706.632 µs and D2H 13.423/89.241/1,510.339 µs; GPU1 was
11.564/99.274/1,669.427 and 12.741/87.630/1,511.045 µs. These bracket warm
steady state but are too sparse for a tail claim.

Pinned allocation/registration cost was approximately 0.68–0.81 ms for small
regions and 4.0–4.6 ms for 13.2 MiB. **Conclusion:** staging regions must be
bounded and pooled; allocation cannot sit on the per-route critical path.

## Simultaneous and relayed paths

Independent pinned regions and one stream per device complete simultaneously:

| Bytes per GPU | dual H2D median/p95 µs | dual D2H median/p95 µs |
|---:|---:|---:|
| 5,760 | 8.455 / 9.607 | 6.246 / 6.718 |
| 23,040 | 16.363 / 17.531 | 7.719 / 8.650 |
| 184,320 | 25.193 / 26.461 | 19.877 / 20.218 |
| 737,280 | 85.423 / 87.535 | 83.769 / 84.354 |
| 2,949,120 | 357.715 / 373.452 | 309.010 / 319.021 |
| 13,236,480 | 1,657.836 / 1,672.364 | 1,548.751 / 1,552.477 |

At the largest payload, `2 × 13,236,480 B / 0.001657836 s = 15.97 GB/s`
decimal aggregate. The numerator includes both devices.

A serialized one-way GPU→pinned-host→other-GPU relay completed as follows:

| Bytes | GPU0→host→GPU1 µs | GPU1→host→GPU0 µs |
|---:|---:|---:|
| 5,760 | 11.803 / 12.547 | 11.944 / 12.631 |
| 23,040 | 18.395 / 19.338 | 18.503 / 19.377 |
| 184,320 | 40.123 / 41.216 | 39.740 / 41.603 |
| 737,280 | 154.047 / 182.546 | 154.891 / 183.038 |
| 2,949,120 | 674.642 / 678.691 | 665.666 / 677.548 |
| 13,236,480 | 3,161.877 / 3,176.196 | 3,165.731 / 3,174.905 |

This is one relay leg. A remote expert whose layer owner and worker are GPUs
requires an activation leg and a returned-output leg, plus packing, events, and
kernel execution. Two isolated runs agreed within about 4.5% for most medians;
the 184,320-byte simultaneous H2D case varied by about 14%, so it is retained
as an uncertainty rather than averaged away.

## Compute and interference controls

The CUDA cuBLAS control expands weights to FP16 and times only gate/up plus down
projections: approximately 80 µs at `M=1`, 74 µs at `M=32`, and 113 µs at
`M=128`. It excludes MXFP4 dequantization, exact GPT SwiGLU, routing, reduction,
and current-kernel inefficiencies and is therefore **not an expert latency**.

At `M=32`, a roughly 75 µs cuBLAS control overlapped a 737,280-byte transfer on
a separate stream: combined H2D completion was 108–111 µs rather than roughly
170 µs serial, and combined D2H was 98–99 µs rather than roughly 157 µs. This
**verifies capability to overlap**, not exact end-to-end overlap.

The real-weight CPU reference probe used layer 0 expert 0:

| Threads | Exact gate/up median | Exact down median | Combined |
|---:|---:|---:|---:|
| 1 | 51.768 ms | 25.991 ms | 77.759 ms |
| 4 | 16.109 ms | 7.170 ms | 23.279 ms |
| 8 | 7.665 ms | 3.824 ms | 11.489 ms |
| 16 | 8.188 ms | 3.978 ms | 12.166 ms |

All thread counts produced identical output hashes. Canonical-to-x8 repack was
3.88–3.93 ms for gate/up and 1.85–1.86 ms for down; this is a once-per-owner
construction cost, not arithmetic.

Two explicit affinity repeats compared CPUs `0-7` (one hardware thread from
each physical core) with `0-15` (both SMT siblings). At eight workers, combined
gate/up+down medians overlapped: 11.887/12.209 ms for physical-only and
12.281/11.861 ms for all logical CPUs. At 16 workers, the physical-only mask
(therefore oversubscribed) was 11.291/11.782 ms versus 12.480/12.361 ms with SMT
available. All hashes remained identical. **Inferred:** SMT can add contention,
but the 5–9% 16-worker difference and run-to-run variability are too small and
probe-specific to freeze affinity policy. Physical-core affinity remains a
cost-model input, not a host setting changed by this phase.

The clean repository matrix harness measured current `residual-q8` behavior.
Gate/up plus down medians were:

| M | auto | forced AVX2 | forced AVX-512 VNNI | scalar |
|---:|---:|---:|---:|---:|
| 1 | 4.569 ms | 4.826 ms | 6.358 ms | 35.644 ms |
| 4 | 141.534 ms | 16.583 ms | 16.938 ms | 141.353 ms |
| 16 | 563.141 ms | 64.579 ms | 60.582 ms | 564.066 ms |
| 64 | 2,266.729 ms | 258.483 ms | 242.581 ms | 2,267.554 ms |

All forced outputs matched the scalar hash. **Policy boundary:** `auto` chooses
the approved optimized decode path but remains scalar for `M>1`; the closed
fused-linear policy lane forbids treating these forced results as a promotion
decision.

During simultaneous CUDA copy/compute traffic, the exploratory 8-thread CPU
probe slowed about 30% for gate/up and 16% for down, while GPU transfer medians
changed little. It was one bounded overlap run, so the planning cost model must
carry host-contention uncertainty.

## Crossover model and conclusions

For a destination bucket `b`, use measured components rather than a device-name
threshold:

```text
T_cpu(b)    = pack + queue_cpu + exact_cpu(b) + owner_reduce
T_local(b)  = pack + queue_gpu + exact_gpu(b) + owner_reduce
T_remote(b) = pack + relay_out(b) + queue_gpu + exact_gpu(b)
              + relay_back(b) + owner_reduce
T_stream(b) = H2D(13,236,480 B) + decode/repack + exact_gpu(b) + synchronization
```

**Verified:** decode activation relay is tens of microseconds, versus about
4.57 ms for the current approved exact M=1 CPU matrix core. One expert weight
H2D alone costs 1.66–1.69 ms before decode/repack/kernel and repeats under
weight streaming; resident static ownership is therefore the justified default.

**Verified:** current scalar prefill can dominate: the 63-token profile spent
about 142.2 s in gate/up and 70.9 s in down. Actual buckets cluster around
`M=7..33`, not 64 uniformly.

**Unknown:** an exact resident selected-expert CUDA kernel does not yet exist, so
same-GPU versus relayed-GPU crossover cannot be reduced to a trustworthy numeric
threshold. The transfer terms and current CPU control are established; exact
GPU kernel/packing/event costs remain an explicit planning gate. CPU/GPU
concurrency is useful where independent work exists, but is not required in
every layer or token.
