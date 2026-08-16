# Host and model baseline

All host facts below are sanitized. No hostname, network address, GPU/drive
serial or UUID, filesystem UUID, token, or credential is retained.

## Command record

**Verified by read-only local commands between
2026-08-15T20:40:28-04:00 and 2026-08-15T20:48:47-04:00:**

| Command family | Concise result |
|---|---|
| `uname -r`; `/etc/os-release`; compiler/tool version commands | OS, kernel, Rust, CUDA, C/C++, linker, CMake, Python, Docker versions below |
| `lscpu`; `numactl --hardware`; cpufreq and affinity sysfs/proc reads | One-socket/one-NUMA topology, cache, flags, governor and allowed CPUs below |
| `/proc/meminfo`; `free`; readable EDAC sysfs | Point-in-time memory and six populated 16 GiB channels; configured transfer rate unavailable |
| sanitized `nvidia-smi` queries, `topo -m`, `topo -p2p`, NVLink status; `lspci -tv`; PCI sysfs | Two RTX 3090s, VRAM, bus/topology/link/peer facts below; identifiers omitted |
| `findmnt`, `df`, `lsblk` without UUID/serial columns | Model filesystem is read-write ext4 with 790 GiB available; protected NVMe remains read-only/unmounted |
| `find`/`stat`; JSON parsing; first-eight-byte SafeTensors header reads | Local checkpoint files, indexes, tensor metadata and totals below; no tensor payload was materialized |
| SHA-256 of small configs/indexes/tokenizer/manifest metadata | Small-file identities below; no large shard was hashed |

No benchmark was run. Current link state and free memory are transient capture
facts, not performance measurements.

## Software and toolchain

| Component | Captured value |
|---|---|
| OS | Ubuntu 26.04 LTS |
| Kernel | `7.0.0-29-generic` |
| Rust | `rustc 1.97.1 (8bab26f4f 2026-07-14)`, LLVM 22.1.6; Cargo 1.97.1; stable `x86_64-unknown-linux-gnu` |
| CUDA toolkit | 13.3, `nvcc` 13.3.73 |
| NVIDIA driver | 610.43.02; driver-visible compute capability 8.6 on both GPUs |
| C/C++ | GCC/G++ 15.2.0 |
| Linker / CMake | GNU ld 2.46; CMake 4.2.3 |
| Python | System Python 3.14.4; project `.venv` absent |
| Containers | Docker 29.7.2; Compose 5.4.0 |

**Verified:** the authoritative official CPU oracle remains pinned to Python
3.12.12/PyTorch 2.12.1 inside its immutable image. The host's Python 3.14 is
not a substitute. No local `.venv` is needed for routine work, and no oracle
image was built in this stage.

## CPU, NUMA, and host memory

| Fact | Captured value | Interpretation |
|---|---|---|
| CPU | Intel Xeon Silver 4215R, family 6 model 85 stepping 7 | Identity only; policy must use capability flags. |
| Sockets / cores / threads | 1 / 8 / 16; SMT 2 threads/core | All 16 logical CPUs online and allowed to this shell. |
| NUMA | 1 node; CPUs 0-15; approximately 95,017 MB node size | Both GPUs also report node 0. No inter-socket NUMA case exists here. |
| Cache | L1d 256 KiB total (8), L1i 256 KiB total (8), L2 8 MiB total (8), L3 11 MiB shared | `lscpu` aggregate/instance report. |
| Relevant ISA | AVX2, FMA, AVX-512F/DQ/CD/BW/VL, AVX-512 VNNI | Current automatic CPU path can select AVX-512 VNNI. |
| Absent ISA flags | AMX tile/INT8/BF16 and AVX-512 BF16 are not exposed | AMX feature-gated build evidence cannot execute on this CPU. |
| Frequency driver/governor | `intel_cpufreq` / `schedutil`; allowed range 1.0-4.0 GHz; instantaneous CPU0 about 2.60 GHz | Point-in-time state, not a fixed clock. |
| Affinity | CPUs 0-15; memory node 0 available | No taskset/cgroup restriction observed. |
| Installed memory | 97,298,424 kB (`/proc/meminfo`), about 92 GiB visible | Nominal 96 GiB. |
| Population | Six readable EDAC entries, each 16,384 MiB unbuffered DDR4 x4; one DIMM in each of six channel positions | Six-channel population is visible; module identity and configured rate are not. |
| Configured memory speed | **Unknown** | Unprivileged DMI access was denied; no readable speed source/tool existed. |
| Measured bandwidth | **Unknown / Deferred** | No bandwidth benchmark was authorized in this stage. |
| Capture free memory | 66,672,204 kB free; 93,973,044 kB available; swap 99 GiB total and unused | Transient baseline only. |

## GPU and interconnect topology

| Fact | GPU0 | GPU1 |
|---|---:|---:|
| Model | RTX 3090 | RTX 3090 |
| VRAM total | 24,576 MiB | 24,576 MiB |
| VRAM used / free at capture | 9 / 24,112 MiB | 1 / 24,123 MiB |
| PCI bus location | `0000:19:00.0` | `0000:65:00.0` |
| NUMA node / CPU affinity | node 0 / CPUs 0-15 | node 0 / CPUs 0-15 |
| Current link | Gen1 x16 (idle/downclocked) | Gen1 x16 (idle/downclocked) |
| Device capability / host ceiling | device Gen4; host-visible maximum Gen3 x16 | device Gen4; host-visible maximum Gen3 x16 |

**Verified:** the PCI tree places the GPUs below separate host-bridge branches.
`nvidia-smi topo -m` labels their relationship `NODE`: traffic crosses PCIe
and host bridges within the same NUMA node. This is not a direct GPU link.

**Verified:** NVLink status reports all links inactive. The driver's topology
peer query reports GPU-to-GPU PCIe peer access `NS` (not supported); related
read/write queries report `GNS` (GPU not supported). No CUDA
`cudaDeviceCanAccessPeer`/enable/copy program was run, so an operational peer
copy is **not tested** independently. The runtime contains no peer-copy path in
any case.

**Not measured:** CPU-to-GPU, GPU-to-GPU or NVLink bandwidth/latency; loaded
PCIe generation; copy/compute overlap; GPU memory bandwidth. Product-sheet
numbers are deliberately not substituted for measurements.

## Filesystem and protected storage

**Verified:** repository and model paths are on the root ext4 filesystem,
mounted read-write. At capture it was 55% used with about 790 GiB available.
This is sufficient for metadata work, not yet a proof that every possible 120B
execution representation and scratch plan fits.

**Verified:** `/dev/nvme1n1` is a 1.8 TiB disk with `RO=1`, has no mountpoint,
and `findmnt` reports no mount sourced from it. It was not mounted, opened for
write, checked, benchmarked, or otherwise modified.

## Model package overview

| Fact | GPT-OSS-20B | GPT-OSS-120B |
|---|---:|---:|
| Canonical local package | `/data/models/openai/gpt-oss-20b` | `/data/models/openai/gpt-oss-120b` |
| Weight snapshot inspected | package root (transformed HF layout) | `original/` (native/original layout) |
| Revision evidence | Git HEAD `6cee5e81ee83917806bbde320786a8fb61efebee` | `REVISION`: `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` |
| Layers / experts / top-k | 24 / 32 / 4 | 36 / 128 / 4 |
| Hidden / intermediate | 2,880 / 2,880 | 2,880 / 2,880 |
| Attention | 64 query heads, 8 KV heads, head dimension 64 | same |
| Context fields | initial 4,096; sliding window 128; RoPE theta 150,000; scaling factor 32 | same native fields |
| Vocabulary | 201,088 | 201,088 in native config |
| SafeTensors shards | 3 transformed shards | 7 original shards |
| Logical tensors | 459 | 543 |
| Indexed tensor payload | 13,761,264,768 bytes | 65,248,815,744 bytes |
| Physical shard-file bytes | 13,761,316,904 | 65,248,869,800 |
| Dtype payload | 363 BF16 tensors / 3,608,919,168 B; 96 U8 / 10,152,345,600 B | 399 BF16 / 4,334,742,144 B; 144 U8 / 60,914,073,600 B |
| Current-runtime loadability | Metadata contract matches | Does not match current loader contract |

### 20B manifest and assets

**Verified:** root `model.safetensors.index.json` maps all 459 tensors to these
three existing files and header enumeration found no missing or extra tensor:

```text
model-00000-of-00002.safetensors  4,792,272,488 bytes
model-00001-of-00002.safetensors  4,798,702,184 bytes
model-00002-of-00002.safetensors  4,170,342,232 bytes
```

The `00000..00002-of-00002` spelling is unusual but internally consistent with
the index. Fourteen top-level files total 13,789,264,674 bytes and include
`config.json`, the model index, `tokenizer.json`, tokenizer configuration,
special-token map, `chat_template.jinja`, generation config, license/readme and
policy files. Intended `original/` and `metal/` variants also exist below the
package root; the CPU/GPU production loaders enumerate only top-level shards.

Small-file SHA-256 identities:

| File | SHA-256 |
|---|---|
| `config.json` | `3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce` |
| `model.safetensors.index.json` | `0e085b977c4c9942f85938828e8c989ed7d5cdabf852e4da6a67c116cd502cd1` |
| `tokenizer.json` | `0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3` |

The transformed config declares `GptOssForCausalLM`, `quant_method=mxfp4`, and
the exact layer types/YaRN fields expected by `CpuGptOssConfig`. Its tensor
names (`model.layers.*`, `model.embed_tokens.weight`, `lm_head.weight`) and
BF16/U8 shapes match both current production loaders.

### 120B manifest and assets

**Verified:** the package has a `DOWNLOAD_COMPLETE` marker dated
2026-08-15T14:25:01-04:00, a revision file, and `SHA256SUMS`. The checksum
manifest lists exactly the ten files in `original/`; all ten exist and no
unlisted file is present there. The seven index-referenced shards exist, and
header enumeration matches all 543 index tensors with no extras.

The seven shard sizes are 10,544,040,680; 10,488,721,680; 10,488,721,688;
10,488,721,672; 10,488,721,680; 10,433,402,600; and 2,316,539,800 bytes.
No large shard checksum was recomputed, so the stored SHA-256 values are
proven present, not revalidated in this stage.

Small-file SHA-256 identities:

| File | SHA-256 |
|---|---|
| `original/config.json` | `063aaa98e012cac330d58d18471700f0b0eb90bb15c58fb227809d45090f3e65` |
| `original/model.safetensors.index.json` | `40ffac1e58f77c5307f20990cdb89910ad86961ed556952feda8aaa1c0d5ac97` |
| `REVISION` file | `1ce60ef7f313e9867434420ba27ffc75eeb92153be0be03cdef56ececa61c6f8` |
| `SHA256SUMS` | `c7d67204d133750e1216cbd42c17002ae8c6ec03474992c5cd9ed35aef05d1df` |

**Verified, with distinct scopes:** this is an internally complete original
checkpoint at the metadata/file level. It is not a complete current-runtime
snapshot. The package root has no `config.json`,
top-level shards, tokenizer or Harmony assets. `original/config.json` uses
native keys such as `num_experts` and omits the architecture/layer-type/YaRN
object required by `CpuGptOssConfig`; tensors use `block.N.*`, `embedding`,
`norm`, and `unembedding`, not transformed `model.layers.N.*` names. Pointing
the runtime at either directory therefore fails its existing contract before a
valid full load. No conversion or copy is authorized in this stage.

## Exact checkpoint expert-storage derivation

Both checkpoints use hidden size `H=2,880`, expert intermediate size `I=2,880`,
and MXFP4 groups of 32 values, hence `G=90` groups per row. Each group stores 16
packed U8 bytes plus one U8 scale. Biases are BF16. There is no alignment or
SafeTensors-header allowance in the payload calculation.

| Per-expert item | Shape / width | Calculation | Bytes |
|---|---|---:|---:|
| Gate/up blocks | `[2I,G,16]` U8 | `5,760 * 90 * 16` | 8,294,400 |
| Gate/up scales | `[2I,G]` U8 | `5,760 * 90` | 518,400 |
| Gate/up bias | `[2I]` BF16 | `5,760 * 2` | 11,520 |
| Down blocks | `[H,G,16]` U8 | `2,880 * 90 * 16` | 4,147,200 |
| Down scales | `[H,G]` U8 | `2,880 * 90` | 259,200 |
| Down bias | `[H]` BF16 | `2,880 * 2` | 5,760 |
| **One expert total** | | | **13,236,480 B = 12.6233 MiB** |

The blocks+scales portion alone is 13,219,200 bytes/expert and maps exactly to
the 17-byte CPU repack record. Consequences:

| Checkpoint | Experts/layer | Expert bytes/layer | All expert bytes |
|---|---:|---:|---:|
| 20B | 32 | 423,567,360 B = 403.9453 MiB | 10,165,616,640 B = 9.4675 GiB across 24 layers |
| 120B | 128 | 1,694,269,440 B = 1,615.7813 MiB | 60,993,699,840 B = 56.8048 GiB across 36 layers |

Router BF16 weight+bias is `E*H*2 + E*2`: 184,384 bytes/layer for 20B and
737,536 bytes/layer for 120B. Dense attention, embeddings, unembedding and
norms account for the remainder of indexed payload.

## Activation and transfer-size facts

These are arithmetic lower bounds, not a selected transfer protocol. No
headers, indices, padding, allocator alignment, retry copy, router transfer,
or synchronization cost is included.

For one token:

| Value | BF16 bytes | f32 bytes |
|---|---:|---:|
| Hidden/expert input (`H`) | 5,760 | 11,520 |
| Gate/up result (`2I`) | 11,520 | 23,040 |
| SwiGLU result (`I`) | 5,760 | 11,520 |
| One expert output (`H`) | 5,760 | 11,520 |
| Router logits, 20B / 120B | 64 / 256 | 128 / 512 |

Top-4 device IDs occupy 16 bytes as `i32` in CUDA; current CPU `usize` IDs
occupy 32 bytes on this host. Four f32 route weights occupy 16 bytes.

If all four selected expert inputs and outputs crossed a device boundary in
BF16, the lower bound is `2 * T * 4 * 2,880 * 2 = 46,080*T` bytes:

| Routed rows `T` | Inputs only | Outputs only | Both directions |
|---:|---:|---:|---:|
| 1 decode row | 23,040 B | 23,040 B | 46,080 B |
| 128 prefill rows | 2,949,120 B | 2,949,120 B | 5,898,240 B |
| 512 prefill rows | 11,796,480 B | 11,796,480 B | 23,592,960 B |
| 2,048 prefill rows | 47,185,920 B | 47,185,920 B | 94,371,840 B |

Current CUDA prefill instead transfers one f32 normalized input and one f32 MoE
output per token (`23,040*T` bytes total), because all experts execute on the
host rather than scattering top-4 activations independently. Transfer timing
and effective bandwidth remain unmeasured.

## Persistent, execution, and temporary representations

| Layer | Current representation | Proven byte behavior |
|---|---|---|
| Checkpoint/on disk | BF16 dense tensors; U8 MXFP4 blocks/scales; BF16 biases | Exact indexed payload above. |
| CPU persistent | All source shards mmap-retained; dense matrices borrowed; norms/biases copied f32; full blocks+scales repacked to another 17-byte-record mmap | The MXFP4 source and repack each virtually cover 10,152,345,600 B (20B) or 60,914,073,600 B (120B). Physical RSS/demand paging is unknown. |
| CPU execution | BF16 hidden/KV, Q8 or residual-Q8 activation blocks, caller-owned matrix scratch, f32 accumulators | Descriptor/high-water machinery exists; full-model values on this host are not captured. |
| CUDA load/persistent | BF16 dense converts to device f32 and f16 during load; U8 first owns one host map, then a second per-layer host clone, then a device copy for decode; biases/router also have host/device copies | Multiple full U8 representations are statically proven. Some f32 projections are pruned only after f16 decode setup; load high-water is unknown. |
| CUDA decode temporary | Per expert: full gate/up f16 matrix 33,177,600 B; full down f16 16,588,800 B; dense masked rows and intermediates | Allocation sites are proven; simultaneous pool-retained high-water is unknown. |
| CUDA TP load | Complete maps are loaded per rank, then downloaded/sliced/reuploaded | Not a bounded direct-placement load; peak and correctness unknown. |

Thus “mmap loader” must not be summarized as “no duplicate execution
representation.” CPU avoids dense tensor copies but adds full MXFP4 repacks;
CUDA maps shards sequentially but materializes multiple owned forms.

## Exact-host model validation status

| Gate | 20B | 120B |
|---|---|---|
| Metadata/index/header inspection | **Verified pass** for transformed snapshot | **Verified pass** for original package; current-runtime metadata contract fails |
| Current loader on this host | **Unknown:** not run in this assignment; latest sanity explicitly did not run full load | **Not runnable as stored** by static contract mismatch; no load attempted |
| One-layer official checkpoint gate | **Unknown on this exact host:** the test is opt-in and no retained local artifact exists | Not run |
| Retained-continuation parity | **Unknown on this exact host** | Not run |
| End-to-end generation | **Unknown on this exact host; supplied sanity says no 20B smoke was run** | Not run; prohibited at this stage |

Repository campaign documents prove important 20B CPU behavior in their pinned
environments, but `/home/emmy/gpt-oss-rs-artifacts` is absent and no retained
record ties those runs to this host. They cannot answer the exact-host question.

## Completeness conclusion

- **20B:** appears complete at both metadata and current-runtime snapshot level.
  This is not a load/generation proof.
- **120B:** appears complete as the original checkpoint package at the
  metadata/file-manifest level, without rehashing shard payloads. It is
  incomplete for the current runtime because the required transformed snapshot
  and tokenizer/protocol assets are absent.
