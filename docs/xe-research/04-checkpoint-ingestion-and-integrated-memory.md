# X4 — Checkpoint Ingestion and Integrated Memory

- Result: pass for the bounded research slice
- Policy: persistent compact weights or bounded compact slab, reusable scratch
- Rejected: a second model-scale representation, general allocator, or LRU

## Allocation experiment

The sweep tested 4 KiB, 64 KiB, 1 MiB, 16 MiB, 128 MiB, and
`min(512 MiB, 10% of MemAvailable)`. The cap resolved to 512 MiB. Each size and
class received thirty independent allocation/write/read/visibility/reuse/
cleanup repetitions:

- OpenCL ordinary device buffer, host-backed buffer, mapped host buffer, and
  coarse-grained SVM;
- Level Zero device, host, and shared allocations.

Every one of the 720 OpenCL and 540 Level Zero roundtrips returned success and
byte-identical visibility. The OpenCL path uses actual map/unmap for the mapped
class and SVM map/unmap for the shared class. The Level Zero path explicitly
allocates each selected class. An allocation 4 KiB above the queried
4,294,959,104-byte maximum failed cleanly with OpenCL `-61` and Level Zero
`2013265929`; host OOM was not induced.

At 128 MiB, representative median phase timings in nanoseconds were:

| API/class | allocate | first write | read | warm write | cleanup |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenCL device | 15,770,301 | 17,474,688 | 13,609,215 | 13,824,108 | 500,504 |
| OpenCL mapped | 15,782,576 | 17,415,276 | 41,169,803 | 13,705,482 | 495,241 |
| OpenCL coarse SVM | 87,216 | 32,656,287 | 41,157,489 | 13,724,806 | 488,529 |
| Level Zero device | 45,891 | 40,018,944 | 18,696,919 | 24,083,829 | 7,820 |
| Level Zero host | 15,697,043 | 24,414,107 | 19,384,335 | 24,260,304 | 35,517 |
| Level Zero shared | 15,823,745 | 24,256,023 | 18,433,064 | 24,142,809 | 36,751 |

These are visibility/synchronization observations, not proof of physical
zero-copy. The integrated GPU shares DDR bandwidth with the CPU. A 128 MiB CPU
copy alone had a 13.42 ms OpenCL-run median, while the same CPU copy begun with
a concurrent OpenCL device roundtrip had a 60.32 ms median. The corresponding
Level Zero contention median was 62.23 ms. This establishes material shared
memory contention and argues against staging on the critical path.

Peak RSS/high-water marks reached approximately 1.10–1.14 GiB during the
512 MiB cases because the research validator intentionally holds source and
destination host buffers while the tested allocation exists. That is measured
temporary duplication, not a proposed serving policy.

## Real checkpoint ingestion

The checkpoint shard is opened read-only and memory-mapped. SafeTensors shapes
and dtypes are validated before expert slicing. The exact selected bundle is:

| Tensor | Full shape | Selected bytes |
| --- | --- | ---: |
| `model.layers.0.mlp.experts.gate_up_proj_blocks` | `[32,5760,90,16]` U8 | 8,294,400 |
| `model.layers.0.mlp.experts.gate_up_proj_scales` | `[32,5760,90]` U8 | 518,400 |
| `model.layers.0.mlp.experts.gate_up_proj_bias` | `[32,5760]` BF16 | 11,520 source bytes |

This is layer 0, expert 0, `N=5760`, `K=2880`, with 90 K=32 blocks per
output. Only the selected slices are copied; the shard remains read-only. The
canonical compact research representation is 8,835,840 bytes including FP32
bias. The CPU `InterleavedSplitX8V2` representation is also 8,835,840 bytes and
is retained only for CPU baseline measurement. No GPU-derived layout was
created.

## Residency decision

If a future forced experiment is authorized, use canonical compact weights as
one persistent selected region or a fixed-size layer/expert slab. Allocate one
reusable activation/output scratch set sized to the bounded request. Populate
and validate residency before publishing the model attachment. No second
model-scale x8/GPU cache may coexist as part of this lane; CPU fallback must
retain access to the canonical checkpoint and may derive bounded CPU work
locally under the existing CPU policy.

Per-tensor allocation is rejected because it multiplies lifecycle state.
Whole-model persistent residency is not proven by one slice and is rejected.
Direct checkpoint mapping as a GPU buffer and zero-copy claims are rejected
because neither was demonstrated. Page advice, prefetch, and registration
remain unused rather than inferred.

## Evidence records

| ID | Manifest SHA-256 | Raw memory/checkpoint SHA-256 |
| --- | --- | --- |
| X4-OCL | `09b656628b650fe2456fcedf1d16a4125ff37c08b28d29f91e9387464dd8af4e` | `3636ec50e79eefba52c519685b21d37a1659fc6b603e2f8e03517d5fd003e3b4` |
| X4-L0 | `5e9b6c9081958367276da6ea22db0aa95b544e8f2e90a2cd31ed0bd9436d411d` | `941105a8432c1cd31f6f8f46fee94680abf6d7092262e54b2c7d6062f9527bcd` |

Both records are `pass` and report repository revision
`09014c0f82304f72507a7ef23aa2206b5be09615`.
