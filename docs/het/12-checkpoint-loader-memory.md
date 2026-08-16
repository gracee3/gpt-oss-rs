# Checkpoint, loader, and memory viability

## Native-to-runtime format delta

The complete maps are retained as
[`checkpoint-map-20b-full.json`](evidence/research-2026-08/checkpoint-map-20b-full.json)
and
[`checkpoint-map-120b-metadata.json`](evidence/research-2026-08/checkpoint-map-120b-metadata.json).
The 120B comparison used the local native headers and the official transformed
index pinned to model revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`; it did not download transformed
weight payloads.

| Fact | 20B | 120B |
|---|---:|---:|
| Native shards / tensors | 1 / 363 | 7 / 543 |
| Runtime shards / tensors | 3 / 459 | 15 / 687 |
| Layers / experts per layer / top-k | 24 / 32 / 4 | 36 / 128 / 4 |
| Hidden / expert intermediate | 2,880 / 2,880 | 2,880 / 2,880 |
| Attention heads / KV heads / head dim | 64 / 8 / 64 | 64 / 8 / 64 |
| Vocabulary / initial context / max position | 201,088 / 4,096 / 131,072 | 201,088 / 4,096 / 131,072 |
| Sliding window / RoPE theta / YaRN factor | 128 / 150,000 / 32 | 128 / 150,000 / 32 |
| Layer pattern / SwiGLU limit | alternating sliding/full / 7 | alternating sliding/full / 7 |
| Native-to-runtime map | 459/459 | 687/687 |
| Payload proof | **Verified:** all 13,761,264,768 payload bytes equal | **Unknown:** payloads were not downloaded; names, shapes, dtypes, slices, and shard index are complete |

**Verified:** each native layer contains one QKV weight `[5120, 2880]` and
bias `[5120]`. The runtime namespace exposes contiguous rows as Q
`[4096, 2880]`, K `[512, 2880]`, and V `[512, 2880]`, with the same split for
biases. This adds four names per layer: `96 = 4 × 24` for 20B and
`144 = 4 × 36` for 120B. Every other mapped tensor is renamed without a payload
transformation.

The repeating namespace map is:

| Native | Runtime |
|---|---|
| `embedding.weight`, `norm.scale`, `unembedding.weight` | `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight` |
| `block.N.attn.norm.scale` | `model.layers.N.input_layernorm.weight` |
| `block.N.attn.qkv.{weight,bias}` | contiguous `self_attn.{q,k,v}_proj.{weight,bias}` slices |
| `block.N.attn.out.{weight,bias}` / `attn.sinks` | `self_attn.o_proj.{weight,bias}` / `self_attn.sinks` |
| `block.N.mlp.norm.scale` | `model.layers.N.post_attention_layernorm.weight` |
| `block.N.mlp.gate.{weight,bias}` | `model.layers.N.mlp.router.{weight,bias}` |
| `block.N.mlp.mlp1_{weight.{blocks,scales},bias}` | `model.layers.N.mlp.experts.gate_up_proj_{blocks,scales,bias}` |
| `block.N.mlp.mlp2_{weight.{blocks,scales},bias}` | `model.layers.N.mlp.experts.down_proj_{blocks,scales,bias}` |

This pattern plus the layer count enumerates the complete map; the retained JSON
lists every concrete name, shard, byte slice, shape, and dtype.

**Verified:** expert blocks and scales keep their native U8 bytes and shapes:
gate/up `[E, 5760, 90, 16]` plus scales `[E, 5760, 90]`, down
`[E, 2880, 90, 16]` plus scales `[E, 2880, 90]`. Biases are BF16. OpenAI's
`gpt_oss/torch/weights.py` treats each 16-byte block with one U8 exponent scale;
the current CPU loader repacks each `(16 block bytes, 1 scale byte)` into a
17-byte x8 record. No permutation, padding, transposition, or scale conversion
occurs in the snapshot transformation. Backend-specific expansion or
requantization is an execution representation and must not be confused with
checkpoint conversion.

| Asset | Result |
|---|---|
| `chat_template.jinja` | **Verified:** byte-identical between local 20B and pinned official 120B asset |
| `generation_config.json` | **Verified:** byte-identical |
| `special_tokens_map.json` | **Verified:** byte-identical |
| `tokenizer_config.json` | **Verified:** byte-identical |
| `tokenizer.json` | **Verified:** identical SHA-256; official 120B identity obtained from its LFS metadata |

**Inferred:** the 20B tokenizer/Harmony assets are semantically reusable for the
pinned 120B revision because every asset identity matches. **Verified:** the
local 120B native directory does not colocate them or a current-runtime index,
so it remains unloadable by the present loader without a bounded format/view
step.

### Completion and identity rules

**Verified:** the local 20B runtime index references all and only the tensors
found in its three top-level shards. The package contains the expected config,
index, tokenizer, special-token, chat template, generation, license/readme, and
policy assets. Its exact small-file identities remain in the
[host/model baseline](03-host-model-baseline.md#20b-manifest-and-assets).

**Verified:** local 120B has `DOWNLOAD_COMPLETE`, `REVISION`, and `SHA256SUMS`.
`REVISION` is exactly the official revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`; the checksum manifest lists exactly
the ten files in `original/`, all seven index shards exist, and shard headers
contain all and only 543 indexed tensors. Large stored hashes were not
recomputed. A viable loading family must bind revision, config/index hashes,
shard list/sizes, and any generated alias/repack manifest; a mere completion
marker is not a payload revalidation.

## Byte-exact storage

All units below are binary GiB (`bytes / 2^30`); file totals include safetensors
headers, while payload totals do not.

| Representation | 20B bytes (GiB) | 120B bytes (GiB) |
|---|---:|---:|
| Native payload | 13,761,264,768 (12.8162) | 65,248,815,744 (60.7677) |
| Expert checkpoint payload | 10,165,616,640 (9.4675) | 60,993,699,840 (56.8048) |
| Non-expert payload | 3,595,648,128 (3.3487) | 4,255,115,904 (3.9629) |
| Native file bytes | 13,761,300,984 | 65,248,869,800 |
| Runtime transformed file bytes | 13,761,316,904 | approximately native payload plus runtime headers; official index total is equivalent payload |

One checkpoint expert is exactly:

```text
gate/up blocks   = 5,760 × 90 × 16 × 1       = 8,294,400 B
gate/up scales   = 5,760 × 90 × 1            =   518,400 B
gate/up bias     = 5,760 × 2                  =    11,520 B
down blocks      = 2,880 × 90 × 16 × 1       = 4,147,200 B
down scales      = 2,880 × 90 × 1            =   259,200 B
down bias        = 2,880 × 2                  =     5,760 B
total                                            13,236,480 B
```

The CPU x8 execution representation is `17 × 90 × (5,760 + 2,880) =
13,219,200 B` per expert, excluding bias. Conservatively retaining f32 biases
adds `4 × (5,760 + 2,880) = 34,560 B`, so planning uses
**13,253,760 B per resident owner expert**. All 4,608 120B experts therefore
require `61,073,326,080 B = 56.8790 GiB` in this owner representation. A full
120B x8 repack without biases is `60,914,073,600 B = 56.7307 GiB`.

**Verified:** the 20B CPU control generated 48 x8 cache files totaling
10,152,372,268 bytes; the 26,668-byte difference from the theoretical payload
is container/header overhead.

## State and transfer-sized allocations

For BF16 K and V, checkpoint dimensions imply:

```text
120B KV/token = 36 layers × 2 × 8 KV heads × 64 × 2 B = 73,728 B
20B  KV/token = 24 layers × 2 × 8 KV heads × 64 × 2 B = 49,152 B
```

Thus 120B KV is 0.28125 GiB at 4,096 tokens, 2.25 GiB at 32,768, and
9.0 GiB at 131,072. These figures exclude block tables, allocator slack, and
request metadata. One BF16 activation row is `2,880 × 2 = 5,760 B`; top-4
destination packing is `23,040 B/token`, plus 16 bytes of i32 expert/rank IDs
and 16 bytes of f32 weights if all metadata is sent.

**Unknown:** exact allocator retention, CUDA context/module/cuBLAS workspace,
attention scratch, and prefill peak for a future exact CUDA path. They remain
inside explicit safety reserves, not silently assigned zero.

## Loading-family evaluation

| Family | Disk and restart | Host/VRAM ownership | Viability and decision evidence |
|---|---|---|---|
| Bounded offline runtime snapshot | Adds 65,248,893,184 bytes of transformed shards; all 26 official root files total 65,276,859,410 bytes (60.7938 GiB). A streamed transform can bound RAM. Restart uses the current mmap contract. | Still requires owner-selective CPU repacks and GPU uploads; the snapshot itself is not an execution expansion. | **Technically viable.** The complete mapping and 20B byte proof bound the transform. It duplicates disk and creates an artifact-integrity/restart surface without changing expert bytes. |
| Direct native loader with views | No duplicate model; restart remaps seven native shards. Q/K/V are slice views and all other tensors are aliases. | Upload GPU-owner packed bytes directly; build x8 only for CPU-owned experts; retain no unused alternate form. | **Technically viable and smallest on disk.** Requires validated shard/view lifetimes and owner-selective construction not present today. |
| Hybrid persistent representation | Native package stays authoritative; persist only CPU-owner x8 records and a versioned manifest. | Direct packed native bytes for GPU owners; x8 for CPU owners; common BF16/f32 bias policy; bounded staging. | **Surviving research preference.** It minimizes restart repack without retaining all native, all x8, and all device copies as resident allocations. The choice is a loading-family conclusion, not an implementation plan. |

At the final capture, the ext4 filesystem containing `/data` had
834,550,788,096 bytes available. The offline family therefore fits current disk
capacity, but its extra 65.28 GB remains a deliberate duplicate rather than a
free prerequisite.

Failure/restart identity differs by family. A bounded offline transform must
write temporary shards, validate every emitted slice/name/size, and publish its
config/index/completion manifest only after all files succeed; an interrupted
temporary snapshot is discardable. Direct-native construction has no derived
artifact to recover, but must rebuild/upload each owner representation after a
restart and discard a partially constructed owner before registration. Hybrid
x8 records need a manifest binding source revision, source tensor slice/hash,
repack version, owner/layout, byte count, and completion; incomplete or stale
records are individually rebuildable and must never be mixed with a new source
revision.

**Rejected:** a lazy current-style loader that faults the entire 60.77 GiB
native package while simultaneously retaining a full x8 repack, device copies,
and full FP16 expert expansions. Native payload plus the conservative x8 owner
form including f32 biases is `65,248,815,744 + 61,073,326,080 =
126,322,141,824 B = 117.647 GiB`, above the host's approximately 92.79 GiB
physical memory. A memory map is virtual and reclaimable; it is not permission
to count every mapped page as free or to touch all pages while building every
alternate representation.

### Family memory envelopes

The following exact known terms apply to every surviving family under the
4-GiB/all-dense-on-GPU0 example. Unknown allocator/context terms must fit inside
the stated reserves rather than be treated as zero.

| Family | Persistent disk added | Persistent host owner allocation | GPU0 / GPU1 expert allocation | Original mapping and construction peak |
|---|---:|---:|---:|---|
| Offline snapshot | 65,276,859,410 B including official root assets | 22,385,600,640 B for 1,689 conservative owner experts | 17,216,634,240 B / 21,471,091,200 B | Runtime snapshot mmap is 65,248,815,744 payload bytes virtual/reclaimable. A bounded transform needs an 8-MiB-style stream plus output page-cache control; exact transform peak is **Unknown** until measured. |
| Direct native views | zero model bytes | same 22,385,600,640 B, built only for CPU owners | same | Native mmap is 65,248,815,744 payload bytes virtual/reclaimable; per-expert staging lower bound is 13,236,480 B. View/shard lifetime and current upload-copy peak are **Unknown**. |
| Hybrid native + persisted CPU x8 | 22,327,228,800 B of x8 records for 1,689 CPU experts, plus bounded container/manifest overhead | same conservative 22,385,600,640 B including f32 biases | same | Restart maps native plus CPU-owner x8. It must avoid simultaneously faulting both complete mappings; cold peak and page-cache eviction are **Unknown**. |

All three also place 4,255,115,904 bytes of non-expert payload, assumed on GPU0
for this envelope, and reserve 4,294,967,296 bytes per GPU for KV, scratch,
staging, CUDA context/modules, allocator retention, fragmentation, and safety.
The reserve is justified as an admission bound, not a measured component split.

## 120B placement envelopes

Each GPU has 25,769,803,776 bytes (24 GiB). The table uses the conservative
13,253,760-byte expert owner form. “Reserve” covers CUDA context, modules,
attention/KV/scratch, staging, fragmentation, and safety; it is not a measured
breakdown. Dense placement uses the exact 4,255,115,904-byte non-expert payload.

| Envelope | GPU0 experts / bytes | GPU1 experts / bytes | CPU experts / bytes | Meaning |
|---|---:|---:|---:|---|
| Arithmetic best case; all dense on GPU0; no reserve | 1,623 / 20.0335 GiB | 1,944 / 23.9958 GiB | 1,041 / 12.8496 GiB | **Unsafe bound:** demonstrates aggregate capacity only. |
| 4 GiB reserve/GPU; all dense on GPU0 | 1,299 / 16.0342 GiB | 1,620 / 19.9965 GiB | 1,689 / 20.8482 GiB | **Viable static envelope** at short/moderate context if native pages are demand-evictable and alternates are not retained. |
| 6 GiB reserve/GPU; all dense on GPU0 | 1,137 / 14.0346 GiB | 1,458 / 17.9969 GiB | 2,013 / 24.8475 GiB | **More conservative GPU envelope;** leaves CPU ample physical headroom. |
| 4 GiB reserve/GPU; dense payload split evenly | 1,459 / 18.0092 GiB | 1,459 / 18.0092 GiB | 1,690 / 20.8606 GiB | **Capacity comparison only:** splitting every dense tensor is not a selected architecture. |

At the 4 GiB/all-dense-on-GPU0 envelope, owner-resident expert bytes plus dense
payload are approximately 60.84 GiB across physical memories. Adding 32K 120B
KV gives about 63.09 GiB before the two 4 GiB GPU reserves. The host-resident
expert share is about 20.85 GiB, leaving substantial host RAM for reclaimable
page cache and bounded staging. At the maximum 131K context, KV alone is 9 GiB
and the 4 GiB reserve assumption is invalid; placement/context admission must
be recomputed.

**Conclusion:** 120B is physically viable only with single-owner expert
residency, bounded shard access, no full alternate copy, and explicit context
and runtime reserves. This establishes a safe design space; it does not prove
the future loader's peak. A planning gate must measure cold construction and
steady residency against the selected envelope.
