# Source ledger

All code checkouts are shallow or filtered and remain unmodified under
`~/src/sources`. Commit dates are informative; the commit hash is the identity.
License statements describe the inspected revision and do not authorize copying
code into this project.

## Pinned code and specifications

| Source | Pin / license | Question and inspected locations | Finding and applicability |
|---|---|---|---|
| [OpenAI `gpt-oss`](https://github.com/openai/gpt-oss) | tag `v0.0.9`, commit `599476783c6f88508dab8577808b5ead5cbee8d2`, Apache-2.0 | `gpt_oss/torch/{weights,model}.py`, `gpt_oss/triton/{moe,model}.py`, tokenizer/config code | **Verified:** official native names, MXFP4 block/scale interpretation, BF16 router/top-k/selected softmax and GPT SwiGLU. The reference reads the native checkpoint directly; Triton expands/transposes/requantizes for its backend. It is the semantic and format authority, not a host-fit executor. |
| [OpenAI GPT-OSS-120B model](https://huggingface.co/openai/gpt-oss-120b) | revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`; Apache-2.0 model release | Root config/index, file/LFS metadata, five small tokenizer/protocol assets; no weight payload | **Verified:** official transformed namespace has 687 tensors and the same payload total as local native 120B. Assets match local 20B. The revision is the remote side of the metadata map. |
| [OpenAI GPT-OSS-20B model](https://huggingface.co/openai/gpt-oss-20b) | local package revision `6cee5e81ee83917806bbde320786a8fb61efebee`; Apache-2.0 model release | Local native/runtime headers, configs, index, assets, every tensor payload | **Verified:** complete 363→459 mapping and byte equality; supplies the retained correctness control. No remote payload was fetched. |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | `eb9b70c4115cff151ace2cae5b0fc9db3690e31e`, Apache-2.0; submodules not initialized | `archive/ktransformers/operators/experts.py`, archived CPU-infer examples, GGUF loader/operator paths, build/readme | **Verified:** persistent pinned buffers and explicit submit/sync enable CPU/GPU overlap; GGUF expert-range mapping uses quant-block alignment. Whole-module backends and AMX-heavy assumptions do not supply an exact three-owner GPT-OSS seam on this non-AMX host. |
| [Fiddler](https://github.com/efeslab/fiddler) | `227715bfd6e8c731b29548eab01d9919c4fe9564`, Apache-2.0 | `src/fiddler/mixtral.py`, cost/partition logic | **Verified:** static hot/cold ownership and activation-not-weight movement are established techniques. The code is CUDA0-only, Mixtral/top-2-shaped, and executes GPU then CPU loops without the required transaction boundary. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a`, MIT | `src/models/openai-moe.cpp`, `src/llama-graph.cpp`, GGML MXFP4 kernels/layouts | **Verified:** selected-expert `mul_mat_id`, packed 17-byte MXFP4 blocks, x86 repack, and explicit rank-order weighted sum are credible patterns. Device mapping is principally layer/row/tensor split; the checkout does not prove native safetensors ingestion or static per-expert three-device ownership. |
| [NCCL](https://github.com/NVIDIA/nccl) | tag `v2.31.2-1`, commit `7b83616df3ae082a1f32bb74c27458bfe8153a13`, Apache-2.0 | source-built library, transport diagnostics, all-reduce harness | **Verified:** this device pair uses SHM/direct host transport when CUDA P2P is unavailable. NCCL all-reduce is correct in the harness but is a poor semantic fit for irregular selected-expert dispatch. |
| [mistral.rs](https://github.com/EricLB/mistral.rs) | `48257ce666cc9f6c7c6f24476b49f7d96ce9ed80`, MIT | GPT-OSS model and MXFP4 selected-expert paths | **Verified:** selected IDs/weights, selected-expert MXFP4 gather execution, and Rust device/error ownership provide useful API patterns. Its mapper is per layer, not proof of this exact contract or three-owner placement. |
| CUDA Runtime 13.3.1 [peer API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html) | retrieved 2026-08-15 | `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess` | **Verified:** access is directional; zero means direct peer memory access is unavailable. The local API results are recorded in the topology evidence. |
| NCCL [environment guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) and [P2P API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html) | retrieved 2026-08-15 | P2P/SHM transport meaning and synchronization | **Verified:** SHM is the host-memory fallback when P2P cannot occur. Vendor semantics support, but do not replace, the local transport trace. |

The external clean control checkout of this repository is detached at
`0113e8214e765d168216bbee2120654555a4cfe4`. It was used to ensure the real 20B
control and matrix benchmark did not build from the user-owned dirty tree.

### Exact inspected code locations

- OpenAI: `gpt_oss/torch/weights.py` (`PARAM_NAME_MAP`,
  `_get_mxfp4_tensor`, `WeightLoader`), `gpt_oss/torch/model.py`
  (`swiglu`, `MLPBlock.forward`), and `gpt_oss/triton/moe.py` grouped routing/
  GEMM plus the model-side expand/transpose/requantize calls.
- KTransformers: `archive/ktransformers/operators/experts.py` pinned-buffer
  construction around lines 271–291, `submit_for_one_decode` and
  `sync_for_one_decode` around 293–316, and grouped route execution around
  1,040–1,068. Archived CPU-infer MoE examples supplied the submit/sync contract.
- Fiddler: `src/fiddler/mixtral.py` layer routing and CPU/GPU expert loop around
  480–570, plus the prefill cost/partition continuation immediately following.
- llama.cpp: `src/llama-graph.cpp` selected-expert routing, `mul_mat_id`, and
  weighted reduction around 2,028–2,251; `src/models/openai-moe.cpp` supplied
  GPT-OSS graph/model construction; GGML CPU/CUDA quant sources supplied
  `block_mxfp4`/x86 repack behavior.
- mistral.rs: `mistralrs-core/src/models/gpt_oss.rs` `gptoss_swiglu` around
  113–124 and selected `gather_forward` route execution around 523–614;
  `mistralrs-quant` MXFP4 modules supplied the packed selected-expert API.
- NCCL: `src/transport/*`, graph/transport diagnostics, public headers, and the
  vendor SHM/P2P documentation were cross-checked with the local probe.

## Pinned papers

| Paper | Identity | Question answered | Limits |
|---|---|---|---|
| [Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models](https://arxiv.org/abs/2402.07033) | arXiv v3; ICLR 2025; local PDF SHA-256 `40c1502e...74c9` | Why move activations rather than weights; phase-specific cost models; static hot/cold placement. | Evaluated models/hardware and top-2 policy do not establish GPT-OSS exactness or no-P2P dual-GPU behavior. |
| [KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models](https://doi.org/10.1145/3731569.3764843) | SOSP 2025; local PDF SHA-256 `d3c5af7a...28f3` | Pinned buffers, asynchronous CPU/GPU scheduling, expert representations, NUMA/AMX assumptions. | Its dual 36-core Xeon 8452Y/1 TiB/AMX evaluation is not this 8-core non-AMX host; expert deferral is approximate and remains deferred. |
| [HybriMoE](https://arxiv.org/abs/2504.05897) | arXiv `2504.05897`, 2025-04-08; local PDF SHA-256 `7e20287b...c6bd` | Phase-varying expert utilization, cache/prefetch scheduling, and static-mapping load-imbalance risk. | Adaptive mechanisms are deferred; its model/hardware set does not validate GPT-OSS or two-GPU ownership. |

The inspected paper portions were Fiddler's system/cost-model and phase-policy
sections (including PDF page 5), KTransformers' architecture and asynchronous
CPU/GPU scheduling sections (including PDF page 6), and HybriMoE's motivation,
system overview, and evaluated-hardware summary (including PDF page 3). Page
numbers here are human-visible one-based PDF pages; no long source text is
reproduced.

**Conflict:** the initially queued identifier `2409.19934` resolved to an
unrelated kidney-stone paper, not KTransformers. It is not a technical source;
the disposable local PDF was removed after identification. The official
SOSP/ACM paper above resolves the source identity.

## Provenance boundary

No external source code was copied into `gpt-oss-rs`. The adopted terminology
and equations are generic research concepts: static expert ownership,
activation-versus-weight movement, selected-expert grouping, pinned staging,
and explicit commit barriers. Any later implementation must re-evaluate license
and provenance at the code level.
