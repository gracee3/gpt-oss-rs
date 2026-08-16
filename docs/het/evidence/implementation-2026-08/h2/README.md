# H2 gate record — exact selected-expert CUDA `M=1`

**Status:** passed locally on 2026-08-16. The package start commit is
`64c83e63421c0254bec7843787acf95257635cbf` on
`agent/het-implementation`; the commit introducing this record is the H2
completion commit. H3 was not started before this gate closed.

## Result

**Proven:** the new selected-expert primitive executes one already-selected
resident expert directly from its 13,236,480-byte native MXFP4
blocks/scales/bias representation. It does not route, scan all experts, call
the existing CUDA MoE path, use cuBLAS, or construct an FP16 expert matrix.
Unsupported phases fail before enqueue.

The final synthetic fixture passed bit-exact BF16 comparisons at gate/up,
scaled-gate, sigmoid, GLU, linear, SwiGLU, and down/output boundaries on both
local RTX 3090s. The real layer-0 fixture independently ran retained-route
experts `[31, 21, 22, 6]` on each GPU and matched the exact CPU semantic
reference at every recorded boundary. All repeated outputs were identical.

`real-expert-oracle.json` is the bounded raw record. It binds the executable,
PTX, model config/index, retained trace, individual expert payloads, stable PCI
device identities, route ranks, boundary hashes, event-completion times, and
staged device-memory samples. `resolution.json` summarizes the gate and its
repair history separately from those raw results.

## Preserved failed attempt and arithmetic repair

`failure.json` is retained unchanged. The first exact synthetic attempt found
the earliest divergence at `gate_up[4]` for E8M0 scale byte `0xff`:

```text
CPU expected: 0xffc0 (negative quiet NaN)
CUDA actual:  0x7fff (canonical positive NaN)
```

A wider diagnostic found the corresponding positive-payload case at the next
row (`0x7fc0` versus `0x7fff`). The cause was CUDA's native BF16 conversion
canonicalizing NaNs while `half` 2.7.1 preserves the f32 sign/high payload and
sets the quiet bit. Invalid finite operations on the CPU also produce the x86
negative-indefinite payload. The repair implements the exact CPU conversion
and ordered f32 add/multiply NaN rules inside the new kernel. It did not relax
bit exactness, add a tolerance, or special-case the fixture output.

## Enqueue, drain, and borrow lifetime

Validation and the injected `SubmitBeforeEnqueue` fault return before any CUDA
enqueue. Beginning with the input H2D attempt, every submit error path now
synchronizes the primitive's private stream before releasing executor,
weights, input, scratch, or caller result-slot borrows. If both the primary
operation and mandatory drain fail, the returned error retains both causes in
deterministic order.

The deterministic `SubmitAfterInputEnqueue` fault proves this post-enqueue
path. The test observes a successful mandatory drain, immediately reuses the
same resources, and obtains the same exact output. Submitted work retains a
terminal event and all borrows until `drain`, `cancel`, or the defensive drop
path has synchronized. The injected drain failure remains a publication
failure and does not expose a result.

## Allocation and high-water evidence

The primitive has no device allocation during submit or drain. Its logical
device work is:

| Allocation class | Bytes |
|---|---:|
| BF16 input | 5,760 |
| Gate/up plus SwiGLU scratch | 17,280 |
| Caller-owned BF16 result slot | 5,760 |
| First-divergence trace buffers | 23,040 |
| **Logical total** | **51,840** |
| Declared bounded workspace pool class | **65,536** |

On each recorded route, free-device-memory samples were taken before executor
construction, after executor construction, after the result slot, after the
13,236,480-byte expert upload, after first execution, after two repeats, and
after explicit teardown. The CUDA asynchronous allocator reserved 33,554,432
bytes at executor construction. Result-slot allocation, expert upload, and all
executions caused no additional visible reservation on this allocator. Every
route returned exactly to its pre-executor free-byte sample after teardown.
This measured allocator reservation is distinct from the 51,840-byte logical
work set and from the resident expert payload.

Recorded kernel-completion intervals for the eight final oracle routes ranged
from 3.897344 ms to 4.704256 ms. These are correctness-fixture measurements,
not an H10 performance claim or placement threshold.

## Gate commands

All commands ran from `/home/emmy/gpt-oss-rs` with the repository lockfile,
Rust 1.97.1, CUDA 13.3 (`sm_86`), and NVIDIA driver 610.43.02.

| Command or check | Result |
|---|---|
| `CUDA_ARCH=sm_86 cargo test -q -p gpt-oss-model-runner --features cuda,heterogeneous-test-faults --test selected_expert_cuda --release --locked -- --nocapture` | Passed on both GPUs, including exact special values, unsupported-shape/identity/route/device checks, cancellation, pre/post-enqueue and drain faults, reuse, and repeat output |
| `GPT_OSS_RUN_SELECTED_EXPERT_REAL=1 GPT_OSS_TEST_MODEL=/data/models/openai/gpt-oss-20b GPT_OSS_SELECTED_EXPERT_TRACE=/home/emmy/src/het-research/results/control-20b/retained-result.json CUDA_ARCH=sm_86 cargo test -q -p gpt-oss-model-runner --features cuda --test selected_expert_cuda_real --release --locked -- --nocapture` | Passed all four real experts on both GPUs at every boundary |
| `target/release/heterogeneous_expert_oracle --model /data/models/openai/gpt-oss-20b --retained-trace /home/emmy/src/het-research/results/control-20b/retained-result.json --output docs/het/evidence/implementation-2026-08/h2/real-expert-oracle.json` | `exact=true`; eight expert/device records, exact repeats, no run-time allocation growth, exact teardown recovery |
| `cargo check --workspace --locked` | Passed |
| `cargo test --workspace --locked` | Passed |
| Three configured strict Clippy lanes from the phase plan | Passed |
| H2 CUDA oracle and integration-test non-strict Clippy inspection | Passed; no warning originated in H2 files after the final refactor |
| `CUDA_ARCH=sm_86 cargo build --release --locked --features cuda` | Passed; 30 `sm_86` PTX modules including `gpt_oss_selected_expert.ptx` |
| Python unit discovery | 35 benchmark-tool and 10 oracle tests passed |
| `cargo fmt --all -- --check`, `git diff --check`, Markdown links, and scope/fallback audit | Passed at package close |

The release build and exploratory broad model-runner Clippy retained the
pre-existing experimental model-runner/engine warning inventory. The latter
is not a configured strict repository lane; no unrelated warning was changed
or suppressed.

## Scope and safety

No full model load, end-to-end generation, H3 construction, 120B execution,
model download/copy/transform, NCCL, P2P, or existing all-expert CUDA fallback
was used. At the final capture, host swap use was zero and both GPUs were back
at idle driver allocations. `/dev/nvme1n1` remained `RO=1` and unmounted. No
remote Git state changed.
