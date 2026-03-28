# Ideal Forward Model (target)

Same model (Qwen2.5-1.5B f16), same hardware (A100 80GB).
This is what the forward path should look like after all 6 agents.

## Per-Step Outer Loop

```
1. prepare_input()           -- CPU: reuse scratch Vecs (no alloc)
2. upload_metadata()         -- 1x memcpy_htod (packed buffer, all 6 fields)
3. graph.replay()            -- single CUDA graph launch
4. read_graph_output()       -- 1x memcpy_dtoh (i32 token IDs)
5. sample_tokens()           -- CPU: extract token ID
```

Changes from current:
- 6 HtoD -> 1 HtoD (Agent 3)
- prepare_input reuses buffers (Agent 5)

## Per-Layer Operations (forward_f16, x28 layers)

Each `forward_mixed` call does: alloc output + cublasGemmEx = **1 alloc, 0 cast kernels, 1 cuBLAS call**.

```
 #  Operation                    Kernels  Allocs  Memcpy   Change from current
--- ----------------------------  -------  ------  ------  --------------------
 1  fused_residual_rmsnorm         1       2       0      WAS: rms_norm (1 kern, 1 alloc). Now fused with step 16 of PREV layer. First layer uses plain rms_norm.
 2  Q = forward_mixed(normed)      1       1       0      WAS: 3 kern, 3 alloc. Eliminated 2 cast kernels + 2 allocs.
 3  K = forward_mixed(normed)      1       1       0      same
 4  V = forward_mixed(normed)      1       1       0      same
 5  add_bias(Q)                    1       0       0      unchanged
 6  add_bias(K)                    1       0       0      unchanged
 7  add_bias(V)                    1       0       0      unchanged
 8  RoPE: rotary_kernel(Q,K)       1       0       0      WAS: 2 alloc + 2 dtod + 1 kern. Now IN-PLACE on Q,K.
 9  cache_write(k, v)              1       0       0      unchanged
10  decode_attention(q)            1       1       0      unchanged (FA2 f16kv)
11  O = forward_mixed(attn)        1       1       0      WAS: 3 kern, 3 alloc
12  fused_residual_rmsnorm         1       2       0      WAS: add(1k,1a) + rms_norm(1k,1a) = 2 kern, 2 alloc. Now 1 kern, 2 alloc. Returns (normed, residual).
13  gate = forward_mixed(normed2)  1       1       0      WAS: 3 kern, 3 alloc
14  up = forward_mixed(normed2)    1       1       0      WAS: 3 kern, 3 alloc
15  fused = silu_mul(gate, up)     1       1       0      unchanged
16  down = forward_mixed(fused)    1       1       0      WAS: 3 kern, 3 alloc
    (residual add folded into next layer's fused_residual_rmsnorm)
--- ----------------------------  -------  ------  ------
    TOTAL PER LAYER:                17      13      0
    TOTAL 28 LAYERS:               476     364      0
```

## Post-Layer

```
17  final_rms_norm                   1       1       0
18  fused_lm_head_argmax_f16         2       3       0
19  dtod to graph_output             0       0       1
--- ----------------------------  -------  ------  ------
    TOTAL FORWARD:                 479     368      1
```

## Comparison: Current vs Ideal

| Metric | Current | Ideal | Reduction |
|--------|---------|-------|-----------|
| Kernel launches | 899 | 479 | **-47%** |
| GPU allocs | 816 | 368 | **-55%** |
| Device-to-device memcpy | 57 | 1 | **-98%** |
| HtoD per step | 6 | 1 | **-83%** |
| Cast kernels | 392 | 0 | **-100%** |
| cuBLAS calls | 196 | 196 | same |

## Estimated Impact

At 0.5-1us per kernel launch + 0.5-1us per alloc:
- 420 fewer kernel launches: ~210-420us saved
- 448 fewer allocs: ~224-448us saved
- 56 fewer dtod memcpy: ~56us saved
- 5 fewer HtoD: ~50-100us saved
- **Total estimated savings: ~540-1024us per token**

Current: 7.7ms/tok. Target: ~6.5-7.0ms/tok from these changes alone.
With engine loop optimization (Agent 4) + input prep (Agent 5): target ~5-6ms/tok.
With pre-allocated scratch (Agent 6): target ~4-5ms/tok.

## Agent Mapping

Each row shows which agent implements the change:

| Change | Agent | Files |
|--------|-------|-------|
| forward_mixed (eliminate casts) | Agent 1 | cublas.rs, linear_cuda.rs |
| Wire up forward_mixed + fused ops + in-place RoPE | Agent 2 | gpu_layer.rs |
| Packed metadata upload | Agent 3 | gpu_runner.rs |
| Engine loop optimization | Agent 4 | async_gpu_engine.rs, gpu_engine.rs |
| Input prep buffer reuse | Agent 5 | input.rs, gpu_worker.rs |
| Pre-allocated scratch buffers | Agent 6 | gpu_runner.rs, gpu_worker.rs |

## Merge Order

```
1. Agent 1 (cublas.rs, linear_cuda.rs)      -- new API, no callers yet
2. Agent 3 (gpu_runner.rs upload_metadata)   -- internal refactor, same interface
3. Agent 4 (engine loop)                     -- independent crate
4. Agent 5 (input prep)                      -- independent path
5. Agent 2 (gpu_layer.rs)                    -- calls Agent 1 API, biggest change
6. Agent 6 (scratch buffers)                 -- builds on Agent 2's forward path
```
