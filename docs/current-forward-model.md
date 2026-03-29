# Current Forward Model (as of d9cd982)

Exact operation sequence for ONE decode step, ONE token, Qwen2.5-1.5B f16.
Reference: `crates/gpt-oss-model-runner/src/gpu_layer.rs` forward_f16().

## Per-Step Outer Loop (gpu_worker.rs)

```
1. prepare_input()           -- CPU: build ModelInput from metadata
2. upload_metadata()         -- 6x memcpy_htod (token_ids, positions, context_lens, block_tables, slot_mapping, seq_start_pos)
3. graph.replay()            -- if graph captured, else forward_gpu_only()
4. read_graph_output()       -- 1x memcpy_dtoh (i32 token IDs)
5. sample_tokens()           -- CPU: extract token ID from array
```

## Per-Layer Operations (forward_f16, x28 layers)

Each `forward_once_f16` call does: cast_f32_to_f16 + alloc + hgemm + cast_f16_to_f32 + alloc = **3 allocs, 2 cast kernels, 1 cuBLAS call**.

```
 #  Operation                    Kernels  Allocs  Memcpy   Source
--- ----------------------------  -------  ------  ------  --------
 1  rms_norm(input)                  1       1       0     gpu_layer.rs:158
 2  Q = forward_once_f16(normed)     3       3       0     gpu_layer.rs:172  (cast+hgemm+cast)
 3  K = forward_once_f16(normed)     3       3       0     gpu_layer.rs:175
 4  V = forward_once_f16(normed)     3       3       0     gpu_layer.rs:178
 5  add_bias(Q)                      1       0       0     gpu_layer.rs:184  (Qwen2.5 only)
 6  add_bias(K)                      1       0       0     gpu_layer.rs:187
 7  add_bias(V)                      1       0       0     gpu_layer.rs:189
 8  RoPE: alloc q_out                0       1       0     gpu_layer.rs:671
 9  RoPE: alloc k_out                0       1       0     gpu_layer.rs:674
10  RoPE: dtod q -> q_out            0       0       1     gpu_layer.rs:678
11  RoPE: dtod k -> k_out            0       0       1     gpu_layer.rs:681
12  RoPE: rotary_kernel              1       0       0     gpu_layer.rs:703
13  cache_write(k_rot, v)            1       0       0     gpu_layer.rs:209
14  decode_attention(q_rot)          1       1       0     gpu_layer.rs:241  (FA2 f16kv)
15  O = forward_once_f16(attn)       3       3       0     gpu_layer.rs:260
16  residual = add(input, O)         1       1       0     gpu_layer.rs:265
17  normed2 = rms_norm(residual)     1       1       0     gpu_layer.rs:274
18  gate = forward_once_f16(norm2)   3       3       0     gpu_layer.rs:285
19  up = forward_once_f16(norm2)     3       3       0     gpu_layer.rs:288
20  fused = silu_mul(gate, up)       1       1       0     gpu_layer.rs:291
21  down = forward_once_f16(fused)   3       3       0     gpu_layer.rs:292
22  output = add(residual, down)     1       1       0     gpu_layer.rs:297
--- ----------------------------  -------  ------  ------
    TOTAL PER LAYER:                 32      29      2
    TOTAL 28 LAYERS:                896     812     56
```

## Post-Layer (gpu_runner.rs forward_gpu_only)

```
23  final_rms_norm                   1       1       0
24  fused_lm_head_argmax_f16         2       3       0     (pass1 + reduce)
25  dtod to graph_output             0       0       1
--- ----------------------------  -------  ------  ------
    TOTAL FORWARD:                 899     816     57
```

## Summary

| Metric | Count |
|--------|-------|
| Kernel launches | 899 |
| GPU memory allocations | 816 |
| Device-to-device memcpy | 57 |
| Host-to-device memcpy (metadata) | 6 |
| Device-to-host memcpy (output) | 1 |
| cuBLAS GEMM calls | 196 (7 per layer x 28) |
| Cast kernels (f32<->f16) | 392 (14 per layer x 28) |

With CUDA graph replay: the 899 kernel launches + 816 allocs inside forward_gpu_only
are captured as one graph. But the 6 HtoD metadata uploads + 1 DtoH output read
happen outside the graph every step.
