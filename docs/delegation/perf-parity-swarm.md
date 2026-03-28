# Performance Parity Swarm -- Close the 2x Gap

## Current state
- rvLLM: 86 tok/s, Python vLLM: 200 tok/s
- Gap is from: no continuous batching, no CUDA graphs, naive CPU prefill attention
- FA2 decode kernel already does 1 launch per layer (same as Python vLLM)

## Three workstreams, 16 agents

### Stream A: CUDA Graph Capture (Agents 1-4)
Capture the decode step kernel sequence into a CUDA graph and replay it.

### Stream B: Fix FA2 Prefill Kernel (Agents 5-10)
Fix the causal mask bug, remove the naive CPU prefill fallback.

### Stream C: Continuous Batching (Agents 11-14)
Batch multiple sequences into a single forward() call.

### Stream D: Cleanup + README + Paper (Agents 15-16)
Remove debug probes, update README with real bench numbers, update paper.

---

## Agent 1: Read CUDA graph infrastructure
- READ: crates/rvllm-gpu/src/cuda_graph.rs
- READ: crates/rvllm-worker/src/graph_runner.rs (if exists)
- OUTPUT: what exists, what's missing to wire graph capture to GpuModelRunner

## Agent 2: Design CUDA graph capture for decode
- READ: Agent 1 output, crates/rvllm-model-runner/src/gpu_runner.rs (forward path)
- PROPOSE: where to insert begin_capture/end_capture around the decode step
- DO NOT EDIT files

## Agent 3: Implement CUDA graph capture in gpu_runner.rs
- EDIT: crates/rvllm-model-runner/src/gpu_runner.rs
- Add: capture first decode step, replay on subsequent steps
- Use: CudaGraphPool from cuda_graph.rs
- Gate behind: runtime check (only capture after warmup)

## Agent 4: Test CUDA graph capture
- EDIT: nothing (read-only verification)
- Verify Agent 3's changes compile and the graph API is used correctly

## Agent 5: Analyze FA2 prefill causal mask
- READ: kernels/flash_attention.cu (prefill kernel, lines 140-340)
- FIND: exact causal mask condition and whether it's correct for single-sequence prefill
- OUTPUT: the bug and the fix

## Agent 6: Fix FA2 prefill causal mask
- EDIT: kernels/flash_attention.cu (prefill kernel only)
- Fix the causal mask to correctly handle single-sequence multi-token prefill
- DO NOT touch the decode kernel

## Agent 7: Fix FA2 prefill output write indexing
- READ: kernels/flash_attention.cu (prefill kernel, output write at line ~333)
- VERIFY: q_global_pos indexing is correct for all query tokens
- If wrong, fix it

## Agent 8: Wire FA2 prefill back into gpu_layer.rs
- EDIT: crates/rvllm-model-runner/src/gpu_layer.rs
- Change: replace naive_prefill_attention dispatch with prefill_attention (FA2)
- Keep naive_prefill_attention as dead code fallback

## Agent 9: Test FA2 prefill on A100
- ACCESS: ssh -p 16806 root@ssh4.vast.ai
- Build and test with FA2 prefill enabled
- Send 5-token prompt, verify output is coherent
- Report pass/fail

## Agent 10: Remove naive prefill if FA2 works
- EDIT: crates/rvllm-model-runner/src/gpu_layer.rs
- Remove or gate the naive_prefill_attention behind a feature flag
- Clean up

## Agent 11: Analyze continuous batching requirements
- READ: crates/rvllm-engine/src/gpu_engine.rs (scheduler, build_metadata)
- READ: crates/rvllm-worker/src/input.rs (prepare_prefill, prepare_decode)
- OUTPUT: what changes needed to batch N sequences in one forward() call

## Agent 12: Implement multi-sequence decode batching
- EDIT: crates/rvllm-engine/src/gpu_engine.rs
- Change: schedule multiple sequences per step instead of one
- The scheduler already has budget limits (max_num_seqs, max_num_batched_tokens)

## Agent 13: Update input preparation for batched decode
- EDIT: crates/rvllm-worker/src/input.rs
- Verify prepare_decode handles multiple sequences correctly
- Verify block_tables flattening works for N > 1

## Agent 14: Test continuous batching
- ACCESS: ssh -p 16806 root@ssh4.vast.ai
- Send 4 concurrent requests
- Verify all produce coherent output
- Measure throughput improvement

## Agent 15: Remove debug probes + update README
- EDIT: crates/rvllm-model-runner/src/gpu_runner.rs (remove probes)
- EDIT: README.md (update benchmark table with real numbers)
- Bench numbers: rvLLM 86 tok/s, Python vLLM 200 tok/s, startup 8s vs 121s, RSS 348MB vs 1033MB, binary 16MB

## Agent 16: Update paper + GitHub Pages
- EDIT: docs/paper/rvllm.tex, docs/paper/rvllm-bw.tex (update benchmark tables)
- EDIT: docs/index.html (update benchmark section)
- Use verified numbers only

## Rules
1. Agents in same stream coordinate (A reads before B writes)
2. No agent touches files outside its assignment
3. Remote server access only for agents 9, 14
4. All edits must compile (cargo check)
5. No fabricated numbers
