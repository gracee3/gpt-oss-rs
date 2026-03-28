# Benchmark History

All results on A100 80GB SXM4, Qwen2.5-1.5B, greedy decoding, 32 tokens/request unless noted.

## Phase 4 (2026-03-28) -- CUDA graph capture fix

Measured with concurrent Python HTTP requests after graph capture fix.

| N | tok/s | ms/tok | Notes |
|---|---|---|---|
| 1 | 128 | 7.7 | 22.7% mem BW utilization |
| 4 | 540 | - | |
| 8 | 1,091 | - | |
| 16 | 2,118 | - | |
| 32 | 3,467 | - | |

Per-token overhead: 5.95ms (77% of total), theoretical peak 574 tok/s.

## Phase 3 (earlier) -- Sampling + attention backend

Previous head-to-head numbers (measured with bench/run.sh batched harness, not reproducible with current code):

| N | rvLLM (tok/s) | vLLM 0.18 (tok/s) |
|---|---|---|
| 1 | 117 | 69 |
| 4 | 882 | 256 |
| 8 | 1,213 | 517 |
| 16 | 1,391 | 1,060 |
| 32 | 1,434 | 1,943 |
| 48 | 3,918 | 2,887 |
| 64 | 4,796 | 3,828 |
| 96 | 5,965 | 5,197 |
| 128 | 7,380 | 6,400 |
| 256 | 9,905 | 9,437 |
| 512 | 10,291 | 10,771 |
| 768 | 10,235 | -- |
| 1024 | 10,051 | 12,740 |

Note: These numbers included optimizations (fused QKV, fused gate+up, vectorized float4, packed HtoD, pre-alloc buffers) that were lost in subsequent code changes and are being re-implemented in Phase 5.

## Phase 2 -- FP16 inference

- 8,339 tok/s peak at N=768
- Matched vLLM at N=48-128

## Phase 1 -- FP32 baseline

- 3,191 tok/s peak at N=512
- 86 tok/s single-sequence

## B200 Results (FP32, earlier)

| N | Tokens | Wall time | tok/s |
|---|---|---|---|
| 1 | 32 | 279ms | 114 |
| 64 | 2,048 | 798ms | 2,566 |
| 256 | 8,192 | 2,106ms | 3,889 |
| 768 | 24,576 | 6,227ms | 3,946 |
| 4,096 | 131,072 | 34,002ms | 3,854 |
