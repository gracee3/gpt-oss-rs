# gpt-oss-rs

This repository is a narrowed fork of `rvllm` focused only on serving OpenAI's GPT-OSS checkpoints from Rust. Multi-model architecture support, embedding-model support, and unrelated research collateral have been removed so the codebase can evolve around one target instead of a generic matrix.

The crate and binary names are still `rvllm*` for now. This cleanup keeps the existing workspace structure intact while removing non-GPT-OSS implementation paths.

## Scope

- Supported architecture: `GptOssForCausalLM`
- Supported serving mode: text generation via the OpenAI-compatible API
- Supported endpoints: `/v1/completions`, `/v1/chat/completions`, `/v1/responses`, `/v1/models`, `/health`, `/metrics`, and batch routes
- Removed from this fork: non-GPT-OSS architectures, `/v1/embeddings`, vision-model planning docs, and stale experiment directories

## Quick start

```bash
# CPU / mock backend
cargo build --release -p rvllm-server

# CUDA backend
cargo build --release --features cuda -p rvllm-server
```

```bash
./target/release/rvllm serve --model openai/gpt-oss-20b
```

For 24 GB consumer GPUs, the server defaults to the GPT-OSS profile:

- `max_model_len=8192`
- `gpu_memory_utilization=0.90`

Override them explicitly if needed:

```bash
./target/release/rvllm serve \
  --model openai/gpt-oss-20b \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

## API example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role":"user","content":"Explain mixture-of-experts routing."}],
    "max_tokens": 200
  }'
```

## Repository notes

- The CUDA engine now fails fast on non-`GptOssForCausalLM` checkpoints.
- The CPU/mock model runner only instantiates `GptOssForCausalLM`.
- Some naming still reflects the original upstream (`rvllm`, `rvllm-server`, etc.). If you want a full rename pass next, that should be done as a separate change because it touches crates, package metadata, and CLI behavior.
