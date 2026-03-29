#!/bin/bash
# Publish gpt-oss-rs crates to crates.io in dependency order
set -euo pipefail

CRATES=(
    gpt-oss-core
    gpt-oss-config
    gpt-oss-gpu
    gpt-oss-memory
    gpt-oss-sequence
    gpt-oss-tokenizer
    gpt-oss-telemetry
    gpt-oss-block-manager
    gpt-oss-kv-cache
    gpt-oss-attention
    gpt-oss-model-loader
    gpt-oss-quant
    gpt-oss-sampling
    gpt-oss-model-runner
    gpt-oss-scheduler
    gpt-oss-worker
    gpt-oss-executor
    gpt-oss-speculative
    gpt-oss-engine
    gpt-oss-api
    gpt-oss-server
)

for crate in "${CRATES[@]}"; do
    echo "Publishing $crate..."
    cargo publish -p "$crate" --allow-dirty
    sleep 10  # crates.io rate limit
done
