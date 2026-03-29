#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building rvllm Docker image..."
echo "  CUDA 13.0 + Rust (release mode)"
echo "  CUDA kernels compiled inside the builder image"
if [ -n "${CUDA_ARCH:-}" ]; then
    echo "  Target arch: $CUDA_ARCH"
fi

cd "$ROOT_DIR"
docker build \
    --build-arg CUDA_ARCH="${CUDA_ARCH:-}" \
    -t rvllm:latest \
    -f Dockerfile \
    .

echo ""
echo "Build complete: rvllm:latest"
echo ""
echo "Run with:"
echo "  docker run --gpus all -p 8000:8000 -v /path/to/models:/models rvllm:latest serve --model /models/your-model"
