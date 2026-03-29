#!/bin/bash
set -euo pipefail
echo "Building gpt-oss-rs wheel..."
maturin build --release
echo "Publishing to PyPI..."
maturin publish
