#!/bin/bash
# Run API compatibility tests against both servers
set -euo pipefail
echo "Testing gpt-oss-rs API compatibility..."
echo "Server: ${GPT_OSS_RS_URL:-http://localhost:8000}"
pip install -q requests pytest
python3 -m pytest tests/api_compat/ -v --tb=short
