.PHONY: build build-cuda check check-cuda test test-cuda kernels bench docker smoke loc clean

# Local development (Mac, mock-gpu)
build:
	cargo build --release -p gpt-oss-server

# CUDA build (Linux + NVIDIA GPU)
build-cuda:
	cargo build --release --features cuda -p gpt-oss-server

# Check workspace compiles (mock-gpu, Mac)
check:
	cargo check --workspace

# Check workspace compiles with CUDA features (needs cudarc, OK to fail without CUDA toolkit)
check-cuda:
	cargo check --workspace --features gpt-oss-server/cuda

# Compile .cu kernels to .ptx (requires nvcc)
kernels:
	bash kernels/build.sh

# Run all tests
test:
	cargo test --workspace

# Run tests with CUDA features
test-cuda:
	cargo test --workspace --features gpt-oss-server/cuda

# Run benchmarks (Rust)
bench:
	cargo bench --package gpt-oss-bench --bench sampling_bench

# Build Docker image
docker:
	bash scripts/build-docker.sh

# Run a basic HTTP smoke test against a local server
smoke:
	bash scripts/smoke_test.sh

# Count lines of code
loc:
	@find crates -name "*.rs" | xargs wc -l | tail -1
	@echo "CUDA kernels:"
	@find kernels -name "*.cu" 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 || echo "  0 lines"

# Clean
clean:
	cargo clean
