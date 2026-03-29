# Multi-stage build for gpt-oss-rs on CUDA
# Stage 1: Build the Rust binary with CUDA support
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 AS builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install Rust toolchain
RUN apt-get update && apt-get install -y curl build-essential pkg-config libssl-dev && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ARG CUDA_ARCH
ENV CUDA_ARCH=${CUDA_ARCH}

WORKDIR /build
COPY . .

# Build with CUDA support, release mode
RUN cargo build --release --features cuda -p gpt-oss-server 2>&1 | tail -20 && \
    ptx_dir="$(find target/release/build -type d -path '*/out/ptx' -print -quit)" && \
    test -n "$ptx_dir" && \
    find "$ptx_dir" -maxdepth 1 -name '*.ptx' | grep -q . && \
    mkdir -p /tmp/gpt-oss-kernels && \
    cp "$ptx_dir"/*.ptx /tmp/gpt-oss-kernels/

# Stage 2: Runtime image (smaller)
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y libssl3t64 ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/gpt-oss-rs /usr/local/bin/gpt-oss-rs
COPY --from=builder /tmp/gpt-oss-kernels/*.ptx /usr/local/share/gpt-oss-rs/kernels/

# Default port
EXPOSE 8000
# Metrics port
EXPOSE 9090

ENV GPT_OSS_RS_KERNEL_DIR=/usr/local/share/gpt-oss-rs/kernels
ENV RUST_LOG=info

ENTRYPOINT ["gpt-oss-rs"]
CMD ["serve", "--model", "/models/default", "--host", "0.0.0.0", "--port", "8000"]
