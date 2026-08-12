#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <kernel.cl> <output.spv>" >&2
    exit 2
fi

kernel_source=$1
spirv_output=$2
xe_corpus=${XE_RESEARCH_CORPUS:-/home/emmy/src/xe-research}
xe_tools="$xe_corpus/toolchain/sysroot/usr/bin"
xe_libraries="$xe_corpus/toolchain/sysroot/usr/lib/x86_64-linux-gnu"
clang_bin=${CLANG_18:-/usr/bin/clang-18}
translator="$xe_tools/llvm-spirv-18"
validator="$xe_tools/spirv-val"
disassembler="$xe_tools/spirv-dis"

for executable in "$clang_bin" "$translator" "$validator" "$disassembler"; do
    if [[ ! -x "$executable" ]]; then
        echo "required offline tool is unavailable: $executable" >&2
        exit 1
    fi
done

output_dir=$(dirname "$spirv_output")
mkdir -p "$output_dir"
bitcode_output="${spirv_output%.spv}.bc"
disassembly_output="${spirv_output%.spv}.spvasm"

"$clang_bin" \
    -cc1 \
    -triple spir64-unknown-unknown \
    -cl-std=CL3.0 \
    -finclude-default-header \
    -O2 \
    -emit-llvm-bc \
    "$kernel_source" \
    -o "$bitcode_output"
LD_LIBRARY_PATH="$xe_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$translator" \
    -spirv-max-version=1.2 \
    "$bitcode_output" \
    -o "$spirv_output"
LD_LIBRARY_PATH="$xe_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$validator" --target-env opencl2.2 "$spirv_output"
LD_LIBRARY_PATH="$xe_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$disassembler" "$spirv_output" -o "$disassembly_output"

sha256sum "$kernel_source" "$bitcode_output" "$spirv_output" "$disassembly_output"
