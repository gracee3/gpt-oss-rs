// SPDX-License-Identifier: Apache-2.0
// Canonical X3 kernel. The exact source and generated SPIR-V are used by both
// host APIs; no fast-math option is permitted by the research harness.
__kernel void xe_i32_add(
    __global const int *left,
    __global const int *right,
    __global int *output,
    const uint count) {
    const size_t index = get_global_id(0);
    if (index < count) {
        output[index] = left[index] + right[index];
    }
}
