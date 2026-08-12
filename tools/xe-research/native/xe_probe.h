#ifndef GPT_OSS_XE_PROBE_H
#define GPT_OSS_XE_PROBE_H

#include <stddef.h>
#include <stdint.h>

#define XE_TEXT 1024
#define XE_EXTENSIONS 16384

enum xe_backend {
    XE_BACKEND_OPENCL = 1,
    XE_BACKEND_LEVEL_ZERO = 2,
};

enum xe_artifact_kind {
    XE_ARTIFACT_OPENCL_SOURCE = 1,
    XE_ARTIFACT_SPIRV = 2,
    XE_ARTIFACT_NATIVE = 3,
    XE_ARTIFACT_OPENCL_BINARY = 4,
};

enum xe_memory_kind {
    XE_MEMORY_DEVICE = 1,
    XE_MEMORY_HOST = 2,
    XE_MEMORY_SHARED = 3,
    XE_MEMORY_MAPPED = 4,
};

struct xe_session_info {
    int32_t status;
    uint32_t backend;
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t api_version;
    uint32_t compute_units;
    uint32_t max_group_size;
    uint32_t subgroup_count;
    uint32_t subgroups[8];
    uint32_t timestamp_valid_bits;
    uint32_t kernel_timestamp_valid_bits;
    uint64_t timer_resolution;
    uint64_t global_memory_bytes;
    uint64_t max_allocation_bytes;
    uint64_t local_memory_bytes;
    uint64_t creation_ns;
    uint64_t native_binary_bytes;
    uint8_t integrated;
    uint8_t compiler_available;
    uint8_t il_supported;
    uint8_t integer_dot_supported;
    uint8_t host_clock_correlation_supported;
    uint8_t immediate;
    char library_path[XE_TEXT];
    char platform_name[XE_TEXT];
    char device_name[XE_TEXT];
    char driver_version[XE_TEXT];
    char device_version[XE_TEXT];
    char extensions[XE_EXTENSIONS];
    char build_log[XE_EXTENSIONS];
    char error[XE_TEXT];
};

struct xe_run_timing {
    int32_t status;
    uint64_t host_ns;
    uint64_t submit_ns;
    uint64_t wait_ns;
    uint64_t device_ns;
    char error[XE_TEXT];
};

struct xe_memory_timing {
    int32_t status;
    uint64_t allocation_ns;
    uint64_t first_write_ns;
    uint64_t read_ns;
    uint64_t reuse_write_ns;
    uint64_t cleanup_ns;
    char error[XE_TEXT];
};

void *xe_session_create(
    uint32_t backend,
    uint32_t expected_vendor,
    uint32_t expected_device,
    uint32_t artifact_kind,
    const uint8_t *artifact,
    size_t artifact_len,
    const char *build_options,
    const char *entry_point,
    uint8_t immediate,
    struct xe_session_info *info);

int32_t xe_session_probe(
    uint32_t backend,
    uint32_t expected_vendor,
    uint32_t expected_device,
    uint8_t immediate,
    struct xe_session_info *info);

int32_t xe_session_native_binary(
    void *session,
    uint8_t **bytes,
    size_t *length,
    char *error,
    size_t error_len);

void xe_bytes_free(uint8_t *bytes);

int32_t xe_session_select_kernel(
    void *session,
    const char *entry_point,
    char *error,
    size_t error_len);

void *xe_buffer_create(
    void *session,
    uint32_t kind,
    size_t size,
    struct xe_run_timing *timing);

int32_t xe_buffer_write(
    void *session,
    void *buffer,
    const void *source,
    size_t size,
    struct xe_run_timing *timing);

int32_t xe_buffer_read(
    void *session,
    void *buffer,
    void *destination,
    size_t size,
    struct xe_run_timing *timing);

int32_t xe_kernel_arg_buffer(
    void *session,
    uint32_t index,
    void *buffer,
    char *error,
    size_t error_len);

int32_t xe_kernel_arg_scalar(
    void *session,
    uint32_t index,
    const void *value,
    size_t size,
    char *error,
    size_t error_len);

int32_t xe_kernel_group_size(
    void *session,
    uint32_t x,
    uint32_t y,
    uint32_t z,
    char *error,
    size_t error_len);

int32_t xe_kernel_run(
    void *session,
    size_t global_x,
    size_t global_y,
    size_t global_z,
    size_t local_x,
    size_t local_y,
    size_t local_z,
    uint64_t timeout_ns,
    struct xe_run_timing *timing);

int32_t xe_buffer_destroy(
    void *session,
    void *buffer,
    struct xe_run_timing *timing);

void xe_session_destroy(void *session);

int32_t xe_memory_roundtrip(
    uint32_t backend,
    uint32_t expected_vendor,
    uint32_t expected_device,
    uint32_t kind,
    size_t size,
    uint8_t immediate,
    struct xe_memory_timing *timing);

#endif
