#![allow(dead_code)]

use std::ffi::{c_char, c_void};

pub const XE_TEXT: usize = 1024;
pub const XE_EXTENSIONS: usize = 16_384;
pub const XE_BACKEND_OPENCL: u32 = 1;
pub const XE_BACKEND_LEVEL_ZERO: u32 = 2;
pub const XE_ARTIFACT_OPENCL_SOURCE: u32 = 1;
pub const XE_ARTIFACT_SPIRV: u32 = 2;
pub const XE_ARTIFACT_NATIVE: u32 = 3;
pub const XE_ARTIFACT_OPENCL_BINARY: u32 = 4;
pub const XE_MEMORY_DEVICE: u32 = 1;
pub const XE_MEMORY_HOST: u32 = 2;
pub const XE_MEMORY_SHARED: u32 = 3;
pub const XE_MEMORY_MAPPED: u32 = 4;

#[repr(C)]
#[derive(Clone)]
pub struct XeSessionInfo {
    pub status: i32,
    pub backend: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version: u32,
    pub compute_units: u32,
    pub max_group_size: u32,
    pub subgroup_count: u32,
    pub subgroups: [u32; 8],
    pub timestamp_valid_bits: u32,
    pub kernel_timestamp_valid_bits: u32,
    pub timer_resolution: u64,
    pub global_memory_bytes: u64,
    pub max_allocation_bytes: u64,
    pub local_memory_bytes: u64,
    pub creation_ns: u64,
    pub native_binary_bytes: u64,
    pub integrated: u8,
    pub compiler_available: u8,
    pub il_supported: u8,
    pub integer_dot_supported: u8,
    pub host_clock_correlation_supported: u8,
    pub immediate: u8,
    pub library_path: [c_char; XE_TEXT],
    pub platform_name: [c_char; XE_TEXT],
    pub device_name: [c_char; XE_TEXT],
    pub driver_version: [c_char; XE_TEXT],
    pub device_version: [c_char; XE_TEXT],
    pub extensions: [c_char; XE_EXTENSIONS],
    pub build_log: [c_char; XE_EXTENSIONS],
    pub error: [c_char; XE_TEXT],
}

impl Default for XeSessionInfo {
    fn default() -> Self {
        // SAFETY: this C-compatible record has no invalid zero bit patterns.
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone)]
pub struct XeRunTiming {
    pub status: i32,
    pub host_ns: u64,
    pub device_ns: u64,
    pub error: [c_char; XE_TEXT],
}

impl Default for XeRunTiming {
    fn default() -> Self {
        // SAFETY: this C-compatible record has no invalid zero bit patterns.
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone)]
pub struct XeMemoryTiming {
    pub status: i32,
    pub allocation_ns: u64,
    pub first_write_ns: u64,
    pub read_ns: u64,
    pub reuse_write_ns: u64,
    pub cleanup_ns: u64,
    pub error: [c_char; XE_TEXT],
}

impl Default for XeMemoryTiming {
    fn default() -> Self {
        // SAFETY: this C-compatible record has no invalid zero bit patterns.
        unsafe { std::mem::zeroed() }
    }
}

unsafe extern "C" {
    pub fn xe_session_create(
        backend: u32,
        expected_vendor: u32,
        expected_device: u32,
        artifact_kind: u32,
        artifact: *const u8,
        artifact_len: usize,
        build_options: *const c_char,
        entry_point: *const c_char,
        immediate: u8,
        info: *mut XeSessionInfo,
    ) -> *mut c_void;
    pub fn xe_session_probe(
        backend: u32,
        expected_vendor: u32,
        expected_device: u32,
        immediate: u8,
        info: *mut XeSessionInfo,
    ) -> i32;
    pub fn xe_session_native_binary(
        session: *mut c_void,
        bytes: *mut *mut u8,
        length: *mut usize,
        error: *mut c_char,
        error_len: usize,
    ) -> i32;
    pub fn xe_bytes_free(bytes: *mut u8);
    pub fn xe_buffer_create(
        session: *mut c_void,
        kind: u32,
        size: usize,
        timing: *mut XeRunTiming,
    ) -> *mut c_void;
    pub fn xe_buffer_write(
        session: *mut c_void,
        buffer: *mut c_void,
        source: *const c_void,
        size: usize,
        timing: *mut XeRunTiming,
    ) -> i32;
    pub fn xe_buffer_read(
        session: *mut c_void,
        buffer: *mut c_void,
        destination: *mut c_void,
        size: usize,
        timing: *mut XeRunTiming,
    ) -> i32;
    pub fn xe_kernel_arg_buffer(
        session: *mut c_void,
        index: u32,
        buffer: *mut c_void,
        error: *mut c_char,
        error_len: usize,
    ) -> i32;
    pub fn xe_kernel_arg_scalar(
        session: *mut c_void,
        index: u32,
        value: *const c_void,
        size: usize,
        error: *mut c_char,
        error_len: usize,
    ) -> i32;
    pub fn xe_kernel_group_size(
        session: *mut c_void,
        x: u32,
        y: u32,
        z: u32,
        error: *mut c_char,
        error_len: usize,
    ) -> i32;
    pub fn xe_kernel_run(
        session: *mut c_void,
        global_x: usize,
        global_y: usize,
        global_z: usize,
        local_x: usize,
        local_y: usize,
        local_z: usize,
        timeout_ns: u64,
        timing: *mut XeRunTiming,
    ) -> i32;
    pub fn xe_buffer_destroy(
        session: *mut c_void,
        buffer: *mut c_void,
        timing: *mut XeRunTiming,
    ) -> i32;
    pub fn xe_session_destroy(session: *mut c_void);
    pub fn xe_memory_roundtrip(
        backend: u32,
        expected_vendor: u32,
        expected_device: u32,
        kind: u32,
        size: usize,
        immediate: u8,
        timing: *mut XeMemoryTiming,
    ) -> i32;
}
