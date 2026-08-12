use std::collections::BTreeSet;
use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt;
use std::path::PathBuf;
use std::ptr::NonNull;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::ffi;

pub const EXPECTED_VENDOR: u32 = 0x8086;
pub const EXPECTED_DEVICE: u32 = 0x9a49;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    Opencl,
    LevelZero,
}

impl Backend {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "opencl" => Ok(Self::Opencl),
            "level-zero" => Ok(Self::LevelZero),
            _ => bail!("backend must be exactly 'opencl' or 'level-zero'"),
        }
    }

    pub const fn ffi(self) -> u32 {
        match self {
            Self::Opencl => ffi::XE_BACKEND_OPENCL,
            Self::LevelZero => ffi::XE_BACKEND_LEVEL_ZERO,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Opencl => "opencl",
            Self::LevelZero => "level-zero",
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ArtifactKind {
    OpenclSource,
    Spirv,
    Native,
    OpenclBinary,
}

impl ArtifactKind {
    const fn ffi(self) -> u32 {
        match self {
            Self::OpenclSource => ffi::XE_ARTIFACT_OPENCL_SOURCE,
            Self::Spirv => ffi::XE_ARTIFACT_SPIRV,
            Self::Native => ffi::XE_ARTIFACT_NATIVE,
            Self::OpenclBinary => ffi::XE_ARTIFACT_OPENCL_BINARY,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryKind {
    Device,
    Host,
    Shared,
    Mapped,
}

impl MemoryKind {
    pub const fn ffi(self) -> u32 {
        match self {
            Self::Device => ffi::XE_MEMORY_DEVICE,
            Self::Host => ffi::XE_MEMORY_HOST,
            Self::Shared => ffi::XE_MEMORY_SHARED,
            Self::Mapped => ffi::XE_MEMORY_MAPPED,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub backend: Backend,
    pub vendor_id: String,
    pub device_id: String,
    pub api_version_raw: u32,
    pub compute_units: u32,
    pub max_group_size: u32,
    pub subgroup_sizes: Vec<u32>,
    pub timestamp_valid_bits: u32,
    pub kernel_timestamp_valid_bits: u32,
    pub timer_resolution: u64,
    pub global_memory_bytes: u64,
    pub max_allocation_bytes: u64,
    pub local_memory_bytes: u64,
    pub creation_ns: u64,
    pub integrated: bool,
    pub compiler_available: bool,
    pub il_supported: bool,
    pub integer_dot_supported: bool,
    pub host_clock_correlation_supported: bool,
    pub immediate: bool,
    pub loader_path: String,
    pub platform_name: String,
    pub device_name: String,
    pub driver_version: String,
    pub device_version: String,
    pub extensions: String,
    pub build_log: String,
}

impl SessionInfo {
    fn from_ffi(backend: Backend, info: &ffi::XeSessionInfo) -> Self {
        Self {
            backend,
            vendor_id: format!("{:04x}", info.vendor_id),
            device_id: format!("{:04x}", info.device_id),
            api_version_raw: info.api_version,
            compute_units: info.compute_units,
            max_group_size: info.max_group_size,
            subgroup_sizes: info.subgroups[..info.subgroup_count.min(8) as usize].to_vec(),
            timestamp_valid_bits: info.timestamp_valid_bits,
            kernel_timestamp_valid_bits: info.kernel_timestamp_valid_bits,
            timer_resolution: info.timer_resolution,
            global_memory_bytes: info.global_memory_bytes,
            max_allocation_bytes: info.max_allocation_bytes,
            local_memory_bytes: info.local_memory_bytes,
            creation_ns: info.creation_ns,
            integrated: info.integrated != 0,
            compiler_available: info.compiler_available != 0,
            il_supported: info.il_supported != 0,
            integer_dot_supported: info.integer_dot_supported != 0,
            host_clock_correlation_supported: info.host_clock_correlation_supported != 0,
            immediate: info.immediate != 0,
            loader_path: c_array(&info.library_path),
            platform_name: c_array(&info.platform_name),
            device_name: c_array(&info.device_name),
            driver_version: c_array(&info.driver_version),
            device_version: c_array(&info.device_version),
            extensions: c_array(&info.extensions),
            build_log: c_array(&info.build_log),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timing {
    pub host_ns: u64,
    pub device_ns: Option<u64>,
}

pub struct Session {
    pointer: NonNull<c_void>,
    backend: Backend,
    info: SessionInfo,
}

impl Session {
    pub fn probe(backend: Backend, immediate: bool) -> Result<SessionInfo> {
        Self::probe_device(backend, EXPECTED_VENDOR, EXPECTED_DEVICE, immediate)
    }

    pub fn probe_device(
        backend: Backend,
        expected_vendor: u32,
        expected_device: u32,
        immediate: bool,
    ) -> Result<SessionInfo> {
        let mut raw = ffi::XeSessionInfo::default();
        // SAFETY: `raw` is writable for the call and the C shim does not retain it.
        let status = unsafe {
            ffi::xe_session_probe(
                backend.ffi(),
                expected_vendor,
                expected_device,
                immediate as u8,
                &mut raw,
            )
        };
        if status != 0 {
            bail!(
                "{} capability probe failed: {}",
                backend,
                c_array(&raw.error)
            );
        }
        if raw.vendor_id != expected_vendor || raw.device_id != expected_device {
            bail!(
                "driver returned unexpected device {:04x}:{:04x}; expected {:04x}:{:04x}",
                raw.vendor_id,
                raw.device_id,
                expected_vendor,
                expected_device
            );
        }
        Ok(SessionInfo::from_ffi(backend, &raw))
    }

    pub fn create(
        backend: Backend,
        kind: ArtifactKind,
        artifact: &[u8],
        build_options: &str,
        entry_point: &str,
        immediate: bool,
    ) -> Result<Self> {
        let options = CString::new(build_options).context("build options contain NUL")?;
        let entry = CString::new(entry_point).context("entry point contains NUL")?;
        let mut raw = ffi::XeSessionInfo::default();
        // SAFETY: all slices and C strings remain alive for the duration of the call.
        let pointer = unsafe {
            ffi::xe_session_create(
                backend.ffi(),
                EXPECTED_VENDOR,
                EXPECTED_DEVICE,
                kind.ffi(),
                artifact.as_ptr(),
                artifact.len(),
                options.as_ptr(),
                entry.as_ptr(),
                immediate as u8,
                &mut raw,
            )
        };
        let pointer = NonNull::new(pointer).ok_or_else(|| {
            anyhow!(
                "{} session creation failed for {entry_point}: {}; build log: {}",
                backend,
                c_array(&raw.error),
                c_array(&raw.build_log)
            )
        })?;
        verify_device(&raw)?;
        let info = SessionInfo::from_ffi(backend, &raw);
        let session = Self {
            pointer,
            backend,
            info,
        };
        session.verify_mixed_generation_guard()?;
        Ok(session)
    }

    pub fn info(&self) -> &SessionInfo {
        &self.info
    }

    pub fn buffer(&self, kind: MemoryKind, size: usize) -> Result<Buffer<'_>> {
        let mut timing = ffi::XeRunTiming::default();
        // SAFETY: session is alive and timing is writable for the call.
        let pointer =
            unsafe { ffi::xe_buffer_create(self.pointer.as_ptr(), kind.ffi(), size, &mut timing) };
        let pointer =
            NonNull::new(pointer).ok_or_else(|| operation_error("buffer create", &timing))?;
        Ok(Buffer {
            pointer,
            session: self,
            size,
            allocation_ns: timing.host_ns,
        })
    }

    pub fn set_buffer(&self, index: u32, buffer: &Buffer<'_>) -> Result<()> {
        if !std::ptr::eq(self, buffer.session) {
            bail!("kernel argument buffer belongs to a different session");
        }
        let mut error = [0 as c_char; ffi::XE_TEXT];
        // SAFETY: both handles are live and the error buffer is writable.
        let status = unsafe {
            ffi::xe_kernel_arg_buffer(
                self.pointer.as_ptr(),
                index,
                buffer.pointer.as_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            bail!("kernel buffer argument {index} failed: {}", c_array(&error));
        }
        Ok(())
    }

    pub fn set_scalar<T: Copy>(&self, index: u32, value: &T) -> Result<()> {
        let mut error = [0 as c_char; ffi::XE_TEXT];
        // SAFETY: value and error buffer remain valid for the duration of the call.
        let status = unsafe {
            ffi::xe_kernel_arg_scalar(
                self.pointer.as_ptr(),
                index,
                (value as *const T).cast(),
                std::mem::size_of::<T>(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            bail!("kernel scalar argument {index} failed: {}", c_array(&error));
        }
        Ok(())
    }

    pub fn set_group_size(&self, x: u32, y: u32, z: u32) -> Result<()> {
        let mut error = [0 as c_char; ffi::XE_TEXT];
        // SAFETY: session and error buffer remain valid for the call.
        let status = unsafe {
            ffi::xe_kernel_group_size(
                self.pointer.as_ptr(),
                x,
                y,
                z,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            bail!("kernel group size failed: {}", c_array(&error));
        }
        Ok(())
    }

    pub fn run(&self, global: [usize; 3], local: [usize; 3], timeout_ns: u64) -> Result<Timing> {
        let mut timing = ffi::XeRunTiming::default();
        // SAFETY: the live session owns a fully configured kernel.
        let status = unsafe {
            ffi::xe_kernel_run(
                self.pointer.as_ptr(),
                global[0],
                global[1],
                global[2],
                local[0],
                local[1],
                local[2],
                timeout_ns,
                &mut timing,
            )
        };
        if status != 0 {
            return Err(operation_error("kernel run", &timing));
        }
        Ok(Timing {
            host_ns: timing.host_ns,
            device_ns: (timing.device_ns != 0).then_some(timing.device_ns),
        })
    }

    pub fn native_binary(&self) -> Result<Vec<u8>> {
        let mut pointer = std::ptr::null_mut();
        let mut length = 0_usize;
        let mut error = [0 as c_char; ffi::XE_TEXT];
        // SAFETY: output pointers and error storage are valid for the call.
        let status = unsafe {
            ffi::xe_session_native_binary(
                self.pointer.as_ptr(),
                &mut pointer,
                &mut length,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            bail!("native binary retrieval failed: {}", c_array(&error));
        }
        let pointer = NonNull::new(pointer).ok_or_else(|| anyhow!("native binary was empty"))?;
        // SAFETY: the C shim allocated exactly `length` bytes and transfers ownership here.
        let bytes = unsafe { std::slice::from_raw_parts(pointer.as_ptr(), length) }.to_vec();
        // SAFETY: this pointer was allocated by and is returned to the C shim.
        unsafe { ffi::xe_bytes_free(pointer.as_ptr()) };
        Ok(bytes)
    }

    pub fn loaded_library_paths(&self) -> Result<Vec<PathBuf>> {
        loaded_libraries()
    }

    pub fn verify_mixed_generation_guard(&self) -> Result<()> {
        let paths = loaded_libraries()?;
        let forbidden = paths.iter().find(|path| {
            let text = path.to_string_lossy();
            text.contains("23.43")
                || text.contains("compute-runtime-23.43")
                || text.contains("level-zero-v1.16.1")
                || text.contains("/toolchain/sysroot/usr/lib/")
        });
        if let Some(path) = forbidden {
            bail!(
                "mixed-generation guard rejected mapped library {}",
                path.display()
            );
        }
        let backend_library_seen = match self.backend {
            Backend::Opencl => paths.iter().any(|path| {
                path.file_name()
                    .and_then(|value| value.to_str())
                    .is_some_and(|value| value.starts_with("libigdrcl.so"))
            }),
            Backend::LevelZero => paths.iter().any(|path| {
                path.file_name()
                    .and_then(|value| value.to_str())
                    .is_some_and(|value| value.starts_with("libze_intel_gpu.so"))
            }),
        };
        if !backend_library_seen {
            bail!(
                "mixed-generation guard could not identify the selected {} Intel driver mapping",
                self.backend
            );
        }
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: the pointer is owned by this session and destroyed exactly once.
        unsafe { ffi::xe_session_destroy(self.pointer.as_ptr()) };
    }
}

pub struct Buffer<'a> {
    pointer: NonNull<c_void>,
    session: &'a Session,
    size: usize,
    allocation_ns: u64,
}

impl Buffer<'_> {
    pub const fn allocation_ns(&self) -> u64 {
        self.allocation_ns
    }

    pub fn write<T>(&self, values: &[T]) -> Result<Timing> {
        let bytes = std::mem::size_of_val(values);
        if bytes > self.size {
            bail!("buffer write of {bytes} bytes exceeds {}", self.size);
        }
        let mut timing = ffi::XeRunTiming::default();
        // SAFETY: slice bytes remain valid and the destination buffer is live.
        let status = unsafe {
            ffi::xe_buffer_write(
                self.session.pointer.as_ptr(),
                self.pointer.as_ptr(),
                values.as_ptr().cast(),
                bytes,
                &mut timing,
            )
        };
        if status != 0 {
            return Err(operation_error("buffer write", &timing));
        }
        Ok(Timing {
            host_ns: timing.host_ns,
            device_ns: None,
        })
    }

    pub fn read<T>(&self, values: &mut [T]) -> Result<Timing> {
        let bytes = std::mem::size_of_val(values);
        if bytes > self.size {
            bail!("buffer read of {bytes} bytes exceeds {}", self.size);
        }
        let mut timing = ffi::XeRunTiming::default();
        // SAFETY: destination bytes remain valid and the source buffer is live.
        let status = unsafe {
            ffi::xe_buffer_read(
                self.session.pointer.as_ptr(),
                self.pointer.as_ptr(),
                values.as_mut_ptr().cast(),
                bytes,
                &mut timing,
            )
        };
        if status != 0 {
            return Err(operation_error("buffer read", &timing));
        }
        Ok(Timing {
            host_ns: timing.host_ns,
            device_ns: None,
        })
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        let mut timing = ffi::XeRunTiming::default();
        // SAFETY: the buffer belongs to the still-live borrowed session.
        unsafe {
            ffi::xe_buffer_destroy(
                self.session.pointer.as_ptr(),
                self.pointer.as_ptr(),
                &mut timing,
            )
        };
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTiming {
    pub status: i32,
    pub allocation_ns: u64,
    pub first_write_ns: u64,
    pub read_ns: u64,
    pub reuse_write_ns: u64,
    pub cleanup_ns: u64,
    pub error: String,
}

pub fn memory_roundtrip(
    backend: Backend,
    kind: MemoryKind,
    size: usize,
    immediate: bool,
) -> MemoryTiming {
    let mut raw = ffi::XeMemoryTiming::default();
    // SAFETY: raw is writable for the duration of the call.
    unsafe {
        ffi::xe_memory_roundtrip(
            backend.ffi(),
            EXPECTED_VENDOR,
            EXPECTED_DEVICE,
            kind.ffi(),
            size,
            immediate as u8,
            &mut raw,
        );
    }
    MemoryTiming {
        status: raw.status,
        allocation_ns: raw.allocation_ns,
        first_write_ns: raw.first_write_ns,
        read_ns: raw.read_ns,
        reuse_write_ns: raw.reuse_write_ns,
        cleanup_ns: raw.cleanup_ns,
        error: c_array(&raw.error),
    }
}

fn verify_device(info: &ffi::XeSessionInfo) -> Result<()> {
    if info.vendor_id != EXPECTED_VENDOR || info.device_id != EXPECTED_DEVICE {
        bail!(
            "driver returned unexpected device {:04x}:{:04x}; expected {:04x}:{:04x}",
            info.vendor_id,
            info.device_id,
            EXPECTED_VENDOR,
            EXPECTED_DEVICE
        );
    }
    Ok(())
}

fn operation_error(operation: &str, timing: &ffi::XeRunTiming) -> anyhow::Error {
    anyhow!(
        "{operation} failed with status {}: {}",
        timing.status,
        c_array(&timing.error)
    )
}

fn c_array<const N: usize>(value: &[c_char; N]) -> String {
    // SAFETY: all buffers are zero-initialized and C writers use bounded snprintf.
    unsafe { CStr::from_ptr(value.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn loaded_libraries() -> Result<Vec<PathBuf>> {
    let maps = std::fs::read_to_string("/proc/self/maps").context("read /proc/self/maps")?;
    let mut paths = BTreeSet::new();
    for line in maps.lines() {
        let Some(path) = line.split_whitespace().nth(5) else {
            continue;
        };
        if path.starts_with('/') && path.contains(".so") {
            paths.insert(PathBuf::from(path));
        }
    }
    Ok(paths.into_iter().collect())
}
