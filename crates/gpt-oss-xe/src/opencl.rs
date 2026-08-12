//! Runtime OpenCL loader and the only unsafe boundary in `gpt-oss-xe`.

use std::collections::BTreeSet;
use std::ffi::{c_char, c_void, CStr, CString};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::{Instant, SystemTime};

use bytemuck::Zeroable;
use half::bf16;
use libloading::Library;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    sha256_bytes, ActivationRecordV2, AttachConfig, AttachmentMode, KernelVariant, PhaseTiming,
    ProjectionRequest, ProjectionRuntime, PromotionRecord, ValidationClass, XeDescriptor, XeError,
    XeIdentity, XeMemoryDescriptor, BUILD_OPTIONS, EXPECTED_DEVICE_ID, EXPECTED_VENDOR_ID,
    KERNEL_ABI_SHA256, KERNEL_SOURCE, KERNEL_SOURCE_SHA256, WORKGROUP_SIZE,
    XE_ACTIVATION_RECORD_BYTES, XE_WEIGHT_PLANES,
};

type ClInt = i32;
type ClUint = u32;
type ClUlong = u64;
type ClBool = ClUint;
type ClBitfield = ClUlong;
type ClDeviceType = ClBitfield;
type ClMemFlags = ClBitfield;
type ClPlatformId = *mut c_void;
type ClDeviceId = *mut c_void;
type ClContext = *mut c_void;
type ClCommandQueue = *mut c_void;
type ClProgram = *mut c_void;
type ClKernel = *mut c_void;
type ClMem = *mut c_void;
type ClEvent = *mut c_void;
type ClContextProperties = isize;
type ClQueueProperties = isize;

const CL_SUCCESS: ClInt = 0;
const CL_DEVICE_NOT_FOUND: ClInt = -1;
const CL_TRUE: ClBool = 1;
const CL_DEVICE_TYPE_GPU: ClDeviceType = 1 << 2;
const CL_DEVICE_VENDOR_ID: ClUint = 0x1001;
const CL_DEVICE_MAX_WORK_GROUP_SIZE: ClUint = 0x1004;
const CL_DEVICE_MAX_MEM_ALLOC_SIZE: ClUint = 0x1010;
const CL_DEVICE_GLOBAL_MEM_SIZE: ClUint = 0x101f;
const CL_DEVICE_COMPILER_AVAILABLE: ClUint = 0x1028;
const CL_DEVICE_NAME: ClUint = 0x102b;
const CL_DRIVER_VERSION: ClUint = 0x102d;
const CL_DEVICE_VERSION: ClUint = 0x102f;
const CL_DEVICE_EXTENSIONS: ClUint = 0x1030;
const CL_DEVICE_LOCAL_MEM_SIZE: ClUint = 0x1023;
const CL_DEVICE_HOST_UNIFIED_MEMORY: ClUint = 0x1035;
const CL_DEVICE_ID_INTEL: ClUint = 0x4251;
const CL_DEVICE_SUB_GROUP_SIZES_INTEL: ClUint = 0x4108;
const CL_PLATFORM_NAME: ClUint = 0x0902;
const CL_QUEUE_PROPERTIES: ClQueueProperties = 0x1093;
const CL_QUEUE_PROFILING_ENABLE: ClQueueProperties = 1 << 1;
const CL_MEM_READ_WRITE: ClMemFlags = 1 << 0;
const CL_PROGRAM_BINARY_SIZES: ClUint = 0x1165;
const CL_PROGRAM_BINARIES: ClUint = 0x1166;
const CL_PROGRAM_BUILD_LOG: ClUint = 0x1183;

type GetPlatformIds = unsafe extern "C" fn(ClUint, *mut ClPlatformId, *mut ClUint) -> ClInt;
type GetPlatformInfo =
    unsafe extern "C" fn(ClPlatformId, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
type GetDeviceIds =
    unsafe extern "C" fn(ClPlatformId, ClDeviceType, ClUint, *mut ClDeviceId, *mut ClUint) -> ClInt;
type GetDeviceInfo =
    unsafe extern "C" fn(ClDeviceId, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
type CreateContext = unsafe extern "C" fn(
    *const ClContextProperties,
    ClUint,
    *const ClDeviceId,
    Option<unsafe extern "C" fn(*const c_char, *const c_void, usize, *mut c_void)>,
    *mut c_void,
    *mut ClInt,
) -> ClContext;
type ReleaseContext = unsafe extern "C" fn(ClContext) -> ClInt;
type CreateCommandQueueWithProperties = unsafe extern "C" fn(
    ClContext,
    ClDeviceId,
    *const ClQueueProperties,
    *mut ClInt,
) -> ClCommandQueue;
type ReleaseCommandQueue = unsafe extern "C" fn(ClCommandQueue) -> ClInt;
type CreateProgramWithSource = unsafe extern "C" fn(
    ClContext,
    ClUint,
    *const *const c_char,
    *const usize,
    *mut ClInt,
) -> ClProgram;
type CreateProgramWithBinary = unsafe extern "C" fn(
    ClContext,
    ClUint,
    *const ClDeviceId,
    *const usize,
    *const *const u8,
    *mut ClInt,
    *mut ClInt,
) -> ClProgram;
type BuildProgram = unsafe extern "C" fn(
    ClProgram,
    ClUint,
    *const ClDeviceId,
    *const c_char,
    Option<unsafe extern "C" fn(ClProgram, *mut c_void)>,
    *mut c_void,
) -> ClInt;
type GetProgramInfo =
    unsafe extern "C" fn(ClProgram, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
type GetProgramBuildInfo =
    unsafe extern "C" fn(ClProgram, ClDeviceId, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
type ReleaseProgram = unsafe extern "C" fn(ClProgram) -> ClInt;
type CreateKernel = unsafe extern "C" fn(ClProgram, *const c_char, *mut ClInt) -> ClKernel;
type SetKernelArg = unsafe extern "C" fn(ClKernel, ClUint, usize, *const c_void) -> ClInt;
type ReleaseKernel = unsafe extern "C" fn(ClKernel) -> ClInt;
type CreateBuffer =
    unsafe extern "C" fn(ClContext, ClMemFlags, usize, *mut c_void, *mut ClInt) -> ClMem;
type ReleaseMemObject = unsafe extern "C" fn(ClMem) -> ClInt;
type EnqueueWriteBuffer = unsafe extern "C" fn(
    ClCommandQueue,
    ClMem,
    ClBool,
    usize,
    usize,
    *const c_void,
    ClUint,
    *const ClEvent,
    *mut ClEvent,
) -> ClInt;
type EnqueueReadBuffer = unsafe extern "C" fn(
    ClCommandQueue,
    ClMem,
    ClBool,
    usize,
    usize,
    *mut c_void,
    ClUint,
    *const ClEvent,
    *mut ClEvent,
) -> ClInt;
type EnqueueNdRangeKernel = unsafe extern "C" fn(
    ClCommandQueue,
    ClKernel,
    ClUint,
    *const usize,
    *const usize,
    *const usize,
    ClUint,
    *const ClEvent,
    *mut ClEvent,
) -> ClInt;
type WaitForEvents = unsafe extern "C" fn(ClUint, *const ClEvent) -> ClInt;
type Finish = unsafe extern "C" fn(ClCommandQueue) -> ClInt;
type ReleaseEvent = unsafe extern "C" fn(ClEvent) -> ClInt;

struct OpenClApi {
    get_platform_ids: GetPlatformIds,
    get_platform_info: GetPlatformInfo,
    get_device_ids: GetDeviceIds,
    get_device_info: GetDeviceInfo,
    create_context: CreateContext,
    release_context: ReleaseContext,
    create_command_queue_with_properties: CreateCommandQueueWithProperties,
    release_command_queue: ReleaseCommandQueue,
    create_program_with_source: CreateProgramWithSource,
    create_program_with_binary: CreateProgramWithBinary,
    build_program: BuildProgram,
    get_program_info: GetProgramInfo,
    get_program_build_info: GetProgramBuildInfo,
    release_program: ReleaseProgram,
    create_kernel: CreateKernel,
    set_kernel_arg: SetKernelArg,
    release_kernel: ReleaseKernel,
    create_buffer: CreateBuffer,
    release_mem_object: ReleaseMemObject,
    enqueue_write_buffer: EnqueueWriteBuffer,
    enqueue_read_buffer: EnqueueReadBuffer,
    enqueue_nd_range_kernel: EnqueueNdRangeKernel,
    wait_for_events: WaitForEvents,
    finish: Finish,
    release_event: ReleaseEvent,
    library_path: PathBuf,
    _library: Library,
}

impl OpenClApi {
    fn load() -> Result<Self, XeError> {
        if let Some(path) = std::env::var_os("GPT_OSS_XE_OPENCL_LIBRARY") {
            return Self::load_candidates(&[PathBuf::from(path)]);
        }
        Self::load_candidates(&[
            PathBuf::from("libOpenCL.so.1"),
            PathBuf::from("/usr/lib/x86_64-linux-gnu/libOpenCL.so.1"),
            PathBuf::from("libOpenCL.so"),
        ])
    }

    fn load_candidates(candidates: &[PathBuf]) -> Result<Self, XeError> {
        let mut errors = Vec::new();
        for candidate in candidates {
            // SAFETY: the library remains owned by the returned table for at
            // least as long as every copied function pointer.
            let library = match unsafe { Library::new(candidate) } {
                Ok(library) => library,
                Err(error) => {
                    errors.push(format!("{}: {error}", candidate.display()));
                    continue;
                }
            };
            // SAFETY: each requested symbol uses the Khronos OpenCL C ABI and
            // is copied while `library` remains alive in the table.
            let table = unsafe { Self::from_library(library, candidate) };
            match table {
                Ok(table) => return Ok(table),
                Err(error) => errors.push(format!("{}: {error}", candidate.display())),
            }
        }
        Err(XeError::Unsupported(format!(
            "OpenCL loader unavailable ({})",
            errors.join("; ")
        )))
    }

    unsafe fn from_library(library: Library, candidate: &Path) -> Result<Self, XeError> {
        unsafe fn load<T: Copy>(library: &Library, name: &[u8]) -> Result<T, XeError> {
            // SAFETY: the caller provides the exact OpenCL C ABI type for the
            // named symbol and retains the library for the pointer lifetime.
            unsafe { library.get::<T>(name) }
                .map(|symbol| *symbol)
                .map_err(|error| {
                    XeError::Unsupported(format!(
                        "missing OpenCL symbol {}: {error}",
                        String::from_utf8_lossy(name).trim_end_matches('\0')
                    ))
                })
        }
        let library_path = candidate
            .canonicalize()
            .unwrap_or_else(|_| candidate.to_path_buf());
        Ok(Self {
            // SAFETY: all names and types are the OpenCL C ABI declarations.
            get_platform_ids: unsafe { load(&library, b"clGetPlatformIDs\0")? },
            get_platform_info: unsafe { load(&library, b"clGetPlatformInfo\0")? },
            get_device_ids: unsafe { load(&library, b"clGetDeviceIDs\0")? },
            get_device_info: unsafe { load(&library, b"clGetDeviceInfo\0")? },
            create_context: unsafe { load(&library, b"clCreateContext\0")? },
            release_context: unsafe { load(&library, b"clReleaseContext\0")? },
            create_command_queue_with_properties: unsafe {
                load(&library, b"clCreateCommandQueueWithProperties\0")?
            },
            release_command_queue: unsafe { load(&library, b"clReleaseCommandQueue\0")? },
            create_program_with_source: unsafe { load(&library, b"clCreateProgramWithSource\0")? },
            create_program_with_binary: unsafe { load(&library, b"clCreateProgramWithBinary\0")? },
            build_program: unsafe { load(&library, b"clBuildProgram\0")? },
            get_program_info: unsafe { load(&library, b"clGetProgramInfo\0")? },
            get_program_build_info: unsafe { load(&library, b"clGetProgramBuildInfo\0")? },
            release_program: unsafe { load(&library, b"clReleaseProgram\0")? },
            create_kernel: unsafe { load(&library, b"clCreateKernel\0")? },
            set_kernel_arg: unsafe { load(&library, b"clSetKernelArg\0")? },
            release_kernel: unsafe { load(&library, b"clReleaseKernel\0")? },
            create_buffer: unsafe { load(&library, b"clCreateBuffer\0")? },
            release_mem_object: unsafe { load(&library, b"clReleaseMemObject\0")? },
            enqueue_write_buffer: unsafe { load(&library, b"clEnqueueWriteBuffer\0")? },
            enqueue_read_buffer: unsafe { load(&library, b"clEnqueueReadBuffer\0")? },
            enqueue_nd_range_kernel: unsafe { load(&library, b"clEnqueueNDRangeKernel\0")? },
            wait_for_events: unsafe { load(&library, b"clWaitForEvents\0")? },
            finish: unsafe { load(&library, b"clFinish\0")? },
            release_event: unsafe { load(&library, b"clReleaseEvent\0")? },
            library_path,
            _library: library,
        })
    }
}

#[derive(Debug, Clone)]
struct DeviceFacts {
    device: ClDeviceId,
    driver_version: String,
    device_version: String,
    extensions: String,
    compiler_available: bool,
    integrated: bool,
    subgroup_sizes: Vec<usize>,
    max_group_size: usize,
    global_memory_bytes: u64,
    max_allocation_bytes: u64,
    local_memory_bytes: u64,
}

#[derive(Debug, Clone)]
struct CoreLibraries {
    loader_sha256: String,
    driver_sha256: String,
    igc_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CacheManifest {
    schema: String,
    key: String,
    source_sha256: String,
    abi_sha256: String,
    build_options: String,
    pci_vendor_id: String,
    pci_device_id: String,
    driver_version: String,
    loader_sha256: String,
    driver_sha256: String,
    igc_sha256: String,
    native_sha256: String,
}

pub(crate) struct OpenClRuntime {
    api: OpenClApi,
    device: ClDeviceId,
    context: Option<ClContext>,
    queue: Option<ClCommandQueue>,
    program: Option<ClProgram>,
    kernels: [Option<ClKernel>; 3],
    weight: Option<ClMem>,
    bias: Option<ClMem>,
    activation: Option<ClMem>,
    output: Option<ClMem>,
    descriptor: XeDescriptor,
    max_columns: usize,
    max_blocks: usize,
    shutdown: bool,
}

// SAFETY: OpenCL handles are used only while `OpenClRuntime` is held behind
// `XeProjectionEngine`'s mutex. The runtime owns every handle and shuts them
// down before the dynamically loaded function table is dropped.
unsafe impl Send for OpenClRuntime {}

impl OpenClRuntime {
    pub(crate) fn attach(config: &AttachConfig, record: &PromotionRecord) -> Result<Self, XeError> {
        let api = OpenClApi::load()?;
        let facts = select_device(&api)?;
        validate_capabilities(&facts)?;
        let libraries = core_libraries(&api.library_path)?;
        let exact_validated_stack = facts.driver_version == record.driver_version
            && libraries.loader_sha256 == record.opencl_loader_sha256
            && libraries.driver_sha256 == record.opencl_driver_sha256
            && libraries.igc_sha256 == record.igc_sha256;
        let validation_class = validation_class(config.mode, exact_validated_stack)?;

        let context = create_context(&api, facts.device)?;
        let queue = match create_queue(&api, context, facts.device) {
            Ok(queue) => queue,
            Err(error) => {
                // SAFETY: context was created by this API and has no queue.
                unsafe { (api.release_context)(context) };
                return Err(error);
            }
        };
        let key = native_cache_key(&facts, &libraries);
        let native_cache = read_native_cache(&config.cache_root, &key, &facts, &libraries);
        let compile_source = || {
            create_program_source(&api, context, facts.device)
                .map(|program| (program, false))
                .inspect_err(|_| {
                    release_queue_context(&api, queue, context);
                })
        };
        let (program, cache_hit) = match native_cache {
            Ok(Some(binary)) => match create_program_binary(&api, context, facts.device, &binary) {
                Ok(program) => (program, true),
                Err(error) if facts.compiler_available => {
                    tracing::warn!(reason = %error, "Xe native cache rejected; recompiling source");
                    compile_source()?
                }
                Err(error) => {
                    release_queue_context(&api, queue, context);
                    return Err(error);
                }
            },
            Ok(None) | Err(_) if facts.compiler_available => compile_source()?,
            Ok(None) => {
                release_queue_context(&api, queue, context);
                return Err(XeError::Capability(
                    "OpenCL compiler is unavailable and no valid native cache exists".into(),
                ));
            }
            Err(error) => {
                release_queue_context(&api, queue, context);
                return Err(error);
            }
        };
        let kernels = match create_kernels(&api, program) {
            Ok(kernels) => kernels,
            Err(error) => {
                // SAFETY: program is live and no kernels were retained.
                unsafe { (api.release_program)(program) };
                release_queue_context(&api, queue, context);
                return Err(error);
            }
        };
        let memory = memory_descriptor(config)?;
        if memory.device_resident_bytes as u64 > facts.global_memory_bytes
            || memory.weight_capacity_bytes as u64 > facts.max_allocation_bytes
            || memory.activation_capacity_bytes as u64 > facts.max_allocation_bytes
            || memory.output_capacity_bytes as u64 > facts.max_allocation_bytes
        {
            release_kernels_program_queue_context(&api, kernels, program, queue, context);
            return Err(XeError::ResidentLimit(
                "checked slab exceeds an OpenCL global or per-allocation limit".into(),
            ));
        }
        let buffers = match allocate_buffers(&api, context, &memory) {
            Ok(buffers) => buffers,
            Err(error) => {
                release_kernels_program_queue_context(&api, kernels, program, queue, context);
                return Err(error);
            }
        };
        let identity = XeIdentity {
            pci_vendor_id: format!("{EXPECTED_VENDOR_ID:04x}"),
            pci_device_id: format!("{EXPECTED_DEVICE_ID:04x}"),
            driver_version: facts.driver_version.clone(),
            device_version: facts.device_version.clone(),
            opencl_loader_sha256: libraries.loader_sha256.clone(),
            opencl_driver_sha256: libraries.driver_sha256.clone(),
            igc_sha256: libraries.igc_sha256.clone(),
        };
        let descriptor = XeDescriptor {
            effective_backend: "cpu_xe".into(),
            validation_class,
            identity,
            source_sha256: KERNEL_SOURCE_SHA256.into(),
            abi_sha256: KERNEL_ABI_SHA256.into(),
            build_options: BUILD_OPTIONS.into(),
            native_cache_key: key.clone(),
            native_cache_hit: cache_hit,
            gate_up_min_rows: record.gate_up_min_rows,
            down_min_rows: record.down_min_rows,
            workgroup_size: WORKGROUP_SIZE,
            memory,
            runtime_fault_policy:
                "drain terminal event, discard uncommitted output, one CPU recomputation, open process-wide breaker"
                    .into(),
        };
        let mut runtime = Self {
            api,
            device: facts.device,
            context: Some(context),
            queue: Some(queue),
            program: Some(program),
            kernels: kernels.map(Some),
            weight: Some(buffers.0),
            bias: Some(buffers.1),
            activation: Some(buffers.2),
            output: Some(buffers.3),
            descriptor,
            max_columns: config.max_columns,
            max_blocks: config.max_blocks,
            shutdown: false,
        };
        if let Err(cache_error) = runtime.startup_self_test() {
            if cache_hit && facts.compiler_available {
                tracing::warn!(reason = %cache_error, "Xe native cache failed startup self-test; recompiling source");
                runtime.replace_program_from_source()?;
                runtime.descriptor.native_cache_hit = false;
                runtime.startup_self_test()?;
            } else {
                let _ = runtime.shutdown();
                return Err(cache_error);
            }
        }
        if !runtime.descriptor.native_cache_hit {
            let binary = runtime.native_binary()?;
            write_native_cache(&config.cache_root, &key, &facts, &libraries, &binary)?;
        }
        tracing::info!(
            pci_vendor = "8086",
            pci_device = "9a49",
            driver_version = %facts.driver_version,
            validation = ?runtime.descriptor.validation_class,
            native_cache_hit = runtime.descriptor.native_cache_hit,
            resident_bytes = runtime.descriptor.memory.device_resident_bytes,
            "attached serialized CPU+Xe projection engine"
        );
        Ok(runtime)
    }

    fn startup_self_test(&mut self) -> Result<(), XeError> {
        let rows = 4;
        let columns = 32;
        let blocks = 1;
        let mut weights = vec![0_u8; columns * blocks * XE_WEIGHT_PLANES];
        let mut packed = [[0_u8; 16]; 32];
        for (lane, lane_packed) in packed.iter_mut().enumerate() {
            weights[lane] = 127;
            for (byte, packed_byte) in lane_packed.iter_mut().enumerate() {
                *packed_byte =
                    ((lane + byte) as u8 & 0x0f) | (((lane * 3 + byte + 1) as u8 & 0x0f) << 4);
                weights[(byte + 1) * 32 + lane] = *packed_byte;
            }
        }
        let mut activations = vec![ActivationRecordV2::zeroed(); rows];
        for (row, record) in activations.iter_mut().enumerate() {
            for index in 0..32 {
                record.primary[index] = (index as i8).wrapping_sub(16 + row as i8);
                record.residual[index] = ((index * 3 + row) as i8 % 11) - 5;
            }
            record.primary_scale = 0.03125 * (row + 1) as f32;
            record.residual_scale = 0.0078125 * (row + 1) as f32;
        }
        let bias = (0..columns)
            .map(|column| column as f32 * 0.000_976_562_5)
            .collect::<Vec<_>>();
        let request = ProjectionRequest {
            role: crate::ProjectionRole::GateUp,
            rows,
            columns,
            blocks,
            weights_v2: &weights,
            activations_v2: &activations,
            bias: &bias,
        };
        let mut actual = vec![0.0; rows * columns];
        self.project_impl(&request, KernelVariant::Tile32M4, &mut actual)?;
        for row in 0..rows {
            for column in 0..columns {
                let record = &activations[row];
                let mut primary = 0_i32;
                let mut residual = 0_i32;
                for (byte, packed_byte) in packed[column].iter().copied().enumerate() {
                    let low = e2m1_x2(packed_byte & 0x0f) as i32;
                    let high = e2m1_x2(packed_byte >> 4) as i32;
                    primary += low * record.primary[byte * 2] as i32
                        + high * record.primary[byte * 2 + 1] as i32;
                    residual += low * record.residual[byte * 2] as i32
                        + high * record.residual[byte * 2 + 1] as i32;
                }
                let expected = bias[column]
                    + primary as f32 * 0.5 * record.primary_scale
                    + residual as f32 * 0.5 * record.residual_scale;
                let observed = actual[row * columns + column];
                if !observed.is_finite()
                    || (expected - observed).abs() > 1e-6
                    || bf16::from_f32(expected).to_bits() != bf16::from_f32(observed).to_bits()
                {
                    return Err(XeError::Capability(format!(
                        "startup numerical self-test mismatch at row {row}, column {column}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn project_impl(
        &mut self,
        request: &ProjectionRequest<'_>,
        variant: KernelVariant,
        output: &mut [f32],
    ) -> Result<PhaseTiming, XeError> {
        if request.columns > self.max_columns || request.blocks > self.max_blocks {
            return Err(XeError::ResidentLimit(format!(
                "projection {}x{} exceeds configured slab {}x{}",
                request.columns, request.blocks, self.max_columns, self.max_blocks
            )));
        }
        if output.len() != request.rows * request.columns {
            return Err(XeError::Dimensions("output extent mismatch".into()));
        }
        let queue = self.queue()?;
        let weight = self.weight()?;
        let bias = self.bias()?;
        let activation = self.activation()?;
        let device_output = self.output()?;
        let mut timing = PhaseTiming::default();
        let started = Instant::now();
        self.write_buffer(queue, weight, request.weights_v2)?;
        self.write_buffer(queue, bias, bytemuck::cast_slice(request.bias))?;
        timing.weight = started.elapsed();

        let divisor = variant.rows_per_dispatch();
        let max_rows = self.descriptor.memory.max_rows_per_chunk;
        let kernel = self.kernel(variant)?;
        let mut source_row = 0;
        while source_row < request.rows {
            let real_rows = (request.rows - source_row).min(max_rows);
            let dispatch_rows = round_up(real_rows, divisor);
            if dispatch_rows > max_rows {
                return Err(XeError::ResidentLimit(
                    "padded Xe row chunk exceeds configured slab".into(),
                ));
            }
            let activation_count = dispatch_rows
                .checked_mul(request.blocks)
                .ok_or_else(|| XeError::Dimensions("chunk activation extent overflows".into()))?;
            let mut staged = vec![ActivationRecordV2::zeroed(); activation_count];
            let source_start = source_row * request.blocks;
            let source_end = source_start + real_rows * request.blocks;
            staged[..real_rows * request.blocks]
                .copy_from_slice(&request.activations_v2[source_start..source_end]);
            let phase = Instant::now();
            self.write_buffer(queue, activation, bytemuck::cast_slice(&staged))?;
            timing.activation += phase.elapsed();

            let rows_u32 = u32::try_from(dispatch_rows)
                .map_err(|_| XeError::Dimensions("rows exceed u32".into()))?;
            let columns_u32 = u32::try_from(request.columns)
                .map_err(|_| XeError::Dimensions("columns exceed u32".into()))?;
            let blocks_u32 = u32::try_from(request.blocks)
                .map_err(|_| XeError::Dimensions("blocks exceed u32".into()))?;
            self.set_mem_arg(kernel, 0, weight)?;
            self.set_mem_arg(kernel, 1, activation)?;
            self.set_mem_arg(kernel, 2, bias)?;
            self.set_mem_arg(kernel, 3, device_output)?;
            self.set_scalar_arg(kernel, 4, &rows_u32)?;
            self.set_scalar_arg(kernel, 5, &columns_u32)?;
            self.set_scalar_arg(kernel, 6, &blocks_u32)?;
            let phase = Instant::now();
            self.run_terminal_event(
                queue,
                kernel,
                [
                    round_up(request.columns, WORKGROUP_SIZE),
                    dispatch_rows / divisor,
                ],
            )?;
            timing.submit_wait += phase.elapsed();
            let mut staged_output = vec![0.0_f32; dispatch_rows * request.columns];
            let phase = Instant::now();
            self.read_buffer(
                queue,
                device_output,
                bytemuck::cast_slice_mut(&mut staged_output),
            )?;
            timing.readback += phase.elapsed();
            let destination = &mut output
                [source_row * request.columns..(source_row + real_rows) * request.columns];
            destination.copy_from_slice(&staged_output[..real_rows * request.columns]);
            source_row += real_rows;
        }
        Ok(timing)
    }

    fn replace_program_from_source(&mut self) -> Result<(), XeError> {
        self.drain()?;
        for kernel in self.kernels.iter_mut().rev() {
            if let Some(kernel) = kernel.take() {
                // SAFETY: kernel is live, queue has been drained, and is owned here.
                unsafe { (self.api.release_kernel)(kernel) };
            }
        }
        if let Some(program) = self.program.take() {
            // SAFETY: all kernels are released and program is owned here.
            unsafe { (self.api.release_program)(program) };
        }
        let program = create_program_source(&self.api, self.context()?, self.device)?;
        self.kernels = create_kernels(&self.api, program)?.map(Some);
        self.program = Some(program);
        Ok(())
    }

    fn native_binary(&self) -> Result<Vec<u8>, XeError> {
        let program = self.program()?;
        let mut size = 0_usize;
        // SAFETY: size is writable and program is live.
        check("clGetProgramInfo(binary size)", unsafe {
            (self.api.get_program_info)(
                program,
                CL_PROGRAM_BINARY_SIZES,
                std::mem::size_of::<usize>(),
                (&mut size as *mut usize).cast(),
                ptr::null_mut(),
            )
        })?;
        if size == 0 {
            return Err(XeError::Artifact(
                "OpenCL returned an empty native program".into(),
            ));
        }
        let mut binary = vec![0_u8; size];
        let mut pointer = binary.as_mut_ptr();
        // SAFETY: OpenCL writes at most the queried binary size through the
        // one-element pointer array.
        check("clGetProgramInfo(binary)", unsafe {
            (self.api.get_program_info)(
                program,
                CL_PROGRAM_BINARIES,
                std::mem::size_of::<*mut u8>(),
                (&mut pointer as *mut *mut u8).cast(),
                ptr::null_mut(),
            )
        })?;
        Ok(binary)
    }

    fn write_buffer(
        &self,
        queue: ClCommandQueue,
        buffer: ClMem,
        bytes: &[u8],
    ) -> Result<(), XeError> {
        if bytes.is_empty() {
            return Err(XeError::Dimensions("zero-byte OpenCL write".into()));
        }
        // SAFETY: queue/buffer are live and `bytes` remains readable through
        // the blocking call.
        check("clEnqueueWriteBuffer", unsafe {
            (self.api.enqueue_write_buffer)(
                queue,
                buffer,
                CL_TRUE,
                0,
                bytes.len(),
                bytes.as_ptr().cast(),
                0,
                ptr::null(),
                ptr::null_mut(),
            )
        })
    }

    fn read_buffer(
        &self,
        queue: ClCommandQueue,
        buffer: ClMem,
        bytes: &mut [u8],
    ) -> Result<(), XeError> {
        if bytes.is_empty() {
            return Err(XeError::Dimensions("zero-byte OpenCL read".into()));
        }
        // SAFETY: queue/buffer are live and `bytes` remains writable through
        // the blocking call.
        check("clEnqueueReadBuffer", unsafe {
            (self.api.enqueue_read_buffer)(
                queue,
                buffer,
                CL_TRUE,
                0,
                bytes.len(),
                bytes.as_mut_ptr().cast(),
                0,
                ptr::null(),
                ptr::null_mut(),
            )
        })
    }

    fn set_mem_arg(&self, kernel: ClKernel, index: u32, memory: ClMem) -> Result<(), XeError> {
        // SAFETY: kernel/memory are live and OpenCL copies the handle value.
        check("clSetKernelArg(buffer)", unsafe {
            (self.api.set_kernel_arg)(
                kernel,
                index,
                std::mem::size_of::<ClMem>(),
                (&memory as *const ClMem).cast(),
            )
        })
    }

    fn set_scalar_arg<T: Copy>(
        &self,
        kernel: ClKernel,
        index: u32,
        value: &T,
    ) -> Result<(), XeError> {
        // SAFETY: kernel is live and OpenCL copies exactly `size_of::<T>()`.
        check("clSetKernelArg(scalar)", unsafe {
            (self.api.set_kernel_arg)(
                kernel,
                index,
                std::mem::size_of::<T>(),
                (value as *const T).cast(),
            )
        })
    }

    fn run_terminal_event(
        &self,
        queue: ClCommandQueue,
        kernel: ClKernel,
        global: [usize; 2],
    ) -> Result<(), XeError> {
        let local = [WORKGROUP_SIZE, 1_usize];
        let mut event: ClEvent = ptr::null_mut();
        // SAFETY: handles and dimension arrays are live; event is writable.
        let enqueue = unsafe {
            (self.api.enqueue_nd_range_kernel)(
                queue,
                kernel,
                2,
                ptr::null(),
                global.as_ptr(),
                local.as_ptr(),
                0,
                ptr::null(),
                &mut event,
            )
        };
        if enqueue != CL_SUCCESS {
            let _ = self.finish_queue(queue);
            return Err(runtime_status("clEnqueueNDRangeKernel", enqueue));
        }
        if event.is_null() {
            let _ = self.finish_queue(queue);
            return Err(XeError::Runtime(
                "OpenCL submission returned no terminal event".into(),
            ));
        }
        // SAFETY: the terminal event is live and owned by this invocation.
        let wait = unsafe { (self.api.wait_for_events)(1, &event) };
        // SAFETY: wait completed (or failed terminally) and the event is not reused.
        let release = unsafe { (self.api.release_event)(event) };
        if wait != CL_SUCCESS {
            let _ = self.finish_queue(queue);
            return Err(runtime_status("clWaitForEvents", wait));
        }
        check("clReleaseEvent", release)
    }

    fn finish_queue(&self, queue: ClCommandQueue) -> Result<(), XeError> {
        // SAFETY: queue is live and owned by the runtime.
        check("clFinish", unsafe { (self.api.finish)(queue) })
    }

    fn context(&self) -> Result<ClContext, XeError> {
        self.context
            .filter(|handle| !handle.is_null())
            .ok_or_else(|| XeError::Shutdown("OpenCL context is closed".into()))
    }

    fn queue(&self) -> Result<ClCommandQueue, XeError> {
        self.queue
            .filter(|handle| !handle.is_null())
            .ok_or_else(|| XeError::Shutdown("OpenCL queue is closed".into()))
    }

    fn program(&self) -> Result<ClProgram, XeError> {
        self.program
            .filter(|handle| !handle.is_null())
            .ok_or_else(|| XeError::Shutdown("OpenCL program is closed".into()))
    }

    fn kernel(&self, variant: KernelVariant) -> Result<ClKernel, XeError> {
        let index = match variant {
            KernelVariant::Tile32M1 => 0,
            KernelVariant::Tile32M2 => 1,
            KernelVariant::Tile32M4 => 2,
        };
        self.kernels[index]
            .filter(|handle| !handle.is_null())
            .ok_or_else(|| XeError::Shutdown("OpenCL kernel is closed".into()))
    }

    fn weight(&self) -> Result<ClMem, XeError> {
        live_buffer(self.weight, "weight")
    }
    fn bias(&self) -> Result<ClMem, XeError> {
        live_buffer(self.bias, "bias")
    }
    fn activation(&self) -> Result<ClMem, XeError> {
        live_buffer(self.activation, "activation")
    }
    fn output(&self) -> Result<ClMem, XeError> {
        live_buffer(self.output, "output")
    }
}

fn validation_class(
    mode: AttachmentMode,
    exact_validated_stack: bool,
) -> Result<ValidationClass, XeError> {
    match (mode, exact_validated_stack) {
        (AttachmentMode::Automatic, true) => Ok(ValidationClass::ValidatedAutomatic),
        (AttachmentMode::Explicit, true) => Ok(ValidationClass::ValidatedExplicit),
        (AttachmentMode::Explicit, false) => Ok(ValidationClass::UnvalidatedExplicit),
        (AttachmentMode::Automatic, false) => Err(XeError::Capability(
            "automatic Xe selection requires the exact checked-in X8 driver/library identity"
                .into(),
        )),
    }
}

impl ProjectionRuntime for OpenClRuntime {
    fn descriptor(&self) -> &XeDescriptor {
        &self.descriptor
    }

    fn project(
        &mut self,
        request: &ProjectionRequest<'_>,
        variant: KernelVariant,
        output: &mut [f32],
    ) -> Result<PhaseTiming, XeError> {
        if self.shutdown {
            return Err(XeError::Shutdown("OpenCL runtime is shut down".into()));
        }
        self.project_impl(request, variant, output)
    }

    fn drain(&mut self) -> Result<(), XeError> {
        if self.shutdown {
            return Ok(());
        }
        self.finish_queue(self.queue()?)
    }

    fn shutdown(&mut self) -> Result<(), XeError> {
        if self.shutdown {
            return Ok(());
        }
        let mut first_error = self.drain().err();
        for buffer in [
            &mut self.output,
            &mut self.activation,
            &mut self.bias,
            &mut self.weight,
        ] {
            if let Some(buffer) = buffer.take() {
                // SAFETY: queue is drained and each owned buffer is released once.
                let status = unsafe { (self.api.release_mem_object)(buffer) };
                if status != CL_SUCCESS && first_error.is_none() {
                    first_error = Some(runtime_status("clReleaseMemObject", status));
                }
            }
        }
        for kernel in self.kernels.iter_mut().rev() {
            if let Some(kernel) = kernel.take() {
                // SAFETY: queue is drained and each owned kernel is released once.
                let status = unsafe { (self.api.release_kernel)(kernel) };
                if status != CL_SUCCESS && first_error.is_none() {
                    first_error = Some(runtime_status("clReleaseKernel", status));
                }
            }
        }
        if let Some(program) = self.program.take() {
            // SAFETY: kernels are released and program is owned here.
            let status = unsafe { (self.api.release_program)(program) };
            if status != CL_SUCCESS && first_error.is_none() {
                first_error = Some(runtime_status("clReleaseProgram", status));
            }
        }
        if let Some(queue) = self.queue.take() {
            // SAFETY: queue is drained and owned here.
            let status = unsafe { (self.api.release_command_queue)(queue) };
            if status != CL_SUCCESS && first_error.is_none() {
                first_error = Some(runtime_status("clReleaseCommandQueue", status));
            }
        }
        if let Some(context) = self.context.take() {
            // SAFETY: all child objects are released and context is owned here.
            let status = unsafe { (self.api.release_context)(context) };
            if status != CL_SUCCESS && first_error.is_none() {
                first_error = Some(runtime_status("clReleaseContext", status));
            }
        }
        self.shutdown = true;
        first_error.map_or(Ok(()), Err)
    }
}

impl Drop for OpenClRuntime {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

fn select_device(api: &OpenClApi) -> Result<DeviceFacts, XeError> {
    let mut platform_count = 0;
    // SAFETY: count is writable and the null list requests its size.
    check("clGetPlatformIDs(count)", unsafe {
        (api.get_platform_ids)(0, ptr::null_mut(), &mut platform_count)
    })?;
    if platform_count == 0 {
        return Err(XeError::Unsupported("no OpenCL platforms found".into()));
    }
    let mut platforms = vec![ptr::null_mut(); platform_count as usize];
    // SAFETY: platform storage matches the queried count.
    check("clGetPlatformIDs(list)", unsafe {
        (api.get_platform_ids)(platform_count, platforms.as_mut_ptr(), ptr::null_mut())
    })?;
    let mut all_devices = Vec::new();
    for platform in platforms {
        let mut device_count = 0;
        // SAFETY: count is writable and null list requests its size.
        let status = unsafe {
            (api.get_device_ids)(
                platform,
                CL_DEVICE_TYPE_GPU,
                0,
                ptr::null_mut(),
                &mut device_count,
            )
        };
        if status == CL_DEVICE_NOT_FOUND || device_count == 0 {
            continue;
        }
        check("clGetDeviceIDs(count)", status)?;
        let mut devices = vec![ptr::null_mut(); device_count as usize];
        // SAFETY: device storage matches the queried count.
        check("clGetDeviceIDs(list)", unsafe {
            (api.get_device_ids)(
                platform,
                CL_DEVICE_TYPE_GPU,
                device_count,
                devices.as_mut_ptr(),
                ptr::null_mut(),
            )
        })?;
        all_devices.extend(devices.into_iter().map(|device| (platform, device)));
    }
    validate_device_selection(all_devices.len(), None)?;
    let (platform, device) = all_devices[0];
    let vendor: u32 = device_scalar(api, device, CL_DEVICE_VENDOR_ID)?;
    let device_id: u32 = device_scalar(api, device, CL_DEVICE_ID_INTEL)?;
    validate_device_selection(all_devices.len(), Some((vendor, device_id)))?;
    let subgroup_sizes = device_vec::<usize>(api, device, CL_DEVICE_SUB_GROUP_SIZES_INTEL)?;
    let _platform_name = platform_string(api, platform, CL_PLATFORM_NAME)?;
    let _device_name = device_string(api, device, CL_DEVICE_NAME)?;
    Ok(DeviceFacts {
        device,
        driver_version: device_string(api, device, CL_DRIVER_VERSION)?,
        device_version: device_string(api, device, CL_DEVICE_VERSION)?,
        extensions: device_string(api, device, CL_DEVICE_EXTENSIONS)?,
        compiler_available: device_scalar::<ClBool>(api, device, CL_DEVICE_COMPILER_AVAILABLE)?
            == CL_TRUE,
        integrated: device_scalar::<ClBool>(api, device, CL_DEVICE_HOST_UNIFIED_MEMORY)? == CL_TRUE,
        subgroup_sizes,
        max_group_size: device_scalar(api, device, CL_DEVICE_MAX_WORK_GROUP_SIZE)?,
        global_memory_bytes: device_scalar(api, device, CL_DEVICE_GLOBAL_MEM_SIZE)?,
        max_allocation_bytes: device_scalar(api, device, CL_DEVICE_MAX_MEM_ALLOC_SIZE)?,
        local_memory_bytes: device_scalar(api, device, CL_DEVICE_LOCAL_MEM_SIZE)?,
    })
}

fn validate_device_selection(count: usize, identity: Option<(u32, u32)>) -> Result<(), XeError> {
    if count != 1 {
        return Err(XeError::Capability(format!(
            "production Xe attachment requires exactly one OpenCL GPU device, found {count}"
        )));
    }
    if let Some((vendor, device)) = identity {
        if vendor != EXPECTED_VENDOR_ID || device != EXPECTED_DEVICE_ID {
            return Err(XeError::Capability(format!(
                "OpenCL GPU is {vendor:04x}:{device:04x}; expected 8086:9a49"
            )));
        }
    }
    Ok(())
}

fn validate_capabilities(facts: &DeviceFacts) -> Result<(), XeError> {
    if !facts.integrated {
        return Err(XeError::Capability(
            "8086:9a49 did not report unified host/device memory".into(),
        ));
    }
    if !facts.subgroup_sizes.contains(&32) {
        return Err(XeError::Capability(
            "required subgroup size 32 is unavailable".into(),
        ));
    }
    if !facts
        .extensions
        .split_ascii_whitespace()
        .any(|extension| extension == "cl_khr_integer_dot_product")
    {
        return Err(XeError::Capability(
            "cl_khr_integer_dot_product is unavailable".into(),
        ));
    }
    if facts.max_group_size < WORKGROUP_SIZE
        || facts.global_memory_bytes == 0
        || facts.max_allocation_bytes == 0
        || facts.local_memory_bytes == 0
    {
        return Err(XeError::Capability(
            "OpenCL workgroup or allocation limits are insufficient".into(),
        ));
    }
    Ok(())
}

fn create_context(api: &OpenClApi, device: ClDeviceId) -> Result<ClContext, XeError> {
    let mut status = CL_SUCCESS;
    // SAFETY: device is selected/live; status is writable; no callback retained.
    let context = unsafe {
        (api.create_context)(ptr::null(), 1, &device, None, ptr::null_mut(), &mut status)
    };
    if context.is_null() || status != CL_SUCCESS {
        return Err(runtime_status("clCreateContext", status));
    }
    Ok(context)
}

fn create_queue(
    api: &OpenClApi,
    context: ClContext,
    device: ClDeviceId,
) -> Result<ClCommandQueue, XeError> {
    let properties = [CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0];
    let mut status = CL_SUCCESS;
    // SAFETY: context/device are live and properties are terminated.
    let queue = unsafe {
        (api.create_command_queue_with_properties)(
            context,
            device,
            properties.as_ptr(),
            &mut status,
        )
    };
    if queue.is_null() || status != CL_SUCCESS {
        return Err(runtime_status("clCreateCommandQueueWithProperties", status));
    }
    Ok(queue)
}

fn create_program_source(
    api: &OpenClApi,
    context: ClContext,
    device: ClDeviceId,
) -> Result<ClProgram, XeError> {
    let source = KERNEL_SOURCE.as_ptr().cast::<c_char>();
    let length = KERNEL_SOURCE.len();
    let mut status = CL_SUCCESS;
    // SAFETY: source bytes/length remain live through creation.
    let program =
        unsafe { (api.create_program_with_source)(context, 1, &source, &length, &mut status) };
    if program.is_null() || status != CL_SUCCESS {
        return Err(runtime_status("clCreateProgramWithSource", status));
    }
    match build_program(api, program, device) {
        Ok(()) => Ok(program),
        Err(error) => {
            // SAFETY: program was created and is owned here.
            unsafe { (api.release_program)(program) };
            Err(error)
        }
    }
}

fn create_program_binary(
    api: &OpenClApi,
    context: ClContext,
    device: ClDeviceId,
    binary: &[u8],
) -> Result<ClProgram, XeError> {
    let length = binary.len();
    let pointer = binary.as_ptr();
    let mut binary_status = CL_SUCCESS;
    let mut status = CL_SUCCESS;
    // SAFETY: device/binary arrays each have one live element.
    let program = unsafe {
        (api.create_program_with_binary)(
            context,
            1,
            &device,
            &length,
            &pointer,
            &mut binary_status,
            &mut status,
        )
    };
    if program.is_null() || status != CL_SUCCESS || binary_status != CL_SUCCESS {
        return Err(runtime_status(
            "clCreateProgramWithBinary",
            if status != CL_SUCCESS {
                status
            } else {
                binary_status
            },
        ));
    }
    match build_program(api, program, device) {
        Ok(()) => Ok(program),
        Err(error) => {
            // SAFETY: program was created and is owned here.
            unsafe { (api.release_program)(program) };
            Err(error)
        }
    }
}

fn build_program(api: &OpenClApi, program: ClProgram, device: ClDeviceId) -> Result<(), XeError> {
    let options = CString::new(BUILD_OPTIONS).expect("static build options contain no NUL");
    // SAFETY: program/device/options are live for the synchronous build call.
    let status = unsafe {
        (api.build_program)(program, 1, &device, options.as_ptr(), None, ptr::null_mut())
    };
    if status == CL_SUCCESS {
        return Ok(());
    }
    let log = program_build_log(api, program, device).unwrap_or_default();
    Err(XeError::Runtime(format!(
        "clBuildProgram failed with status {status}; build log: {}",
        bounded_text(&log, 2048)
    )))
}

fn create_kernels(api: &OpenClApi, program: ClProgram) -> Result<[ClKernel; 3], XeError> {
    let names = [
        KernelVariant::Tile32M1.entry_point(),
        KernelVariant::Tile32M2.entry_point(),
        KernelVariant::Tile32M4.entry_point(),
    ];
    let mut kernels: [ClKernel; 3] = [ptr::null_mut(); 3];
    for (index, name) in names.into_iter().enumerate() {
        let name = CString::new(name).expect("static entry has no NUL");
        let mut status = CL_SUCCESS;
        // SAFETY: program/name are live and status is writable.
        let kernel = unsafe { (api.create_kernel)(program, name.as_ptr(), &mut status) };
        if kernel.is_null() || status != CL_SUCCESS {
            for kernel in kernels.into_iter().take(index).rev() {
                if !kernel.is_null() {
                    // SAFETY: earlier kernels are owned by this function.
                    unsafe { (api.release_kernel)(kernel) };
                }
            }
            return Err(runtime_status("clCreateKernel", status));
        }
        kernels[index] = kernel;
    }
    Ok(kernels)
}

fn memory_descriptor(config: &AttachConfig) -> Result<XeMemoryDescriptor, XeError> {
    let weight_capacity_bytes = config.max_columns * config.max_blocks * XE_WEIGHT_PLANES;
    let bias_capacity_bytes = config.max_columns * 4;
    let fixed = weight_capacity_bytes + bias_capacity_bytes;
    let row_activation = config.max_blocks * XE_ACTIVATION_RECORD_BYTES;
    let row_output = config.max_columns * 4;
    let row_total = row_activation + row_output;
    let mut max_rows_per_chunk = (config.max_resident_bytes - fixed) / row_total;
    max_rows_per_chunk -= max_rows_per_chunk % 4;
    if max_rows_per_chunk < 4 {
        return Err(XeError::ResidentLimit(
            "resident cap cannot provide a four-row streaming chunk".into(),
        ));
    }
    let activation_capacity_bytes = max_rows_per_chunk * row_activation;
    let output_capacity_bytes = max_rows_per_chunk * row_output;
    let device_resident_bytes = fixed + activation_capacity_bytes + output_capacity_bytes;
    Ok(XeMemoryDescriptor {
        max_resident_bytes: config.max_resident_bytes,
        device_resident_bytes,
        host_staging_bound_bytes: device_resident_bytes,
        weight_capacity_bytes,
        bias_capacity_bytes,
        activation_capacity_bytes,
        output_capacity_bytes,
        max_rows_per_chunk,
    })
}

fn allocate_buffers(
    api: &OpenClApi,
    context: ClContext,
    memory: &XeMemoryDescriptor,
) -> Result<(ClMem, ClMem, ClMem, ClMem), XeError> {
    let sizes = [
        memory.weight_capacity_bytes,
        memory.bias_capacity_bytes,
        memory.activation_capacity_bytes,
        memory.output_capacity_bytes,
    ];
    let mut buffers: [ClMem; 4] = [ptr::null_mut(); 4];
    for (index, size) in sizes.into_iter().enumerate() {
        let mut status = CL_SUCCESS;
        // SAFETY: context is live, status writable, and no host pointer supplied.
        let buffer = unsafe {
            (api.create_buffer)(
                context,
                CL_MEM_READ_WRITE,
                size,
                ptr::null_mut(),
                &mut status,
            )
        };
        if buffer.is_null() || status != CL_SUCCESS {
            for buffer in buffers.into_iter().take(index).rev() {
                if !buffer.is_null() {
                    // SAFETY: earlier buffers are owned by this function.
                    unsafe { (api.release_mem_object)(buffer) };
                }
            }
            return Err(runtime_status("clCreateBuffer", status));
        }
        buffers[index] = buffer;
    }
    Ok((buffers[0], buffers[1], buffers[2], buffers[3]))
}

fn release_queue_context(api: &OpenClApi, queue: ClCommandQueue, context: ClContext) {
    // SAFETY: these handles were created together and no child objects remain.
    unsafe {
        (api.release_command_queue)(queue);
        (api.release_context)(context);
    }
}

fn release_kernels_program_queue_context(
    api: &OpenClApi,
    kernels: [ClKernel; 3],
    program: ClProgram,
    queue: ClCommandQueue,
    context: ClContext,
) {
    // SAFETY: handles are owned by the failed construction path and released
    // in reverse dependency order.
    unsafe {
        for kernel in kernels.into_iter().rev() {
            (api.release_kernel)(kernel);
        }
        (api.release_program)(program);
        (api.release_command_queue)(queue);
        (api.release_context)(context);
    }
}

fn core_libraries(loader_path: &Path) -> Result<CoreLibraries, XeError> {
    let mapped = mapped_libraries()?;
    let loader_path = select_library_path(
        "OpenCL loader",
        &mapped,
        |name| name.starts_with("libOpenCL.so"),
        Some(loader_path),
    )?;
    let driver_path = select_library_path(
        "Intel OpenCL driver",
        &mapped,
        |name| name == "libigdrcl.so",
        Some(Path::new(
            "/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so",
        )),
    )?;
    let igc_path = select_library_path(
        "IGC",
        &mapped,
        |name| name.starts_with("libigc.so"),
        Some(Path::new("/usr/lib/x86_64-linux-gnu/libigc.so.2")),
    )?;
    for path in [&loader_path, &driver_path, &igc_path] {
        let text = path.to_string_lossy();
        if (!text.starts_with("/usr/lib/") && !text.starts_with("/lib/"))
            || text.contains("/xe-research/")
            || text.contains("level-zero-v1.16.1")
        {
            return Err(XeError::Capability(format!(
                "mixed-generation library guard rejected {}",
                path.display()
            )));
        }
    }
    Ok(CoreLibraries {
        loader_sha256: sha256_file(&loader_path)?,
        driver_sha256: sha256_file(&driver_path)?,
        igc_sha256: sha256_file(&igc_path)?,
    })
}

fn mapped_libraries() -> Result<Vec<PathBuf>, XeError> {
    let maps = std::fs::read_to_string("/proc/self/maps")
        .map_err(|error| XeError::Capability(format!("cannot read /proc/self/maps: {error}")))?;
    let mut paths = BTreeSet::new();
    for line in maps.lines() {
        if let Some(path) = line
            .split_whitespace()
            .last()
            .filter(|path| path.starts_with('/'))
        {
            if path.contains("libOpenCL.so")
                || path.contains("libigdrcl.so")
                || path.contains("libigc.so")
            {
                let canonical = Path::new(path)
                    .canonicalize()
                    .unwrap_or_else(|_| PathBuf::from(path));
                paths.insert(canonical);
            }
        }
    }
    Ok(paths.into_iter().collect())
}

fn select_library_path(
    label: &str,
    mapped: &[PathBuf],
    predicate: impl Fn(&str) -> bool,
    fallback: Option<&Path>,
) -> Result<PathBuf, XeError> {
    let mut matches = mapped
        .iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(&predicate)
        })
        .cloned()
        .collect::<BTreeSet<_>>();
    if matches.is_empty() {
        if let Some(path) = fallback.filter(|path| path.exists()) {
            matches.insert(path.canonicalize().map_err(|error| {
                XeError::Capability(format!("canonicalize {}: {error}", path.display()))
            })?);
        }
    }
    if matches.len() != 1 {
        return Err(XeError::Capability(format!(
            "mixed-generation guard expected one {label}, found {}",
            matches.len()
        )));
    }
    Ok(matches.pop_first().expect("one match checked"))
}

fn native_cache_key(facts: &DeviceFacts, libraries: &CoreLibraries) -> String {
    let mut hash = Sha256::new();
    hash.update(b"gpt-oss-rs-xe-native-program-v1");
    hash.update(KERNEL_SOURCE_SHA256.as_bytes());
    hash.update(KERNEL_ABI_SHA256.as_bytes());
    hash.update(BUILD_OPTIONS.as_bytes());
    hash.update(EXPECTED_VENDOR_ID.to_le_bytes());
    hash.update(EXPECTED_DEVICE_ID.to_le_bytes());
    hash.update(facts.driver_version.as_bytes());
    hash.update(libraries.loader_sha256.as_bytes());
    hash.update(libraries.driver_sha256.as_bytes());
    hash.update(libraries.igc_sha256.as_bytes());
    format!("{:x}", hash.finalize())
}

fn cache_manifest(
    key: &str,
    facts: &DeviceFacts,
    libraries: &CoreLibraries,
    native: &[u8],
) -> CacheManifest {
    CacheManifest {
        schema: "gpt-oss-rs.xe-native-cache/v1".into(),
        key: key.into(),
        source_sha256: KERNEL_SOURCE_SHA256.into(),
        abi_sha256: KERNEL_ABI_SHA256.into(),
        build_options: BUILD_OPTIONS.into(),
        pci_vendor_id: format!("{EXPECTED_VENDOR_ID:04x}"),
        pci_device_id: format!("{EXPECTED_DEVICE_ID:04x}"),
        driver_version: facts.driver_version.clone(),
        loader_sha256: libraries.loader_sha256.clone(),
        driver_sha256: libraries.driver_sha256.clone(),
        igc_sha256: libraries.igc_sha256.clone(),
        native_sha256: sha256_bytes(native),
    }
}

fn read_native_cache(
    root: &Path,
    key: &str,
    facts: &DeviceFacts,
    libraries: &CoreLibraries,
) -> Result<Option<Vec<u8>>, XeError> {
    let directory = root.join("xe/native").join(key);
    let manifest_path = directory.join("manifest.json");
    let binary_path = directory.join("program.bin");
    let manifest_bytes = match std::fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(XeError::Artifact(format!(
                "read native cache manifest: {error}"
            )))
        }
    };
    let actual: CacheManifest = match serde_json::from_slice(&manifest_bytes) {
        Ok(manifest) => manifest,
        Err(_) => return Ok(None),
    };
    let binary = match std::fs::read(&binary_path) {
        Ok(bytes) => bytes,
        Err(_) => return Ok(None),
    };
    let expected = cache_manifest(key, facts, libraries, &binary);
    if actual != expected || binary.is_empty() {
        return Ok(None);
    }
    Ok(Some(binary))
}

fn write_native_cache(
    root: &Path,
    key: &str,
    facts: &DeviceFacts,
    libraries: &CoreLibraries,
    binary: &[u8],
) -> Result<(), XeError> {
    let directory = root.join("xe/native").join(key);
    std::fs::create_dir_all(&directory).map_err(|error| {
        XeError::Artifact(format!(
            "create native cache {}: {error}",
            directory.display()
        ))
    })?;
    let manifest = serde_json::to_vec_pretty(&cache_manifest(key, facts, libraries, binary))
        .map_err(|error| XeError::Artifact(format!("serialize native cache: {error}")))?;
    atomic_write(&directory, "program.bin", binary)?;
    atomic_write(&directory, "manifest.json", &manifest)?;
    File::open(&directory)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| XeError::Artifact(format!("sync native cache directory: {error}")))?;
    Ok(())
}

fn atomic_write(directory: &Path, name: &str, bytes: &[u8]) -> Result<(), XeError> {
    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temporary = directory.join(format!(".{name}.{}.{}.tmp", std::process::id(), nonce));
    let target = directory.join(name);
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| XeError::Artifact(format!("create {}: {error}", temporary.display())))?;
    let result = file
        .write_all(bytes)
        .and_then(|_| file.sync_all())
        .and_then(|_| std::fs::rename(&temporary, &target));
    if let Err(error) = result {
        let _ = std::fs::remove_file(&temporary);
        return Err(XeError::Artifact(format!(
            "publish native cache {}: {error}",
            target.display()
        )));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, XeError> {
    let mut file = File::open(path).map_err(|error| {
        XeError::Capability(format!("open library {}: {error}", path.display()))
    })?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            XeError::Capability(format!("hash library {}: {error}", path.display()))
        })?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn device_scalar<T: Copy + Default>(
    api: &OpenClApi,
    device: ClDeviceId,
    property: ClUint,
) -> Result<T, XeError> {
    let mut value = T::default();
    // SAFETY: value is writable for its exact size and device is live.
    check("clGetDeviceInfo(scalar)", unsafe {
        (api.get_device_info)(
            device,
            property,
            std::mem::size_of::<T>(),
            (&mut value as *mut T).cast(),
            ptr::null_mut(),
        )
    })?;
    Ok(value)
}

fn device_vec<T: Copy + Default>(
    api: &OpenClApi,
    device: ClDeviceId,
    property: ClUint,
) -> Result<Vec<T>, XeError> {
    let mut bytes = 0_usize;
    // SAFETY: byte count is writable and null destination queries size.
    check("clGetDeviceInfo(vector size)", unsafe {
        (api.get_device_info)(device, property, 0, ptr::null_mut(), &mut bytes)
    })?;
    if bytes == 0 || !bytes.is_multiple_of(std::mem::size_of::<T>()) {
        return Err(XeError::Capability(
            "OpenCL returned an invalid vector property size".into(),
        ));
    }
    let mut values = vec![T::default(); bytes / std::mem::size_of::<T>()];
    // SAFETY: vector has exactly the queried writable byte capacity.
    check("clGetDeviceInfo(vector)", unsafe {
        (api.get_device_info)(
            device,
            property,
            bytes,
            values.as_mut_ptr().cast(),
            ptr::null_mut(),
        )
    })?;
    Ok(values)
}

fn device_string(api: &OpenClApi, device: ClDeviceId, property: ClUint) -> Result<String, XeError> {
    get_string(|size, destination, returned| {
        // SAFETY: delegated buffer follows get_string's queried-size contract.
        unsafe { (api.get_device_info)(device, property, size, destination, returned) }
    })
}

fn platform_string(
    api: &OpenClApi,
    platform: ClPlatformId,
    property: ClUint,
) -> Result<String, XeError> {
    get_string(|size, destination, returned| {
        // SAFETY: delegated buffer follows get_string's queried-size contract.
        unsafe { (api.get_platform_info)(platform, property, size, destination, returned) }
    })
}

fn get_string(
    mut call: impl FnMut(usize, *mut c_void, *mut usize) -> ClInt,
) -> Result<String, XeError> {
    let mut length = 0_usize;
    check(
        "OpenCL string size query",
        call(0, ptr::null_mut(), &mut length),
    )?;
    if length == 0 {
        return Err(XeError::Capability(
            "OpenCL returned an empty string property".into(),
        ));
    }
    let mut bytes = vec![0_u8; length];
    check(
        "OpenCL string query",
        call(length, bytes.as_mut_ptr().cast(), ptr::null_mut()),
    )?;
    CStr::from_bytes_until_nul(&bytes)
        .map(|text| text.to_string_lossy().into_owned())
        .map_err(|error| XeError::Capability(format!("invalid OpenCL string: {error}")))
}

fn program_build_log(
    api: &OpenClApi,
    program: ClProgram,
    device: ClDeviceId,
) -> Result<String, XeError> {
    let mut length = 0_usize;
    // SAFETY: length is writable; null destination queries size.
    check("clGetProgramBuildInfo(size)", unsafe {
        (api.get_program_build_info)(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            0,
            ptr::null_mut(),
            &mut length,
        )
    })?;
    if length == 0 {
        return Ok(String::new());
    }
    let mut bytes = vec![0_u8; length];
    // SAFETY: bytes matches the queried writable size.
    check("clGetProgramBuildInfo(log)", unsafe {
        (api.get_program_build_info)(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            length,
            bytes.as_mut_ptr().cast(),
            ptr::null_mut(),
        )
    })?;
    Ok(CStr::from_bytes_until_nul(&bytes)
        .map(|text| text.to_string_lossy().into_owned())
        .unwrap_or_else(|_| String::from_utf8_lossy(&bytes).into_owned()))
}

fn check(operation: &str, status: ClInt) -> Result<(), XeError> {
    if status == CL_SUCCESS {
        Ok(())
    } else {
        Err(runtime_status(operation, status))
    }
}

fn runtime_status(operation: &str, status: ClInt) -> XeError {
    XeError::Runtime(format!("{operation} failed with status {status}"))
}

fn live_buffer(buffer: Option<ClMem>, name: &str) -> Result<ClMem, XeError> {
    buffer
        .filter(|handle| !handle.is_null())
        .ok_or_else(|| XeError::Shutdown(format!("OpenCL {name} buffer is closed")))
}

fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

fn bounded_text(value: &str, max: usize) -> &str {
    if value.len() <= max {
        return value;
    }
    let boundary = value
        .char_indices()
        .take_while(|(index, _)| *index <= max)
        .map(|(index, _)| index)
        .last()
        .unwrap_or(0);
    &value[..boundary]
}

const fn e2m1_x2(value: u8) -> i8 {
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12][(value & 0x0f) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_dynamic_loader_is_portably_reported() {
        let error = OpenClApi::load_candidates(&[PathBuf::from(
            "/definitely/missing/gpt-oss-rs/libOpenCL.so",
        )])
        .err()
        .unwrap();
        assert!(matches!(error, XeError::Unsupported(_)));
    }

    #[test]
    fn cache_corruption_and_identity_drift_are_misses() {
        let temp = tempfile::tempdir().unwrap();
        let facts = DeviceFacts {
            device: ptr::null_mut(),
            driver_version: "test-driver".into(),
            device_version: "OpenCL 3.0".into(),
            extensions: "cl_khr_integer_dot_product".into(),
            compiler_available: true,
            integrated: true,
            subgroup_sizes: vec![32],
            max_group_size: 32,
            global_memory_bytes: 1 << 30,
            max_allocation_bytes: 1 << 29,
            local_memory_bytes: 65536,
        };
        let libraries = CoreLibraries {
            loader_sha256: "a".repeat(64),
            driver_sha256: "b".repeat(64),
            igc_sha256: "c".repeat(64),
        };
        let key = native_cache_key(&facts, &libraries);
        write_native_cache(temp.path(), &key, &facts, &libraries, b"native").unwrap();
        assert_eq!(
            read_native_cache(temp.path(), &key, &facts, &libraries)
                .unwrap()
                .unwrap(),
            b"native"
        );
        std::fs::write(
            temp.path().join("xe/native").join(&key).join("program.bin"),
            b"corrupt",
        )
        .unwrap();
        assert!(read_native_cache(temp.path(), &key, &facts, &libraries)
            .unwrap()
            .is_none());
        let mut changed = facts.clone();
        changed.driver_version.push_str("-changed");
        assert!(read_native_cache(temp.path(), &key, &changed, &libraries)
            .unwrap()
            .is_none());
    }

    #[test]
    fn capability_omissions_are_rejected() {
        let facts = DeviceFacts {
            device: ptr::null_mut(),
            driver_version: String::new(),
            device_version: String::new(),
            extensions: String::new(),
            compiler_available: true,
            integrated: true,
            subgroup_sizes: vec![8, 16],
            max_group_size: 512,
            global_memory_bytes: 1,
            max_allocation_bytes: 1,
            local_memory_bytes: 1,
        };
        assert!(matches!(
            validate_capabilities(&facts),
            Err(XeError::Capability(_))
        ));
    }

    #[test]
    fn wrong_identity_and_multiple_devices_are_rejected() {
        assert!(matches!(
            validate_device_selection(2, None),
            Err(XeError::Capability(message)) if message.contains("exactly one")
        ));
        assert!(matches!(
            validate_device_selection(1, Some((0x8086, 0x9a40))),
            Err(XeError::Capability(message)) if message.contains("8086:9a49")
        ));
        validate_device_selection(1, Some((EXPECTED_VENDOR_ID, EXPECTED_DEVICE_ID))).unwrap();
    }

    #[test]
    fn changed_stack_is_explicit_only_and_labeled_unvalidated() {
        assert_eq!(
            validation_class(AttachmentMode::Explicit, false).unwrap(),
            ValidationClass::UnvalidatedExplicit
        );
        assert!(matches!(
            validation_class(AttachmentMode::Automatic, false),
            Err(XeError::Capability(message)) if message.contains("exact checked-in")
        ));
    }

    #[test]
    fn mixed_library_generations_are_rejected() {
        let mapped = [
            PathBuf::from("/usr/lib/x86_64-linux-gnu/libigc.so.2"),
            PathBuf::from("/opt/other/libigc.so.2"),
        ];
        assert!(matches!(
            select_library_path(
                "IGC",
                &mapped,
                |name| name.starts_with("libigc.so"),
                None,
            ),
            Err(XeError::Capability(message)) if message.contains("found 2")
        ));
    }

    #[test]
    fn resident_descriptor_chunks_and_stays_bounded() {
        let config = AttachConfig::new(AttachmentMode::Explicit, ".", 128 * 1024 * 1024, 5760, 90);
        let memory = memory_descriptor(&config).unwrap();
        assert!(memory.max_rows_per_chunk >= 4);
        assert!(memory.max_rows_per_chunk.is_multiple_of(4));
        assert!(memory.device_resident_bytes <= memory.max_resident_bytes);
    }
}
