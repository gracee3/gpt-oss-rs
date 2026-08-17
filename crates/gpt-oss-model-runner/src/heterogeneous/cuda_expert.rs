//! Exact selected-expert CUDA decode primitive.
//!
//! This module is deliberately narrower than the existing CUDA MoE path: one
//! already-selected expert, one BF16 activation row, and native resident
//! MXFP4 blocks/scales. It neither routes nor scans experts and has no fallback
//! to whole-expert FP16 expansion.

use std::sync::Arc;

use cudarc::driver::{
    sys::CUevent_flags, CudaContext, CudaEvent, CudaSlice, CudaStream, LaunchConfig,
    PinnedHostSlice, PushKernelArg,
};
use gpt_oss_core::error::{LLMError, Result};
use gpt_oss_cpu_kernels::{
    accumulate_mxfp4_bf16_block, Mxfp4Block, MXFP4_PACKED_BYTES, QUANT_BLOCK_SIZE,
};
use gpt_oss_gpu::device::{list_devices, resolve_stable_device, StableCudaDeviceId};
use gpt_oss_gpu::event::CorrelatedTimeline;
use gpt_oss_gpu::kernel_loader::{compiled_ptx_dir, KernelLoader};
use gpt_oss_gpu::pinned_memory::BoundedPinnedLease;
use half::bf16;

use super::contract::{
    CanonicalRouteContract, ExpertRepresentationTag, ExpertResultDescriptor,
    ExpertWeightDescriptor, GptOssPhase, PackedRouteDescriptor,
};
use super::placement::{ExpertOwner, GptOssExpertKey};

pub const HIDDEN_SIZE: usize = 2_880;
pub const INTERMEDIATE_SIZE: usize = 2_880;
pub const INPUT_BLOCKS: usize = HIDDEN_SIZE / QUANT_BLOCK_SIZE;
pub const GATE_UP_ROWS: usize = INTERMEDIATE_SIZE * 2;
pub const GATE_UP_BLOCK_BYTES: usize = GATE_UP_ROWS * INPUT_BLOCKS * MXFP4_PACKED_BYTES;
pub const GATE_UP_SCALE_BYTES: usize = GATE_UP_ROWS * INPUT_BLOCKS;
pub const GATE_UP_BIAS_VALUES: usize = GATE_UP_ROWS;
pub const DOWN_BLOCK_BYTES: usize = HIDDEN_SIZE * INPUT_BLOCKS * MXFP4_PACKED_BYTES;
pub const DOWN_SCALE_BYTES: usize = HIDDEN_SIZE * INPUT_BLOCKS;
pub const DOWN_BIAS_VALUES: usize = HIDDEN_SIZE;
pub const GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES: usize = GATE_UP_BLOCK_BYTES
    + GATE_UP_SCALE_BYTES
    + GATE_UP_BIAS_VALUES * size_of::<u16>()
    + DOWN_BLOCK_BYTES
    + DOWN_SCALE_BYTES
    + DOWN_BIAS_VALUES * size_of::<u16>();
pub const GPT_OSS_SELECTED_EXPERT_INPUT_BYTES: usize = HIDDEN_SIZE * size_of::<u16>();
pub const GPT_OSS_SELECTED_EXPERT_SCRATCH_BYTES: usize =
    (GATE_UP_ROWS + INTERMEDIATE_SIZE) * size_of::<u16>();
pub const GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES: usize = HIDDEN_SIZE * size_of::<u16>();
pub const GPT_OSS_SELECTED_EXPERT_TRACE_BYTES: usize = INTERMEDIATE_SIZE * 4 * size_of::<u16>();
/// Persistent executor-owned input, arithmetic scratch, and divergence trace.
/// Result slots are caller-owned and accounted separately by the execution
/// reserve plan.
pub const GPT_OSS_SELECTED_EXPERT_EXECUTOR_BYTES: usize = GPT_OSS_SELECTED_EXPERT_INPUT_BYTES
    + GPT_OSS_SELECTED_EXPERT_SCRATCH_BYTES
    + GPT_OSS_SELECTED_EXPERT_TRACE_BYTES;
pub const GPT_OSS_SELECTED_EXPERT_DEVICE_WORK_BYTES: usize = GPT_OSS_SELECTED_EXPERT_INPUT_BYTES
    + GPT_OSS_SELECTED_EXPERT_SCRATCH_BYTES
    + GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES
    + GPT_OSS_SELECTED_EXPERT_TRACE_BYTES;
pub const GPT_OSS_SELECTED_EXPERT_WORKSPACE_POOL_CLASS_BYTES: usize = 64 * 1024;

const MODULE: &str = "gpt_oss_selected_expert";
const GEMV: &str = "gpt_oss_selected_mxfp4_bf16_gemv_kernel";
const SWIGLU: &str = "gpt_oss_selected_swiglu_bf16_kernel";
const THREADS: u32 = 256;
const SWIGLU_ALPHA: f32 = 1.702;
const SWIGLU_LIMIT: f32 = 7.0;

/// Borrowed native representation for exactly one expert.
#[derive(Debug, Clone, Copy)]
pub struct NativeMxfp4ExpertView<'a> {
    pub key: GptOssExpertKey,
    pub gate_up_blocks: &'a [u8],
    pub gate_up_scales: &'a [u8],
    pub gate_up_bias_bf16_bits: &'a [u16],
    pub down_blocks: &'a [u8],
    pub down_scales: &'a [u8],
    pub down_bias_bf16_bits: &'a [u16],
    pub identity_sha256: &'a str,
}

impl NativeMxfp4ExpertView<'_> {
    fn validate(&self) -> Result<()> {
        for (name, observed, expected) in [
            (
                "gate_up blocks",
                self.gate_up_blocks.len(),
                GATE_UP_BLOCK_BYTES,
            ),
            (
                "gate_up scales",
                self.gate_up_scales.len(),
                GATE_UP_SCALE_BYTES,
            ),
            (
                "gate_up bias",
                self.gate_up_bias_bf16_bits.len(),
                GATE_UP_BIAS_VALUES,
            ),
            ("down blocks", self.down_blocks.len(), DOWN_BLOCK_BYTES),
            ("down scales", self.down_scales.len(), DOWN_SCALE_BYTES),
            (
                "down bias",
                self.down_bias_bf16_bits.len(),
                DOWN_BIAS_VALUES,
            ),
        ] {
            if observed != expected {
                return Err(LLMError::ModelError(format!(
                    "selected expert {name} length {observed} != {expected}"
                )));
            }
        }
        if self.identity_sha256.len() != 64
            || !self
                .identity_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(LLMError::ModelError(
                "selected expert identity is not a SHA-256".into(),
            ));
        }
        Ok(())
    }
}

/// Single-owner resident native MXFP4 weights for one selected expert.
pub struct CudaSelectedExpertWeights {
    descriptor: ExpertWeightDescriptor,
    device: StableCudaDeviceId,
    gate_up_blocks: CudaSlice<u8>,
    gate_up_scales: CudaSlice<u8>,
    gate_up_bias: CudaSlice<u16>,
    down_blocks: CudaSlice<u8>,
    down_scales: CudaSlice<u8>,
    down_bias: CudaSlice<u16>,
}

impl CudaSelectedExpertWeights {
    pub fn descriptor(&self) -> &ExpertWeightDescriptor {
        &self.descriptor
    }

    pub fn device(&self) -> &StableCudaDeviceId {
        &self.device
    }
}

/// Caller-owned device result slot for one selected-expert output row.
///
/// H4 may pool these slots by canonical route result. Keeping the allocation
/// outside the executor prevents an executor-private buffer from becoming an
/// implicit publication or reuse boundary.
pub struct CudaSelectedExpertResultSlot {
    device: StableCudaDeviceId,
    transaction_generation: u64,
    route_contract: Option<CanonicalRouteContract>,
    buffer: CudaSlice<u16>,
}

#[must_use = "a failed result-slot rebind still owns the reusable CUDA slot"]
pub(crate) struct ResultSlotBindFailure {
    pub(crate) error: LLMError,
    pub(crate) result_slot: CudaSelectedExpertResultSlot,
}

impl CudaSelectedExpertResultSlot {
    pub fn device(&self) -> &StableCudaDeviceId {
        &self.device
    }

    pub const fn transaction_generation(&self) -> u64 {
        self.transaction_generation
    }

    pub(crate) const fn route_contract(&self) -> Option<CanonicalRouteContract> {
        self.route_contract
    }

    pub(crate) fn buffer(&self) -> &CudaSlice<u16> {
        &self.buffer
    }

    /// Rebind a caller-owned slot only after the previous job and canonical
    /// relay copy have returned it through their terminal-owning values. The
    /// consuming API makes the drained slot the ownership ticket; no CUDA
    /// allocation occurs while a new route is being admitted.
    /// H7 fail-closed variant: malformed route admission never destroys the
    /// fixed result-slot pool merely because validation rejected a rebind.
    pub(crate) fn bind_drained_for_route_owned(
        mut self,
        transaction_generation: u64,
        route: &PackedRouteDescriptor,
    ) -> std::result::Result<Self, ResultSlotBindFailure> {
        if let Err(error) = self.validate_rebind(transaction_generation, route) {
            return Err(ResultSlotBindFailure {
                error,
                result_slot: self,
            });
        }
        self.transaction_generation = transaction_generation;
        self.route_contract = Some(CanonicalRouteContract::from_packed_route(route));
        Ok(self)
    }

    fn validate_rebind(
        &self,
        transaction_generation: u64,
        route: &PackedRouteDescriptor,
    ) -> Result<()> {
        let route_device = match &route.owner {
            ExpertOwner::LayerOwnerGpu { device } | ExpertOwner::RemoteGpu { device } => device,
            ExpertOwner::Cpu { .. } => {
                return Err(LLMError::GpuError(
                    "selected-expert result slot route has no CUDA owner".into(),
                ));
            }
        };
        if transaction_generation == 0
            || transaction_generation <= self.transaction_generation
            || &self.device != route_device
        {
            return Err(LLMError::GpuError(
                "drained result slot generation/device cannot be rebound".into(),
            ));
        }
        Ok(())
    }
}

/// Whether drain should retain BF16 first-divergence boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedExpertCapture {
    OutputOnly,
    FirstDivergence,
}

/// Deterministic lifecycle faults available only to explicit integration-test
/// builds. These exercise ownership and drain behavior without corrupting a
/// CUDA context to manufacture a driver failure.
#[cfg(feature = "heterogeneous-test-faults")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedExpertInjectedFault {
    SubmitBeforeEnqueue,
    SubmitAfterInputEnqueue,
    SubmitAfterInputEnqueueAndFallbackDrainFailure,
    Drain,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedExpertFirstDivergenceTrace {
    pub gate_up_bf16_bits: Vec<u16>,
    pub scaled_gate_bf16_bits: Vec<u16>,
    pub sigmoid_bf16_bits: Vec<u16>,
    pub glu_bf16_bits: Vec<u16>,
    pub linear_bf16_bits: Vec<u16>,
    pub swiglu_bf16_bits: Vec<u16>,
    pub down_bf16_bits: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectedExpertExecution {
    pub result: ExpertResultDescriptor,
    pub output_bf16_bits: Vec<u16>,
    pub trace: Option<SelectedExpertFirstDivergenceTrace>,
    pub kernel_elapsed_ms: f32,
}

/// H4 output descriptor for a result downloaded directly into a bounded
/// pinned relay lease. No pageable result allocation is made by this path.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedExpertPinnedExecution {
    pub result: ExpertResultDescriptor,
    pub kernel_elapsed_ms: f32,
    pub output_bytes: usize,
}

/// Allocation-free post-dispatch completion record. Route identity is owned
/// by the prepared step and is not cloned or reconstructed here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectedExpertDeviceExecution {
    pub route_contract: CanonicalRouteContract,
    pub kernel_elapsed_ms: f32,
    pub output_bytes: usize,
}

/// Preallocated host evidence for one real selected-expert invocation.
///
/// H6 allocates these records before dispatch. The owned CUDA path moves this
/// storage with the pending job so an asynchronous D2H can never outlive it.
pub(crate) struct SelectedExpertTraceStorage {
    pub(crate) gate_up_bf16_bits: Vec<u16>,
    pub(crate) scaled_gate_bf16_bits: Vec<u16>,
    pub(crate) sigmoid_bf16_bits: Vec<u16>,
    pub(crate) glu_bf16_bits: Vec<u16>,
    pub(crate) linear_bf16_bits: Vec<u16>,
    pub(crate) swiglu_bf16_bits: Vec<u16>,
    pub(crate) down_bf16_bits: Vec<u16>,
}

impl SelectedExpertTraceStorage {
    pub(crate) fn new() -> Self {
        Self {
            gate_up_bf16_bits: vec![0; GATE_UP_ROWS],
            scaled_gate_bf16_bits: vec![0; INTERMEDIATE_SIZE],
            sigmoid_bf16_bits: vec![0; INTERMEDIATE_SIZE],
            glu_bf16_bits: vec![0; INTERMEDIATE_SIZE],
            linear_bf16_bits: vec![0; INTERMEDIATE_SIZE],
            swiglu_bf16_bits: vec![0; INTERMEDIATE_SIZE],
            down_bf16_bits: vec![0; HIDDEN_SIZE],
        }
    }

    pub(crate) fn into_trace(self) -> SelectedExpertFirstDivergenceTrace {
        SelectedExpertFirstDivergenceTrace {
            gate_up_bf16_bits: self.gate_up_bf16_bits,
            scaled_gate_bf16_bits: self.scaled_gate_bf16_bits,
            sigmoid_bf16_bits: self.sigmoid_bf16_bits,
            glu_bf16_bits: self.glu_bf16_bits,
            linear_bf16_bits: self.linear_bf16_bits,
            swiglu_bf16_bits: self.swiglu_bf16_bits,
            down_bf16_bits: self.down_bf16_bits,
        }
    }
}

/// Successful owned device-input execution. The route-bound result allocation
/// returns only after every kernel and evidence D2H is terminal.
pub(crate) struct OwnedSelectedExpertExecution {
    pub route_contract: CanonicalRouteContract,
    pub result_slot: CudaSelectedExpertResultSlot,
    pub trace: SelectedExpertFirstDivergenceTrace,
    pub kernel_elapsed_ms: f32,
    pub input_d2d_bytes: usize,
    pub input_h2d_bytes: usize,
    pub output_d2h_bytes: usize,
}

/// Successful output-only owned execution used by the all-layer H7 control.
/// The result slot returns only after the selected-expert terminal and any
/// requested pinned result D2H are proven complete.
pub(crate) struct OwnedSelectedExpertOutput {
    pub route_contract: CanonicalRouteContract,
    pub result_slot: CudaSelectedExpertResultSlot,
    pub kernel_elapsed_ms: f32,
    pub input_d2d_bytes: usize,
    pub input_h2d_bytes: usize,
    pub output_d2h_bytes: usize,
}

/// Typed failure for the owned H6 path.
///
/// A proven drain returns its result slot for a deterministic retry. An
/// uncertain drain never returns storage: `into_parts` leaks the Arc-retained
/// input/weights, result slot, and host evidence while the outer coordinator
/// quarantines the executor/model and shell state.
#[must_use = "owned selected-expert failures must be classified before cleanup"]
pub(crate) struct OwnedSelectedExpertFailure {
    pub error: LLMError,
    drain_proven: bool,
    result_slot: Option<CudaSelectedExpertResultSlot>,
    weights: Option<Arc<CudaSelectedExpertWeights>>,
    device_input: Option<Arc<CudaSlice<u16>>>,
    trace: Option<SelectedExpertTraceStorage>,
    pinned_input_may_be_referenced: bool,
    pinned_output_may_be_referenced: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct OwnedSelectedExpertRetention {
    pub(crate) result_slot: bool,
    pub(crate) weights: bool,
    pub(crate) device_input: bool,
    pub(crate) trace: bool,
}

impl OwnedSelectedExpertRetention {
    #[cfg(feature = "heterogeneous-test-faults")]
    pub(crate) const fn all_device_owned(self) -> bool {
        self.result_slot && self.weights && self.device_input && self.trace
    }
}

impl OwnedSelectedExpertFailure {
    pub(crate) fn into_parts(
        mut self,
    ) -> (
        LLMError,
        Option<CudaSelectedExpertResultSlot>,
        bool,
        bool,
        OwnedSelectedExpertRetention,
    ) {
        let drain_proven = self.drain_proven;
        let pinned = self.pinned_input_may_be_referenced || self.pinned_output_may_be_referenced;
        let retained = OwnedSelectedExpertRetention {
            result_slot: !drain_proven && self.result_slot.is_some(),
            weights: !drain_proven && self.weights.is_some(),
            device_input: !drain_proven && self.device_input.is_some(),
            trace: !drain_proven && self.trace.is_some(),
        };
        let slot = if drain_proven {
            self.result_slot.take()
        } else {
            if let Some(slot) = self.result_slot.take() {
                std::mem::forget(slot);
            }
            if let Some(weights) = self.weights.take() {
                std::mem::forget(weights);
            }
            if let Some(input) = self.device_input.take() {
                std::mem::forget(input);
            }
            if let Some(trace) = self.trace.take() {
                std::mem::forget(trace);
            }
            None
        };
        let error = std::mem::replace(
            &mut self.error,
            LLMError::GpuError("consumed owned selected-expert failure".into()),
        );
        (error, slot, drain_proven, pinned, retained)
    }
}

impl Drop for OwnedSelectedExpertFailure {
    fn drop(&mut self) {
        if !self.drain_proven {
            if let Some(slot) = self.result_slot.take() {
                std::mem::forget(slot);
            }
            if let Some(weights) = self.weights.take() {
                std::mem::forget(weights);
            }
            if let Some(input) = self.device_input.take() {
                std::mem::forget(input);
            }
            if let Some(trace) = self.trace.take() {
                std::mem::forget(trace);
            }
        }
    }
}

/// Owns the one-stream, bounded scratch needed by a selected-expert decode.
pub struct CudaSelectedExpertExecutor {
    stable_device: StableCudaDeviceId,
    stream: Arc<CudaStream>,
    loader: KernelLoader,
    input: CudaSlice<u16>,
    gate_up: CudaSlice<u16>,
    swiglu: CudaSlice<u16>,
    scaled_gate_trace: CudaSlice<u16>,
    sigmoid_trace: CudaSlice<u16>,
    glu_trace: CudaSlice<u16>,
    linear_trace: CudaSlice<u16>,
    owned_drain_unproven: bool,
    #[cfg(feature = "heterogeneous-test-faults")]
    injected_fault: Option<SelectedExpertInjectedFault>,
    #[cfg(feature = "heterogeneous-test-faults")]
    injected_fault_delay: usize,
    #[cfg(feature = "heterogeneous-test-faults")]
    last_post_enqueue_fault_drained: bool,
}

impl CudaSelectedExpertExecutor {
    pub fn new(stable_device: StableCudaDeviceId) -> Result<Self> {
        let resolved = resolve_stable_device(&stable_device, &list_devices())
            .map_err(|error| LLMError::GpuError(format!("stable CUDA device: {error}")))?;
        let context = CudaContext::new(resolved.transient_ordinal)
            .map_err(|error| LLMError::GpuError(format!("CUDA context: {error}")))?;
        let stream = context
            .new_stream()
            .map_err(|error| LLMError::GpuError(format!("selected expert stream: {error}")))?;
        let loader = KernelLoader::new(
            Arc::clone(&context),
            Arc::clone(&stream),
            compiled_ptx_dir(),
        )?;
        if !loader.has_func(MODULE, GEMV) || !loader.has_func(MODULE, SWIGLU) {
            return Err(LLMError::GpuError(
                "selected-expert PTX functions are unavailable".into(),
            ));
        }
        let input = stream
            .alloc_zeros::<u16>(HIDDEN_SIZE)
            .map_err(cuda_error("selected expert input allocation"))?;
        let gate_up = stream
            .alloc_zeros::<u16>(GATE_UP_ROWS)
            .map_err(cuda_error("selected expert gate/up allocation"))?;
        let swiglu = stream
            .alloc_zeros::<u16>(INTERMEDIATE_SIZE)
            .map_err(cuda_error("selected expert SwiGLU allocation"))?;
        let scaled_gate_trace = stream
            .alloc_zeros::<u16>(INTERMEDIATE_SIZE)
            .map_err(cuda_error("selected expert scaled-gate trace allocation"))?;
        let sigmoid_trace = stream
            .alloc_zeros::<u16>(INTERMEDIATE_SIZE)
            .map_err(cuda_error("selected expert sigmoid trace allocation"))?;
        let glu_trace = stream
            .alloc_zeros::<u16>(INTERMEDIATE_SIZE)
            .map_err(cuda_error("selected expert GLU trace allocation"))?;
        let linear_trace = stream
            .alloc_zeros::<u16>(INTERMEDIATE_SIZE)
            .map_err(cuda_error("selected expert linear trace allocation"))?;
        Ok(Self {
            stable_device,
            stream,
            loader,
            input,
            gate_up,
            swiglu,
            scaled_gate_trace,
            sigmoid_trace,
            glu_trace,
            linear_trace,
            owned_drain_unproven: false,
            #[cfg(feature = "heterogeneous-test-faults")]
            injected_fault: None,
            #[cfg(feature = "heterogeneous-test-faults")]
            injected_fault_delay: 0,
            #[cfg(feature = "heterogeneous-test-faults")]
            last_post_enqueue_fault_drained: false,
        })
    }

    pub fn stable_device(&self) -> &StableCudaDeviceId {
        &self.stable_device
    }

    /// Construction-only access to the executor's private stream.
    ///
    /// H3 uses this to materialize immutable weights into the same context as
    /// the H2 executor. Execution modules outside this crate do not receive a
    /// stream handle, so stream/event ownership remains behind the narrow
    /// heterogeneous boundary.
    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub const fn scratch_bytes(&self) -> usize {
        GPT_OSS_SELECTED_EXPERT_SCRATCH_BYTES
    }

    /// Driver-visible free/total bytes for bounded gate evidence.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        self.stream
            .context()
            .bind_to_thread()
            .map_err(cuda_error("selected expert memory-info bind"))?;
        cudarc::driver::result::mem_get_info()
            .map_err(cuda_error("selected expert memory-info query"))
    }

    /// Allocate one bounded result slot for ownership by the caller's route
    /// result pool. Allocation is never performed during `prepare` or `submit`.
    pub fn allocate_result_slot(&self) -> Result<CudaSelectedExpertResultSlot> {
        self.allocate_result_slot_for_generation(0)
    }

    /// Allocate a result slot bound to one prepared heterogeneous step. A
    /// nonzero generation is required by the H5 canonical relay; generation
    /// zero remains available only to detached H2/H4 primitive controls.
    pub fn allocate_result_slot_for_generation(
        &self,
        transaction_generation: u64,
    ) -> Result<CudaSelectedExpertResultSlot> {
        self.allocate_result_slot_inner(transaction_generation, None)
    }

    /// Allocate a production H5 slot bound to the exact route identity that
    /// will populate it. The relay rejects unbound or relabeled slots.
    pub fn allocate_result_slot_for_route(
        &self,
        transaction_generation: u64,
        route: &PackedRouteDescriptor,
    ) -> Result<CudaSelectedExpertResultSlot> {
        if transaction_generation == 0 {
            return Err(LLMError::GpuError(
                "route-bound result slot generation must be nonzero".into(),
            ));
        }
        self.allocate_result_slot_inner(
            transaction_generation,
            Some(CanonicalRouteContract::from_packed_route(route)),
        )
    }

    fn allocate_result_slot_inner(
        &self,
        transaction_generation: u64,
        route_contract: Option<CanonicalRouteContract>,
    ) -> Result<CudaSelectedExpertResultSlot> {
        let buffer = self
            .stream
            .alloc_zeros::<u16>(HIDDEN_SIZE)
            .map_err(cuda_error("selected expert result-slot allocation"))?;
        Ok(CudaSelectedExpertResultSlot {
            device: self.stable_device.clone(),
            transaction_generation,
            route_contract,
            buffer,
        })
    }

    /// Arm one deterministic fault for the next submitted job.
    #[cfg(feature = "heterogeneous-test-faults")]
    pub fn inject_next_failure(&mut self, fault: SelectedExpertInjectedFault) -> Result<()> {
        self.inject_failure_after_successful_submissions(fault, 0)
    }

    /// Arm one deterministic fault after `successful_submissions_before_fault`
    /// earlier jobs have consumed this capacity-one executor. This exists only
    /// to exercise cleanup after a multi-result dispatch is partially complete.
    #[cfg(feature = "heterogeneous-test-faults")]
    pub fn inject_failure_after_successful_submissions(
        &mut self,
        fault: SelectedExpertInjectedFault,
        successful_submissions_before_fault: usize,
    ) -> Result<()> {
        if self.injected_fault.is_some() {
            return Err(LLMError::GpuError(
                "selected expert already has an armed test fault".into(),
            ));
        }
        self.injected_fault = Some(fault);
        self.injected_fault_delay = successful_submissions_before_fault;
        self.last_post_enqueue_fault_drained = false;
        Ok(())
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    fn take_injected_fault(&mut self) -> Option<SelectedExpertInjectedFault> {
        if self.injected_fault.is_some() && self.injected_fault_delay > 0 {
            self.injected_fault_delay -= 1;
            None
        } else {
            self.injected_fault.take()
        }
    }

    /// Whether the most recent injected post-enqueue submit failure completed
    /// its mandatory stream drain before returning its borrows.
    #[cfg(feature = "heterogeneous-test-faults")]
    pub fn last_post_enqueue_fault_drained(&self) -> bool {
        self.last_post_enqueue_fault_drained
    }

    pub fn upload_expert(
        &self,
        owner: ExpertOwner,
        source: NativeMxfp4ExpertView<'_>,
    ) -> Result<CudaSelectedExpertWeights> {
        source.validate()?;
        match &owner {
            ExpertOwner::LayerOwnerGpu { device } | ExpertOwner::RemoteGpu { device }
                if device == &self.stable_device => {}
            _ => {
                return Err(LLMError::GpuError(
                    "selected expert owner does not match executor device".into(),
                ));
            }
        }
        let upload = |label: &'static str, data: &[u8]| {
            self.stream.clone_htod(data).map_err(cuda_error(label))
        };
        let upload_u16 = |label: &'static str, data: &[u16]| {
            self.stream.clone_htod(data).map_err(cuda_error(label))
        };
        let weights = CudaSelectedExpertWeights {
            descriptor: ExpertWeightDescriptor {
                key: source.key,
                owner,
                representation: ExpertRepresentationTag::CudaNativeMxfp4BlocksScalesV1,
                payload_bytes: GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES as u64,
                identity_sha256: source.identity_sha256.to_ascii_lowercase(),
            },
            device: self.stable_device.clone(),
            gate_up_blocks: upload("selected expert gate/up blocks", source.gate_up_blocks)?,
            gate_up_scales: upload("selected expert gate/up scales", source.gate_up_scales)?,
            gate_up_bias: upload_u16(
                "selected expert gate/up bias",
                source.gate_up_bias_bf16_bits,
            )?,
            down_blocks: upload("selected expert down blocks", source.down_blocks)?,
            down_scales: upload("selected expert down scales", source.down_scales)?,
            down_bias: upload_u16("selected expert down bias", source.down_bias_bf16_bits)?,
        };
        self.stream
            .synchronize()
            .map_err(cuda_error("selected expert upload synchronize"))?;
        Ok(weights)
    }

    /// Upload one immutable expert through a caller-owned bounded pinned lease.
    ///
    /// H3 uses this construction path so every blocks/scales/bias surface is
    /// staged without creating a whole-expert host allocation. The lease is
    /// reused only after a stream synchronization and never becomes part of
    /// the resident weight handle.
    pub(crate) fn upload_expert_staged(
        &self,
        owner: ExpertOwner,
        source: NativeMxfp4ExpertView<'_>,
        pinned: &mut PinnedHostSlice<u8>,
    ) -> Result<CudaSelectedExpertWeights> {
        self.upload_expert_staged_classified(owner, source, pinned)
            .map_err(|failure| failure.error)
    }

    /// Construction upload with the only error classification relevant to a
    /// source-mapping lifetime: whether an H2D enqueue may have occurred
    /// without a successful stream synchronization.
    pub(crate) fn upload_expert_staged_classified(
        &self,
        owner: ExpertOwner,
        source: NativeMxfp4ExpertView<'_>,
        pinned: &mut PinnedHostSlice<u8>,
    ) -> std::result::Result<CudaSelectedExpertWeights, StagedExpertUploadFailure> {
        source
            .validate()
            .map_err(StagedExpertUploadFailure::before_enqueue)?;
        match &owner {
            ExpertOwner::LayerOwnerGpu { device } | ExpertOwner::RemoteGpu { device }
                if device == &self.stable_device => {}
            _ => {
                return Err(StagedExpertUploadFailure::before_enqueue(
                    LLMError::GpuError(
                        "selected expert owner does not match executor device".into(),
                    ),
                ));
            }
        }
        let gate_up_blocks =
            upload_pinned_u8_classified(&self.stream, pinned, source.gate_up_blocks)?;
        let gate_up_scales =
            upload_pinned_u8_classified(&self.stream, pinned, source.gate_up_scales)?;
        let gate_up_bias =
            upload_pinned_u16_classified(&self.stream, pinned, source.gate_up_bias_bf16_bits)?;
        let down_blocks = upload_pinned_u8_classified(&self.stream, pinned, source.down_blocks)?;
        let down_scales = upload_pinned_u8_classified(&self.stream, pinned, source.down_scales)?;
        let down_bias =
            upload_pinned_u16_classified(&self.stream, pinned, source.down_bias_bf16_bits)?;
        Ok(CudaSelectedExpertWeights {
            descriptor: ExpertWeightDescriptor {
                key: source.key,
                owner,
                representation: ExpertRepresentationTag::CudaNativeMxfp4BlocksScalesV1,
                payload_bytes: GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES as u64,
                identity_sha256: source.identity_sha256.to_ascii_lowercase(),
            },
            device: self.stable_device.clone(),
            gate_up_blocks,
            gate_up_scales,
            gate_up_bias,
            down_blocks,
            down_scales,
            down_bias,
        })
    }

    pub fn execute(
        &mut self,
        phase: GptOssPhase,
        route: &PackedRouteDescriptor,
        weights: &CudaSelectedExpertWeights,
        input_bf16_bits: &[u16],
        result_slot: &mut CudaSelectedExpertResultSlot,
        capture: SelectedExpertCapture,
    ) -> Result<SelectedExpertExecution> {
        self.prepare(phase, route, weights, input_bf16_bits, result_slot)?
            .submit()?
            .drain(capture)
    }

    /// Validate a fixed-shape job without launching or changing device state.
    pub fn prepare<'a>(
        &'a mut self,
        phase: GptOssPhase,
        route: &'a PackedRouteDescriptor,
        weights: &'a CudaSelectedExpertWeights,
        input_bf16_bits: &'a [u16],
        result_slot: &'a mut CudaSelectedExpertResultSlot,
    ) -> Result<PreparedSelectedExpert<'a>> {
        if input_bf16_bits.len() != HIDDEN_SIZE {
            return Err(LLMError::GpuError(format!(
                "selected-expert CUDA v1 input length {} != {HIDDEN_SIZE}",
                input_bf16_bits.len()
            )));
        }
        if input_bf16_bits
            .iter()
            .copied()
            .map(bf16::from_bits)
            .any(|value| !value.to_f32().is_finite())
        {
            return Err(LLMError::GpuError(
                "selected-expert CUDA v1 input contains a non-finite BF16 value".into(),
            ));
        }
        self.prepare_inner(phase, route, weights, input_bf16_bits, result_slot)
    }

    fn prepare_inner<'a>(
        &'a mut self,
        phase: GptOssPhase,
        route: &'a PackedRouteDescriptor,
        weights: &'a CudaSelectedExpertWeights,
        input_bf16_bits: &'a [u16],
        result_slot: &'a mut CudaSelectedExpertResultSlot,
    ) -> Result<PreparedSelectedExpert<'a>> {
        if phase != GptOssPhase::Decode {
            return Err(LLMError::GpuError(
                "selected-expert CUDA v1 supports decode M=1 only".into(),
            ));
        }
        if route.route.expert_id != weights.descriptor.key.expert
            || route.owner != weights.descriptor.owner
            || weights.device != self.stable_device
            || result_slot.device != self.stable_device
            || route.route.source_row != 0
            || route.source_activation_slot != 0
            || route.canonical_result_slot != route.route.canonical_result_slot()
            || result_slot.route_contract.is_some_and(|contract| {
                contract != CanonicalRouteContract::from_packed_route(route)
            })
        {
            return Err(LLMError::GpuError(
                "selected expert route/weight/device identity mismatch".into(),
            ));
        }
        Ok(PreparedSelectedExpert {
            executor: self,
            route,
            weights,
            input_bf16_bits,
            result_slot,
        })
    }

    /// Prepare H6 GPU0-local work from the shell's Arc-retained resident
    /// activation. The source context is read from the `CudaSlice` itself; no
    /// caller-supplied context token can spoof same-context admission.
    pub(crate) fn prepare_owned_device(
        &mut self,
        phase: GptOssPhase,
        route: PackedRouteDescriptor,
        weights: Arc<CudaSelectedExpertWeights>,
        device_input: Arc<CudaSlice<u16>>,
        stable_device: &StableCudaDeviceId,
        result_slot: CudaSelectedExpertResultSlot,
        trace: SelectedExpertTraceStorage,
    ) -> std::result::Result<PreparedOwnedSelectedExpert<'_>, OwnedSelectedExpertFailure> {
        self.prepare_owned_device_inner(
            phase,
            route,
            weights,
            device_input,
            stable_device,
            result_slot,
            Some(trace),
        )
    }

    fn prepare_owned_device_inner(
        &mut self,
        phase: GptOssPhase,
        route: PackedRouteDescriptor,
        weights: Arc<CudaSelectedExpertWeights>,
        device_input: Arc<CudaSlice<u16>>,
        stable_device: &StableCudaDeviceId,
        result_slot: CudaSelectedExpertResultSlot,
        trace: Option<SelectedExpertTraceStorage>,
    ) -> std::result::Result<PreparedOwnedSelectedExpert<'_>, OwnedSelectedExpertFailure> {
        let valid = phase == GptOssPhase::Decode
            && !self.owned_drain_unproven
            && device_input.len() == HIDDEN_SIZE
            && stable_device == &self.stable_device
            && device_input.context().cu_ctx() == self.stream.context().cu_ctx()
            && result_slot.buffer.context().cu_ctx() == self.stream.context().cu_ctx()
            && route.route.expert_id == weights.descriptor.key.expert
            && route.owner == weights.descriptor.owner
            && weights.device == self.stable_device
            && result_slot.device == self.stable_device
            && route.route.source_row == 0
            && route.source_activation_slot == 0
            && route.canonical_result_slot == route.route.canonical_result_slot()
            && result_slot.route_contract.is_some_and(|contract| {
                contract == CanonicalRouteContract::from_packed_route(&route)
            });
        if !valid {
            return Err(OwnedSelectedExpertFailure {
                error: LLMError::GpuError(
                    "owned selected expert route/weight/source/context identity mismatch".into(),
                ),
                drain_proven: true,
                result_slot: Some(result_slot),
                weights: Some(weights),
                device_input: Some(device_input),
                trace,
                pinned_input_may_be_referenced: false,
                pinned_output_may_be_referenced: false,
            });
        }
        Ok(PreparedOwnedSelectedExpert {
            executor: self,
            route,
            weights,
            input: OwnedSelectedExpertInput::Device(device_input),
            result_slot,
            trace,
        })
    }

    /// Prepare a resident same-GPU job without host first-divergence traces.
    /// Every CUDA-referenced object remains owned exactly as in the traced H6
    /// path; only evidence D2H storage is absent.
    pub(crate) fn prepare_owned_device_output_only(
        &mut self,
        phase: GptOssPhase,
        route: PackedRouteDescriptor,
        weights: Arc<CudaSelectedExpertWeights>,
        device_input: Arc<CudaSlice<u16>>,
        stable_device: &StableCudaDeviceId,
        result_slot: CudaSelectedExpertResultSlot,
    ) -> std::result::Result<PreparedOwnedSelectedExpert<'_>, OwnedSelectedExpertFailure> {
        self.prepare_owned_device_inner(
            phase,
            route,
            weights,
            device_input,
            stable_device,
            result_slot,
            None,
        )
    }

    /// Prepare the remote-GPU H6 worker from the single bounded pinned relay
    /// row. The result slot and resident weights are owned by the job; an
    /// uncertain H2D drain is reported so the caller can quarantine the whole
    /// relay reservation that owns `input_bf16_bits`.
    pub(crate) fn prepare_owned_pinned<'a>(
        &'a mut self,
        phase: GptOssPhase,
        route: PackedRouteDescriptor,
        weights: Arc<CudaSelectedExpertWeights>,
        input_bf16_bits: &'a [u16],
        result_slot: CudaSelectedExpertResultSlot,
        trace: SelectedExpertTraceStorage,
    ) -> std::result::Result<PreparedOwnedSelectedExpert<'a>, OwnedSelectedExpertFailure> {
        self.prepare_owned_pinned_inner(
            phase,
            route,
            weights,
            input_bf16_bits,
            result_slot,
            Some(trace),
        )
    }

    fn prepare_owned_pinned_inner<'a>(
        &'a mut self,
        phase: GptOssPhase,
        route: PackedRouteDescriptor,
        weights: Arc<CudaSelectedExpertWeights>,
        input_bf16_bits: &'a [u16],
        result_slot: CudaSelectedExpertResultSlot,
        trace: Option<SelectedExpertTraceStorage>,
    ) -> std::result::Result<PreparedOwnedSelectedExpert<'a>, OwnedSelectedExpertFailure> {
        let valid = phase == GptOssPhase::Decode
            && !self.owned_drain_unproven
            && input_bf16_bits.len() == HIDDEN_SIZE
            && input_bf16_bits
                .iter()
                .copied()
                .map(bf16::from_bits)
                .all(|value| value.to_f32().is_finite())
            && result_slot.buffer.context().cu_ctx() == self.stream.context().cu_ctx()
            && route.route.expert_id == weights.descriptor.key.expert
            && route.owner == weights.descriptor.owner
            && weights.device == self.stable_device
            && result_slot.device == self.stable_device
            && route.route.source_row == 0
            && route.source_activation_slot == 0
            && route.canonical_result_slot == route.route.canonical_result_slot()
            && result_slot.route_contract.is_some_and(|contract| {
                contract == CanonicalRouteContract::from_packed_route(&route)
            });
        if !valid {
            return Err(OwnedSelectedExpertFailure {
                error: LLMError::GpuError(
                    "owned pinned selected expert route/weight/input identity mismatch".into(),
                ),
                drain_proven: true,
                result_slot: Some(result_slot),
                weights: Some(weights),
                device_input: None,
                trace,
                pinned_input_may_be_referenced: false,
                pinned_output_may_be_referenced: false,
            });
        }
        Ok(PreparedOwnedSelectedExpert {
            executor: self,
            route,
            weights,
            input: OwnedSelectedExpertInput::Pinned(input_bf16_bits),
            result_slot,
            trace,
        })
    }

    /// Prepare a pinned remote-GPU job without host first-divergence traces.
    pub(crate) fn prepare_owned_pinned_output_only<'a>(
        &'a mut self,
        phase: GptOssPhase,
        route: PackedRouteDescriptor,
        weights: Arc<CudaSelectedExpertWeights>,
        input_bf16_bits: &'a [u16],
        result_slot: CudaSelectedExpertResultSlot,
    ) -> std::result::Result<PreparedOwnedSelectedExpert<'a>, OwnedSelectedExpertFailure> {
        self.prepare_owned_pinned_inner(phase, route, weights, input_bf16_bits, result_slot, None)
    }

    pub(crate) const fn owned_drain_unproven(&self) -> bool {
        self.owned_drain_unproven
    }
}

pub(crate) struct StagedExpertUploadFailure {
    pub(crate) error: LLMError,
    pub(crate) terminal_unproven: bool,
}

impl StagedExpertUploadFailure {
    fn before_enqueue(error: LLMError) -> Self {
        Self {
            error,
            terminal_unproven: false,
        }
    }

    fn after_enqueue(error: LLMError) -> Self {
        Self {
            error,
            terminal_unproven: true,
        }
    }
}

fn upload_pinned_u8_classified(
    stream: &Arc<CudaStream>,
    pinned: &mut PinnedHostSlice<u8>,
    source: &[u8],
) -> std::result::Result<CudaSlice<u8>, StagedExpertUploadFailure> {
    if source.len() > pinned.len() {
        return Err(StagedExpertUploadFailure::before_enqueue(
            LLMError::GpuError(format!(
                "selected expert surface {} exceeds pinned construction lease {}",
                source.len(),
                pinned.len()
            )),
        ));
    }
    pinned
        .as_mut_slice()
        .map_err(cuda_error("selected expert pinned write access"))
        .map_err(StagedExpertUploadFailure::before_enqueue)?[..source.len()]
        .copy_from_slice(source);
    // SAFETY: the full allocation is initialized by the immediately following
    // copy and synchronized before the pinned lease can be reused.
    let mut destination = unsafe { stream.alloc::<u8>(source.len()) }
        .map_err(cuda_error("selected expert staged allocation"))
        .map_err(StagedExpertUploadFailure::before_enqueue)?;
    let staged = &pinned
        .as_slice()
        .map_err(cuda_error("selected expert pinned read access"))
        .map_err(StagedExpertUploadFailure::before_enqueue)?[..source.len()];
    stream
        .memcpy_htod(staged, &mut destination)
        .map_err(cuda_error("selected expert staged H2D"))
        .map_err(StagedExpertUploadFailure::after_enqueue)?;
    stream
        .synchronize()
        .map_err(cuda_error("selected expert staged H2D drain"))
        .map_err(StagedExpertUploadFailure::after_enqueue)?;
    Ok(destination)
}

fn upload_pinned_u16_classified(
    stream: &Arc<CudaStream>,
    pinned: &mut PinnedHostSlice<u8>,
    source: &[u16],
) -> std::result::Result<CudaSlice<u16>, StagedExpertUploadFailure> {
    let source_bytes = bytemuck::cast_slice(source);
    if source_bytes.len() > pinned.len() {
        return Err(StagedExpertUploadFailure::before_enqueue(
            LLMError::GpuError(format!(
                "selected expert bias surface {} exceeds pinned construction lease {}",
                source_bytes.len(),
                pinned.len()
            )),
        ));
    }
    pinned
        .as_mut_slice()
        .map_err(cuda_error("selected expert pinned bias write access"))
        .map_err(StagedExpertUploadFailure::before_enqueue)?[..source_bytes.len()]
        .copy_from_slice(source_bytes);
    // SAFETY: the full allocation is initialized by the immediately following
    // exact-length copy and synchronized before lease reuse.
    let mut destination = unsafe { stream.alloc::<u16>(source.len()) }
        .map_err(cuda_error("selected expert staged bias allocation"))
        .map_err(StagedExpertUploadFailure::before_enqueue)?;
    let staged_bytes = &pinned
        .as_slice()
        .map_err(cuda_error("selected expert pinned bias read access"))
        .map_err(StagedExpertUploadFailure::before_enqueue)?[..source_bytes.len()];
    let staged: &[u16] = bytemuck::try_cast_slice(staged_bytes)
        .map_err(|error| LLMError::GpuError(format!("selected expert pinned BF16 view: {error}")))
        .map_err(StagedExpertUploadFailure::before_enqueue)?;
    stream
        .memcpy_htod(staged, &mut destination)
        .map_err(cuda_error("selected expert staged bias H2D"))
        .map_err(StagedExpertUploadFailure::after_enqueue)?;
    stream
        .synchronize()
        .map_err(cuda_error("selected expert staged bias H2D drain"))
        .map_err(StagedExpertUploadFailure::after_enqueue)?;
    Ok(destination)
}

/// Validated H6 device-input work that owns every CUDA-referenced object other
/// than the executor itself. The executor remains exclusively borrowed until
/// submission produces a terminal-owning pending job or a classified failure.
pub(crate) struct PreparedOwnedSelectedExpert<'a> {
    executor: &'a mut CudaSelectedExpertExecutor,
    route: PackedRouteDescriptor,
    weights: Arc<CudaSelectedExpertWeights>,
    input: OwnedSelectedExpertInput<'a>,
    result_slot: CudaSelectedExpertResultSlot,
    trace: Option<SelectedExpertTraceStorage>,
}

enum OwnedSelectedExpertInput<'a> {
    Device(Arc<CudaSlice<u16>>),
    Pinned(&'a [u16]),
}

impl<'a> PreparedOwnedSelectedExpert<'a> {
    pub(crate) fn submit_with_timeline(
        self,
        timeline: &CorrelatedTimeline,
        actor: &str,
    ) -> std::result::Result<PendingOwnedSelectedExpert<'a>, OwnedSelectedExpertFailure> {
        let Self {
            executor,
            route,
            weights,
            input,
            mut result_slot,
            trace,
        } = self;
        let (device_input, pinned_input) = match input {
            OwnedSelectedExpertInput::Device(input) => (Some(input), None),
            OwnedSelectedExpertInput::Pinned(input) => (None, Some(input)),
        };
        let input_d2d_bytes =
            usize::from(device_input.is_some()) * GPT_OSS_SELECTED_EXPERT_INPUT_BYTES;
        let input_h2d_bytes =
            usize::from(pinned_input.is_some()) * GPT_OSS_SELECTED_EXPERT_INPUT_BYTES;
        #[cfg(feature = "heterogeneous-test-faults")]
        let injected_fault = executor.take_injected_fault();
        #[cfg(feature = "heterogeneous-test-faults")]
        if injected_fault == Some(SelectedExpertInjectedFault::SubmitBeforeEnqueue) {
            return Err(OwnedSelectedExpertFailure {
                error: LLMError::GpuError(
                    "injected owned selected-expert pre-enqueue submit failure".into(),
                ),
                drain_proven: true,
                result_slot: Some(result_slot),
                weights: Some(weights),
                device_input,
                trace,
                pinned_input_may_be_referenced: false,
                pinned_output_may_be_referenced: false,
            });
        }
        let submitted = (|| -> Result<(CudaEvent, CudaEvent)> {
            match (&device_input, pinned_input) {
                (Some(device_input), None) => {
                    timeline.enqueue_cuda_marker(&executor.stream, actor, "input_d2d_begin")?;
                    executor
                        .stream
                        .memcpy_dtod(device_input.as_ref(), &mut executor.input)
                        .map_err(cuda_error("owned selected expert input D2D"))?;
                    timeline.enqueue_cuda_marker(&executor.stream, actor, "input_d2d_end")?;
                }
                (None, Some(pinned_input)) => {
                    timeline.enqueue_cuda_marker(&executor.stream, actor, "input_h2d_begin")?;
                    executor
                        .stream
                        .memcpy_htod(pinned_input, &mut executor.input)
                        .map_err(cuda_error("owned selected expert input H2D"))?;
                    timeline.enqueue_cuda_marker(&executor.stream, actor, "input_h2d_end")?;
                }
                _ => unreachable!("owned selected expert has exactly one input domain"),
            }
            #[cfg(feature = "heterogeneous-test-faults")]
            if matches!(
                injected_fault,
                Some(SelectedExpertInjectedFault::SubmitAfterInputEnqueue)
                    | Some(
                        SelectedExpertInjectedFault::SubmitAfterInputEnqueueAndFallbackDrainFailure
                    )
            ) {
                return Err(LLMError::GpuError(
                    "injected owned selected-expert post-input-transfer submit failure".into(),
                ));
            }
            let start = executor
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("owned selected expert start event"))?;
            timeline.enqueue_cuda_marker(&executor.stream, actor, "compute_begin")?;
            launch_gemv(
                &executor.stream,
                &executor.loader,
                &executor.input,
                &weights.gate_up_blocks,
                &weights.gate_up_scales,
                &weights.gate_up_bias,
                &mut executor.gate_up,
                GATE_UP_ROWS,
            )?;
            launch_swiglu(
                &executor.stream,
                &executor.loader,
                &executor.gate_up,
                &mut executor.swiglu,
                SwigluTraceDevice {
                    scaled_gate: &mut executor.scaled_gate_trace,
                    sigmoid: &mut executor.sigmoid_trace,
                    glu: &mut executor.glu_trace,
                    linear: &mut executor.linear_trace,
                },
            )?;
            launch_gemv(
                &executor.stream,
                &executor.loader,
                &executor.swiglu,
                &weights.down_blocks,
                &weights.down_scales,
                &weights.down_bias,
                &mut result_slot.buffer,
                HIDDEN_SIZE,
            )?;
            timeline.enqueue_cuda_marker(&executor.stream, actor, "compute_end")?;
            let terminal = executor
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("owned selected expert terminal event"))?;
            Ok((start, terminal))
        })();
        let (start, terminal) = match submitted {
            Ok(events) => events,
            Err(primary) => {
                #[cfg(feature = "heterogeneous-test-faults")]
                if injected_fault
                    == Some(
                        SelectedExpertInjectedFault::SubmitAfterInputEnqueueAndFallbackDrainFailure,
                    )
                {
                    executor.owned_drain_unproven = true;
                    return Err(OwnedSelectedExpertFailure {
                        error: LLMError::GpuError(format!(
                            "owned selected expert submit failed ({primary}); injected mandatory fallback drain failure"
                        )),
                        drain_proven: false,
                        result_slot: Some(result_slot),
                        weights: Some(weights),
                        device_input,
                        trace,
                        pinned_input_may_be_referenced: pinned_input.is_some(),
                        pinned_output_may_be_referenced: false,
                    });
                }
                let drained = executor.stream.synchronize();
                #[cfg(feature = "heterogeneous-test-faults")]
                if injected_fault == Some(SelectedExpertInjectedFault::SubmitAfterInputEnqueue)
                    && drained.is_ok()
                {
                    executor.last_post_enqueue_fault_drained = true;
                }
                if let Err(drain) = drained {
                    executor.owned_drain_unproven = true;
                    return Err(OwnedSelectedExpertFailure {
                        error: LLMError::GpuError(format!(
                            "owned selected expert submit failed ({primary}); mandatory drain failed ({drain})"
                        )),
                        drain_proven: false,
                        result_slot: Some(result_slot),
                        weights: Some(weights),
                        device_input,
                        trace,
                        pinned_input_may_be_referenced: pinned_input.is_some(),
                        pinned_output_may_be_referenced: false,
                    });
                }
                return Err(OwnedSelectedExpertFailure {
                    error: primary,
                    drain_proven: true,
                    result_slot: Some(result_slot),
                    weights: Some(weights),
                    device_input,
                    trace,
                    pinned_input_may_be_referenced: false,
                    pinned_output_may_be_referenced: false,
                });
            }
        };
        Ok(PendingOwnedSelectedExpert {
            executor,
            route,
            weights: Some(weights),
            device_input,
            pinned_input,
            pinned_input_may_be_referenced: pinned_input.is_some(),
            result_slot: Some(result_slot),
            trace,
            start,
            terminal: Some(terminal),
            drained: false,
            input_d2d_bytes,
            input_h2d_bytes,
            #[cfg(feature = "heterogeneous-test-faults")]
            inject_drain_failure: injected_fault == Some(SelectedExpertInjectedFault::Drain),
        })
    }
}

/// Terminal-owning H6 work. All referenced device and host storage is owned by
/// this value until a proven drain; an uncertain drain poisons the executor and
/// transfers storage to a failure that cannot free it accidentally.
pub(crate) struct PendingOwnedSelectedExpert<'a> {
    executor: &'a mut CudaSelectedExpertExecutor,
    route: PackedRouteDescriptor,
    weights: Option<Arc<CudaSelectedExpertWeights>>,
    device_input: Option<Arc<CudaSlice<u16>>>,
    pinned_input: Option<&'a [u16]>,
    pinned_input_may_be_referenced: bool,
    result_slot: Option<CudaSelectedExpertResultSlot>,
    trace: Option<SelectedExpertTraceStorage>,
    start: CudaEvent,
    terminal: Option<CudaEvent>,
    drained: bool,
    input_d2d_bytes: usize,
    input_h2d_bytes: usize,
    #[cfg(feature = "heterogeneous-test-faults")]
    inject_drain_failure: bool,
}

impl PendingOwnedSelectedExpert<'_> {
    pub(crate) fn drain_with_trace(
        self,
        pinned_output: Option<&mut BoundedPinnedLease<u16>>,
        timeline: &CorrelatedTimeline,
        actor: &str,
    ) -> std::result::Result<OwnedSelectedExpertExecution, OwnedSelectedExpertFailure> {
        self.drain_with_trace_at(pinned_output, 0, timeline, actor)
    }

    pub(crate) fn drain_with_trace_at(
        mut self,
        mut pinned_output: Option<&mut BoundedPinnedLease<u16>>,
        pinned_output_slot: u32,
        timeline: &CorrelatedTimeline,
        actor: &str,
    ) -> std::result::Result<OwnedSelectedExpertExecution, OwnedSelectedExpertFailure> {
        let terminal = self.terminal.as_ref().expect("owned terminal is present");
        if let Err(error) = terminal.synchronize() {
            let primary = cuda_error("owned selected expert terminal drain")(error);
            return Err(match self.executor.stream.synchronize() {
                Ok(()) => self.failure(primary, true, false),
                Err(drain) => self.failure(
                    LLMError::GpuError(format!(
                        "owned selected expert terminal failed ({primary}); mandatory drain failed ({drain})"
                    )),
                    false,
                    false,
                ),
            });
        }
        // The input transfer and all kernels are terminal. A later evidence
        // D2H failure cannot retain the source pinned relay row.
        self.pinned_input_may_be_referenced = false;
        self.pinned_input = None;
        let output_start = pinned_output_slot as usize * HIDDEN_SIZE;
        let output_end = output_start.saturating_add(HIDDEN_SIZE);
        if pinned_output
            .as_ref()
            .is_some_and(|output| output.as_slice().len() < output_end)
        {
            return Err(self.failure(
                LLMError::MemoryError("owned selected expert pinned output is undersized".into()),
                true,
                false,
            ));
        }
        #[cfg(feature = "heterogeneous-test-faults")]
        if self.inject_drain_failure {
            return Err(self.failure(
                LLMError::GpuError("injected owned selected-expert drain failure".into()),
                true,
                false,
            ));
        }
        let kernel_elapsed_ms = match self.start.elapsed_ms(terminal) {
            Ok(value) => value,
            Err(error) => {
                return Err(self.failure(
                    cuda_error("owned selected expert event timing")(error),
                    true,
                    false,
                ));
            }
        };
        let submitted = (|| -> Result<(CudaEvent, bool)> {
            let trace = self.trace.as_mut().expect("owned trace is present");
            timeline.enqueue_cuda_marker(&self.executor.stream, actor, "trace_d2h_begin")?;
            self.executor
                .stream
                .memcpy_dtoh(&self.executor.gate_up, &mut trace.gate_up_bf16_bits)
                .map_err(cuda_error("owned selected expert gate/up D2H"))?;
            self.executor
                .stream
                .memcpy_dtoh(
                    &self.executor.scaled_gate_trace,
                    &mut trace.scaled_gate_bf16_bits,
                )
                .map_err(cuda_error("owned selected expert scaled-gate D2H"))?;
            self.executor
                .stream
                .memcpy_dtoh(&self.executor.sigmoid_trace, &mut trace.sigmoid_bf16_bits)
                .map_err(cuda_error("owned selected expert sigmoid D2H"))?;
            self.executor
                .stream
                .memcpy_dtoh(&self.executor.glu_trace, &mut trace.glu_bf16_bits)
                .map_err(cuda_error("owned selected expert GLU D2H"))?;
            self.executor
                .stream
                .memcpy_dtoh(&self.executor.linear_trace, &mut trace.linear_bf16_bits)
                .map_err(cuda_error("owned selected expert linear D2H"))?;
            self.executor
                .stream
                .memcpy_dtoh(&self.executor.swiglu, &mut trace.swiglu_bf16_bits)
                .map_err(cuda_error("owned selected expert SwiGLU D2H"))?;
            let result_slot = self
                .result_slot
                .as_ref()
                .expect("owned result slot is present");
            self.executor
                .stream
                .memcpy_dtoh(&result_slot.buffer, &mut trace.down_bf16_bits)
                .map_err(cuda_error("owned selected expert down D2H"))?;
            let mut pinned_enqueued = false;
            if let Some(output) = pinned_output.as_deref_mut() {
                timeline.enqueue_cuda_marker(&self.executor.stream, actor, "result_d2h_begin")?;
                self.executor
                    .stream
                    .memcpy_dtoh(
                        &result_slot.buffer,
                        &mut output.as_mut_slice()[output_start..output_end],
                    )
                    .map_err(cuda_error("owned selected expert pinned result D2H"))?;
                pinned_enqueued = true;
                timeline.enqueue_cuda_marker(&self.executor.stream, actor, "result_d2h_end")?;
            }
            timeline.enqueue_cuda_marker(&self.executor.stream, actor, "trace_d2h_end")?;
            let evidence_terminal = self
                .executor
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("owned selected expert evidence terminal event"))?;
            Ok((evidence_terminal, pinned_enqueued))
        })();
        let (evidence_terminal, pinned_enqueued) = match submitted {
            Ok(value) => value,
            Err(primary) => {
                return Err(match self.executor.stream.synchronize() {
                    Ok(()) => self.failure(primary, true, false),
                    Err(drain) => self.failure(
                        LLMError::GpuError(format!(
                            "owned selected expert evidence submit failed ({primary}); mandatory drain failed ({drain})"
                        )),
                        false,
                        pinned_output.is_some(),
                    ),
                });
            }
        };
        if let Err(error) = evidence_terminal.synchronize() {
            let primary = cuda_error("owned selected expert evidence terminal drain")(error);
            return Err(match self.executor.stream.synchronize() {
                Ok(()) => self.failure(primary, true, false),
                Err(drain) => self.failure(
                    LLMError::GpuError(format!(
                        "owned selected expert evidence terminal failed ({primary}); mandatory drain failed ({drain})"
                    )),
                    false,
                    pinned_enqueued,
                ),
            });
        }
        self.drained = true;
        let route_contract = CanonicalRouteContract::from_packed_route(&self.route);
        Ok(OwnedSelectedExpertExecution {
            route_contract,
            result_slot: self
                .result_slot
                .take()
                .expect("owned result slot is present"),
            trace: self
                .trace
                .take()
                .expect("owned trace is present")
                .into_trace(),
            kernel_elapsed_ms,
            input_d2d_bytes: self.input_d2d_bytes,
            input_h2d_bytes: self.input_h2d_bytes,
            output_d2h_bytes: if pinned_output.is_some() {
                GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES
            } else {
                0
            },
        })
    }

    /// Drain a selected-expert job without allocating or downloading
    /// first-divergence traces. An optional pinned output is the only host
    /// evidence transfer; the CUDA result slot is always returned after its
    /// terminal event is proven complete.
    pub(crate) fn drain_output_only_at(
        mut self,
        mut pinned_output: Option<&mut BoundedPinnedLease<u16>>,
        pinned_output_slot: u32,
        timeline: &CorrelatedTimeline,
        actor: &str,
    ) -> std::result::Result<OwnedSelectedExpertOutput, OwnedSelectedExpertFailure> {
        let terminal = self.terminal.as_ref().expect("owned terminal is present");
        if let Err(error) = terminal.synchronize() {
            let primary = cuda_error("owned selected expert terminal drain")(error);
            return Err(match self.executor.stream.synchronize() {
                Ok(()) => self.failure(primary, true, false),
                Err(drain) => self.failure(
                    LLMError::GpuError(format!(
                        "owned selected expert terminal failed ({primary}); mandatory drain failed ({drain})"
                    )),
                    false,
                    false,
                ),
            });
        }
        self.pinned_input_may_be_referenced = false;
        self.pinned_input = None;
        if self.trace.is_some() {
            return Err(self.failure(
                LLMError::GpuError(
                    "traced selected-expert job cannot use output-only drain".into(),
                ),
                true,
                false,
            ));
        }
        let output_start = pinned_output_slot as usize * HIDDEN_SIZE;
        let output_end = output_start.saturating_add(HIDDEN_SIZE);
        if pinned_output
            .as_ref()
            .is_some_and(|output| output.as_slice().len() < output_end)
        {
            return Err(self.failure(
                LLMError::MemoryError("owned selected expert pinned output is undersized".into()),
                true,
                false,
            ));
        }
        #[cfg(feature = "heterogeneous-test-faults")]
        if self.inject_drain_failure {
            return Err(self.failure(
                LLMError::GpuError("injected owned selected-expert drain failure".into()),
                true,
                false,
            ));
        }
        let kernel_elapsed_ms = match self.start.elapsed_ms(terminal) {
            Ok(value) => value,
            Err(error) => {
                return Err(self.failure(
                    cuda_error("owned selected expert event timing")(error),
                    true,
                    false,
                ));
            }
        };
        let mut output_enqueued = false;
        if let Some(output) = pinned_output.as_deref_mut() {
            let submitted = (|| -> Result<CudaEvent> {
                timeline.enqueue_cuda_marker(&self.executor.stream, actor, "result_d2h_begin")?;
                let result_slot = self
                    .result_slot
                    .as_ref()
                    .expect("owned result slot is present");
                self.executor
                    .stream
                    .memcpy_dtoh(
                        &result_slot.buffer,
                        &mut output.as_mut_slice()[output_start..output_end],
                    )
                    .map_err(cuda_error("owned selected expert pinned result D2H"))?;
                output_enqueued = true;
                timeline.enqueue_cuda_marker(&self.executor.stream, actor, "result_d2h_end")?;
                self.executor
                    .stream
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(cuda_error("owned selected expert output terminal event"))
            })();
            let output_terminal = match submitted {
                Ok(event) => event,
                Err(primary) => {
                    return Err(match self.executor.stream.synchronize() {
                        Ok(()) => self.failure(primary, true, false),
                        Err(drain) => self.failure(
                            LLMError::GpuError(format!(
                                "owned selected expert output submit failed ({primary}); mandatory drain failed ({drain})"
                            )),
                            false,
                            output_enqueued,
                        ),
                    });
                }
            };
            if let Err(error) = output_terminal.synchronize() {
                let primary = cuda_error("owned selected expert output terminal drain")(error);
                return Err(match self.executor.stream.synchronize() {
                    Ok(()) => self.failure(primary, true, false),
                    Err(drain) => self.failure(
                        LLMError::GpuError(format!(
                            "owned selected expert output terminal failed ({primary}); mandatory drain failed ({drain})"
                        )),
                        false,
                        true,
                    ),
                });
            }
        }
        self.drained = true;
        Ok(OwnedSelectedExpertOutput {
            route_contract: CanonicalRouteContract::from_packed_route(&self.route),
            result_slot: self
                .result_slot
                .take()
                .expect("owned result slot is present"),
            kernel_elapsed_ms,
            input_d2d_bytes: self.input_d2d_bytes,
            input_h2d_bytes: self.input_h2d_bytes,
            output_d2h_bytes: usize::from(pinned_output.is_some())
                * GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES,
        })
    }

    fn failure(
        &mut self,
        error: LLMError,
        drain_proven: bool,
        pinned_output_may_be_referenced: bool,
    ) -> OwnedSelectedExpertFailure {
        // Suppress Drop's second synchronization attempt. The classified
        // failure now owns all movable storage; an uncertain result also marks
        // the executor so its model owner must quarantine the CUDA state.
        self.drained = true;
        if !drain_proven {
            self.executor.owned_drain_unproven = true;
        }
        OwnedSelectedExpertFailure {
            error,
            drain_proven,
            result_slot: self.result_slot.take(),
            weights: self.weights.take(),
            device_input: self.device_input.take(),
            trace: self.trace.take(),
            pinned_input_may_be_referenced: self.pinned_input_may_be_referenced,
            pinned_output_may_be_referenced,
        }
    }
}

impl Drop for PendingOwnedSelectedExpert<'_> {
    fn drop(&mut self) {
        if self.drained {
            return;
        }
        if self.executor.stream.synchronize().is_err() {
            self.executor.owned_drain_unproven = true;
            if let Some(slot) = self.result_slot.take() {
                std::mem::forget(slot);
            }
            if let Some(weights) = self.weights.take() {
                std::mem::forget(weights);
            }
            if let Some(input) = self.device_input.take() {
                std::mem::forget(input);
            }
            if let Some(trace) = self.trace.take() {
                std::mem::forget(trace);
            }
        }
    }
}

/// Validated but not yet enqueued selected-expert work.
pub struct PreparedSelectedExpert<'a> {
    executor: &'a mut CudaSelectedExpertExecutor,
    route: &'a PackedRouteDescriptor,
    weights: &'a CudaSelectedExpertWeights,
    input_bf16_bits: &'a [u16],
    result_slot: &'a mut CudaSelectedExpertResultSlot,
}

impl<'a> PreparedSelectedExpert<'a> {
    /// Enqueue input upload and all three exact kernels, returning the terminal
    /// event owner that must be drained before any borrowed resource can move.
    pub fn submit(self) -> Result<PendingSelectedExpert<'a>> {
        self.submit_inner(None)
    }

    /// Submit with GPU markers on the same process-monotonic timeline used by
    /// CPU work and the layer-owner relay. The markers are evidence only; they
    /// do not change stream dependencies.
    pub fn submit_with_timeline(
        self,
        timeline: &CorrelatedTimeline,
        actor: &str,
    ) -> Result<PendingSelectedExpert<'a>> {
        self.submit_inner(Some((timeline, actor)))
    }

    fn submit_inner(
        self,
        timeline: Option<(&CorrelatedTimeline, &str)>,
    ) -> Result<PendingSelectedExpert<'a>> {
        let executor = self.executor;
        #[cfg(feature = "heterogeneous-test-faults")]
        let injected_fault = executor.take_injected_fault();
        #[cfg(feature = "heterogeneous-test-faults")]
        if injected_fault == Some(SelectedExpertInjectedFault::SubmitBeforeEnqueue) {
            return Err(LLMError::GpuError(
                "injected selected-expert pre-enqueue submit failure".into(),
            ));
        }
        // Once the first enqueue is attempted, every error path synchronizes
        // this private stream before any executor, weights, scratch, or result
        // borrow can be released. This includes H2D, event, and launch errors.
        let submitted = (|| -> Result<(CudaEvent, CudaEvent)> {
            if let Some((timeline, actor)) = timeline {
                timeline.enqueue_cuda_marker(&executor.stream, actor, "input_h2d_begin")?;
            }
            executor
                .stream
                .memcpy_htod(self.input_bf16_bits, &mut executor.input)
                .map_err(cuda_error("selected expert input H2D"))?;
            if let Some((timeline, actor)) = timeline {
                timeline.enqueue_cuda_marker(&executor.stream, actor, "input_h2d_end")?;
            }
            #[cfg(feature = "heterogeneous-test-faults")]
            if injected_fault == Some(SelectedExpertInjectedFault::SubmitAfterInputEnqueue) {
                return Err(LLMError::GpuError(
                    "injected selected-expert post-enqueue submit failure".into(),
                ));
            }
            let start = executor
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("selected expert start event"))?;
            if let Some((timeline, actor)) = timeline {
                timeline.enqueue_cuda_marker(&executor.stream, actor, "compute_begin")?;
            }
            launch_gemv(
                &executor.stream,
                &executor.loader,
                &executor.input,
                &self.weights.gate_up_blocks,
                &self.weights.gate_up_scales,
                &self.weights.gate_up_bias,
                &mut executor.gate_up,
                GATE_UP_ROWS,
            )?;
            launch_swiglu(
                &executor.stream,
                &executor.loader,
                &executor.gate_up,
                &mut executor.swiglu,
                SwigluTraceDevice {
                    scaled_gate: &mut executor.scaled_gate_trace,
                    sigmoid: &mut executor.sigmoid_trace,
                    glu: &mut executor.glu_trace,
                    linear: &mut executor.linear_trace,
                },
            )?;
            launch_gemv(
                &executor.stream,
                &executor.loader,
                &executor.swiglu,
                &self.weights.down_blocks,
                &self.weights.down_scales,
                &self.weights.down_bias,
                &mut self.result_slot.buffer,
                HIDDEN_SIZE,
            )?;
            if let Some((timeline, actor)) = timeline {
                timeline.enqueue_cuda_marker(&executor.stream, actor, "compute_end")?;
            }
            let terminal = executor
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("selected expert terminal event"))?;
            Ok((start, terminal))
        })();
        let (start, terminal) = match submitted {
            Ok(events) => events,
            Err(primary) => {
                let drained = executor.stream.synchronize();
                #[cfg(feature = "heterogeneous-test-faults")]
                if injected_fault == Some(SelectedExpertInjectedFault::SubmitAfterInputEnqueue)
                    && drained.is_ok()
                {
                    executor.last_post_enqueue_fault_drained = true;
                }
                if let Err(drain_error) = drained {
                    return Err(LLMError::GpuError(format!(
                        "selected expert submit failed ({primary}); mandatory stream drain also failed ({drain_error})"
                    )));
                }
                return Err(primary);
            }
        };
        Ok(PendingSelectedExpert {
            executor,
            route: self.route,
            _weights: self.weights,
            _input_bf16_bits: self.input_bf16_bits,
            result_slot: self.result_slot,
            start,
            terminal: Some(terminal),
            drained: false,
            #[cfg(feature = "heterogeneous-test-faults")]
            inject_drain_failure: injected_fault == Some(SelectedExpertInjectedFault::Drain),
        })
    }
}

/// Submitted work. Its exclusive executor borrow and drop-time drain prevent
/// scratch, modules, streams, and weight borrows from outliving CUDA work.
pub struct PendingSelectedExpert<'a> {
    executor: &'a mut CudaSelectedExpertExecutor,
    route: &'a PackedRouteDescriptor,
    _weights: &'a CudaSelectedExpertWeights,
    _input_bf16_bits: &'a [u16],
    result_slot: &'a mut CudaSelectedExpertResultSlot,
    start: CudaEvent,
    terminal: Option<CudaEvent>,
    drained: bool,
    #[cfg(feature = "heterogeneous-test-faults")]
    inject_drain_failure: bool,
}

impl PendingSelectedExpert<'_> {
    pub fn is_complete(&self) -> bool {
        self.terminal.as_ref().is_some_and(CudaEvent::is_complete)
    }

    pub fn drain(mut self, capture: SelectedExpertCapture) -> Result<SelectedExpertExecution> {
        let terminal = self.terminal.as_ref().ok_or_else(|| {
            LLMError::GpuError("selected expert terminal event was already consumed".into())
        })?;
        terminal
            .synchronize()
            .map_err(cuda_error("selected expert terminal drain"))?;
        #[cfg(feature = "heterogeneous-test-faults")]
        if self.inject_drain_failure {
            self.drained = true;
            return Err(LLMError::GpuError(
                "injected selected-expert asynchronous drain failure".into(),
            ));
        }
        let kernel_elapsed_ms = self
            .start
            .elapsed_ms(terminal)
            .map_err(cuda_error("selected expert event timing"))?;
        let output_bf16_bits = self
            .executor
            .stream
            .clone_dtoh(&self.result_slot.buffer)
            .map_err(cuda_error("selected expert output D2H"))?;
        let trace = if capture == SelectedExpertCapture::FirstDivergence {
            let gate_up_bf16_bits = self
                .executor
                .stream
                .clone_dtoh(&self.executor.gate_up)
                .map_err(cuda_error("selected expert gate/up trace D2H"))?;
            let swiglu_bf16_bits = self
                .executor
                .stream
                .clone_dtoh(&self.executor.swiglu)
                .map_err(cuda_error("selected expert SwiGLU trace D2H"))?;
            let scaled_gate_bf16_bits = self
                .executor
                .stream
                .clone_dtoh(&self.executor.scaled_gate_trace)
                .map_err(cuda_error("selected expert scaled-gate trace D2H"))?;
            let sigmoid_bf16_bits = self
                .executor
                .stream
                .clone_dtoh(&self.executor.sigmoid_trace)
                .map_err(cuda_error("selected expert sigmoid trace D2H"))?;
            let glu_bf16_bits = self
                .executor
                .stream
                .clone_dtoh(&self.executor.glu_trace)
                .map_err(cuda_error("selected expert GLU trace D2H"))?;
            let linear_bf16_bits = self
                .executor
                .stream
                .clone_dtoh(&self.executor.linear_trace)
                .map_err(cuda_error("selected expert linear trace D2H"))?;
            Some(SelectedExpertFirstDivergenceTrace {
                gate_up_bf16_bits,
                scaled_gate_bf16_bits,
                sigmoid_bf16_bits,
                glu_bf16_bits,
                linear_bf16_bits,
                swiglu_bf16_bits,
                down_bf16_bits: output_bf16_bits.clone(),
            })
        } else {
            None
        };
        self.drained = true;
        Ok(SelectedExpertExecution {
            result: ExpertResultDescriptor::from_packed_route(self.route),
            output_bf16_bits,
            trace,
            kernel_elapsed_ms,
        })
    }

    /// Drain one GPU0-local result while leaving it resident in the caller's
    /// generation-bound result slot. This path allocates no host output or
    /// result descriptor after dispatch.
    pub fn drain_device_only(mut self) -> Result<SelectedExpertDeviceExecution> {
        let terminal = self.terminal.as_ref().ok_or_else(|| {
            LLMError::GpuError("selected expert terminal event was already consumed".into())
        })?;
        terminal
            .synchronize()
            .map_err(cuda_error("selected expert terminal drain"))?;
        #[cfg(feature = "heterogeneous-test-faults")]
        if self.inject_drain_failure {
            self.drained = true;
            return Err(LLMError::GpuError(
                "injected selected-expert asynchronous drain failure".into(),
            ));
        }
        let kernel_elapsed_ms = self
            .start
            .elapsed_ms(terminal)
            .map_err(cuda_error("selected expert event timing"))?;
        self.drained = true;
        Ok(SelectedExpertDeviceExecution {
            route_contract: CanonicalRouteContract::from_packed_route(self.route),
            kernel_elapsed_ms,
            output_bytes: GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES,
        })
    }

    /// Download the selected result directly into one already-reserved pinned
    /// slot. All CUDA work, the D2H, and timeline callbacks are terminal before
    /// this function returns, so the caller may then return the lease.
    pub fn drain_into_pinned(
        self,
        output: &mut BoundedPinnedLease<u16>,
        timeline: Option<(&CorrelatedTimeline, &str)>,
    ) -> Result<SelectedExpertPinnedExecution> {
        let result = ExpertResultDescriptor::from_packed_route(self.route);
        let execution = self.drain_into_pinned_device_only(output, timeline)?;
        Ok(SelectedExpertPinnedExecution {
            result,
            kernel_elapsed_ms: execution.kernel_elapsed_ms,
            output_bytes: execution.output_bytes,
        })
    }

    /// Drain into a pre-reserved pinned destination without allocating or
    /// cloning route identity after dispatch.
    pub fn drain_into_pinned_device_only(
        mut self,
        output: &mut BoundedPinnedLease<u16>,
        timeline: Option<(&CorrelatedTimeline, &str)>,
    ) -> Result<SelectedExpertDeviceExecution> {
        if output.as_slice().len() < HIDDEN_SIZE {
            return Err(LLMError::GpuError(format!(
                "selected expert pinned output length {} < {HIDDEN_SIZE}",
                output.as_slice().len()
            )));
        }
        let terminal = self.terminal.as_ref().ok_or_else(|| {
            LLMError::GpuError("selected expert terminal event was already consumed".into())
        })?;
        #[cfg(feature = "heterogeneous-test-faults")]
        if self.inject_drain_failure {
            terminal
                .synchronize()
                .map_err(cuda_error("selected expert injected terminal drain"))?;
            self.drained = true;
            return Err(LLMError::GpuError(
                "injected selected-expert asynchronous drain failure".into(),
            ));
        }
        let submitted = (|| -> Result<_> {
            if let Some((timeline, actor)) = timeline {
                timeline.enqueue_cuda_marker(&self.executor.stream, actor, "result_d2h_begin")?;
            }
            self.executor
                .stream
                .memcpy_dtoh(
                    &self.result_slot.buffer,
                    &mut output.as_mut_slice()[..HIDDEN_SIZE],
                )
                .map_err(cuda_error("selected expert pinned output D2H"))?;
            if let Some((timeline, actor)) = timeline {
                timeline.enqueue_cuda_marker(&self.executor.stream, actor, "result_d2h_end")?;
            }
            self.executor
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("selected expert pinned-output terminal event"))
        })();
        let output_terminal = match submitted {
            Ok(event) => event,
            Err(primary) => {
                let drained = self.executor.stream.synchronize();
                if let Err(drain) = drained {
                    return Err(LLMError::GpuError(format!(
                        "selected expert pinned D2H submit failed ({primary}); mandatory drain failed ({drain})"
                    )));
                }
                self.drained = true;
                return Err(primary);
            }
        };
        if let Err(error) = output_terminal.synchronize() {
            let primary = cuda_error("selected expert pinned-output drain")(error);
            let drained = self.executor.stream.synchronize();
            if let Err(drain) = drained {
                return Err(LLMError::GpuError(format!(
                    "selected expert pinned-output drain failed ({primary}); mandatory stream drain failed ({drain})"
                )));
            }
            self.drained = true;
            return Err(primary);
        }
        let kernel_elapsed_ms = self
            .start
            .elapsed_ms(terminal)
            .map_err(cuda_error("selected expert event timing"))?;
        self.drained = true;
        Ok(SelectedExpertDeviceExecution {
            route_contract: CanonicalRouteContract::from_packed_route(self.route),
            kernel_elapsed_ms,
            output_bytes: GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES,
        })
    }

    /// Suppress output publication while still draining non-cancelable CUDA
    /// work. H5 attaches this acknowledgement to the prepared transaction.
    pub fn cancel(mut self) -> Result<ExpertResultDescriptor> {
        let terminal = self.terminal.as_ref().ok_or_else(|| {
            LLMError::GpuError("selected expert terminal event was already consumed".into())
        })?;
        terminal
            .synchronize()
            .map_err(cuda_error("selected expert cancellation drain"))?;
        self.drained = true;
        Ok(ExpertResultDescriptor::from_packed_route(self.route))
    }
}

impl Drop for PendingSelectedExpert<'_> {
    fn drop(&mut self) {
        if !self.drained {
            if let Some(terminal) = &self.terminal {
                let _ = terminal.synchronize();
            }
        }
    }
}

/// Driver-visible memory state before constructing a selected-expert executor.
/// The stable identity is resolved for this process; callers must compare
/// staged samples on an otherwise idle device rather than treating free bytes
/// as a persistent capacity promise.
pub fn selected_expert_device_memory_info(
    stable_device: &StableCudaDeviceId,
) -> Result<(usize, usize)> {
    let resolved = resolve_stable_device(stable_device, &list_devices())
        .map_err(|error| LLMError::GpuError(format!("stable CUDA device: {error}")))?;
    let context = CudaContext::new(resolved.transient_ordinal)
        .map_err(|error| LLMError::GpuError(format!("CUDA context: {error}")))?;
    context
        .bind_to_thread()
        .map_err(cuda_error("selected expert memory-info bind"))?;
    cudarc::driver::result::mem_get_info().map_err(cuda_error("selected expert memory-info query"))
}

#[allow(clippy::too_many_arguments)]
fn launch_gemv(
    stream: &Arc<CudaStream>,
    loader: &KernelLoader,
    input: &CudaSlice<u16>,
    blocks: &CudaSlice<u8>,
    scales: &CudaSlice<u8>,
    bias: &CudaSlice<u16>,
    output: &mut CudaSlice<u16>,
    rows: usize,
) -> Result<()> {
    let kernel = loader.get_func(MODULE, GEMV)?;
    let rows_i32 = rows as i32;
    let blocks_i32 = INPUT_BLOCKS as i32;
    let config = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(THREADS), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: every argument matches the CUDA signature; all slices belong to
    // this context, remain alive through the terminal event, and have the exact
    // fixed lengths validated during construction.
    unsafe {
        stream
            .launch_builder(&kernel)
            .arg(input)
            .arg(blocks)
            .arg(scales)
            .arg(bias)
            .arg(output)
            .arg(&rows_i32)
            .arg(&blocks_i32)
            .launch(config)
            .map_err(cuda_error("selected expert MXFP4 GEMV launch"))?;
    }
    Ok(())
}

fn launch_swiglu(
    stream: &Arc<CudaStream>,
    loader: &KernelLoader,
    gate_up: &CudaSlice<u16>,
    output: &mut CudaSlice<u16>,
    trace: SwigluTraceDevice<'_>,
) -> Result<()> {
    let kernel = loader.get_func(MODULE, SWIGLU)?;
    let intermediate = INTERMEDIATE_SIZE as i32;
    let config = LaunchConfig {
        grid_dim: ((INTERMEDIATE_SIZE as u32).div_ceil(THREADS), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: arguments match the CUDA signature and the fixed slices remain
    // owned until the recorded terminal event has been drained.
    unsafe {
        stream
            .launch_builder(&kernel)
            .arg(gate_up)
            .arg(output)
            .arg(trace.scaled_gate)
            .arg(trace.sigmoid)
            .arg(trace.glu)
            .arg(trace.linear)
            .arg(&intermediate)
            .arg(&SWIGLU_ALPHA)
            .arg(&SWIGLU_LIMIT)
            .launch(config)
            .map_err(cuda_error("selected expert SwiGLU launch"))?;
    }
    Ok(())
}

struct SwigluTraceDevice<'a> {
    scaled_gate: &'a mut CudaSlice<u16>,
    sigmoid: &'a mut CudaSlice<u16>,
    glu: &'a mut CudaSlice<u16>,
    linear: &'a mut CudaSlice<u16>,
}

fn cuda_error(stage: &'static str) -> impl FnOnce(cudarc::driver::DriverError) -> LLMError {
    move |error| LLMError::GpuError(format!("{stage}: {error}"))
}

/// CPU semantic authority for one exact native-packed selected expert.
pub fn exact_selected_expert_reference(
    source: NativeMxfp4ExpertView<'_>,
    input_bf16_bits: &[u16],
) -> Result<SelectedExpertFirstDivergenceTrace> {
    source.validate()?;
    if input_bf16_bits.len() != HIDDEN_SIZE {
        return Err(LLMError::ModelError(format!(
            "selected expert reference input length {} != {HIDDEN_SIZE}",
            input_bf16_bits.len()
        )));
    }
    let input = input_bf16_bits
        .iter()
        .copied()
        .map(bf16::from_bits)
        .collect::<Vec<_>>();
    let gate_up_bf16_bits = exact_gemv(
        &input,
        source.gate_up_blocks,
        source.gate_up_scales,
        source.gate_up_bias_bf16_bits,
        GATE_UP_ROWS,
    )?;
    let swiglu = exact_swiglu(&gate_up_bf16_bits);
    let activated = swiglu
        .output_bf16_bits
        .iter()
        .copied()
        .map(bf16::from_bits)
        .collect::<Vec<_>>();
    let down_bf16_bits = exact_gemv(
        &activated,
        source.down_blocks,
        source.down_scales,
        source.down_bias_bf16_bits,
        HIDDEN_SIZE,
    )?;
    Ok(SelectedExpertFirstDivergenceTrace {
        gate_up_bf16_bits,
        scaled_gate_bf16_bits: swiglu.scaled_gate_bf16_bits,
        sigmoid_bf16_bits: swiglu.sigmoid_bf16_bits,
        glu_bf16_bits: swiglu.glu_bf16_bits,
        linear_bf16_bits: swiglu.linear_bf16_bits,
        swiglu_bf16_bits: swiglu.output_bf16_bits,
        down_bf16_bits,
    })
}

fn exact_gemv(
    input: &[bf16],
    blocks: &[u8],
    scales: &[u8],
    bias: &[u16],
    rows: usize,
) -> Result<Vec<u16>> {
    if input.len() != INPUT_BLOCKS * QUANT_BLOCK_SIZE
        || blocks.len() != rows * INPUT_BLOCKS * MXFP4_PACKED_BYTES
        || scales.len() != rows * INPUT_BLOCKS
        || bias.len() != rows
    {
        return Err(LLMError::ModelError(
            "invalid selected expert exact GEMV dimensions".into(),
        ));
    }
    let activation = input
        .chunks_exact(QUANT_BLOCK_SIZE)
        .map(|block| <&[bf16; QUANT_BLOCK_SIZE]>::try_from(block).expect("exact chunk"))
        .collect::<Vec<_>>();
    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut lanes = [0.0_f32; 16];
        for (block_index, activation) in activation.iter().enumerate() {
            let packed_start = (row * INPUT_BLOCKS + block_index) * MXFP4_PACKED_BYTES;
            let packed = blocks[packed_start..packed_start + MXFP4_PACKED_BYTES]
                .try_into()
                .expect("validated MXFP4 block length");
            let weight = Mxfp4Block {
                scale: scales[row * INPUT_BLOCKS + block_index],
                packed,
            };
            accumulate_mxfp4_bf16_block(&weight, activation, &mut lanes);
        }
        let total = bf16::from_bits(bias[row]).to_f32() + lanes.into_iter().sum::<f32>();
        output.push(bf16::from_f32(total).to_bits());
    }
    Ok(output)
}

struct ExactSwiGluTrace {
    scaled_gate_bf16_bits: Vec<u16>,
    sigmoid_bf16_bits: Vec<u16>,
    glu_bf16_bits: Vec<u16>,
    linear_bf16_bits: Vec<u16>,
    output_bf16_bits: Vec<u16>,
}

fn exact_swiglu(gate_up: &[u16]) -> ExactSwiGluTrace {
    let mut scaled_gate_bf16_bits = Vec::with_capacity(INTERMEDIATE_SIZE);
    let mut sigmoid_bf16_bits = Vec::with_capacity(INTERMEDIATE_SIZE);
    let mut glu_bf16_bits = Vec::with_capacity(INTERMEDIATE_SIZE);
    let mut linear_bf16_bits = Vec::with_capacity(INTERMEDIATE_SIZE);
    let mut output_bf16_bits = Vec::with_capacity(INTERMEDIATE_SIZE);
    for index in 0..INTERMEDIATE_SIZE {
        let gate = bf16::from_bits(gate_up[index * 2])
            .to_f32()
            .min(SWIGLU_LIMIT);
        let up = bf16::from_bits(gate_up[index * 2 + 1])
            .to_f32()
            .clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
        let scaled_gate = bf16::from_f32(gate * SWIGLU_ALPHA).to_f32();
        let sigmoid = bf16::from_f32(1.0 / (1.0 + (-scaled_gate).exp())).to_f32();
        let glu = bf16::from_f32(gate * sigmoid).to_f32();
        let linear = bf16::from_f32(up + 1.0).to_f32();
        scaled_gate_bf16_bits.push(bf16::from_f32(scaled_gate).to_bits());
        sigmoid_bf16_bits.push(bf16::from_f32(sigmoid).to_bits());
        glu_bf16_bits.push(bf16::from_f32(glu).to_bits());
        linear_bf16_bits.push(bf16::from_f32(linear).to_bits());
        output_bf16_bits.push(bf16::from_f32(glu * linear).to_bits());
    }
    ExactSwiGluTrace {
        scaled_gate_bf16_bits,
        sigmoid_bf16_bits,
        glu_bf16_bits,
        linear_bf16_bits,
        output_bf16_bits,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_memory_contract_matches_phase_one_bytes() {
        assert_eq!(GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES, 13_236_480);
        assert_eq!(GPT_OSS_SELECTED_EXPERT_INPUT_BYTES, 5_760);
        assert_eq!(GPT_OSS_SELECTED_EXPERT_SCRATCH_BYTES, 17_280);
        assert_eq!(GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES, 5_760);
        assert_eq!(GPT_OSS_SELECTED_EXPERT_TRACE_BYTES, 23_040);
        assert_eq!(GPT_OSS_SELECTED_EXPERT_DEVICE_WORK_BYTES, 51_840);
    }

    #[test]
    fn unsupported_view_is_rejected_before_upload() {
        let source = NativeMxfp4ExpertView {
            key: GptOssExpertKey {
                layer: 0,
                expert: 0,
            },
            gate_up_blocks: &[],
            gate_up_scales: &[],
            gate_up_bias_bf16_bits: &[],
            down_blocks: &[],
            down_scales: &[],
            down_bias_bf16_bits: &[],
            identity_sha256: &"0".repeat(64),
        };
        assert!(source.validate().is_err());
    }
}
