//! Exact GPU-authored GPT-OSS router projection and stable top-4 records.

use std::{mem::ManuallyDrop, sync::Arc};

use cudarc::driver::{
    sys::CUevent_flags, CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use gpt_oss_core::error::{LLMError, Result};
use gpt_oss_cpu_kernels::{KernelPath, Kernels};
use gpt_oss_gpu::device::{list_devices, resolve_stable_device, StableCudaDeviceId};
use gpt_oss_gpu::event::CorrelatedTimeline;
use gpt_oss_gpu::kernel_loader::{compiled_ptx_dir, KernelLoader};
use gpt_oss_gpu::pinned_memory::BoundedPinnedLease;
use gpt_oss_moe_semantics::{softmax_weights, stable_top_k_indices};
use half::bf16;

use super::contract::{
    GptOssPhase, GptOssRouteDescriptor, GptOssRouteWireV1, GptOssRoutedBatchDescriptor,
    GPT_OSS_HIDDEN_SIZE, GPT_OSS_ROUTE_WIRE_V1_BYTES, GPT_OSS_TOP_K,
};

pub const GPT_OSS_ROUTER_MAX_ROWS: usize = 64;
pub const GPT_OSS_ROUTER_DESCRIPTOR_BYTES_PER_ROW: usize =
    GPT_OSS_TOP_K * GPT_OSS_ROUTE_WIRE_V1_BYTES;

pub fn exact_router_owned_device_bytes(experts: usize, max_rows: usize) -> Result<usize> {
    if !matches!(experts, 32 | 128) || max_rows == 0 || max_rows > GPT_OSS_ROUTER_MAX_ROWS {
        return Err(LLMError::ModelError(
            "router byte reporter received unsupported dimensions".into(),
        ));
    }
    let bf16 = size_of::<u16>();
    let activation = max_rows
        .checked_mul(GPT_OSS_HIDDEN_SIZE)
        .and_then(|values| values.checked_mul(bf16))
        .ok_or_else(|| LLMError::ModelError("router activation bytes overflow".into()))?;
    let weights = experts
        .checked_mul(GPT_OSS_HIDDEN_SIZE)
        .and_then(|values| values.checked_mul(bf16))
        .ok_or_else(|| LLMError::ModelError("router weight bytes overflow".into()))?;
    let bias = experts
        .checked_mul(bf16)
        .ok_or_else(|| LLMError::ModelError("router bias bytes overflow".into()))?;
    let logits = max_rows
        .checked_mul(experts)
        .and_then(|values| values.checked_mul(bf16))
        .ok_or_else(|| LLMError::ModelError("router logit bytes overflow".into()))?;
    let descriptors = max_rows
        .checked_mul(GPT_OSS_ROUTER_DESCRIPTOR_BYTES_PER_ROW)
        .ok_or_else(|| LLMError::ModelError("router descriptor bytes overflow".into()))?;
    let invalid_flags = max_rows
        .checked_mul(size_of::<u32>())
        .ok_or_else(|| LLMError::ModelError("router invalid-flag bytes overflow".into()))?;
    [
        activation,
        weights,
        bias,
        logits,
        descriptors,
        invalid_flags,
    ]
    .into_iter()
    .try_fold(0_usize, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or_else(|| LLMError::ModelError("router device bytes overflow".into()))
    })
}

const MODULE: &str = "gpt_oss_router";
const PROJECTION: &str = "gpt_oss_router_bf16_projection_kernel";
const TOP4: &str = "gpt_oss_router_stable_top4_kernel";
const PROJECTION_THREADS: u32 = 256;
const ROUTER_THREADS: u32 = 64;

#[derive(Debug, Clone, Copy)]
pub struct ExactRouterWeightsView<'a> {
    pub experts: usize,
    pub weight_bf16_bits: &'a [u16],
    pub bias_bf16_bits: &'a [u16],
}

/// Owned, already-resident BF16 router surfaces for the narrow H3 handoff.
///
/// The byte allocations deliberately match `LayerOwnerDenseTensor`'s
/// storage representation. Owner-selective construction transfers the two
/// router allocations into this value instead of retaining them as generic
/// dense tensors. Construction validates the durable device identity, exact
/// CUDA context, expert count, and byte lengths before any handoff work is
/// enqueued. Borrowing these allocations would make an unproven CUDA drain
/// impossible to quarantine safely.
pub struct ResidentExactRouterWeights {
    stable_device: StableCudaDeviceId,
    experts: usize,
    weight_bf16_bytes: CudaSlice<u8>,
    bias_bf16_bytes: CudaSlice<u8>,
    #[cfg(feature = "heterogeneous-test-faults")]
    injected_fault: Option<ResidentRouterHandoffInjectedFault>,
    #[cfg(feature = "heterogeneous-test-faults")]
    drop_probe: Option<Arc<std::sync::atomic::AtomicBool>>,
}

impl ResidentExactRouterWeights {
    pub fn new(
        stable_device: StableCudaDeviceId,
        experts: usize,
        weight_bf16_bytes: CudaSlice<u8>,
        bias_bf16_bytes: CudaSlice<u8>,
    ) -> Result<Self> {
        let source = Self {
            stable_device,
            experts,
            weight_bf16_bytes,
            bias_bf16_bytes,
            #[cfg(feature = "heterogeneous-test-faults")]
            injected_fault: None,
            #[cfg(feature = "heterogeneous-test-faults")]
            drop_probe: None,
        };
        source.validate()?;
        Ok(source)
    }

    fn validate(&self) -> Result<()> {
        let (weight_bytes, bias_bytes) = exact_router_weight_surface_bytes(self.experts)?;
        if self.weight_bf16_bytes.len() != weight_bytes || self.bias_bf16_bytes.len() != bias_bytes
        {
            return Err(LLMError::GpuError(format!(
                "resident router BF16 byte shape mismatch: weights={} expected={} bias={} expected={}",
                self.weight_bf16_bytes.len(),
                weight_bytes,
                self.bias_bf16_bytes.len(),
                bias_bytes
            )));
        }
        if self.weight_bf16_bytes.context().cu_ctx() != self.bias_bf16_bytes.context().cu_ctx() {
            return Err(LLMError::GpuError(
                "resident router weight and bias allocations do not share one CUDA context".into(),
            ));
        }
        let resolved =
            resolve_stable_device(&self.stable_device, &list_devices()).map_err(|error| {
                LLMError::GpuError(format!("resident router stable device: {error}"))
            })?;
        if self.weight_bf16_bytes.context().ordinal() != resolved.transient_ordinal
            || self.bias_bf16_bytes.context().ordinal() != resolved.transient_ordinal
        {
            return Err(LLMError::GpuError(
                "resident router allocation ordinal does not match stable device identity".into(),
            ));
        }
        Ok(())
    }

    pub const fn stable_device(&self) -> &StableCudaDeviceId {
        &self.stable_device
    }

    pub const fn experts(&self) -> usize {
        self.experts
    }

    pub fn device_bytes(&self) -> Result<usize> {
        self.weight_bf16_bytes
            .len()
            .checked_add(self.bias_bf16_bytes.len())
            .ok_or_else(|| LLMError::GpuError("resident router source bytes overflow".into()))
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub fn inject_handoff_failure(
        &mut self,
        fault: ResidentRouterHandoffInjectedFault,
    ) -> Result<()> {
        if self.injected_fault.replace(fault).is_some() {
            return Err(LLMError::GpuError(
                "resident router handoff already has an armed test fault".into(),
            ));
        }
        Ok(())
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub fn set_drop_probe_for_test(&mut self, probe: Arc<std::sync::atomic::AtomicBool>) {
        self.drop_probe = Some(probe);
    }
}

#[cfg(feature = "heterogeneous-test-faults")]
impl Drop for ResidentExactRouterWeights {
    fn drop(&mut self) {
        if let Some(probe) = self.drop_probe.as_ref() {
            probe.store(true, std::sync::atomic::Ordering::Release);
        }
    }
}

#[cfg(feature = "heterogeneous-test-faults")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentRouterHandoffInjectedFault {
    AfterWeightCopyEnqueue,
    AfterWeightCopyEnqueueAndFallbackDrainFailure,
}

impl ExactRouterWeightsView<'_> {
    fn validate(&self) -> Result<()> {
        if !matches!(self.experts, 32 | 128) {
            return Err(LLMError::ModelError(format!(
                "exact router supports E=32 or E=128, observed {}",
                self.experts
            )));
        }
        let expected_weights = self
            .experts
            .checked_mul(GPT_OSS_HIDDEN_SIZE)
            .ok_or_else(|| LLMError::ModelError("router weight shape overflows".into()))?;
        if self.weight_bf16_bits.len() != expected_weights
            || self.bias_bf16_bits.len() != self.experts
        {
            return Err(LLMError::ModelError(format!(
                "exact router weight/bias shape mismatch: weights={} expected={} bias={} expected={}",
                self.weight_bf16_bits.len(),
                expected_weights,
                self.bias_bf16_bits.len(),
                self.experts
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactRouterReference {
    pub batch: GptOssRoutedBatchDescriptor,
    pub router_logits_bf16_bits: Vec<u16>,
}

/// CPU semantic authority for the H4 router oracle.
pub fn exact_router_reference(
    layer: u16,
    phase: GptOssPhase,
    placement_epoch: u64,
    rows: usize,
    activation_bf16_bits: &[u16],
    weights: ExactRouterWeightsView<'_>,
) -> Result<ExactRouterReference> {
    weights.validate()?;
    validate_router_input(rows, activation_bf16_bits)?;
    let kernels = Kernels::new(KernelPath::Scalar)
        .map_err(|error| LLMError::ModelError(format!("router scalar kernels: {error}")))?;
    let activation = activation_bf16_bits
        .iter()
        .copied()
        .map(bf16::from_bits)
        .collect::<Vec<_>>();
    let router_weights = weights
        .weight_bf16_bits
        .iter()
        .copied()
        .map(bf16::from_bits)
        .collect::<Vec<_>>();
    let mut logits_bits = Vec::with_capacity(rows * weights.experts);
    let mut routes = Vec::with_capacity(rows * GPT_OSS_TOP_K);
    for source_row in 0..rows {
        let input =
            &activation[source_row * GPT_OSS_HIDDEN_SIZE..(source_row + 1) * GPT_OSS_HIDDEN_SIZE];
        let mut logits = vec![0.0_f32; weights.experts];
        kernels
            .bf16_matvec(
                &router_weights,
                weights.experts,
                GPT_OSS_HIDDEN_SIZE,
                input,
                &mut logits,
            )
            .map_err(|error| LLMError::ModelError(format!("router scalar projection: {error}")))?;
        for (logit, bias) in logits.iter_mut().zip(weights.bias_bf16_bits) {
            *logit += bf16::from_bits(*bias).to_f32();
            *logit = bf16::from_f32(*logit).to_f32();
        }
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(LLMError::ModelError(
                "exact router produced non-finite BF16 logits".into(),
            ));
        }
        logits_bits.extend(
            logits
                .iter()
                .copied()
                .map(bf16::from_f32)
                .map(bf16::to_bits),
        );
        let selected = stable_top_k_indices(&logits, GPT_OSS_TOP_K);
        let selected_logits = selected
            .iter()
            .map(|expert| logits[*expert])
            .collect::<Vec<_>>();
        let selected_weights = softmax_weights(&selected_logits);
        for (rank, (expert, weight)) in selected.into_iter().zip(selected_weights).enumerate() {
            routes.push(GptOssRouteDescriptor {
                source_row: source_row as u32,
                route_rank: rank as u8,
                expert_id: expert as u16,
                weight_bf16_bits: bf16::from_f32(weight).to_bits(),
                activation_slot: source_row as u32,
            });
        }
    }
    let batch = GptOssRoutedBatchDescriptor {
        layer,
        phase,
        rows: rows as u32,
        hidden_size: GPT_OSS_HIDDEN_SIZE as u16,
        experts_per_layer: weights.experts as u16,
        placement_epoch,
        activation_bf16_bits: activation_bf16_bits.to_vec(),
        routes,
    };
    batch
        .validate()
        .map_err(|error| LLMError::ModelError(format!("router reference batch: {error}")))?;
    Ok(ExactRouterReference {
        batch,
        router_logits_bf16_bits: logits_bits,
    })
}

#[cfg(feature = "heterogeneous-test-faults")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactRouterInjectedFault {
    SubmitAfterInputEnqueue,
    SubmitAfterInputEnqueueAndFallbackDrainFailure,
    RelayAfterSourceEnqueue,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExactRouterExecution {
    pub batch: GptOssRoutedBatchDescriptor,
    pub router_logits_bf16_bits: Vec<u16>,
    pub router_elapsed_ms: f32,
    pub source_d2h_bytes: usize,
    pub descriptor_d2h_bytes: usize,
}

/// Fixed-shape GPU0 router executor. Selection and BF16 weights remain GPU
/// authored; host output is a checked dispatch/evidence view.
pub struct CudaExactRouter {
    stable_device: StableCudaDeviceId,
    compute_stream: ManuallyDrop<Arc<CudaStream>>,
    relay_stream: ManuallyDrop<Arc<CudaStream>>,
    loader: ManuallyDrop<KernelLoader>,
    experts: usize,
    max_rows: usize,
    input: ManuallyDrop<CudaSlice<u16>>,
    weights: ManuallyDrop<CudaSlice<u16>>,
    bias: ManuallyDrop<CudaSlice<u16>>,
    logits: ManuallyDrop<CudaSlice<u16>>,
    route_records: ManuallyDrop<CudaSlice<u8>>,
    status: ManuallyDrop<CudaSlice<u32>>,
    poisoned: bool,
    #[cfg(feature = "heterogeneous-test-faults")]
    injected_fault: Option<ExactRouterInjectedFault>,
    #[cfg(feature = "heterogeneous-test-faults")]
    last_fault_drained: bool,
}

struct ResidentRouterHandoffState {
    source: ManuallyDrop<ResidentExactRouterWeights>,
    compute_stream: ManuallyDrop<Arc<CudaStream>>,
    relay_stream: ManuallyDrop<Arc<CudaStream>>,
    loader: ManuallyDrop<KernelLoader>,
    input: ManuallyDrop<CudaSlice<u16>>,
    weights: ManuallyDrop<CudaSlice<u16>>,
    bias: ManuallyDrop<CudaSlice<u16>>,
    logits: ManuallyDrop<CudaSlice<u16>>,
    route_records: ManuallyDrop<CudaSlice<u8>>,
    status: ManuallyDrop<CudaSlice<u32>>,
    quarantined: bool,
    disarmed: bool,
}

impl ResidentRouterHandoffState {
    fn quarantine(&mut self) {
        self.quarantined = true;
        #[cfg(feature = "heterogeneous-test-faults")]
        RESIDENT_ROUTER_HANDOFF_QUARANTINES.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }
}

impl Drop for ResidentRouterHandoffState {
    fn drop(&mut self) {
        if self.quarantined || self.disarmed {
            return;
        }
        // SAFETY: either no source-referencing copy was submitted, or the
        // handoff stream reached a proven terminal point before this state is
        // released. The quarantined path intentionally retains every field.
        unsafe {
            ManuallyDrop::drop(&mut self.status);
            ManuallyDrop::drop(&mut self.route_records);
            ManuallyDrop::drop(&mut self.logits);
            ManuallyDrop::drop(&mut self.bias);
            ManuallyDrop::drop(&mut self.weights);
            ManuallyDrop::drop(&mut self.input);
            ManuallyDrop::drop(&mut self.loader);
            ManuallyDrop::drop(&mut self.relay_stream);
            ManuallyDrop::drop(&mut self.compute_stream);
            ManuallyDrop::drop(&mut self.source);
        }
    }
}

#[cfg(feature = "heterogeneous-test-faults")]
static RESIDENT_ROUTER_HANDOFF_QUARANTINES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "heterogeneous-test-faults")]
pub fn resident_router_handoff_quarantines_for_test() -> usize {
    RESIDENT_ROUTER_HANDOFF_QUARANTINES.load(std::sync::atomic::Ordering::Acquire)
}

#[derive(Clone, Copy)]
enum RouterInput<'a> {
    Host(&'a [u16]),
    Device(&'a CudaSlice<u16>),
}

impl CudaExactRouter {
    pub fn new(
        stable_device: StableCudaDeviceId,
        max_rows: usize,
        weights: ExactRouterWeightsView<'_>,
    ) -> Result<Self> {
        weights.validate()?;
        if max_rows == 0 || max_rows > GPT_OSS_ROUTER_MAX_ROWS {
            return Err(LLMError::GpuError(format!(
                "exact router max rows {max_rows} outside 1..={GPT_OSS_ROUTER_MAX_ROWS}"
            )));
        }
        let resolved = resolve_stable_device(&stable_device, &list_devices())
            .map_err(|error| LLMError::GpuError(format!("stable CUDA router device: {error}")))?;
        let context = CudaContext::new(resolved.transient_ordinal)
            .map_err(cuda_error("exact router context"))?;
        let compute_stream = context
            .new_stream()
            .map_err(cuda_error("exact router compute stream"))?;
        let relay_stream = context
            .new_stream()
            .map_err(cuda_error("exact router relay stream"))?;
        let loader = KernelLoader::new(
            Arc::clone(&context),
            Arc::clone(&compute_stream),
            compiled_ptx_dir(),
        )?;
        if !loader.has_func(MODULE, PROJECTION) || !loader.has_func(MODULE, TOP4) {
            return Err(LLMError::GpuError(
                "exact-router PTX functions are unavailable".into(),
            ));
        }
        let input = compute_stream
            .alloc_zeros::<u16>(max_rows * GPT_OSS_HIDDEN_SIZE)
            .map_err(cuda_error("exact router input allocation"))?;
        let device_weights = compute_stream
            .clone_htod(weights.weight_bf16_bits)
            .map_err(cuda_error("exact router weight upload"))?;
        let bias = compute_stream
            .clone_htod(weights.bias_bf16_bits)
            .map_err(cuda_error("exact router bias upload"))?;
        let logits = compute_stream
            .alloc_zeros::<u16>(max_rows * weights.experts)
            .map_err(cuda_error("exact router logit allocation"))?;
        let route_records = compute_stream
            .alloc_zeros::<u8>(max_rows * GPT_OSS_ROUTER_DESCRIPTOR_BYTES_PER_ROW)
            .map_err(cuda_error("exact router canonical-record allocation"))?;
        let status = compute_stream
            .alloc_zeros::<u32>(max_rows)
            .map_err(cuda_error("exact router status allocation"))?;
        compute_stream
            .synchronize()
            .map_err(cuda_error("exact router construction drain"))?;
        Ok(Self {
            stable_device,
            compute_stream: ManuallyDrop::new(compute_stream),
            relay_stream: ManuallyDrop::new(relay_stream),
            loader: ManuallyDrop::new(loader),
            experts: weights.experts,
            max_rows,
            input: ManuallyDrop::new(input),
            weights: ManuallyDrop::new(device_weights),
            bias: ManuallyDrop::new(bias),
            logits: ManuallyDrop::new(logits),
            route_records: ManuallyDrop::new(route_records),
            status: ManuallyDrop::new(status),
            poisoned: false,
            #[cfg(feature = "heterogeneous-test-faults")]
            injected_fault: None,
            #[cfg(feature = "heterogeneous-test-faults")]
            last_fault_drained: false,
        })
    }

    /// Build an exact router by copying an owned pair of already-resident
    /// layer-owner BF16 byte allocations into router-owned storage.
    ///
    /// This is an isolated H3 handoff primitive. It is intentionally not wired
    /// into `OwnerSelectiveModel` or `HeterogeneousControlRuntime`. The source
    /// is consumed so an unproven post-enqueue drain can quarantine both the
    /// source and destination storage for process life.
    pub fn from_resident_weights(
        max_rows: usize,
        source: ResidentExactRouterWeights,
    ) -> Result<Self> {
        source.validate()?;
        validate_router_max_rows(max_rows)?;
        let stable_device = source.stable_device.clone();
        let experts = source.experts;
        let context = Arc::clone(source.weight_bf16_bytes.context());
        if context.cu_ctx() != source.bias_bf16_bytes.context().cu_ctx() {
            return Err(LLMError::GpuError(
                "resident router source context changed before handoff".into(),
            ));
        }
        let compute_stream = context
            .new_stream()
            .map_err(cuda_error("resident router compute stream"))?;
        let relay_stream = context
            .new_stream()
            .map_err(cuda_error("resident router relay stream"))?;
        let loader = KernelLoader::new(
            Arc::clone(&context),
            Arc::clone(&compute_stream),
            compiled_ptx_dir(),
        )?;
        if !loader.has_func(MODULE, PROJECTION) || !loader.has_func(MODULE, TOP4) {
            return Err(LLMError::GpuError(
                "exact-router PTX functions are unavailable".into(),
            ));
        }

        let input_values = max_rows
            .checked_mul(GPT_OSS_HIDDEN_SIZE)
            .ok_or_else(|| LLMError::GpuError("resident router input shape overflows".into()))?;
        let logit_values = max_rows
            .checked_mul(experts)
            .ok_or_else(|| LLMError::GpuError("resident router logit shape overflows".into()))?;
        let route_bytes = max_rows
            .checked_mul(GPT_OSS_ROUTER_DESCRIPTOR_BYTES_PER_ROW)
            .ok_or_else(|| LLMError::GpuError("resident router route arena overflows".into()))?;
        let (weight_bytes, bias_bytes) = exact_router_weight_surface_bytes(experts)?;
        let weight_values = weight_bytes
            .checked_div(size_of::<u16>())
            .ok_or_else(|| LLMError::GpuError("resident router weight values overflow".into()))?;
        let bias_values = bias_bytes
            .checked_div(size_of::<u16>())
            .ok_or_else(|| LLMError::GpuError("resident router bias values overflow".into()))?;

        let input = allocate_terminal::<u16>(
            &compute_stream,
            input_values,
            "resident router input allocation",
        )?;
        let device_weights = allocate_terminal::<u16>(
            &compute_stream,
            weight_values,
            "resident router weight allocation",
        )?;
        let device_bias = allocate_terminal::<u16>(
            &compute_stream,
            bias_values,
            "resident router bias allocation",
        )?;
        let logits = allocate_terminal::<u16>(
            &compute_stream,
            logit_values,
            "resident router logit allocation",
        )?;
        let route_records = allocate_terminal::<u8>(
            &compute_stream,
            route_bytes,
            "resident router route allocation",
        )?;
        let status = allocate_terminal::<u32>(
            &compute_stream,
            max_rows,
            "resident router status allocation",
        )?;

        let mut state = ResidentRouterHandoffState {
            source: ManuallyDrop::new(source),
            compute_stream: ManuallyDrop::new(compute_stream),
            relay_stream: ManuallyDrop::new(relay_stream),
            loader: ManuallyDrop::new(loader),
            input: ManuallyDrop::new(input),
            weights: ManuallyDrop::new(device_weights),
            bias: ManuallyDrop::new(device_bias),
            logits: ManuallyDrop::new(logits),
            route_records: ManuallyDrop::new(route_records),
            status: ManuallyDrop::new(status),
            quarantined: false,
            disarmed: false,
        };

        #[cfg(feature = "heterogeneous-test-faults")]
        let injected_fault = state.source.injected_fault;
        // SAFETY: both sources are whole CUDA allocations (not offset views),
        // CUDA allocation alignment exceeds u16 alignment, exact even byte
        // lengths were validated above, and every u16 bit pattern is valid.
        // The borrowed typed views cannot outlive the owned source in `state`.
        let source_weights = unsafe {
            state
                .source
                .weight_bf16_bytes
                .transmute::<u16>(weight_values)
        }
        .ok_or_else(|| LLMError::GpuError("resident router weight view cast failed".into()))?;
        // SAFETY: identical argument to the weight view, for the bias surface.
        let source_bias = unsafe { state.source.bias_bf16_bytes.transmute::<u16>(bias_values) }
            .ok_or_else(|| LLMError::GpuError("resident router bias view cast failed".into()))?;
        let result = state
            .compute_stream
            .memcpy_dtod(&source_weights, &mut *state.weights)
            .map_err(cuda_error("resident router weight D2D"));
        if result.is_ok() {
            #[cfg(feature = "heterogeneous-test-faults")]
            if matches!(
                injected_fault,
                Some(
                    ResidentRouterHandoffInjectedFault::AfterWeightCopyEnqueue
                        | ResidentRouterHandoffInjectedFault::AfterWeightCopyEnqueueAndFallbackDrainFailure
                )
            ) {
                return resident_handoff_submit_failure(
                    state,
                    LLMError::GpuError(
                        "injected resident-router post-weight-enqueue failure".into(),
                    ),
                    injected_fault
                        == Some(ResidentRouterHandoffInjectedFault::AfterWeightCopyEnqueueAndFallbackDrainFailure),
                );
            }
        }
        let submitted = result.and_then(|()| {
            state
                .compute_stream
                .memcpy_dtod(&source_bias, &mut *state.bias)
                .map_err(cuda_error("resident router bias D2D"))
        });
        if let Err(primary) = submitted {
            return resident_handoff_submit_failure(state, primary, false);
        }
        if let Err(error) = state.compute_stream.synchronize() {
            return resident_handoff_submit_failure(
                state,
                cuda_error("resident router terminal drain")(error),
                false,
            );
        }

        // SAFETY: the terminal stream drain proves both copies complete. Every field
        // is moved exactly once and `disarmed` suppresses the state destructor.
        let router = unsafe {
            ManuallyDrop::drop(&mut state.source);
            let compute_stream = ManuallyDrop::take(&mut state.compute_stream);
            let relay_stream = ManuallyDrop::take(&mut state.relay_stream);
            let loader = ManuallyDrop::take(&mut state.loader);
            let input = ManuallyDrop::take(&mut state.input);
            let weights = ManuallyDrop::take(&mut state.weights);
            let bias = ManuallyDrop::take(&mut state.bias);
            let logits = ManuallyDrop::take(&mut state.logits);
            let route_records = ManuallyDrop::take(&mut state.route_records);
            let status = ManuallyDrop::take(&mut state.status);
            state.disarmed = true;
            Self {
                stable_device,
                compute_stream: ManuallyDrop::new(compute_stream),
                relay_stream: ManuallyDrop::new(relay_stream),
                loader: ManuallyDrop::new(loader),
                experts,
                max_rows,
                input: ManuallyDrop::new(input),
                weights: ManuallyDrop::new(weights),
                bias: ManuallyDrop::new(bias),
                logits: ManuallyDrop::new(logits),
                route_records: ManuallyDrop::new(route_records),
                status: ManuallyDrop::new(status),
                poisoned: false,
                #[cfg(feature = "heterogeneous-test-faults")]
                injected_fault: None,
                #[cfg(feature = "heterogeneous-test-faults")]
                last_fault_drained: false,
            }
        };
        Ok(router)
    }

    pub fn stable_device(&self) -> &StableCudaDeviceId {
        &self.stable_device
    }

    pub const fn experts(&self) -> usize {
        self.experts
    }

    pub fn owned_device_bytes(&self) -> Result<usize> {
        exact_router_owned_device_bytes(self.experts, self.max_rows)
    }

    pub const fn max_rows(&self) -> usize {
        self.max_rows
    }

    pub(crate) fn relay_stream(&self) -> &Arc<CudaStream> {
        &self.relay_stream
    }

    pub fn drain(&mut self) -> Result<()> {
        if self.poisoned {
            return Err(LLMError::GpuError(
                "poisoned exact router cannot prove a later drain".into(),
            ));
        }
        let compute = self.compute_stream.synchronize();
        let relay = self.relay_stream.synchronize();
        let result = match (compute, relay) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) => Err(cuda_error("exact router compute drain")(error)),
            (Ok(()), Err(error)) => Err(cuda_error("exact router relay drain")(error)),
            (Err(compute), Err(relay)) => Err(LLMError::GpuError(format!(
                "exact router compute drain failed ({compute}); relay drain failed ({relay})"
            ))),
        };
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    /// An outer transaction observed an unproven CUDA lifetime. Make this
    /// router permanently unusable so its `Drop` path retains every stream,
    /// module and allocation that may still be referenced.
    pub(crate) fn quarantine_unproven_device_work(&mut self) {
        self.poisoned = true;
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub fn inject_next_failure(&mut self, fault: ExactRouterInjectedFault) -> Result<()> {
        if self.injected_fault.is_some() {
            return Err(LLMError::GpuError(
                "exact router already has an armed test fault".into(),
            ));
        }
        self.injected_fault = Some(fault);
        self.last_fault_drained = false;
        Ok(())
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub const fn last_fault_drained(&self) -> bool {
        self.last_fault_drained
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub const fn device_state_quarantined_for_test(&self) -> bool {
        self.poisoned
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_and_download(
        &mut self,
        layer: u16,
        phase: GptOssPhase,
        placement_epoch: u64,
        rows: usize,
        activation_bf16_bits: &[u16],
        source_activation: &mut BoundedPinnedLease<u16>,
        route_records: &mut BoundedPinnedLease<u8>,
        timeline: Option<&CorrelatedTimeline>,
    ) -> Result<ExactRouterExecution> {
        validate_router_input(rows, activation_bf16_bits)?;
        self.execute_inner(
            layer,
            phase,
            placement_epoch,
            rows,
            RouterInput::Host(activation_bf16_bits),
            source_activation,
            route_records,
            timeline,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_device_and_download(
        &mut self,
        layer: u16,
        phase: GptOssPhase,
        placement_epoch: u64,
        rows: usize,
        activation: &CudaSlice<u16>,
        source_device: &StableCudaDeviceId,
        source_activation: &mut BoundedPinnedLease<u16>,
        route_records: &mut BoundedPinnedLease<u8>,
        timeline: Option<&CorrelatedTimeline>,
    ) -> Result<ExactRouterExecution> {
        let expected = rows.checked_mul(GPT_OSS_HIDDEN_SIZE).ok_or_else(|| {
            LLMError::ModelError("device router activation shape overflows".into())
        })?;
        if rows == 0
            || rows > GPT_OSS_ROUTER_MAX_ROWS
            || activation.len() != expected
            || source_device != &self.stable_device
        {
            return Err(LLMError::GpuError(
                "device-resident router input shape/device mismatch".into(),
            ));
        }
        self.execute_inner(
            layer,
            phase,
            placement_epoch,
            rows,
            RouterInput::Device(activation),
            source_activation,
            route_records,
            timeline,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_inner(
        &mut self,
        layer: u16,
        phase: GptOssPhase,
        placement_epoch: u64,
        rows: usize,
        activation: RouterInput<'_>,
        source_activation: &mut BoundedPinnedLease<u16>,
        route_records: &mut BoundedPinnedLease<u8>,
        timeline: Option<&CorrelatedTimeline>,
    ) -> Result<ExactRouterExecution> {
        if self.poisoned || rows > self.max_rows {
            return Err(LLMError::GpuError(format!(
                "exact router is poisoned or rows {rows} exceed reserved maximum {}",
                self.max_rows,
            )));
        }
        let input_len = rows * GPT_OSS_HIDDEN_SIZE;
        let route_count = rows * GPT_OSS_TOP_K;
        if source_activation.as_slice().len() < input_len
            || route_records.as_slice().len() < route_count * GPT_OSS_ROUTE_WIRE_V1_BYTES
        {
            return Err(LLMError::GpuError(
                "exact router pinned relay leases are too small".into(),
            ));
        }
        #[cfg(feature = "heterogeneous-test-faults")]
        let injected_fault = self.injected_fault.take();

        let submitted = (|| -> Result<_> {
            match activation {
                RouterInput::Host(values) => self
                    .compute_stream
                    .memcpy_htod(values, &mut self.input.slice_mut(..input_len))
                    .map_err(cuda_error("exact router input H2D"))?,
                RouterInput::Device(values) => self
                    .compute_stream
                    .memcpy_dtod(values, &mut self.input.slice_mut(..input_len))
                    .map_err(cuda_error("exact router input D2D"))?,
            }
            #[cfg(feature = "heterogeneous-test-faults")]
            if matches!(
                injected_fault,
                Some(
                    ExactRouterInjectedFault::SubmitAfterInputEnqueue
                        | ExactRouterInjectedFault::SubmitAfterInputEnqueueAndFallbackDrainFailure
                )
            ) {
                return Err(LLMError::GpuError(
                    "injected exact-router post-input-enqueue failure".into(),
                ));
            }
            if let Some(timeline) = timeline {
                timeline.enqueue_cuda_marker(
                    &self.compute_stream,
                    "gpu0_router",
                    "router_begin",
                )?;
            }
            let start = self
                .compute_stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("exact router start event"))?;
            launch_projection(
                &self.compute_stream,
                &self.loader,
                &self.input,
                &self.weights,
                &self.bias,
                &mut self.logits,
                rows,
                self.experts,
            )?;
            launch_top4(
                &self.compute_stream,
                &self.loader,
                &self.logits,
                &mut self.route_records,
                &mut self.status,
                rows,
                self.experts,
            )?;
            if let Some(timeline) = timeline {
                timeline.enqueue_cuda_marker(&self.compute_stream, "gpu0_router", "router_end")?;
            }
            let routes_ready = self
                .compute_stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("exact router routes-ready event"))?;
            self.relay_stream
                .wait(&routes_ready)
                .map_err(cuda_error("exact router relay wait"))?;
            if let Some(timeline) = timeline {
                timeline.enqueue_cuda_marker(
                    &self.relay_stream,
                    "gpu0_relay",
                    "source_d2h_begin",
                )?;
            }
            self.relay_stream
                .memcpy_dtoh(
                    &self.input.slice(..input_len),
                    &mut source_activation.as_mut_slice()[..input_len],
                )
                .map_err(cuda_error("exact router source activation D2H"))?;
            #[cfg(feature = "heterogeneous-test-faults")]
            if injected_fault == Some(ExactRouterInjectedFault::RelayAfterSourceEnqueue) {
                return Err(LLMError::GpuError(
                    "injected exact-router post-source-D2H-enqueue failure".into(),
                ));
            }
            let descriptor_bytes = route_count * GPT_OSS_ROUTE_WIRE_V1_BYTES;
            self.relay_stream
                .memcpy_dtoh(
                    &self.route_records.slice(..descriptor_bytes),
                    &mut route_records.as_mut_slice()[..descriptor_bytes],
                )
                .map_err(cuda_error("exact router canonical descriptors D2H"))?;
            if let Some(timeline) = timeline {
                timeline.enqueue_cuda_marker(&self.relay_stream, "gpu0_relay", "source_d2h_end")?;
            }
            let relay_terminal = self
                .relay_stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("exact router relay terminal event"))?;
            Ok((start, routes_ready, relay_terminal))
        })();

        let (start, routes_ready, relay_terminal) = match submitted {
            Ok(events) => events,
            Err(primary) => {
                #[cfg(feature = "heterogeneous-test-faults")]
                if injected_fault
                    == Some(
                        ExactRouterInjectedFault::SubmitAfterInputEnqueueAndFallbackDrainFailure,
                    )
                {
                    self.poisoned = true;
                    return Err(LLMError::GpuError(format!(
                        "exact router submit failed ({primary}); injected mandatory fallback drain failure; router CUDA state and caller-owned pinned leases must remain quarantined"
                    )));
                }
                let drained = self.drain();
                #[cfg(feature = "heterogeneous-test-faults")]
                if matches!(
                    injected_fault,
                    Some(
                        ExactRouterInjectedFault::SubmitAfterInputEnqueue
                            | ExactRouterInjectedFault::RelayAfterSourceEnqueue
                    )
                ) && drained.is_ok()
                {
                    self.last_fault_drained = true;
                }
                if let Err(drain) = drained {
                    return Err(LLMError::GpuError(format!(
                        "exact router submit failed ({primary}); mandatory drain failed ({drain})"
                    )));
                }
                return Err(primary);
            }
        };
        if let Err(error) = relay_terminal.synchronize() {
            let primary = cuda_error("exact router relay terminal drain")(error);
            let drained = self.drain();
            return match drained {
                Ok(()) => Err(primary),
                Err(drain) => Err(LLMError::GpuError(format!(
                    "exact router terminal failed ({primary}); mandatory drain failed ({drain})"
                ))),
            };
        }
        let router_elapsed_ms = start
            .elapsed_ms(&routes_ready)
            .map_err(cuda_error("exact router event timing"))?;
        let status = self
            .compute_stream
            .clone_dtoh(&self.status.slice(..rows))
            .map_err(cuda_error("exact router status D2H"))?;
        let logits = self
            .compute_stream
            .clone_dtoh(&self.logits.slice(..rows * self.experts))
            .map_err(cuda_error("exact router logits D2H"))?;
        if let Err(error) = self.compute_stream.synchronize() {
            self.poisoned = true;
            return Err(cuda_error("exact router evidence D2H drain")(error));
        }
        if let Some((row, code)) = status
            .iter()
            .copied()
            .enumerate()
            .find(|(_, code)| *code != 0)
        {
            return Err(LLMError::GpuError(match code {
                1 => format!("exact router rejected non-finite BF16 logit in row {row}"),
                2 => format!("exact router rejected unsupported expert count in row {row}"),
                _ => format!("exact router returned unknown status {code} in row {row}"),
            }));
        }
        let descriptor_bytes = route_count * GPT_OSS_ROUTE_WIRE_V1_BYTES;
        let wire_records: &[GptOssRouteWireV1] = bytemuck::try_cast_slice(
            &route_records.as_slice()[..descriptor_bytes],
        )
        .map_err(|error| LLMError::GpuError(format!("GPU route wire alignment/layout: {error}")))?;
        let mut routes = Vec::with_capacity(route_count);
        for (slot, wire) in wire_records.iter().copied().enumerate() {
            let route = wire
                .into_descriptor()
                .map_err(|error| LLMError::GpuError(format!("GPU route wire: {error}")))?;
            let expected_row = (slot / GPT_OSS_TOP_K) as u32;
            let expected_rank = (slot % GPT_OSS_TOP_K) as u8;
            if route.source_row != expected_row
                || route.route_rank != expected_rank
                || route.activation_slot != expected_row
                || usize::from(route.expert_id) >= self.experts
            {
                return Err(LLMError::GpuError(format!(
                    "GPU-authored route {slot} is noncanonical: row={} rank={} activation={} expert={}",
                    route.source_row, route.route_rank, route.activation_slot, route.expert_id
                )));
            }
            routes.push(route);
        }
        let batch = GptOssRoutedBatchDescriptor {
            layer,
            phase,
            rows: rows as u32,
            hidden_size: GPT_OSS_HIDDEN_SIZE as u16,
            experts_per_layer: self.experts as u16,
            placement_epoch,
            activation_bf16_bits: source_activation.as_slice()[..input_len].to_vec(),
            routes,
        };
        batch
            .validate()
            .map_err(|error| LLMError::GpuError(format!("GPU router batch: {error}")))?;
        Ok(ExactRouterExecution {
            batch,
            router_logits_bf16_bits: logits,
            router_elapsed_ms,
            source_d2h_bytes: input_len * size_of::<u16>(),
            descriptor_d2h_bytes: descriptor_bytes,
        })
    }
}

impl Drop for CudaExactRouter {
    fn drop(&mut self) {
        if self.poisoned {
            return;
        }
        // SAFETY: a healthy router drains before any fallible return that may
        // follow enqueue. The poisoned path intentionally retains every CUDA
        // allocation, loader, stream, and their context Arcs for process life.
        unsafe {
            ManuallyDrop::drop(&mut self.status);
            ManuallyDrop::drop(&mut self.route_records);
            ManuallyDrop::drop(&mut self.logits);
            ManuallyDrop::drop(&mut self.bias);
            ManuallyDrop::drop(&mut self.weights);
            ManuallyDrop::drop(&mut self.input);
            ManuallyDrop::drop(&mut self.loader);
            ManuallyDrop::drop(&mut self.relay_stream);
            ManuallyDrop::drop(&mut self.compute_stream);
        }
    }
}

fn validate_router_max_rows(max_rows: usize) -> Result<()> {
    if max_rows == 0 || max_rows > GPT_OSS_ROUTER_MAX_ROWS {
        return Err(LLMError::GpuError(format!(
            "exact router max rows {max_rows} outside 1..={GPT_OSS_ROUTER_MAX_ROWS}"
        )));
    }
    Ok(())
}

pub(crate) fn exact_router_weight_surface_bytes(experts: usize) -> Result<(usize, usize)> {
    if !matches!(experts, 32 | 128) {
        return Err(LLMError::GpuError(format!(
            "resident exact router supports E=32 or E=128, observed {experts}"
        )));
    }
    let weight_values = experts
        .checked_mul(GPT_OSS_HIDDEN_SIZE)
        .ok_or_else(|| LLMError::GpuError("resident router weight shape overflows".into()))?;
    let weight_bytes = weight_values
        .checked_mul(size_of::<u16>())
        .ok_or_else(|| LLMError::GpuError("resident router weight bytes overflow".into()))?;
    let bias_bytes = experts
        .checked_mul(size_of::<u16>())
        .ok_or_else(|| LLMError::GpuError("resident router bias bytes overflow".into()))?;
    Ok((weight_bytes, bias_bytes))
}

fn allocate_terminal<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    len: usize,
    label: &'static str,
) -> Result<CudaSlice<T>> {
    // SAFETY: every allocation is filled before the router can execute. The
    // allocation itself is synchronized here so a later construction error
    // cannot strand an untracked cudaMallocAsync operation.
    let allocation = unsafe { stream.alloc::<T>(len) }.map_err(cuda_error(label))?;
    if let Err(error) = stream.synchronize() {
        // The allocation may still be referenced by an unproven stream. Its
        // slice retains the stream and context for process life.
        std::mem::forget(allocation);
        return Err(cuda_error("resident router allocation drain")(error));
    }
    Ok(allocation)
}

fn resident_handoff_submit_failure(
    mut state: ResidentRouterHandoffState,
    primary: LLMError,
    inject_unproven_drain: bool,
) -> Result<CudaExactRouter> {
    if inject_unproven_drain {
        state.quarantine();
        return Err(LLMError::GpuError(format!(
            "resident router handoff failed ({primary}); injected mandatory fallback drain failure; source, destination, stream, and context are quarantined"
        )));
    }
    match state.compute_stream.synchronize() {
        Ok(()) => Err(primary),
        Err(drain) => {
            state.quarantine();
            Err(LLMError::GpuError(format!(
                "resident router handoff failed ({primary}); mandatory fallback drain failed ({drain}); source, destination, stream, and context are quarantined"
            )))
        }
    }
}

fn validate_router_input(rows: usize, activation_bf16_bits: &[u16]) -> Result<()> {
    if rows == 0 || rows > GPT_OSS_ROUTER_MAX_ROWS {
        return Err(LLMError::ModelError(format!(
            "exact router rows {rows} outside 1..={GPT_OSS_ROUTER_MAX_ROWS}"
        )));
    }
    let expected = rows
        .checked_mul(GPT_OSS_HIDDEN_SIZE)
        .ok_or_else(|| LLMError::ModelError("router activation shape overflows".into()))?;
    if activation_bf16_bits.len() != expected {
        return Err(LLMError::ModelError(format!(
            "exact router activation length {} != {expected}",
            activation_bf16_bits.len()
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_projection(
    stream: &Arc<CudaStream>,
    loader: &KernelLoader,
    input: &CudaSlice<u16>,
    weights: &CudaSlice<u16>,
    bias: &CudaSlice<u16>,
    logits: &mut CudaSlice<u16>,
    rows: usize,
    experts: usize,
) -> Result<()> {
    let function = loader.get_func(MODULE, PROJECTION)?;
    let rows_i32 = rows as i32;
    let experts_i32 = experts as i32;
    let hidden_i32 = GPT_OSS_HIDDEN_SIZE as i32;
    let outputs = rows as u32 * experts as u32;
    let config = LaunchConfig {
        grid_dim: (outputs.div_ceil(PROJECTION_THREADS), 1, 1),
        block_dim: (PROJECTION_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: fixed validated shapes match the CUDA signature and every slice
    // remains owned by the executor until its routes-ready event drains.
    unsafe {
        stream
            .launch_builder(&function)
            .arg(input)
            .arg(weights)
            .arg(bias)
            .arg(logits)
            .arg(&rows_i32)
            .arg(&experts_i32)
            .arg(&hidden_i32)
            .launch(config)
            .map_err(cuda_error("exact router projection launch"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_top4(
    stream: &Arc<CudaStream>,
    loader: &KernelLoader,
    logits: &CudaSlice<u16>,
    route_records: &mut CudaSlice<u8>,
    status: &mut CudaSlice<u32>,
    rows: usize,
    experts: usize,
) -> Result<()> {
    let function = loader.get_func(MODULE, TOP4)?;
    let rows_i32 = rows as i32;
    let experts_i32 = experts as i32;
    let config = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(ROUTER_THREADS), 1, 1),
        block_dim: (ROUTER_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: the route arena has four 16-byte records per reserved row and
    // remains owned by the executor until the relay terminal event drains.
    unsafe {
        stream
            .launch_builder(&function)
            .arg(logits)
            .arg(route_records)
            .arg(status)
            .arg(&rows_i32)
            .arg(&experts_i32)
            .launch(config)
            .map_err(cuda_error("exact router stable-top4 launch"))?;
    }
    Ok(())
}

fn cuda_error(context: &'static str) -> impl FnOnce(cudarc::driver::DriverError) -> LLMError {
    move |error| LLMError::GpuError(format!("{context}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(experts: usize, rows: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let activation = (0..rows * GPT_OSS_HIDDEN_SIZE)
            .map(|index| bf16::from_f32((index as f32 % 31.0 - 15.0) / 16.0).to_bits())
            .collect::<Vec<_>>();
        let weights = (0..experts * GPT_OSS_HIDDEN_SIZE)
            .map(|index| bf16::from_f32((index as f32 % 17.0 - 8.0) / 64.0).to_bits())
            .collect::<Vec<_>>();
        let bias = (0..experts)
            .map(|expert| bf16::from_f32((expert % 7) as f32 / 16.0).to_bits())
            .collect::<Vec<_>>();
        (activation, weights, bias)
    }

    #[test]
    fn reference_covers_e32_e128_and_stable_lower_id_ties() {
        for experts in [32, 128] {
            let (activation, mut weights, mut bias) = fixture(experts, 2);
            weights.fill(bf16::from_f32(0.0).to_bits());
            bias.fill(bf16::from_f32(0.0).to_bits());
            let result = exact_router_reference(
                0,
                GptOssPhase::Decode,
                1,
                2,
                &activation,
                ExactRouterWeightsView {
                    experts,
                    weight_bf16_bits: &weights,
                    bias_bf16_bits: &bias,
                },
            )
            .unwrap();
            for row in result.batch.routes.chunks_exact(GPT_OSS_TOP_K) {
                assert_eq!(
                    row.iter().map(|route| route.expert_id).collect::<Vec<_>>(),
                    vec![0, 1, 2, 3]
                );
                assert!(row
                    .iter()
                    .all(|route| route.weight_bf16_bits == bf16::from_f32(0.25).to_bits()));
            }
        }
    }

    #[test]
    fn reference_rejects_non_finite_after_bias_placement() {
        let (activation, weights, mut bias) = fixture(32, 1);
        bias[7] = bf16::NAN.to_bits();
        assert!(exact_router_reference(
            0,
            GptOssPhase::Decode,
            1,
            1,
            &activation,
            ExactRouterWeightsView {
                experts: 32,
                weight_bf16_bits: &weights,
                bias_bf16_bits: &bias,
            },
        )
        .is_err());
    }

    #[test]
    fn resident_handoff_dimensions_are_exact_and_checked() {
        assert_eq!(
            exact_router_weight_surface_bytes(32).unwrap(),
            (368_640, 64)
        );
        assert_eq!(
            exact_router_weight_surface_bytes(128).unwrap(),
            (737_280, 256)
        );
        assert!(exact_router_weight_surface_bytes(0).is_err());
        assert!(exact_router_weight_surface_bytes(64).is_err());
        assert!(validate_router_max_rows(0).is_err());
        assert!(validate_router_max_rows(GPT_OSS_ROUTER_MAX_ROWS + 1).is_err());
        assert!(validate_router_max_rows(1).is_ok());
        assert!(validate_router_max_rows(GPT_OSS_ROUTER_MAX_ROWS).is_ok());
    }
}
