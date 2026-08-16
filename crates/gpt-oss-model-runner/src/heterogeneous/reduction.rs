//! Exact GPU0 weighting and reduction in original routing-rank order.

use std::sync::Arc;

use cudarc::driver::{sys::CUevent_flags, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use gpt_oss_core::error::{LLMError, Result};
use gpt_oss_gpu::device::StableCudaDeviceId;
use gpt_oss_gpu::kernel_loader::{compiled_ptx_dir, KernelLoader};
use half::bf16;

use super::contract::{
    group_routes_stably, CanonicalRouteContract, ExpertResultDescriptor, GptOssPhase,
    GptOssRoutedBatchDescriptor, GPT_OSS_HIDDEN_SIZE, GPT_OSS_TOP_K,
};
use super::placement::ResolvedExpertPlacement;
use super::relay::CudaResultRelay;

const MODULE: &str = "gpt_oss_rank_reduction";
const REDUCE: &str = "gpt_oss_rank_order_reduce_bf16_kernel";
const THREADS: u32 = 256;

pub const GPT_OSS_REDUCTION_CONTRIBUTION_BYTES: usize =
    GPT_OSS_TOP_K * GPT_OSS_HIDDEN_SIZE * size_of::<u16>();
pub const GPT_OSS_REDUCTION_WEIGHT_BYTES: usize = GPT_OSS_TOP_K * size_of::<u16>();
pub const GPT_OSS_REDUCTION_OUTPUT_BYTES: usize = GPT_OSS_HIDDEN_SIZE * size_of::<u16>();
pub const GPT_OSS_REDUCTION_TRACE_BYTES: usize =
    2 * GPT_OSS_TOP_K * GPT_OSS_HIDDEN_SIZE * size_of::<u32>();
/// Bytes owned by the reducer in addition to H4's canonical contribution arena.
pub const GPT_OSS_REDUCER_OWNED_DEVICE_BYTES: usize =
    GPT_OSS_REDUCTION_WEIGHT_BYTES + GPT_OSS_REDUCTION_OUTPUT_BYTES + GPT_OSS_REDUCTION_TRACE_BYTES;
/// Full H4+H5 GPU0 reduction pipeline high-water accounting.
pub const GPT_OSS_REDUCTION_DEVICE_WORK_BYTES: usize =
    GPT_OSS_REDUCTION_CONTRIBUTION_BYTES + GPT_OSS_REDUCER_OWNED_DEVICE_BYTES;
pub const GPT_OSS_REDUCTION_WORKSPACE_CLASS_BYTES: usize = 128 * 1024;

const _: () = assert!(GPT_OSS_REDUCER_OWNED_DEVICE_BYTES == 97_928);
const _: () = assert!(GPT_OSS_REDUCTION_DEVICE_WORK_BYTES == 120_968);
const _: () =
    assert!(GPT_OSS_REDUCTION_DEVICE_WORK_BYTES <= GPT_OSS_REDUCTION_WORKSPACE_CLASS_BYTES);

/// Oracle-only host contribution. Production reduction consumes H4's resident
/// GPU0 canonical contribution arena and never reuploads these rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalExpertContribution {
    pub descriptor: ExpertResultDescriptor,
    pub output_bf16_bits: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankOrderedReductionTrace {
    /// One f32 bit pattern per `[route_rank][hidden]` weighted contribution.
    pub weighted_f32_bits: Vec<u32>,
    /// Accumulator after each rank, in the same canonical layout.
    pub accumulator_f32_bits: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankOrderedReductionExecution {
    pub output_bf16_bits: Vec<u16>,
    pub trace: RankOrderedReductionTrace,
    pub kernel_elapsed_ms: f32,
}

/// Every fallible host allocation and route canonicalization happens here,
/// before any H4 dispatch or CUDA enqueue. Submission only validates fixed
/// descriptors and moves these pre-sized buffers into the result.
pub struct PreparedRankOrderedReduction {
    transaction_generation: u64,
    expected: Vec<ExpertResultDescriptor>,
    route_contracts: [CanonicalRouteContract; GPT_OSS_TOP_K],
    weights_bf16_bits: Vec<u16>,
    output_bf16_bits: Vec<u16>,
    weighted_f32_bits: Vec<u32>,
    accumulator_f32_bits: Vec<u32>,
}

impl PreparedRankOrderedReduction {
    pub fn prepare(
        batch: &GptOssRoutedBatchDescriptor,
        placement: &ResolvedExpertPlacement,
        transaction_generation: u64,
    ) -> Result<Self> {
        if transaction_generation == 0 {
            return Err(LLMError::ModelError(
                "rank reduction transaction generation must be nonzero".into(),
            ));
        }
        validate_decode_batch(batch)?;
        let packed = group_routes_stably(batch, placement)
            .map_err(|error| LLMError::ModelError(format!("reduction placement: {error}")))?;
        let mut expected = vec![None::<ExpertResultDescriptor>; GPT_OSS_TOP_K];
        let mut route_contracts = [None::<CanonicalRouteContract>; GPT_OSS_TOP_K];
        for route in packed {
            let slot = route.canonical_result_slot as usize;
            if slot >= expected.len()
                || expected[slot]
                    .replace(ExpertResultDescriptor::from_packed_route(&route))
                    .is_some()
                || route_contracts[slot]
                    .replace(CanonicalRouteContract::from_packed_route(&route))
                    .is_some()
            {
                return Err(LLMError::ModelError(
                    "reduction expected-route slots are not unique".into(),
                ));
            }
        }
        let expected = expected
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                LLMError::ModelError("reduction expected-route slot is missing".into())
            })?;
        let route_contracts = match route_contracts {
            [Some(rank0), Some(rank1), Some(rank2), Some(rank3)] => [rank0, rank1, rank2, rank3],
            _ => {
                return Err(LLMError::ModelError(
                    "reduction expected-route contract is missing".into(),
                ));
            }
        };
        Ok(Self {
            transaction_generation,
            expected,
            route_contracts,
            weights_bf16_bits: batch
                .routes
                .iter()
                .map(|route| route.weight_bf16_bits)
                .collect(),
            output_bf16_bits: vec![0_u16; GPT_OSS_HIDDEN_SIZE],
            weighted_f32_bits: vec![0_u32; GPT_OSS_TOP_K * GPT_OSS_HIDDEN_SIZE],
            accumulator_f32_bits: vec![0_u32; GPT_OSS_TOP_K * GPT_OSS_HIDDEN_SIZE],
        })
    }

    /// Canonical descriptors are materialized before dispatch. CPU/GPU
    /// completion paths borrow these records rather than allocating or
    /// reconstructing result identity after work has entered a stream.
    pub fn expected_results(&self) -> &[ExpertResultDescriptor] {
        &self.expected
    }

    fn validate_results(&self, results: &[ExpertResultDescriptor]) -> Result<()> {
        if results.len() != GPT_OSS_TOP_K {
            return Err(LLMError::ModelError(format!(
                "rank reduction received {} results, expected {GPT_OSS_TOP_K}",
                results.len()
            )));
        }
        let mut populated = [false; GPT_OSS_TOP_K];
        for result in results {
            let slot = result.result_slot as usize;
            if slot >= GPT_OSS_TOP_K || populated[slot] {
                return Err(LLMError::ModelError(format!(
                    "reduction result slot {} is out of range or duplicated",
                    result.result_slot
                )));
            }
            if result != &self.expected[slot] {
                return Err(LLMError::ModelError(format!(
                    "reduction result identity mismatch for slot {}",
                    result.result_slot
                )));
            }
            populated[slot] = true;
        }
        if populated.iter().any(|present| !present) {
            return Err(LLMError::ModelError(
                "rank reduction is missing a canonical result slot".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankReductionInjectedFault {
    AfterWeightEnqueue,
    AfterKernelLaunch,
    AfterEvidenceEnqueue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankReductionConstructionFault {
    AfterWeights,
    AfterOutput,
    AfterWeightedTrace,
    AfterAccumulatorTrace,
}

fn construction_fault_after(
    stream: &CudaStream,
    armed: Option<RankReductionConstructionFault>,
    boundary: RankReductionConstructionFault,
) -> Result<()> {
    if armed != Some(boundary) {
        return Ok(());
    }
    // `alloc_zeros` may have enqueued initialization. Prove it terminal before
    // returning and allowing already-created device slices to drop.
    stream.synchronize().map_err(cuda_error(
        "rank reduction injected construction-fault drain",
    ))?;
    Err(LLMError::MemoryError(format!(
        "injected rank reduction construction failure at {boundary:?}"
    )))
}

pub fn exact_rank_ordered_reduction_reference(
    batch: &GptOssRoutedBatchDescriptor,
    placement: &ResolvedExpertPlacement,
    contributions: &[CanonicalExpertContribution],
) -> Result<RankOrderedReductionExecution> {
    let (arena, weights) = canonicalize_oracle_contributions(batch, placement, contributions)?;
    let mut output_bf16_bits = vec![0_u16; GPT_OSS_HIDDEN_SIZE];
    let mut weighted_f32_bits = vec![0_u32; GPT_OSS_TOP_K * GPT_OSS_HIDDEN_SIZE];
    let mut accumulator_f32_bits = vec![0_u32; GPT_OSS_TOP_K * GPT_OSS_HIDDEN_SIZE];
    for column in 0..GPT_OSS_HIDDEN_SIZE {
        let mut accumulator = 0.0_f32;
        for rank in 0..GPT_OSS_TOP_K {
            let index = rank * GPT_OSS_HIDDEN_SIZE + column;
            let value = bf16::from_bits(arena[index]).to_f32();
            let weight = bf16::from_bits(weights[rank]).to_f32();
            let weighted = value * weight;
            accumulator += weighted;
            weighted_f32_bits[index] = weighted.to_bits();
            accumulator_f32_bits[index] = accumulator.to_bits();
        }
        output_bf16_bits[column] = bf16::from_f32(accumulator).to_bits();
    }
    Ok(RankOrderedReductionExecution {
        output_bf16_bits,
        trace: RankOrderedReductionTrace {
            weighted_f32_bits,
            accumulator_f32_bits,
        },
        kernel_elapsed_ms: 0.0,
    })
}

/// Decode-M=1 production reducer. It shares H4's GPU0 relay stream/context and
/// reads H4's canonical contribution arena in place. Only weights, output and
/// optional first-divergence traces are owned here.
pub struct CudaRankOrderedReducer {
    stable_device: StableCudaDeviceId,
    stream: Arc<CudaStream>,
    loader: KernelLoader,
    weights: CudaSlice<u16>,
    output: CudaSlice<u16>,
    weighted_trace: CudaSlice<u32>,
    accumulator_trace: CudaSlice<u32>,
    poisoned: bool,
    quarantined_host: Option<PreparedRankOrderedReduction>,
    #[cfg(feature = "heterogeneous-test-faults")]
    injected_fault: Option<RankReductionInjectedFault>,
    #[cfg(feature = "heterogeneous-test-faults")]
    last_fault_drained: bool,
}

impl CudaRankOrderedReducer {
    pub fn new(relay: &CudaResultRelay) -> Result<Self> {
        Self::new_inner(relay, None)
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub fn new_with_construction_fault(
        relay: &CudaResultRelay,
        fault: RankReductionConstructionFault,
    ) -> Result<Self> {
        Self::new_inner(relay, Some(fault))
    }

    fn new_inner(
        relay: &CudaResultRelay,
        construction_fault: Option<RankReductionConstructionFault>,
    ) -> Result<Self> {
        if relay.max_routes() < GPT_OSS_TOP_K {
            return Err(LLMError::GpuError(
                "rank reducer requires four H4 canonical result slots".into(),
            ));
        }
        let stable_device = relay.stable_device().clone();
        let stream = Arc::clone(relay.stream());
        let loader = KernelLoader::new(
            Arc::clone(stream.context()),
            Arc::clone(&stream),
            compiled_ptx_dir(),
        )?;
        if !loader.has_func(MODULE, REDUCE) {
            return Err(LLMError::GpuError(
                "rank-ordered reduction PTX function is unavailable".into(),
            ));
        }
        let weights = stream
            .alloc_zeros::<u16>(GPT_OSS_TOP_K)
            .map_err(cuda_error("rank reduction weight allocation"))?;
        construction_fault_after(
            &stream,
            construction_fault,
            RankReductionConstructionFault::AfterWeights,
        )?;
        let output = stream
            .alloc_zeros::<u16>(GPT_OSS_HIDDEN_SIZE)
            .map_err(cuda_error("rank reduction output allocation"))?;
        construction_fault_after(
            &stream,
            construction_fault,
            RankReductionConstructionFault::AfterOutput,
        )?;
        let weighted_trace = stream
            .alloc_zeros::<u32>(GPT_OSS_TOP_K * GPT_OSS_HIDDEN_SIZE)
            .map_err(cuda_error("rank reduction weighted-trace allocation"))?;
        construction_fault_after(
            &stream,
            construction_fault,
            RankReductionConstructionFault::AfterWeightedTrace,
        )?;
        let accumulator_trace = stream
            .alloc_zeros::<u32>(GPT_OSS_TOP_K * GPT_OSS_HIDDEN_SIZE)
            .map_err(cuda_error("rank reduction accumulator-trace allocation"))?;
        construction_fault_after(
            &stream,
            construction_fault,
            RankReductionConstructionFault::AfterAccumulatorTrace,
        )?;
        stream
            .synchronize()
            .map_err(cuda_error("rank reduction construction drain"))?;
        Ok(Self {
            stable_device,
            stream,
            loader,
            weights,
            output,
            weighted_trace,
            accumulator_trace,
            poisoned: false,
            quarantined_host: None,
            #[cfg(feature = "heterogeneous-test-faults")]
            injected_fault: None,
            #[cfg(feature = "heterogeneous-test-faults")]
            last_fault_drained: false,
        })
    }

    pub fn stable_device(&self) -> &StableCudaDeviceId {
        &self.stable_device
    }

    pub const fn owned_device_bytes(&self) -> usize {
        GPT_OSS_REDUCER_OWNED_DEVICE_BYTES
    }

    pub const fn pipeline_device_bytes(&self) -> usize {
        GPT_OSS_REDUCTION_DEVICE_WORK_BYTES
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub fn inject_next_failure(&mut self, fault: RankReductionInjectedFault) -> Result<()> {
        if self.injected_fault.is_some() {
            return Err(LLMError::GpuError(
                "rank reduction already has an armed fault".into(),
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

    pub fn reduce_relay(
        &mut self,
        relay: &mut CudaResultRelay,
        mut prepared: PreparedRankOrderedReduction,
    ) -> Result<RankOrderedReductionExecution> {
        if self.poisoned {
            return Err(LLMError::GpuError(
                "rank reduction stream is poisoned".into(),
            ));
        }
        if relay.stable_device() != &self.stable_device
            || !Arc::ptr_eq(relay.stream(), &self.stream)
        {
            return Err(LLMError::GpuError(
                "rank reducer and H4 canonical arena do not share GPU0 relay stream/context".into(),
            ));
        }
        if prepared.transaction_generation != relay.arena_generation() {
            return Err(LLMError::GpuError(format!(
                "prepared reduction generation {} does not match canonical arena generation {}",
                prepared.transaction_generation,
                relay.arena_generation()
            )));
        }
        relay.validate_reduction_contract(
            prepared.transaction_generation,
            &prepared.route_contracts,
        )?;
        relay.validate_complete_decode_results(&prepared.expected)?;
        #[cfg(feature = "heterogeneous-test-faults")]
        let injected_fault = self.injected_fault.take();
        #[cfg(not(feature = "heterogeneous-test-faults"))]
        let injected_fault = None;

        let submitted = (|| -> Result<_> {
            self.stream
                .memcpy_htod(&prepared.weights_bf16_bits, &mut self.weights)
                .map_err(cuda_error("rank reduction weight H2D"))?;
            #[cfg(feature = "heterogeneous-test-faults")]
            if injected_fault == Some(RankReductionInjectedFault::AfterWeightEnqueue) {
                return Err(LLMError::GpuError(
                    "injected rank reduction post-weight-enqueue failure".into(),
                ));
            }
            let start = self
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("rank reduction start event"))?;
            let function = self.loader.get_func(MODULE, REDUCE)?;
            let rows = 1_i32;
            let hidden = GPT_OSS_HIDDEN_SIZE as i32;
            let config = LaunchConfig {
                grid_dim: ((GPT_OSS_HIDDEN_SIZE as u32).div_ceil(THREADS), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: H4 owns four canonical decode rows until this terminal
            // event drains. Every other fixed allocation is reducer-owned.
            unsafe {
                self.stream
                    .launch_builder(&function)
                    .arg(relay.decode_contribution_arena())
                    .arg(&self.weights)
                    .arg(&mut self.output)
                    .arg(&mut self.weighted_trace)
                    .arg(&mut self.accumulator_trace)
                    .arg(&rows)
                    .arg(&hidden)
                    .launch(config)
                    .map_err(cuda_error("rank reduction launch"))?;
            }
            #[cfg(feature = "heterogeneous-test-faults")]
            if injected_fault == Some(RankReductionInjectedFault::AfterKernelLaunch) {
                return Err(LLMError::GpuError(
                    "injected rank reduction post-launch failure".into(),
                ));
            }
            let kernel_end = self
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("rank reduction kernel-end event"))?;
            self.stream
                .memcpy_dtoh(&self.output, &mut prepared.output_bf16_bits)
                .map_err(cuda_error("rank reduction output D2H"))?;
            self.stream
                .memcpy_dtoh(&self.weighted_trace, &mut prepared.weighted_f32_bits)
                .map_err(cuda_error("rank reduction weighted trace D2H"))?;
            self.stream
                .memcpy_dtoh(&self.accumulator_trace, &mut prepared.accumulator_f32_bits)
                .map_err(cuda_error("rank reduction accumulator trace D2H"))?;
            #[cfg(feature = "heterogeneous-test-faults")]
            if injected_fault == Some(RankReductionInjectedFault::AfterEvidenceEnqueue) {
                return Err(LLMError::GpuError(
                    "injected rank reduction post-evidence-enqueue failure".into(),
                ));
            }
            let terminal = self
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("rank reduction terminal event"))?;
            Ok((start, kernel_end, terminal))
        })();

        let (start, kernel_end, terminal) = match submitted {
            Ok(events) => events,
            Err(primary) => return self.fail_after_enqueue(primary, injected_fault, prepared),
        };
        if let Err(error) = terminal.synchronize() {
            return self.fail_after_enqueue(
                cuda_error("rank reduction terminal drain")(error),
                injected_fault,
                prepared,
            );
        }
        let kernel_elapsed_ms = start
            .elapsed_ms(&kernel_end)
            .map_err(cuda_error("rank reduction event timing"))?;
        relay.finish_reduction_generation(prepared.transaction_generation)?;
        Ok(RankOrderedReductionExecution {
            output_bf16_bits: prepared.output_bf16_bits,
            trace: RankOrderedReductionTrace {
                weighted_f32_bits: prepared.weighted_f32_bits,
                accumulator_f32_bits: prepared.accumulator_f32_bits,
            },
            kernel_elapsed_ms,
        })
    }

    fn fail_after_enqueue(
        &mut self,
        primary: LLMError,
        _fault: Option<RankReductionInjectedFault>,
        prepared: PreparedRankOrderedReduction,
    ) -> Result<RankOrderedReductionExecution> {
        let drained = self.stream.synchronize();
        #[cfg(feature = "heterogeneous-test-faults")]
        if _fault.is_some() && drained.is_ok() {
            self.last_fault_drained = true;
        }
        match drained {
            Ok(()) => Err(primary),
            Err(drain) => {
                self.poisoned = true;
                self.quarantined_host = Some(prepared);
                Err(LLMError::GpuError(format!(
                    "rank reduction failed ({primary}); mandatory drain failed ({drain}); stream poisoned and host D2H storage quarantined"
                )))
            }
        }
    }
}

impl Drop for CudaRankOrderedReducer {
    fn drop(&mut self) {
        if self.poisoned {
            if let Some(host) = self.quarantined_host.take() {
                // A failed terminal and fallback drain cannot prove that the
                // asynchronous D2H no longer references these allocations.
                std::mem::forget(host);
            }
        }
    }
}

fn validate_decode_batch(batch: &GptOssRoutedBatchDescriptor) -> Result<()> {
    batch
        .validate()
        .map_err(|error| LLMError::ModelError(format!("reduction routes: {error}")))?;
    if batch.phase != GptOssPhase::Decode || batch.rows != 1 {
        return Err(LLMError::ModelError(
            "rank reduction supports decode M=1 only".into(),
        ));
    }
    Ok(())
}

fn canonicalize_oracle_contributions(
    batch: &GptOssRoutedBatchDescriptor,
    placement: &ResolvedExpertPlacement,
    contributions: &[CanonicalExpertContribution],
) -> Result<(Vec<u16>, Vec<u16>)> {
    let prepared = PreparedRankOrderedReduction::prepare(batch, placement, 1)?;
    let descriptors = contributions
        .iter()
        .map(|contribution| contribution.descriptor.clone())
        .collect::<Vec<_>>();
    prepared.validate_results(&descriptors)?;
    let mut arena = vec![0_u16; GPT_OSS_TOP_K * GPT_OSS_HIDDEN_SIZE];
    for contribution in contributions {
        if contribution.output_bf16_bits.len() != GPT_OSS_HIDDEN_SIZE {
            return Err(LLMError::ModelError(format!(
                "reduction result slot {} has {} values, expected {GPT_OSS_HIDDEN_SIZE}",
                contribution.descriptor.result_slot,
                contribution.output_bf16_bits.len()
            )));
        }
        let start = contribution.descriptor.result_slot as usize * GPT_OSS_HIDDEN_SIZE;
        arena[start..start + GPT_OSS_HIDDEN_SIZE].copy_from_slice(&contribution.output_bf16_bits);
    }
    Ok((arena, prepared.weights_bf16_bits))
}

fn cuda_error(context: &'static str) -> impl FnOnce(cudarc::driver::DriverError) -> LLMError {
    move |error| LLMError::GpuError(format!("{context}: {error}"))
}
