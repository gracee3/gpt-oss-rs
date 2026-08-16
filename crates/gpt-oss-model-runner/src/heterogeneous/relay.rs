//! Fixed-capacity pinned relay reservations and GPU0 result-slot uploads.
//!
//! H4 owns five prewarmed host buffers. Reservations never allocate and are
//! all-or-none. A lease can return only after every CPU/CUDA reference drains.

use std::mem::ManuallyDrop;

use cudarc::driver::{sys::CUevent_flags, CudaSlice, CudaStream};
use gpt_oss_core::error::{LLMError, Result};
use gpt_oss_gpu::event::CorrelatedTimeline;
use gpt_oss_gpu::pinned_memory::{BoundedPinnedLease, BoundedPinnedPool, BoundedPinnedPoolStats};

use super::contract::{
    CanonicalExpertOwner, CanonicalRouteContract, ExpertResultDescriptor, GptOssPhase,
    GPT_OSS_HIDDEN_SIZE, GPT_OSS_ROUTE_WIRE_V1_BYTES, GPT_OSS_TOP_K,
};
use super::cuda_expert::CudaSelectedExpertResultSlot;
use super::packing::{
    relay_pinned_capacity_bytes, PackedDispatchPlan, RelayBytePlan, H4_PREFILL_MAX_ROWS,
    H4_ROUTE_DESCRIPTOR_MAX_BYTES,
};
use super::placement::ExpertOwner;
use super::router::CudaExactRouter;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelayPinnedPoolStats {
    pub source_activation: BoundedPinnedPoolStats,
    pub route_descriptors: BoundedPinnedPoolStats,
    pub remote_gpu_input: BoundedPinnedPoolStats,
    pub remote_gpu_result: BoundedPinnedPoolStats,
    pub cpu_result: BoundedPinnedPoolStats,
    pub raw_capacity_bytes: usize,
    pub hard_cap_bytes: usize,
}

pub fn result_relay_owned_device_bytes(max_rows: usize) -> Result<usize> {
    max_rows
        .checked_mul(GPT_OSS_TOP_K)
        .and_then(|routes| routes.checked_mul(GPT_OSS_HIDDEN_SIZE))
        .and_then(|values| values.checked_mul(size_of::<u16>()))
        .ok_or_else(|| LLMError::ModelError("result-relay device bytes overflow".into()))
}

/// Exactly five capacity-one pools sized for one decode row or one bounded
/// prefill chunk. Their capacities do not depend on the observed route mix.
pub struct RelayPinnedPools {
    max_rows: usize,
    source_activation: BoundedPinnedPool<u16>,
    route_descriptors: BoundedPinnedPool<u8>,
    remote_gpu_input: BoundedPinnedPool<u16>,
    remote_gpu_result: BoundedPinnedPool<u16>,
    cpu_result: BoundedPinnedPool<u16>,
}

impl RelayPinnedPools {
    pub fn warm_exact(router: &CudaExactRouter, max_rows: usize) -> Result<Self> {
        if max_rows == 0 || max_rows > H4_PREFILL_MAX_ROWS {
            return Err(LLMError::MemoryError(format!(
                "relay pool rows {max_rows} outside 1..={H4_PREFILL_MAX_ROWS}"
            )));
        }
        router
            .relay_stream()
            .context()
            .bind_to_thread()
            .map_err(cuda_error("relay pool context bind"))?;
        let route_capacity = max_rows * GPT_OSS_TOP_K;
        let pools = Self {
            max_rows,
            source_activation: BoundedPinnedPool::warm_exact(max_rows * GPT_OSS_HIDDEN_SIZE, 1)?,
            route_descriptors: BoundedPinnedPool::warm_exact(
                route_capacity * GPT_OSS_ROUTE_WIRE_V1_BYTES,
                1,
            )?,
            remote_gpu_input: BoundedPinnedPool::warm_exact(
                route_capacity * GPT_OSS_HIDDEN_SIZE,
                1,
            )?,
            remote_gpu_result: BoundedPinnedPool::warm_exact(
                route_capacity * GPT_OSS_HIDDEN_SIZE,
                1,
            )?,
            cpu_result: BoundedPinnedPool::warm_exact(route_capacity * GPT_OSS_HIDDEN_SIZE, 1)?,
        };
        let stats = pools.stats();
        if stats.raw_capacity_bytes > stats.hard_cap_bytes {
            return Err(LLMError::MemoryError(format!(
                "relay fixed capacity {} exceeds hard cap {}",
                stats.raw_capacity_bytes, stats.hard_cap_bytes
            )));
        }
        Ok(pools)
    }

    /// Reserve all five fixed buffers before any CPU task or CUDA enqueue.
    /// On exhaustion, every earlier reservation is returned immediately.
    pub fn try_reserve_all(&self, generation: u64) -> Result<RelayPinnedReservation> {
        let mut source_activation = None;
        let mut route_descriptors = None;
        let mut remote_gpu_input = None;
        let mut remote_gpu_result = None;
        let mut cpu_result = None;
        let acquired = (|| -> Result<()> {
            source_activation = Some(self.source_activation.try_acquire(generation)?);
            route_descriptors = Some(self.route_descriptors.try_acquire(generation)?);
            remote_gpu_input = Some(self.remote_gpu_input.try_acquire(generation)?);
            remote_gpu_result = Some(self.remote_gpu_result.try_acquire(generation)?);
            cpu_result = Some(self.cpu_result.try_acquire(generation)?);
            Ok(())
        })();
        if let Err(error) = acquired {
            release_if_present(cpu_result)?;
            release_if_present(remote_gpu_result)?;
            release_if_present(remote_gpu_input)?;
            release_if_present(route_descriptors)?;
            release_if_present(source_activation)?;
            return Err(error);
        }
        Ok(RelayPinnedReservation {
            generation,
            source_activation: source_activation.expect("source lease acquired"),
            route_descriptors: route_descriptors.expect("descriptor lease acquired"),
            remote_gpu_input: remote_gpu_input.expect("remote input lease acquired"),
            remote_gpu_result: remote_gpu_result.expect("remote result lease acquired"),
            cpu_result: cpu_result.expect("CPU result lease acquired"),
        })
    }

    pub fn stats(&self) -> RelayPinnedPoolStats {
        let source_activation = self.source_activation.stats();
        let route_descriptors = self.route_descriptors.stats();
        let remote_gpu_input = self.remote_gpu_input.stats();
        let remote_gpu_result = self.remote_gpu_result.stats();
        let cpu_result = self.cpu_result.stats();
        let raw_capacity_bytes = source_activation.bytes_per_buffer
            + route_descriptors.bytes_per_buffer
            + remote_gpu_input.bytes_per_buffer
            + remote_gpu_result.bytes_per_buffer
            + cpu_result.bytes_per_buffer;
        let phase = if self.max_rows == 1 {
            GptOssPhase::Decode
        } else {
            GptOssPhase::Prefill
        };
        let (reported_raw_capacity_bytes, hard_cap_bytes) =
            relay_pinned_capacity_bytes(phase, self.max_rows)
                .expect("validated relay pool dimensions have a byte plan");
        assert_eq!(
            raw_capacity_bytes, reported_raw_capacity_bytes,
            "relay pool allocations drifted from the reviewed byte reporter"
        );
        RelayPinnedPoolStats {
            source_activation,
            route_descriptors,
            remote_gpu_input,
            remote_gpu_result,
            cpu_result,
            raw_capacity_bytes,
            hard_cap_bytes,
        }
    }

    /// Hold the second pool so a test can prove that failure after acquiring
    /// the source lease rolls that earlier acquisition back without allocating.
    #[cfg(feature = "heterogeneous-test-faults")]
    pub fn hold_route_descriptors_for_test(
        &self,
        generation: u64,
    ) -> Result<BoundedPinnedLease<u8>> {
        self.route_descriptors.try_acquire(generation)
    }
}

fn release_if_present<T: bytemuck::Pod + Send>(lease: Option<BoundedPinnedLease<T>>) -> Result<()> {
    if let Some(lease) = lease {
        lease.release_drained()?;
    }
    Ok(())
}

pub struct RelayPinnedReservation {
    generation: u64,
    pub source_activation: BoundedPinnedLease<u16>,
    pub route_descriptors: BoundedPinnedLease<u8>,
    pub remote_gpu_input: BoundedPinnedLease<u16>,
    pub remote_gpu_result: BoundedPinnedLease<u16>,
    pub cpu_result: BoundedPinnedLease<u16>,
}

impl RelayPinnedReservation {
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn release_drained(self) -> Result<()> {
        // Continue releasing if a defensive invariant fails so no later lease
        // is accidentally quarantined merely because an earlier return failed.
        let mut first_error = None;
        for result in [
            self.cpu_result.release_drained(),
            self.remote_gpu_result.release_drained(),
            self.remote_gpu_input.release_drained(),
            self.route_descriptors.release_drained(),
            self.source_activation.release_drained(),
        ] {
            if first_error.is_none() {
                first_error = result.err();
            }
        }
        if let Some(error) = first_error {
            Err(error)
        } else {
            Ok(())
        }
    }
}

/// Copy canonical source rows into the stable remote-owner route slots. The
/// source arena remains full row-major; no host compaction changes row identity.
pub fn pack_remote_inputs(
    plan: &PackedDispatchPlan,
    source_activation: &BoundedPinnedLease<u16>,
    remote_gpu_input: &mut BoundedPinnedLease<u16>,
) -> Result<()> {
    let source_required = plan.rows as usize * GPT_OSS_HIDDEN_SIZE;
    if source_activation.as_slice().len() < source_required {
        return Err(LLMError::MemoryError(
            "relay source activation lease is undersized".into(),
        ));
    }
    remote_gpu_input.as_mut_slice().fill(0);
    for owner in &plan.remote_gpu {
        for route in &owner.routes {
            let source_row = route.relay_activation_slot as usize;
            if source_row != route.descriptor.route.source_row as usize {
                return Err(LLMError::ModelError(
                    "relay route lost canonical source-row identity".into(),
                ));
            }
            let source_start = source_row * GPT_OSS_HIDDEN_SIZE;
            let destination_start = route.owner_route_slot as usize * GPT_OSS_HIDDEN_SIZE;
            let source =
                &source_activation.as_slice()[source_start..source_start + GPT_OSS_HIDDEN_SIZE];
            remote_gpu_input.as_mut_slice()
                [destination_start..destination_start + GPT_OSS_HIDDEN_SIZE]
                .copy_from_slice(source);
        }
    }
    Ok(())
}

#[cfg(feature = "heterogeneous-test-faults")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultRelayInjectedFault {
    AfterFirstResultEnqueue,
    CpuAuthorityAfterFirstEnqueue,
    AfterFirstResultEnqueueAndFallbackDrainFailure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResultRelayExecution {
    pub cpu_h2d_bytes: usize,
    pub remote_gpu_h2d_bytes: usize,
    pub evidence_d2h_bytes: usize,
    pub arena_generation: u64,
}

pub struct CompletedResultRelay {
    pub execution: ResultRelayExecution,
    pub reservation: RelayPinnedReservation,
}

pub struct CompletedCanonicalArenaEvidence {
    pub bytes: usize,
    pub arena_generation: u64,
    pub reservation: RelayPinnedReservation,
}

pub struct ResultRelayFailure {
    pub error: LLMError,
    /// Present only when the stream was proven drained and the caller may
    /// safely retry/release the reservation.
    pub reservation: Option<RelayPinnedReservation>,
}

impl std::fmt::Debug for CompletedResultRelay {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompletedResultRelay")
            .field("execution", &self.execution)
            .field("reservation_generation", &self.reservation.generation())
            .finish()
    }
}

impl std::fmt::Debug for CompletedCanonicalArenaEvidence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompletedCanonicalArenaEvidence")
            .field("bytes", &self.bytes)
            .field("arena_generation", &self.arena_generation)
            .field("reservation_generation", &self.reservation.generation())
            .finish()
    }
}

impl std::fmt::Debug for ResultRelayFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResultRelayFailure")
            .field("error", &self.error)
            .field("reservation_recoverable", &self.reservation.is_some())
            .finish()
    }
}

pub struct CompletedLocalResultRelay {
    pub bytes: usize,
    pub arena_generation: u64,
    pub result_slot: CudaSelectedExpertResultSlot,
}

pub struct LocalResultRelayFailure {
    pub error: LLMError,
    pub result_slot: Option<CudaSelectedExpertResultSlot>,
}

impl std::fmt::Debug for CompletedLocalResultRelay {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompletedLocalResultRelay")
            .field("bytes", &self.bytes)
            .field("arena_generation", &self.arena_generation)
            .field(
                "result_slot_generation",
                &self.result_slot.transaction_generation(),
            )
            .finish()
    }
}

impl std::fmt::Debug for LocalResultRelayFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalResultRelayFailure")
            .field("error", &self.error)
            .field("result_slot_recoverable", &self.result_slot.is_some())
            .finish()
    }
}

/// Detached H4 result uploader. It places CPU and GPU1 outputs into canonical
/// GPU0 route slots but deliberately performs no weighting or reduction.
pub struct CudaResultRelay {
    stable_device: gpt_oss_gpu::device::StableCudaDeviceId,
    stream: ManuallyDrop<std::sync::Arc<CudaStream>>,
    contribution_arena: ManuallyDrop<CudaSlice<u16>>,
    max_routes: usize,
    active_bound_generation: Option<u64>,
    last_bound_generation: u64,
    expected_decode_contracts: [Option<CanonicalRouteContract>; GPT_OSS_TOP_K],
    arena_generation: u64,
    remote_upload_complete: bool,
    populated_slots: Vec<bool>,
    poisoned: bool,
    quarantined_reservation: Option<RelayPinnedReservation>,
    quarantined_local_slots: Vec<CudaSelectedExpertResultSlot>,
    quarantined_oracle_outputs: Option<Vec<Vec<u16>>>,
    #[cfg(feature = "heterogeneous-test-faults")]
    injected_fault: Option<ResultRelayInjectedFault>,
    #[cfg(feature = "heterogeneous-test-faults")]
    last_fault_drained: bool,
}

impl CudaResultRelay {
    pub fn new(router: &CudaExactRouter, max_rows: usize) -> Result<Self> {
        if max_rows == 0 || max_rows > H4_PREFILL_MAX_ROWS {
            return Err(LLMError::GpuError(format!(
                "result relay rows {max_rows} outside 1..={H4_PREFILL_MAX_ROWS}"
            )));
        }
        let stream = std::sync::Arc::clone(router.relay_stream());
        let max_routes = max_rows * GPT_OSS_TOP_K;
        let contribution_arena = stream
            .alloc_zeros::<u16>(max_routes * GPT_OSS_HIDDEN_SIZE)
            .map_err(cuda_error("result relay arena allocation"))?;
        stream
            .synchronize()
            .map_err(cuda_error("result relay construction drain"))?;
        Ok(Self {
            stable_device: router.stable_device().clone(),
            stream: ManuallyDrop::new(stream),
            contribution_arena: ManuallyDrop::new(contribution_arena),
            max_routes,
            active_bound_generation: None,
            last_bound_generation: 0,
            expected_decode_contracts: [None; GPT_OSS_TOP_K],
            arena_generation: 0,
            remote_upload_complete: false,
            populated_slots: vec![false; max_routes],
            poisoned: false,
            quarantined_reservation: None,
            quarantined_local_slots: Vec::with_capacity(GPT_OSS_TOP_K),
            quarantined_oracle_outputs: None,
            #[cfg(feature = "heterogeneous-test-faults")]
            injected_fault: None,
            #[cfg(feature = "heterogeneous-test-faults")]
            last_fault_drained: false,
        })
    }

    pub fn owned_device_bytes(&self) -> Result<usize> {
        result_relay_owned_device_bytes(self.max_routes / GPT_OSS_TOP_K)
    }

    pub(crate) fn stable_device(&self) -> &gpt_oss_gpu::device::StableCudaDeviceId {
        &self.stable_device
    }

    pub(crate) fn stream(&self) -> &std::sync::Arc<CudaStream> {
        &self.stream
    }

    pub(crate) const fn max_routes(&self) -> usize {
        self.max_routes
    }

    pub(crate) const fn arena_generation(&self) -> u64 {
        self.arena_generation
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub const fn published_arena_generation_for_test(&self) -> u64 {
        self.arena_generation
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub const fn has_active_generation_for_test(&self) -> bool {
        self.active_bound_generation.is_some()
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub const fn device_state_quarantined_for_test(&self) -> bool {
        self.poisoned
    }

    /// A reducer sharing this stream/context observed an unproven drain. The
    /// canonical arena may still be referenced, so make the relay permanently
    /// unusable and retain its entire CUDA state in Drop.
    pub(crate) fn quarantine_unproven_device_work(&mut self) {
        self.poisoned = true;
    }

    pub(crate) const fn drain_is_unproven(&self) -> bool {
        self.poisoned
    }

    pub fn memory_info(&self) -> Result<(usize, usize)> {
        self.stream
            .context()
            .bind_to_thread()
            .map_err(cuda_error("result relay memory-info bind"))?;
        cudarc::driver::result::mem_get_info().map_err(cuda_error("result relay memory-info query"))
    }

    /// Prove that the shared relay stream has no outstanding transaction
    /// references. A failed synchronization poisons the relay so Drop retains
    /// its full CUDA state.
    pub fn prove_transaction_drain(&mut self) -> Result<()> {
        if self.poisoned {
            return Err(LLMError::GpuError(
                "poisoned result relay cannot prove transaction drain".into(),
            ));
        }
        if let Err(error) = self.stream.synchronize() {
            self.poisoned = true;
            return Err(cuda_error("result relay transaction drain")(error));
        }
        Ok(())
    }

    pub(crate) fn decode_contribution_arena(&self) -> &CudaSlice<u16> {
        debug_assert!(self.max_routes >= GPT_OSS_TOP_K);
        &self.contribution_arena
    }

    pub(crate) const fn active_decode_generation(&self) -> Option<u64> {
        self.active_bound_generation
    }

    /// Bind the next decode arena generation to exactly four canonical route
    /// identities before any expert dispatch. The stored contract is fixed
    /// size and every later upload/reduction must match it.
    pub fn bind_decode_generation(
        &mut self,
        transaction_generation: u64,
        plan: &PackedDispatchPlan,
    ) -> Result<()> {
        if self.poisoned
            || transaction_generation == 0
            || transaction_generation <= self.last_bound_generation.max(self.arena_generation)
            || self.active_bound_generation.is_some()
            || plan.phase != GptOssPhase::Decode
            || plan.rows != 1
        {
            return Err(LLMError::GpuError(
                "canonical relay generation cannot be rebound or is not decode M=1".into(),
            ));
        }
        plan.validate_round_trip()?;
        let contracts = contracts_from_plan(plan)?;
        self.active_bound_generation = Some(transaction_generation);
        self.last_bound_generation = transaction_generation;
        self.expected_decode_contracts = contracts.map(Some);
        self.remote_upload_complete = false;
        self.populated_slots.fill(false);
        Ok(())
    }

    /// Close a discarded bound generation only after the caller proves every
    /// CPU/GPU owner terminal. This method additionally drains the shared GPU0
    /// relay stream before route contracts or arena slots can be reused.
    pub fn abandon_decode_generation(
        &mut self,
        transaction_generation: u64,
        all_owners_proven_drained: bool,
    ) -> Result<()> {
        if self.active_bound_generation != Some(transaction_generation) {
            return Err(LLMError::GpuError(
                "cannot abandon a non-active canonical relay generation".into(),
            ));
        }
        if !all_owners_proven_drained {
            return Err(LLMError::GpuError(
                "canonical relay generation cannot be abandoned before every owner drains".into(),
            ));
        }
        if let Err(error) = self.stream.synchronize() {
            self.poisoned = true;
            return Err(cuda_error("canonical relay abandon drain")(error));
        }
        self.close_active_generation(transaction_generation)
    }

    pub(crate) fn finish_reduction_generation(
        &mut self,
        transaction_generation: u64,
    ) -> Result<()> {
        if self.arena_generation != transaction_generation
            || !self.remote_upload_complete
            || self.populated_slots[..GPT_OSS_TOP_K]
                .iter()
                .any(|populated| !populated)
        {
            return Err(LLMError::GpuError(
                "successful reduction cannot close an incomplete canonical arena".into(),
            ));
        }
        self.close_active_generation(transaction_generation)
    }

    fn close_active_generation(&mut self, transaction_generation: u64) -> Result<()> {
        if self.active_bound_generation != Some(transaction_generation) {
            return Err(LLMError::GpuError(
                "canonical relay generation close does not match the active ticket".into(),
            ));
        }
        self.active_bound_generation = None;
        self.expected_decode_contracts = [None; GPT_OSS_TOP_K];
        self.remote_upload_complete = false;
        self.populated_slots.fill(false);
        Ok(())
    }

    pub(crate) fn validate_reduction_contract(
        &self,
        transaction_generation: u64,
        contracts: &[CanonicalRouteContract; GPT_OSS_TOP_K],
    ) -> Result<()> {
        if self.active_bound_generation != Some(transaction_generation)
            || self.arena_generation != transaction_generation
        {
            return Err(LLMError::GpuError(
                "rank reducer generation does not match the bound canonical arena".into(),
            ));
        }
        for (slot, contract) in contracts.iter().enumerate() {
            if self.expected_decode_contracts[slot] != Some(*contract) {
                return Err(LLMError::GpuError(format!(
                    "rank reducer route contract mismatch at slot {slot}"
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn validate_complete_decode_results(
        &self,
        results: &[ExpertResultDescriptor],
    ) -> Result<()> {
        if self.poisoned || !self.remote_upload_complete || self.arena_generation == 0 {
            return Err(LLMError::GpuError(
                "canonical result arena is poisoned or lacks a completed clear/upload generation"
                    .into(),
            ));
        }
        if results.len() != GPT_OSS_TOP_K {
            return Err(LLMError::GpuError(
                "canonical result validation requires exactly four descriptors".into(),
            ));
        }
        let mut seen = [false; GPT_OSS_TOP_K];
        for result in results {
            let slot = result.result_slot as usize;
            let expected = self.expected_decode_contracts.get(slot).copied().flatten();
            if slot >= GPT_OSS_TOP_K
                || seen[slot]
                || !self.populated_slots.get(slot).copied().unwrap_or(false)
                || expected.is_none_or(|contract| contract.validate_result(result).is_err())
            {
                return Err(LLMError::GpuError(format!(
                    "canonical result slot {} is missing, duplicated, or identity-mismatched in generation {}",
                    result.result_slot, self.arena_generation
                )));
            }
            seen[slot] = true;
        }
        if seen.iter().any(|present| !present) {
            return Err(LLMError::GpuError(
                "canonical result descriptors are missing a route slot".into(),
            ));
        }
        Ok(())
    }

    /// H6a oracle-only control: place four already-computed CPU-authority
    /// contributions into the canonical GPU0 arena. Production dispatch must
    /// use the bound CPU/GPU1 upload plus local D2D APIs instead.
    pub fn upload_cpu_authority_control(
        &mut self,
        transaction_generation: u64,
        descriptors: &[ExpertResultDescriptor],
        outputs_bf16_bits: Vec<Vec<u16>>,
    ) -> Result<usize> {
        if self.poisoned
            || self.active_bound_generation != Some(transaction_generation)
            || transaction_generation <= self.arena_generation
            || descriptors.len() != GPT_OSS_TOP_K
            || outputs_bf16_bits.len() != GPT_OSS_TOP_K
        {
            return Err(LLMError::GpuError(
                "CPU-authority control does not match the active canonical generation".into(),
            ));
        }
        let mut seen = [false; GPT_OSS_TOP_K];
        for (descriptor, output) in descriptors.iter().zip(&outputs_bf16_bits) {
            let slot = descriptor.result_slot as usize;
            if slot >= GPT_OSS_TOP_K
                || seen[slot]
                || output.len() != GPT_OSS_HIDDEN_SIZE
                || self.expected_decode_contracts[slot]
                    .is_none_or(|contract| contract.validate_result(descriptor).is_err())
            {
                return Err(LLMError::GpuError(
                    "CPU-authority control contribution identity/shape mismatch".into(),
                ));
            }
            seen[slot] = true;
        }
        if seen.iter().any(|present| !present) {
            return Err(LLMError::GpuError(
                "CPU-authority control is missing a canonical route".into(),
            ));
        }
        #[cfg(feature = "heterogeneous-test-faults")]
        let injected_fault = self.injected_fault.take();
        let submitted = (|| -> Result<()> {
            self.stream
                .memset_zeros(&mut *self.contribution_arena)
                .map_err(cuda_error("CPU-authority arena clear"))?;
            for (_index, (descriptor, output)) in
                descriptors.iter().zip(&outputs_bf16_bits).enumerate()
            {
                let start = descriptor.result_slot as usize * GPT_OSS_HIDDEN_SIZE;
                self.stream
                    .memcpy_htod(
                        output,
                        &mut self
                            .contribution_arena
                            .slice_mut(start..start + GPT_OSS_HIDDEN_SIZE),
                    )
                    .map_err(cuda_error("CPU-authority contribution H2D"))?;
                #[cfg(feature = "heterogeneous-test-faults")]
                if injected_fault == Some(ResultRelayInjectedFault::CpuAuthorityAfterFirstEnqueue)
                    && _index == 0
                {
                    return Err(LLMError::GpuError(
                        "injected CPU-authority post-enqueue failure".into(),
                    ));
                }
            }
            self.stream
                .synchronize()
                .map_err(cuda_error("CPU-authority contribution drain"))
        })();
        if let Err(primary) = submitted {
            return match self.stream.synchronize() {
                Ok(()) => {
                    #[cfg(feature = "heterogeneous-test-faults")]
                    if injected_fault
                        == Some(ResultRelayInjectedFault::CpuAuthorityAfterFirstEnqueue)
                    {
                        self.last_fault_drained = true;
                    }
                    Err(primary)
                }
                Err(drain) => {
                    self.poisoned = true;
                    // At least one H2D may still reference these pageable host
                    // vectors. Logical poisoning is insufficient: retain the
                    // owned storage until process teardown, where Drop leaks it
                    // if CUDA still cannot prove the stream terminal.
                    self.quarantined_oracle_outputs = Some(outputs_bf16_bits);
                    Err(LLMError::GpuError(format!(
                        "CPU-authority upload failed ({primary}); mandatory drain failed ({drain}); relay poisoned and host authority storage quarantined"
                    )))
                }
            };
        }
        self.arena_generation = transaction_generation;
        self.remote_upload_complete = true;
        self.populated_slots[..GPT_OSS_TOP_K].fill(true);
        Ok(GPT_OSS_TOP_K * GPT_OSS_HIDDEN_SIZE * size_of::<u16>())
    }

    /// Copy one already-drained GPU0-local selected-expert result into its H4
    /// canonical slot without a host round trip. The relay stream owns the
    /// terminal event, so neither slot nor arena can be reused early.
    pub fn upload_local_device_result(
        &mut self,
        transaction_generation: u64,
        descriptor: &ExpertResultDescriptor,
        result_slot: CudaSelectedExpertResultSlot,
    ) -> std::result::Result<CompletedLocalResultRelay, LocalResultRelayFailure> {
        self.upload_local_device_result_inner(transaction_generation, descriptor, result_slot, None)
    }

    pub fn upload_local_device_result_with_timeline(
        &mut self,
        transaction_generation: u64,
        descriptor: &ExpertResultDescriptor,
        result_slot: CudaSelectedExpertResultSlot,
        timeline: &CorrelatedTimeline,
    ) -> std::result::Result<CompletedLocalResultRelay, LocalResultRelayFailure> {
        self.upload_local_device_result_inner(
            transaction_generation,
            descriptor,
            result_slot,
            Some(timeline),
        )
    }

    fn upload_local_device_result_inner(
        &mut self,
        transaction_generation: u64,
        descriptor: &ExpertResultDescriptor,
        result_slot: CudaSelectedExpertResultSlot,
        timeline: Option<&CorrelatedTimeline>,
    ) -> std::result::Result<CompletedLocalResultRelay, LocalResultRelayFailure> {
        let expected = usize::try_from(descriptor.result_slot)
            .ok()
            .filter(|slot| *slot < GPT_OSS_TOP_K)
            .and_then(|slot| self.expected_decode_contracts[slot]);
        if self.poisoned
            || !self.remote_upload_complete
            || self.active_bound_generation != Some(transaction_generation)
            || transaction_generation != self.arena_generation
        {
            return Err(LocalResultRelayFailure {
                error: LLMError::GpuError(
                    "local result D2D requires a completed arena clear and CPU/GPU1 upload".into(),
                ),
                result_slot: Some(result_slot),
            });
        }
        if result_slot.device() != &self.stable_device
            || result_slot.transaction_generation() != transaction_generation
            || !matches!(descriptor.owner, ExpertOwner::LayerOwnerGpu { .. })
            || descriptor.result_slot as usize >= self.max_routes
            || self.populated_slots[descriptor.result_slot as usize]
            || expected.is_none()
            || result_slot.route_contract() != expected
            || expected.is_some_and(|contract| contract.validate_result(descriptor).is_err())
            || !matches!(
                expected.map(|contract| contract.owner),
                Some(CanonicalExpertOwner::LayerOwnerGpu { .. })
            )
        {
            return Err(LocalResultRelayFailure {
                error: LLMError::GpuError(
                    "local result relay device/owner/slot identity mismatch or duplicate slot"
                        .into(),
                ),
                result_slot: Some(result_slot),
            });
        }
        let start = descriptor.result_slot as usize * GPT_OSS_HIDDEN_SIZE;
        let submitted = (|| -> Result<cudarc::driver::CudaEvent> {
            let actor = match descriptor.route_rank {
                0 => "gpu0_local_result_rank0",
                1 => "gpu0_local_result_rank1",
                2 => "gpu0_local_result_rank2",
                _ => "gpu0_local_result_rank3",
            };
            if let Some(timeline) = timeline {
                timeline.enqueue_cuda_marker(&self.stream, actor, "result_d2d_begin")?;
            }
            self.stream
                .memcpy_dtod(
                    result_slot.buffer(),
                    &mut self
                        .contribution_arena
                        .slice_mut(start..start + GPT_OSS_HIDDEN_SIZE),
                )
                .map_err(cuda_error("local result relay D2D"))?;
            if let Some(timeline) = timeline {
                timeline.enqueue_cuda_marker(&self.stream, actor, "result_d2d_end")?;
            }
            self.stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("local result relay terminal event"))
        })();
        let terminal = match submitted {
            Ok(terminal) => terminal,
            Err(primary) => {
                return match self.stream.synchronize() {
                    Ok(()) => Err(LocalResultRelayFailure {
                        error: primary,
                        result_slot: Some(result_slot),
                    }),
                    Err(drain) => {
                        self.poisoned = true;
                        self.quarantined_local_slots.push(result_slot);
                        Err(LocalResultRelayFailure {
                            error: LLMError::GpuError(format!(
                                "local result relay failed ({primary}); mandatory drain failed ({drain}); stream poisoned and result slot quarantined"
                            )),
                            result_slot: None,
                        })
                    }
                }
            }
        };
        if let Err(error) = terminal.synchronize() {
            let primary = cuda_error("local result relay terminal drain")(error);
            return match self.stream.synchronize() {
                Ok(()) => Err(LocalResultRelayFailure {
                    error: primary,
                    result_slot: Some(result_slot),
                }),
                Err(drain) => {
                    self.poisoned = true;
                    self.quarantined_local_slots.push(result_slot);
                    Err(LocalResultRelayFailure {
                        error: LLMError::GpuError(format!(
                            "local result relay terminal failed ({primary}); mandatory drain failed ({drain}); stream poisoned and result slot quarantined"
                        )),
                        result_slot: None,
                    })
                }
            };
        }
        self.populated_slots[descriptor.result_slot as usize] = true;
        Ok(CompletedLocalResultRelay {
            bytes: GPT_OSS_HIDDEN_SIZE * size_of::<u16>(),
            arena_generation: self.arena_generation,
            result_slot,
        })
    }

    /// Download the completed four-rank contribution arena into the already
    /// reserved GPU1-input lease for bounded H6 evidence. This occurs only
    /// after CPU/GPU1 H2D and every GPU0 local D2D are terminal. Consuming the
    /// reservation lets an uncertain fallback drain quarantine the exact host
    /// storage that an asynchronous D2H may still reference.
    pub fn download_complete_decode_evidence(
        &mut self,
        transaction_generation: u64,
        descriptors: &[ExpertResultDescriptor],
        mut reservation: RelayPinnedReservation,
        timeline: Option<&CorrelatedTimeline>,
    ) -> std::result::Result<CompletedCanonicalArenaEvidence, ResultRelayFailure> {
        if reservation.generation() != transaction_generation
            || self.active_bound_generation != Some(transaction_generation)
            || self.arena_generation != transaction_generation
        {
            return Err(ResultRelayFailure {
                error: LLMError::GpuError(
                    "canonical evidence reservation/generation mismatch".into(),
                ),
                reservation: Some(reservation),
            });
        }
        if let Err(error) = self.validate_complete_decode_results(descriptors) {
            return Err(ResultRelayFailure {
                error,
                reservation: Some(reservation),
            });
        }
        let values = GPT_OSS_TOP_K * GPT_OSS_HIDDEN_SIZE;
        if reservation.remote_gpu_input.as_slice().len() < values {
            return Err(ResultRelayFailure {
                error: LLMError::MemoryError(
                    "canonical evidence pinned lease is undersized".into(),
                ),
                reservation: Some(reservation),
            });
        }
        let submitted = (|| -> Result<cudarc::driver::CudaEvent> {
            if let Some(timeline) = timeline {
                timeline.enqueue_cuda_marker(
                    &self.stream,
                    "gpu0_relay",
                    "canonical_evidence_d2h_begin",
                )?;
            }
            self.stream
                .memcpy_dtoh(
                    &self.contribution_arena.slice(..values),
                    &mut reservation.remote_gpu_input.as_mut_slice()[..values],
                )
                .map_err(cuda_error("canonical evidence D2H"))?;
            if let Some(timeline) = timeline {
                timeline.enqueue_cuda_marker(
                    &self.stream,
                    "gpu0_relay",
                    "canonical_evidence_d2h_end",
                )?;
            }
            self.stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("canonical evidence terminal event"))
        })();
        let terminal = match submitted {
            Ok(terminal) => terminal,
            Err(primary) => {
                return match self.stream.synchronize() {
                    Ok(()) => Err(ResultRelayFailure {
                        error: primary,
                        reservation: Some(reservation),
                    }),
                    Err(drain) => {
                        self.poisoned = true;
                        self.quarantined_reservation = Some(reservation);
                        Err(ResultRelayFailure {
                            error: LLMError::GpuError(format!(
                                "canonical evidence submit failed ({primary}); mandatory drain failed ({drain}); relay poisoned and pinned reservation quarantined"
                            )),
                            reservation: None,
                        })
                    }
                };
            }
        };
        if let Err(error) = terminal.synchronize() {
            let primary = cuda_error("canonical evidence terminal drain")(error);
            return match self.stream.synchronize() {
                Ok(()) => Err(ResultRelayFailure {
                    error: primary,
                    reservation: Some(reservation),
                }),
                Err(drain) => {
                    self.poisoned = true;
                    self.quarantined_reservation = Some(reservation);
                    Err(ResultRelayFailure {
                        error: LLMError::GpuError(format!(
                            "canonical evidence terminal failed ({primary}); mandatory drain failed ({drain}); relay poisoned and pinned reservation quarantined"
                        )),
                        reservation: None,
                    })
                }
            };
        }
        Ok(CompletedCanonicalArenaEvidence {
            bytes: values * size_of::<u16>(),
            arena_generation: self.arena_generation,
            reservation,
        })
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub fn inject_next_failure(&mut self, fault: ResultRelayInjectedFault) -> Result<()> {
        if self.injected_fault.is_some() {
            return Err(LLMError::GpuError(
                "result relay already has an armed test fault".into(),
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

    /// Upload actual CPU/GPU1 result rows and download the canonical arena into
    /// the no-longer-needed remote-input lease for bounded evidence.
    pub fn upload_results(
        &mut self,
        plan: &PackedDispatchPlan,
        reservation: RelayPinnedReservation,
        timeline: Option<&CorrelatedTimeline>,
    ) -> std::result::Result<CompletedResultRelay, ResultRelayFailure> {
        self.upload_results_inner(plan, reservation, None, timeline)
    }

    /// Production H5 upload. Every CPU/GPU1 completion is a compact identity
    /// emitted by the exact worker that filled the pinned result row.
    pub fn upload_results_bound(
        &mut self,
        plan: &PackedDispatchPlan,
        reservation: RelayPinnedReservation,
        nonlocal_completions: &[CanonicalRouteContract],
        timeline: Option<&CorrelatedTimeline>,
    ) -> std::result::Result<CompletedResultRelay, ResultRelayFailure> {
        self.upload_results_inner(plan, reservation, Some(nonlocal_completions), timeline)
    }

    fn upload_results_inner(
        &mut self,
        plan: &PackedDispatchPlan,
        mut reservation: RelayPinnedReservation,
        nonlocal_completions: Option<&[CanonicalRouteContract]>,
        timeline: Option<&CorrelatedTimeline>,
    ) -> std::result::Result<CompletedResultRelay, ResultRelayFailure> {
        if self.poisoned {
            return Err(ResultRelayFailure {
                error: LLMError::GpuError("result relay stream is poisoned".into()),
                reservation: Some(reservation),
            });
        }
        let route_count = plan.rows as usize * GPT_OSS_TOP_K;
        if route_count > self.max_routes
            || reservation.remote_gpu_input.as_slice().len() < route_count * GPT_OSS_HIDDEN_SIZE
        {
            return Err(ResultRelayFailure {
                error: LLMError::GpuError(
                    "result relay reservation is smaller than canonical arena".into(),
                ),
                reservation: Some(reservation),
            });
        }
        let next_generation = reservation.generation();
        if let Some(bound_generation) = self.active_bound_generation {
            if next_generation != bound_generation
                || plan_contracts_match(plan, &self.expected_decode_contracts).is_err()
                || nonlocal_completions.is_none_or(|completions| {
                    validate_nonlocal_completions(&self.expected_decode_contracts, completions)
                        .is_err()
                })
            {
                return Err(ResultRelayFailure {
                    error: LLMError::GpuError(
                        "result relay plan/completion identity does not match the bound generation"
                            .into(),
                    ),
                    reservation: Some(reservation),
                });
            }
        } else if nonlocal_completions.is_some() {
            return Err(ResultRelayFailure {
                error: LLMError::GpuError(
                    "bound result upload requires bind_decode_generation before dispatch".into(),
                ),
                reservation: Some(reservation),
            });
        }
        if next_generation == 0 || next_generation <= self.arena_generation {
            return Err(ResultRelayFailure {
                error: LLMError::GpuError(format!(
                    "result relay reservation generation {next_generation} is not newer than arena generation {}",
                    self.arena_generation
                )),
                reservation: Some(reservation),
            });
        }
        self.remote_upload_complete = false;
        self.populated_slots.fill(false);
        #[cfg(feature = "heterogeneous-test-faults")]
        let injected_fault = self.injected_fault.take();
        let submitted = (|| -> Result<(usize, usize, cudarc::driver::CudaEvent)> {
            self.stream
                .memset_zeros(&mut *self.contribution_arena)
                .map_err(cuda_error("result relay arena clear"))?;
            if let Some(timeline) = timeline {
                timeline.enqueue_cuda_marker(&self.stream, "gpu0_relay", "result_h2d_begin")?;
            }
            let mut cpu_bytes = 0;
            let mut remote_bytes = 0;
            #[cfg(feature = "heterogeneous-test-faults")]
            let mut enqueued = 0;
            for (owners, source, bytes) in [
                (&plan.cpu, reservation.cpu_result.as_slice(), &mut cpu_bytes),
                (
                    &plan.remote_gpu,
                    reservation.remote_gpu_result.as_slice(),
                    &mut remote_bytes,
                ),
            ] {
                for owner in owners {
                    for route in &owner.routes {
                        let source_start = route.owner_route_slot as usize * GPT_OSS_HIDDEN_SIZE;
                        let destination_start =
                            route.descriptor.canonical_result_slot as usize * GPT_OSS_HIDDEN_SIZE;
                        self.stream
                            .memcpy_htod(
                                &source[source_start..source_start + GPT_OSS_HIDDEN_SIZE],
                                &mut self.contribution_arena.slice_mut(
                                    destination_start..destination_start + GPT_OSS_HIDDEN_SIZE,
                                ),
                            )
                            .map_err(cuda_error("result relay contribution H2D"))?;
                        *bytes += GPT_OSS_HIDDEN_SIZE * size_of::<u16>();
                        #[cfg(feature = "heterogeneous-test-faults")]
                        {
                            enqueued += 1;
                            if matches!(
                                injected_fault,
                                Some(ResultRelayInjectedFault::AfterFirstResultEnqueue)
                                    | Some(
                                        ResultRelayInjectedFault::AfterFirstResultEnqueueAndFallbackDrainFailure
                                    )
                            )
                                && enqueued == 1
                            {
                                return Err(LLMError::GpuError(
                                    "injected result relay post-enqueue failure".into(),
                                ));
                            }
                        }
                    }
                }
            }
            if let Some(timeline) = timeline {
                timeline.enqueue_cuda_marker(&self.stream, "gpu0_relay", "result_h2d_end")?;
            }
            self.stream
                .memcpy_dtoh(
                    &self
                        .contribution_arena
                        .slice(..route_count * GPT_OSS_HIDDEN_SIZE),
                    &mut reservation.remote_gpu_input.as_mut_slice()
                        [..route_count * GPT_OSS_HIDDEN_SIZE],
                )
                .map_err(cuda_error("result relay evidence D2H"))?;
            let terminal = self
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("result relay terminal event"))?;
            Ok((cpu_bytes, remote_bytes, terminal))
        })();
        let (cpu_h2d_bytes, remote_gpu_h2d_bytes, terminal) = match submitted {
            Ok(value) => value,
            Err(primary) => {
                #[cfg(feature = "heterogeneous-test-faults")]
                if injected_fault
                    == Some(
                        ResultRelayInjectedFault::AfterFirstResultEnqueueAndFallbackDrainFailure,
                    )
                {
                    self.poisoned = true;
                    self.quarantined_reservation = Some(reservation);
                    return Err(ResultRelayFailure {
                        error: LLMError::GpuError(format!(
                            "result relay submit failed ({primary}); injected mandatory fallback drain failure; all CUDA and pinned state quarantined"
                        )),
                        reservation: None,
                    });
                }
                let drained = self.stream.synchronize();
                #[cfg(feature = "heterogeneous-test-faults")]
                if injected_fault == Some(ResultRelayInjectedFault::AfterFirstResultEnqueue)
                    && drained.is_ok()
                {
                    self.last_fault_drained = true;
                }
                if let Err(drain) = drained {
                    self.poisoned = true;
                    self.quarantined_reservation = Some(reservation);
                    return Err(ResultRelayFailure {
                        error: LLMError::GpuError(format!(
                            "result relay submit failed ({primary}); mandatory drain failed ({drain}); stream poisoned and pinned reservation quarantined"
                        )),
                        reservation: None,
                    });
                }
                return Err(ResultRelayFailure {
                    error: primary,
                    reservation: Some(reservation),
                });
            }
        };
        if let Err(error) = terminal.synchronize() {
            let primary = cuda_error("result relay terminal drain")(error);
            return match self.stream.synchronize() {
                Ok(()) => Err(ResultRelayFailure {
                    error: primary,
                    reservation: Some(reservation),
                }),
                Err(drain) => {
                    self.poisoned = true;
                    self.quarantined_reservation = Some(reservation);
                    Err(ResultRelayFailure {
                        error: LLMError::GpuError(format!(
                            "result relay terminal failed ({primary}); mandatory drain failed ({drain}); stream poisoned and pinned reservation quarantined"
                        )),
                        reservation: None,
                    })
                }
            };
        }
        for owner in plan.cpu.iter().chain(&plan.remote_gpu) {
            for route in &owner.routes {
                self.populated_slots[route.descriptor.canonical_result_slot as usize] = true;
            }
        }
        self.arena_generation = next_generation;
        self.remote_upload_complete = true;
        Ok(CompletedResultRelay {
            execution: ResultRelayExecution {
                cpu_h2d_bytes,
                remote_gpu_h2d_bytes,
                evidence_d2h_bytes: route_count * GPT_OSS_HIDDEN_SIZE * size_of::<u16>(),
                arena_generation: self.arena_generation,
            },
            reservation,
        })
    }
}

impl Drop for CudaResultRelay {
    fn drop(&mut self) {
        if self.poisoned {
            if let Some(reservation) = self.quarantined_reservation.take() {
                // A failed CUDA drain cannot prove that DMA released these
                // pages. Retain them for process lifetime rather than free or
                // return them to a pool.
                std::mem::forget(reservation);
            }
            for slot in self.quarantined_local_slots.drain(..) {
                std::mem::forget(slot);
            }
            if let Some(outputs) = self.quarantined_oracle_outputs.take() {
                std::mem::forget(outputs);
            }
            // The failed stream may still name the arena. ManuallyDrop keeps
            // both allocations (and the stream/context retained by their
            // Arcs) alive for process lifetime.
            return;
        }
        // SAFETY: these fields are wrapped solely to permit fail-closed leak
        // on an unproven drain. The healthy path drops each exactly once, with
        // the arena before its owning stream.
        unsafe {
            ManuallyDrop::drop(&mut self.contribution_arena);
            ManuallyDrop::drop(&mut self.stream);
        }
    }
}

fn contracts_from_plan(
    plan: &PackedDispatchPlan,
) -> Result<[CanonicalRouteContract; GPT_OSS_TOP_K]> {
    let mut contracts = [None; GPT_OSS_TOP_K];
    for route in plan.all_routes() {
        let contract = CanonicalRouteContract::from_packed_route(&route.descriptor);
        let slot = contract.result_slot as usize;
        if slot >= GPT_OSS_TOP_K || contracts[slot].replace(contract).is_some() {
            return Err(LLMError::ModelError(
                "canonical decode plan has an out-of-range or duplicate result slot".into(),
            ));
        }
    }
    match contracts {
        [Some(rank0), Some(rank1), Some(rank2), Some(rank3)] => Ok([rank0, rank1, rank2, rank3]),
        _ => Err(LLMError::ModelError(
            "canonical decode plan is missing a result contract".into(),
        )),
    }
}

fn plan_contracts_match(
    plan: &PackedDispatchPlan,
    expected: &[Option<CanonicalRouteContract>; GPT_OSS_TOP_K],
) -> Result<()> {
    let actual = contracts_from_plan(plan)?;
    if actual
        .iter()
        .enumerate()
        .all(|(slot, contract)| expected[slot] == Some(*contract))
    {
        Ok(())
    } else {
        Err(LLMError::ModelError(
            "dispatch plan changed after canonical generation binding".into(),
        ))
    }
}

fn validate_nonlocal_completions(
    expected: &[Option<CanonicalRouteContract>; GPT_OSS_TOP_K],
    completions: &[CanonicalRouteContract],
) -> Result<()> {
    let expected_count = expected
        .iter()
        .flatten()
        .filter(|contract| {
            matches!(
                contract.owner,
                CanonicalExpertOwner::Cpu { .. } | CanonicalExpertOwner::RemoteGpu { .. }
            )
        })
        .count();
    if completions.len() != expected_count {
        return Err(LLMError::ModelError(format!(
            "nonlocal result completion count {} != expected {expected_count}",
            completions.len()
        )));
    }
    let mut seen = [false; GPT_OSS_TOP_K];
    for completion in completions {
        let slot = completion.result_slot as usize;
        if slot >= GPT_OSS_TOP_K
            || seen[slot]
            || expected[slot] != Some(*completion)
            || !matches!(
                completion.owner,
                CanonicalExpertOwner::Cpu { .. } | CanonicalExpertOwner::RemoteGpu { .. }
            )
        {
            return Err(LLMError::ModelError(format!(
                "nonlocal result completion identity is missing, duplicated, or mismatched at slot {}",
                completion.result_slot
            )));
        }
        seen[slot] = true;
    }
    for contract in expected.iter().flatten().filter(|contract| {
        matches!(
            contract.owner,
            CanonicalExpertOwner::Cpu { .. } | CanonicalExpertOwner::RemoteGpu { .. }
        )
    }) {
        if !seen[contract.result_slot as usize] {
            return Err(LLMError::ModelError(format!(
                "nonlocal result completion is missing slot {}",
                contract.result_slot
            )));
        }
    }
    Ok(())
}

pub fn fixed_relay_byte_plan(max_rows: usize) -> Result<RelayBytePlan> {
    if max_rows == 0 || max_rows > H4_PREFILL_MAX_ROWS {
        return Err(LLMError::MemoryError(
            "fixed relay byte plan rows outside 1..=64".into(),
        ));
    }
    let phase = if max_rows == 1 {
        GptOssPhase::Decode
    } else {
        GptOssPhase::Prefill
    };
    let row_bytes = GPT_OSS_HIDDEN_SIZE
        .checked_mul(size_of::<u16>())
        .ok_or_else(|| LLMError::ModelError("fixed relay row bytes overflow".into()))?;
    let source_activation_capacity = max_rows
        .checked_mul(row_bytes)
        .ok_or_else(|| LLMError::ModelError("fixed relay source bytes overflow".into()))?;
    let route_capacity = max_rows
        .checked_mul(GPT_OSS_TOP_K)
        .ok_or_else(|| LLMError::ModelError("fixed relay route count overflow".into()))?;
    let route_descriptor_capacity = route_capacity
        .checked_mul(H4_ROUTE_DESCRIPTOR_MAX_BYTES)
        .ok_or_else(|| LLMError::ModelError("fixed relay descriptor bytes overflow".into()))?;
    let route_arena = route_capacity
        .checked_mul(row_bytes)
        .ok_or_else(|| LLMError::ModelError("fixed relay route arena bytes overflow".into()))?;
    let (raw_pinned_bytes, hard_cap_bytes) = relay_pinned_capacity_bytes(phase, max_rows)?;
    Ok(RelayBytePlan {
        source_activation_d2h: source_activation_capacity,
        route_descriptor_d2h: route_descriptor_capacity,
        remote_gpu_h2d: 0,
        remote_gpu_d2h: 0,
        cpu_result_bytes: 0,
        source_activation_capacity,
        route_descriptor_capacity,
        remote_gpu_input_capacity: route_arena,
        remote_gpu_result_capacity: route_arena,
        cpu_result_capacity: route_arena,
        raw_pinned_bytes,
        hard_cap_bytes,
    })
}

fn cuda_error(stage: &'static str) -> impl FnOnce(cudarc::driver::DriverError) -> LLMError {
    move |error| LLMError::GpuError(format!("{stage}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_five_buffer_plan_matches_phase_2_arithmetic() {
        let decode = fixed_relay_byte_plan(1).unwrap();
        assert_eq!(decode.raw_pinned_bytes, 74_944);
        assert_eq!(decode.hard_cap_bytes, 128 * 1024);
        let prefill = fixed_relay_byte_plan(64).unwrap();
        assert_eq!(prefill.raw_pinned_bytes, 4_796_416);
        assert_eq!(prefill.hard_cap_bytes, 8 * 1024 * 1024);
    }
}
