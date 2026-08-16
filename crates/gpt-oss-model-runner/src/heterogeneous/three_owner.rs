//! Prepared decode-M=1 execution across the H3 resident CPU/GPU0/GPU1 owners.
//!
//! This is deliberately a routed-expert seam, not a model graph. GPU0 router
//! and layer semantics stay in `layer`; this module begins only after an exact
//! routed batch and immutable placement have produced a bounded dispatch plan.

use std::sync::Arc;

use gpt_oss_core::error::{LLMError, Result};
use gpt_oss_gpu::event::CorrelatedTimeline;

use crate::model_loader::owner_selective::OwnerSelectiveModel;

use super::contract::{
    CanonicalRouteContract, ExpertResultDescriptor, GptOssPhase, GPT_OSS_HIDDEN_SIZE, GPT_OSS_TOP_K,
};
use super::cpu_expert::CpuX8SelectedExpertWorker;
use super::cuda_expert::{
    CudaSelectedExpertResultSlot, CudaSelectedExpertWeights, OwnedSelectedExpertFailure,
    SelectedExpertFirstDivergenceTrace, SelectedExpertTraceStorage,
};
use super::layer::CudaLayerOwnerShell;
use super::packing::{PackedDispatchPlan, PackedDispatchRoute};
use super::placement::GptOssExpertKey;
use super::reduction::{
    CudaRankOrderedReducer, PreparedRankOrderedReduction, RankOrderedReductionExecution,
};
use super::relay::{
    pack_remote_inputs, CudaResultRelay, RelayPinnedReservation, ResultRelayExecution,
};
use super::router::CudaExactRouter;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreeOwnerTerminal {
    CpuExpert,
    LayerOwnerExpert,
    RemoteGpuExpert,
    LayerOwnerRelay,
    RankReduction,
}

#[cfg(feature = "heterogeneous-test-faults")]
pub struct OwnedDeviceInputFaultProbe {
    pub fault_drained: bool,
    pub retry_trace: SelectedExpertFirstDivergenceTrace,
    pub retry_input_d2d_bytes: usize,
    pub retry_input_host_bytes: usize,
}

#[cfg(feature = "heterogeneous-test-faults")]
pub struct OwnedDeviceInputUnprovenFaultProbe {
    pub drain_proven: bool,
    pub result_slot_returned: bool,
    pub input_retained: bool,
    pub weights_retained: bool,
    pub trace_retained: bool,
    pub slot_retained: bool,
    pub executor_marked_unproven: bool,
    pub model_quarantined: bool,
    pub shell_quarantined: bool,
}

/// Exercise the exact post-D2D-enqueue ownership boundary against one H3
/// resident local expert, then immediately reuse the proven-drained slot and
/// executor. This never uploads or duplicates the resident expert payload.
#[cfg(feature = "heterogeneous-test-faults")]
pub fn exercise_owned_device_input_fault_and_retry(
    model: &mut OwnerSelectiveModel,
    shell: &CudaLayerOwnerShell,
    layer: u16,
    generation: u64,
    route: &PackedDispatchRoute,
    timeline: &CorrelatedTimeline,
) -> Result<OwnedDeviceInputFaultProbe> {
    use super::cuda_expert::SelectedExpertInjectedFault;

    let resident = shell.resident_expert_input()?;
    let parts = model.execution_parts();
    let weights = require_gpu_weight(
        parts.layer_owner_experts,
        layer,
        route,
        "layer-owner fault probe",
    )?;
    let slot = parts
        .layer_owner_executor
        .allocate_result_slot_for_route(generation, &route.descriptor)?;
    parts
        .layer_owner_executor
        .inject_next_failure(SelectedExpertInjectedFault::SubmitAfterInputEnqueue)?;
    let prepared = parts.layer_owner_executor.prepare_owned_device(
        GptOssPhase::Decode,
        route.descriptor.clone(),
        Arc::clone(&weights),
        Arc::clone(&resident.slice),
        &resident.stable_device,
        slot,
        SelectedExpertTraceStorage::new(),
    );
    let failure = match prepared {
        Ok(prepared) => match prepared.submit_with_timeline(timeline, "gpu0_d2d_fault_probe") {
            Ok(_) => {
                return Err(LLMError::GpuError(
                    "owned D2D injected fault unexpectedly submitted".into(),
                ));
            }
            Err(failure) => failure,
        },
        Err(failure) => failure,
    };
    let (fault, slot, drain_proven, pinned_referenced, _) = failure.into_parts();
    if !drain_proven
        || pinned_referenced
        || !parts.layer_owner_executor.last_post_enqueue_fault_drained()
    {
        return Err(LLMError::GpuError(format!(
            "owned D2D fault did not prove its mandatory fallback drain: {fault}"
        )));
    }
    let slot = slot.ok_or_else(|| {
        LLMError::GpuError("proven owned D2D fault did not return its result slot".into())
    })?;
    let retry = match parts.layer_owner_executor.prepare_owned_device(
        GptOssPhase::Decode,
        route.descriptor.clone(),
        weights,
        resident.slice,
        &resident.stable_device,
        slot,
        SelectedExpertTraceStorage::new(),
    ) {
        Ok(prepared) => prepared
            .submit_with_timeline(timeline, "gpu0_d2d_fault_retry")
            .map_err(|failure| failure.into_parts().0)?,
        Err(failure) => return Err(failure.into_parts().0),
    }
    .drain_with_trace(None, timeline, "gpu0_d2d_fault_retry")
    .map_err(|failure| failure.into_parts().0)?;
    Ok(OwnedDeviceInputFaultProbe {
        fault_drained: true,
        retry_trace: retry.trace,
        retry_input_d2d_bytes: retry.input_d2d_bytes,
        retry_input_host_bytes: retry.input_h2d_bytes,
    })
}

/// Destructive fault probe for the unproven fallback-drain branch. The model
/// and shell become permanently unusable; the function intentionally exposes
/// no retry object.
#[cfg(feature = "heterogeneous-test-faults")]
pub fn exercise_owned_device_input_unproven_fault(
    model: &mut OwnerSelectiveModel,
    shell: &mut CudaLayerOwnerShell,
    layer: u16,
    generation: u64,
    route: &PackedDispatchRoute,
    timeline: &CorrelatedTimeline,
) -> Result<OwnedDeviceInputUnprovenFaultProbe> {
    use super::cuda_expert::SelectedExpertInjectedFault;

    let resident = shell.resident_expert_input()?;
    let (failure, executor_marked_unproven) = {
        let parts = model.execution_parts();
        let weights = require_gpu_weight(
            parts.layer_owner_experts,
            layer,
            route,
            "layer-owner unproven fault probe",
        )?;
        let slot = parts
            .layer_owner_executor
            .allocate_result_slot_for_route(generation, &route.descriptor)?;
        parts.layer_owner_executor.inject_next_failure(
            SelectedExpertInjectedFault::SubmitAfterInputEnqueueAndFallbackDrainFailure,
        )?;
        let prepared = parts.layer_owner_executor.prepare_owned_device(
            GptOssPhase::Decode,
            route.descriptor.clone(),
            weights,
            resident.slice,
            &resident.stable_device,
            slot,
            SelectedExpertTraceStorage::new(),
        );
        let failure = match prepared {
            Ok(prepared) => {
                match prepared.submit_with_timeline(timeline, "gpu0_d2d_unproven_fault") {
                    Ok(_) => {
                        return Err(LLMError::GpuError(
                            "owned D2D unproven fault unexpectedly submitted".into(),
                        ));
                    }
                    Err(failure) => failure,
                }
            }
            Err(failure) => failure,
        };
        let marked = parts.layer_owner_executor.owned_drain_unproven();
        (failure, marked)
    };
    let (_error, slot, drain_proven, pinned_referenced, retained) = failure.into_parts();
    if drain_proven
        || slot.is_some()
        || pinned_referenced
        || !retained.all_device_owned()
        || !executor_marked_unproven
    {
        return Err(LLMError::GpuError(
            "owned D2D unproven fault returned reusable state or missed executor poisoning".into(),
        ));
    }
    model.quarantine_execution();
    shell.quarantine_external_device_use();
    Ok(OwnedDeviceInputUnprovenFaultProbe {
        drain_proven,
        result_slot_returned: false,
        input_retained: retained.device_input,
        weights_retained: retained.weights,
        trace_retained: retained.trace,
        slot_retained: retained.result_slot,
        executor_marked_unproven,
        model_quarantined: model.execution_quarantined_for_test(),
        shell_quarantined: shell.is_poisoned_for_test(),
    })
}

pub struct PreparedThreeOwnerDecode {
    generation: u64,
    plan: PackedDispatchPlan,
    descriptors: Vec<ExpertResultDescriptor>,
    reservation: Option<RelayPinnedReservation>,
    reduction: Option<PreparedRankOrderedReduction>,
    local_slots: [Option<CudaSelectedExpertResultSlot>; 2],
    remote_slot: Option<CudaSelectedExpertResultSlot>,
    trace_storage: [Option<SelectedExpertTraceStorage>; GPT_OSS_TOP_K],
    completed_traces: [Option<SelectedExpertFirstDivergenceTrace>; GPT_OSS_TOP_K],
    cpu_worker: CpuX8SelectedExpertWorker,
    relay_active: bool,
    completion_contracts: [Option<CanonicalRouteContract>; GPT_OSS_TOP_K],
    local_kernel_elapsed_ms: [f32; 2],
    remote_kernel_elapsed_ms: f32,
    cpu_elapsed_ns: u64,
    relay_execution: Option<ResultRelayExecution>,
    reduced_execution: Option<RankOrderedReductionExecution>,
    local_activation_d2d_bytes: usize,
    remote_activation_h2d_bytes: usize,
    remote_result_d2h_bytes: usize,
    local_result_d2d_bytes: usize,
    canonical_evidence_d2h_bytes: usize,
    quarantine_required: bool,
    pinned_storage_may_be_referenced: bool,
    router_admitted: bool,
}

pub struct ThreeOwnerDecodeExecution {
    pub generation: u64,
    pub descriptors: Vec<ExpertResultDescriptor>,
    pub completion_contracts: [CanonicalRouteContract; GPT_OSS_TOP_K],
    pub expert_traces: [SelectedExpertFirstDivergenceTrace; GPT_OSS_TOP_K],
    pub local_kernel_elapsed_ms: [f32; 2],
    pub remote_kernel_elapsed_ms: f32,
    pub cpu_elapsed_ns: u64,
    pub cpu_scratch_bytes: usize,
    pub cpu_high_water_jobs: usize,
    pub local_activation_d2d_bytes: usize,
    pub local_activation_host_execution_bytes: usize,
    pub remote_activation_h2d_bytes: usize,
    pub remote_result_d2h_bytes: usize,
    pub local_result_d2d_bytes: usize,
    pub canonical_evidence_d2h_bytes: usize,
    pub relay_execution: ResultRelayExecution,
    pub reduction: RankOrderedReductionExecution,
    pub reservation: RelayPinnedReservation,
}

pub struct ThreeOwnerDecodeFailure {
    pub error: LLMError,
    pub drain_proven: bool,
}

/// Retain the durable model/shell CUDA owners after an outer transaction edge
/// cannot prove its fallback drain. This is deliberately one-way.
pub fn quarantine_three_owner_after_unproven_drain(
    model: &mut OwnerSelectiveModel,
    shell: &mut CudaLayerOwnerShell,
) {
    model.quarantine_execution();
    shell.quarantine_external_device_use();
}

impl ThreeOwnerDecodeFailure {
    pub fn into_parts(self) -> (LLMError, bool) {
        (self.error, self.drain_proven)
    }
}

impl PreparedThreeOwnerDecode {
    /// Reserve every route-bound GPU slot, CPU scratch, and evidence record
    /// before expert dispatch, then bind the canonical relay generation. The
    /// H3 model's existing resident weights are Arc-retained, never uploaded.
    pub fn prepare(
        model: &mut OwnerSelectiveModel,
        plan: PackedDispatchPlan,
        reservation: RelayPinnedReservation,
        reduction: PreparedRankOrderedReduction,
        relay: &mut CudaResultRelay,
    ) -> Result<Self> {
        let generation = reservation.generation();
        if generation == 0
            || reduction.transaction_generation() != generation
            || plan.phase != GptOssPhase::Decode
            || plan.rows != 1
            || plan.local_route_count() != 2
            || plan.cpu_route_count() != 1
            || plan.remote_gpu_route_count() != 1
            || plan.local_gpu.len() != 1
            || plan.cpu.len() != 1
            || plan.remote_gpu.len() != 1
        {
            return Err(release_predispatch_reservation(
                reservation,
                LLMError::ModelError(
                    "H6 three-owner preparation requires decode M=1 with a 2/1/1 owner split"
                        .into(),
                ),
            ));
        }
        if let Err(error) = plan.validate_round_trip() {
            return Err(release_predispatch_reservation(reservation, error));
        }
        let descriptors = reduction.expected_results().to_vec();
        if descriptors.len() != GPT_OSS_TOP_K {
            return Err(release_predispatch_reservation(
                reservation,
                LLMError::ModelError(
                    "H6 three-owner reduction does not contain four canonical descriptors".into(),
                ),
            ));
        }

        let mut local_slots = [None, None];
        let mut remote_slot = None;
        let allocated = (|| -> Result<()> {
            let parts = model.execution_parts();
            for route in &plan.local_gpu[0].routes {
                require_gpu_weight(parts.layer_owner_experts, plan.layer, route, "layer-owner")?;
            }
            let remote_route = &plan.remote_gpu[0].routes[0];
            require_gpu_weight(parts.remote_gpu_experts, plan.layer, remote_route, "remote")?;
            parts
                .cpu_layers
                .get(&plan.layer)
                .ok_or_else(|| LLMError::ModelError("CPU owner layer record is missing".into()))?
                .expert_view(plan.cpu[0].routes[0].descriptor.route.expert_id)?;

            local_slots[0] = Some(parts.layer_owner_executor.allocate_result_slot_for_route(
                generation,
                &plan.local_gpu[0].routes[0].descriptor,
            )?);
            local_slots[1] = Some(parts.layer_owner_executor.allocate_result_slot_for_route(
                generation,
                &plan.local_gpu[0].routes[1].descriptor,
            )?);
            remote_slot = Some(
                parts
                    .remote_executor
                    .allocate_result_slot_for_route(generation, &remote_route.descriptor)?,
            );
            Ok(())
        })();
        if let Err(primary) = allocated {
            if let Err(drain) = model.drain() {
                model.quarantine_execution();
                retain_result_slots(&mut local_slots, &mut remote_slot);
                return Err(release_predispatch_reservation(
                    reservation,
                    LLMError::GpuError(format!(
                        "H6 three-owner preparation failed ({primary}); allocation cleanup drain was not proven ({drain}); allocated CUDA slots quarantined"
                    )),
                ));
            }
            return Err(release_predispatch_reservation(reservation, primary));
        }
        if let Err(drain) = model.drain() {
            model.quarantine_execution();
            retain_result_slots(&mut local_slots, &mut remote_slot);
            return Err(release_predispatch_reservation(
                reservation,
                LLMError::GpuError(format!(
                    "H6 three-owner preparation allocation drain was not proven ({drain}); allocated CUDA slots quarantined"
                )),
            ));
        }

        if let Err(error) = relay.bind_decode_generation(generation, &plan) {
            return Err(release_predispatch_reservation(reservation, error));
        }
        Ok(Self {
            generation,
            plan,
            descriptors,
            reservation: Some(reservation),
            reduction: Some(reduction),
            local_slots,
            remote_slot,
            trace_storage: std::array::from_fn(|_| Some(SelectedExpertTraceStorage::new())),
            completed_traces: std::array::from_fn(|_| None),
            cpu_worker: CpuX8SelectedExpertWorker::new(),
            relay_active: true,
            completion_contracts: [None; GPT_OSS_TOP_K],
            local_kernel_elapsed_ms: [0.0; 2],
            remote_kernel_elapsed_ms: 0.0,
            cpu_elapsed_ns: 0,
            relay_execution: None,
            reduced_execution: None,
            local_activation_d2d_bytes: 0,
            remote_activation_h2d_bytes: 0,
            remote_result_d2h_bytes: 0,
            local_result_d2d_bytes: 0,
            canonical_evidence_d2h_bytes: 0,
            quarantine_required: false,
            pinned_storage_may_be_referenced: false,
            router_admitted: false,
        })
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Borrow the two already-reserved router download targets. The
    /// coordinator has assigned `generation` and every expert/result/trace
    /// resource before these buffers can be handed to GPU0's router.
    pub fn router_download_buffers(
        &mut self,
    ) -> Result<(
        &mut gpt_oss_gpu::pinned_memory::BoundedPinnedLease<u16>,
        &mut gpt_oss_gpu::pinned_memory::BoundedPinnedLease<u8>,
    )> {
        if self.router_admitted {
            return Err(LLMError::ModelError(
                "H6 three-owner router buffers are already admitted".into(),
            ));
        }
        let reservation = self
            .reservation
            .as_mut()
            .ok_or_else(|| LLMError::ModelError("H6 router reservation is missing".into()))?;
        Ok((
            &mut reservation.source_activation,
            &mut reservation.route_descriptors,
        ))
    }

    /// Admit the actual GPU-authored route only when it is byte-for-byte the
    /// pre-reserved oracle plan. This is an oracle admission check, never host
    /// route substitution. Remote compaction happens only after the router's
    /// D2H terminal has proven the canonical activation arena complete.
    pub fn admit_gpu_route(&mut self, actual: &PackedDispatchPlan) -> Result<()> {
        if self.router_admitted || actual != &self.plan {
            return Err(LLMError::ModelError(
                "GPU-authored route does not match the pre-reserved H6 oracle admission".into(),
            ));
        }
        let reservation = self
            .reservation
            .as_mut()
            .ok_or_else(|| LLMError::ModelError("H6 route reservation is missing".into()))?;
        pack_remote_inputs(
            actual,
            &reservation.source_activation,
            &mut reservation.remote_gpu_input,
        )?;
        self.router_admitted = true;
        Ok(())
    }

    pub fn execute<F>(
        mut self,
        model: &mut OwnerSelectiveModel,
        shell: &mut CudaLayerOwnerShell,
        relay: &mut CudaResultRelay,
        reducer: &mut CudaRankOrderedReducer,
        timeline: &CorrelatedTimeline,
        mut terminal: F,
    ) -> std::result::Result<ThreeOwnerDecodeExecution, ThreeOwnerDecodeFailure>
    where
        F: FnMut(ThreeOwnerTerminal) -> Result<()>,
    {
        if !self.router_admitted {
            return Err(self.cleanup_after_error(
                model,
                shell,
                relay,
                reducer,
                LLMError::ModelError(
                    "H6 three-owner expert dispatch preceded GPU route admission".into(),
                ),
            ));
        }
        if let Err(primary) =
            self.execute_inner(model, shell, relay, reducer, timeline, &mut terminal)
        {
            return Err(self.cleanup_after_error(model, shell, relay, reducer, primary));
        }
        let completion_contracts = match self.completion_contracts {
            [Some(rank0), Some(rank1), Some(rank2), Some(rank3)] => [rank0, rank1, rank2, rank3],
            _ => {
                return Err(self.cleanup_after_error(
                    model,
                    shell,
                    relay,
                    reducer,
                    LLMError::ModelError("H6 three-owner completion contract is incomplete".into()),
                ));
            }
        };
        let expert_traces = match self.completed_traces {
            [Some(rank0), Some(rank1), Some(rank2), Some(rank3)] => [rank0, rank1, rank2, rank3],
            _ => {
                return Err(self.cleanup_after_error(
                    model,
                    shell,
                    relay,
                    reducer,
                    LLMError::ModelError("H6 three-owner boundary evidence is incomplete".into()),
                ));
            }
        };
        Ok(ThreeOwnerDecodeExecution {
            generation: self.generation,
            descriptors: self.descriptors,
            completion_contracts,
            expert_traces,
            local_kernel_elapsed_ms: self.local_kernel_elapsed_ms,
            remote_kernel_elapsed_ms: self.remote_kernel_elapsed_ms,
            cpu_elapsed_ns: self.cpu_elapsed_ns,
            cpu_scratch_bytes: self.cpu_worker.scratch_bytes(),
            cpu_high_water_jobs: self.cpu_worker.high_water_jobs(),
            local_activation_d2d_bytes: self.local_activation_d2d_bytes,
            local_activation_host_execution_bytes: 0,
            remote_activation_h2d_bytes: self.remote_activation_h2d_bytes,
            remote_result_d2h_bytes: self.remote_result_d2h_bytes,
            local_result_d2d_bytes: self.local_result_d2d_bytes,
            canonical_evidence_d2h_bytes: self.canonical_evidence_d2h_bytes,
            relay_execution: self
                .relay_execution
                .take()
                .expect("successful relay execution is retained"),
            reduction: self
                .reduced_execution
                .take()
                .expect("successful reduction execution is retained"),
            reservation: self
                .reservation
                .take()
                .expect("successful pinned reservation is retained"),
        })
    }

    /// Explicitly dispose a prepared generation when routing, admission, or
    /// another postdispatch pre-expert check fails. No Drop implementation is
    /// allowed to decide transaction terminality implicitly.
    pub fn abandon_before_expert_dispatch(
        mut self,
        model: &mut OwnerSelectiveModel,
        shell: &mut CudaLayerOwnerShell,
        router: &mut CudaExactRouter,
        relay: &mut CudaResultRelay,
        reducer: &mut CudaRankOrderedReducer,
        mut primary: LLMError,
    ) -> ThreeOwnerDecodeFailure {
        if let Err(router_drain) = router.drain() {
            self.quarantine_required = true;
            self.pinned_storage_may_be_referenced = true;
            primary = LLMError::GpuError(format!(
                "H6 pre-expert failure ({primary}); router drain was not proven ({router_drain})"
            ));
        }
        self.cleanup_after_error(model, shell, relay, reducer, primary)
    }

    fn execute_inner<F>(
        &mut self,
        model: &mut OwnerSelectiveModel,
        shell: &mut CudaLayerOwnerShell,
        relay: &mut CudaResultRelay,
        reducer: &mut CudaRankOrderedReducer,
        timeline: &CorrelatedTimeline,
        terminal: &mut F,
    ) -> Result<()>
    where
        F: FnMut(ThreeOwnerTerminal) -> Result<()>,
    {
        let local_routes = self.plan.local_gpu[0].routes.clone();
        let mut reservation = self
            .reservation
            .take()
            .expect("prepared reservation is present");
        let worker_result =
            self.execute_workers(model, shell, &mut reservation, timeline, terminal);
        self.reservation = Some(reservation);
        let nonlocal = worker_result?;
        let reservation = self
            .reservation
            .take()
            .expect("prepared reservation is present for result upload");
        let completed =
            match relay.upload_results_bound(&self.plan, reservation, &nonlocal, Some(timeline)) {
                Ok(completed) => completed,
                Err(failure) => {
                    self.reservation = failure.reservation;
                    return Err(failure.error);
                }
            };
        self.relay_execution = Some(completed.execution);
        self.reservation = Some(completed.reservation);

        for index in 0..2 {
            let descriptor =
                &self.descriptors[local_routes[index].descriptor.canonical_result_slot as usize];
            let slot = self.local_slots[index]
                .take()
                .expect("drained local result slot is present");
            let completed = match relay.upload_local_device_result_with_timeline(
                self.generation,
                descriptor,
                slot,
                timeline,
            ) {
                Ok(completed) => completed,
                Err(failure) => {
                    self.local_slots[index] = failure.result_slot;
                    return Err(failure.error);
                }
            };
            self.local_result_d2d_bytes += completed.bytes;
            self.local_slots[index] = Some(completed.result_slot);
        }

        let reservation = self
            .reservation
            .take()
            .expect("prepared reservation is present for canonical evidence");
        let completed = match relay.download_complete_decode_evidence(
            self.generation,
            &self.descriptors,
            reservation,
            Some(timeline),
        ) {
            Ok(completed) => completed,
            Err(failure) => {
                self.reservation = failure.reservation;
                return Err(failure.error);
            }
        };
        self.canonical_evidence_d2h_bytes = completed.bytes;
        self.reservation = Some(completed.reservation);
        terminal(ThreeOwnerTerminal::LayerOwnerRelay)?;

        let prepared = self
            .reduction
            .take()
            .expect("prepared rank reduction is present");
        let reduced = reducer.reduce_relay(relay, prepared)?;
        self.relay_active = false;
        terminal(ThreeOwnerTerminal::RankReduction)?;
        self.reduced_execution = Some(reduced);
        Ok(())
    }

    fn execute_workers<F>(
        &mut self,
        model: &mut OwnerSelectiveModel,
        shell: &mut CudaLayerOwnerShell,
        reservation: &mut RelayPinnedReservation,
        timeline: &CorrelatedTimeline,
        terminal: &mut F,
    ) -> Result<[CanonicalRouteContract; 2]>
    where
        F: FnMut(ThreeOwnerTerminal) -> Result<()>,
    {
        let local_routes = self.plan.local_gpu[0].routes.clone();
        let cpu_route = self.plan.cpu[0].routes[0].clone();
        let remote_route = self.plan.remote_gpu[0].routes[0].clone();
        let resident_input = shell.resident_expert_input()?;
        let local_input = Arc::clone(&resident_input.slice);
        let stable_input_device = resident_input.stable_device;
        let parts = model.execution_parts();
        let local0_weights = require_gpu_weight(
            parts.layer_owner_experts,
            self.plan.layer,
            &local_routes[0],
            "layer-owner",
        )?;
        let local1_weights = require_gpu_weight(
            parts.layer_owner_experts,
            self.plan.layer,
            &local_routes[1],
            "layer-owner",
        )?;
        let remote_weights = require_gpu_weight(
            parts.remote_gpu_experts,
            self.plan.layer,
            &remote_route,
            "remote",
        )?;
        let cpu_record = parts.cpu_layers.get(&self.plan.layer).ok_or_else(|| {
            LLMError::ModelError("CPU owner layer record disappeared after preparation".into())
        })?;

        let local0_slot = self.local_slots[0]
            .take()
            .expect("prepared local rank slot 0");
        let local0_trace = self.take_trace(&local_routes[0])?;
        let local0_pending = match parts.layer_owner_executor.prepare_owned_device(
            GptOssPhase::Decode,
            local_routes[0].descriptor.clone(),
            local0_weights,
            Arc::clone(&local_input),
            &stable_input_device,
            local0_slot,
            local0_trace,
        ) {
            Ok(prepared) => {
                match prepared.submit_with_timeline(timeline, "gpu0_local_expert_rank0") {
                    Ok(pending) => pending,
                    Err(failure) => return Err(self.classify_owned_failure(Some(0), failure)),
                }
            }
            Err(failure) => return Err(self.classify_owned_failure(Some(0), failure)),
        };

        let remote_start = remote_route.owner_route_slot as usize * GPT_OSS_HIDDEN_SIZE;
        let remote_slot = self
            .remote_slot
            .take()
            .expect("prepared remote result slot");
        let remote_trace = self.take_trace(&remote_route)?;
        let remote_pending = match parts.remote_executor.prepare_owned_pinned(
            GptOssPhase::Decode,
            remote_route.descriptor.clone(),
            remote_weights,
            &reservation.remote_gpu_input.as_slice()
                [remote_start..remote_start + GPT_OSS_HIDDEN_SIZE],
            remote_slot,
            remote_trace,
        ) {
            Ok(prepared) => match prepared.submit_with_timeline(timeline, "gpu1_expert_rank2") {
                Ok(pending) => pending,
                Err(failure) => return Err(self.classify_owned_failure(None, failure)),
            },
            Err(failure) => return Err(self.classify_owned_failure(None, failure)),
        };

        let cpu_slot = cpu_route.descriptor.canonical_result_slot as usize;
        let mut cpu_trace = self.trace_storage[cpu_slot]
            .take()
            .expect("prepared CPU trace storage");
        let cpu_execution = self.cpu_worker.execute_into_pinned_with_trace(
            self.plan.layer,
            &cpu_route.descriptor,
            cpu_route.owner_route_slot,
            cpu_record.expert_view(cpu_route.descriptor.route.expert_id)?,
            &reservation.source_activation.as_slice()[..GPT_OSS_HIDDEN_SIZE],
            &mut reservation.cpu_result,
            &mut cpu_trace,
            Some(timeline),
        )?;
        self.completed_traces[cpu_slot] = Some(cpu_trace.into_trace());
        self.cpu_elapsed_ns = cpu_execution.elapsed_ns;
        insert_completion(&mut self.completion_contracts, cpu_execution.route_contract)?;
        terminal(ThreeOwnerTerminal::CpuExpert)?;

        let local0_execution =
            match local0_pending.drain_with_trace(None, timeline, "gpu0_local_expert_rank0") {
                Ok(execution) => execution,
                Err(failure) => return Err(self.classify_owned_failure(Some(0), failure)),
            };
        self.local_activation_d2d_bytes += local0_execution.input_d2d_bytes;
        self.local_kernel_elapsed_ms[0] = local0_execution.kernel_elapsed_ms;
        self.store_gpu_completion(0, local0_execution)?;

        let remote_execution = match remote_pending.drain_with_trace(
            Some(&mut reservation.remote_gpu_result),
            timeline,
            "gpu1_expert_rank2",
        ) {
            Ok(execution) => execution,
            Err(failure) => return Err(self.classify_owned_failure(None, failure)),
        };
        let remote_contract = remote_execution.route_contract;
        self.remote_activation_h2d_bytes = remote_execution.input_h2d_bytes;
        self.remote_result_d2h_bytes = remote_execution.output_d2h_bytes;
        self.remote_kernel_elapsed_ms = remote_execution.kernel_elapsed_ms;
        self.store_remote_completion(remote_execution)?;
        terminal(ThreeOwnerTerminal::RemoteGpuExpert)?;

        let local1_slot = self.local_slots[1]
            .take()
            .expect("prepared local rank slot 1");
        let local1_trace = self.take_trace(&local_routes[1])?;
        let local1_pending = match parts.layer_owner_executor.prepare_owned_device(
            GptOssPhase::Decode,
            local_routes[1].descriptor.clone(),
            local1_weights,
            local_input,
            &stable_input_device,
            local1_slot,
            local1_trace,
        ) {
            Ok(prepared) => {
                match prepared.submit_with_timeline(timeline, "gpu0_local_expert_rank3") {
                    Ok(pending) => pending,
                    Err(failure) => return Err(self.classify_owned_failure(Some(1), failure)),
                }
            }
            Err(failure) => return Err(self.classify_owned_failure(Some(1), failure)),
        };
        let local1_execution =
            match local1_pending.drain_with_trace(None, timeline, "gpu0_local_expert_rank3") {
                Ok(execution) => execution,
                Err(failure) => return Err(self.classify_owned_failure(Some(1), failure)),
            };
        self.local_activation_d2d_bytes += local1_execution.input_d2d_bytes;
        self.local_kernel_elapsed_ms[1] = local1_execution.kernel_elapsed_ms;
        self.store_gpu_completion(1, local1_execution)?;
        terminal(ThreeOwnerTerminal::LayerOwnerExpert)?;
        Ok([cpu_execution.route_contract, remote_contract])
    }

    fn take_trace(&mut self, route: &PackedDispatchRoute) -> Result<SelectedExpertTraceStorage> {
        let slot = route.descriptor.canonical_result_slot as usize;
        self.trace_storage
            .get_mut(slot)
            .and_then(Option::take)
            .ok_or_else(|| LLMError::ModelError("selected-expert trace slot is missing".into()))
    }

    fn store_gpu_completion(
        &mut self,
        local_index: usize,
        execution: super::cuda_expert::OwnedSelectedExpertExecution,
    ) -> Result<()> {
        let slot = execution.route_contract.result_slot as usize;
        insert_completion(&mut self.completion_contracts, execution.route_contract)?;
        self.completed_traces[slot] = Some(execution.trace);
        self.local_slots[local_index] = Some(execution.result_slot);
        Ok(())
    }

    fn store_remote_completion(
        &mut self,
        execution: super::cuda_expert::OwnedSelectedExpertExecution,
    ) -> Result<()> {
        let slot = execution.route_contract.result_slot as usize;
        insert_completion(&mut self.completion_contracts, execution.route_contract)?;
        self.completed_traces[slot] = Some(execution.trace);
        self.remote_slot = Some(execution.result_slot);
        Ok(())
    }

    fn classify_owned_failure(
        &mut self,
        local_index: Option<usize>,
        failure: OwnedSelectedExpertFailure,
    ) -> LLMError {
        let (error, slot, drain_proven, pinned_storage_may_be_referenced, _) = failure.into_parts();
        if let Some(slot) = slot {
            if let Some(index) = local_index {
                self.local_slots[index] = Some(slot);
            } else {
                self.remote_slot = Some(slot);
            }
        }
        if !drain_proven {
            self.quarantine_required = true;
            self.pinned_storage_may_be_referenced |= pinned_storage_may_be_referenced;
        }
        error
    }

    fn cleanup_after_error(
        &mut self,
        model: &mut OwnerSelectiveModel,
        shell: &mut CudaLayerOwnerShell,
        relay: &mut CudaResultRelay,
        reducer: &mut CudaRankOrderedReducer,
        primary: LLMError,
    ) -> ThreeOwnerDecodeFailure {
        let model_drain = if self.quarantine_required {
            Err(LLMError::GpuError(
                "owned selected-expert drain was not proven".into(),
            ))
        } else {
            model.drain()
        };
        let shell_drain = if model_drain.is_ok() {
            shell.drain()
        } else {
            Err(LLMError::GpuError(
                "model drain was not proven before shell drain".into(),
            ))
        };
        let reducer_drain = if model_drain.is_ok() && shell_drain.is_ok() {
            reducer.prove_transaction_drain()
        } else {
            Err(LLMError::GpuError(
                "earlier owner drain was not proven before reducer drain".into(),
            ))
        };
        let relay_close = if !self.relay_active {
            // Successful rank reduction already terminally closed this relay
            // generation. A later callback/evidence error has nothing left to
            // abandon and remains a recoverable postdispatch failure.
            Ok(())
        } else if model_drain.is_ok() && shell_drain.is_ok() && reducer_drain.is_ok() {
            relay.abandon_decode_generation(self.generation, true)
        } else {
            Err(LLMError::GpuError(
                "earlier owner drain was not proven before relay close".into(),
            ))
        };
        if model_drain.is_ok()
            && shell_drain.is_ok()
            && reducer_drain.is_ok()
            && relay_close.is_ok()
        {
            self.relay_active = false;
            if let Some(reservation) = self.reservation.take() {
                if let Err(release) = reservation.release_drained() {
                    return ThreeOwnerDecodeFailure {
                        error: LLMError::GpuError(format!(
                            "H6 three-owner execution failed ({primary}); drained cleanup release failed ({release})"
                        )),
                        drain_proven: true,
                    };
                }
            }
            return ThreeOwnerDecodeFailure {
                error: primary,
                drain_proven: true,
            };
        }

        // No storage that may still be named by CUDA is allowed to fall out of
        // this consumed prepared step. Poison both durable owners and retain
        // every route slot/reservation for process lifetime.
        model.quarantine_execution();
        shell.quarantine_external_device_use();
        relay.quarantine_unproven_device_work();
        if let Some(reservation) = self.reservation.take() {
            std::mem::forget(reservation);
        }
        for slot in &mut self.local_slots {
            if let Some(slot) = slot.take() {
                std::mem::forget(slot);
            }
        }
        if let Some(slot) = self.remote_slot.take() {
            std::mem::forget(slot);
        }
        ThreeOwnerDecodeFailure {
            error: LLMError::GpuError(format!(
                "H6 three-owner execution failed ({primary}); cleanup could not prove every owner drain: model={model_drain:?} shell={shell_drain:?} reducer={reducer_drain:?} relay={relay_close:?}; GPU state and possibly referenced storage quarantined (pinned_referenced={})",
                self.pinned_storage_may_be_referenced
            )),
            drain_proven: false,
        }
    }
}

fn release_predispatch_reservation(
    reservation: RelayPinnedReservation,
    primary: LLMError,
) -> LLMError {
    match reservation.release_drained() {
        Ok(()) => primary,
        Err(release) => LLMError::GpuError(format!(
            "H6 three-owner preparation failed ({primary}); pinned reservation rollback failed ({release})"
        )),
    }
}

fn retain_result_slots(
    local_slots: &mut [Option<CudaSelectedExpertResultSlot>; 2],
    remote_slot: &mut Option<CudaSelectedExpertResultSlot>,
) {
    for slot in local_slots {
        if let Some(slot) = slot.take() {
            std::mem::forget(slot);
        }
    }
    if let Some(slot) = remote_slot.take() {
        std::mem::forget(slot);
    }
}

fn require_gpu_weight(
    weights: &std::collections::BTreeMap<GptOssExpertKey, Arc<CudaSelectedExpertWeights>>,
    layer: u16,
    route: &PackedDispatchRoute,
    label: &str,
) -> Result<Arc<CudaSelectedExpertWeights>> {
    let key = GptOssExpertKey {
        layer,
        expert: route.descriptor.route.expert_id,
    };
    let weight = weights.get(&key).ok_or_else(|| {
        LLMError::ModelError(format!(
            "{label} resident expert {layer}/{} is missing",
            key.expert
        ))
    })?;
    if weight.descriptor().key != key || weight.descriptor().owner != route.descriptor.owner {
        return Err(LLMError::ModelError(format!(
            "{label} resident expert descriptor does not match the canonical route"
        )));
    }
    Ok(Arc::clone(weight))
}

fn insert_completion(
    completions: &mut [Option<CanonicalRouteContract>; GPT_OSS_TOP_K],
    completion: CanonicalRouteContract,
) -> Result<()> {
    let slot = usize::try_from(completion.result_slot).map_err(|_| {
        LLMError::ModelError("three-owner completion slot cannot be represented".into())
    })?;
    if slot >= GPT_OSS_TOP_K || completions[slot].replace(completion).is_some() {
        return Err(LLMError::ModelError(
            "three-owner completion slot is out of range or duplicated".into(),
        ));
    }
    Ok(())
}
