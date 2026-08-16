use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use gpt_oss_core::error::LLMError;
use gpt_oss_cpu_kernels::{KernelPath, Mxfp4MatmulBackend};
use gpt_oss_engine::{
    DrainRole, GpuSequenceVisibility, HeterogeneousStepId, HeterogeneousTransactionCoordinator,
    SequenceCommitImage, TransactionOutcome, TransactionTerminalRecord,
};
use gpt_oss_gpu::event::{CorrelatedTimeline, TimelinePoint};
use gpt_oss_model_runner::heterogeneous::{
    exact_rank_ordered_reduction_reference, exact_selected_expert_reference, pack_routes_bounded,
    quarantine_three_owner_after_unproven_drain, CanonicalExpertContribution,
    CanonicalRouteContract, CudaExactRouter, CudaLayerOwnerShell, CudaRankOrderedReducer,
    CudaResultRelay, ExactRouterWeightsView, ExpertOwner, GptOssExpertKey, GptOssPhase,
    GptOssRouteDescriptor, GptOssRoutedBatchDescriptor, NativeMxfp4ExpertView,
    PreparedRankOrderedReduction, PreparedThreeOwnerDecode, RelayPinnedPools,
    SelectedExpertFirstDivergenceTrace, ThreeOwnerTerminal, DOWN_BIAS_VALUES, DOWN_BLOCK_BYTES,
    DOWN_SCALE_BYTES, GATE_UP_BIAS_VALUES, GATE_UP_BLOCK_BYTES, GATE_UP_SCALE_BYTES,
    GPT_OSS_REDUCTION_OUTPUT_BYTES, GPT_OSS_REDUCTION_TRACE_BYTES,
};
use gpt_oss_model_runner::model_loader::gpt_oss_native::GptOssCheckpointView;
use gpt_oss_model_runner::model_loader::owner_selective::{
    OwnerSelectiveConstructor, OwnerSelectiveModel,
};
use gpt_oss_model_runner::{
    CpuExpertProjection, CpuGptOssConfig, CpuKvCacheSnapshot, CpuLayerTrace, CpuModelRunner,
    CpuModelRunnerOptions, CpuTensorStore,
};
use half::bf16;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(feature = "heterogeneous-test-faults")]
use gpt_oss_model_runner::heterogeneous::{
    exercise_owned_device_input_fault_and_retry, exercise_owned_device_input_unproven_fault,
    LayerOwnerInjectedFault,
};

const EXPECTED_ROUTE: [usize; 4] = [31, 21, 22, 6];

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    native_model: PathBuf,
    #[arg(long)]
    cpu_repack_cache: PathBuf,
    #[arg(long)]
    owner_cache: PathBuf,
    #[arg(long)]
    placement: PathBuf,
    #[arg(long)]
    retained_trace: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[cfg(feature = "heterogeneous-test-faults")]
    #[arg(long)]
    exercise_shell_faults: bool,
    #[cfg(feature = "heterogeneous-test-faults")]
    #[arg(long)]
    exercise_unproven_device_input_fault_only: bool,
    #[arg(long)]
    exercise_three_owner: bool,
    #[arg(long, requires = "exercise_three_owner")]
    h6b_output: Option<PathBuf>,
}

#[derive(Deserialize)]
struct ControlDocument {
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
}

#[derive(Serialize)]
struct H6aEvidence {
    schema: &'static str,
    pre_moe_authority: &'static str,
    post_router_authority: &'static str,
    model_config_sha256: String,
    model_index_sha256: String,
    native_mapping_sha256: String,
    placement_sha256: String,
    retained_trace_sha256: String,
    layer_owner_pci_bus_id: String,
    token_id: u32,
    position: usize,
    prior_kv_rows: usize,
    prior_kv_bytes: usize,
    owner_shell_work_bytes: usize,
    owner_shell_host_staging_bytes: usize,
    fault_feature_enabled: bool,
    fault_exercise_requested: bool,
    shell_faults_drained: Vec<&'static str>,
    shell_fault_retries_passed: bool,
    owned_device_input_fault_drained: bool,
    owned_device_input_retry_exact: bool,
    owned_device_input_retry_d2d_bytes: usize,
    owned_device_input_retry_host_bytes: usize,
    owner_shell_kernel_elapsed_ms: [f32; 2],
    selected_experts: Vec<usize>,
    routing_weights_bf16_bits: Vec<u16>,
    router_logits_sha256: String,
    router_input_device_handoff_bytes: usize,
    router_input_execution_host_bytes: usize,
    router_evidence_activation_d2h_bytes: usize,
    router_descriptor_d2h_bytes: usize,
    cpu_authority_contribution_h2d_bytes: usize,
    exact_expert_output_sha256: Vec<String>,
    moe_output_sha256: String,
    reducer_output_device_handoff_bytes: usize,
    reducer_output_execution_host_bytes: usize,
    reducer_evidence_d2h_bytes: usize,
    layer_output_sha256: String,
    reduction_kernel_elapsed_ms: f32,
    boundaries: Vec<BoundaryEvidence>,
    owner_shell_prefix_repeat_exact: bool,
    passed: bool,
}

#[derive(Serialize)]
struct BoundaryEvidence {
    name: &'static str,
    values: usize,
    sha256: String,
    bit_exact: bool,
}

#[derive(Serialize)]
struct H6bExpertEvidence {
    route_rank: u8,
    expert_id: u16,
    owner: String,
    gate_up_sha256: String,
    scaled_gate_sha256: String,
    sigmoid_sha256: String,
    glu_sha256: String,
    linear_sha256: String,
    swiglu_sha256: String,
    down_sha256: String,
    bit_exact: bool,
}

#[derive(Serialize)]
struct H6bRouteIdentityEvidence {
    source_row: u32,
    activation_slot: u32,
    source_activation_slot: u32,
    route_rank: u8,
    expert_id: u16,
    weight_bf16_bits: u16,
    owner: ExpertOwner,
    owner_role: String,
    placement_epoch: u64,
    canonical_result_slot: u32,
}

#[derive(Serialize)]
struct H6bPackedRouteEvidence {
    identity: H6bRouteIdentityEvidence,
    relay_activation_slot: u32,
    owner_route_slot: u32,
}

#[derive(Debug, Clone, Serialize)]
struct H6bResourceSnapshot {
    captured_unix_seconds: u64,
    process_swap_used_bytes: u64,
    swap_used_bytes: u64,
    swap_cached_bytes: u64,
    gpu_free_bytes: [u64; 2],
    gpu_total_bytes: [u64; 2],
}

#[derive(Serialize)]
struct H6bCaseEvidence {
    label: &'static str,
    step_id: u64,
    transaction_generation: u64,
    oracle_admission: &'static str,
    gpu_route_matched_admission: bool,
    packed_admission_descriptors: Vec<H6bPackedRouteEvidence>,
    completion_identities: Vec<H6bRouteIdentityEvidence>,
    completion_contracts_exact: bool,
    selected_experts: Vec<u16>,
    selected_weight_bf16_bits: Vec<u16>,
    owner_counts: [usize; 3],
    local_activation_d2d_bytes: usize,
    local_activation_host_execution_bytes: usize,
    remote_activation_h2d_bytes: usize,
    remote_result_d2h_bytes: usize,
    cpu_result_h2d_bytes: usize,
    remote_result_h2d_bytes: usize,
    local_result_d2d_bytes: usize,
    canonical_evidence_d2h_bytes: usize,
    cpu_scratch_bytes: usize,
    cpu_high_water_jobs: usize,
    local_kernel_elapsed_ms: [f32; 2],
    remote_kernel_elapsed_ms: f32,
    cpu_elapsed_ns: u64,
    expert_boundaries: Vec<H6bExpertEvidence>,
    reduction_output_sha256: String,
    layer_output_sha256: String,
    reduction_bit_exact: bool,
    layer_output_bit_exact: bool,
    cpu_gpu0_gpu1_compute_overlap: bool,
    compute_intervals_ns: [[u64; 2]; 3],
    timeline: Vec<TimelinePoint>,
    before_visibility: GpuSequenceVisibility,
    after_visibility: GpuSequenceVisibility,
    terminal: TransactionTerminalRecord,
    active_steps_after: usize,
    free_blocks_after: usize,
}

#[derive(Serialize)]
struct H6bEvidence {
    schema: &'static str,
    transaction_scope: &'static str,
    physical_kv_publication_claimed: bool,
    route_authority: &'static str,
    generation_authority: &'static str,
    placement_sha256: String,
    layer_owner_pci_bus_id: String,
    remote_worker_pci_bus_id: String,
    resources_before: H6bResourceSnapshot,
    resources_after: H6bResourceSnapshot,
    swap_growth_bytes: u64,
    gpu_free_loss_bytes: [u64; 2],
    gpu_free_loss_tolerance_bytes: u64,
    allocator_drift_gate_passed: bool,
    cases: Vec<H6bCaseEvidence>,
    committed_case_passed: bool,
    drained_discard_passed: bool,
    clean_repeat_passed: bool,
    coordinator_shutdown_zero: bool,
    coordinator_free_blocks_after_recycle: usize,
    pinned_pool_active_leases_after: usize,
    passed: bool,
}

/// Owns the coordinator reservation from `reserve_step` until publication or
/// discard. A predispatch drop cancels and releases immediately. After
/// dispatch, Drop suppresses publication but finalizes only when the caller
/// has explicitly proven that every submitted CPU/CUDA role is drained.
struct H6bTransactionGuard<'a> {
    coordinator: &'a mut HeterogeneousTransactionCoordinator,
    step_id: HeterogeneousStepId,
    dispatched: bool,
    cancelled: bool,
    drain_proven: bool,
    terminal_roles: [bool; 6],
    finished: bool,
}

impl<'a> H6bTransactionGuard<'a> {
    fn reserve(
        coordinator: &'a mut HeterogeneousTransactionCoordinator,
        sequence_id: u64,
        placement_epoch: u64,
    ) -> Result<Self> {
        let step_id = coordinator.reserve_step(sequence_id, 1, placement_epoch)?;
        Ok(Self {
            coordinator,
            step_id,
            dispatched: false,
            cancelled: false,
            drain_proven: false,
            terminal_roles: [false; 6],
            finished: false,
        })
    }

    const fn step_id(&self) -> HeterogeneousStepId {
        self.step_id
    }

    fn committed_view(&self, sequence_id: u64) -> Option<&GpuSequenceVisibility> {
        self.coordinator.committed_view(sequence_id)
    }

    fn mark_prepared(&mut self) -> Result<()> {
        self.coordinator.mark_prepared(self.step_id)?;
        Ok(())
    }

    fn mark_dispatched(&mut self) -> Result<()> {
        self.coordinator
            .mark_dispatched(self.step_id, &H6B_DRAIN_ROLES)?;
        self.dispatched = true;
        Ok(())
    }

    fn mark_terminal(
        &mut self,
        role: DrainRole,
    ) -> std::result::Result<(), gpt_oss_core::error::LLMError> {
        self.coordinator.mark_terminal(self.step_id, role)?;
        self.terminal_roles[drain_role_index(role)] = true;
        Ok(())
    }

    fn cancel(&mut self) -> Result<Option<TransactionTerminalRecord>> {
        let terminal = self.coordinator.cancel_step(self.step_id)?;
        self.cancelled = true;
        if terminal.is_some() {
            self.finished = true;
        }
        Ok(terminal)
    }

    fn prove_drained(&mut self) {
        self.drain_proven = true;
    }

    fn mark_reduced(&mut self, output: &[u16]) -> Result<()> {
        self.coordinator.mark_reduced(self.step_id, output)?;
        Ok(())
    }

    fn prepare_commit(&mut self, image: SequenceCommitImage) -> Result<()> {
        self.coordinator.prepare_commit(self.step_id, image)?;
        Ok(())
    }

    fn commit(&mut self) -> Result<TransactionTerminalRecord> {
        let terminal = self.coordinator.commit(self.step_id)?;
        self.finished = true;
        Ok(terminal)
    }

    fn finalize_discard(&mut self) -> Result<TransactionTerminalRecord> {
        let terminal = self.coordinator.finalize_discard(self.step_id)?;
        self.finished = true;
        Ok(terminal)
    }

    fn active_step_count(&self) -> usize {
        self.coordinator.active_step_count()
    }

    fn free_block_count(&self) -> usize {
        self.coordinator.free_block_count()
    }
}

impl Drop for H6bTransactionGuard<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        if !self.cancelled {
            match self.coordinator.cancel_step(self.step_id) {
                Ok(Some(_)) => {
                    self.finished = true;
                    return;
                }
                Ok(None) => self.cancelled = true,
                Err(_) => return,
            }
        }
        if !self.dispatched || !self.drain_proven {
            // Predispatch cancellation was already finalized above. An
            // uncertain postdispatch drain intentionally remains active and
            // publication-forbidden so shutdown cannot claim safe cleanup.
            return;
        }
        for role in H6B_DRAIN_ROLES {
            let index = drain_role_index(role);
            if !self.terminal_roles[index] {
                if self.coordinator.mark_terminal(self.step_id, role).is_err() {
                    return;
                }
                self.terminal_roles[index] = true;
            }
        }
        if self.coordinator.finalize_discard(self.step_id).is_ok() {
            self.finished = true;
        }
    }
}

const fn drain_role_index(role: DrainRole) -> usize {
    match role {
        DrainRole::LayerOwnerRouter => 0,
        DrainRole::LayerOwnerExpert => 1,
        DrainRole::LayerOwnerRelay => 2,
        DrainRole::CpuExpert => 3,
        DrainRole::RemoteGpuExpert => 4,
        DrainRole::RankReduction => 5,
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let control: ControlDocument = serde_json::from_slice(&std::fs::read(&cli.retained_trace)?)?;
    let token_id = *control
        .generated_token_ids
        .first()
        .context("retained control has no generated token")?;
    if token_id != 200_005 || control.prompt_token_ids.len() != 63 {
        bail!("retained control is not the pinned 63-token/200005 fixture");
    }

    let mut cpu = CpuModelRunner::load_with_options(
        &cli.model,
        &cli.cpu_repack_cache,
        CpuModelRunnerOptions {
            kernel_path: KernelPath::Auto,
            matmul_backend: Mxfp4MatmulBackend::Auto,
            threads: 8,
            context_cap: 128,
            expert_projection: CpuExpertProjection::ResidualQ8,
            xe: None,
            profile_capacity_bytes: None,
        },
    )?;
    cpu.prefill(&control.prompt_token_ids)?;
    let cache = cpu
        .caches()
        .first()
        .context("CPU authority has no layer-0 cache")?
        .oracle_snapshot();
    let (_, trace) = cpu.decode_trace(token_id, &[0], 8, 1)?;
    let authority = trace.layers.first().context("CPU layer-0 trace missing")?;
    if authority.selected_experts != EXPECTED_ROUTE {
        bail!(
            "real CPU route changed: {:?} != {:?}",
            authority.selected_experts,
            EXPECTED_ROUTE
        );
    }
    let config = cpu.config().clone();
    drop(cpu);

    let store = CpuTensorStore::open(&cli.model)?;
    let embedding_tensor = store.tensor("model.embed_tokens.weight")?;
    let embeddings = embedding_tensor.bf16()?;
    let hidden_start = token_id as usize * config.hidden_size;
    let hidden = embeddings[hidden_start..hidden_start + config.hidden_size]
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    drop(store);

    let manifest = serde_json::from_slice(&std::fs::read(&cli.placement)?)?;
    let checkpoint = GptOssCheckpointView::open(&cli.native_model)?;
    let native_mapping_sha256 = checkpoint.mapping_sha256().to_owned();
    let router_weights = tensor_u16(
        &checkpoint,
        "model.layers.0.mlp.router.weight",
        config.num_local_experts * config.hidden_size,
    )?;
    let router_bias = tensor_u16(
        &checkpoint,
        "model.layers.0.mlp.router.bias",
        config.num_local_experts,
    )?;
    let constructor = OwnerSelectiveConstructor::new(&cli.owner_cache);
    let mut model = constructor.construct(checkpoint, &manifest, |_| Ok(()))?;
    let placement_sha256 = model.placement().manifest_hash().to_owned();
    let layer_owner_pci_bus_id = model
        .placement()
        .layer_owner()
        .stable_id
        .pci_bus_id
        .to_string();
    let mut shell = CudaLayerOwnerShell::new(&model, &config)?;
    #[allow(unused_mut)]
    let mut shell_faults_drained = Vec::with_capacity(5);
    #[allow(unused_mut)]
    let mut shell_fault_retries_passed = false;
    #[cfg(feature = "heterogeneous-test-faults")]
    let fault_exercise_requested = cli.exercise_shell_faults;
    #[cfg(not(feature = "heterogeneous-test-faults"))]
    let fault_exercise_requested = false;
    #[cfg(feature = "heterogeneous-test-faults")]
    if cli.exercise_shell_faults {
        for (fault, label) in [
            (
                LayerOwnerInjectedFault::SubmitAfterPriorKeyEnqueue,
                "submit_after_prior_key_enqueue",
            ),
            (
                LayerOwnerInjectedFault::TerminalDrain,
                "terminal_fallback_drain",
            ),
            (
                LayerOwnerInjectedFault::BoundaryDownloadAfterFirstEnqueue,
                "boundary_download_after_first_enqueue",
            ),
        ] {
            shell.inject_next_failure(fault)?;
            if shell
                .execute_layer0_decode(&model, &config, token_id, 63, &cache)
                .is_ok()
                || !shell.last_fault_drained()
                || shell.is_poisoned_for_test()
            {
                bail!("layer-owner fault {label} did not drain and remain safely reusable");
            }
            shell_faults_drained.push(label);
            shell.execute_layer0_decode(&model, &config, token_id, 63, &cache)?;
        }
    }
    let first = shell.execute_layer0_decode(&model, &config, token_id, 63, &cache)?;
    let second = shell.execute_layer0_decode(&model, &config, token_id, 63, &cache)?;

    let expected = ExpectedBoundaries::new(authority, hidden);
    let actual = ActualBoundaries::new(&first);
    let repeated = ActualBoundaries::new(&second);
    let mut boundaries = Vec::with_capacity(expected.values.len());
    for ((name, expected), (repeated_name, repeated)) in
        expected.values.iter().zip(repeated.values.iter())
    {
        let actual = actual
            .values
            .iter()
            .find(|(actual_name, _)| actual_name == name)
            .map(|(_, values)| values)
            .context("actual boundary missing")?;
        if name != repeated_name {
            bail!("repeat boundary ordering changed");
        }
        exact(name, expected, actual)?;
        exact(name, actual, repeated)?;
        boundaries.push(BoundaryEvidence {
            name,
            values: actual.len(),
            sha256: hash_u16(actual),
            bit_exact: true,
        });
    }

    const GENERATION: u64 = 6_001;
    let mut router = CudaExactRouter::new(
        model.placement().layer_owner().stable_id.clone(),
        1,
        ExactRouterWeightsView {
            experts: config.num_local_experts,
            weight_bf16_bits: &router_weights,
            bias_bf16_bits: &router_bias,
        },
    )?;
    let pools = RelayPinnedPools::warm_exact(&router, 1)?;
    let mut reservation = pools.try_reserve_all(GENERATION)?;
    let routed = shell.route_resident_decode(
        &mut router,
        0,
        model.placement().placement_epoch(),
        &mut reservation.source_activation,
        &mut reservation.route_descriptors,
        None,
    )?;
    exact(
        "router_input_evidence",
        &first.router_input_bf16_bits,
        &routed.batch.activation_bf16_bits,
    )?;
    exact(
        "router_logits",
        &bits(&authority.router_logits),
        &routed.router_logits_bf16_bits,
    )?;
    let routed_ids = routed
        .batch
        .routes
        .iter()
        .map(|route| usize::from(route.expert_id))
        .collect::<Vec<_>>();
    let routed_weights = routed
        .batch
        .routes
        .iter()
        .map(|route| route.weight_bf16_bits)
        .collect::<Vec<_>>();
    if routed_ids != authority.selected_experts
        || routed_weights != bits(&authority.routing_weights)
    {
        bail!("GPU-authored route IDs or BF16 weights diverged from CPU authority");
    }
    let plan = pack_routes_bounded(&routed.batch, model.placement())?;
    #[cfg(feature = "heterogeneous-test-faults")]
    if cli.exercise_unproven_device_input_fault_only {
        let local_route = plan
            .local_gpu
            .first()
            .and_then(|owner| owner.routes.first())
            .context("H6 destructive D2D fault probe has no GPU0-local route")?;
        let fault_timeline = gpt_oss_gpu::event::CorrelatedTimeline::new();
        let probe = exercise_owned_device_input_unproven_fault(
            &mut model,
            &mut shell,
            0,
            GENERATION,
            local_route,
            &fault_timeline,
        )?;
        if probe.drain_proven
            || probe.result_slot_returned
            || !probe.input_retained
            || !probe.weights_retained
            || !probe.trace_retained
            || !probe.slot_retained
            || !probe.executor_marked_unproven
            || !probe.model_quarantined
            || !probe.shell_quarantined
        {
            bail!("destructive owned D2D fault did not quarantine every owner");
        }
        return write_json(
            &cli.output,
            &serde_json::json!({
                "schema": "gpt-oss-rs.heterogeneous-owned-d2d-unproven-fault/v2",
                "drain_proven": probe.drain_proven,
                "result_slot_returned": probe.result_slot_returned,
                "input_retained": probe.input_retained,
                "weights_retained": probe.weights_retained,
                "trace_retained": probe.trace_retained,
                "slot_retained": probe.slot_retained,
                "executor_marked_unproven": probe.executor_marked_unproven,
                "model_quarantined": probe.model_quarantined,
                "shell_quarantined": probe.shell_quarantined,
                "retry_permitted": false,
                "passed": true
            }),
        );
    }
    #[allow(unused_mut)]
    let mut owned_device_input_fault_drained = false;
    #[allow(unused_mut)]
    let mut owned_device_input_retry_exact = false;
    #[allow(unused_mut)]
    let mut owned_device_input_retry_d2d_bytes = 0;
    #[allow(unused_mut)]
    let mut owned_device_input_retry_host_bytes = 0;
    #[cfg(feature = "heterogeneous-test-faults")]
    if cli.exercise_shell_faults {
        let local_route = plan
            .local_gpu
            .first()
            .and_then(|owner| owner.routes.first())
            .context("H6a fault probe has no GPU0-local route")?;
        let expected_trace = exact_expert_trace(
            model.checkpoint(),
            0,
            local_route.descriptor.route.expert_id,
            &routed.batch.activation_bf16_bits,
        )?;
        let fault_timeline = gpt_oss_gpu::event::CorrelatedTimeline::new();
        let probe = exercise_owned_device_input_fault_and_retry(
            &mut model,
            &shell,
            0,
            GENERATION,
            local_route,
            &fault_timeline,
        )?;
        exact_trace(
            "owned_device_input_fault_retry",
            &expected_trace,
            &probe.retry_trace,
        )?;
        if probe.retry_input_d2d_bytes != config.hidden_size * size_of::<u16>()
            || probe.retry_input_host_bytes != 0
        {
            bail!("owned device-input retry did not preserve the resident D2D boundary");
        }
        owned_device_input_fault_drained = probe.fault_drained;
        owned_device_input_retry_exact = true;
        owned_device_input_retry_d2d_bytes = probe.retry_input_d2d_bytes;
        owned_device_input_retry_host_bytes = probe.retry_input_host_bytes;
    }
    let prepared =
        PreparedRankOrderedReduction::prepare(&routed.batch, model.placement(), GENERATION)?;
    let descriptors = prepared.expected_results().to_vec();
    // The retained runner is ResidualQ8. Its dense/K/V/router boundaries are
    // authoritative here, but its expert/MoE/layer outputs are not the exact
    // H2 selected-expert contract. Recompute every contribution directly from
    // the native packed expert view using the real GPU-authored route.
    let authority_outputs = descriptors
        .iter()
        .map(|descriptor| {
            exact_expert_output(
                model.checkpoint(),
                0,
                descriptor.expert_id,
                &routed.batch.activation_bf16_bits,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let contributions = descriptors
        .iter()
        .cloned()
        .zip(authority_outputs.iter().cloned())
        .map(
            |(descriptor, output_bf16_bits)| CanonicalExpertContribution {
                descriptor,
                output_bf16_bits,
            },
        )
        .collect::<Vec<_>>();
    let exact_reduction =
        exact_rank_ordered_reduction_reference(&routed.batch, model.placement(), &contributions)?;
    let exact_expert_output_sha256 = authority_outputs
        .iter()
        .map(|output| hash_u16(output))
        .collect::<Vec<_>>();
    let mut relay = CudaResultRelay::new(&router, 1)?;
    relay.bind_decode_generation(GENERATION, &plan)?;
    let cpu_authority_contribution_h2d_bytes =
        relay.upload_cpu_authority_control(GENERATION, &descriptors, authority_outputs)?;
    let mut reducer = CudaRankOrderedReducer::new(&relay)?;
    let reduced = reducer.reduce_relay(&mut relay, prepared)?;
    exact(
        "moe_output",
        &exact_reduction.output_bf16_bits,
        &reduced.output_bf16_bits,
    )?;
    let expected_layer_output = exact_residual(
        &first.post_attention_residual_bf16_bits,
        &exact_reduction.output_bf16_bits,
    )?;
    #[cfg(feature = "heterogeneous-test-faults")]
    if cli.exercise_shell_faults {
        for (fault, label) in [
            (
                LayerOwnerInjectedFault::FinalResidualAfterD2dEnqueue,
                "final_residual_after_d2d_enqueue",
            ),
            (
                LayerOwnerInjectedFault::FinalOutputAfterD2hEnqueue,
                "final_output_after_d2h_enqueue",
            ),
        ] {
            shell.inject_next_failure(fault)?;
            if shell.finish_layer_residual_resident(&reducer).is_ok()
                || !shell.last_fault_drained()
                || shell.is_poisoned_for_test()
            {
                bail!(
                    "layer-owner residual fault {label} did not drain and remain safely reusable"
                );
            }
            shell_faults_drained.push(label);
            let retry = shell.finish_layer_residual_resident(&reducer)?;
            exact("layer_output_fault_retry", &expected_layer_output, &retry)?;
        }
        if shell_faults_drained.len() != 5 {
            bail!("not all five layer-owner lifecycle faults were exercised");
        }
        shell_fault_retries_passed = true;
    }
    let layer_output = shell.finish_layer_residual_resident(&reducer)?;
    exact("layer_output", &expected_layer_output, &layer_output)?;
    for (name, values) in [
        ("router_logits", routed.router_logits_bf16_bits.as_slice()),
        ("moe_output", reduced.output_bf16_bits.as_slice()),
        ("layer_output", layer_output.as_slice()),
    ] {
        boundaries.push(BoundaryEvidence {
            name,
            values: values.len(),
            sha256: hash_u16(values),
            bit_exact: true,
        });
    }
    reservation.release_drained()?;
    drop(reducer);
    drop(relay);

    let h6b_evidence = if cli.exercise_three_owner {
        Some(run_h6b_campaign(
            &mut model,
            &mut shell,
            &mut router,
            &pools,
            &config,
            &cache,
            token_id,
            &control.prompt_token_ids,
            authority,
            &expected,
            placement_sha256.clone(),
            layer_owner_pci_bus_id.clone(),
        )?)
    } else {
        None
    };

    let evidence = H6aEvidence {
        schema: "gpt-oss-rs.heterogeneous-layer-oracle-h6a/v4",
        pre_moe_authority: "retained-residual-q8-dense-kv-router-only",
        post_router_authority: "native-mxfp4-exact-selected-expert-reference",
        model_config_sha256: hash_file(&cli.model.join("config.json"))?,
        model_index_sha256: hash_file(&cli.model.join("model.safetensors.index.json"))?,
        native_mapping_sha256,
        placement_sha256,
        retained_trace_sha256: hash_file(&cli.retained_trace)?,
        layer_owner_pci_bus_id,
        token_id,
        position: 63,
        prior_kv_rows: cache.len,
        prior_kv_bytes: (cache.keys_bf16_bits.len() + cache.values_bf16_bits.len()) * 2,
        owner_shell_work_bytes: shell.owned_device_bytes(),
        owner_shell_host_staging_bytes: shell.owned_host_staging_bytes(),
        fault_feature_enabled: cfg!(feature = "heterogeneous-test-faults"),
        fault_exercise_requested,
        shell_faults_drained,
        shell_fault_retries_passed,
        owned_device_input_fault_drained,
        owned_device_input_retry_exact,
        owned_device_input_retry_d2d_bytes,
        owned_device_input_retry_host_bytes,
        owner_shell_kernel_elapsed_ms: [first.kernel_elapsed_ms, second.kernel_elapsed_ms],
        selected_experts: authority.selected_experts.clone(),
        routing_weights_bf16_bits: routed_weights,
        router_logits_sha256: hash_u16(&routed.router_logits_bf16_bits),
        router_input_device_handoff_bytes: first.router_input_bf16_bits.len() * size_of::<u16>(),
        router_input_execution_host_bytes: 0,
        router_evidence_activation_d2h_bytes: routed.source_d2h_bytes,
        router_descriptor_d2h_bytes: routed.descriptor_d2h_bytes,
        cpu_authority_contribution_h2d_bytes,
        exact_expert_output_sha256,
        moe_output_sha256: hash_u16(&reduced.output_bf16_bits),
        reducer_output_device_handoff_bytes: GPT_OSS_REDUCTION_OUTPUT_BYTES,
        reducer_output_execution_host_bytes: 0,
        reducer_evidence_d2h_bytes: GPT_OSS_REDUCTION_OUTPUT_BYTES + GPT_OSS_REDUCTION_TRACE_BYTES,
        layer_output_sha256: hash_u16(&layer_output),
        reduction_kernel_elapsed_ms: reduced.kernel_elapsed_ms,
        boundaries,
        owner_shell_prefix_repeat_exact: true,
        passed: true,
    };
    shell.drain()?;
    router.drain()?;
    model.drain()?;
    if let Some(h6b_evidence) = h6b_evidence {
        write_json(
            cli.h6b_output
                .as_deref()
                .context("--h6b-output is required with --exercise-three-owner")?,
            &h6b_evidence,
        )?;
    }
    write_json(&cli.output, &evidence)
}

const H6B_DRAIN_ROLES: [DrainRole; 6] = [
    DrainRole::LayerOwnerRouter,
    DrainRole::LayerOwnerExpert,
    DrainRole::LayerOwnerRelay,
    DrainRole::CpuExpert,
    DrainRole::RemoteGpuExpert,
    DrainRole::RankReduction,
];

#[allow(clippy::too_many_arguments)]
fn run_h6b_campaign(
    model: &mut OwnerSelectiveModel,
    shell: &mut CudaLayerOwnerShell,
    router: &mut CudaExactRouter,
    pools: &RelayPinnedPools,
    config: &CpuGptOssConfig,
    cache: &CpuKvCacheSnapshot,
    token_id: u32,
    prompt_token_ids: &[u32],
    authority: &CpuLayerTrace,
    expected_prefix: &ExpectedBoundaries,
    placement_sha256: String,
    layer_owner_pci_bus_id: String,
) -> Result<H6bEvidence> {
    const GPU_FREE_LOSS_TOLERANCE_BYTES: u64 = 8 * 1024 * 1024;
    let resources_before = h6b_resource_snapshot(model)?;
    let admission = cpu_oracle_admission(model, config, authority)?;
    let admission_plan = pack_routes_bounded(&admission, model.placement())?;
    if admission_plan.local_route_count() != 2
        || admission_plan.cpu_route_count() != 1
        || admission_plan.remote_gpu_route_count() != 1
    {
        bail!("H6b CPU oracle admission is not the required 2/1/1 owner split");
    }

    let mut coordinator = HeterogeneousTransactionCoordinator::new(16, 32, false)?;
    const COMMIT_SEQUENCE: u64 = 6_001;
    const DISCARD_SEQUENCE: u64 = 6_002;
    let placement_epoch = model.placement().placement_epoch();
    let prompt_len = u32::try_from(prompt_token_ids.len())?;
    coordinator.register_sequence(
        COMMIT_SEQUENCE,
        prompt_len,
        placement_epoch,
        prompt_token_ids.to_vec(),
    )?;
    coordinator.register_sequence(
        DISCARD_SEQUENCE,
        prompt_len,
        placement_epoch,
        prompt_token_ids.to_vec(),
    )?;

    let cases = [
        run_h6b_case(
            "committed",
            COMMIT_SEQUENCE,
            false,
            &mut coordinator,
            model,
            shell,
            router,
            pools,
            config,
            cache,
            token_id,
            prompt_token_ids,
            authority,
            expected_prefix,
            &admission,
            &admission_plan,
        )?,
        run_h6b_case(
            "cancelled_and_discarded",
            DISCARD_SEQUENCE,
            true,
            &mut coordinator,
            model,
            shell,
            router,
            pools,
            config,
            cache,
            token_id,
            prompt_token_ids,
            authority,
            expected_prefix,
            &admission,
            &admission_plan,
        )?,
        run_h6b_case(
            "clean_repeat_after_discard",
            DISCARD_SEQUENCE,
            false,
            &mut coordinator,
            model,
            shell,
            router,
            pools,
            config,
            cache,
            token_id,
            prompt_token_ids,
            authority,
            expected_prefix,
            &admission,
            &admission_plan,
        )?,
    ]
    .into_iter()
    .collect::<Vec<_>>();

    let committed_case_passed = cases[0].terminal.outcome == TransactionOutcome::Committed
        && cases[0].after_visibility.visibility_epoch
            == cases[0].before_visibility.visibility_epoch + 1;
    let drained_discard_passed = cases[1].terminal.outcome == TransactionOutcome::Discarded
        && cases[1].after_visibility == cases[1].before_visibility;
    let clean_repeat_passed = cases[2].terminal.outcome == TransactionOutcome::Committed
        && cases[2].after_visibility.visibility_epoch
            == cases[2].before_visibility.visibility_epoch + 1;
    if !committed_case_passed || !drained_discard_passed || !clean_repeat_passed {
        bail!("H6b commit/discard/clean-repeat transaction outcomes diverged");
    }

    let immediate = coordinator.begin_shutdown()?;
    let drained = coordinator.finish_shutdown()?;
    let coordinator_shutdown_zero =
        immediate.is_empty() && drained.is_empty() && coordinator.active_step_count() == 0;
    if !coordinator_shutdown_zero {
        bail!("H6b coordinator shutdown retained request-owned work");
    }
    coordinator.recycle_sequence(COMMIT_SEQUENCE)?;
    coordinator.recycle_sequence(DISCARD_SEQUENCE)?;
    let coordinator_free_blocks_after_recycle = coordinator.free_block_count();
    if coordinator_free_blocks_after_recycle != 32 {
        bail!("H6b coordinator did not reclaim all generation-tagged blocks");
    }
    let pool_stats = pools.stats();
    let pinned_pool_active_leases_after = pool_stats.source_activation.checked_out
        + pool_stats.route_descriptors.checked_out
        + pool_stats.remote_gpu_input.checked_out
        + pool_stats.remote_gpu_result.checked_out
        + pool_stats.cpu_result.checked_out;
    if pinned_pool_active_leases_after != 0 {
        bail!("H6b pinned relay pools retained checked-out storage");
    }
    let resources_after = h6b_resource_snapshot(model)?;
    if resources_after.gpu_total_bytes != resources_before.gpu_total_bytes {
        bail!("H6b CUDA total-memory identity changed during the campaign");
    }
    let swap_growth_bytes = resources_after
        .swap_used_bytes
        .saturating_sub(resources_before.swap_used_bytes);
    let gpu_free_loss_bytes = [
        resources_before.gpu_free_bytes[0].saturating_sub(resources_after.gpu_free_bytes[0]),
        resources_before.gpu_free_bytes[1].saturating_sub(resources_after.gpu_free_bytes[1]),
    ];
    let allocator_drift_gate_passed = resources_before.process_swap_used_bytes == 0
        && resources_after.process_swap_used_bytes == 0
        && swap_growth_bytes == 0
        && gpu_free_loss_bytes
            .iter()
            .all(|loss| *loss <= GPU_FREE_LOSS_TOLERANCE_BYTES);
    if !allocator_drift_gate_passed {
        bail!(
            "H6b resource gate failed: process_swap_before={} process_swap_after={} global_swap_before={} global_swap_after={} global_swap_growth={swap_growth_bytes} GPU free losses={gpu_free_loss_bytes:?}",
            resources_before.process_swap_used_bytes,
            resources_after.process_swap_used_bytes,
            resources_before.swap_used_bytes,
            resources_after.swap_used_bytes,
        );
    }

    Ok(H6bEvidence {
        schema: "gpt-oss-rs.heterogeneous-layer-oracle-h6b/v4",
        transaction_scope: "generation-tied coordinator metadata commit/discard; layer-owner flat K/V shell is private oracle storage",
        physical_kv_publication_claimed: false,
        route_authority: "CPU authority pre-reserves the exact 2/1/1 oracle admission; GPU0 native-BF16 router must author and exactly match every descriptor before expert dispatch",
        generation_authority: "HeterogeneousTransactionCoordinator step_id is the relay/result/reducer generation",
        placement_sha256,
        layer_owner_pci_bus_id,
        remote_worker_pci_bus_id: model
            .placement()
            .remote_worker()
            .stable_id
            .pci_bus_id
            .to_string(),
        resources_before,
        resources_after,
        swap_growth_bytes,
        gpu_free_loss_bytes,
        gpu_free_loss_tolerance_bytes: GPU_FREE_LOSS_TOLERANCE_BYTES,
        allocator_drift_gate_passed,
        cases,
        committed_case_passed,
        drained_discard_passed,
        clean_repeat_passed,
        coordinator_shutdown_zero,
        coordinator_free_blocks_after_recycle,
        pinned_pool_active_leases_after,
        passed: true,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_h6b_case(
    label: &'static str,
    sequence_id: u64,
    cancel_after_dispatch: bool,
    coordinator: &mut HeterogeneousTransactionCoordinator,
    model: &mut OwnerSelectiveModel,
    shell: &mut CudaLayerOwnerShell,
    router: &mut CudaExactRouter,
    pools: &RelayPinnedPools,
    config: &CpuGptOssConfig,
    cache: &CpuKvCacheSnapshot,
    token_id: u32,
    prompt_token_ids: &[u32],
    authority: &CpuLayerTrace,
    expected_prefix: &ExpectedBoundaries,
    admission: &GptOssRoutedBatchDescriptor,
    admission_plan: &gpt_oss_model_runner::heterogeneous::PackedDispatchPlan,
) -> Result<H6bCaseEvidence> {
    // Dense/attention creates GPU0's private router input. The transaction
    // proof starts at router admission; this shell cache is explicitly not the
    // coordinator's physical generation-tagged block store.
    let prefix = shell.execute_layer0_decode(model, config, token_id, 63, cache)?;
    let actual_prefix = ActualBoundaries::new(&prefix);
    for (name, expected) in &expected_prefix.values {
        let actual = actual_prefix
            .values
            .iter()
            .find(|(actual_name, _)| actual_name == name)
            .map(|(_, values)| *values)
            .context("H6b owner-shell prefix boundary missing")?;
        exact(&format!("H6b.{label}.{name}"), expected, actual)?;
    }

    // Build the entire exact reference before dispatch. These allocations are
    // oracle evidence only and cannot mask allocation in the target path.
    let expected_traces = admission
        .routes
        .iter()
        .map(|route| {
            exact_expert_trace(
                model.checkpoint(),
                admission.layer,
                route.expert_id,
                &admission.activation_bf16_bits,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut transaction =
        H6bTransactionGuard::reserve(coordinator, sequence_id, admission.placement_epoch)?;
    let step_id = transaction.step_id();
    let prepared_reduction =
        PreparedRankOrderedReduction::prepare(admission, model.placement(), step_id)?;
    let before_visibility = transaction
        .committed_view(sequence_id)
        .cloned()
        .context("H6b committed visibility missing before dispatch")?;
    let descriptors = prepared_reduction.expected_results().to_vec();
    let contributions = descriptors
        .iter()
        .cloned()
        .zip(expected_traces.iter())
        .map(|(descriptor, trace)| CanonicalExpertContribution {
            descriptor,
            output_bf16_bits: trace.down_bf16_bits.clone(),
        })
        .collect::<Vec<_>>();
    let exact_reduction =
        exact_rank_ordered_reduction_reference(admission, model.placement(), &contributions)?;
    let expected_layer_output = exact_residual(
        &prefix.post_attention_residual_bf16_bits,
        &exact_reduction.output_bf16_bits,
    )?;

    // Every bounded lease, device slot, trace, relay arena, and reducer buffer
    // is reserved from the coordinator generation before GPU0's router runs.
    let mut relay = CudaResultRelay::new(router, 1)?;
    let mut reducer = CudaRankOrderedReducer::new(&relay)?;
    let reservation = pools.try_reserve_all(step_id)?;
    let mut prepared = PreparedThreeOwnerDecode::prepare(
        model,
        admission_plan.clone(),
        reservation,
        prepared_reduction,
        &mut relay,
    )?;
    if prepared.generation() != step_id {
        bail!("H6b prepared generation differs from coordinator step ID");
    }
    transaction.mark_prepared()?;
    transaction.mark_dispatched()?;
    if cancel_after_dispatch && transaction.cancel()?.is_some() {
        bail!("H6b dispatched cancellation finalized before mandatory drains");
    }

    let timeline = CorrelatedTimeline::new();
    let routed_and_plan = (|| -> Result<_> {
        let routed = {
            let (source_activation, route_descriptors) = prepared.router_download_buffers()?;
            shell.route_resident_decode(
                router,
                admission.layer,
                admission.placement_epoch,
                source_activation,
                route_descriptors,
                Some(&timeline),
            )?
        };
        exact(
            &format!("H6b.{label}.router_input"),
            &admission.activation_bf16_bits,
            &routed.batch.activation_bf16_bits,
        )?;
        exact(
            &format!("H6b.{label}.router_logits"),
            &bits(&authority.router_logits),
            &routed.router_logits_bf16_bits,
        )?;
        let actual_plan = pack_routes_bounded(&routed.batch, model.placement())?;
        prepared.admit_gpu_route(&actual_plan)?;
        transaction.mark_terminal(DrainRole::LayerOwnerRouter)?;
        Ok((routed, actual_plan))
    })();
    let (routed, actual_plan) = match routed_and_plan {
        Ok(result) => result,
        Err(primary) => {
            let failure = prepared.abandon_before_expert_dispatch(
                model,
                shell,
                router,
                &mut relay,
                &mut reducer,
                LLMError::ModelError(primary.to_string()),
            );
            let (error, drain_proven) = failure.into_parts();
            if drain_proven {
                transaction.prove_drained();
            }
            return Err(anyhow::anyhow!(error));
        }
    };

    let execution = match prepared.execute(
        model,
        shell,
        &mut relay,
        &mut reducer,
        &timeline,
        |terminal| {
            if terminal == ThreeOwnerTerminal::RankReduction {
                // The coordinator's reduction obligation includes the
                // resident residual/result terminal. Do not expose an
                // all-terminal step while that CUDA work remains untracked.
                Ok(())
            } else {
                transaction.mark_terminal(drain_role(terminal))
            }
        },
    ) {
        Ok(execution) => execution,
        Err(failure) => {
            let (error, drain_proven) = failure.into_parts();
            if drain_proven {
                transaction.prove_drained();
            }
            return Err(anyhow::anyhow!(error));
        }
    };
    // Expert/relay/reduction work is terminal at this boundary. Its pinned
    // reservation is not an input to the resident residual and can be returned
    // before that separately fallible CUDA edge begins.
    execution.reservation.release_drained()?;
    let layer_output = match shell.finish_layer_residual_resident(&reducer) {
        Ok(output) => output,
        Err(primary) => {
            let model_drain = model.drain();
            let shell_drain = shell.drain();
            let router_drain = router.drain();
            let reducer_drain = reducer.prove_transaction_drain();
            let relay_drain = relay.prove_transaction_drain();
            let drain_proven = model_drain.is_ok()
                && shell_drain.is_ok()
                && router_drain.is_ok()
                && reducer_drain.is_ok()
                && relay_drain.is_ok();
            if drain_proven {
                transaction.prove_drained();
            } else {
                quarantine_three_owner_after_unproven_drain(model, shell);
            }
            return Err(anyhow::anyhow!(
                "H6b resident residual failed ({primary}); drain_proven={drain_proven} model={model_drain:?} shell={shell_drain:?} router={router_drain:?} reducer={reducer_drain:?} relay={relay_drain:?}"
            ));
        }
    };
    // Every target CUDA stream has returned through a terminal/fallback drain,
    // and no pinned reservation remains live. Exactness/evidence failures from
    // here are ordinary postdispatch errors that the guard can safely discard.
    transaction.prove_drained();
    exact(
        &format!("H6b.{label}.rank_reduction"),
        &exact_reduction.output_bf16_bits,
        &execution.reduction.output_bf16_bits,
    )?;
    exact(
        &format!("H6b.{label}.layer_output"),
        &expected_layer_output,
        &layer_output,
    )?;
    transaction.mark_terminal(DrainRole::RankReduction)?;

    if execution.generation != step_id
        || execution.descriptors != descriptors
        || execution.local_activation_d2d_bytes != 2 * config.hidden_size * size_of::<u16>()
        || execution.local_activation_host_execution_bytes != 0
        || execution.remote_activation_h2d_bytes != config.hidden_size * size_of::<u16>()
        || execution.remote_result_d2h_bytes != config.hidden_size * size_of::<u16>()
        || execution.local_result_d2d_bytes != 2 * config.hidden_size * size_of::<u16>()
        || execution.relay_execution.cpu_h2d_bytes != config.hidden_size * size_of::<u16>()
        || execution.relay_execution.remote_gpu_h2d_bytes != config.hidden_size * size_of::<u16>()
    {
        bail!("H6b generation or target-path byte accounting diverged");
    }
    let mut packed_admission_descriptors = Vec::with_capacity(4);
    for route in admission_plan.all_routes() {
        packed_admission_descriptors.push(H6bPackedRouteEvidence {
            identity: route_identity(
                CanonicalRouteContract::from_packed_route(&route.descriptor),
                &route.descriptor.owner,
            ),
            relay_activation_slot: route.relay_activation_slot,
            owner_route_slot: route.owner_route_slot,
        });
    }
    let mut completion_identities = Vec::with_capacity(4);
    for (slot, (contract, descriptor)) in execution
        .completion_contracts
        .iter()
        .copied()
        .zip(&descriptors)
        .enumerate()
    {
        let packed = actual_plan
            .all_routes()
            .find(|route| route.descriptor.canonical_result_slot as usize == slot)
            .with_context(|| format!("H6b packed route missing canonical slot {slot}"))?;
        let expected_contract = CanonicalRouteContract::from_packed_route(&packed.descriptor);
        if contract != expected_contract {
            bail!("H6b completion contract changed packed identity at slot {slot}");
        }
        contract.validate_result(descriptor).map_err(|error| {
            anyhow::anyhow!("H6b result descriptor mismatch at slot {slot}: {error}")
        })?;
        completion_identities.push(route_identity(contract, &descriptor.owner));
    }
    let completion_contracts_exact = true;
    let mut expert_boundaries = Vec::with_capacity(4);
    for (rank, ((descriptor, expected), actual)) in descriptors
        .iter()
        .zip(&expected_traces)
        .zip(&execution.expert_traces)
        .enumerate()
    {
        exact_trace(&format!("H6b.{label}.expert_rank_{rank}"), expected, actual)?;
        expert_boundaries.push(H6bExpertEvidence {
            route_rank: descriptor.route_rank,
            expert_id: descriptor.expert_id,
            owner: descriptor.owner.role_name().to_owned(),
            gate_up_sha256: hash_u16(&actual.gate_up_bf16_bits),
            scaled_gate_sha256: hash_u16(&actual.scaled_gate_bf16_bits),
            sigmoid_sha256: hash_u16(&actual.sigmoid_bf16_bits),
            glu_sha256: hash_u16(&actual.glu_bf16_bits),
            linear_sha256: hash_u16(&actual.linear_bf16_bits),
            swiglu_sha256: hash_u16(&actual.swiglu_bf16_bits),
            down_sha256: hash_u16(&actual.down_bf16_bits),
            bit_exact: true,
        });
    }

    let points = timeline.points();
    let compute_intervals_ns = [
        timeline_interval(&points, "cpu_expert", "compute_begin", "compute_end")?,
        timeline_interval(
            &points,
            "gpu0_local_expert_rank0",
            "compute_begin",
            "compute_end",
        )?,
        timeline_interval(&points, "gpu1_expert_rank2", "compute_begin", "compute_end")?,
    ];
    let intersection_begin = compute_intervals_ns
        .iter()
        .map(|interval| interval[0])
        .max()
        .expect("three H6b compute intervals");
    let intersection_end = compute_intervals_ns
        .iter()
        .map(|interval| interval[1])
        .min()
        .expect("three H6b compute intervals");
    let cpu_gpu0_gpu1_compute_overlap = intersection_begin < intersection_end;
    if !cpu_gpu0_gpu1_compute_overlap {
        bail!(
            "H6b {label} has no strict three-way CPU/GPU0/GPU1 compute intersection: {:?}",
            compute_intervals_ns
        );
    }

    let terminal = if cancel_after_dispatch {
        transaction.finalize_discard()?
    } else {
        transaction.mark_reduced(&execution.reduction.output_bf16_bits)?;
        let mut committed_tokens = Vec::with_capacity(prompt_token_ids.len() + 1);
        committed_tokens.extend_from_slice(prompt_token_ids);
        committed_tokens.push(token_id);
        transaction.prepare_commit(SequenceCommitImage {
            next_revision: before_visibility.request_revision + 1,
            token_ids: committed_tokens,
            output_image: u16_le_bytes(&layer_output),
            evidence_image: hash_u16(&execution.reduction.output_bf16_bits).into_bytes(),
        })?;
        transaction.commit()?
    };
    let after_visibility = transaction
        .committed_view(sequence_id)
        .cloned()
        .context("H6b committed visibility missing after terminal outcome")?;
    let active_steps_after = transaction.active_step_count();
    if terminal.drained_roles.len() != H6B_DRAIN_ROLES.len()
        || H6B_DRAIN_ROLES
            .iter()
            .any(|role| !terminal.drained_roles.contains(role))
        || active_steps_after != 0
    {
        bail!("H6b terminal record omitted a mandatory drain role");
    }
    if cancel_after_dispatch {
        if terminal.outcome != TransactionOutcome::Discarded
            || after_visibility != before_visibility
        {
            bail!("H6b cancelled step changed committed visibility");
        }
    } else {
        let expected_length = before_visibility
            .committed_length
            .checked_add(1)
            .context("H6b committed length overflow")?;
        if terminal.outcome != TransactionOutcome::Committed
            || after_visibility.committed_length != expected_length
            || after_visibility.request_revision != before_visibility.request_revision + 1
            || after_visibility.visibility_epoch != before_visibility.visibility_epoch + 1
            || after_visibility.token_ids.last() != Some(&token_id)
            || after_visibility.output_image != u16_le_bytes(&layer_output)
        {
            bail!("H6b committed step publication image or epoch changed unexpectedly");
        }
    }

    Ok(H6bCaseEvidence {
        label,
        step_id,
        transaction_generation: execution.generation,
        oracle_admission: "CPU route/weights reserve exact resources; GPU0-authored canonical descriptors must match before expert dispatch",
        gpu_route_matched_admission: actual_plan == *admission_plan,
        packed_admission_descriptors,
        completion_identities,
        completion_contracts_exact,
        selected_experts: routed.batch.routes.iter().map(|route| route.expert_id).collect(),
        selected_weight_bf16_bits: routed
            .batch
            .routes
            .iter()
            .map(|route| route.weight_bf16_bits)
            .collect(),
        owner_counts: [2, 1, 1],
        local_activation_d2d_bytes: execution.local_activation_d2d_bytes,
        local_activation_host_execution_bytes: execution.local_activation_host_execution_bytes,
        remote_activation_h2d_bytes: execution.remote_activation_h2d_bytes,
        remote_result_d2h_bytes: execution.remote_result_d2h_bytes,
        cpu_result_h2d_bytes: execution.relay_execution.cpu_h2d_bytes,
        remote_result_h2d_bytes: execution.relay_execution.remote_gpu_h2d_bytes,
        local_result_d2d_bytes: execution.local_result_d2d_bytes,
        canonical_evidence_d2h_bytes: execution.canonical_evidence_d2h_bytes,
        cpu_scratch_bytes: execution.cpu_scratch_bytes,
        cpu_high_water_jobs: execution.cpu_high_water_jobs,
        local_kernel_elapsed_ms: execution.local_kernel_elapsed_ms,
        remote_kernel_elapsed_ms: execution.remote_kernel_elapsed_ms,
        cpu_elapsed_ns: execution.cpu_elapsed_ns,
        expert_boundaries,
        reduction_output_sha256: hash_u16(&execution.reduction.output_bf16_bits),
        layer_output_sha256: hash_u16(&layer_output),
        reduction_bit_exact: true,
        layer_output_bit_exact: true,
        cpu_gpu0_gpu1_compute_overlap,
        compute_intervals_ns,
        timeline: points,
        before_visibility,
        after_visibility,
        terminal,
        active_steps_after,
        free_blocks_after: transaction.free_block_count(),
    })
}

fn cpu_oracle_admission(
    model: &OwnerSelectiveModel,
    config: &CpuGptOssConfig,
    authority: &CpuLayerTrace,
) -> Result<GptOssRoutedBatchDescriptor> {
    let weights = bits(&authority.routing_weights);
    let routes = authority
        .selected_experts
        .iter()
        .zip(weights)
        .enumerate()
        .map(
            |(rank, (&expert, weight_bf16_bits))| GptOssRouteDescriptor {
                source_row: 0,
                route_rank: rank as u8,
                expert_id: u16::try_from(expert).expect("GPT-OSS expert ID is u16"),
                weight_bf16_bits,
                activation_slot: 0,
            },
        )
        .collect();
    let batch = GptOssRoutedBatchDescriptor {
        layer: 0,
        phase: GptOssPhase::Decode,
        rows: 1,
        hidden_size: u16::try_from(config.hidden_size)?,
        experts_per_layer: u16::try_from(config.num_local_experts)?,
        placement_epoch: model.placement().placement_epoch(),
        activation_bf16_bits: bits(&authority.router_input),
        routes,
    };
    batch
        .validate()
        .map_err(|error| anyhow::anyhow!("invalid CPU oracle admission: {error}"))?;
    Ok(batch)
}

fn route_identity(
    contract: CanonicalRouteContract,
    owner: &ExpertOwner,
) -> H6bRouteIdentityEvidence {
    H6bRouteIdentityEvidence {
        source_row: contract.source_row,
        activation_slot: contract.activation_slot,
        source_activation_slot: contract.source_activation_slot,
        route_rank: contract.route_rank,
        expert_id: contract.expert_id,
        weight_bf16_bits: contract.weight_bf16_bits,
        owner: owner.clone(),
        owner_role: owner.role_name().to_owned(),
        placement_epoch: contract.placement_epoch,
        canonical_result_slot: contract.result_slot,
    }
}

fn h6b_resource_snapshot(model: &OwnerSelectiveModel) -> Result<H6bResourceSnapshot> {
    let memory = model.device_memory_info()?;
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    let mut swap_total = None;
    let mut swap_free = None;
    let mut swap_cached = None;
    for line in meminfo.lines() {
        let mut fields = line.split_whitespace();
        match fields.next() {
            Some("SwapTotal:") => {
                swap_total = fields.next().and_then(|value| value.parse::<u64>().ok())
            }
            Some("SwapFree:") => {
                swap_free = fields.next().and_then(|value| value.parse::<u64>().ok())
            }
            Some("SwapCached:") => {
                swap_cached = fields.next().and_then(|value| value.parse::<u64>().ok())
            }
            _ => {}
        }
    }
    let swap_total = swap_total.context("/proc/meminfo omitted SwapTotal")? * 1024;
    let swap_free = swap_free.context("/proc/meminfo omitted SwapFree")? * 1024;
    let swap_cached = swap_cached.context("/proc/meminfo omitted SwapCached")? * 1024;
    let status = std::fs::read_to_string("/proc/self/status")?;
    let process_swap_used_bytes = status
        .lines()
        .find_map(|line| {
            let mut fields = line.split_whitespace();
            (fields.next() == Some("VmSwap:"))
                .then(|| fields.next()?.parse::<u64>().ok())
                .flatten()
        })
        .context("/proc/self/status omitted VmSwap")?
        * 1024;
    let captured_unix_seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    Ok(H6bResourceSnapshot {
        captured_unix_seconds,
        process_swap_used_bytes,
        swap_used_bytes: swap_total.saturating_sub(swap_free),
        swap_cached_bytes: swap_cached,
        gpu_free_bytes: [u64::try_from(memory[0].0)?, u64::try_from(memory[1].0)?],
        gpu_total_bytes: [u64::try_from(memory[0].1)?, u64::try_from(memory[1].1)?],
    })
}

const fn drain_role(terminal: ThreeOwnerTerminal) -> DrainRole {
    match terminal {
        ThreeOwnerTerminal::CpuExpert => DrainRole::CpuExpert,
        ThreeOwnerTerminal::LayerOwnerExpert => DrainRole::LayerOwnerExpert,
        ThreeOwnerTerminal::RemoteGpuExpert => DrainRole::RemoteGpuExpert,
        ThreeOwnerTerminal::LayerOwnerRelay => DrainRole::LayerOwnerRelay,
        ThreeOwnerTerminal::RankReduction => DrainRole::RankReduction,
    }
}

fn timeline_interval(
    points: &[TimelinePoint],
    actor: &str,
    begin: &str,
    end: &str,
) -> Result<[u64; 2]> {
    let begin = points
        .iter()
        .find(|point| point.actor == actor && point.label == begin)
        .map(|point| point.monotonic_ns)
        .with_context(|| format!("timeline missing {actor} begin"))?;
    let end = points
        .iter()
        .find(|point| point.actor == actor && point.label == end)
        .map(|point| point.monotonic_ns)
        .with_context(|| format!("timeline missing {actor} end"))?;
    if begin >= end {
        bail!("timeline interval {actor} is empty or reversed");
    }
    Ok([begin, end])
}

fn u16_le_bytes(values: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::H6bTransactionGuard;
    use gpt_oss_engine::HeterogeneousTransactionCoordinator;

    #[test]
    fn h6b_predispatch_guard_releases_reserved_step_and_blocks() {
        let mut coordinator = HeterogeneousTransactionCoordinator::new(16, 4, false).unwrap();
        coordinator.register_sequence(41, 1, 7, vec![123]).unwrap();
        let free_before = coordinator.free_block_count();
        let before = coordinator.committed_view(41).cloned().unwrap();
        {
            let guard = H6bTransactionGuard::reserve(&mut coordinator, 41, 7).unwrap();
            assert_eq!(guard.active_step_count(), 1);
            // Simulate any preparation failure after reserve by dropping the
            // guard before Prepared/Dispatched.
        }
        assert_eq!(coordinator.active_step_count(), 0);
        assert_eq!(coordinator.free_block_count(), free_before);
        assert_eq!(coordinator.committed_view(41), Some(&before));
    }

    #[test]
    fn h6b_recoverable_postdispatch_guard_drains_and_discards() {
        let mut coordinator = HeterogeneousTransactionCoordinator::new(16, 4, false).unwrap();
        coordinator.register_sequence(42, 1, 9, vec![456]).unwrap();
        let free_before = coordinator.free_block_count();
        let before = coordinator.committed_view(42).cloned().unwrap();
        {
            let mut guard = H6bTransactionGuard::reserve(&mut coordinator, 42, 9).unwrap();
            guard.mark_prepared().unwrap();
            guard.mark_dispatched().unwrap();
            // Models an explicit PreparedThreeOwnerDecode failure classified
            // drain_proven=true after its model/shell/reducer/relay cleanup.
            guard.prove_drained();
        }
        assert_eq!(coordinator.active_step_count(), 0);
        assert_eq!(coordinator.free_block_count(), free_before);
        assert_eq!(coordinator.committed_view(42), Some(&before));
    }
}

struct ExpectedBoundaries {
    values: Vec<(&'static str, Vec<u16>)>,
}

impl ExpectedBoundaries {
    fn new(trace: &CpuLayerTrace, hidden: Vec<u16>) -> Self {
        Self {
            values: vec![
                ("hidden", hidden),
                ("input_norm", bits(&trace.input_norm)),
                ("query_after_rope", bits(&trace.query_after_rope)),
                ("key_after_rope", bits(&trace.key_after_rope)),
                ("value_projection", bits(&trace.value_projection)),
                ("attention_context", bits(&trace.attention_context)),
                ("attention_projection", bits(&trace.attention_projection)),
                (
                    "post_attention_residual",
                    bits(&trace.post_attention_residual),
                ),
                ("router_input", bits(&trace.router_input)),
            ],
        }
    }
}

struct ActualBoundaries<'a> {
    values: Vec<(&'static str, &'a [u16])>,
}

impl<'a> ActualBoundaries<'a> {
    fn new(execution: &'a gpt_oss_model_runner::heterogeneous::LayerOwnerShellExecution) -> Self {
        Self {
            values: vec![
                ("hidden", &execution.hidden_bf16_bits),
                ("input_norm", &execution.input_norm_bf16_bits),
                ("query_after_rope", &execution.query_after_rope_bf16_bits),
                ("key_after_rope", &execution.key_after_rope_bf16_bits),
                ("value_projection", &execution.value_projection_bf16_bits),
                ("attention_context", &execution.attention_context_bf16_bits),
                (
                    "attention_projection",
                    &execution.attention_projection_bf16_bits,
                ),
                (
                    "post_attention_residual",
                    &execution.post_attention_residual_bf16_bits,
                ),
                ("router_input", &execution.router_input_bf16_bits),
            ],
        }
    }
}

fn bits(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .copied()
        .map(bf16::from_f32)
        .map(bf16::to_bits)
        .collect()
}

fn exact(label: &str, expected: &[u16], actual: &[u16]) -> Result<()> {
    if expected.len() != actual.len() {
        bail!(
            "first divergence at {label}: expected length {}, actual {}",
            expected.len(),
            actual.len()
        );
    }
    if let Some((index, (&expected, &actual))) = expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (expected, actual))| expected != actual)
    {
        bail!(
            "first divergence at {label}[{index}]: expected bits=0x{expected:04x} value={} actual bits=0x{actual:04x} value={}",
            bf16::from_bits(expected).to_f32(),
            bf16::from_bits(actual).to_f32()
        );
    }
    Ok(())
}

fn exact_expert_output(
    checkpoint: &GptOssCheckpointView,
    layer: u16,
    expert: u16,
    input_bf16_bits: &[u16],
) -> Result<Vec<u16>> {
    Ok(exact_expert_trace(checkpoint, layer, expert, input_bf16_bits)?.down_bf16_bits)
}

fn exact_expert_trace(
    checkpoint: &GptOssCheckpointView,
    layer: u16,
    expert: u16,
    input_bf16_bits: &[u16],
) -> Result<gpt_oss_model_runner::heterogeneous::SelectedExpertFirstDivergenceTrace> {
    let prefix = format!("model.layers.{layer}.mlp.experts");
    let gate_blocks = checkpoint.tensor(&format!("{prefix}.gate_up_proj_blocks"))?;
    let gate_scales = checkpoint.tensor(&format!("{prefix}.gate_up_proj_scales"))?;
    let gate_bias = checkpoint.tensor(&format!("{prefix}.gate_up_proj_bias"))?;
    let down_blocks = checkpoint.tensor(&format!("{prefix}.down_proj_blocks"))?;
    let down_scales = checkpoint.tensor(&format!("{prefix}.down_proj_scales"))?;
    let down_bias = checkpoint.tensor(&format!("{prefix}.down_proj_bias"))?;
    let expert = usize::from(expert);
    let gate_bias_bf16_bits = bytes_to_u16(expert_slice(
        gate_bias.bytes(),
        expert,
        GATE_UP_BIAS_VALUES * size_of::<u16>(),
    ))?;
    let down_bias_bf16_bits = bytes_to_u16(expert_slice(
        down_bias.bytes(),
        expert,
        DOWN_BIAS_VALUES * size_of::<u16>(),
    ))?;
    let gate_up_blocks = expert_slice(gate_blocks.bytes(), expert, GATE_UP_BLOCK_BYTES);
    let gate_up_scales = expert_slice(gate_scales.bytes(), expert, GATE_UP_SCALE_BYTES);
    let gate_up_bias_bytes = expert_slice(
        gate_bias.bytes(),
        expert,
        GATE_UP_BIAS_VALUES * size_of::<u16>(),
    );
    let down_projection_blocks = expert_slice(down_blocks.bytes(), expert, DOWN_BLOCK_BYTES);
    let down_projection_scales = expert_slice(down_scales.bytes(), expert, DOWN_SCALE_BYTES);
    let down_projection_bias_bytes = expert_slice(
        down_bias.bytes(),
        expert,
        DOWN_BIAS_VALUES * size_of::<u16>(),
    );
    let identity_sha256 = hash_surfaces(&[
        gate_up_blocks,
        gate_up_scales,
        gate_up_bias_bytes,
        down_projection_blocks,
        down_projection_scales,
        down_projection_bias_bytes,
    ]);
    let source = NativeMxfp4ExpertView {
        key: GptOssExpertKey {
            layer,
            expert: expert as u16,
        },
        gate_up_blocks,
        gate_up_scales,
        gate_up_bias_bf16_bits: &gate_bias_bf16_bits,
        down_blocks: down_projection_blocks,
        down_scales: down_projection_scales,
        down_bias_bf16_bits: &down_bias_bf16_bits,
        identity_sha256: &identity_sha256,
    };
    Ok(exact_selected_expert_reference(source, input_bf16_bits)?)
}

fn exact_trace(
    label: &str,
    expected: &SelectedExpertFirstDivergenceTrace,
    actual: &SelectedExpertFirstDivergenceTrace,
) -> Result<()> {
    for (boundary, expected, actual) in [
        (
            "gate_up",
            expected.gate_up_bf16_bits.as_slice(),
            actual.gate_up_bf16_bits.as_slice(),
        ),
        (
            "scaled_gate",
            expected.scaled_gate_bf16_bits.as_slice(),
            actual.scaled_gate_bf16_bits.as_slice(),
        ),
        (
            "sigmoid",
            expected.sigmoid_bf16_bits.as_slice(),
            actual.sigmoid_bf16_bits.as_slice(),
        ),
        (
            "glu",
            expected.glu_bf16_bits.as_slice(),
            actual.glu_bf16_bits.as_slice(),
        ),
        (
            "linear",
            expected.linear_bf16_bits.as_slice(),
            actual.linear_bf16_bits.as_slice(),
        ),
        (
            "swiglu",
            expected.swiglu_bf16_bits.as_slice(),
            actual.swiglu_bf16_bits.as_slice(),
        ),
        (
            "down",
            expected.down_bf16_bits.as_slice(),
            actual.down_bf16_bits.as_slice(),
        ),
    ] {
        exact(&format!("{label}.{boundary}"), expected, actual)?;
    }
    Ok(())
}

fn exact_residual(residual: &[u16], update: &[u16]) -> Result<Vec<u16>> {
    if residual.len() != update.len() {
        bail!(
            "exact residual shape mismatch: residual={} update={}",
            residual.len(),
            update.len()
        );
    }
    Ok(residual
        .iter()
        .zip(update)
        .map(|(&residual, &update)| {
            bf16::from_f32(bf16::from_bits(residual).to_f32() + bf16::from_bits(update).to_f32())
                .to_bits()
        })
        .collect())
}

fn expert_slice<T>(values: &[T], expert: usize, stride: usize) -> &[T] {
    &values[expert * stride..(expert + 1) * stride]
}

fn bytes_to_u16(bytes: &[u8]) -> Result<Vec<u16>> {
    if !bytes.len().is_multiple_of(size_of::<u16>()) {
        bail!("BF16 byte extent is not u16-aligned");
    }
    Ok(bytes
        .chunks_exact(size_of::<u16>())
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect())
}

fn hash_u16(values: &[u16]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn hash_surfaces(surfaces: &[&[u8]]) -> String {
    let mut digest = Sha256::new();
    for surface in surfaces {
        digest.update(surface);
    }
    format!("{:x}", digest.finalize())
}

fn hash_file(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
}

fn tensor_u16(
    checkpoint: &GptOssCheckpointView,
    name: &str,
    expected_values: usize,
) -> Result<Vec<u16>> {
    let tensor = checkpoint.tensor(name)?;
    if tensor.bytes().len() != expected_values * size_of::<u16>() {
        bail!(
            "native tensor {name} has {} BF16 values, expected {expected_values}",
            tensor.bytes().len() / size_of::<u16>()
        );
    }
    Ok(tensor
        .bytes()
        .chunks_exact(size_of::<u16>())
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect())
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    std::fs::write(path, bytes)?;
    Ok(())
}
