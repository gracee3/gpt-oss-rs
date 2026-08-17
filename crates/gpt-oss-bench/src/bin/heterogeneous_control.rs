use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use gpt_oss_bench::r2_release_handshake::{
    child_release_handshake, ChildReleaseHandshake, ReleaseProof, ReleaseReadyMarker,
};
use gpt_oss_core::error::LLMError;
use gpt_oss_engine::{
    DrainRole, HeterogeneousTransactionCoordinator, SequenceCommitImage, TransactionOutcome,
};
use gpt_oss_gpu::device::StableCudaDeviceId;
use gpt_oss_gpu::event::{CorrelatedTimeline, TimelinePoint};
use gpt_oss_gpu::pinned_memory::BoundedPinnedPoolStats;
use gpt_oss_model_runner::heterogeneous::{
    ExpertOwner, ExpertResultDescriptor, HeterogeneousControlLayerExecution,
    HeterogeneousControlRuntime, PackedRouteDescriptor, RelayPinnedPoolStats,
};
use gpt_oss_model_runner::model_loader::capacity_one::CAPACITY_ONE_POLICY_SHA256;
use gpt_oss_model_runner::model_loader::gpt_oss_native::GptOssCheckpointReleaseEvidence;
use gpt_oss_model_runner::model_loader::gpt_oss_native::GptOssCheckpointView;
use gpt_oss_model_runner::model_loader::gpt_oss_native::GptOssNativeCatalogMap;
use gpt_oss_model_runner::model_loader::owner_selective::{
    CapacityOneConstructionEvidence, ConstructionLedger, OwnerSelectiveConstructor,
    OwnerSelectiveEnvelope, OwnerSelectiveModel,
};
use gpt_oss_model_runner::model_loader::shard_catalog::SafeTensorShardCatalog;
use gpt_oss_model_runner::{cpu_repack::CpuOwnerRecordReleaseTelemetry, CpuGptOssConfig};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const EXPECTED: [u32; 8] = [200_005, 35_644, 200_008, 976, 1_825, 5_003, 25, 392];
const ROLES: [DrainRole; 6] = [
    DrainRole::LayerOwnerRouter,
    DrainRole::LayerOwnerExpert,
    DrainRole::LayerOwnerRelay,
    DrainRole::CpuExpert,
    DrainRole::RemoteGpuExpert,
    DrainRole::RankReduction,
];

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ConstructorMode {
    MonolithicControl,
    CapacityOne,
}

impl ConstructorMode {
    const fn release_name(self) -> &'static str {
        match self {
            Self::MonolithicControl => "monolithic-control",
            Self::CapacityOne => "capacity-one",
        }
    }
}

#[derive(Parser)]
struct Cli {
    #[arg(long, value_enum, default_value = "monolithic-control")]
    constructor: ConstructorMode,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    native_model: PathBuf,
    #[arg(long)]
    owner_cache: PathBuf,
    #[arg(long)]
    placement: PathBuf,
    #[arg(long)]
    retained_trace: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 8)]
    max_new_tokens: usize,
    #[arg(long)]
    max_input_tokens: Option<usize>,
    #[arg(long, default_value_t = 1)]
    repeat: usize,
    #[arg(long)]
    release_handshake_root: Option<PathBuf>,
    #[arg(long)]
    release_handshake_nonce: Option<String>,
    #[arg(long)]
    release_handshake_cell: Option<String>,
}

#[derive(Deserialize)]
struct RetainedControl {
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
}

#[derive(Serialize)]
struct Evidence {
    schema: &'static str,
    execution_path: &'static str,
    cuda_prefill_or_all_expert_fallback_used: bool,
    tensor_parallel_or_nccl_used: bool,
    peer_access_used: bool,
    decode_expert_weight_transfer_bytes: u64,
    constructor: ConstructorMode,
    r2_policy_sha256: &'static str,
    binary_sha256: String,
    placement_file_sha256: String,
    retained_trace_sha256: String,
    prompt_tokens_requested: usize,
    expected_token_ids: Vec<u32>,
    repeat_requested: usize,
    runs: Vec<RunEvidence>,
    all_runs_passed: bool,
}

#[derive(Serialize)]
struct RunEvidence {
    run_index: usize,
    load_class: &'static str,
    model_identity: ModelIdentityEvidence,
    placement: PlacementEvidence,
    envelope: OwnerSelectiveEnvelope,
    final_ledger: ConstructionLedger,
    capacity_one: Option<CapacityOneConstructionEvidence>,
    checkpoint_release: Option<GptOssCheckpointReleaseEvidence>,
    source_release_handshake: Option<ReleaseReadyMarker>,
    cpu_record_releases: Vec<CpuOwnerRecordReleaseTelemetry>,
    construction_stages: Vec<ConstructionStageEvidence>,
    resources_before: ResourceSnapshot,
    resources_after_construct: ResourceSnapshot,
    resources_after_runtime: ResourceSnapshot,
    resources_after_execution: ResourceSnapshot,
    resources_before_drop: ResourceSnapshot,
    resources_after_drop: ResourceSnapshot,
    cancellation_retry: CancellationRetryEvidence,
    committed_steps: Vec<StepCommitEvidence>,
    prompt_tokens_committed: usize,
    generated_token_ids: Vec<u32>,
    exact_retained_tokens: bool,
    committed_length: u32,
    request_revision: u64,
    visibility_epoch: u64,
    active_steps_after: usize,
    cpu_worker_high_water_jobs: usize,
    pinned_pool: PinnedPoolEvidence,
    pinned_raw_capacity_bytes: usize,
    pinned_checked_out_after: usize,
    pinned_quarantined_after: u64,
    captured_layer: Option<CapturedLayerEvidence>,
    memory_gate_passed: bool,
    cleanup_gate_passed: bool,
    epoch_gate_passed: bool,
    passed: bool,
}

#[derive(Serialize)]
struct ModelIdentityEvidence {
    revision: String,
    config_sha256: String,
    index_sha256: String,
    mapping_sha256: String,
}

#[derive(Serialize)]
struct PlacementEvidence {
    manifest_hash: String,
    placement_epoch: u64,
    layer_owner: StableCudaDeviceId,
    layer_owner_ordinal: usize,
    remote_worker: StableCudaDeviceId,
    remote_worker_ordinal: usize,
    cpu_experts: u32,
    layer_owner_experts: u32,
    remote_gpu_experts: u32,
}

#[derive(Serialize)]
struct ConstructionStageEvidence {
    ledger: ConstructionLedger,
    resources: ResourceSnapshot,
}

#[derive(Debug, Clone, Serialize)]
struct ResourceSnapshot {
    unix_time_ms: u128,
    process_rss_bytes: u64,
    process_high_water_bytes: u64,
    process_swap_used_bytes: u64,
    mem_available_bytes: u64,
    global_swap_used_bytes: u64,
    swap_cached_bytes: u64,
    gpus: Vec<GpuMemorySnapshot>,
}

#[derive(Debug, Clone, Serialize)]
struct GpuMemorySnapshot {
    pci_bus_id: String,
    total_bytes: u64,
    free_bytes: u64,
    used_bytes: u64,
}

#[derive(Serialize)]
struct CancellationRetryEvidence {
    discarded_step: u64,
    discarded_token: u32,
    discarded_prediction: u32,
    discarded_logits_sha256: String,
    committed_length_before: u32,
    committed_length_after: u32,
    visibility_epoch_before: u64,
    visibility_epoch_after: u64,
    runtime_committed_tokens_after: usize,
    active_steps_after: usize,
    pinned_checked_out_after: usize,
    pinned_quarantined_after: u64,
    clean_retry_identity_matched: bool,
}

#[derive(Serialize)]
struct StepCommitEvidence {
    step_id: u64,
    input_token: u32,
    output_token: u32,
    output_logits_sha256: String,
    prepared_hidden_sha256: String,
    committed_length: u32,
    request_revision: u64,
    visibility_epoch: u64,
    runtime_committed_tokens: usize,
}

#[derive(Serialize)]
struct PinnedPoolEvidence {
    source_activation: PinnedLeaseClassEvidence,
    route_descriptors: PinnedLeaseClassEvidence,
    remote_gpu_input: PinnedLeaseClassEvidence,
    remote_gpu_result: PinnedLeaseClassEvidence,
    cpu_result: PinnedLeaseClassEvidence,
    raw_capacity_bytes: usize,
    hard_cap_bytes: usize,
}

#[derive(Serialize)]
struct PinnedLeaseClassEvidence {
    capacity: usize,
    available: usize,
    checked_out: usize,
    high_water: usize,
    fixed_allocations: usize,
    exhaustions: u64,
    quarantined: u64,
    bytes_per_buffer: usize,
}

#[derive(Serialize)]
struct CapturedLayerEvidence {
    layer: u16,
    expert_ids: Vec<u16>,
    owners: Vec<ExpertOwner>,
    weights_bf16_bits: Vec<u16>,
    packed_admission: Vec<PackedRouteDescriptor>,
    completion_descriptors: Vec<ExpertResultDescriptor>,
    router_logits_bf16_sha256: String,
    router_elapsed_ms: f32,
    router_source_d2h_bytes: usize,
    router_descriptor_d2h_bytes: usize,
    expert_boundary_sha256: Vec<ExpertBoundaryHashes>,
    reduction_output_sha256: String,
    reduction_kernel_elapsed_ms: f32,
    strict_three_way_intersection: bool,
    timeline: Vec<TimelinePoint>,
}

#[derive(Serialize)]
struct ExpertBoundaryHashes {
    rank: u8,
    expert: u16,
    kernel_elapsed_ms: f32,
    cpu_elapsed_ns: Option<u64>,
    input_d2d_bytes: usize,
    input_h2d_bytes: usize,
    output_d2h_bytes: usize,
    gate_up: Option<String>,
    swiglu: Option<String>,
    down: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    validate_owner_cache(&cli.owner_cache, cli.constructor)?;
    let retained: RetainedControl = serde_json::from_slice(&std::fs::read(&cli.retained_trace)?)?;
    if retained.generated_token_ids.get(..EXPECTED.len()) != Some(EXPECTED.as_slice()) {
        bail!("retained CPU control identity does not contain the required continuation");
    }
    let prompt_limit = cli
        .max_input_tokens
        .unwrap_or(retained.prompt_token_ids.len());
    if prompt_limit == 0 || prompt_limit > retained.prompt_token_ids.len() {
        bail!("input-token bound is outside the retained prompt");
    }
    if cli.max_new_tokens == 0 || cli.max_new_tokens > EXPECTED.len() {
        bail!("new-token bound must be within 1..={}", EXPECTED.len());
    }
    if cli.repeat == 0 || cli.repeat > 2 {
        bail!("H7 repeat count must be one or two");
    }
    let release_handshake = release_handshake_config(&cli)?;

    let config = CpuGptOssConfig::from_snapshot(&cli.model)?;
    let manifest_bytes = std::fs::read(&cli.placement)?;
    let manifest = serde_json::from_slice(&manifest_bytes)?;
    let binary_sha256 = sha256_file(&std::env::current_exe()?)?;
    let placement_file_sha256 = hash_bytes(&manifest_bytes);
    let retained_trace_sha256 = sha256_file(&cli.retained_trace)?;
    let mut runs = Vec::with_capacity(cli.repeat);
    for run_index in 0..cli.repeat {
        runs.push(run_once(
            run_index,
            &cli,
            &retained,
            prompt_limit,
            &config,
            &manifest,
            release_handshake.as_ref(),
        )?);
    }
    let all_runs_passed = runs.iter().all(|run| run.passed);
    let evidence = Evidence {
        schema: "gpt-oss-rs.heterogeneous-control-h7/v4",
        execution_path: "serial_m1_exact_router_selected_expert_rank_reduction",
        cuda_prefill_or_all_expert_fallback_used: false,
        tensor_parallel_or_nccl_used: false,
        peer_access_used: false,
        decode_expert_weight_transfer_bytes: 0,
        constructor: cli.constructor,
        r2_policy_sha256: CAPACITY_ONE_POLICY_SHA256,
        binary_sha256,
        placement_file_sha256,
        retained_trace_sha256,
        prompt_tokens_requested: prompt_limit,
        expected_token_ids: EXPECTED[..cli.max_new_tokens].to_vec(),
        repeat_requested: cli.repeat,
        runs,
        all_runs_passed,
    };
    if let Some(path) = cli.output {
        let mut bytes = serde_json::to_vec_pretty(&evidence)?;
        bytes.push(b'\n');
        std::fs::write(path, bytes)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&evidence)?);
    }
    if !all_runs_passed {
        bail!("H7 control gate failed");
    }
    Ok(())
}

fn validate_owner_cache(path: &Path, constructor: ConstructorMode) -> Result<()> {
    let legacy = Path::new("/home/emmy/workspace/gpt-oss-rs-het-cache");
    if path == legacy && constructor == ConstructorMode::MonolithicControl {
        return Ok(());
    }
    if !path.is_absolute() {
        bail!("comparison owner cache must be absolute");
    }
    let workspace = Path::new("/home/emmy/workspace");
    let relative = path
        .strip_prefix(workspace)
        .context("comparison owner cache is outside the authorized workspace")?;
    let components = relative.components().collect::<Vec<_>>();
    if components.len() != 2 {
        bail!("comparison owner cache must be an immediate child of one R4 run root");
    }
    let run_name = match components[0] {
        std::path::Component::Normal(name) => name.to_string_lossy(),
        _ => bail!("comparison run root contains a non-normal component"),
    };
    let expected_cache = match constructor {
        ConstructorMode::MonolithicControl => "monolithic-cache",
        ConstructorMode::CapacityOne => "capacity-one-cache",
    };
    let cache_name = match components[1] {
        std::path::Component::Normal(name) => name.to_string_lossy(),
        _ => bail!("comparison cache root contains a non-normal component"),
    };
    if !run_name.starts_with("gpt-oss-rs-het-r4-") || cache_name != expected_cache {
        bail!("comparison owner cache does not match its constructor-bound R4 namespace");
    }
    let run_root = workspace.join(run_name.as_ref());
    if run_root.is_symlink() || !run_root.is_dir() {
        bail!("comparison run root must be an existing non-symlink directory");
    }
    if path.is_symlink() {
        bail!("comparison owner cache must not be a symlink");
    }
    Ok(())
}

fn release_handshake_config(cli: &Cli) -> Result<Option<ChildReleaseHandshake>> {
    let configured = match (
        cli.release_handshake_root.as_ref(),
        cli.release_handshake_nonce.as_ref(),
        cli.release_handshake_cell.as_ref(),
    ) {
        (None, None, None) => None,
        (Some(root), Some(nonce), Some(cell)) => Some(ChildReleaseHandshake {
            root: root.clone(),
            nonce: nonce.clone(),
            cell: cell.clone(),
            constructor: cli.constructor.release_name().into(),
            expected_releases: cli.repeat,
        }),
        _ => bail!("R2 release handshake arguments must be supplied together"),
    };
    if cli.owner_cache != Path::new("/home/emmy/workspace/gpt-oss-rs-het-cache")
        && configured.is_none()
    {
        bail!("R4 H7 control requires the R2 release handshake");
    }
    Ok(configured)
}

fn perform_release_handshake(
    model: &OwnerSelectiveModel,
    constructor: ConstructorMode,
    handshake: Option<&ChildReleaseHandshake>,
    ordinal: usize,
) -> Result<Option<ReleaseReadyMarker>> {
    let proof = match constructor {
        ConstructorMode::MonolithicControl => {
            let release = model
                .checkpoint_release_evidence()
                .context("monolithic model omitted checkpoint release evidence")?;
            ReleaseProof {
                release_report_count: release.shard_releases.len(),
                source_mapping_count_after_release: release.source_mapping_count_after_release,
                source_mapping_pss_bytes_after_release: release
                    .source_mapping_pss_bytes_after_release,
                source_payload_fds_after_release: if release.descriptors_closed { 0 } else { 1 },
                mappings_removed: release.mappings_removed,
                descriptors_closed: release.descriptors_closed,
                capacity_one_mapping_high_water: None,
            }
        }
        ConstructorMode::CapacityOne => {
            let evidence = model
                .capacity_one_evidence()
                .context("capacity-one model omitted source release evidence")?;
            ReleaseProof {
                release_report_count: evidence.shard_releases.len(),
                source_mapping_count_after_release: evidence
                    .shard_releases
                    .iter()
                    .map(|release| release.post_release.source_inode_mapping_count)
                    .sum(),
                source_mapping_pss_bytes_after_release: evidence
                    .shard_releases
                    .iter()
                    .map(|release| release.post_release.source_inode_pss_bytes)
                    .sum(),
                source_payload_fds_after_release: evidence
                    .publication_proof
                    .active_source_payload_fds,
                mappings_removed: evidence
                    .shard_releases
                    .iter()
                    .all(|release| release.mapping_removed),
                descriptors_closed: evidence
                    .shard_releases
                    .iter()
                    .all(|release| release.fd_closed),
                capacity_one_mapping_high_water: Some(evidence.active_mapping_high_water),
            }
        }
    };
    proof.validate(constructor.release_name())?;
    handshake
        .map(|config| child_release_handshake(config, ordinal, CAPACITY_ONE_POLICY_SHA256, proof))
        .transpose()
}

#[allow(clippy::too_many_lines)]
fn run_once(
    run_index: usize,
    cli: &Cli,
    retained: &RetainedControl,
    prompt_limit: usize,
    config: &CpuGptOssConfig,
    manifest: &gpt_oss_model_runner::heterogeneous::GptOssExpertPlacementManifestV1,
    release_handshake: Option<&ChildReleaseHandshake>,
) -> Result<RunEvidence> {
    let resources_before = resource_snapshot()?;
    let constructor = OwnerSelectiveConstructor::new(&cli.owner_cache);
    let mut construction_stages = Vec::with_capacity(8);
    let mut observer = |ledger: &ConstructionLedger| {
        let resources = resource_snapshot()
            .map_err(|error| LLMError::MemoryError(format!("H7 stage snapshot: {error:#}")))?;
        construction_stages.push(ConstructionStageEvidence {
            ledger: ledger.clone(),
            resources,
        });
        Ok(())
    };
    let (mut model, model_identity) = match cli.constructor {
        ConstructorMode::MonolithicControl => {
            let checkpoint = GptOssCheckpointView::open(&cli.native_model)?;
            let identity = ModelIdentityEvidence {
                revision: checkpoint.revision().to_owned(),
                config_sha256: checkpoint.config_sha256().to_owned(),
                index_sha256: checkpoint.metadata_sha256().to_owned(),
                mapping_sha256: checkpoint.mapping_sha256().to_owned(),
            };
            (
                constructor.construct(checkpoint, manifest, &mut observer)?,
                identity,
            )
        }
        ConstructorMode::CapacityOne => {
            let catalog = SafeTensorShardCatalog::open(&cli.native_model)?;
            let native = GptOssNativeCatalogMap::from_source_root(&cli.native_model, &catalog)?;
            let identity = ModelIdentityEvidence {
                revision: native.revision().to_owned(),
                config_sha256: native.config_sha256().to_owned(),
                index_sha256: native.metadata_sha256().to_owned(),
                mapping_sha256: native.mapping_sha256().to_owned(),
            };
            drop(native);
            drop(catalog);
            (
                constructor.construct_capacity_one(
                    &cli.native_model,
                    manifest,
                    CAPACITY_ONE_POLICY_SHA256,
                    &mut observer,
                )?,
                identity,
            )
        }
    };
    let capacity_one = model.capacity_one_evidence().cloned();
    let checkpoint_release = model.checkpoint_release_evidence().cloned();
    model.drain()?;
    let source_release_handshake =
        perform_release_handshake(&model, cli.constructor, release_handshake, run_index)?;
    let resources_after_construct = resource_snapshot()?;
    let placement = {
        let resolved = model.placement();
        let counts = resolved.counts();
        PlacementEvidence {
            manifest_hash: resolved.manifest_hash().to_owned(),
            placement_epoch: resolved.placement_epoch(),
            layer_owner: resolved.layer_owner().stable_id.clone(),
            layer_owner_ordinal: resolved.layer_owner().transient_ordinal,
            remote_worker: resolved.remote_worker().stable_id.clone(),
            remote_worker_ordinal: resolved.remote_worker().transient_ordinal,
            cpu_experts: counts.cpu,
            layer_owner_experts: counts.layer_owner_gpu,
            remote_gpu_experts: counts.remote_gpu,
        }
    };
    let envelope = model.envelope().clone();
    let final_ledger = model.ledger().clone();
    let placement_epoch = model.placement().placement_epoch();
    let mut runtime = HeterogeneousControlRuntime::new(&mut model, config)?;
    let resources_after_runtime = resource_snapshot()?;
    let mut coordinator = HeterogeneousTransactionCoordinator::new(1, 96, false)?;
    coordinator.register_sequence(1, 0, placement_epoch, Vec::new())?;
    let mut committed_inputs = Vec::with_capacity(prompt_limit + cli.max_new_tokens - 1);
    let mut committed_steps = Vec::with_capacity(prompt_limit + cli.max_new_tokens - 1);
    let mut cancellation_retry = execute_and_discard(
        &mut coordinator,
        &mut runtime,
        &mut model,
        config,
        retained.prompt_token_ids[0],
    )?;
    let mut generated = Vec::with_capacity(cli.max_new_tokens);
    let mut captured_layer = None;

    let mut prediction = 0_u32;
    for (position, &input) in retained.prompt_token_ids[..prompt_limit].iter().enumerate() {
        let (next, capture_evidence, commit_evidence) = execute_and_commit(
            &mut coordinator,
            &mut runtime,
            &mut model,
            config,
            &mut committed_inputs,
            input,
            None,
        )
        .with_context(|| format!("H7 serial-prefill token {position}"))?;
        prediction = next;
        if position == 0 {
            if prediction != cancellation_retry.discarded_prediction
                || commit_evidence.output_logits_sha256
                    != cancellation_retry.discarded_logits_sha256
            {
                bail!("H7 clean retry after discard changed output identity");
            }
            cancellation_retry.clean_retry_identity_matched = true;
        }
        committed_steps.push(commit_evidence);
        if capture_evidence.is_some() {
            captured_layer = capture_evidence;
        }
    }
    generated.push(prediction);
    for generated_index in 1..cli.max_new_tokens {
        let capture =
            (prompt_limit == retained.prompt_token_ids.len() && generated_index == 1).then_some(0);
        let (next, capture_evidence, commit_evidence) = execute_and_commit(
            &mut coordinator,
            &mut runtime,
            &mut model,
            config,
            &mut committed_inputs,
            prediction,
            capture,
        )
        .with_context(|| format!("H7 retained decode token {generated_index}"))?;
        committed_steps.push(commit_evidence);
        prediction = next;
        generated.push(prediction);
        if capture_evidence.is_some() {
            captured_layer = capture_evidence;
        }
    }

    let resources_after_execution = resource_snapshot()?;
    runtime.drain(&mut model)?;
    let committed = coordinator
        .committed_view(1)
        .context("H7 committed sequence disappeared")?;
    let committed_length = committed.committed_length;
    let request_revision = committed.request_revision;
    let visibility_epoch = committed.visibility_epoch;
    let pool = runtime.pinned_pool_stats();
    let pinned_checked_out_after = pool_checked_out(&pool);
    let pinned_quarantined_after = pool_quarantined(&pool);
    let active_steps_after = coordinator.active_step_count();
    let cpu_worker_high_water_jobs = runtime.cpu_high_water_jobs();
    let pinned_pool = pinned_pool_evidence(&pool);
    let pinned_raw_capacity_bytes = pool.raw_capacity_bytes;
    let resources_before_drop = resource_snapshot()?;
    let exact_retained_tokens = prompt_limit == retained.prompt_token_ids.len()
        && generated == EXPECTED[..cli.max_new_tokens];
    let target_gate = if prompt_limit == retained.prompt_token_ids.len() {
        exact_retained_tokens && captured_layer.is_some()
    } else {
        true
    };
    let in_process_snapshots = [
        &resources_before,
        &resources_after_construct,
        &resources_after_runtime,
        &resources_after_execution,
        &resources_before_drop,
    ];
    let in_process_memory_gate = memory_gate(&in_process_snapshots)
        && in_process_snapshots.iter().all(|snapshot| {
            snapshot.global_swap_used_bytes <= resources_before.global_swap_used_bytes
        });
    drop(runtime);
    let cpu_record_releases = model.release_cpu_record_mappings_with_advice()?;
    drop(model);
    drop(coordinator);
    let resources_after_drop = resource_snapshot()?;
    let memory_gate_passed = in_process_memory_gate
        && memory_gate(&[&resources_after_drop])
        && resources_after_drop.global_swap_used_bytes <= resources_before.global_swap_used_bytes;
    let cleanup_gate_passed = pinned_checked_out_after == 0
        && pinned_quarantined_after == 0
        && active_steps_after == 0
        && gpu_cleanup_gate(&resources_before, &resources_after_drop);
    let epoch_gate_passed = committed_steps.iter().enumerate().all(|(index, step)| {
        let expected = u64::try_from(index + 1).ok();
        expected.is_some_and(|expected| {
            step.request_revision == expected
                && step.visibility_epoch == expected
                && u64::from(step.committed_length) == expected
                && u64::try_from(step.runtime_committed_tokens) == Ok(expected)
        })
    });
    let passed = target_gate
        && cancellation_retry.clean_retry_identity_matched
        && memory_gate_passed
        && cleanup_gate_passed
        && epoch_gate_passed;
    Ok(RunEvidence {
        run_index,
        load_class: if run_index == 0 {
            "cold_process_load_with_existing_identity_valid_cpu_cache"
        } else {
            "warm_reload_same_process"
        },
        model_identity,
        placement,
        envelope,
        final_ledger,
        capacity_one,
        checkpoint_release,
        source_release_handshake,
        cpu_record_releases,
        construction_stages,
        resources_before,
        resources_after_construct,
        resources_after_runtime,
        resources_after_execution,
        resources_before_drop,
        resources_after_drop,
        cancellation_retry,
        committed_steps,
        prompt_tokens_committed: prompt_limit,
        generated_token_ids: generated,
        exact_retained_tokens,
        committed_length,
        request_revision,
        visibility_epoch,
        active_steps_after,
        cpu_worker_high_water_jobs,
        pinned_pool,
        pinned_raw_capacity_bytes,
        pinned_checked_out_after,
        pinned_quarantined_after,
        captured_layer,
        memory_gate_passed,
        cleanup_gate_passed,
        epoch_gate_passed,
        passed,
    })
}

fn execute_and_discard(
    coordinator: &mut HeterogeneousTransactionCoordinator,
    runtime: &mut HeterogeneousControlRuntime,
    model: &mut gpt_oss_model_runner::model_loader::owner_selective::OwnerSelectiveModel,
    config: &CpuGptOssConfig,
    input_token: u32,
) -> Result<CancellationRetryEvidence> {
    let before = coordinator
        .committed_view(1)
        .context("H7 sequence missing before discard probe")?
        .clone();
    let step = coordinator.reserve_step(1, 1, model.placement().placement_epoch())?;
    coordinator.mark_prepared(step)?;
    coordinator.mark_dispatched(step, &ROLES)?;
    let execution = runtime
        .execute_step(
            model,
            config,
            input_token,
            step,
            None,
            &CorrelatedTimeline::new(),
        )
        .map_err(|failure| failure.error)?;
    if coordinator.cancel_step(step)?.is_some() {
        bail!("post-dispatch H7 cancellation terminated before drain accounting");
    }
    runtime.discard_prepared_token(model)?;
    for role in ROLES {
        coordinator.mark_terminal(step, role)?;
    }
    let terminal = coordinator.finalize_discard(step)?;
    if terminal.outcome != TransactionOutcome::Discarded {
        bail!("H7 cancellation probe did not terminate as discarded");
    }
    let after = coordinator
        .committed_view(1)
        .context("H7 sequence missing after discard probe")?;
    let pool = runtime.pinned_pool_stats();
    let pinned_checked_out_after = pool_checked_out(&pool);
    let pinned_quarantined_after = pool_quarantined(&pool);
    if before != *after
        || runtime.shell().committed_tokens() != usize::try_from(before.committed_length)?
        || coordinator.active_step_count() != 0
        || pinned_checked_out_after != 0
        || pinned_quarantined_after != 0
    {
        bail!("H7 cancellation probe exposed state or retained bounded capacity");
    }
    Ok(CancellationRetryEvidence {
        discarded_step: step,
        discarded_token: input_token,
        discarded_prediction: execution.output.token_id,
        discarded_logits_sha256: execution.output.logits_bf16_sha256,
        committed_length_before: before.committed_length,
        committed_length_after: after.committed_length,
        visibility_epoch_before: before.visibility_epoch,
        visibility_epoch_after: after.visibility_epoch,
        runtime_committed_tokens_after: runtime.shell().committed_tokens(),
        active_steps_after: coordinator.active_step_count(),
        pinned_checked_out_after,
        pinned_quarantined_after,
        clean_retry_identity_matched: false,
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_and_commit(
    coordinator: &mut HeterogeneousTransactionCoordinator,
    runtime: &mut HeterogeneousControlRuntime,
    model: &mut gpt_oss_model_runner::model_loader::owner_selective::OwnerSelectiveModel,
    config: &CpuGptOssConfig,
    committed_inputs: &mut Vec<u32>,
    input_token: u32,
    capture_layer: Option<usize>,
) -> Result<(u32, Option<CapturedLayerEvidence>, StepCommitEvidence)> {
    // Every fallible publication-image allocation is complete before the
    // router or any selected expert enters a stream. Commit later only fills
    // fixed-size bytes and moves this image into the coordinator.
    let committed = coordinator
        .committed_view(1)
        .context("H7 sequence missing before reservation")?;
    let next_revision = committed
        .request_revision
        .checked_add(1)
        .context("H7 request revision overflow")?;
    let mut next_token_ids = Vec::with_capacity(committed_inputs.len() + 1);
    next_token_ids.extend_from_slice(committed_inputs);
    next_token_ids.push(input_token);
    let mut commit_image = SequenceCommitImage {
        next_revision,
        token_ids: next_token_ids,
        output_image: vec![0_u8; size_of::<u32>()],
        evidence_image: vec![0_u8; 64],
    };
    let step = coordinator.reserve_step(1, 1, model.placement().placement_epoch())?;
    coordinator.mark_prepared(step)?;
    coordinator.mark_dispatched(step, &ROLES)?;
    let timeline = CorrelatedTimeline::new();
    let execution =
        match runtime.execute_step(model, config, input_token, step, capture_layer, &timeline) {
            Ok(execution) => execution,
            Err(failure) => {
                let _ = coordinator.cancel_step(step);
                if failure.drain_proven {
                    for role in ROLES {
                        let _ = coordinator.mark_terminal(step, role);
                    }
                    let _ = coordinator.finalize_discard(step);
                }
                return Err(failure.error.into());
            }
        };
    for role in ROLES {
        coordinator.mark_terminal(step, role)?;
    }
    let hidden = runtime.prepared_hidden_bf16_bits()?;
    coordinator.mark_reduced(step, &hidden)?;
    let prepared_hidden_sha256 = hash_u16(&hidden);
    commit_image
        .output_image
        .copy_from_slice(&execution.output.token_id.to_le_bytes());
    commit_image
        .evidence_image
        .copy_from_slice(execution.output.logits_bf16_sha256.as_bytes());
    coordinator.prepare_commit(step, commit_image)?;
    let terminal =
        coordinator.commit_with_external_visibility(step, || runtime.commit_prepared_token())?;
    if terminal.outcome != TransactionOutcome::Committed {
        bail!("H7 transaction did not commit");
    }
    committed_inputs.push(input_token);
    let committed = coordinator
        .committed_view(1)
        .context("H7 sequence missing after commit")?;
    let commit_evidence = StepCommitEvidence {
        step_id: step,
        input_token,
        output_token: execution.output.token_id,
        output_logits_sha256: execution.output.logits_bf16_sha256.clone(),
        prepared_hidden_sha256,
        committed_length: committed.committed_length,
        request_revision: committed.request_revision,
        visibility_epoch: committed.visibility_epoch,
        runtime_committed_tokens: runtime.shell().committed_tokens(),
    };
    let captured = execution
        .captured_layer
        .map(|layer| capture_layer_evidence(layer, timeline.points()))
        .transpose()?;
    Ok((execution.output.token_id, captured, commit_evidence))
}

fn capture_layer_evidence(
    layer: HeterogeneousControlLayerExecution,
    timeline: Vec<TimelinePoint>,
) -> Result<CapturedLayerEvidence> {
    let expert_ids = layer
        .router
        .batch
        .routes
        .iter()
        .map(|route| route.expert_id)
        .collect::<Vec<_>>();
    let weights_bf16_bits = layer
        .router
        .batch
        .routes
        .iter()
        .map(|route| route.weight_bf16_bits)
        .collect::<Vec<_>>();
    let owners = layer
        .experts
        .iter()
        .map(|expert| expert.descriptor.owner.clone())
        .collect::<Vec<_>>();
    let packed_admission = layer
        .plan
        .local_gpu
        .iter()
        .chain(layer.plan.cpu.iter())
        .chain(layer.plan.remote_gpu.iter())
        .flat_map(|owner| owner.routes.iter().map(|route| route.descriptor.clone()))
        .collect::<Vec<_>>();
    let completion_descriptors = layer
        .experts
        .iter()
        .map(|expert| expert.descriptor.clone())
        .collect::<Vec<_>>();
    if packed_admission.len() != 4
        || completion_descriptors.len() != 4
        || completion_descriptors.iter().any(|completion| {
            !packed_admission
                .iter()
                .any(|route| ExpertResultDescriptor::from_packed_route(route) == *completion)
        })
    {
        bail!("captured H7 completion identity does not match packed admission");
    }
    let strict_three_way_intersection = triple_intersection(&timeline);
    if owners
        .iter()
        .filter(|owner| matches!(owner, ExpertOwner::Cpu { .. }))
        .count()
        != 1
        || owners
            .iter()
            .filter(|owner| matches!(owner, ExpertOwner::RemoteGpu { .. }))
            .count()
            != 1
        || owners
            .iter()
            .filter(|owner| matches!(owner, ExpertOwner::LayerOwnerGpu { .. }))
            .count()
            != 2
        || !strict_three_way_intersection
    {
        bail!("captured retained layer did not prove the required 2/1/1 concurrent ownership");
    }
    let expert_boundary_sha256 = layer
        .experts
        .iter()
        .map(|expert| ExpertBoundaryHashes {
            rank: expert.descriptor.route_rank,
            expert: expert.descriptor.expert_id,
            kernel_elapsed_ms: expert.kernel_elapsed_ms,
            cpu_elapsed_ns: expert.cpu_elapsed_ns,
            input_d2d_bytes: expert.input_d2d_bytes,
            input_h2d_bytes: expert.input_h2d_bytes,
            output_d2h_bytes: expert.output_d2h_bytes,
            gate_up: expert
                .trace
                .as_ref()
                .map(|trace| hash_u16(&trace.gate_up_bf16_bits)),
            swiglu: expert
                .trace
                .as_ref()
                .map(|trace| hash_u16(&trace.swiglu_bf16_bits)),
            down: expert
                .trace
                .as_ref()
                .map(|trace| hash_u16(&trace.down_bf16_bits)),
        })
        .collect();
    Ok(CapturedLayerEvidence {
        layer: layer.layer,
        expert_ids,
        owners,
        weights_bf16_bits,
        packed_admission,
        completion_descriptors,
        router_logits_bf16_sha256: hash_u16(&layer.router.router_logits_bf16_bits),
        router_elapsed_ms: layer.router.router_elapsed_ms,
        router_source_d2h_bytes: layer.router.source_d2h_bytes,
        router_descriptor_d2h_bytes: layer.router.descriptor_d2h_bytes,
        expert_boundary_sha256,
        reduction_output_sha256: hash_u16(&layer.reduction.output_bf16_bits),
        reduction_kernel_elapsed_ms: layer.reduction.kernel_elapsed_ms,
        strict_three_way_intersection,
        timeline,
    })
}

fn triple_intersection(points: &[TimelinePoint]) -> bool {
    let interval = |actor: &str| {
        let begin = points
            .iter()
            .find(|point| point.actor == actor && point.label == "compute_begin")?
            .monotonic_ns;
        let end = points
            .iter()
            .find(|point| point.actor == actor && point.label == "compute_end")?
            .monotonic_ns;
        Some((begin, end))
    };
    let Some(cpu) = interval("cpu_expert") else {
        return false;
    };
    let Some(gpu0) = interval("h7_gpu0_local_first") else {
        return false;
    };
    let Some(gpu1) = interval("h7_gpu1_remote") else {
        return false;
    };
    cpu.0.max(gpu0.0).max(gpu1.0) < cpu.1.min(gpu0.1).min(gpu1.1)
}

fn hash_u16(values: &[u16]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn pool_checked_out(pool: &RelayPinnedPoolStats) -> usize {
    pool.source_activation.checked_out
        + pool.route_descriptors.checked_out
        + pool.remote_gpu_input.checked_out
        + pool.remote_gpu_result.checked_out
        + pool.cpu_result.checked_out
}

fn pool_quarantined(pool: &RelayPinnedPoolStats) -> u64 {
    pool.source_activation.quarantined
        + pool.route_descriptors.quarantined
        + pool.remote_gpu_input.quarantined
        + pool.remote_gpu_result.quarantined
        + pool.cpu_result.quarantined
}

fn pinned_pool_evidence(pool: &RelayPinnedPoolStats) -> PinnedPoolEvidence {
    PinnedPoolEvidence {
        source_activation: pinned_lease_class(pool.source_activation),
        route_descriptors: pinned_lease_class(pool.route_descriptors),
        remote_gpu_input: pinned_lease_class(pool.remote_gpu_input),
        remote_gpu_result: pinned_lease_class(pool.remote_gpu_result),
        cpu_result: pinned_lease_class(pool.cpu_result),
        raw_capacity_bytes: pool.raw_capacity_bytes,
        hard_cap_bytes: pool.hard_cap_bytes,
    }
}

const fn pinned_lease_class(stats: BoundedPinnedPoolStats) -> PinnedLeaseClassEvidence {
    PinnedLeaseClassEvidence {
        capacity: stats.capacity,
        available: stats.available,
        checked_out: stats.checked_out,
        high_water: stats.high_water,
        fixed_allocations: stats.fixed_allocations,
        exhaustions: stats.exhaustions,
        quarantined: stats.quarantined,
        bytes_per_buffer: stats.bytes_per_buffer,
    }
}

fn resource_snapshot() -> Result<ResourceSnapshot> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    let process_rss_bytes = proc_kib(&status, "VmRSS:")?;
    let process_high_water_bytes = proc_kib(&status, "VmHWM:")?;
    let process_swap_used_bytes = proc_kib(&status, "VmSwap:")?;
    let mem_available_bytes = proc_kib(&meminfo, "MemAvailable:")?;
    let swap_total = proc_kib(&meminfo, "SwapTotal:")?;
    let swap_free = proc_kib(&meminfo, "SwapFree:")?;
    let swap_cached_bytes = proc_kib(&meminfo, "SwapCached:")?;
    let global_swap_used_bytes = swap_total
        .checked_sub(swap_free)
        .context("global swap accounting underflow")?;
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=pci.bus_id,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .context("run nvidia-smi for H7 resource evidence")?;
    if !output.status.success() {
        bail!("nvidia-smi failed during H7 resource snapshot");
    }
    let stdout = String::from_utf8(output.stdout)?;
    let mut gpus = Vec::with_capacity(2);
    for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
        let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() != 4 {
            bail!("unexpected nvidia-smi memory row: {line}");
        }
        let mib = |value: &str| -> Result<u64> {
            value
                .parse::<u64>()?
                .checked_mul(1024 * 1024)
                .context("GPU MiB conversion overflow")
        };
        gpus.push(GpuMemorySnapshot {
            pci_bus_id: fields[0].to_ascii_lowercase(),
            total_bytes: mib(fields[1])?,
            free_bytes: mib(fields[2])?,
            used_bytes: mib(fields[3])?,
        });
    }
    if gpus.len() != 2 {
        bail!("H7 requires exactly two driver-visible GPUs");
    }
    let unix_time_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    Ok(ResourceSnapshot {
        unix_time_ms,
        process_rss_bytes,
        process_high_water_bytes,
        process_swap_used_bytes,
        mem_available_bytes,
        global_swap_used_bytes,
        swap_cached_bytes,
        gpus,
    })
}

fn proc_kib(contents: &str, key: &str) -> Result<u64> {
    let value = contents
        .lines()
        .find_map(|line| {
            let rest = line.strip_prefix(key)?;
            rest.split_whitespace().next()
        })
        .with_context(|| format!("missing {key} in proc snapshot"))?
        .parse::<u64>()?;
    value
        .checked_mul(1024)
        .with_context(|| format!("{key} byte conversion overflow"))
}

fn memory_gate(snapshots: &[&ResourceSnapshot]) -> bool {
    const MIN_MEM_AVAILABLE: u64 = 12 * 1024 * 1024 * 1024;
    const MAX_PROCESS_RSS: u64 = 72 * 1024 * 1024 * 1024;
    const MIN_GPU_FREE: u64 = 4 * 1024 * 1024 * 1024;
    snapshots.iter().all(|snapshot| {
        snapshot.process_swap_used_bytes == 0
            && snapshot.mem_available_bytes >= MIN_MEM_AVAILABLE
            && snapshot.process_rss_bytes <= MAX_PROCESS_RSS
            && snapshot
                .gpus
                .iter()
                .all(|gpu| gpu.free_bytes >= MIN_GPU_FREE)
    })
}

fn gpu_cleanup_gate(before: &ResourceSnapshot, after: &ResourceSnapshot) -> bool {
    const CLEANUP_TOLERANCE: u64 = 64 * 1024 * 1024;
    before.gpus.len() == after.gpus.len()
        && before.gpus.iter().all(|baseline| {
            after.gpus.iter().any(|current| {
                current.pci_bus_id == baseline.pci_bus_id
                    && current.total_bytes == baseline.total_bytes
                    && current.used_bytes <= baseline.used_bytes.saturating_add(CLEANUP_TOLERANCE)
            })
        })
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(hash_bytes(&std::fs::read(path)?))
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
