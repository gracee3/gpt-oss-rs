use std::collections::BTreeMap;
#[cfg(feature = "heterogeneous-test-faults")]
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use gpt_oss_core::error::LLMError;
use gpt_oss_gpu::device::{list_devices, GpuDevice, StableCudaDeviceId};
#[cfg(feature = "heterogeneous-test-faults")]
use gpt_oss_model_runner::heterogeneous::CudaSelectedExpertExecutor;
use gpt_oss_model_runner::heterogeneous::{
    selected_expert_device_memory_info, CpuPoolId, ExpertOwner, GptOssExpertKey,
    GptOssExpertPlacementManifestV1, GptOssPlacementModel, PlacementBudgets, PlacementPolicyClass,
    CONSERVATIVE_OWNER_EXPERT_BYTES, HETEROGENEOUS_PLACEMENT_SCHEMA_V1,
};
use gpt_oss_model_runner::model_loader::gpt_oss_native::GptOssCheckpointView;
#[cfg(feature = "heterogeneous-test-faults")]
use gpt_oss_model_runner::model_loader::owner_selective::{
    owner_selective_pinned_current_bytes, owner_selective_pinned_high_water_bytes,
    OWNER_SELECTIVE_PINNED_UPLOAD_BYTES,
};
use gpt_oss_model_runner::model_loader::owner_selective::{
    ConstructionLedger, ConstructionStage, ExecutionReserveDisposition, OwnerSelectiveConstructor,
    OwnerSelectiveEnvelope, OWNER_SELECTIVE_GPU_RESERVE_BYTES, OWNER_SELECTIVE_PROOF_CONTEXT_CAP,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CACHE_ROOT: &str = "/home/emmy/workspace/gpt-oss-rs-het-cache";
const MIN_AVAILABLE_BYTES: u64 = 12 * 1024 * 1024 * 1024;
const MAX_PROCESS_RSS_BYTES: u64 = 72 * 1024 * 1024 * 1024;
const GPU_CLEANUP_TOLERANCE_BYTES: usize = 16 * 1024 * 1024;
#[cfg(feature = "heterogeneous-test-faults")]
const H8_ADMISSION_MARGIN_BYTES: u64 = 64 * 1024 * 1024;
#[cfg(feature = "heterogeneous-test-faults")]
const H8_MIN_DISK_REMAINING_BYTES: u64 = 64 * 1024 * 1024 * 1024;
#[cfg(feature = "heterogeneous-test-faults")]
const H8_POLICY_SEED: u64 = 0x4845_5438_3132_3042;

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Validate,
    Cold,
    Warm,
    #[cfg(feature = "heterogeneous-test-faults")]
    Faults,
    #[cfg(feature = "heterogeneous-test-faults")]
    H8,
}

#[derive(Parser)]
struct Cli {
    #[arg(long, value_enum)]
    mode: Mode,
    #[arg(long)]
    model_20b: PathBuf,
    #[arg(long)]
    model_120b: PathBuf,
    #[arg(long)]
    mapping_20b: PathBuf,
    #[arg(long)]
    mapping_120b: PathBuf,
    #[arg(long)]
    cache_root: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Deserialize)]
struct ResearchMappingDocument {
    native_tensor_count: usize,
    runtime_tensor_count: usize,
    mapping_count: usize,
    mappings: Vec<ResearchMapping>,
}

#[derive(Debug, Deserialize)]
struct ResearchMapping {
    native: String,
    runtime: String,
    native_shard: String,
    native_slice: [usize; 2],
    dtype: String,
    #[serde(default)]
    native_shape: Option<Vec<usize>>,
    #[serde(default)]
    runtime_shape: Option<Vec<usize>>,
    #[serde(default)]
    runtime_shape_derived: Option<Vec<usize>>,
    bytes: usize,
}

#[derive(Debug, Serialize)]
struct EvidenceRecord {
    schema: &'static str,
    mode: Mode,
    captured_unix_ms: u128,
    repository_head: String,
    executable_sha256: String,
    command: Vec<String>,
    toolchain: ToolchainRecord,
    cache_root: String,
    cache_bytes_before: u64,
    cache_bytes_after: u64,
    process_before: ProcessMemory,
    process_after: ProcessMemory,
    system_before: SystemMemory,
    system_after: SystemMemory,
    checkpoint_20b: CheckpointRecord,
    checkpoint_120b: CheckpointRecord,
    placement_20b: PlacementRecord,
    placement_120b: PlacementRecord,
    envelope_20b: OwnerSelectiveEnvelope,
    envelope_120b: OwnerSelectiveEnvelope,
    construction: Option<ConstructionRecord>,
    fault_campaign: Option<FaultCampaignRecord>,
    h8_campaign: Option<H8CampaignRecord>,
    protected_nvme: ProtectedNvmeRecord,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct ToolchainRecord {
    rustc: String,
    cargo: String,
    nvcc: String,
    nvidia_driver: String,
}

#[derive(Debug, Serialize)]
struct CheckpointRecord {
    source: String,
    revision: String,
    config_sha256: String,
    metadata_sha256: String,
    mapping_sha256: String,
    mapping_equivalence_sha256: String,
    native_tensor_count: usize,
    runtime_tensor_count: usize,
    mapped_payload_bytes: u64,
    expert_payload_bytes: u64,
    non_expert_payload_bytes: u64,
}

#[derive(Debug, Serialize)]
struct PlacementRecord {
    manifest_sha256: String,
    policy: &'static str,
    layer_owner_pci: String,
    remote_gpu_pci: String,
    cpu_experts: u32,
    layer_owner_experts: u32,
    remote_gpu_experts: u32,
}

#[derive(Debug, Serialize)]
struct ConstructionRecord {
    elapsed_ms: u128,
    snapshots: Vec<ResourceSnapshot>,
    final_ledger: ConstructionLedger,
    dense_tensors: usize,
    dense_device_bytes: u64,
    cpu_layer_files: Vec<CpuLayerFileRecord>,
    cuda_memory_while_loaded: [CudaMemoryRecord; 2],
    cuda_memory_before: [CudaMemoryRecord; 2],
    cuda_memory_after: [CudaMemoryRecord; 2],
    cleanup_within_tolerance: bool,
    partial_artifacts_after: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ResourceSnapshot {
    elapsed_ms: u128,
    ledger: ConstructionLedger,
    process: ProcessMemory,
    system: SystemMemory,
    gpus: Vec<GpuMemory>,
}

#[derive(Debug, Clone, Default, Serialize)]
struct ProcessMemory {
    vm_size_bytes: u64,
    vm_rss_bytes: u64,
    vm_hwm_bytes: u64,
    vm_swap_bytes: u64,
    rollup_rss_bytes: u64,
    rollup_pss_bytes: u64,
    rollup_pss_anon_bytes: u64,
    rollup_pss_file_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
struct SystemMemory {
    mem_available_bytes: u64,
    swap_total_bytes: u64,
    swap_free_bytes: u64,
    swap_used_bytes: u64,
    swap_cached_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct GpuMemory {
    pci_bus_id: String,
    used_mib: u64,
    free_mib: u64,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct CudaMemoryRecord {
    free_bytes: usize,
    total_bytes: usize,
}

#[derive(Debug, Serialize)]
struct CpuLayerFileRecord {
    layer: u16,
    expert_ids: Vec<u16>,
    payload_bytes: u64,
    file_bytes: u64,
    relative_path: String,
}

#[derive(Debug, Serialize)]
struct FaultCampaignRecord {
    stages: Vec<ConstructionFaultRecord>,
    clean_construction: ConstructionRecord,
    clean_cache_bytes_before: u64,
    clean_cache_bytes_after: u64,
    clean_reused_published_record: bool,
    pinned_high_water_bytes: u64,
    final_pinned_current_bytes: u64,
    final_cache_files: Vec<String>,
    final_partial_artifacts: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ConstructionFaultRecord {
    stage: ConstructionStage,
    elapsed_ms: u128,
    observed_stages: Vec<ConstructionStage>,
    injected_error: String,
    cuda_memory_before: [CudaMemoryRecord; 2],
    cuda_memory_after: [CudaMemoryRecord; 2],
    cleanup_within_tolerance: bool,
    pinned_current_bytes_after: u64,
    pinned_high_water_bytes_after: u64,
    cache_bytes_before: u64,
    cache_bytes_after: u64,
    cache_files_after: Vec<String>,
    partial_artifacts_after: Vec<String>,
    system_after: SystemMemory,
}

#[derive(Debug, Serialize)]
struct H8AdmissionRecord {
    simultaneous_executor_memory: [CudaMemoryRecord; 2],
    additional_margin_bytes_per_gpu: u64,
    layer_owner_experts: u32,
    remote_gpu_experts: u32,
    cpu_experts: u32,
    layer_owner_required_free_bytes: u64,
    remote_gpu_required_free_bytes: u64,
    layer_owner_headroom_bytes: u64,
    remote_gpu_headroom_bytes: u64,
    assignment_algorithm: &'static str,
    policy_seed: u64,
    manifest_sha256: String,
    deterministic_manifest_rebuild_matched: bool,
    all_expert_keys_exactly_once: bool,
    layers_with_all_three_owners: u16,
    per_layer_owner_order: [&'static str; 3],
    per_layer_owner_counts: Vec<[u32; 3]>,
    per_layer_owner_minimums: [u32; 3],
    per_layer_owner_maximums: [u32; 3],
}

#[cfg(feature = "heterogeneous-test-faults")]
struct H8ManifestCoverage {
    manifest_sha256: String,
    all_expert_keys_exactly_once: bool,
    layers_with_all_three_owners: u16,
    per_layer_owner_counts: Vec<[u32; 3]>,
    per_layer_owner_minimums: [u32; 3],
    per_layer_owner_maximums: [u32; 3],
}

#[derive(Debug, Serialize)]
struct H8PublishFaultRecord {
    stage: ConstructionStage,
    elapsed_ms: u128,
    observed_stages: Vec<ConstructionStage>,
    injected_error: String,
    cuda_memory_before: [CudaMemoryRecord; 2],
    loaded_reserve_at_publish: H8LoadedReserveRecord,
    cuda_memory_after: [CudaMemoryRecord; 2],
    cleanup_within_tolerance: bool,
    pinned_current_bytes_after: u64,
    cache_bytes_before: u64,
    cache_bytes_after: u64,
    partial_artifacts_after: Vec<String>,
    process_after: ProcessMemory,
    system_after: SystemMemory,
}

#[derive(Debug, Serialize)]
struct H8CampaignRecord {
    admission: H8AdmissionRecord,
    placement_path: String,
    disk_available_before_bytes: u64,
    disk_required_for_cpu_payload_and_safety_bytes: u64,
    cache_bytes_before: u64,
    cache_bytes_after_cold: u64,
    cache_bytes_after_fault: u64,
    cache_bytes_after_warm: u64,
    cold: ConstructionRecord,
    cold_loaded_reserve: H8LoadedReserveRecord,
    publish_fault: H8PublishFaultRecord,
    warm: ConstructionRecord,
    warm_loaded_reserve: H8LoadedReserveRecord,
    pinned_high_water_bytes: u64,
    final_pinned_current_bytes: u64,
    final_cache_files: Vec<String>,
    final_partial_artifacts: Vec<String>,
    process_after: ProcessMemory,
    system_after: SystemMemory,
}

#[derive(Debug, Serialize)]
struct H8LoadedReserveRecord {
    cuda_memory_while_loaded: [CudaMemoryRecord; 2],
    execution_reserve_bytes_per_gpu: u64,
    admission_margin_bytes_per_gpu: u64,
    reserve_plus_margin_bytes_per_gpu: u64,
    execution_reserve_met: [bool; 2],
    reserve_plus_margin_met: [bool; 2],
}

#[derive(Debug, Serialize)]
struct ProtectedNvmeRecord {
    read_only: bool,
    mounted: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cache_root = PathBuf::from(CACHE_ROOT);
    if cli.cache_root != cache_root || !cli.cache_root.is_absolute() {
        bail!("H3 probe cache root must be exactly {CACHE_ROOT}");
    }
    if std::fs::canonicalize("/home/emmy/workspace")? != Path::new("/home/emmy/workspace") {
        bail!("authorized workspace root resolves through an unexpected path");
    }
    match cli.mode {
        Mode::Validate | Mode::Cold if cli.cache_root.exists() => {
            bail!("validate/cold H3 probe requires the authorized cache root to be absent")
        }
        Mode::Warm if !cli.cache_root.is_dir() => {
            bail!("warm H3 probe requires the cold cache directory")
        }
        #[cfg(feature = "heterogeneous-test-faults")]
        Mode::Faults if !cli.cache_root.is_dir() => {
            bail!("fault H3 probe requires the preserved cold/warm cache directory")
        }
        #[cfg(feature = "heterogeneous-test-faults")]
        Mode::H8 if !cli.cache_root.is_dir() => {
            bail!("H8 requires the preserved project-scoped owner cache directory")
        }
        _ => {}
    }
    if !partial_artifacts(&cli.cache_root)?.is_empty() {
        bail!("owner cache contains a partial artifact before construction")
    }

    let protected_nvme = protected_nvme_state()?;
    if !protected_nvme.read_only || protected_nvme.mounted {
        bail!("protected /dev/nvme1n1 is not read-only and unmounted");
    }
    let system_before = system_memory()?;
    let process_before = process_memory()?;
    if process_before.vm_swap_bytes != 0 || system_before.mem_available_bytes < MIN_AVAILABLE_BYTES
    {
        bail!(
            "host preflight memory/swap guard failed: process={process_before:?} system={system_before:?}"
        );
    }

    let devices = ordered_devices()?;
    let stable = [
        StableCudaDeviceId::from_device(&devices[0])?,
        StableCudaDeviceId::from_device(&devices[1])?,
    ];
    let checkpoint_20b = GptOssCheckpointView::open(&cli.model_20b)?;
    let map_equivalence_20b = compare_research_mapping(&checkpoint_20b, &cli.mapping_20b)?;
    assert_checkpoint_bytes(
        &checkpoint_20b,
        13_761_264_768,
        10_165_616_640,
        3_595_648_128,
    )?;
    let manifest_20b = proof_manifest_20b(&checkpoint_20b, &stable)?;
    #[cfg(feature = "heterogeneous-test-faults")]
    let manifest_20b = {
        let mut manifest = manifest_20b;
        if matches!(cli.mode, Mode::Faults) {
            // A distinct immutable placement identity lets the CPU-stage fault
            // publish one valid record without mutating the cold/warm record.
            manifest.policy_seed = 0x4845_5433_4641_554c;
            manifest.placement_epoch = 2;
        }
        manifest
    };
    let resolved_20b = manifest_20b.validate(&devices)?;
    let envelope_20b =
        OwnerSelectiveEnvelope::from_checkpoint_and_placement(&checkpoint_20b, &resolved_20b)?;
    verify_execution_reserve_plan(&envelope_20b)?;

    let checkpoint_120b = GptOssCheckpointView::open(&cli.model_120b)?;
    let map_equivalence_120b = compare_research_mapping(&checkpoint_120b, &cli.mapping_120b)?;
    assert_checkpoint_bytes(
        &checkpoint_120b,
        65_248_815_744,
        60_993_699_840,
        4_255_115_904,
    )?;
    #[cfg(feature = "heterogeneous-test-faults")]
    let (manifest_120b, h8_admission) = if matches!(cli.mode, Mode::H8) {
        let (manifest, admission) = admitted_manifest_120b(&checkpoint_120b, &stable)?;
        (manifest, Some(admission))
    } else {
        (existence_manifest_120b(&checkpoint_120b, &stable)?, None)
    };
    #[cfg(not(feature = "heterogeneous-test-faults"))]
    let manifest_120b = existence_manifest_120b(&checkpoint_120b, &stable)?;
    let resolved_120b = manifest_120b.validate(&devices)?;
    let envelope_120b =
        OwnerSelectiveEnvelope::from_checkpoint_and_placement(&checkpoint_120b, &resolved_120b)?;
    verify_execution_reserve_plan(&envelope_120b)?;
    #[cfg(feature = "heterogeneous-test-faults")]
    if let Some(admission) = &h8_admission {
        verify_h8_envelope(&envelope_120b, admission)?;
    } else {
        verify_120b_envelope(&envelope_120b)?;
    }
    #[cfg(not(feature = "heterogeneous-test-faults"))]
    verify_120b_envelope(&envelope_120b)?;

    if matches!(cli.mode, Mode::Validate) {
        let parent = cli.output.parent().context("output has no parent")?;
        std::fs::create_dir_all(parent)?;
        atomic_write(
            &parent.join("mapping-20b.generated.json"),
            &checkpoint_20b.mapping_json()?,
        )?;
        atomic_write(
            &parent.join("mapping-120b.generated.json"),
            &checkpoint_120b.mapping_json()?,
        )?;
        atomic_write(
            &parent.join("placement-20b.json"),
            &manifest_20b.stable_json()?,
        )?;
        atomic_write(
            &parent.join("placement-120b-existence.json"),
            &manifest_120b.stable_json()?,
        )?;
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    let h8_placement_path = if matches!(cli.mode, Mode::H8) {
        let parent = cli.output.parent().context("output has no parent")?;
        std::fs::create_dir_all(parent)?;
        let path = parent.join("placement-120b-h8.json");
        atomic_write(&path, &manifest_120b.stable_json()?)?;
        Some(path)
    } else {
        None
    };

    let checkpoint_record_20b = checkpoint_record(&checkpoint_20b, map_equivalence_20b);
    let checkpoint_record_120b = checkpoint_record(&checkpoint_120b, map_equivalence_120b);
    let placement_record_20b = placement_record(&manifest_20b, &resolved_20b);
    let placement_record_120b = placement_record(&manifest_120b, &resolved_120b);
    drop(checkpoint_120b);
    let cache_bytes_before = cache_bytes(&cli.cache_root)?;

    let (construction, fault_campaign, h8_campaign) = match cli.mode {
        Mode::Validate => (None, None, None),
        Mode::Cold | Mode::Warm => (
            Some(run_construction(
                checkpoint_20b,
                &manifest_20b,
                &cli.cache_root,
                &stable,
                system_before.swap_used_bytes,
            )?),
            None,
            None,
        ),
        #[cfg(feature = "heterogeneous-test-faults")]
        Mode::Faults => {
            drop(checkpoint_20b);
            (
                None,
                Some(run_fault_campaign(
                    &cli.model_20b,
                    &manifest_20b,
                    &cli.cache_root,
                    &stable,
                    system_before.swap_used_bytes,
                )?),
                None,
            )
        }
        #[cfg(feature = "heterogeneous-test-faults")]
        Mode::H8 => {
            drop(checkpoint_20b);
            (
                None,
                None,
                Some(run_h8_campaign(
                    &cli.model_120b,
                    &manifest_120b,
                    h8_admission.context("H8 admission record is missing")?,
                    &envelope_120b,
                    &cli.cache_root,
                    &stable,
                    system_before.swap_used_bytes,
                    h8_placement_path
                        .as_deref()
                        .context("H8 placement path is missing")?,
                )?),
            )
        }
    };
    let cache_bytes_after = cache_bytes(&cli.cache_root)?;
    if matches!(cli.mode, Mode::Warm) && cache_bytes_after != cache_bytes_before {
        bail!("warm load changed the immutable persistent cache byte count");
    }
    let process_after = process_memory()?;
    let system_after = system_memory()?;
    if process_after.vm_swap_bytes != 0
        || system_after
            .swap_used_bytes
            .saturating_sub(system_before.swap_used_bytes)
            != 0
        || system_after.mem_available_bytes < MIN_AVAILABLE_BYTES
    {
        bail!(
            "host final memory/swap guard failed: process={process_after:?} system_before={system_before:?} system_after={system_after:?}"
        );
    }

    let record = EvidenceRecord {
        schema: "gpt-oss-rs.heterogeneous-construction/v4",
        mode: cli.mode,
        captured_unix_ms: now_unix_ms(),
        repository_head: command_text("git", &["rev-parse", "HEAD"])?,
        executable_sha256: hash_file(&std::env::current_exe()?)?,
        command: std::env::args().collect(),
        toolchain: ToolchainRecord {
            rustc: command_text("rustc", &["--version"])?,
            cargo: command_text("cargo", &["--version"])?,
            nvcc: command_text("nvcc", &["--version"])?,
            nvidia_driver: command_text(
                "nvidia-smi",
                &["--query-gpu=driver_version", "--format=csv,noheader"],
            )?,
        },
        cache_root: CACHE_ROOT.into(),
        cache_bytes_before,
        cache_bytes_after,
        process_before,
        process_after,
        system_before,
        system_after,
        checkpoint_20b: checkpoint_record_20b,
        checkpoint_120b: checkpoint_record_120b,
        placement_20b: placement_record_20b,
        placement_120b: placement_record_120b,
        envelope_20b,
        envelope_120b,
        construction,
        fault_campaign,
        h8_campaign,
        protected_nvme,
        passed: true,
    };
    let mut bytes = serde_json::to_vec_pretty(&record)?;
    bytes.push(b'\n');
    atomic_write(&cli.output, &bytes)?;
    Ok(())
}

fn run_construction(
    checkpoint: GptOssCheckpointView,
    manifest: &GptOssExpertPlacementManifestV1,
    cache_root: &Path,
    stable: &[StableCudaDeviceId; 2],
    swap_baseline: u64,
) -> Result<ConstructionRecord> {
    let cuda_memory_before = cuda_memory(stable)?;
    let started = Instant::now();
    let mut snapshots = Vec::new();
    let constructor = OwnerSelectiveConstructor::new(cache_root);
    let mut model = constructor.construct(checkpoint, manifest, |ledger| {
        let snapshot = resource_snapshot(started, ledger.clone())
            .map_err(|error| LLMError::ModelError(format!("H3 resource snapshot: {error:#}")))?;
        enforce_resource_guards(&snapshot, swap_baseline)
            .map_err(|error| LLMError::ModelError(format!("H3 resource guard: {error:#}")))?;
        snapshots.push(snapshot);
        Ok(())
    })?;
    model.drain()?;
    let cuda_while = model.device_memory_info()?;
    let cuda_memory_while_loaded = [
        CudaMemoryRecord {
            free_bytes: cuda_while[0].0,
            total_bytes: cuda_while[0].1,
        },
        CudaMemoryRecord {
            free_bytes: cuda_while[1].0,
            total_bytes: cuda_while[1].1,
        },
    ];
    let final_ledger = model.ledger().clone();
    let dense_tensors = model.layer_owner_dense().len();
    let dense_device_bytes = model
        .layer_owner_dense()
        .iter()
        .map(|tensor| tensor.device_bytes() as u64)
        .sum();
    let cpu_layer_files = model
        .cpu_layer_records()
        .map(|(layer, record)| {
            Ok(CpuLayerFileRecord {
                layer: *layer,
                expert_ids: record.expert_ids().to_vec(),
                payload_bytes: record.payload_bytes(),
                file_bytes: std::fs::metadata(record.path())?.len(),
                relative_path: record
                    .path()
                    .strip_prefix(cache_root)?
                    .to_string_lossy()
                    .into_owned(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    drop(model);
    let cuda_memory_after = cuda_memory(stable)?;
    let cleanup_within_tolerance =
        cuda_cleanup_within_tolerance(&cuda_memory_before, &cuda_memory_after);
    if !cleanup_within_tolerance {
        bail!("CUDA memory did not return to the configured cleanup tolerance");
    }
    let partial_artifacts_after = partial_artifacts(cache_root)?;
    if !partial_artifacts_after.is_empty() {
        bail!("owner cache retained partial artifacts: {partial_artifacts_after:?}");
    }
    let after = resource_snapshot(started, final_ledger.clone())?;
    enforce_resource_guards(&after, swap_baseline)?;
    snapshots.push(after);
    Ok(ConstructionRecord {
        elapsed_ms: started.elapsed().as_millis(),
        snapshots,
        final_ledger,
        dense_tensors,
        dense_device_bytes,
        cpu_layer_files,
        cuda_memory_while_loaded,
        cuda_memory_before,
        cuda_memory_after,
        cleanup_within_tolerance,
        partial_artifacts_after,
    })
}

#[cfg(feature = "heterogeneous-test-faults")]
fn run_fault_campaign(
    model_path: &Path,
    manifest: &GptOssExpertPlacementManifestV1,
    cache_root: &Path,
    stable: &[StableCudaDeviceId; 2],
    swap_baseline: u64,
) -> Result<FaultCampaignRecord> {
    const STAGES: [ConstructionStage; 8] = [
        ConstructionStage::Identity,
        ConstructionStage::RuntimeBaseline,
        ConstructionStage::Mappings,
        ConstructionStage::LayerOwnerDense,
        ConstructionStage::GpuExperts,
        ConstructionStage::CpuExperts,
        ConstructionStage::ExecutionReserve,
        ConstructionStage::Publish,
    ];

    if owner_selective_pinned_current_bytes() != 0 || owner_selective_pinned_high_water_bytes() != 0
    {
        bail!("fault campaign did not start with an empty pinned-lease tracker");
    }

    let constructor = OwnerSelectiveConstructor::new(cache_root);
    let mut records = Vec::with_capacity(STAGES.len());
    let mut cache_after_cpu_fault = None;
    for stage in STAGES {
        let checkpoint = GptOssCheckpointView::open(model_path)?;
        let cuda_memory_before = cuda_memory(stable)?;
        let cache_bytes_before = cache_bytes(cache_root)?;
        let started = Instant::now();
        let mut observed_stages = Vec::new();
        let result = constructor.construct_with_fault(checkpoint, manifest, stage, |ledger| {
            let snapshot = resource_snapshot(started, ledger.clone())
                .map_err(|error| LLMError::ModelError(format!("H3 fault snapshot: {error:#}")))?;
            enforce_resource_guards(&snapshot, swap_baseline)
                .map_err(|error| LLMError::ModelError(format!("H3 fault guard: {error:#}")))?;
            observed_stages.push(ledger.stage);
            Ok(())
        });
        let injected_error = match result {
            Ok(mut model) => {
                model.drain()?;
                drop(model);
                bail!("constructor unexpectedly succeeded with fault at {stage:?}");
            }
            Err(error) => error.to_string(),
        };
        let expected_error = format!("injected owner-selective construction failure at {stage:?}");
        if !injected_error.contains(&expected_error) {
            bail!("constructor fault at {stage:?} returned a different error: {injected_error}");
        }
        if observed_stages.last().copied() != Some(stage) {
            bail!(
                "constructor fault at {stage:?} was not observed at the real stage: {observed_stages:?}"
            );
        }

        let pinned_current_bytes_after = owner_selective_pinned_current_bytes();
        let pinned_high_water_bytes_after = owner_selective_pinned_high_water_bytes();
        if pinned_current_bytes_after != 0
            || pinned_high_water_bytes_after > OWNER_SELECTIVE_PINNED_UPLOAD_BYTES as u64
        {
            bail!(
                "pinned lease tracker failed after {stage:?}: current={pinned_current_bytes_after} high_water={pinned_high_water_bytes_after}"
            );
        }
        let cuda_memory_after = cuda_memory(stable)?;
        let cleanup_within_tolerance =
            cuda_cleanup_within_tolerance(&cuda_memory_before, &cuda_memory_after);
        if !cleanup_within_tolerance {
            bail!("CUDA memory failed to return after fault at {stage:?}");
        }
        let partial_artifacts_after = partial_artifacts(cache_root)?;
        if !partial_artifacts_after.is_empty() {
            bail!(
                "partial CPU cache artifacts remained after fault at {stage:?}: {partial_artifacts_after:?}"
            );
        }
        let cache_bytes_after = cache_bytes(cache_root)?;
        if stage != ConstructionStage::CpuExperts
            && cache_after_cpu_fault.is_none()
            && cache_bytes_after != cache_bytes_before
        {
            bail!("cache changed before the CPU publication stage at {stage:?}");
        }
        if stage == ConstructionStage::CpuExperts {
            if cache_bytes_after < cache_bytes_before {
                bail!("CPU-stage fault removed a published cache record");
            }
            cache_after_cpu_fault = Some(cache_bytes_after);
        } else if let Some(expected) = cache_after_cpu_fault {
            if cache_bytes_after != expected || cache_bytes_before != expected {
                bail!("a post-CPU fault rewrote the published cache record at {stage:?}");
            }
        }
        let system_after = system_memory()?;
        if process_memory()?.vm_swap_bytes != 0
            || system_after.swap_used_bytes.saturating_sub(swap_baseline) != 0
            || system_after.mem_available_bytes < MIN_AVAILABLE_BYTES
        {
            bail!("host resource guard failed after fault at {stage:?}");
        }
        records.push(ConstructionFaultRecord {
            stage,
            elapsed_ms: started.elapsed().as_millis(),
            observed_stages,
            injected_error,
            cuda_memory_before,
            cuda_memory_after,
            cleanup_within_tolerance,
            pinned_current_bytes_after,
            pinned_high_water_bytes_after,
            cache_bytes_before,
            cache_bytes_after,
            cache_files_after: owner_cache_files(cache_root)?,
            partial_artifacts_after,
            system_after,
        });
    }

    let clean_cache_bytes_before = cache_bytes(cache_root)?;
    let clean_checkpoint = GptOssCheckpointView::open(model_path)?;
    let clean_construction = run_construction(
        clean_checkpoint,
        manifest,
        cache_root,
        stable,
        swap_baseline,
    )?;
    let clean_cache_bytes_after = cache_bytes(cache_root)?;
    let clean_reused_published_record = clean_cache_bytes_after == clean_cache_bytes_before;
    if !clean_reused_published_record {
        bail!("clean construction did not reuse the identity-valid published CPU record");
    }
    let final_pinned_current_bytes = owner_selective_pinned_current_bytes();
    let pinned_high_water_bytes = owner_selective_pinned_high_water_bytes();
    if final_pinned_current_bytes != 0
        || pinned_high_water_bytes != OWNER_SELECTIVE_PINNED_UPLOAD_BYTES as u64
    {
        bail!(
            "final pinned lease accounting failed: current={final_pinned_current_bytes} high_water={pinned_high_water_bytes}"
        );
    }
    let final_partial_artifacts = partial_artifacts(cache_root)?;
    if !final_partial_artifacts.is_empty() {
        bail!("fault campaign retained partial artifacts: {final_partial_artifacts:?}");
    }
    Ok(FaultCampaignRecord {
        stages: records,
        clean_construction,
        clean_cache_bytes_before,
        clean_cache_bytes_after,
        clean_reused_published_record,
        pinned_high_water_bytes,
        final_pinned_current_bytes,
        final_cache_files: owner_cache_files(cache_root)?,
        final_partial_artifacts,
    })
}

#[cfg(feature = "heterogeneous-test-faults")]
#[allow(clippy::too_many_arguments)]
fn run_h8_campaign(
    model_path: &Path,
    manifest: &GptOssExpertPlacementManifestV1,
    admission: H8AdmissionRecord,
    envelope: &OwnerSelectiveEnvelope,
    cache_root: &Path,
    stable: &[StableCudaDeviceId; 2],
    swap_baseline: u64,
    placement_path: &Path,
) -> Result<H8CampaignRecord> {
    let placement_hash = manifest.sha256()?;
    let cache_files_before = owner_cache_files(cache_root)?;
    let identity_fragment = format!(
        "owner-x8-v2/{}/{placement_hash}/",
        manifest.model.mapping_sha256
    );
    if cache_files_before
        .iter()
        .any(|path| path.contains(&identity_fragment))
    {
        bail!("H8 cold construction found an existing 120B placement cache record");
    }
    let cache_bytes_before = cache_bytes(cache_root)?;
    let disk_available_before_bytes = disk_available_bytes(cache_root)?;
    let header_allowance = u64::from(manifest.model.num_layers) * 1024 * 1024;
    let disk_required_for_cpu_payload_and_safety_bytes = envelope
        .cpu_x8_record_bytes
        .checked_add(header_allowance)
        .and_then(|bytes| bytes.checked_add(H8_MIN_DISK_REMAINING_BYTES))
        .context("H8 disk requirement overflows")?;
    if disk_available_before_bytes < disk_required_for_cpu_payload_and_safety_bytes {
        bail!(
            "H8 disk guard failed: available={disk_available_before_bytes} required={disk_required_for_cpu_payload_and_safety_bytes}"
        );
    }
    if owner_selective_pinned_current_bytes() != 0 {
        bail!("H8 did not start with an empty construction pinned tracker");
    }

    let cold = run_construction(
        GptOssCheckpointView::open(model_path)?,
        manifest,
        cache_root,
        stable,
        swap_baseline,
    )?;
    let cold_loaded_reserve = verify_h8_construction(
        &cold,
        admission.layer_owner_experts,
        admission.remote_gpu_experts,
        admission.cpu_experts,
    )?;
    let cache_bytes_after_cold = cache_bytes(cache_root)?;
    if cache_bytes_after_cold <= cache_bytes_before {
        bail!("H8 cold construction did not publish owner-filtered CPU records");
    }

    let publish_fault =
        run_h8_publish_fault(model_path, manifest, cache_root, stable, swap_baseline)?;
    let cache_bytes_after_fault = cache_bytes(cache_root)?;
    if cache_bytes_after_fault != cache_bytes_after_cold {
        bail!("H8 publish fault changed immutable CPU cache bytes");
    }

    let warm = run_construction(
        GptOssCheckpointView::open(model_path)?,
        manifest,
        cache_root,
        stable,
        swap_baseline,
    )?;
    let warm_loaded_reserve = verify_h8_construction(
        &warm,
        admission.layer_owner_experts,
        admission.remote_gpu_experts,
        admission.cpu_experts,
    )?;
    let cache_bytes_after_warm = cache_bytes(cache_root)?;
    if cache_bytes_after_warm != cache_bytes_after_cold {
        bail!("H8 warm construction rewrote immutable CPU cache bytes");
    }

    let final_pinned_current_bytes = owner_selective_pinned_current_bytes();
    let pinned_high_water_bytes = owner_selective_pinned_high_water_bytes();
    if final_pinned_current_bytes != 0
        || pinned_high_water_bytes > OWNER_SELECTIVE_PINNED_UPLOAD_BYTES as u64
    {
        bail!(
            "H8 pinned accounting failed: current={final_pinned_current_bytes} high_water={pinned_high_water_bytes}"
        );
    }
    let final_partial_artifacts = partial_artifacts(cache_root)?;
    if !final_partial_artifacts.is_empty() {
        bail!("H8 retained partial artifacts: {final_partial_artifacts:?}");
    }
    let final_cache_files = owner_cache_files(cache_root)?;
    let expected_new_files = cold.cpu_layer_files.len();
    let actual_new_files = final_cache_files
        .iter()
        .filter(|path| path.contains(&identity_fragment))
        .count();
    if actual_new_files != expected_new_files {
        bail!(
            "H8 owner-cache file count differs: expected={expected_new_files} actual={actual_new_files}"
        );
    }
    let process_after = process_memory()?;
    let system_after = system_memory()?;
    enforce_process_system_guards(&process_after, &system_after, swap_baseline, "H8 final")?;

    Ok(H8CampaignRecord {
        admission,
        placement_path: placement_path.to_string_lossy().into_owned(),
        disk_available_before_bytes,
        disk_required_for_cpu_payload_and_safety_bytes,
        cache_bytes_before,
        cache_bytes_after_cold,
        cache_bytes_after_fault,
        cache_bytes_after_warm,
        cold,
        cold_loaded_reserve,
        publish_fault,
        warm,
        warm_loaded_reserve,
        pinned_high_water_bytes,
        final_pinned_current_bytes,
        final_cache_files,
        final_partial_artifacts,
        process_after,
        system_after,
    })
}

#[cfg(feature = "heterogeneous-test-faults")]
fn run_h8_publish_fault(
    model_path: &Path,
    manifest: &GptOssExpertPlacementManifestV1,
    cache_root: &Path,
    stable: &[StableCudaDeviceId; 2],
    swap_baseline: u64,
) -> Result<H8PublishFaultRecord> {
    let stage = ConstructionStage::Publish;
    let cache_bytes_before = cache_bytes(cache_root)?;
    let cuda_memory_before = cuda_memory(stable)?;
    let started = Instant::now();
    let mut observed_stages = Vec::new();
    let mut loaded_reserve_at_publish = None;
    let result = OwnerSelectiveConstructor::new(cache_root).construct_with_fault(
        GptOssCheckpointView::open(model_path)?,
        manifest,
        stage,
        |ledger| {
            let snapshot = resource_snapshot(started, ledger.clone()).map_err(|error| {
                LLMError::ModelError(format!("H8 publish-fault snapshot: {error:#}"))
            })?;
            enforce_resource_guards(&snapshot, swap_baseline).map_err(|error| {
                LLMError::ModelError(format!("H8 publish-fault resource guard: {error:#}"))
            })?;
            observed_stages.push(ledger.stage);
            if ledger.stage == ConstructionStage::Publish {
                let memory = cuda_memory(stable).map_err(|error| {
                    LLMError::ModelError(format!(
                        "H8 publish-fault loaded CUDA snapshot: {error:#}"
                    ))
                })?;
                loaded_reserve_at_publish = Some(
                    h8_loaded_reserve_record("H8 publish fault", memory).map_err(|error| {
                        LLMError::ModelError(format!(
                            "H8 publish-fault loaded reserve gate: {error:#}"
                        ))
                    })?,
                );
            }
            Ok(())
        },
    );
    let injected_error = match result {
        Ok(mut model) => {
            model.drain()?;
            drop(model);
            bail!("H8 constructor unexpectedly published despite the injected fault");
        }
        Err(error) => error.to_string(),
    };
    if !injected_error.contains("injected owner-selective construction failure at Publish")
        || observed_stages.last().copied() != Some(ConstructionStage::Publish)
    {
        bail!("H8 publish fault did not reach the real publish boundary: {injected_error}");
    }
    let loaded_reserve_at_publish = loaded_reserve_at_publish
        .context("H8 publish fault did not retain its loaded-stage CUDA snapshot")?;
    let pinned_current_bytes_after = owner_selective_pinned_current_bytes();
    if pinned_current_bytes_after != 0 {
        bail!("H8 publish fault retained a construction pinned lease");
    }
    let cuda_memory_after = cuda_memory(stable)?;
    let cleanup_within_tolerance =
        cuda_cleanup_within_tolerance(&cuda_memory_before, &cuda_memory_after);
    if !cleanup_within_tolerance {
        bail!("H8 publish fault retained GPU allocations");
    }
    let cache_bytes_after = cache_bytes(cache_root)?;
    if cache_bytes_after != cache_bytes_before {
        bail!("H8 publish fault changed an identity-valid CPU cache record");
    }
    let partial_artifacts_after = partial_artifacts(cache_root)?;
    if !partial_artifacts_after.is_empty() {
        bail!("H8 publish fault retained partial cache artifacts");
    }
    let process_after = process_memory()?;
    let system_after = system_memory()?;
    enforce_process_system_guards(
        &process_after,
        &system_after,
        swap_baseline,
        "H8 publish fault",
    )?;
    Ok(H8PublishFaultRecord {
        stage,
        elapsed_ms: started.elapsed().as_millis(),
        observed_stages,
        injected_error,
        cuda_memory_before,
        loaded_reserve_at_publish,
        cuda_memory_after,
        cleanup_within_tolerance,
        pinned_current_bytes_after,
        cache_bytes_before,
        cache_bytes_after,
        partial_artifacts_after,
        process_after,
        system_after,
    })
}

#[cfg(feature = "heterogeneous-test-faults")]
fn verify_h8_construction(
    construction: &ConstructionRecord,
    layer_owner_experts: u32,
    remote_gpu_experts: u32,
    cpu_experts: u32,
) -> Result<H8LoadedReserveRecord> {
    let ledger = &construction.final_ledger;
    let cpu_file_experts =
        construction
            .cpu_layer_files
            .iter()
            .try_fold(0_u32, |count, layer| {
                count
                    .checked_add(u32::try_from(layer.expert_ids.len())?)
                    .context("H8 CPU file expert count overflows")
            })?;
    if ledger.stage != ConstructionStage::Publish
        || ledger.layer_owner_experts != layer_owner_experts
        || ledger.remote_gpu_experts != remote_gpu_experts
        || ledger.cpu_experts != cpu_experts
        || cpu_file_experts != cpu_experts
        || !construction.cleanup_within_tolerance
        || !construction.partial_artifacts_after.is_empty()
    {
        bail!("H8 construction ledger, cache, or cleanup invariant failed");
    }
    h8_loaded_reserve_record("H8 construction", construction.cuda_memory_while_loaded)
}

#[cfg(feature = "heterogeneous-test-faults")]
fn h8_loaded_reserve_record(
    label: &str,
    cuda_memory_while_loaded: [CudaMemoryRecord; 2],
) -> Result<H8LoadedReserveRecord> {
    let reserve_plus_margin_bytes_per_gpu = OWNER_SELECTIVE_GPU_RESERVE_BYTES
        .checked_add(H8_ADMISSION_MARGIN_BYTES)
        .context("H8 reserve plus admission margin overflows")?;
    let free = [
        u64::try_from(cuda_memory_while_loaded[0].free_bytes)?,
        u64::try_from(cuda_memory_while_loaded[1].free_bytes)?,
    ];
    let execution_reserve_met = free.map(|bytes| bytes >= OWNER_SELECTIVE_GPU_RESERVE_BYTES);
    let reserve_plus_margin_met = free.map(|bytes| bytes >= reserve_plus_margin_bytes_per_gpu);
    if !execution_reserve_met.into_iter().all(|met| met)
        || !reserve_plus_margin_met.into_iter().all(|met| met)
    {
        bail!(
            "{label} eroded the H8 GPU execution reserve or admission margin: free={free:?} reserve={} margin={}",
            OWNER_SELECTIVE_GPU_RESERVE_BYTES,
            H8_ADMISSION_MARGIN_BYTES
        );
    }
    Ok(H8LoadedReserveRecord {
        cuda_memory_while_loaded,
        execution_reserve_bytes_per_gpu: OWNER_SELECTIVE_GPU_RESERVE_BYTES,
        admission_margin_bytes_per_gpu: H8_ADMISSION_MARGIN_BYTES,
        reserve_plus_margin_bytes_per_gpu,
        execution_reserve_met,
        reserve_plus_margin_met,
    })
}

fn cuda_cleanup_within_tolerance(
    before: &[CudaMemoryRecord; 2],
    after: &[CudaMemoryRecord; 2],
) -> bool {
    before.iter().zip(after).all(|(before, after)| {
        before.free_bytes.abs_diff(after.free_bytes) <= GPU_CLEANUP_TOLERANCE_BYTES
            && before.total_bytes == after.total_bytes
    })
}

fn proof_manifest_20b(
    checkpoint: &GptOssCheckpointView,
    devices: &[StableCudaDeviceId; 2],
) -> Result<GptOssExpertPlacementManifestV1> {
    build_manifest(checkpoint, devices, |key, _| {
        if key
            == (GptOssExpertKey {
                layer: 0,
                expert: 21,
            })
        {
            0
        } else if key
            == (GptOssExpertKey {
                layer: 0,
                expert: 22,
            })
        {
            2
        } else {
            1
        }
    })
}

fn existence_manifest_120b(
    checkpoint: &GptOssCheckpointView,
    devices: &[StableCudaDeviceId; 2],
) -> Result<GptOssExpertPlacementManifestV1> {
    build_manifest(checkpoint, devices, |_, flat| {
        if flat < 1_299 {
            1
        } else if flat < 1_299 + 1_620 {
            2
        } else {
            0
        }
    })
}

#[cfg(feature = "heterogeneous-test-faults")]
fn admitted_manifest_120b(
    checkpoint: &GptOssCheckpointView,
    devices: &[StableCudaDeviceId; 2],
) -> Result<(GptOssExpertPlacementManifestV1, H8AdmissionRecord)> {
    let layer_owner = CudaSelectedExpertExecutor::new(devices[0].clone())?;
    let remote = CudaSelectedExpertExecutor::new(devices[1].clone())?;
    let layer_memory = layer_owner.memory_info()?;
    let remote_memory = remote.memory_info()?;
    let simultaneous_executor_memory = [
        CudaMemoryRecord {
            free_bytes: layer_memory.0,
            total_bytes: layer_memory.1,
        },
        CudaMemoryRecord {
            free_bytes: remote_memory.0,
            total_bytes: remote_memory.1,
        },
    ];

    let capacity = |label: &str, free: usize, dense: u64| -> Result<u32> {
        let usable = u64::try_from(free)
            .context("CUDA free byte count does not fit u64")?
            .checked_sub(dense)
            .and_then(|bytes| bytes.checked_sub(OWNER_SELECTIVE_GPU_RESERVE_BYTES))
            .and_then(|bytes| bytes.checked_sub(H8_ADMISSION_MARGIN_BYTES))
            .with_context(|| format!("{label} has no capacity after dense/reserve/margin"))?;
        u32::try_from(usable / CONSERVATIVE_OWNER_EXPERT_BYTES)
            .with_context(|| format!("{label} expert capacity does not fit u32"))
    };
    let layer_owner_experts = capacity(
        "layer-owner GPU",
        layer_memory.0,
        checkpoint.non_expert_payload_bytes(),
    )?;
    let remote_gpu_experts = capacity("remote GPU", remote_memory.0, 0)?;
    let total_experts = u32::try_from(
        checkpoint
            .config()
            .num_hidden_layers
            .checked_mul(checkpoint.config().num_experts)
            .context("H8 total expert count overflows")?,
    )?;
    let cpu_experts = total_experts
        .checked_sub(layer_owner_experts)
        .and_then(|remaining| remaining.checked_sub(remote_gpu_experts))
        .context("measured GPU capacities exceed the model expert count")?;
    if cpu_experts == 0 {
        bail!("H8 proof placement must retain a nonempty CPU owner set");
    }

    // Role order throughout the hash policy and its retained coverage record is
    // CPU, layer-owner GPU, remote GPU. Capacity measurement fixes only the
    // exact global quotas; it must not determine contiguous ownership.
    let quotas = [cpu_experts, layer_owner_experts, remote_gpu_experts];
    let manifest = quota_balanced_hashed_manifest(checkpoint, devices, quotas, H8_POLICY_SEED)?;
    let rebuilt = quota_balanced_hashed_manifest(checkpoint, devices, quotas, H8_POLICY_SEED)?;
    let manifest_json = manifest.stable_json()?;
    let deterministic_manifest_rebuild_matched = manifest_json == rebuilt.stable_json()?;
    if !deterministic_manifest_rebuild_matched || manifest.sha256()? != rebuilt.sha256()? {
        bail!("H8 quota-balanced manifest is not deterministic");
    }
    let coverage = verify_h8_manifest(checkpoint, &manifest, devices, quotas)?;
    let layer_owner_required_free_bytes = checkpoint
        .non_expert_payload_bytes()
        .checked_add(u64::from(layer_owner_experts) * CONSERVATIVE_OWNER_EXPERT_BYTES)
        .and_then(|bytes| bytes.checked_add(OWNER_SELECTIVE_GPU_RESERVE_BYTES))
        .context("H8 layer-owner admission byte count overflows")?;
    let remote_gpu_required_free_bytes = u64::from(remote_gpu_experts)
        .checked_mul(CONSERVATIVE_OWNER_EXPERT_BYTES)
        .and_then(|bytes| bytes.checked_add(OWNER_SELECTIVE_GPU_RESERVE_BYTES))
        .context("H8 remote admission byte count overflows")?;
    let record = H8AdmissionRecord {
        simultaneous_executor_memory,
        additional_margin_bytes_per_gpu: H8_ADMISSION_MARGIN_BYTES,
        layer_owner_experts,
        remote_gpu_experts,
        cpu_experts,
        layer_owner_required_free_bytes,
        remote_gpu_required_free_bytes,
        layer_owner_headroom_bytes: u64::try_from(layer_memory.0)?
            .checked_sub(layer_owner_required_free_bytes)
            .context("H8 layer-owner headroom underflows")?,
        remote_gpu_headroom_bytes: u64::try_from(remote_memory.0)?
            .checked_sub(remote_gpu_required_free_bytes)
            .context("H8 remote headroom underflows")?,
        assignment_algorithm: "sha256-quota-balanced-per-layer-v1",
        policy_seed: H8_POLICY_SEED,
        manifest_sha256: coverage.manifest_sha256,
        deterministic_manifest_rebuild_matched,
        all_expert_keys_exactly_once: coverage.all_expert_keys_exactly_once,
        layers_with_all_three_owners: coverage.layers_with_all_three_owners,
        per_layer_owner_order: ["cpu", "layer_owner_gpu", "remote_gpu"],
        per_layer_owner_counts: coverage.per_layer_owner_counts,
        per_layer_owner_minimums: coverage.per_layer_owner_minimums,
        per_layer_owner_maximums: coverage.per_layer_owner_maximums,
    };
    drop(remote);
    drop(layer_owner);
    Ok((manifest, record))
}

#[cfg(feature = "heterogeneous-test-faults")]
fn quota_balanced_hashed_manifest(
    checkpoint: &GptOssCheckpointView,
    devices: &[StableCudaDeviceId; 2],
    quotas: [u32; 3],
    policy_seed: u64,
) -> Result<GptOssExpertPlacementManifestV1> {
    let config = checkpoint.config();
    let owner_roles = quota_balanced_owner_roles(
        config.num_hidden_layers,
        config.num_experts,
        quotas,
        policy_seed,
        checkpoint.mapping_sha256(),
    )?;
    let mut manifest = build_manifest(checkpoint, devices, |_, flat| owner_roles[flat])?;
    manifest.policy_seed = policy_seed;
    Ok(manifest)
}

#[cfg(feature = "heterogeneous-test-faults")]
fn quota_balanced_owner_roles(
    num_layers: usize,
    experts_per_layer: usize,
    quotas: [u32; 3],
    policy_seed: u64,
    mapping_sha256: &str,
) -> Result<Vec<u8>> {
    if num_layers == 0 || experts_per_layer == 0 {
        bail!("H8 hash assignment requires nonzero model dimensions");
    }
    let layers_u32 = u32::try_from(num_layers).context("H8 layer count does not fit u32")?;
    let experts_u32 =
        u32::try_from(experts_per_layer).context("H8 expert count does not fit u32")?;
    let total = layers_u32
        .checked_mul(experts_u32)
        .context("H8 hash assignment total overflows")?;
    if quotas.into_iter().sum::<u32>() != total {
        bail!("H8 hash assignment quotas do not cover the expert rectangle");
    }

    let bases = quotas.map(|quota| quota / layers_u32);
    let remainders = quotas.map(|quota| quota % layers_u32);
    let base_sum = bases.into_iter().sum::<u32>();
    let extras_per_layer = experts_u32
        .checked_sub(base_sum)
        .context("H8 per-layer quota bases exceed the expert count")?;
    if extras_per_layer > 2 || remainders.into_iter().sum::<u32>() != extras_per_layer * layers_u32
    {
        bail!("H8 three-owner quota apportionment is inconsistent");
    }

    let mut layer_order = (0..num_layers).collect::<Vec<_>>();
    layer_order.sort_by_key(|layer| {
        h8_assignment_hash(
            b"layer-extra-order",
            policy_seed,
            mapping_sha256,
            u32::try_from(*layer).unwrap_or(u32::MAX),
            0,
        )
    });
    let mut role_order = [0_usize, 1, 2];
    role_order.sort_by_key(|role| {
        h8_assignment_hash(
            b"owner-extra-order",
            policy_seed,
            mapping_sha256,
            0,
            u32::try_from(*role).unwrap_or(u32::MAX),
        )
    });

    let mut extras = vec![[false; 3]; num_layers];
    match extras_per_layer {
        0 => {}
        1 => {
            let mut cursor = 0_usize;
            for role in role_order {
                for _ in 0..remainders[role] {
                    let layer = *layer_order
                        .get(cursor)
                        .context("H8 extra-owner apportionment exhausted layers")?;
                    extras[layer][role] = true;
                    cursor += 1;
                }
            }
            if cursor != num_layers {
                bail!("H8 extra-owner apportionment did not cover every layer");
            }
        }
        2 => {
            // With three owners and two extras per layer, assigning the single
            // excluded owner gives an exact, duplicate-free complement.
            let exclusions = remainders.map(|remainder| layers_u32 - remainder);
            let mut cursor = 0_usize;
            for role in role_order {
                for _ in 0..exclusions[role] {
                    let layer = *layer_order
                        .get(cursor)
                        .context("H8 excluded-owner apportionment exhausted layers")?;
                    for (included, is_extra) in extras[layer].iter_mut().enumerate() {
                        *is_extra = included != role;
                    }
                    cursor += 1;
                }
            }
            if cursor != num_layers {
                bail!("H8 excluded-owner apportionment did not cover every layer");
            }
        }
        _ => unreachable!("validated above"),
    }

    let mut roles = vec![u8::MAX; usize::try_from(total)?];
    for layer in 0..num_layers {
        let mut layer_counts = bases;
        for role in 0..3 {
            layer_counts[role] += u32::from(extras[layer][role]);
        }
        if layer_counts.into_iter().sum::<u32>() != experts_u32
            || layer_counts.into_iter().any(|count| count == 0)
        {
            bail!("H8 per-layer quota is incomplete or lacks an owner");
        }

        let mut expert_order = (0..experts_per_layer).collect::<Vec<_>>();
        expert_order.sort_by_key(|expert| {
            h8_assignment_hash(
                b"expert-order",
                policy_seed,
                mapping_sha256,
                u32::try_from(layer).unwrap_or(u32::MAX),
                u32::try_from(*expert).unwrap_or(u32::MAX),
            )
        });
        let mut layer_role_order = [0_usize, 1, 2];
        layer_role_order.sort_by_key(|role| {
            h8_assignment_hash(
                b"layer-owner-order",
                policy_seed,
                mapping_sha256,
                u32::try_from(layer).unwrap_or(u32::MAX),
                u32::try_from(*role).unwrap_or(u32::MAX),
            )
        });
        let mut cursor = 0_usize;
        for role in layer_role_order {
            for _ in 0..layer_counts[role] {
                let expert = *expert_order
                    .get(cursor)
                    .context("H8 expert hash ordering exhausted a layer")?;
                roles[layer * experts_per_layer + expert] = u8::try_from(role)?;
                cursor += 1;
            }
        }
        if cursor != experts_per_layer {
            bail!("H8 expert hash ordering did not cover a layer");
        }
    }
    if roles.contains(&u8::MAX) {
        bail!("H8 hash assignment left an expert without an owner");
    }
    Ok(roles)
}

#[cfg(feature = "heterogeneous-test-faults")]
fn h8_assignment_hash(
    domain: &[u8],
    policy_seed: u64,
    mapping_sha256: &str,
    layer: u32,
    item: u32,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"gpt-oss-rs/het/h8/");
    digest.update(domain);
    digest.update(policy_seed.to_le_bytes());
    digest.update(mapping_sha256.as_bytes());
    digest.update(layer.to_le_bytes());
    digest.update(item.to_le_bytes());
    digest.finalize().into()
}

#[cfg(feature = "heterogeneous-test-faults")]
fn verify_h8_manifest(
    checkpoint: &GptOssCheckpointView,
    manifest: &GptOssExpertPlacementManifestV1,
    devices: &[StableCudaDeviceId; 2],
    expected_quotas: [u32; 3],
) -> Result<H8ManifestCoverage> {
    let config = checkpoint.config();
    let expected_total = config
        .num_hidden_layers
        .checked_mul(config.num_experts)
        .context("H8 manifest rectangle overflows")?;
    if manifest.assignments.len() != expected_total
        || manifest.policy_seed != H8_POLICY_SEED
        || manifest.layer_owner != devices[0]
        || manifest.remote_worker != devices[1]
    {
        bail!("H8 manifest identity or assignment cardinality is invalid");
    }

    let mut seen = BTreeSet::new();
    let mut global = [0_u32; 3];
    let mut per_layer = vec![[0_u32; 3]; config.num_hidden_layers];
    for assignment in &manifest.assignments {
        if !seen.insert(assignment.key) {
            bail!("H8 manifest repeats expert key {:?}", assignment.key);
        }
        let role = match &assignment.owner {
            ExpertOwner::Cpu { pool } if *pool == CpuPoolId(0) => 0,
            ExpertOwner::LayerOwnerGpu { device } if device == &devices[0] => 1,
            ExpertOwner::RemoteGpu { device } if device == &devices[1] => 2,
            _ => bail!("H8 manifest contains an unexpected owner identity"),
        };
        let layer = usize::from(assignment.key.layer);
        let expert = usize::from(assignment.key.expert);
        if layer >= config.num_hidden_layers || expert >= config.num_experts {
            bail!("H8 manifest contains an out-of-range expert key");
        }
        global[role] += 1;
        per_layer[layer][role] += 1;
    }
    for layer in 0..config.num_hidden_layers {
        for expert in 0..config.num_experts {
            if !seen.contains(&GptOssExpertKey {
                layer: u16::try_from(layer)?,
                expert: u16::try_from(expert)?,
            }) {
                bail!("H8 manifest lacks layer {layer} expert {expert}");
            }
        }
    }
    if global != expected_quotas
        || manifest.budgets.max_cpu_experts != expected_quotas[0]
        || manifest.budgets.max_layer_owner_experts != expected_quotas[1]
        || manifest.budgets.max_remote_gpu_experts != expected_quotas[2]
    {
        bail!("H8 manifest global owner counts differ from measured quotas");
    }

    let mut minimums = [u32::MAX; 3];
    let mut maximums = [0_u32; 3];
    let mut layers_with_all_three = 0_u16;
    for counts in &per_layer {
        if counts.iter().sum::<u32>() != u32::try_from(config.num_experts)? {
            bail!("H8 manifest per-layer owner counts are incomplete");
        }
        if counts.iter().all(|count| *count > 0) {
            layers_with_all_three = layers_with_all_three
                .checked_add(1)
                .context("H8 three-owner layer count overflows")?;
        }
        for role in 0..3 {
            minimums[role] = minimums[role].min(counts[role]);
            maximums[role] = maximums[role].max(counts[role]);
        }
    }
    if usize::from(layers_with_all_three) != config.num_hidden_layers
        || minimums
            .iter()
            .zip(maximums)
            .any(|(minimum, maximum)| maximum - minimum > 1)
    {
        bail!("H8 manifest does not provide balanced three-owner layer coverage");
    }
    Ok(H8ManifestCoverage {
        manifest_sha256: manifest.sha256()?,
        all_expert_keys_exactly_once: seen.len() == expected_total,
        layers_with_all_three_owners: layers_with_all_three,
        per_layer_owner_counts: per_layer,
        per_layer_owner_minimums: minimums,
        per_layer_owner_maximums: maximums,
    })
}

fn build_manifest<F>(
    checkpoint: &GptOssCheckpointView,
    devices: &[StableCudaDeviceId; 2],
    mut owner_for: F,
) -> Result<GptOssExpertPlacementManifestV1>
where
    F: FnMut(GptOssExpertKey, usize) -> u8,
{
    let config = checkpoint.config();
    let mut assignments = Vec::with_capacity(config.num_hidden_layers * config.num_experts);
    let mut counts = [0_u32; 3];
    for layer in 0..config.num_hidden_layers {
        for expert in 0..config.num_experts {
            let key = GptOssExpertKey {
                layer: layer as u16,
                expert: expert as u16,
            };
            let role = owner_for(key, layer * config.num_experts + expert);
            counts[role as usize] += 1;
            let owner = match role {
                0 => ExpertOwner::Cpu { pool: CpuPoolId(0) },
                1 => ExpertOwner::LayerOwnerGpu {
                    device: devices[0].clone(),
                },
                2 => ExpertOwner::RemoteGpu {
                    device: devices[1].clone(),
                },
                _ => bail!("invalid owner role"),
            };
            assignments.push(
                gpt_oss_model_runner::heterogeneous::placement::ExpertAssignment { key, owner },
            );
        }
    }
    Ok(GptOssExpertPlacementManifestV1 {
        schema: HETEROGENEOUS_PLACEMENT_SCHEMA_V1.into(),
        model: GptOssPlacementModel {
            revision: checkpoint.revision().into(),
            config_sha256: checkpoint.config_sha256().into(),
            index_sha256: checkpoint.metadata_sha256().into(),
            mapping_sha256: checkpoint.mapping_sha256().into(),
            num_layers: config.num_hidden_layers as u16,
            experts_per_layer: config.num_experts as u16,
            hidden_size: config.hidden_size as u16,
            intermediate_size: config.intermediate_size as u16,
            top_k: config.experts_per_token as u8,
        },
        layer_owner: devices[0].clone(),
        remote_worker: devices[1].clone(),
        policy: PlacementPolicyClass::Proof,
        policy_seed: 0x4845_5433,
        placement_epoch: 1,
        budgets: PlacementBudgets {
            max_cpu_experts: counts[0],
            max_layer_owner_experts: counts[1],
            max_remote_gpu_experts: counts[2],
            max_host_owner_bytes: u64::from(counts[0]) * CONSERVATIVE_OWNER_EXPERT_BYTES,
            max_layer_owner_bytes: u64::from(counts[1]) * CONSERVATIVE_OWNER_EXPERT_BYTES,
            max_remote_gpu_bytes: u64::from(counts[2]) * CONSERVATIVE_OWNER_EXPERT_BYTES,
        },
        assignments,
    })
}

fn ordered_devices() -> Result<Vec<GpuDevice>> {
    let mut devices = list_devices();
    devices.sort_by_key(|device| device.pci_bus_id);
    if devices.len() != 2
        || devices[0].pci_bus_id.map(|id| id.to_string()).as_deref() != Some("0000:19:00.0")
        || devices[1].pci_bus_id.map(|id| id.to_string()).as_deref() != Some("0000:65:00.0")
    {
        bail!("H3 requires the two pinned RTX 3090 stable PCI identities");
    }
    Ok(devices)
}

fn compare_research_mapping(checkpoint: &GptOssCheckpointView, path: &Path) -> Result<String> {
    let expected: ResearchMappingDocument = serde_json::from_slice(&std::fs::read(path)?)?;
    if expected.native_tensor_count != checkpoint.config().native_tensor_count()
        || expected.runtime_tensor_count != checkpoint.config().runtime_tensor_count()
        || expected.mapping_count != checkpoint.config().runtime_tensor_count()
    {
        bail!("research mapping cardinality differs from native checkpoint");
    }
    let expected = expected
        .mappings
        .into_iter()
        .map(|mapping| (mapping.runtime.clone(), mapping))
        .collect::<BTreeMap<_, _>>();
    let actual = checkpoint
        .mappings()
        .map(|mapping| (mapping.runtime.clone(), mapping))
        .collect::<BTreeMap<_, _>>();
    if expected.len() != actual.len() {
        bail!("research/native runtime mapping lengths differ");
    }
    let mut canonical = Vec::new();
    for (runtime, actual) in actual {
        let expected = expected
            .get(&runtime)
            .with_context(|| format!("research mapping lacks {runtime}"))?;
        let expected_runtime_shape = expected
            .runtime_shape
            .as_ref()
            .or(expected.runtime_shape_derived.as_ref())
            .context("research mapping has no runtime shape")?;
        if expected.native != actual.native
            || expected.runtime != actual.runtime
            || expected.native_shard != actual.native_shard
            || expected.native_slice != actual.native_slice
            || expected.dtype != actual.dtype
            || expected_runtime_shape != &actual.runtime_shape
            || expected.bytes != actual.bytes
            || expected
                .native_shape
                .as_ref()
                .is_some_and(|shape| shape != &actual.native_shape)
        {
            bail!("research mapping differs at runtime tensor {runtime}");
        }
        canonical.extend_from_slice(runtime.as_bytes());
        canonical.extend_from_slice(&serde_json::to_vec(actual)?);
    }
    Ok(hash_bytes(&canonical))
}

fn assert_checkpoint_bytes(
    checkpoint: &GptOssCheckpointView,
    mapped: u64,
    experts: u64,
    non_experts: u64,
) -> Result<()> {
    if checkpoint.mapped_payload_bytes() != mapped
        || checkpoint.expert_payload_bytes() != experts
        || checkpoint.non_expert_payload_bytes() != non_experts
    {
        bail!("native checkpoint payload arithmetic differs from the Phase 1 envelope");
    }
    Ok(())
}

fn verify_120b_envelope(envelope: &OwnerSelectiveEnvelope) -> Result<()> {
    if envelope.layer_owner_experts != 1_299
        || envelope.remote_gpu_experts != 1_620
        || envelope.cpu_experts != 1_689
        || envelope.host_conservative_owner_bytes != 22_385_600_640
        || envelope.layer_owner_conservative_admission_bytes != 17_216_634_240
        || envelope.remote_gpu_conservative_admission_bytes != 21_471_091_200
        || envelope.layer_owner_execution_reserve_bytes != OWNER_SELECTIVE_GPU_RESERVE_BYTES
        || envelope.remote_gpu_execution_reserve_bytes != OWNER_SELECTIVE_GPU_RESERVE_BYTES
    {
        bail!("120B existence envelope differs from the Phase 1/2 proof");
    }
    Ok(())
}

fn verify_execution_reserve_plan(envelope: &OwnerSelectiveEnvelope) -> Result<()> {
    let plan = &envelope.execution_reserve_plan;
    plan.validate().context("execution reserve plan")?;
    if plan.disposition != ExecutionReserveDisposition::PostExecutorAdmissionRuntimePlanReviewed
        || plan.context_cap as usize != OWNER_SELECTIVE_PROOF_CONTEXT_CAP
        || plan.max_dispatch_rows != 1
        || plan.decode_pinned_relay_raw_capacity_bytes != 74_944
        || plan.decode_pinned_relay_cap_bytes != 128 * 1024
        || plan.decode_pinned_relay_raw_capacity_bytes > plan.decode_pinned_relay_cap_bytes
        || plan.decode_pinned_relay_materialized_at_construction
        || plan.prefill_pinned_relay_cap_bytes != 8 * 1024 * 1024
        || plan.prefill_pinned_relay_materialized_at_construction
    {
        bail!("execution reserve policy differs from the reviewed proof policy");
    }
    for (label, device) in [
        ("layer owner", &plan.layer_owner),
        ("remote GPU", &plan.remote_gpu),
    ] {
        if device.materialized_before_admission_bytes != device.selected_expert_executor_bytes
            || device
                .materialized_before_admission_bytes
                .checked_add(device.reviewed_deferred_after_admission_bytes)
                != Some(device.planned_owned_bytes)
            || device
                .reviewed_deferred_after_admission_bytes
                .checked_add(device.runtime_and_safety_remainder_bytes)
                != Some(device.reserve_cap_bytes)
            || device.reserve_cap_bytes != OWNER_SELECTIVE_GPU_RESERVE_BYTES
        {
            bail!("{label} execution reserve measurement boundary is inconsistent");
        }
    }
    Ok(())
}

#[cfg(feature = "heterogeneous-test-faults")]
fn verify_h8_envelope(
    envelope: &OwnerSelectiveEnvelope,
    admission: &H8AdmissionRecord,
) -> Result<()> {
    if envelope.layer_owner_experts != admission.layer_owner_experts
        || envelope.remote_gpu_experts != admission.remote_gpu_experts
        || envelope.cpu_experts != admission.cpu_experts
        || envelope.layer_owner_required_free_bytes()? != admission.layer_owner_required_free_bytes
        || envelope.remote_gpu_required_free_bytes()? != admission.remote_gpu_required_free_bytes
        || admission.layer_owner_headroom_bytes < H8_ADMISSION_MARGIN_BYTES
        || admission.remote_gpu_headroom_bytes < H8_ADMISSION_MARGIN_BYTES
        || u64::from(envelope.layer_owner_experts)
            + u64::from(envelope.remote_gpu_experts)
            + u64::from(envelope.cpu_experts)
            != 4_608
    {
        bail!("H8 measured admission and owner-selective envelope disagree");
    }
    Ok(())
}

fn checkpoint_record(
    checkpoint: &GptOssCheckpointView,
    mapping_equivalence_sha256: String,
) -> CheckpointRecord {
    CheckpointRecord {
        source: checkpoint.source_root().to_string_lossy().into_owned(),
        revision: checkpoint.revision().into(),
        config_sha256: checkpoint.config_sha256().into(),
        metadata_sha256: checkpoint.metadata_sha256().into(),
        mapping_sha256: checkpoint.mapping_sha256().into(),
        mapping_equivalence_sha256,
        native_tensor_count: checkpoint.config().native_tensor_count(),
        runtime_tensor_count: checkpoint.config().runtime_tensor_count(),
        mapped_payload_bytes: checkpoint.mapped_payload_bytes(),
        expert_payload_bytes: checkpoint.expert_payload_bytes(),
        non_expert_payload_bytes: checkpoint.non_expert_payload_bytes(),
    }
}

fn placement_record(
    manifest: &GptOssExpertPlacementManifestV1,
    resolved: &gpt_oss_model_runner::heterogeneous::ResolvedExpertPlacement,
) -> PlacementRecord {
    PlacementRecord {
        manifest_sha256: resolved.manifest_hash().into(),
        policy: "proof",
        layer_owner_pci: manifest.layer_owner.pci_bus_id.to_string(),
        remote_gpu_pci: manifest.remote_worker.pci_bus_id.to_string(),
        cpu_experts: resolved.counts().cpu,
        layer_owner_experts: resolved.counts().layer_owner_gpu,
        remote_gpu_experts: resolved.counts().remote_gpu,
    }
}

fn resource_snapshot(started: Instant, ledger: ConstructionLedger) -> Result<ResourceSnapshot> {
    Ok(ResourceSnapshot {
        elapsed_ms: started.elapsed().as_millis(),
        ledger,
        process: process_memory()?,
        system: system_memory()?,
        gpus: gpu_memory()?,
    })
}

fn enforce_resource_guards(snapshot: &ResourceSnapshot, swap_baseline: u64) -> Result<()> {
    enforce_process_system_guards(
        &snapshot.process,
        &snapshot.system,
        swap_baseline,
        &format!("owner-selective {:?}", snapshot.ledger.stage),
    )
}

fn enforce_process_system_guards(
    process: &ProcessMemory,
    system: &SystemMemory,
    swap_baseline: u64,
    label: &str,
) -> Result<()> {
    if process.vm_swap_bytes != 0
        || system.swap_used_bytes.saturating_sub(swap_baseline) != 0
        || system.mem_available_bytes < MIN_AVAILABLE_BYTES
        || process.vm_rss_bytes > MAX_PROCESS_RSS_BYTES
    {
        bail!("{label} resource guard failed: process={process:?} system={system:?}");
    }
    Ok(())
}

fn process_memory() -> Result<ProcessMemory> {
    let status = parse_kb_file("/proc/self/status")?;
    let rollup = parse_kb_file("/proc/self/smaps_rollup")?;
    Ok(ProcessMemory {
        vm_size_bytes: value(&status, "VmSize"),
        vm_rss_bytes: value(&status, "VmRSS"),
        vm_hwm_bytes: value(&status, "VmHWM"),
        vm_swap_bytes: value(&status, "VmSwap"),
        rollup_rss_bytes: value(&rollup, "Rss"),
        rollup_pss_bytes: value(&rollup, "Pss"),
        rollup_pss_anon_bytes: value(&rollup, "Pss_Anon"),
        rollup_pss_file_bytes: value(&rollup, "Pss_File"),
    })
}

fn system_memory() -> Result<SystemMemory> {
    let info = parse_kb_file("/proc/meminfo")?;
    let swap_total = value(&info, "SwapTotal");
    let swap_free = value(&info, "SwapFree");
    Ok(SystemMemory {
        mem_available_bytes: value(&info, "MemAvailable"),
        swap_total_bytes: swap_total,
        swap_free_bytes: swap_free,
        swap_used_bytes: swap_total.saturating_sub(swap_free),
        swap_cached_bytes: value(&info, "SwapCached"),
    })
}

fn parse_kb_file(path: &str) -> Result<BTreeMap<String, u64>> {
    let mut result = BTreeMap::new();
    for line in std::fs::read_to_string(path)?.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let Some(number) = rest.split_whitespace().next() else {
            continue;
        };
        if let Ok(kib) = number.parse::<u64>() {
            result.insert(key.into(), kib * 1024);
        }
    }
    Ok(result)
}

fn value(values: &BTreeMap<String, u64>, key: &str) -> u64 {
    values.get(key).copied().unwrap_or(0)
}

fn gpu_memory() -> Result<Vec<GpuMemory>> {
    let output = command_text(
        "nvidia-smi",
        &[
            "--query-gpu=pci.bus_id,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
    )?;
    output
        .lines()
        .map(|line| {
            let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
            if fields.len() != 3 {
                bail!("unexpected nvidia-smi memory row");
            }
            Ok(GpuMemory {
                pci_bus_id: fields[0].to_ascii_lowercase(),
                used_mib: fields[1].parse()?,
                free_mib: fields[2].parse()?,
            })
        })
        .collect()
}

#[cfg(feature = "heterogeneous-test-faults")]
fn disk_available_bytes(path: &Path) -> Result<u64> {
    let output = Command::new("df")
        .args(["--output=avail", "-B1"])
        .arg(path)
        .output()?;
    if !output.status.success() {
        bail!("df failed while checking H8 cache capacity");
    }
    String::from_utf8(output.stdout)?
        .lines()
        .filter_map(|line| line.trim().parse::<u64>().ok())
        .next_back()
        .context("df did not report H8 available bytes")
}

fn cuda_memory(stable: &[StableCudaDeviceId; 2]) -> Result<[CudaMemoryRecord; 2]> {
    let first = selected_expert_device_memory_info(&stable[0])?;
    let second = selected_expert_device_memory_info(&stable[1])?;
    Ok([
        CudaMemoryRecord {
            free_bytes: first.0,
            total_bytes: first.1,
        },
        CudaMemoryRecord {
            free_bytes: second.0,
            total_bytes: second.1,
        },
    ])
}

fn cache_bytes(path: &Path) -> Result<u64> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_symlink() {
        bail!("cache tree contains symlink: {}", path.display());
    }
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    let mut total = 0_u64;
    for entry in std::fs::read_dir(path)? {
        total = total
            .checked_add(cache_bytes(&entry?.path())?)
            .context("cache byte count overflow")?;
    }
    Ok(total)
}

fn partial_artifacts(root: &Path) -> Result<Vec<String>> {
    let mut paths = Vec::new();
    collect_paths(root, &mut paths)?;
    Ok(paths
        .into_iter()
        .filter(|path| {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            name.starts_with('.') || name.ends_with(".tmp") || name.ends_with(".lock")
        })
        .map(|path| path.to_string_lossy().into_owned())
        .collect())
}

#[cfg(feature = "heterogeneous-test-faults")]
fn owner_cache_files(root: &Path) -> Result<Vec<String>> {
    let mut paths = Vec::new();
    collect_paths(root, &mut paths)?;
    let mut files = paths
        .into_iter()
        .filter(|path| {
            path.extension().and_then(|extension| extension.to_str()) == Some("owner-x8")
        })
        .map(|path| {
            path.strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned()
        })
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

fn collect_paths(root: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    let metadata = match std::fs::symlink_metadata(root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_symlink() {
        bail!(
            "partial-artifact inspection refuses symlink: {}",
            root.display()
        );
    }
    if metadata.is_file() {
        output.push(root.to_path_buf());
        return Ok(());
    }
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        let metadata = std::fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            bail!(
                "partial-artifact inspection refuses symlink: {}",
                path.display()
            );
        }
        if metadata.is_dir() {
            collect_paths(&path, output)?;
        } else {
            output.push(path);
        }
    }
    Ok(())
}

fn protected_nvme_state() -> Result<ProtectedNvmeRecord> {
    let output = command_text("lsblk", &["-dnro", "RO,MOUNTPOINTS", "/dev/nvme1n1"])?;
    let read_only = output.split_whitespace().next() == Some("1");
    let mounted = std::fs::read_to_string("/proc/self/mountinfo")?
        .lines()
        .any(|line| line.contains("nvme1n1"));
    Ok(ProtectedNvmeRecord { read_only, mounted })
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("atomic output has no parent")?;
    std::fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("evidence"),
        std::process::id()
    ));
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(&temporary, path)?;
    Ok(())
}

fn command_text(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    if !output.status.success() {
        bail!("{program} failed with {}", output.status);
    }
    Ok(String::from_utf8(output.stdout)?.trim().into())
}

fn hash_file(path: &Path) -> Result<String> {
    Ok(hash_bytes(&std::fs::read(path)?))
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(all(test, feature = "heterogeneous-test-faults"))]
mod h8_manifest_tests {
    use super::*;

    fn assert_balanced(quotas: [u32; 3]) {
        let first = quota_balanced_owner_roles(
            36,
            128,
            quotas,
            H8_POLICY_SEED,
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        )
        .unwrap();
        let second = quota_balanced_owner_roles(
            36,
            128,
            quotas,
            H8_POLICY_SEED,
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 4_608);

        let mut global = [0_u32; 3];
        let mut minimums = [u32::MAX; 3];
        let mut maximums = [0_u32; 3];
        for layer in first.chunks_exact(128) {
            let mut counts = [0_u32; 3];
            for role in layer {
                counts[usize::from(*role)] += 1;
                global[usize::from(*role)] += 1;
            }
            assert!(counts.into_iter().all(|count| count > 0));
            for role in 0..3 {
                minimums[role] = minimums[role].min(counts[role]);
                maximums[role] = maximums[role].max(counts[role]);
            }
        }
        assert_eq!(global, quotas);
        for role in 0..3 {
            assert!(maximums[role] - minimums[role] <= 1);
        }
    }

    #[test]
    fn h8_quota_balanced_hash_is_exact_and_deterministic() {
        // The measured H8 quota shape seen during preflight has one apportioned
        // extra per layer. Also cover the generic two-extra complement branch.
        assert_balanced([1_814, 1_236, 1_558]);
        assert_balanced([1_835, 1_223, 1_550]);
    }
}
