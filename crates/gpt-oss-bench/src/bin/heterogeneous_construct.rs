use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use gpt_oss_core::error::LLMError;
use gpt_oss_gpu::device::{list_devices, GpuDevice, StableCudaDeviceId};
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
    ConstructionLedger, ConstructionStage, OwnerSelectiveConstructor, OwnerSelectiveEnvelope,
    OWNER_SELECTIVE_GPU_RESERVE_BYTES,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CACHE_ROOT: &str = "/home/emmy/workspace/gpt-oss-rs-het-cache";
const MIN_AVAILABLE_BYTES: u64 = 12 * 1024 * 1024 * 1024;
const MAX_PROCESS_RSS_BYTES: u64 = 72 * 1024 * 1024 * 1024;
const GPU_CLEANUP_TOLERANCE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Validate,
    Cold,
    Warm,
    #[cfg(feature = "heterogeneous-test-faults")]
    Faults,
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
        _ => {}
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

    let checkpoint_120b = GptOssCheckpointView::open(&cli.model_120b)?;
    let map_equivalence_120b = compare_research_mapping(&checkpoint_120b, &cli.mapping_120b)?;
    assert_checkpoint_bytes(
        &checkpoint_120b,
        65_248_815_744,
        60_993_699_840,
        4_255_115_904,
    )?;
    let manifest_120b = existence_manifest_120b(&checkpoint_120b, &stable)?;
    let resolved_120b = manifest_120b.validate(&devices)?;
    let envelope_120b =
        OwnerSelectiveEnvelope::from_checkpoint_and_placement(&checkpoint_120b, &resolved_120b)?;
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

    let checkpoint_record_20b = checkpoint_record(&checkpoint_20b, map_equivalence_20b);
    let checkpoint_record_120b = checkpoint_record(&checkpoint_120b, map_equivalence_120b);
    let placement_record_20b = placement_record(&manifest_20b, &resolved_20b);
    let placement_record_120b = placement_record(&manifest_120b, &resolved_120b);
    drop(checkpoint_120b);
    let cache_bytes_before = cache_bytes(&cli.cache_root)?;

    let (construction, fault_campaign) = match cli.mode {
        Mode::Validate => (None, None),
        Mode::Cold | Mode::Warm => (
            Some(run_construction(
                checkpoint_20b,
                &manifest_20b,
                &cli.cache_root,
                &stable,
                system_before.swap_used_bytes,
            )?),
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
        schema: "gpt-oss-rs.heterogeneous-construction/v2",
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
    if snapshot.process.vm_swap_bytes != 0
        || snapshot
            .system
            .swap_used_bytes
            .saturating_sub(swap_baseline)
            != 0
        || snapshot.system.mem_available_bytes < MIN_AVAILABLE_BYTES
        || snapshot.process.vm_rss_bytes > MAX_PROCESS_RSS_BYTES
    {
        bail!("H3 resource guard failed at {:?}", snapshot.ledger.stage);
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
