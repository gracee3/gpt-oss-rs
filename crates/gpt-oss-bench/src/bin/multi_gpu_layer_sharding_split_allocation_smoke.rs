use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use gpt_oss_model_runner::{
    CudaShardResourceStatus, DeviceId, DeviceMap, LateAllocationKind, SafetensorHeaderManifest,
    ShardAllocationReport, ShardTensorManifest, ShardedCudaResourcePlan, ShardedCudaResourceStatus,
    ShardedModelPlan, SplitAllocationReport, UploadManifestOptions,
};
use serde::{Deserialize, Serialize};

const SUCCESS_CLASSIFICATION: &str = "multi_gpu_layer_sharding_split_allocation_smoke_complete";
const INVALID_DEVICE_MAP_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_split_allocation_smoke_invalid_device_map";
const CONFIG_ERROR_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_split_allocation_smoke_config_error";
const HEADER_ERROR_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_split_allocation_smoke_header_error";
#[allow(dead_code)]
const RESOURCE_ERROR_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_split_allocation_smoke_resource_error";
#[allow(dead_code)]
const LOADER_ERROR_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_split_allocation_smoke_loader_error";
#[allow(dead_code)]
const CUDA_UNAVAILABLE_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_split_allocation_smoke_cuda_unavailable";

const OMITTED_ALLOCATIONS: &[&str] = &[
    "kv_cache",
    "rope_tables",
    "metadata_buffers",
    "graph_output_buffers",
    "f16_scratch",
    "fused_qkv_weights",
    "fused_gate_up_weights",
    "moe_gpu_weights",
    "transformer_layers",
    "gpu_model_runner",
    "activation_transfer",
    "final_norm",
    "lm_head",
    "sampling",
    "logits",
    "graph_capture",
];

#[derive(Debug, Parser)]
#[command(about = "Bench-only manifest-driven CUDA allocation smoke for layer sharding")]
struct Cli {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    device_map: String,

    #[arg(long, default_value_t = 0)]
    selected_device: usize,

    #[arg(long)]
    kernel_dir: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = DTypeMode::F16)]
    dtype: DTypeMode,

    #[arg(long)]
    tie_word_embeddings: Option<bool>,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum DTypeMode {
    F32,
    F16,
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct SplitAllocationSmokeReport {
    classification: String,
    model_path: String,
    device_map_spec: String,
    selected_device: usize,
    num_layers: Option<usize>,
    dtype_mode: DTypeMode,
    has_lm_head_weight: Option<bool>,
    tie_word_embeddings: Option<bool>,
    tensor_count_from_headers: Option<usize>,
    total_header_tensor_bytes: Option<usize>,
    resource_construction_succeeded: bool,
    allocation_smoke_succeeded: bool,
    kernel_dir: Option<String>,
    omitted_allocations: Vec<String>,
    shards: Vec<SplitAllocationSmokeShardReport>,
    unassigned_tensor_names: Vec<String>,
    invalid_tensor_names: Vec<String>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct SplitAllocationSmokeShardReport {
    device_id: usize,
    absolute_layers: Vec<usize>,
    owns_embeddings: bool,
    owns_final_head: bool,
    required_tensor_count: usize,
    required_tensor_names: Vec<String>,
    uploaded_f32_tensor_count: usize,
    uploaded_f16_tensor_count: usize,
    host_u8_tensor_count: usize,
    host_u8_tensor_names: Vec<String>,
    uploaded_f32_shape_count: usize,
    uploaded_f16_shape_count: usize,
    uploaded_f32_total_elements: usize,
    uploaded_f32_total_bytes: usize,
    uploaded_f16_total_elements: usize,
    uploaded_f16_total_bytes: usize,
    host_u8_total_bytes: usize,
    late_allocations: Vec<String>,
    resource_status: CudaResourceStatusReport,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CudaResourceStatusReport {
    context_created: bool,
    stream_created: bool,
    cublas_created: bool,
    kernel_loader_created: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelPlanningConfig {
    num_layers: usize,
    tie_word_embeddings: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ShardUploadCounts {
    uploaded_f32_tensor_count: usize,
    uploaded_f16_tensor_count: usize,
    uploaded_f32_shape_count: usize,
    uploaded_f16_shape_count: usize,
    uploaded_f32_total_elements: usize,
    uploaded_f16_total_elements: usize,
    host_u8_total_bytes: usize,
}

impl ShardUploadCounts {
    fn planned(shard: &ShardTensorManifest, header_bytes: &BTreeMap<String, usize>) -> Self {
        let required_elements = 0;
        let host_u8_total_bytes = shard
            .host_u8_tensor_names
            .iter()
            .filter_map(|name| header_bytes.get(name))
            .sum();

        Self {
            uploaded_f32_tensor_count: 0,
            uploaded_f16_tensor_count: 0,
            uploaded_f32_shape_count: 0,
            uploaded_f16_shape_count: 0,
            uploaded_f32_total_elements: required_elements,
            uploaded_f16_total_elements: required_elements,
            host_u8_total_bytes,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let report = build_split_allocation_smoke_report(
        &cli.model,
        &cli.device_map,
        cli.selected_device,
        cli.kernel_dir.as_deref(),
        cli.dtype,
        cli.tie_word_embeddings,
        true,
    );
    let json = serde_json::to_string_pretty(&report)?;

    if let Some(output) = cli.output {
        if let Some(parent) = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }
        std::fs::write(&output, json.as_bytes())
            .with_context(|| format!("failed to write {}", output.display()))?;
    } else {
        println!("{json}");
    }

    Ok(())
}

fn build_split_allocation_smoke_report(
    model_path: &Path,
    device_map_spec: &str,
    selected_device: usize,
    kernel_dir: Option<&Path>,
    dtype_mode: DTypeMode,
    tie_word_embeddings_override: Option<bool>,
    construct_and_upload: bool,
) -> SplitAllocationSmokeReport {
    let config = match read_model_planning_config(model_path, tie_word_embeddings_override) {
        Ok(config) => config,
        Err(error) => {
            return error_report(
                CONFIG_ERROR_CLASSIFICATION,
                model_path,
                device_map_spec,
                selected_device,
                dtype_mode,
                kernel_dir,
                Some(error.to_string()),
            );
        }
    };

    let header_manifest = match SafetensorHeaderManifest::discover(model_path) {
        Ok(manifest) => manifest,
        Err(error) => {
            return error_report_with_config(
                HEADER_ERROR_CLASSIFICATION,
                model_path,
                device_map_spec,
                selected_device,
                dtype_mode,
                kernel_dir,
                config,
                Some(error.to_string()),
            );
        }
    };

    let device_map = match DeviceMap::parse(
        device_map_spec,
        config.num_layers,
        DeviceId(selected_device),
    ) {
        Ok(device_map) => device_map,
        Err(error) => {
            return error_report_with_config_and_headers(
                INVALID_DEVICE_MAP_CLASSIFICATION,
                model_path,
                device_map_spec,
                selected_device,
                dtype_mode,
                kernel_dir,
                config,
                &header_manifest,
                Some(error.to_string()),
            );
        }
    };

    let plan = match ShardedModelPlan::from_device_map(device_map, config.num_layers) {
        Ok(plan) => plan,
        Err(error) => {
            return error_report_with_config_and_headers(
                INVALID_DEVICE_MAP_CLASSIFICATION,
                model_path,
                device_map_spec,
                selected_device,
                dtype_mode,
                kernel_dir,
                config,
                &header_manifest,
                Some(error.to_string()),
            );
        }
    };

    let upload_manifest = match plan.upload_manifest_for_tensor_names(
        header_manifest.tensor_names(),
        UploadManifestOptions {
            tie_word_embeddings: config.tie_word_embeddings,
        },
    ) {
        Ok(manifest) => manifest,
        Err(error) => {
            return error_report_with_config_and_headers(
                HEADER_ERROR_CLASSIFICATION,
                model_path,
                device_map_spec,
                selected_device,
                dtype_mode,
                kernel_dir,
                config,
                &header_manifest,
                Some(error.to_string()),
            );
        }
    };

    let kv_plan = plan.kv_cache_plan();
    let allocation_report = match plan.split_allocation_report(&upload_manifest, &kv_plan) {
        Ok(report) => report,
        Err(error) => {
            return error_report_with_config_and_headers(
                HEADER_ERROR_CLASSIFICATION,
                model_path,
                device_map_spec,
                selected_device,
                dtype_mode,
                kernel_dir,
                config,
                &header_manifest,
                Some(error.to_string()),
            );
        }
    };

    let header_bytes = tensor_byte_map(&header_manifest);
    let manifest_by_device = upload_manifest
        .shards
        .iter()
        .map(|shard| (shard.device_id, shard))
        .collect::<HashMap<_, _>>();

    if !construct_and_upload {
        let resource_plan = ShardedCudaResourcePlan::from_model_plan(&plan);
        let resource_status = ShardedCudaResourceStatus::from_plan(&resource_plan);
        return render_report(
            SUCCESS_CLASSIFICATION,
            model_path,
            device_map_spec,
            selected_device,
            dtype_mode,
            kernel_dir,
            config,
            &header_manifest,
            &allocation_report,
            &manifest_by_device,
            &header_bytes,
            &resource_status,
            false,
            true,
            None,
        );
    }

    match run_cuda_allocation_smoke(model_path, &plan, &upload_manifest, kernel_dir, dtype_mode) {
        Ok((resource_status, upload_counts)) => render_report_with_counts(
            SUCCESS_CLASSIFICATION,
            model_path,
            device_map_spec,
            selected_device,
            dtype_mode,
            kernel_dir,
            config,
            &header_manifest,
            &allocation_report,
            &manifest_by_device,
            &header_bytes,
            &resource_status,
            &upload_counts,
            true,
            true,
            None,
        ),
        Err((classification, error)) => {
            let resource_plan = ShardedCudaResourcePlan::from_model_plan(&plan);
            let resource_status = ShardedCudaResourceStatus::from_plan(&resource_plan);
            render_report(
                classification,
                model_path,
                device_map_spec,
                selected_device,
                dtype_mode,
                kernel_dir,
                config,
                &header_manifest,
                &allocation_report,
                &manifest_by_device,
                &header_bytes,
                &resource_status,
                true,
                false,
                Some(error),
            )
        }
    }
}

fn read_model_planning_config(
    model_path: &Path,
    tie_word_embeddings_override: Option<bool>,
) -> Result<ModelPlanningConfig> {
    let config_path = config_path_for_model(model_path);
    let text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
    let num_layers = json
        .get("num_hidden_layers")
        .and_then(|value| value.as_u64())
        .context("config.json missing numeric num_hidden_layers")? as usize;
    let tie_word_embeddings = tie_word_embeddings_override.unwrap_or_else(|| {
        json.get("tie_word_embeddings")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
    });

    Ok(ModelPlanningConfig {
        num_layers,
        tie_word_embeddings,
    })
}

fn config_path_for_model(model_path: &Path) -> PathBuf {
    if model_path.is_dir() {
        return model_path.join("config.json");
    }

    model_path
        .parent()
        .map(|parent| parent.join("config.json"))
        .unwrap_or_else(|| PathBuf::from("config.json"))
}

#[cfg(feature = "cuda")]
fn run_cuda_allocation_smoke(
    model_path: &Path,
    plan: &ShardedModelPlan,
    upload_manifest: &gpt_oss_model_runner::ShardedUploadManifest,
    kernel_dir: Option<&Path>,
    dtype_mode: DTypeMode,
) -> std::result::Result<
    (
        ShardedCudaResourceStatus,
        BTreeMap<DeviceId, ShardUploadCounts>,
    ),
    (&'static str, String),
> {
    use gpt_oss_model_runner::model_loader::gpu_loader::{
        load_u8_weights_to_host_filtered, load_weights_to_gpu_f16_with_shapes_filtered,
        load_weights_to_gpu_with_shapes_filtered,
    };
    use gpt_oss_model_runner::ShardedCudaResources;

    let resources = match kernel_dir {
        Some(kernel_dir) => ShardedCudaResources::create_for_plan_with_kernel_dir(plan, kernel_dir),
        None => ShardedCudaResources::create_for_plan(plan),
    }
    .map_err(|error| (RESOURCE_ERROR_CLASSIFICATION, error.to_string()))?;

    let mut upload_counts = BTreeMap::new();
    for resource in &resources.shards {
        let manifest = upload_manifest
            .shards
            .iter()
            .find(|shard| shard.device_id == resource.device_id)
            .ok_or_else(|| {
                (
                    LOADER_ERROR_CLASSIFICATION,
                    format!("missing upload manifest for device {}", resource.device_id),
                )
            })?;

        let required = manifest.required_tensor_filter_set();
        let host_u8 = manifest.host_u8_tensor_filter_set();

        let (uploaded_f32_tensor_count, uploaded_f32_shape_count, uploaded_f32_total_elements) =
            if matches!(dtype_mode, DTypeMode::F32 | DTypeMode::Both) {
                let (weights, shapes) = load_weights_to_gpu_with_shapes_filtered(
                    model_path,
                    &resource.stream,
                    |name| required.contains(name),
                )
                .map_err(|error| (LOADER_ERROR_CLASSIFICATION, error.to_string()))?;
                let total_elements = shapes_for_names(&shapes, weights.keys());
                (weights.len(), shapes.len(), total_elements)
            } else {
                (0, 0, 0)
            };

        let (uploaded_f16_tensor_count, uploaded_f16_shape_count, uploaded_f16_total_elements) =
            if matches!(dtype_mode, DTypeMode::F16 | DTypeMode::Both) {
                let (weights, shapes) = load_weights_to_gpu_f16_with_shapes_filtered(
                    model_path,
                    &resource.stream,
                    |name| required.contains(name),
                )
                .map_err(|error| (LOADER_ERROR_CLASSIFICATION, error.to_string()))?;
                let total_elements = shapes_for_names(&shapes, weights.keys());
                (weights.len(), shapes.len(), total_elements)
            } else {
                (0, 0, 0)
            };

        let u8_map = load_u8_weights_to_host_filtered(model_path, |name| host_u8.contains(name))
            .map_err(|error| (LOADER_ERROR_CLASSIFICATION, error.to_string()))?;
        let host_u8_total_bytes = u8_map.values().map(Vec::len).sum();

        upload_counts.insert(
            resource.device_id,
            ShardUploadCounts {
                uploaded_f32_tensor_count,
                uploaded_f16_tensor_count,
                uploaded_f32_shape_count,
                uploaded_f16_shape_count,
                uploaded_f32_total_elements,
                uploaded_f16_total_elements,
                host_u8_total_bytes,
            },
        );
    }

    Ok((resources.status(), upload_counts))
}

#[cfg(not(feature = "cuda"))]
fn run_cuda_allocation_smoke(
    _model_path: &Path,
    _plan: &ShardedModelPlan,
    _upload_manifest: &gpt_oss_model_runner::ShardedUploadManifest,
    _kernel_dir: Option<&Path>,
    _dtype_mode: DTypeMode,
) -> std::result::Result<
    (
        ShardedCudaResourceStatus,
        BTreeMap<DeviceId, ShardUploadCounts>,
    ),
    (&'static str, String),
> {
    Err((
        CUDA_UNAVAILABLE_CLASSIFICATION,
        "binary was built without the cuda feature".into(),
    ))
}

#[cfg(feature = "cuda")]
fn shapes_for_names<'a>(
    shapes: &HashMap<String, Vec<usize>>,
    names: impl Iterator<Item = &'a String>,
) -> usize {
    names
        .filter_map(|name| shapes.get(name))
        .map(|shape| shape.iter().product::<usize>())
        .sum()
}

fn render_report(
    classification: &str,
    model_path: &Path,
    device_map_spec: &str,
    selected_device: usize,
    dtype_mode: DTypeMode,
    kernel_dir: Option<&Path>,
    config: ModelPlanningConfig,
    header_manifest: &SafetensorHeaderManifest,
    allocation_report: &SplitAllocationReport,
    manifest_by_device: &HashMap<DeviceId, &ShardTensorManifest>,
    header_bytes: &BTreeMap<String, usize>,
    resource_status: &ShardedCudaResourceStatus,
    resource_construction_succeeded: bool,
    allocation_smoke_succeeded: bool,
    error: Option<String>,
) -> SplitAllocationSmokeReport {
    let upload_counts = manifest_by_device
        .iter()
        .map(|(&device_id, manifest)| {
            (
                device_id,
                ShardUploadCounts::planned(manifest, header_bytes),
            )
        })
        .collect();

    render_report_with_counts(
        classification,
        model_path,
        device_map_spec,
        selected_device,
        dtype_mode,
        kernel_dir,
        config,
        header_manifest,
        allocation_report,
        manifest_by_device,
        header_bytes,
        resource_status,
        &upload_counts,
        resource_construction_succeeded,
        allocation_smoke_succeeded,
        error,
    )
}

fn render_report_with_counts(
    classification: &str,
    model_path: &Path,
    device_map_spec: &str,
    selected_device: usize,
    dtype_mode: DTypeMode,
    kernel_dir: Option<&Path>,
    config: ModelPlanningConfig,
    header_manifest: &SafetensorHeaderManifest,
    allocation_report: &SplitAllocationReport,
    manifest_by_device: &HashMap<DeviceId, &ShardTensorManifest>,
    header_bytes: &BTreeMap<String, usize>,
    resource_status: &ShardedCudaResourceStatus,
    upload_counts: &BTreeMap<DeviceId, ShardUploadCounts>,
    resource_construction_succeeded: bool,
    allocation_smoke_succeeded: bool,
    error: Option<String>,
) -> SplitAllocationSmokeReport {
    let resource_by_device = resource_status
        .shards
        .iter()
        .map(|status| (status.device_id, status))
        .collect::<HashMap<_, _>>();

    SplitAllocationSmokeReport {
        classification: classification.into(),
        model_path: model_path.display().to_string(),
        device_map_spec: device_map_spec.into(),
        selected_device,
        num_layers: Some(config.num_layers),
        dtype_mode,
        has_lm_head_weight: Some(header_manifest.has_lm_head_weight()),
        tie_word_embeddings: Some(config.tie_word_embeddings),
        tensor_count_from_headers: Some(header_manifest.tensors.len()),
        total_header_tensor_bytes: Some(header_manifest.tensors.iter().map(|t| t.byte_size).sum()),
        resource_construction_succeeded,
        allocation_smoke_succeeded,
        kernel_dir: kernel_dir.map(|path| path.display().to_string()),
        omitted_allocations: omitted_allocations(),
        shards: allocation_report
            .shards
            .iter()
            .map(|shard| {
                let manifest = manifest_by_device
                    .get(&shard.device_id)
                    .expect("allocation shard should have tensor manifest");
                let counts = upload_counts
                    .get(&shard.device_id)
                    .cloned()
                    .unwrap_or_else(|| ShardUploadCounts::planned(manifest, header_bytes));
                let resource = resource_by_device.get(&shard.device_id).copied();
                render_shard_report(shard, counts, resource, resource_construction_succeeded)
            })
            .collect(),
        unassigned_tensor_names: allocation_report.unassigned_tensor_names.clone(),
        invalid_tensor_names: allocation_report.invalid_tensor_names.clone(),
        error,
    }
}

fn render_shard_report(
    shard: &ShardAllocationReport,
    counts: ShardUploadCounts,
    resource: Option<&CudaShardResourceStatus>,
    resource_created: bool,
) -> SplitAllocationSmokeShardReport {
    SplitAllocationSmokeShardReport {
        device_id: shard.device_id.0,
        absolute_layers: shard.absolute_layers.clone(),
        owns_embeddings: shard.owns_embeddings,
        owns_final_head: shard.owns_final_head,
        required_tensor_count: shard.required_tensor_count,
        required_tensor_names: shard.required_tensor_names.clone(),
        uploaded_f32_tensor_count: counts.uploaded_f32_tensor_count,
        uploaded_f16_tensor_count: counts.uploaded_f16_tensor_count,
        host_u8_tensor_count: shard.host_u8_tensor_count,
        host_u8_tensor_names: shard.host_u8_tensor_names.clone(),
        uploaded_f32_shape_count: counts.uploaded_f32_shape_count,
        uploaded_f16_shape_count: counts.uploaded_f16_shape_count,
        uploaded_f32_total_elements: counts.uploaded_f32_total_elements,
        uploaded_f32_total_bytes: counts.uploaded_f32_total_elements * 4,
        uploaded_f16_total_elements: counts.uploaded_f16_total_elements,
        uploaded_f16_total_bytes: counts.uploaded_f16_total_elements * 2,
        host_u8_total_bytes: counts.host_u8_total_bytes,
        late_allocations: shard
            .late_allocations
            .iter()
            .map(format_late_allocation)
            .collect(),
        resource_status: CudaResourceStatusReport {
            context_created: resource.is_some() && resource_created,
            stream_created: resource.is_some() && resource_created,
            cublas_created: resource.is_some() && resource_created,
            kernel_loader_created: resource.is_some() && resource_created,
        },
    }
}

fn error_report(
    classification: &str,
    model_path: &Path,
    device_map_spec: &str,
    selected_device: usize,
    dtype_mode: DTypeMode,
    kernel_dir: Option<&Path>,
    error: Option<String>,
) -> SplitAllocationSmokeReport {
    SplitAllocationSmokeReport {
        classification: classification.into(),
        model_path: model_path.display().to_string(),
        device_map_spec: device_map_spec.into(),
        selected_device,
        num_layers: None,
        dtype_mode,
        has_lm_head_weight: None,
        tie_word_embeddings: None,
        tensor_count_from_headers: None,
        total_header_tensor_bytes: None,
        resource_construction_succeeded: false,
        allocation_smoke_succeeded: false,
        kernel_dir: kernel_dir.map(|path| path.display().to_string()),
        omitted_allocations: omitted_allocations(),
        shards: Vec::new(),
        unassigned_tensor_names: Vec::new(),
        invalid_tensor_names: Vec::new(),
        error,
    }
}

fn error_report_with_config(
    classification: &str,
    model_path: &Path,
    device_map_spec: &str,
    selected_device: usize,
    dtype_mode: DTypeMode,
    kernel_dir: Option<&Path>,
    config: ModelPlanningConfig,
    error: Option<String>,
) -> SplitAllocationSmokeReport {
    let mut report = error_report(
        classification,
        model_path,
        device_map_spec,
        selected_device,
        dtype_mode,
        kernel_dir,
        error,
    );
    report.num_layers = Some(config.num_layers);
    report.tie_word_embeddings = Some(config.tie_word_embeddings);
    report
}

fn error_report_with_config_and_headers(
    classification: &str,
    model_path: &Path,
    device_map_spec: &str,
    selected_device: usize,
    dtype_mode: DTypeMode,
    kernel_dir: Option<&Path>,
    config: ModelPlanningConfig,
    header_manifest: &SafetensorHeaderManifest,
    error: Option<String>,
) -> SplitAllocationSmokeReport {
    let mut report = error_report_with_config(
        classification,
        model_path,
        device_map_spec,
        selected_device,
        dtype_mode,
        kernel_dir,
        config,
        error,
    );
    report.has_lm_head_weight = Some(header_manifest.has_lm_head_weight());
    report.tensor_count_from_headers = Some(header_manifest.tensors.len());
    report.total_header_tensor_bytes =
        Some(header_manifest.tensors.iter().map(|t| t.byte_size).sum());
    report
}

fn tensor_byte_map(header_manifest: &SafetensorHeaderManifest) -> BTreeMap<String, usize> {
    header_manifest
        .tensors
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor.byte_size))
        .collect()
}

fn omitted_allocations() -> Vec<String> {
    OMITTED_ALLOCATIONS
        .iter()
        .map(|name| (*name).to_string())
        .collect()
}

fn format_late_allocation(kind: &LateAllocationKind) -> String {
    match kind {
        LateAllocationKind::RopeTables => "rope_tables".into(),
        LateAllocationKind::MetadataBuffers => "metadata_buffers".into(),
        LateAllocationKind::KvCache { absolute_layers } => {
            format!("kv_cache:{absolute_layers:?}")
        }
        LateAllocationKind::F16FusedLayerWeights { layer_idx } => {
            format!("f16_fused_layer_weights:{layer_idx}")
        }
        LateAllocationKind::F16ConvertedEmbedding => "f16_converted_embedding".into(),
        LateAllocationKind::F16ConvertedFinalNorm => "f16_converted_final_norm".into(),
        LateAllocationKind::GptOssMoeGpuUpload { layer_idx } => {
            format!("gpt_oss_moe_gpu_upload:{layer_idx}")
        }
        LateAllocationKind::TiedLmHeadFallback => "tied_lm_head_fallback".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_temp_model_dir(test_name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "multi_gpu_layer_sharding_split_allocation_smoke_{test_name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_config(dir: &Path, num_layers: usize, tie_word_embeddings: bool) {
        let config = serde_json::json!({
            "num_hidden_layers": num_layers,
            "tie_word_embeddings": tie_word_embeddings,
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
    }

    fn write_safetensors(path: &Path, tensors: &[(&str, &str, &[usize], usize)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();

        for (name, dtype, shape, byte_len) in tensors {
            let start = data.len();
            data.resize(start + byte_len, 0u8);
            let end = data.len();

            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }

        let header_json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_json);
        bytes.extend_from_slice(&data);
        std::fs::write(path, bytes).unwrap();
    }

    fn fixture_with_lm_head() -> PathBuf {
        let dir = unique_temp_model_dir("with_lm_head");
        write_config(&dir, 24, true);
        write_safetensors(
            &dir.join("model.safetensors"),
            &[
                ("model.embed_tokens.weight", "F16", &[2, 4], 16),
                ("model.layers.0.self_attn.q_proj.weight", "F32", &[2], 8),
                ("model.layers.11.mlp.down_proj.weight", "F32", &[2], 8),
                ("model.layers.12.self_attn.k_proj.weight", "F32", &[2], 8),
                (
                    "model.layers.23.post_attention_layernorm.weight",
                    "F32",
                    &[2],
                    8,
                ),
                (
                    "model.layers.23.mlp.experts.down_proj_scales",
                    "U8",
                    &[4],
                    4,
                ),
                ("model.norm.weight", "F32", &[4], 16),
                ("lm_head.weight", "F16", &[2, 4], 16),
                ("model.extra_unknown.weight", "F32", &[1], 4),
                ("model.layers.24.self_attn.q_proj.weight", "F32", &[1], 4),
            ],
        );
        dir
    }

    fn split_report_for(
        dir: &Path,
        tie_word_embeddings: Option<bool>,
    ) -> SplitAllocationSmokeReport {
        build_split_allocation_smoke_report(
            dir,
            "split:0-11@0,12-23@1",
            0,
            None,
            DTypeMode::F16,
            tie_word_embeddings,
            false,
        )
    }

    fn shard(
        report: &SplitAllocationSmokeReport,
        device_id: usize,
    ) -> &SplitAllocationSmokeShardReport {
        report
            .shards
            .iter()
            .find(|shard| shard.device_id == device_id)
            .unwrap()
    }

    #[test]
    fn split_allocation_smoke_status_builds_from_header_fixture() {
        let dir = fixture_with_lm_head();

        let report = split_report_for(&dir, None);

        assert_eq!(report.classification, SUCCESS_CLASSIFICATION);
        assert_eq!(report.num_layers, Some(24));
        assert_eq!(report.tensor_count_from_headers, Some(10));
        assert_eq!(report.shards.len(), 2);
        assert!(!report.resource_construction_succeeded);
        assert!(report.allocation_smoke_succeeded);
    }

    #[test]
    fn split_allocation_smoke_status_includes_split_layer_ownership() {
        let dir = fixture_with_lm_head();

        let report = split_report_for(&dir, None);

        let gpu0 = shard(&report, 0);
        assert!(gpu0.owns_embeddings);
        assert!(!gpu0.owns_final_head);
        assert_eq!(gpu0.absolute_layers, (0..12).collect::<Vec<_>>());

        let gpu1 = shard(&report, 1);
        assert!(!gpu1.owns_embeddings);
        assert!(gpu1.owns_final_head);
        assert_eq!(gpu1.absolute_layers, (12..24).collect::<Vec<_>>());
    }

    #[test]
    fn split_allocation_smoke_omits_execution_allocations() {
        let dir = fixture_with_lm_head();

        let report = split_report_for(&dir, None);

        for expected in [
            "kv_cache",
            "rope_tables",
            "metadata_buffers",
            "f16_scratch",
            "fused_qkv_weights",
            "fused_gate_up_weights",
            "moe_gpu_weights",
            "transformer_layers",
            "gpu_model_runner",
            "activation_transfer",
        ] {
            assert!(report.omitted_allocations.contains(&expected.to_string()));
        }
    }

    #[test]
    fn tied_lm_head_fallback_remains_deferred_when_lm_head_absent() {
        let dir = unique_temp_model_dir("tied_fallback");
        write_config(&dir, 24, true);
        write_safetensors(
            &dir.join("model.safetensors"),
            &[
                ("model.embed_tokens.weight", "F16", &[2, 4], 16),
                ("model.norm.weight", "F32", &[4], 16),
            ],
        );

        let report = split_report_for(&dir, None);

        assert_eq!(report.has_lm_head_weight, Some(false));
        assert!(shard(&report, 1)
            .late_allocations
            .contains(&"tied_lm_head_fallback".to_string()));
        assert!(!shard(&report, 1)
            .required_tensor_names
            .contains(&"model.embed_tokens.weight".to_string()));
    }

    #[test]
    fn tied_lm_head_fallback_absent_when_lm_head_exists() {
        let dir = fixture_with_lm_head();

        let report = split_report_for(&dir, None);

        assert_eq!(report.has_lm_head_weight, Some(true));
        assert!(!shard(&report, 1)
            .late_allocations
            .contains(&"tied_lm_head_fallback".to_string()));
    }

    #[test]
    fn unknown_and_invalid_tensors_are_not_selected_for_upload() {
        let dir = fixture_with_lm_head();

        let report = split_report_for(&dir, None);

        assert_eq!(
            report.unassigned_tensor_names,
            vec!["model.extra_unknown.weight".to_string()]
        );
        assert_eq!(
            report.invalid_tensor_names,
            vec!["model.layers.24.self_attn.q_proj.weight".to_string()]
        );
        for shard in &report.shards {
            assert!(!shard
                .required_tensor_names
                .contains(&"model.extra_unknown.weight".to_string()));
            assert!(!shard
                .required_tensor_names
                .contains(&"model.layers.24.self_attn.q_proj.weight".to_string()));
        }
    }

    #[test]
    fn split_allocation_smoke_json_contains_success_classification() {
        let dir = fixture_with_lm_head();

        let report = split_report_for(&dir, None);
        let json = serde_json::to_string(&report).unwrap();

        assert!(json.contains(SUCCESS_CLASSIFICATION));
    }
}
