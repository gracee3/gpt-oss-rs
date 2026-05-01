#[cfg(feature = "cuda")]
use std::collections::BTreeSet;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use gpt_oss_model_runner::{
    CudaLayerFusedF16AllocationStatus, CudaShardFusedF16AllocationStatus,
    CudaShardKvCacheAllocationStatus, CudaShardMetadataAllocationStatus, CudaShardResourceStatus,
    CudaShardRuntimeBufferStatus, DeviceId, DeviceMap, F16ScratchAllocationConfig,
    FusedF16AllocationStatus, KvCacheAllocationConfig, LateAllocationKind,
    MetadataAllocationConfig, MetadataMode, RopeRuntimeBufferConfig, SafetensorHeaderManifest,
    SafetensorHeaderMergePolicy, ShardAllocationReport, ShardTensorManifest,
    ShardedCudaResourcePlan, ShardedCudaResourceStatus, ShardedFusedF16AllocationPlan,
    ShardedFusedF16AllocationStatus, ShardedKvCacheAllocationPlan, ShardedKvCacheAllocationStatus,
    ShardedMetadataAllocationPlan, ShardedMetadataAllocationStatus, ShardedModelPlan,
    ShardedRuntimeBufferPlan, ShardedRuntimeBufferStatus, SplitAllocationReport,
    UploadManifestOptions,
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
const ROPE_ALLOCATED_METADATA_DEFERRED_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_rope_allocated_metadata_deferred";
const KV_CACHE_ALLOCATION_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_kv_cache_allocation_smoke_complete";
const METADATA_ALLOCATION_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_metadata_allocation_smoke_complete";
const FUSED_F16_PLAN_STATUS_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_fused_f16_plan_status_complete";
const FUSED_QKV_ALLOCATION_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_fused_qkv_allocation_smoke_complete";
const FUSED_QKV_NORM_ALLOCATION_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_fused_qkv_norm_allocation_smoke_complete";
const FUSED_QKV_NORM_BIAS_ALLOCATION_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_fused_qkv_norm_bias_allocation_smoke_complete";
#[allow(dead_code)]
const FUSED_QKV_ALLOCATION_BLOCKED_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_fused_qkv_allocation_blocked";
const FUSED_QKV_NORM_ALLOCATION_CAST_ERROR_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_fused_qkv_norm_allocation_cast_error";
const BIAS_CONVERSION_ALLOCATION_BLOCKED_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_bias_conversion_allocation_blocked";

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
    allow_restricted_sinks_override: bool,

    #[arg(long)]
    output: Option<PathBuf>,

    #[arg(long)]
    allocate_rope_metadata: bool,

    #[arg(long)]
    allocate_kv_cache: bool,

    #[arg(long)]
    kv_num_blocks: Option<usize>,

    #[arg(long)]
    kv_block_size: Option<usize>,

    #[arg(long)]
    allocate_metadata: bool,

    #[arg(long, value_enum)]
    metadata_mode: Option<MetadataModeArg>,

    #[arg(long)]
    metadata_num_tokens: Option<usize>,

    #[arg(long)]
    metadata_num_seqs: Option<usize>,

    #[arg(long)]
    metadata_context_len: Option<usize>,

    #[arg(long)]
    metadata_block_size: Option<usize>,

    #[arg(long)]
    allocate_fused_f16: bool,

    #[arg(long)]
    allocate_f16_scratch: bool,

    #[arg(long)]
    f16_scratch_max_tokens: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum DTypeMode {
    F32,
    F16,
    Both,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum MetadataModeArg {
    Decode,
}

impl MetadataModeArg {
    fn to_runner_mode(self) -> MetadataMode {
        match self {
            MetadataModeArg::Decode => MetadataMode::Decode,
        }
    }
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
    header_merge_policy: String,
    restricted_sinks_override_enabled: bool,
    overridden_tensor_count: usize,
    overridden_tensor_names: Vec<String>,
    resource_construction_succeeded: bool,
    allocation_smoke_succeeded: bool,
    rope_metadata_allocation_attempted: bool,
    rope_metadata_allocation_succeeded: bool,
    kv_cache_allocation_attempted: bool,
    kv_cache_allocation_succeeded: bool,
    kv_num_blocks: Option<usize>,
    kv_block_size: Option<usize>,
    metadata_allocation_attempted: bool,
    metadata_allocation_succeeded: bool,
    metadata_mode: Option<String>,
    metadata_num_tokens: Option<usize>,
    metadata_num_seqs: Option<usize>,
    metadata_context_len: Option<usize>,
    metadata_block_size: Option<usize>,
    metadata_graph_max_blocks: Option<usize>,
    metadata_error: Option<String>,
    fused_f16_allocation_attempted: bool,
    fused_f16_allocation_succeeded: bool,
    f16_scratch_allocation_attempted: bool,
    f16_scratch_allocation_succeeded: bool,
    f16_scratch_max_tokens: Option<usize>,
    fused_f16_error: Option<String>,
    f16_scratch_error: Option<String>,
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
    rope_allocated: bool,
    rope_cos_elements: usize,
    rope_sin_elements: usize,
    rope_total_bytes: usize,
    metadata_allocated: bool,
    metadata_status: String,
    metadata_deferred_reason: Option<String>,
    runtime_buffer_error: Option<String>,
    kv_cache_allocated: bool,
    kv_cache_entry_count: usize,
    kv_cache_layers: Vec<usize>,
    kv_cache_local_indices: Vec<usize>,
    kv_key_total_bytes: usize,
    kv_value_total_bytes: usize,
    kv_total_bytes: usize,
    kv_cache_entries: Vec<KvCacheEntryReport>,
    kv_cache_error: Option<String>,
    metadata_mode: Option<String>,
    metadata_num_tokens: usize,
    metadata_num_seqs: usize,
    metadata_graph_max_blocks: usize,
    metadata_packed_elements: usize,
    metadata_packed_bytes: usize,
    metadata_token_ids_len: usize,
    metadata_positions_len: usize,
    metadata_context_lens_len: usize,
    metadata_block_tables_len: usize,
    metadata_slot_mapping_len: usize,
    metadata_seq_start_pos_len: usize,
    metadata_max_context_len: usize,
    metadata_error: Option<String>,
    fused_f16_allocated: bool,
    fused_f16_status: String,
    fused_qkv_weight_count: usize,
    fused_qkv_total_bytes: usize,
    fused_gate_up_weight_count: usize,
    fused_gate_up_total_bytes: usize,
    f16_layernorm_total_bytes: usize,
    f16_postnorm_total_bytes: usize,
    f16_layernorm_count: usize,
    f16_postnorm_count: usize,
    f16_qkv_bias_count: usize,
    f16_o_proj_bias_count: usize,
    f16_qkv_bias_total_bytes: usize,
    f16_o_proj_bias_total_bytes: usize,
    embedding_f16_allocated: bool,
    final_norm_f16_allocated: bool,
    fused_total_bytes: usize,
    fused_layer_absolute_indices: Vec<usize>,
    fused_layer_statuses: Vec<FusedLayerReport>,
    fused_deferred_reason: Option<String>,
    fused_error: Option<String>,
    f16_scratch_allocated: bool,
    f16_scratch_status: String,
    f16_scratch_bytes: usize,
    f16_scratch_deferred_reason: Option<String>,
    f16_scratch_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct KvCacheEntryReport {
    absolute_layer_idx: usize,
    local_cache_idx: usize,
    key_bytes: usize,
    value_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CudaResourceStatusReport {
    context_created: bool,
    stream_created: bool,
    cublas_created: bool,
    kernel_loader_created: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct FusedLayerReport {
    absolute_layer_idx: usize,
    local_layer_idx: usize,
    fused_qkv_allocated: bool,
    fused_qkv_status: String,
    fused_qkv_bytes: usize,
    fused_gate_up_allocated: bool,
    fused_gate_up_status: String,
    fused_gate_up_bytes: usize,
    layernorm_f16_status: String,
    layernorm_f16_bytes: usize,
    postnorm_f16_status: String,
    postnorm_f16_bytes: usize,
    qkv_bias_f16_status: String,
    qkv_bias_f16_bytes: usize,
    o_proj_bias_f16_status: String,
    o_proj_bias_f16_bytes: usize,
    layer_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ModelPlanningConfig {
    num_layers: usize,
    tie_word_embeddings: bool,
    head_dim: Option<usize>,
    num_kv_heads: Option<usize>,
    max_position: usize,
    rope_theta: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct FusedF16SmokeOptions {
    allocate_fused_f16: bool,
    allocate_f16_scratch: bool,
    f16_scratch_max_tokens: Option<usize>,
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
        cli.allow_restricted_sinks_override,
        cli.allocate_rope_metadata,
        cli.allocate_kv_cache,
        cli.kv_num_blocks,
        cli.kv_block_size,
        cli.allocate_metadata,
        cli.metadata_mode,
        cli.metadata_num_tokens,
        cli.metadata_num_seqs,
        cli.metadata_context_len,
        cli.metadata_block_size,
        cli.allocate_fused_f16,
        cli.allocate_f16_scratch,
        cli.f16_scratch_max_tokens,
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
    allow_restricted_sinks_override: bool,
    allocate_rope_metadata: bool,
    allocate_kv_cache: bool,
    kv_num_blocks: Option<usize>,
    kv_block_size: Option<usize>,
    allocate_metadata: bool,
    metadata_mode: Option<MetadataModeArg>,
    metadata_num_tokens: Option<usize>,
    metadata_num_seqs: Option<usize>,
    metadata_context_len: Option<usize>,
    metadata_block_size: Option<usize>,
    allocate_fused_f16: bool,
    allocate_f16_scratch: bool,
    f16_scratch_max_tokens: Option<usize>,
    construct_and_upload: bool,
) -> SplitAllocationSmokeReport {
    let fused_f16_options = FusedF16SmokeOptions {
        allocate_fused_f16,
        allocate_f16_scratch,
        f16_scratch_max_tokens,
    };
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
                allocate_rope_metadata,
                allocate_kv_cache,
                kv_num_blocks,
                kv_block_size,
                allocate_metadata,
                metadata_mode,
                metadata_num_tokens,
                metadata_num_seqs,
                metadata_context_len,
                metadata_block_size,
                None,
                fused_f16_options,
                Some(error.to_string()),
            );
        }
    };
    let runtime_buffer_config = if allocate_rope_metadata {
        match build_runtime_buffer_config(config) {
            Ok(config) => Some(config),
            Err(error) => {
                return error_report(
                    CONFIG_ERROR_CLASSIFICATION,
                    model_path,
                    device_map_spec,
                    selected_device,
                    dtype_mode,
                    kernel_dir,
                    allocate_rope_metadata,
                    allocate_kv_cache,
                    kv_num_blocks,
                    kv_block_size,
                    allocate_metadata,
                    metadata_mode,
                    metadata_num_tokens,
                    metadata_num_seqs,
                    metadata_context_len,
                    metadata_block_size,
                    None,
                    fused_f16_options,
                    Some(error.to_string()),
                );
            }
        }
    } else {
        None
    };
    let kv_cache_config = match build_kv_cache_allocation_config(
        config,
        allocate_kv_cache,
        kv_num_blocks,
        kv_block_size,
    ) {
        Ok(config) => config,
        Err(error) => {
            return error_report(
                CONFIG_ERROR_CLASSIFICATION,
                model_path,
                device_map_spec,
                selected_device,
                dtype_mode,
                kernel_dir,
                allocate_rope_metadata,
                allocate_kv_cache,
                kv_num_blocks,
                kv_block_size,
                allocate_metadata,
                metadata_mode,
                metadata_num_tokens,
                metadata_num_seqs,
                metadata_context_len,
                metadata_block_size,
                None,
                fused_f16_options,
                Some(error.to_string()),
            );
        }
    };
    let metadata_config = match build_metadata_allocation_config(
        config,
        allocate_metadata,
        metadata_mode,
        metadata_num_tokens,
        metadata_num_seqs,
        metadata_context_len,
        metadata_block_size,
        kv_cache_config.as_ref(),
    ) {
        Ok(config) => config,
        Err(error) => {
            return error_report(
                CONFIG_ERROR_CLASSIFICATION,
                model_path,
                device_map_spec,
                selected_device,
                dtype_mode,
                kernel_dir,
                allocate_rope_metadata,
                allocate_kv_cache,
                kv_num_blocks,
                kv_block_size,
                allocate_metadata,
                metadata_mode,
                metadata_num_tokens,
                metadata_num_seqs,
                metadata_context_len,
                metadata_block_size,
                None,
                fused_f16_options,
                Some(error.to_string()),
            );
        }
    };
    let f16_scratch_config = match build_f16_scratch_allocation_config(fused_f16_options) {
        Ok(config) => config,
        Err(error) => {
            return error_report(
                CONFIG_ERROR_CLASSIFICATION,
                model_path,
                device_map_spec,
                selected_device,
                dtype_mode,
                kernel_dir,
                allocate_rope_metadata,
                allocate_kv_cache,
                kv_num_blocks,
                kv_block_size,
                allocate_metadata,
                metadata_mode,
                metadata_num_tokens,
                metadata_num_seqs,
                metadata_context_len,
                metadata_block_size,
                metadata_config.map(|config| config.graph_max_blocks()),
                fused_f16_options,
                Some(error.to_string()),
            );
        }
    };
    if fused_f16_options.allocate_fused_f16
        && !matches!(dtype_mode, DTypeMode::F16 | DTypeMode::Both)
    {
        return error_report(
            CONFIG_ERROR_CLASSIFICATION,
            model_path,
            device_map_spec,
            selected_device,
            dtype_mode,
            kernel_dir,
            allocate_rope_metadata,
            allocate_kv_cache,
            kv_num_blocks,
            kv_block_size,
            allocate_metadata,
            metadata_mode,
            metadata_num_tokens,
            metadata_num_seqs,
            metadata_context_len,
            metadata_block_size,
            metadata_config.map(|config| config.graph_max_blocks()),
            fused_f16_options,
            Some("--allocate-fused-f16 requires --dtype f16 or --dtype both".to_string()),
        );
    }

    let header_merge_policy = header_merge_policy(allow_restricted_sinks_override);
    let header_manifest =
        match SafetensorHeaderManifest::discover_with_merge_policy(model_path, header_merge_policy)
        {
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
                    allow_restricted_sinks_override,
                    allocate_rope_metadata,
                    allocate_kv_cache,
                    kv_num_blocks,
                    kv_block_size,
                    allocate_metadata,
                    metadata_mode,
                    metadata_num_tokens,
                    metadata_num_seqs,
                    metadata_context_len,
                    metadata_block_size,
                    metadata_config.map(|config| config.graph_max_blocks()),
                    fused_f16_options,
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
                allow_restricted_sinks_override,
                allocate_rope_metadata,
                allocate_kv_cache,
                kv_num_blocks,
                kv_block_size,
                allocate_metadata,
                metadata_mode,
                metadata_num_tokens,
                metadata_num_seqs,
                metadata_context_len,
                metadata_block_size,
                metadata_config.map(|config| config.graph_max_blocks()),
                fused_f16_options,
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
                allow_restricted_sinks_override,
                allocate_rope_metadata,
                allocate_kv_cache,
                kv_num_blocks,
                kv_block_size,
                allocate_metadata,
                metadata_mode,
                metadata_num_tokens,
                metadata_num_seqs,
                metadata_context_len,
                metadata_block_size,
                metadata_config.map(|config| config.graph_max_blocks()),
                fused_f16_options,
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
                allow_restricted_sinks_override,
                allocate_rope_metadata,
                allocate_kv_cache,
                kv_num_blocks,
                kv_block_size,
                allocate_metadata,
                metadata_mode,
                metadata_num_tokens,
                metadata_num_seqs,
                metadata_context_len,
                metadata_block_size,
                metadata_config.map(|config| config.graph_max_blocks()),
                fused_f16_options,
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
                allow_restricted_sinks_override,
                allocate_rope_metadata,
                allocate_kv_cache,
                kv_num_blocks,
                kv_block_size,
                allocate_metadata,
                metadata_mode,
                metadata_num_tokens,
                metadata_num_seqs,
                metadata_context_len,
                metadata_block_size,
                metadata_config.map(|config| config.graph_max_blocks()),
                fused_f16_options,
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
    let fused_f16_status =
        if fused_f16_options.allocate_fused_f16 || fused_f16_options.allocate_f16_scratch {
            let fused_plan = ShardedFusedF16AllocationPlan::from_upload_manifest(
                &upload_manifest,
                f16_scratch_config,
            );
            Some(ShardedFusedF16AllocationStatus::from_plan(
                &fused_plan,
                false,
                false,
            ))
        } else {
            None
        };

    if !construct_and_upload {
        let resource_plan = ShardedCudaResourcePlan::from_model_plan(&plan);
        let resource_status = ShardedCudaResourceStatus::from_plan(&resource_plan);
        let runtime_buffer_status = runtime_buffer_config.map(|config| {
            let runtime_plan = ShardedRuntimeBufferPlan::from_model_plan(&plan, config);
            ShardedRuntimeBufferStatus::from_plan(&runtime_plan, false)
        });
        let kv_cache_status = kv_cache_config.map(|config| {
            let kv_allocation_plan = ShardedKvCacheAllocationPlan::from_model_plan(&plan, config);
            ShardedKvCacheAllocationStatus::from_plan(&kv_allocation_plan, false)
        });
        let metadata_status = metadata_config.map(|config| {
            let metadata_plan = ShardedMetadataAllocationPlan::from_model_plan(&plan, config);
            ShardedMetadataAllocationStatus::from_plan(&metadata_plan, false)
        });
        let classification = if fused_f16_status.is_some() {
            FUSED_F16_PLAN_STATUS_CLASSIFICATION
        } else {
            SUCCESS_CLASSIFICATION
        };
        return render_report(
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
            allow_restricted_sinks_override,
            false,
            true,
            allocate_rope_metadata,
            false,
            allocate_kv_cache,
            false,
            kv_num_blocks,
            kv_block_size,
            allocate_metadata,
            false,
            metadata_config,
            runtime_buffer_status.as_ref(),
            kv_cache_status.as_ref(),
            metadata_status.as_ref(),
            fused_f16_options,
            fused_f16_status.as_ref(),
            None,
        );
    }

    match run_cuda_allocation_smoke(
        model_path,
        &plan,
        &upload_manifest,
        kernel_dir,
        dtype_mode,
        runtime_buffer_config,
        kv_cache_config,
        metadata_config,
        fused_f16_options,
        f16_scratch_config,
    ) {
        Ok((
            resource_status,
            upload_counts,
            runtime_buffer_status,
            kv_cache_status,
            metadata_status,
            actual_fused_f16_status,
        )) => {
            let runtime_buffer_succeeded =
                allocate_rope_metadata && runtime_buffer_status.is_some();
            let kv_cache_succeeded = allocate_kv_cache && kv_cache_status.is_some();
            let metadata_succeeded = allocate_metadata && metadata_status.is_some();
            let fused_f16_succeeded = fused_f16_options.allocate_fused_f16
                && actual_fused_f16_status
                    .as_ref()
                    .map(fused_status_succeeded)
                    .unwrap_or(false);
            let classification = if fused_f16_succeeded {
                if actual_fused_f16_status
                    .as_ref()
                    .map(fused_status_has_allocated_biases)
                    .unwrap_or(false)
                {
                    FUSED_QKV_NORM_BIAS_ALLOCATION_CLASSIFICATION
                } else if actual_fused_f16_status
                    .as_ref()
                    .map(fused_status_has_allocated_norms)
                    .unwrap_or(false)
                {
                    FUSED_QKV_NORM_ALLOCATION_CLASSIFICATION
                } else {
                    FUSED_QKV_ALLOCATION_CLASSIFICATION
                }
            } else if fused_f16_status.is_some() {
                FUSED_F16_PLAN_STATUS_CLASSIFICATION
            } else if metadata_succeeded {
                METADATA_ALLOCATION_CLASSIFICATION
            } else if kv_cache_succeeded {
                KV_CACHE_ALLOCATION_CLASSIFICATION
            } else if runtime_buffer_succeeded {
                ROPE_ALLOCATED_METADATA_DEFERRED_CLASSIFICATION
            } else {
                SUCCESS_CLASSIFICATION
            };
            render_report_with_counts(
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
                &upload_counts,
                allow_restricted_sinks_override,
                true,
                true,
                allocate_rope_metadata,
                runtime_buffer_succeeded,
                allocate_kv_cache,
                kv_cache_succeeded,
                kv_num_blocks,
                kv_block_size,
                allocate_metadata,
                metadata_succeeded,
                metadata_config,
                runtime_buffer_status.as_ref(),
                kv_cache_status.as_ref(),
                metadata_status.as_ref(),
                fused_f16_options,
                actual_fused_f16_status
                    .as_ref()
                    .or(fused_f16_status.as_ref()),
                None,
            )
        }
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
                allow_restricted_sinks_override,
                true,
                false,
                allocate_rope_metadata,
                false,
                allocate_kv_cache,
                false,
                kv_num_blocks,
                kv_block_size,
                allocate_metadata,
                false,
                metadata_config,
                None,
                None,
                None,
                fused_f16_options,
                fused_f16_status.as_ref(),
                Some(error),
            )
        }
    }
}

fn header_merge_policy(allow_restricted_sinks_override: bool) -> SafetensorHeaderMergePolicy {
    if allow_restricted_sinks_override {
        SafetensorHeaderMergePolicy::AllowRestrictedSinksOverride
    } else {
        SafetensorHeaderMergePolicy::RejectDuplicates
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
    let get_usize = |key: &str| -> Option<usize> {
        json.get(key)
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
    };
    let get_f32 = |key: &str, default: f32| -> f32 {
        json.get(key)
            .and_then(|value| value.as_f64())
            .map(|value| value as f32)
            .unwrap_or(default)
    };
    let hidden_size = get_usize("hidden_size");
    let num_attention_heads = get_usize("num_attention_heads");
    let head_dim = get_usize("head_dim").or_else(|| {
        hidden_size
            .zip(num_attention_heads)
            .and_then(|(hidden_size, heads)| hidden_size.checked_div(heads))
    });
    let num_kv_heads = get_usize("num_key_value_heads").or(num_attention_heads);
    let max_position = get_usize("max_position_embeddings").unwrap_or(2048);
    let rope_theta = get_f32("rope_theta", 10000.0);
    let tie_word_embeddings = tie_word_embeddings_override.unwrap_or_else(|| {
        json.get("tie_word_embeddings")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
    });

    Ok(ModelPlanningConfig {
        num_layers,
        tie_word_embeddings,
        head_dim,
        num_kv_heads,
        max_position,
        rope_theta,
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

fn build_runtime_buffer_config(config: ModelPlanningConfig) -> Result<RopeRuntimeBufferConfig> {
    let head_dim = config
        .head_dim
        .context("config.json missing numeric head_dim or hidden_size/num_attention_heads")?;
    RopeRuntimeBufferConfig::new(head_dim, config.max_position, config.rope_theta)
        .map_err(anyhow::Error::msg)
}

fn build_kv_cache_allocation_config(
    config: ModelPlanningConfig,
    allocate_kv_cache: bool,
    kv_num_blocks: Option<usize>,
    kv_block_size: Option<usize>,
) -> Result<Option<KvCacheAllocationConfig>> {
    if !allocate_kv_cache {
        return Ok(None);
    }

    let num_gpu_blocks =
        kv_num_blocks.context("--allocate-kv-cache requires --kv-num-blocks <N>")?;
    let block_size = kv_block_size.context("--allocate-kv-cache requires --kv-block-size <N>")?;
    let num_kv_heads = config
        .num_kv_heads
        .context("config.json missing numeric num_key_value_heads or num_attention_heads")?;
    let head_dim = config
        .head_dim
        .context("config.json missing numeric head_dim or hidden_size/num_attention_heads")?;

    KvCacheAllocationConfig::new(num_kv_heads, head_dim, num_gpu_blocks, block_size)
        .map(Some)
        .map_err(anyhow::Error::msg)
}

fn build_metadata_allocation_config(
    config: ModelPlanningConfig,
    allocate_metadata: bool,
    metadata_mode: Option<MetadataModeArg>,
    metadata_num_tokens: Option<usize>,
    metadata_num_seqs: Option<usize>,
    metadata_context_len: Option<usize>,
    metadata_block_size: Option<usize>,
    kv_cache_config: Option<&KvCacheAllocationConfig>,
) -> Result<Option<MetadataAllocationConfig>> {
    if !allocate_metadata {
        return Ok(None);
    }

    let mode = metadata_mode.unwrap_or(MetadataModeArg::Decode);
    match mode.to_runner_mode() {
        MetadataMode::Decode => {}
    }

    let num_tokens =
        metadata_num_tokens.context("--allocate-metadata requires --metadata-num-tokens <N>")?;
    let num_seqs =
        metadata_num_seqs.context("--allocate-metadata requires --metadata-num-seqs <N>")?;
    let context_len =
        metadata_context_len.context("--allocate-metadata requires --metadata-context-len <N>")?;
    let block_size =
        metadata_block_size.context("--allocate-metadata requires --metadata-block-size <N>")?;

    MetadataAllocationConfig::new_decode(
        num_tokens,
        num_seqs,
        context_len,
        block_size,
        config.max_position,
        kv_cache_config,
    )
    .map(Some)
    .map_err(anyhow::Error::msg)
}

fn build_f16_scratch_allocation_config(
    options: FusedF16SmokeOptions,
) -> Result<Option<F16ScratchAllocationConfig>> {
    if !options.allocate_f16_scratch {
        return Ok(None);
    }

    let max_tokens = options
        .f16_scratch_max_tokens
        .context("--allocate-f16-scratch requires --f16-scratch-max-tokens <N>")?;
    F16ScratchAllocationConfig::new(max_tokens)
        .map(Some)
        .map_err(anyhow::Error::msg)
}

#[cfg(feature = "cuda")]
fn run_cuda_allocation_smoke(
    model_path: &Path,
    plan: &ShardedModelPlan,
    upload_manifest: &gpt_oss_model_runner::ShardedUploadManifest,
    kernel_dir: Option<&Path>,
    dtype_mode: DTypeMode,
    runtime_buffer_config: Option<RopeRuntimeBufferConfig>,
    kv_cache_config: Option<KvCacheAllocationConfig>,
    metadata_config: Option<MetadataAllocationConfig>,
    fused_f16_options: FusedF16SmokeOptions,
    f16_scratch_config: Option<F16ScratchAllocationConfig>,
) -> std::result::Result<
    (
        ShardedCudaResourceStatus,
        BTreeMap<DeviceId, ShardUploadCounts>,
        Option<ShardedRuntimeBufferStatus>,
        Option<ShardedKvCacheAllocationStatus>,
        Option<ShardedMetadataAllocationStatus>,
        Option<ShardedFusedF16AllocationStatus>,
    ),
    (&'static str, String),
> {
    use gpt_oss_model_runner::model_loader::gpu_loader::{
        load_u8_weights_to_host_filtered, load_weights_to_gpu_f16_with_shapes_filtered,
        load_weights_to_gpu_with_shapes_filtered,
    };
    use gpt_oss_model_runner::{
        ShardedCudaResources, ShardedFusedF16Buffers, ShardedKvCacheBuffers,
        ShardedMetadataBuffers, ShardedRuntimeBuffers,
    };

    let resources = match kernel_dir {
        Some(kernel_dir) => ShardedCudaResources::create_for_plan_with_kernel_dir(plan, kernel_dir),
        None => ShardedCudaResources::create_for_plan(plan),
    }
    .map_err(|error| (RESOURCE_ERROR_CLASSIFICATION, error.to_string()))?;

    let mut upload_counts = BTreeMap::new();
    let mut f32_weights_by_device = BTreeMap::new();
    let mut f32_shapes_by_device = BTreeMap::new();
    let mut f16_weights_by_device = BTreeMap::new();
    let mut f16_shapes_by_device = BTreeMap::new();
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
        let f32_filter = if matches!(dtype_mode, DTypeMode::F32 | DTypeMode::Both) {
            required.clone()
        } else if fused_f16_options.allocate_fused_f16 {
            fused_f32_tensor_filter_set(manifest)
        } else {
            BTreeSet::new()
        };

        let (uploaded_f32_tensor_count, uploaded_f32_shape_count, uploaded_f32_total_elements) =
            if matches!(dtype_mode, DTypeMode::F32 | DTypeMode::Both)
                || (fused_f16_options.allocate_fused_f16 && !f32_filter.is_empty())
            {
                let (weights, shapes) = load_weights_to_gpu_with_shapes_filtered(
                    model_path,
                    &resource.stream,
                    |name| f32_filter.contains(name),
                )
                .map_err(|error| (LOADER_ERROR_CLASSIFICATION, error.to_string()))?;
                let total_elements = shapes_for_names(&shapes, weights.keys());
                let weight_count = weights.len();
                let shape_count = shapes.len();
                if fused_f16_options.allocate_fused_f16 {
                    f32_shapes_by_device.insert(resource.device_id, shapes.clone());
                    f32_weights_by_device.insert(resource.device_id, weights);
                }
                (weight_count, shape_count, total_elements)
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
                let weight_count = weights.len();
                let shape_count = shapes.len();
                if fused_f16_options.allocate_fused_f16 {
                    f16_shapes_by_device.insert(resource.device_id, shapes.clone());
                    f16_weights_by_device.insert(resource.device_id, weights);
                }
                (weight_count, shape_count, total_elements)
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

    let runtime_buffer_status = if let Some(runtime_buffer_config) = runtime_buffer_config {
        Some(
            ShardedRuntimeBuffers::create_for_resources(&resources, runtime_buffer_config)
                .map_err(|error| (RESOURCE_ERROR_CLASSIFICATION, error.to_string()))?
                .status(),
        )
    } else {
        None
    };

    let kv_cache_status = if let Some(kv_cache_config) = kv_cache_config {
        let kv_allocation_plan =
            ShardedKvCacheAllocationPlan::from_model_plan(plan, kv_cache_config);
        Some(
            ShardedKvCacheBuffers::create_for_resources(&resources, &kv_allocation_plan)
                .map_err(|error| (RESOURCE_ERROR_CLASSIFICATION, error.to_string()))?
                .status(),
        )
    } else {
        None
    };

    let metadata_status = if let Some(metadata_config) = metadata_config {
        Some(
            ShardedMetadataBuffers::create_for_resources(&resources, metadata_config)
                .map_err(|error| (RESOURCE_ERROR_CLASSIFICATION, error.to_string()))?
                .status(),
        )
    } else {
        None
    };

    let fused_f16_status =
        if fused_f16_options.allocate_fused_f16 || fused_f16_options.allocate_f16_scratch {
            if fused_f16_options.allocate_fused_f16 {
                Some(
                    ShardedFusedF16Buffers::create_for_resources(
                        &resources,
                        upload_manifest,
                        &f16_weights_by_device,
                        &f16_shapes_by_device,
                        &f32_weights_by_device,
                        &f32_shapes_by_device,
                        f16_scratch_config,
                    )
                    .map_err(|error| {
                        let error = error.to_string();
                        (classify_fused_f16_error(&error), error)
                    })?
                    .status(),
                )
            } else {
                let fused_plan = ShardedFusedF16AllocationPlan::from_upload_manifest(
                    upload_manifest,
                    f16_scratch_config,
                );
                Some(ShardedFusedF16AllocationStatus::from_plan(
                    &fused_plan,
                    false,
                    false,
                ))
            }
        } else {
            None
        };

    Ok((
        resources.status(),
        upload_counts,
        runtime_buffer_status,
        kv_cache_status,
        metadata_status,
        fused_f16_status,
    ))
}

#[cfg(not(feature = "cuda"))]
fn run_cuda_allocation_smoke(
    _model_path: &Path,
    _plan: &ShardedModelPlan,
    _upload_manifest: &gpt_oss_model_runner::ShardedUploadManifest,
    _kernel_dir: Option<&Path>,
    _dtype_mode: DTypeMode,
    _runtime_buffer_config: Option<RopeRuntimeBufferConfig>,
    _kv_cache_config: Option<KvCacheAllocationConfig>,
    _metadata_config: Option<MetadataAllocationConfig>,
    _fused_f16_options: FusedF16SmokeOptions,
    _f16_scratch_config: Option<F16ScratchAllocationConfig>,
) -> std::result::Result<
    (
        ShardedCudaResourceStatus,
        BTreeMap<DeviceId, ShardUploadCounts>,
        Option<ShardedRuntimeBufferStatus>,
        Option<ShardedKvCacheAllocationStatus>,
        Option<ShardedMetadataAllocationStatus>,
        Option<ShardedFusedF16AllocationStatus>,
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

#[cfg(feature = "cuda")]
fn fused_f32_tensor_filter_set(manifest: &ShardTensorManifest) -> BTreeSet<String> {
    let mut filter = BTreeSet::new();
    for &layer_idx in &manifest.absolute_layers {
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.bias",
        ] {
            let name = format!("model.layers.{layer_idx}.{suffix}");
            if manifest.should_load_required_tensor(&name) {
                filter.insert(name);
            }
        }
    }
    filter
}

fn fused_status_succeeded(status: &ShardedFusedF16AllocationStatus) -> bool {
    status.shards.iter().all(|shard| {
        shard.fused_f16_allocated
            || shard.fused_f16_status == FusedF16AllocationStatus::NotApplicable
    })
}

fn fused_status_has_allocated_norms(status: &ShardedFusedF16AllocationStatus) -> bool {
    status.shards.iter().any(|shard| {
        shard.f16_layernorm_count > 0
            || shard.f16_postnorm_count > 0
            || shard.f16_layernorm_total_bytes > 0
            || shard.f16_postnorm_total_bytes > 0
    })
}

fn fused_status_has_allocated_biases(status: &ShardedFusedF16AllocationStatus) -> bool {
    status.shards.iter().any(|shard| {
        shard.f16_qkv_bias_count > 0
            || shard.f16_o_proj_bias_count > 0
            || shard.f16_qkv_bias_total_bytes > 0
            || shard.f16_o_proj_bias_total_bytes > 0
    })
}

fn classify_fused_f16_error(error: &str) -> &'static str {
    if error.contains("f16 bias") || error.contains("f32 bias") {
        BIAS_CONVERSION_ALLOCATION_BLOCKED_CLASSIFICATION
    } else if error.contains("f16 norm cast")
        || error.contains("f16 cast kernel")
        || error.contains("cast_f32_f16")
    {
        FUSED_QKV_NORM_ALLOCATION_CAST_ERROR_CLASSIFICATION
    } else {
        FUSED_QKV_ALLOCATION_BLOCKED_CLASSIFICATION
    }
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
    allow_restricted_sinks_override: bool,
    resource_construction_succeeded: bool,
    allocation_smoke_succeeded: bool,
    rope_metadata_allocation_attempted: bool,
    rope_metadata_allocation_succeeded: bool,
    kv_cache_allocation_attempted: bool,
    kv_cache_allocation_succeeded: bool,
    kv_num_blocks: Option<usize>,
    kv_block_size: Option<usize>,
    metadata_allocation_attempted: bool,
    metadata_allocation_succeeded: bool,
    metadata_config: Option<MetadataAllocationConfig>,
    runtime_buffer_status: Option<&ShardedRuntimeBufferStatus>,
    kv_cache_status: Option<&ShardedKvCacheAllocationStatus>,
    metadata_status: Option<&ShardedMetadataAllocationStatus>,
    fused_f16_options: FusedF16SmokeOptions,
    fused_f16_status: Option<&ShardedFusedF16AllocationStatus>,
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
        allow_restricted_sinks_override,
        resource_construction_succeeded,
        allocation_smoke_succeeded,
        rope_metadata_allocation_attempted,
        rope_metadata_allocation_succeeded,
        kv_cache_allocation_attempted,
        kv_cache_allocation_succeeded,
        kv_num_blocks,
        kv_block_size,
        metadata_allocation_attempted,
        metadata_allocation_succeeded,
        metadata_config,
        runtime_buffer_status,
        kv_cache_status,
        metadata_status,
        fused_f16_options,
        fused_f16_status,
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
    allow_restricted_sinks_override: bool,
    resource_construction_succeeded: bool,
    allocation_smoke_succeeded: bool,
    rope_metadata_allocation_attempted: bool,
    rope_metadata_allocation_succeeded: bool,
    kv_cache_allocation_attempted: bool,
    kv_cache_allocation_succeeded: bool,
    kv_num_blocks: Option<usize>,
    kv_block_size: Option<usize>,
    metadata_allocation_attempted: bool,
    metadata_allocation_succeeded: bool,
    metadata_config: Option<MetadataAllocationConfig>,
    runtime_buffer_status: Option<&ShardedRuntimeBufferStatus>,
    kv_cache_status: Option<&ShardedKvCacheAllocationStatus>,
    metadata_status: Option<&ShardedMetadataAllocationStatus>,
    fused_f16_options: FusedF16SmokeOptions,
    fused_f16_status: Option<&ShardedFusedF16AllocationStatus>,
    error: Option<String>,
) -> SplitAllocationSmokeReport {
    let resource_by_device = resource_status
        .shards
        .iter()
        .map(|status| (status.device_id, status))
        .collect::<HashMap<_, _>>();
    let runtime_buffer_by_device = runtime_buffer_status
        .map(|status| {
            status
                .shards
                .iter()
                .map(|status| (status.device_id, status))
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();
    let kv_cache_by_device = kv_cache_status
        .map(|status| {
            status
                .shards
                .iter()
                .map(|status| (status.device_id, status))
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();
    let metadata_by_device = metadata_status
        .map(|status| {
            status
                .shards
                .iter()
                .map(|status| (status.device_id, status))
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();
    let fused_f16_by_device = fused_f16_status
        .map(|status| {
            status
                .shards
                .iter()
                .map(|status| (status.device_id, status))
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();
    let fused_f16_allocation_succeeded = fused_f16_options.allocate_fused_f16
        && fused_f16_status
            .map(fused_status_succeeded)
            .unwrap_or(false);

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
        header_merge_policy: header_manifest.merge_policy.as_str().into(),
        restricted_sinks_override_enabled: allow_restricted_sinks_override,
        overridden_tensor_count: header_manifest.overridden_tensor_count(),
        overridden_tensor_names: header_manifest.overridden_tensor_names.clone(),
        resource_construction_succeeded,
        allocation_smoke_succeeded,
        rope_metadata_allocation_attempted,
        rope_metadata_allocation_succeeded,
        kv_cache_allocation_attempted,
        kv_cache_allocation_succeeded,
        kv_num_blocks,
        kv_block_size,
        metadata_allocation_attempted,
        metadata_allocation_succeeded,
        metadata_mode: metadata_config.map(|config| config.mode.as_str().to_string()),
        metadata_num_tokens: metadata_config.map(|config| config.num_tokens),
        metadata_num_seqs: metadata_config.map(|config| config.num_seqs),
        metadata_context_len: metadata_config.map(|config| config.context_len),
        metadata_block_size: metadata_config.map(|config| config.block_size),
        metadata_graph_max_blocks: metadata_config.map(|config| config.graph_max_blocks()),
        metadata_error: error.clone().filter(|_| metadata_allocation_attempted),
        fused_f16_allocation_attempted: fused_f16_options.allocate_fused_f16,
        fused_f16_allocation_succeeded,
        f16_scratch_allocation_attempted: fused_f16_options.allocate_f16_scratch,
        f16_scratch_allocation_succeeded: false,
        f16_scratch_max_tokens: fused_f16_options.f16_scratch_max_tokens,
        fused_f16_error: error
            .clone()
            .filter(|_| fused_f16_options.allocate_fused_f16),
        f16_scratch_error: error
            .clone()
            .filter(|_| fused_f16_options.allocate_f16_scratch),
        kernel_dir: kernel_dir.map(|path| path.display().to_string()),
        omitted_allocations: omitted_allocations(
            rope_metadata_allocation_succeeded,
            kv_cache_allocation_attempted,
            metadata_allocation_attempted,
            fused_f16_options.allocate_fused_f16,
            fused_f16_options.allocate_f16_scratch,
        ),
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
                let runtime_buffer = runtime_buffer_by_device.get(&shard.device_id).copied();
                let kv_cache = kv_cache_by_device.get(&shard.device_id).copied();
                let metadata = metadata_by_device.get(&shard.device_id).copied();
                let fused_f16 = fused_f16_by_device.get(&shard.device_id).copied();
                render_shard_report(
                    shard,
                    counts,
                    resource,
                    resource_construction_succeeded,
                    runtime_buffer,
                    kv_cache,
                    metadata,
                    fused_f16,
                )
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
    runtime_buffer: Option<&CudaShardRuntimeBufferStatus>,
    kv_cache: Option<&CudaShardKvCacheAllocationStatus>,
    metadata: Option<&CudaShardMetadataAllocationStatus>,
    fused_f16: Option<&CudaShardFusedF16AllocationStatus>,
) -> SplitAllocationSmokeShardReport {
    let (
        rope_allocated,
        rope_cos_elements,
        rope_sin_elements,
        rope_total_bytes,
        metadata_allocated,
        metadata_status,
        metadata_deferred_reason,
        runtime_buffer_error,
    ) = runtime_buffer
        .map(|status| {
            (
                status.rope_allocated,
                status.rope_cos_elements,
                status.rope_sin_elements,
                status.rope_total_bytes,
                status.metadata_allocated,
                status.metadata_status.as_str().to_string(),
                status.metadata_deferred_reason.clone(),
                status.runtime_buffer_error.clone(),
            )
        })
        .unwrap_or_else(|| {
            (
                false,
                0,
                0,
                0,
                false,
                "not_applicable".to_string(),
                None,
                None,
            )
        });
    let (
        metadata_mode,
        metadata_num_tokens,
        metadata_num_seqs,
        metadata_graph_max_blocks,
        metadata_packed_elements,
        metadata_packed_bytes,
        metadata_token_ids_len,
        metadata_positions_len,
        metadata_context_lens_len,
        metadata_block_tables_len,
        metadata_slot_mapping_len,
        metadata_seq_start_pos_len,
        metadata_max_context_len,
        metadata_error,
        metadata_allocated,
        metadata_status,
        metadata_deferred_reason,
    ) = metadata
        .map(|status| {
            (
                Some(status.mode.as_str().to_string()),
                status.num_tokens,
                status.num_seqs,
                status.graph_max_blocks,
                status.packed_elements,
                status.packed_bytes,
                status.token_ids_len,
                status.positions_len,
                status.context_lens_len,
                status.block_tables_len,
                status.slot_mapping_len,
                status.seq_start_pos_len,
                status.max_context_len,
                status.metadata_error.clone(),
                status.metadata_allocated,
                status.metadata_status.as_str().to_string(),
                None,
            )
        })
        .unwrap_or((
            None,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            metadata_allocated,
            metadata_status,
            metadata_deferred_reason,
        ));

    let (
        kv_cache_allocated,
        kv_cache_entry_count,
        kv_cache_layers,
        kv_cache_local_indices,
        kv_key_total_bytes,
        kv_value_total_bytes,
        kv_total_bytes,
        kv_cache_entries,
        kv_cache_error,
    ) = kv_cache
        .map(|status| {
            (
                status.kv_cache_allocated,
                status.entries.len(),
                status
                    .entries
                    .iter()
                    .map(|entry| entry.absolute_layer_idx)
                    .collect::<Vec<_>>(),
                status
                    .entries
                    .iter()
                    .map(|entry| entry.local_cache_idx)
                    .collect::<Vec<_>>(),
                status.key_total_bytes,
                status.value_total_bytes,
                status.total_bytes,
                status
                    .entries
                    .iter()
                    .map(|entry| KvCacheEntryReport {
                        absolute_layer_idx: entry.absolute_layer_idx,
                        local_cache_idx: entry.local_cache_idx,
                        key_bytes: entry.key_bytes,
                        value_bytes: entry.value_bytes,
                    })
                    .collect::<Vec<_>>(),
                status.kv_cache_error.clone(),
            )
        })
        .unwrap_or_else(|| (false, 0, Vec::new(), Vec::new(), 0, 0, 0, Vec::new(), None));
    let (
        fused_f16_allocated,
        fused_f16_status,
        fused_qkv_weight_count,
        fused_qkv_total_bytes,
        fused_gate_up_weight_count,
        fused_gate_up_total_bytes,
        f16_layernorm_total_bytes,
        f16_postnorm_total_bytes,
        f16_layernorm_count,
        f16_postnorm_count,
        f16_qkv_bias_count,
        f16_o_proj_bias_count,
        f16_qkv_bias_total_bytes,
        f16_o_proj_bias_total_bytes,
        embedding_f16_allocated,
        final_norm_f16_allocated,
        fused_total_bytes,
        fused_layer_absolute_indices,
        fused_layer_statuses,
        fused_deferred_reason,
        fused_error,
        f16_scratch_allocated,
        f16_scratch_status,
        f16_scratch_bytes,
        f16_scratch_deferred_reason,
        f16_scratch_error,
    ) = fused_f16
        .map(|status| {
            (
                status.fused_f16_allocated,
                status.fused_f16_status.as_str().to_string(),
                status.fused_qkv_weight_count,
                status.fused_qkv_total_bytes,
                status.fused_gate_up_weight_count,
                status.fused_gate_up_total_bytes,
                status.f16_layernorm_total_bytes,
                status.f16_postnorm_total_bytes,
                status.f16_layernorm_count,
                status.f16_postnorm_count,
                status.f16_qkv_bias_count,
                status.f16_o_proj_bias_count,
                status.f16_qkv_bias_total_bytes,
                status.f16_o_proj_bias_total_bytes,
                status.embedding_f16_allocated,
                status.final_norm_f16_allocated,
                status.fused_total_bytes,
                status.fused_layer_absolute_indices.clone(),
                status
                    .fused_layer_statuses
                    .iter()
                    .map(render_fused_layer_report)
                    .collect::<Vec<_>>(),
                status.fused_deferred_reason.clone(),
                status.fused_error.clone(),
                status.f16_scratch_allocated,
                status.f16_scratch_status.as_str().to_string(),
                status.f16_scratch_bytes,
                status.f16_scratch_deferred_reason.clone(),
                status.f16_scratch_error.clone(),
            )
        })
        .unwrap_or_else(|| {
            (
                false,
                FusedF16AllocationStatus::NotApplicable.as_str().to_string(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                false,
                false,
                0,
                Vec::new(),
                Vec::new(),
                None,
                None,
                false,
                FusedF16AllocationStatus::NotApplicable.as_str().to_string(),
                0,
                None,
                None,
            )
        });

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
        rope_allocated,
        rope_cos_elements,
        rope_sin_elements,
        rope_total_bytes,
        metadata_allocated,
        metadata_status,
        metadata_deferred_reason,
        runtime_buffer_error,
        kv_cache_allocated,
        kv_cache_entry_count,
        kv_cache_layers,
        kv_cache_local_indices,
        kv_key_total_bytes,
        kv_value_total_bytes,
        kv_total_bytes,
        kv_cache_entries,
        kv_cache_error,
        metadata_mode,
        metadata_num_tokens,
        metadata_num_seqs,
        metadata_graph_max_blocks,
        metadata_packed_elements,
        metadata_packed_bytes,
        metadata_token_ids_len,
        metadata_positions_len,
        metadata_context_lens_len,
        metadata_block_tables_len,
        metadata_slot_mapping_len,
        metadata_seq_start_pos_len,
        metadata_max_context_len,
        metadata_error,
        fused_f16_allocated,
        fused_f16_status,
        fused_qkv_weight_count,
        fused_qkv_total_bytes,
        fused_gate_up_weight_count,
        fused_gate_up_total_bytes,
        f16_layernorm_total_bytes,
        f16_postnorm_total_bytes,
        f16_layernorm_count,
        f16_postnorm_count,
        f16_qkv_bias_count,
        f16_o_proj_bias_count,
        f16_qkv_bias_total_bytes,
        f16_o_proj_bias_total_bytes,
        embedding_f16_allocated,
        final_norm_f16_allocated,
        fused_total_bytes,
        fused_layer_absolute_indices,
        fused_layer_statuses,
        fused_deferred_reason,
        fused_error,
        f16_scratch_allocated,
        f16_scratch_status,
        f16_scratch_bytes,
        f16_scratch_deferred_reason,
        f16_scratch_error,
    }
}

fn render_fused_layer_report(status: &CudaLayerFusedF16AllocationStatus) -> FusedLayerReport {
    FusedLayerReport {
        absolute_layer_idx: status.absolute_layer_idx,
        local_layer_idx: status.local_layer_idx,
        fused_qkv_allocated: status.fused_qkv_allocated,
        fused_qkv_status: status.fused_qkv_status.as_str().to_string(),
        fused_qkv_bytes: status.fused_qkv_bytes,
        fused_gate_up_allocated: status.fused_gate_up_allocated,
        fused_gate_up_status: status.fused_gate_up_status.as_str().to_string(),
        fused_gate_up_bytes: status.fused_gate_up_bytes,
        layernorm_f16_status: status.layernorm_f16_status.as_str().to_string(),
        layernorm_f16_bytes: status.layernorm_f16_bytes,
        postnorm_f16_status: status.postnorm_f16_status.as_str().to_string(),
        postnorm_f16_bytes: status.postnorm_f16_bytes,
        qkv_bias_f16_status: status.qkv_bias_f16_status.as_str().to_string(),
        qkv_bias_f16_bytes: status.qkv_bias_f16_bytes,
        o_proj_bias_f16_status: status.o_proj_bias_f16_status.as_str().to_string(),
        o_proj_bias_f16_bytes: status.o_proj_bias_f16_bytes,
        layer_error: status.layer_error.clone(),
    }
}

fn error_report(
    classification: &str,
    model_path: &Path,
    device_map_spec: &str,
    selected_device: usize,
    dtype_mode: DTypeMode,
    kernel_dir: Option<&Path>,
    rope_metadata_allocation_attempted: bool,
    kv_cache_allocation_attempted: bool,
    kv_num_blocks: Option<usize>,
    kv_block_size: Option<usize>,
    metadata_allocation_attempted: bool,
    metadata_mode: Option<MetadataModeArg>,
    metadata_num_tokens: Option<usize>,
    metadata_num_seqs: Option<usize>,
    metadata_context_len: Option<usize>,
    metadata_block_size: Option<usize>,
    metadata_graph_max_blocks: Option<usize>,
    fused_f16_options: FusedF16SmokeOptions,
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
        header_merge_policy: SafetensorHeaderMergePolicy::RejectDuplicates
            .as_str()
            .into(),
        restricted_sinks_override_enabled: false,
        overridden_tensor_count: 0,
        overridden_tensor_names: Vec::new(),
        resource_construction_succeeded: false,
        allocation_smoke_succeeded: false,
        rope_metadata_allocation_attempted,
        rope_metadata_allocation_succeeded: false,
        kv_cache_allocation_attempted,
        kv_cache_allocation_succeeded: false,
        kv_num_blocks,
        kv_block_size,
        metadata_allocation_attempted,
        metadata_allocation_succeeded: false,
        metadata_mode: metadata_mode.map(|mode| mode.to_runner_mode().as_str().to_string()),
        metadata_num_tokens,
        metadata_num_seqs,
        metadata_context_len,
        metadata_block_size,
        metadata_graph_max_blocks,
        metadata_error: error.clone(),
        fused_f16_allocation_attempted: fused_f16_options.allocate_fused_f16,
        fused_f16_allocation_succeeded: false,
        f16_scratch_allocation_attempted: fused_f16_options.allocate_f16_scratch,
        f16_scratch_allocation_succeeded: false,
        f16_scratch_max_tokens: fused_f16_options.f16_scratch_max_tokens,
        fused_f16_error: error
            .clone()
            .filter(|_| fused_f16_options.allocate_fused_f16),
        f16_scratch_error: error
            .clone()
            .filter(|_| fused_f16_options.allocate_f16_scratch),
        kernel_dir: kernel_dir.map(|path| path.display().to_string()),
        omitted_allocations: omitted_allocations(
            false,
            kv_cache_allocation_attempted,
            metadata_allocation_attempted,
            fused_f16_options.allocate_fused_f16,
            fused_f16_options.allocate_f16_scratch,
        ),
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
    allow_restricted_sinks_override: bool,
    rope_metadata_allocation_attempted: bool,
    kv_cache_allocation_attempted: bool,
    kv_num_blocks: Option<usize>,
    kv_block_size: Option<usize>,
    metadata_allocation_attempted: bool,
    metadata_mode: Option<MetadataModeArg>,
    metadata_num_tokens: Option<usize>,
    metadata_num_seqs: Option<usize>,
    metadata_context_len: Option<usize>,
    metadata_block_size: Option<usize>,
    metadata_graph_max_blocks: Option<usize>,
    fused_f16_options: FusedF16SmokeOptions,
    error: Option<String>,
) -> SplitAllocationSmokeReport {
    let mut report = error_report(
        classification,
        model_path,
        device_map_spec,
        selected_device,
        dtype_mode,
        kernel_dir,
        rope_metadata_allocation_attempted,
        kv_cache_allocation_attempted,
        kv_num_blocks,
        kv_block_size,
        metadata_allocation_attempted,
        metadata_mode,
        metadata_num_tokens,
        metadata_num_seqs,
        metadata_context_len,
        metadata_block_size,
        metadata_graph_max_blocks,
        fused_f16_options,
        error,
    );
    report.num_layers = Some(config.num_layers);
    report.tie_word_embeddings = Some(config.tie_word_embeddings);
    report.header_merge_policy = header_merge_policy(allow_restricted_sinks_override)
        .as_str()
        .into();
    report.restricted_sinks_override_enabled = allow_restricted_sinks_override;
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
    allow_restricted_sinks_override: bool,
    rope_metadata_allocation_attempted: bool,
    kv_cache_allocation_attempted: bool,
    kv_num_blocks: Option<usize>,
    kv_block_size: Option<usize>,
    metadata_allocation_attempted: bool,
    metadata_mode: Option<MetadataModeArg>,
    metadata_num_tokens: Option<usize>,
    metadata_num_seqs: Option<usize>,
    metadata_context_len: Option<usize>,
    metadata_block_size: Option<usize>,
    metadata_graph_max_blocks: Option<usize>,
    fused_f16_options: FusedF16SmokeOptions,
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
        allow_restricted_sinks_override,
        rope_metadata_allocation_attempted,
        kv_cache_allocation_attempted,
        kv_num_blocks,
        kv_block_size,
        metadata_allocation_attempted,
        metadata_mode,
        metadata_num_tokens,
        metadata_num_seqs,
        metadata_context_len,
        metadata_block_size,
        metadata_graph_max_blocks,
        fused_f16_options,
        error,
    );
    report.has_lm_head_weight = Some(header_manifest.has_lm_head_weight());
    report.tensor_count_from_headers = Some(header_manifest.tensors.len());
    report.total_header_tensor_bytes =
        Some(header_manifest.tensors.iter().map(|t| t.byte_size).sum());
    report.header_merge_policy = header_manifest.merge_policy.as_str().into();
    report.restricted_sinks_override_enabled = allow_restricted_sinks_override;
    report.overridden_tensor_count = header_manifest.overridden_tensor_count();
    report.overridden_tensor_names = header_manifest.overridden_tensor_names.clone();
    report
}

fn tensor_byte_map(header_manifest: &SafetensorHeaderManifest) -> BTreeMap<String, usize> {
    header_manifest
        .tensors
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor.byte_size))
        .collect()
}

fn omitted_allocations(
    rope_allocated: bool,
    kv_cache_attempted: bool,
    metadata_attempted: bool,
    fused_f16_attempted: bool,
    f16_scratch_attempted: bool,
) -> Vec<String> {
    OMITTED_ALLOCATIONS
        .iter()
        .filter(|name| {
            !(rope_allocated && **name == "rope_tables")
                && !(kv_cache_attempted && **name == "kv_cache")
                && !(metadata_attempted && **name == "metadata_buffers")
                && !(fused_f16_attempted
                    && (**name == "fused_qkv_weights" || **name == "fused_gate_up_weights"))
                && !(f16_scratch_attempted && **name == "f16_scratch")
        })
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
        write_config_with_max_position(dir, num_layers, tie_word_embeddings, 16);
    }

    fn write_config_with_max_position(
        dir: &Path,
        num_layers: usize,
        tie_word_embeddings: bool,
        max_position: usize,
    ) {
        let config = serde_json::json!({
            "num_hidden_layers": num_layers,
            "tie_word_embeddings": tie_word_embeddings,
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": max_position,
            "rope_theta": 10000.0,
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
        split_report_for_rope_metadata(dir, tie_word_embeddings, false)
    }

    fn split_report_for_rope_metadata(
        dir: &Path,
        tie_word_embeddings: Option<bool>,
        allocate_rope_metadata: bool,
    ) -> SplitAllocationSmokeReport {
        build_split_allocation_smoke_report(
            dir,
            "split:0-11@0,12-23@1",
            0,
            None,
            DTypeMode::F16,
            tie_word_embeddings,
            false,
            allocate_rope_metadata,
            false,
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            false,
            false,
            None,
            false,
        )
    }

    fn split_report_for_kv_cache(
        dir: &Path,
        allocate_rope_metadata: bool,
        allocate_kv_cache: bool,
        kv_num_blocks: Option<usize>,
        kv_block_size: Option<usize>,
    ) -> SplitAllocationSmokeReport {
        build_split_allocation_smoke_report(
            dir,
            "split:0-11@0,12-23@1",
            0,
            None,
            DTypeMode::F16,
            None,
            false,
            allocate_rope_metadata,
            allocate_kv_cache,
            kv_num_blocks,
            kv_block_size,
            false,
            None,
            None,
            None,
            None,
            None,
            false,
            false,
            None,
            false,
        )
    }

    fn split_report_for_metadata(
        dir: &Path,
        allocate_kv_cache: bool,
        kv_num_blocks: Option<usize>,
        kv_block_size: Option<usize>,
        metadata_num_tokens: Option<usize>,
        metadata_num_seqs: Option<usize>,
        metadata_context_len: Option<usize>,
        metadata_block_size: Option<usize>,
    ) -> SplitAllocationSmokeReport {
        build_split_allocation_smoke_report(
            dir,
            "split:0-11@0,12-23@1",
            0,
            None,
            DTypeMode::F16,
            None,
            false,
            false,
            allocate_kv_cache,
            kv_num_blocks,
            kv_block_size,
            true,
            Some(MetadataModeArg::Decode),
            metadata_num_tokens,
            metadata_num_seqs,
            metadata_context_len,
            metadata_block_size,
            false,
            false,
            None,
            false,
        )
    }

    fn split_report_for_fused(
        dir: &Path,
        allocate_fused_f16: bool,
        allocate_f16_scratch: bool,
        f16_scratch_max_tokens: Option<usize>,
    ) -> SplitAllocationSmokeReport {
        build_split_allocation_smoke_report(
            dir,
            "split:0-11@0,12-23@1",
            0,
            None,
            DTypeMode::F16,
            None,
            false,
            false,
            false,
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            allocate_fused_f16,
            allocate_f16_scratch,
            f16_scratch_max_tokens,
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
    fn default_split_allocation_smoke_does_not_attempt_rope_metadata() {
        let dir = fixture_with_lm_head();

        let report = split_report_for(&dir, None);

        assert!(!report.rope_metadata_allocation_attempted);
        assert!(!report.rope_metadata_allocation_succeeded);
        for shard in &report.shards {
            assert!(!shard.rope_allocated);
            assert_eq!(shard.rope_cos_elements, 0);
            assert_eq!(shard.rope_sin_elements, 0);
            assert_eq!(shard.rope_total_bytes, 0);
            assert!(!shard.metadata_allocated);
            assert_eq!(shard.metadata_status, "not_applicable");
            assert!(shard.metadata_deferred_reason.is_none());
        }
    }

    #[test]
    fn rope_metadata_flag_surfaces_planned_rope_and_deferred_metadata() {
        let dir = fixture_with_lm_head();

        let report = split_report_for_rope_metadata(&dir, None, true);

        assert!(report.rope_metadata_allocation_attempted);
        assert!(!report.rope_metadata_allocation_succeeded);
        for shard in &report.shards {
            assert!(!shard.rope_allocated);
            assert_eq!(shard.rope_cos_elements, 32);
            assert_eq!(shard.rope_sin_elements, 32);
            assert_eq!(shard.rope_total_bytes, 256);
            assert!(!shard.metadata_allocated);
            assert_eq!(shard.metadata_status, "deferred");
            assert!(shard
                .metadata_deferred_reason
                .as_deref()
                .unwrap()
                .contains("request-shaped metadata"));
        }
    }

    #[test]
    fn rope_metadata_flag_keeps_non_executing_allocations_omitted() {
        let dir = fixture_with_lm_head();

        let report = split_report_for_rope_metadata(&dir, None, true);

        for expected in [
            "kv_cache",
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
    fn default_split_allocation_smoke_does_not_attempt_kv_cache() {
        let dir = fixture_with_lm_head();

        let report = split_report_for(&dir, None);

        assert!(!report.kv_cache_allocation_attempted);
        assert!(!report.kv_cache_allocation_succeeded);
        assert_eq!(report.kv_num_blocks, None);
        assert_eq!(report.kv_block_size, None);
        assert!(report.omitted_allocations.contains(&"kv_cache".to_string()));
        for shard in &report.shards {
            assert!(!shard.kv_cache_allocated);
            assert_eq!(shard.kv_cache_entry_count, 0);
            assert!(shard.kv_cache_layers.is_empty());
            assert!(shard.kv_cache_local_indices.is_empty());
            assert_eq!(shard.kv_total_bytes, 0);
        }
    }

    #[test]
    fn kv_cache_flag_surfaces_planned_absolute_and_local_indices() {
        let dir = fixture_with_lm_head();

        let report = split_report_for_kv_cache(&dir, false, true, Some(1), Some(16));

        assert!(report.kv_cache_allocation_attempted);
        assert!(!report.kv_cache_allocation_succeeded);
        assert_eq!(report.kv_num_blocks, Some(1));
        assert_eq!(report.kv_block_size, Some(16));
        assert!(!report.omitted_allocations.contains(&"kv_cache".to_string()));

        let gpu0 = shard(&report, 0);
        assert!(!gpu0.kv_cache_allocated);
        assert_eq!(gpu0.kv_cache_entry_count, 12);
        assert_eq!(gpu0.kv_cache_layers, (0..12).collect::<Vec<_>>());
        assert_eq!(gpu0.kv_cache_local_indices, (0..12).collect::<Vec<_>>());

        let gpu1 = shard(&report, 1);
        assert!(!gpu1.kv_cache_allocated);
        assert_eq!(gpu1.kv_cache_entry_count, 12);
        assert_eq!(gpu1.kv_cache_layers, (12..24).collect::<Vec<_>>());
        assert_eq!(gpu1.kv_cache_local_indices, (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn kv_cache_flag_reports_bytes_per_shard() {
        let dir = fixture_with_lm_head();

        let report = split_report_for_kv_cache(&dir, false, true, Some(1), Some(16));

        let per_cache_bytes = 1 * 16 * 2 * 4 * std::mem::size_of::<half::f16>();
        for shard in &report.shards {
            assert_eq!(shard.kv_cache_entry_count, 12);
            assert_eq!(shard.kv_key_total_bytes, per_cache_bytes * 12);
            assert_eq!(shard.kv_value_total_bytes, per_cache_bytes * 12);
            assert_eq!(shard.kv_total_bytes, per_cache_bytes * 24);
            assert_eq!(shard.kv_cache_entries.len(), 12);
            assert!(shard
                .kv_cache_entries
                .iter()
                .all(|entry| entry.key_bytes == per_cache_bytes));
            assert!(shard
                .kv_cache_entries
                .iter()
                .all(|entry| entry.value_bytes == per_cache_bytes));
        }
    }

    #[test]
    fn kv_cache_flag_requires_num_blocks() {
        let dir = fixture_with_lm_head();

        let report = split_report_for_kv_cache(&dir, false, true, None, Some(16));

        assert_eq!(report.classification, CONFIG_ERROR_CLASSIFICATION);
        assert!(report.kv_cache_allocation_attempted);
        assert!(report.error.as_deref().unwrap().contains("--kv-num-blocks"));
    }

    #[test]
    fn kv_cache_flag_requires_block_size() {
        let dir = fixture_with_lm_head();

        let report = split_report_for_kv_cache(&dir, false, true, Some(1), None);

        assert_eq!(report.classification, CONFIG_ERROR_CLASSIFICATION);
        assert!(report.kv_cache_allocation_attempted);
        assert!(report.error.as_deref().unwrap().contains("--kv-block-size"));
    }

    #[test]
    fn kv_cache_status_is_independent_from_deferred_metadata() {
        let dir = fixture_with_lm_head();

        let report = split_report_for_kv_cache(&dir, true, true, Some(1), Some(16));

        assert!(report.rope_metadata_allocation_attempted);
        assert!(!report.rope_metadata_allocation_succeeded);
        assert!(report.kv_cache_allocation_attempted);
        assert!(!report.kv_cache_allocation_succeeded);
        for shard in &report.shards {
            assert_eq!(shard.metadata_status, "deferred");
            assert_eq!(shard.kv_cache_entry_count, 12);
        }
    }

    #[test]
    fn default_split_allocation_smoke_does_not_attempt_metadata_allocation() {
        let dir = fixture_with_lm_head();

        let report = split_report_for(&dir, None);

        assert!(!report.metadata_allocation_attempted);
        assert!(!report.metadata_allocation_succeeded);
        assert_eq!(report.metadata_mode, None);
        assert!(report
            .omitted_allocations
            .contains(&"metadata_buffers".to_string()));
    }

    #[test]
    fn metadata_flag_surfaces_decode_shape_status_per_shard() {
        let dir = fixture_with_lm_head();

        let report =
            split_report_for_metadata(&dir, false, None, None, Some(2), Some(2), Some(1), Some(16));

        assert!(report.metadata_allocation_attempted);
        assert!(!report.metadata_allocation_succeeded);
        assert_eq!(report.metadata_mode.as_deref(), Some("decode"));
        assert_eq!(report.metadata_num_tokens, Some(2));
        assert_eq!(report.metadata_num_seqs, Some(2));
        assert_eq!(report.metadata_context_len, Some(1));
        assert_eq!(report.metadata_block_size, Some(16));
        assert_eq!(report.metadata_graph_max_blocks, Some(1));
        assert!(!report
            .omitted_allocations
            .contains(&"metadata_buffers".to_string()));

        for shard in &report.shards {
            assert!(!shard.metadata_allocated);
            assert_eq!(shard.metadata_status, "deferred");
            assert_eq!(shard.metadata_mode.as_deref(), Some("decode"));
            assert_eq!(shard.metadata_num_tokens, 2);
            assert_eq!(shard.metadata_num_seqs, 2);
            assert_eq!(shard.metadata_graph_max_blocks, 1);
            assert_eq!(shard.metadata_token_ids_len, 2);
            assert_eq!(shard.metadata_positions_len, 2);
            assert_eq!(shard.metadata_context_lens_len, 2);
            assert_eq!(shard.metadata_block_tables_len, 2);
            assert_eq!(shard.metadata_slot_mapping_len, 2);
            assert_eq!(shard.metadata_seq_start_pos_len, 3);
            assert_eq!(shard.metadata_packed_elements, 13);
            assert_eq!(shard.metadata_packed_bytes, 13 * std::mem::size_of::<i32>());
            assert_eq!(shard.metadata_max_context_len, 1);
        }
    }

    #[test]
    fn metadata_flag_rejects_tokens_sequences_mismatch() {
        let dir = fixture_with_lm_head();

        let report =
            split_report_for_metadata(&dir, false, None, None, Some(2), Some(1), Some(1), Some(16));

        assert_eq!(report.classification, CONFIG_ERROR_CLASSIFICATION);
        assert!(report.metadata_allocation_attempted);
        assert!(report
            .error
            .as_deref()
            .unwrap()
            .contains("metadata-num-tokens == metadata-num-seqs"));
    }

    #[test]
    fn metadata_flag_rejects_zero_context_len() {
        let dir = fixture_with_lm_head();

        let report =
            split_report_for_metadata(&dir, false, None, None, Some(1), Some(1), Some(0), Some(16));

        assert_eq!(report.classification, CONFIG_ERROR_CLASSIFICATION);
        assert!(report
            .error
            .as_deref()
            .unwrap()
            .contains("metadata-context-len"));
    }

    #[test]
    fn metadata_flag_rejects_zero_block_size() {
        let dir = fixture_with_lm_head();

        let report =
            split_report_for_metadata(&dir, false, None, None, Some(1), Some(1), Some(1), Some(0));

        assert_eq!(report.classification, CONFIG_ERROR_CLASSIFICATION);
        assert!(report
            .error
            .as_deref()
            .unwrap()
            .contains("metadata-block-size"));
    }

    #[test]
    fn metadata_flag_rejects_kv_block_size_mismatch() {
        let dir = fixture_with_lm_head();

        let report = split_report_for_metadata(
            &dir,
            true,
            Some(1),
            Some(8),
            Some(1),
            Some(1),
            Some(1),
            Some(16),
        );

        assert_eq!(report.classification, CONFIG_ERROR_CLASSIFICATION);
        assert!(report.error.as_deref().unwrap().contains("kv-block-size"));
    }

    #[test]
    fn metadata_flag_rejects_generated_blocks_beyond_kv_blocks() {
        let dir = fixture_with_lm_head();
        write_config_with_max_position(&dir, 24, true, 64);

        let report = split_report_for_metadata(
            &dir,
            true,
            Some(1),
            Some(16),
            Some(1),
            Some(1),
            Some(17),
            Some(16),
        );

        assert_eq!(report.classification, CONFIG_ERROR_CLASSIFICATION);
        assert!(report.error.as_deref().unwrap().contains("kv-num-blocks"));
    }

    #[test]
    fn metadata_mode_cli_rejects_unsupported_values() {
        let parsed = Cli::try_parse_from([
            "multi_gpu_layer_sharding_split_allocation_smoke",
            "--model",
            "/tmp/model",
            "--device-map",
            "single",
            "--metadata-mode",
            "prefill",
        ]);

        assert!(parsed.is_err());
    }

    #[test]
    fn default_split_allocation_smoke_does_not_attempt_fused_or_scratch() {
        let dir = fixture_with_lm_head();

        let report = split_report_for(&dir, None);

        assert!(!report.fused_f16_allocation_attempted);
        assert!(!report.fused_f16_allocation_succeeded);
        assert!(!report.f16_scratch_allocation_attempted);
        assert!(!report.f16_scratch_allocation_succeeded);
        assert_eq!(report.f16_scratch_max_tokens, None);
        assert!(report
            .omitted_allocations
            .contains(&"fused_qkv_weights".to_string()));
        assert!(report
            .omitted_allocations
            .contains(&"fused_gate_up_weights".to_string()));
        assert!(report
            .omitted_allocations
            .contains(&"f16_scratch".to_string()));
        for shard in &report.shards {
            assert_eq!(shard.fused_f16_status, "not_applicable");
            assert_eq!(shard.f16_scratch_status, "not_applicable");
            assert!(shard.fused_layer_absolute_indices.is_empty());
        }
    }

    #[test]
    fn fused_f16_error_classification_maps_norm_cast_errors() {
        let classification = classify_fused_f16_error(
            "gpu error: shard 0 f16 norm cast kernel load failed: gpu error: module 'cast_fp' not loaded",
        );

        assert_eq!(
            classification,
            FUSED_QKV_NORM_ALLOCATION_CAST_ERROR_CLASSIFICATION
        );
    }

    #[test]
    fn fused_f16_error_classification_maps_bias_errors() {
        let classification = classify_fused_f16_error(
            "gpu error: shard 0 f16 bias partial QKV bias set absolute layer 12 missing tensors",
        );

        assert_eq!(
            classification,
            BIAS_CONVERSION_ALLOCATION_BLOCKED_CLASSIFICATION
        );
    }

    #[test]
    fn fused_f16_error_classification_keeps_qkv_errors_distinct() {
        let classification = classify_fused_f16_error(
            "gpu error: shard 0 fused QKV q copy failed absolute layer 12",
        );

        assert_eq!(classification, FUSED_QKV_ALLOCATION_BLOCKED_CLASSIFICATION);
    }

    #[test]
    fn fused_f16_flag_surfaces_deferred_plan_status_per_shard() {
        let dir = fixture_with_lm_head();

        let report = split_report_for_fused(&dir, true, false, None);

        assert_eq!(report.classification, FUSED_F16_PLAN_STATUS_CLASSIFICATION);
        assert!(report.fused_f16_allocation_attempted);
        assert!(!report.fused_f16_allocation_succeeded);
        assert!(!report
            .omitted_allocations
            .contains(&"fused_qkv_weights".to_string()));
        assert!(!report
            .omitted_allocations
            .contains(&"fused_gate_up_weights".to_string()));

        let gpu0 = shard(&report, 0);
        assert_eq!(
            gpu0.fused_layer_absolute_indices,
            (0..12).collect::<Vec<_>>()
        );
        assert_eq!(gpu0.fused_f16_status, "deferred");
        let gpu0_layer0 = gpu0
            .fused_layer_statuses
            .iter()
            .find(|layer| layer.absolute_layer_idx == 0)
            .unwrap();
        assert_eq!(gpu0_layer0.local_layer_idx, 0);
        assert_eq!(gpu0_layer0.fused_qkv_status, "not_applicable");
        assert!(!gpu0_layer0.fused_qkv_allocated);
        assert!(gpu0
            .fused_deferred_reason
            .as_deref()
            .unwrap()
            .contains("GpuModelRunner::fuse_weights"));

        let gpu1 = shard(&report, 1);
        assert_eq!(
            gpu1.fused_layer_absolute_indices,
            (12..24).collect::<Vec<_>>()
        );
        assert_eq!(gpu1.fused_f16_status, "deferred");
        let gpu1_layer12 = gpu1
            .fused_layer_statuses
            .iter()
            .find(|layer| layer.absolute_layer_idx == 12)
            .unwrap();
        assert_eq!(gpu1_layer12.local_layer_idx, 0);
        assert_eq!(gpu1_layer12.fused_qkv_status, "not_applicable");
    }

    #[test]
    fn fused_f16_status_places_embedding_and_final_norm_on_owning_shards() {
        let dir = fixture_with_lm_head();

        let report = split_report_for_fused(&dir, true, false, None);

        let gpu0 = shard(&report, 0);
        assert!(gpu0.owns_embeddings);
        assert!(!gpu0.embedding_f16_allocated);
        assert!(!gpu0.final_norm_f16_allocated);

        let gpu1 = shard(&report, 1);
        assert!(gpu1.owns_final_head);
        assert!(!gpu1.embedding_f16_allocated);
        assert!(!gpu1.final_norm_f16_allocated);
        assert!(gpu1
            .required_tensor_names
            .contains(&"model.norm.weight".to_string()));
    }

    #[test]
    fn f16_scratch_flag_requires_max_tokens() {
        let dir = fixture_with_lm_head();

        let report = split_report_for_fused(&dir, false, true, None);

        assert_eq!(report.classification, CONFIG_ERROR_CLASSIFICATION);
        assert!(report.f16_scratch_allocation_attempted);
        assert!(report
            .error
            .as_deref()
            .unwrap()
            .contains("--f16-scratch-max-tokens"));
    }

    #[test]
    fn fused_f16_flag_requires_f16_dtype_mode() {
        let dir = fixture_with_lm_head();

        let report = build_split_allocation_smoke_report(
            &dir,
            "split:0-11@0,12-23@1",
            0,
            None,
            DTypeMode::F32,
            None,
            false,
            false,
            false,
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            true,
            false,
            None,
            false,
        );

        assert_eq!(report.classification, CONFIG_ERROR_CLASSIFICATION);
        assert!(report.fused_f16_allocation_attempted);
        assert!(report
            .error
            .as_deref()
            .unwrap()
            .contains("--allocate-fused-f16 requires --dtype f16"));
    }

    #[test]
    fn f16_scratch_flag_surfaces_deferred_private_boundary() {
        let dir = fixture_with_lm_head();

        let report = split_report_for_fused(&dir, false, true, Some(1));

        assert_eq!(report.classification, FUSED_F16_PLAN_STATUS_CLASSIFICATION);
        assert!(report.f16_scratch_allocation_attempted);
        assert!(!report.f16_scratch_allocation_succeeded);
        assert_eq!(report.f16_scratch_max_tokens, Some(1));
        assert!(!report
            .omitted_allocations
            .contains(&"f16_scratch".to_string()));

        for shard in &report.shards {
            assert_eq!(shard.f16_scratch_status, "deferred");
            assert!(shard
                .f16_scratch_deferred_reason
                .as_deref()
                .unwrap()
                .contains("F16LayerScratch"));
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

    #[test]
    fn split_allocation_smoke_reports_restricted_sinks_overrides() {
        let dir = unique_temp_model_dir("restricted_sinks_override");
        write_config(&dir, 24, true);
        write_safetensors(
            &dir.join("model-00000-of-00002.safetensors"),
            &[
                ("model.embed_tokens.weight", "F16", &[2, 4], 16),
                ("model.layers.0.self_attn.sinks", "F32", &[2], 8),
                ("model.norm.weight", "F32", &[4], 16),
            ],
        );
        write_safetensors(
            &dir.join("zzzz-sinks-override.safetensors"),
            &[("model.layers.0.self_attn.sinks", "F32", &[2], 8)],
        );

        let report = build_split_allocation_smoke_report(
            &dir,
            "split:0-11@0,12-23@1",
            0,
            None,
            DTypeMode::F16,
            None,
            true,
            false,
            false,
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            false,
            false,
            None,
            false,
        );

        assert_eq!(report.classification, SUCCESS_CLASSIFICATION);
        assert!(report.restricted_sinks_override_enabled);
        assert_eq!(
            report.header_merge_policy,
            "allow_restricted_sinks_override"
        );
        assert_eq!(report.overridden_tensor_count, 1);
        assert_eq!(
            report.overridden_tensor_names,
            vec!["model.layers.0.self_attn.sinks".to_string()]
        );
        assert!(shard(&report, 0)
            .required_tensor_names
            .contains(&"model.layers.0.self_attn.sinks".to_string()));
    }
}
