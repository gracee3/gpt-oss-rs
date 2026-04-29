use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use gpt_oss_model_runner::{
    DeviceId, DeviceMap, LateAllocationKind, SafetensorHeaderManifest, ShardAllocationReport,
    ShardedModelPlan, SplitAllocationReport, UploadManifestOptions,
};
use serde::{Deserialize, Serialize};

const SUCCESS_CLASSIFICATION: &str = "multi_gpu_layer_sharding_dry_run_report_complete";

#[derive(Debug, Parser)]
#[command(about = "CUDA-free multi-GPU layer-sharding dry-run report")]
struct Cli {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    device_map: String,

    #[arg(long, default_value_t = 0)]
    selected_device: usize,

    #[arg(long)]
    tie_word_embeddings: Option<bool>,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DryRunReport {
    classification: String,
    model_path: String,
    device_map_spec: String,
    selected_device: usize,
    num_layers: usize,
    tensor_count: usize,
    tensor_byte_count: usize,
    unassigned_tensor_count: usize,
    invalid_tensor_count: usize,
    has_lm_head_weight: bool,
    tie_word_embeddings: bool,
    shards: Vec<DryRunShardReport>,
    unassigned_tensor_names: Vec<String>,
    invalid_tensor_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DryRunShardReport {
    device_id: usize,
    absolute_layers: Vec<usize>,
    owns_embeddings: bool,
    owns_final_head: bool,
    required_tensor_count: usize,
    optional_tensor_count: usize,
    host_u8_tensor_count: usize,
    late_allocation_count: usize,
    kv_cache_entry_count: usize,
    kv_cache_layers: Vec<usize>,
    required_tensor_names: Vec<String>,
    host_u8_tensor_names: Vec<String>,
    late_allocations: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelPlanningConfig {
    num_layers: usize,
    tie_word_embeddings: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let report = build_dry_run_report(
        &cli.model,
        &cli.device_map,
        cli.selected_device,
        cli.tie_word_embeddings,
    )?;
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

fn build_dry_run_report(
    model_path: &Path,
    device_map_spec: &str,
    selected_device: usize,
    tie_word_embeddings_override: Option<bool>,
) -> Result<DryRunReport> {
    let config = read_model_planning_config(model_path, tie_word_embeddings_override)?;
    let header_manifest = SafetensorHeaderManifest::discover(model_path).with_context(|| {
        format!(
            "multi_gpu_layer_sharding_dry_run_invalid_headers: failed to discover safetensors headers under {}",
            model_path.display()
        )
    })?;
    let device_map = DeviceMap::parse(
        device_map_spec,
        config.num_layers,
        DeviceId(selected_device),
    )
    .with_context(|| "multi_gpu_layer_sharding_dry_run_invalid_device_map")?;
    let plan = ShardedModelPlan::from_device_map(device_map, config.num_layers)
        .with_context(|| "multi_gpu_layer_sharding_dry_run_invalid_device_map")?;
    let upload_manifest = plan
        .upload_manifest_for_tensor_names(
            header_manifest.tensor_names(),
            UploadManifestOptions {
                tie_word_embeddings: config.tie_word_embeddings,
            },
        )
        .with_context(|| "multi_gpu_layer_sharding_dry_run_invalid_tensor_manifest")?;
    let kv_plan = plan.kv_cache_plan();
    let allocation_report = plan
        .split_allocation_report(&upload_manifest, &kv_plan)
        .with_context(|| "multi_gpu_layer_sharding_dry_run_invalid_tensor_manifest")?;

    Ok(render_dry_run_report(
        model_path,
        device_map_spec,
        selected_device,
        config,
        &header_manifest,
        allocation_report,
    ))
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

fn render_dry_run_report(
    model_path: &Path,
    device_map_spec: &str,
    selected_device: usize,
    config: ModelPlanningConfig,
    header_manifest: &SafetensorHeaderManifest,
    allocation_report: SplitAllocationReport,
) -> DryRunReport {
    DryRunReport {
        classification: SUCCESS_CLASSIFICATION.into(),
        model_path: model_path.display().to_string(),
        device_map_spec: device_map_spec.into(),
        selected_device,
        num_layers: config.num_layers,
        tensor_count: header_manifest.tensors.len(),
        tensor_byte_count: header_manifest
            .tensors
            .iter()
            .map(|tensor| tensor.byte_size)
            .sum(),
        unassigned_tensor_count: allocation_report.unassigned_tensor_count,
        invalid_tensor_count: allocation_report.invalid_tensor_count,
        has_lm_head_weight: header_manifest.has_lm_head_weight(),
        tie_word_embeddings: config.tie_word_embeddings,
        shards: allocation_report
            .shards
            .iter()
            .map(render_shard_report)
            .collect(),
        unassigned_tensor_names: allocation_report.unassigned_tensor_names,
        invalid_tensor_names: allocation_report.invalid_tensor_names,
    }
}

fn render_shard_report(shard: &ShardAllocationReport) -> DryRunShardReport {
    DryRunShardReport {
        device_id: shard.device_id.0,
        absolute_layers: shard.absolute_layers.clone(),
        owns_embeddings: shard.owns_embeddings,
        owns_final_head: shard.owns_final_head,
        required_tensor_count: shard.required_tensor_count,
        optional_tensor_count: shard.optional_tensor_count,
        host_u8_tensor_count: shard.host_u8_tensor_count,
        late_allocation_count: shard.late_allocation_count,
        kv_cache_entry_count: shard.kv_cache_entry_count,
        kv_cache_layers: shard.kv_cache_layers.clone(),
        required_tensor_names: shard.required_tensor_names.clone(),
        host_u8_tensor_names: shard.host_u8_tensor_names.clone(),
        late_allocations: shard
            .late_allocations
            .iter()
            .map(format_late_allocation)
            .collect(),
    }
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

    fn unique_temp_model_dir(test_name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "multi_gpu_layer_sharding_dry_run_{test_name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
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
                    "model.layers.23.mlp.experts.down_proj_scales",
                    "U8",
                    &[4],
                    4,
                ),
                ("model.norm.weight", "F32", &[4], 16),
                ("lm_head.weight", "F16", &[2, 4], 16),
                ("model.extra_unknown.weight", "F32", &[1], 4),
            ],
        );
        dir
    }

    fn split_report_for(dir: &Path, tie_word_embeddings: Option<bool>) -> DryRunReport {
        build_dry_run_report(dir, "split:0-11@0,12-23@1", 0, tie_word_embeddings).unwrap()
    }

    fn shard(report: &DryRunReport, device_id: usize) -> &DryRunShardReport {
        report
            .shards
            .iter()
            .find(|shard| shard.device_id == device_id)
            .unwrap()
    }

    #[test]
    fn dry_run_report_builds_from_header_fixture_and_split_map() {
        let dir = fixture_with_lm_head();

        let report = split_report_for(&dir, None);

        assert_eq!(report.classification, SUCCESS_CLASSIFICATION);
        assert_eq!(report.num_layers, 24);
        assert_eq!(report.tensor_count, 8);
        assert!(report.has_lm_head_weight);
        assert!(report.tie_word_embeddings);
        assert_eq!(report.shards.len(), 2);
    }

    #[test]
    fn dry_run_report_includes_split_layer_ownership() {
        let dir = fixture_with_lm_head();

        let report = split_report_for(&dir, None);

        let gpu0 = shard(&report, 0);
        assert!(gpu0.owns_embeddings);
        assert!(!gpu0.owns_final_head);
        assert_eq!(gpu0.absolute_layers, (0..=11).collect::<Vec<_>>());
        assert_eq!(gpu0.kv_cache_layers, (0..=11).collect::<Vec<_>>());

        let gpu1 = shard(&report, 1);
        assert!(!gpu1.owns_embeddings);
        assert!(gpu1.owns_final_head);
        assert_eq!(gpu1.absolute_layers, (12..=23).collect::<Vec<_>>());
        assert_eq!(gpu1.kv_cache_layers, (12..=23).collect::<Vec<_>>());
    }

    #[test]
    fn dry_run_report_routes_header_tensors_to_expected_shards() {
        let dir = fixture_with_lm_head();

        let report = split_report_for(&dir, None);

        let gpu0 = shard(&report, 0);
        assert!(gpu0
            .required_tensor_names
            .contains(&"model.embed_tokens.weight".to_string()));
        assert!(gpu0
            .required_tensor_names
            .contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(gpu0
            .required_tensor_names
            .contains(&"model.layers.11.mlp.down_proj.weight".to_string()));

        let gpu1 = shard(&report, 1);
        assert!(gpu1
            .required_tensor_names
            .contains(&"model.layers.12.self_attn.k_proj.weight".to_string()));
        assert!(gpu1
            .required_tensor_names
            .contains(&"model.norm.weight".to_string()));
        assert!(gpu1
            .required_tensor_names
            .contains(&"lm_head.weight".to_string()));
        assert!(gpu1
            .host_u8_tensor_names
            .contains(&"model.layers.23.mlp.experts.down_proj_scales".to_string()));
    }

    #[test]
    fn dry_run_report_surfaces_unknown_names_globally() {
        let dir = fixture_with_lm_head();

        let report = split_report_for(&dir, None);

        assert_eq!(
            report.unassigned_tensor_names,
            vec!["model.extra_unknown.weight".to_string()]
        );
        assert!(!report.shards.iter().any(|shard| shard
            .required_tensor_names
            .contains(&"model.extra_unknown.weight".to_string())));
    }

    #[test]
    fn dry_run_report_marks_tied_lm_head_fallback_when_lm_head_is_absent() {
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

        assert!(!report.has_lm_head_weight);
        assert!(shard(&report, 1)
            .late_allocations
            .contains(&"tied_lm_head_fallback".to_string()));
    }

    #[test]
    fn dry_run_report_omits_tied_lm_head_fallback_when_lm_head_exists() {
        let dir = fixture_with_lm_head();

        let report = split_report_for(&dir, None);

        assert!(!shard(&report, 1)
            .late_allocations
            .contains(&"tied_lm_head_fallback".to_string()));
    }

    #[test]
    fn dry_run_report_rejects_duplicate_tensor_names_before_report_generation() {
        let dir = unique_temp_model_dir("duplicate");
        write_config(&dir, 24, true);
        write_safetensors(
            &dir.join("a.safetensors"),
            &[("model.norm.weight", "F32", &[4], 16)],
        );
        write_safetensors(
            &dir.join("b.safetensors"),
            &[("model.norm.weight", "F32", &[4], 16)],
        );

        let err = build_dry_run_report(&dir, "split:0-11@0,12-23@1", 0, None).unwrap_err();

        assert!(
            err.to_string()
                .contains("multi_gpu_layer_sharding_dry_run_invalid_headers"),
            "got: {err:?}"
        );
    }

    #[test]
    fn explicit_tie_word_embeddings_override_controls_fallback() {
        let dir = unique_temp_model_dir("override");
        write_config(&dir, 24, true);
        write_safetensors(
            &dir.join("model.safetensors"),
            &[("model.embed_tokens.weight", "F16", &[2, 4], 16)],
        );

        let report = split_report_for(&dir, Some(false));

        assert!(!report.tie_word_embeddings);
        assert!(!shard(&report, 1)
            .late_allocations
            .contains(&"tied_lm_head_fallback".to_string()));
    }
}
