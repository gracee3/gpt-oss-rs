use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use gpt_oss_model_runner::{
    CudaShardResourceStatus, DeviceId, DeviceMap, ShardedCudaResourcePlan,
    ShardedCudaResourceStatus, ShardedModelPlan,
};
use serde::{Deserialize, Serialize};

const SUCCESS_CLASSIFICATION: &str = "multi_gpu_layer_sharding_cuda_resource_smoke_complete";
const INVALID_DEVICE_MAP_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_cuda_resource_smoke_invalid_device_map";
const CONFIG_ERROR_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_cuda_resource_smoke_config_error";
#[allow(dead_code)]
const CUDA_UNAVAILABLE_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_cuda_resource_smoke_cuda_unavailable";
#[allow(dead_code)]
const RESOURCE_ERROR_CLASSIFICATION: &str =
    "multi_gpu_layer_sharding_cuda_resource_smoke_resource_error";

const OMITTED_ALLOCATIONS: &[&str] = &[
    "model_tensor_upload",
    "u8_payload_upload",
    "kv_cache",
    "rope_tables",
    "metadata_buffers",
    "f16_scratch",
    "fused_weights",
    "moe_gpu_weights",
    "transformer_layers",
    "gpu_model_runner",
    "activation_transfer",
];

#[derive(Debug, Parser)]
#[command(about = "Bench-only CUDA resource smoke for multi-GPU layer sharding")]
struct Cli {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    device_map: String,

    #[arg(long, default_value_t = 0)]
    selected_device: usize,

    #[arg(long)]
    kernel_dir: Option<PathBuf>,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CudaResourceSmokeReport {
    classification: String,
    model_path: String,
    device_map_spec: String,
    selected_device: usize,
    num_layers: Option<usize>,
    cuda_feature_enabled: bool,
    resource_construction_attempted: bool,
    resource_construction_succeeded: bool,
    kernel_dir: Option<String>,
    shard_count: usize,
    shards: Vec<CudaResourceSmokeShardReport>,
    omitted_allocations: Vec<String>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CudaResourceSmokeShardReport {
    device_id: usize,
    absolute_layers: Vec<usize>,
    owns_embeddings: bool,
    owns_final_head: bool,
    context_created: bool,
    stream_created: bool,
    cublas_created: bool,
    kernel_loader_created: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let report = build_cuda_resource_smoke_report(
        &cli.model,
        &cli.device_map,
        cli.selected_device,
        cli.kernel_dir.as_deref(),
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

fn build_cuda_resource_smoke_report(
    model_path: &Path,
    device_map_spec: &str,
    selected_device: usize,
    kernel_dir: Option<&Path>,
    construct_resources: bool,
) -> CudaResourceSmokeReport {
    let num_layers = match read_num_layers(model_path) {
        Ok(num_layers) => num_layers,
        Err(error) => {
            return error_report(
                CONFIG_ERROR_CLASSIFICATION,
                model_path,
                device_map_spec,
                selected_device,
                None,
                kernel_dir,
                false,
                Some(error.to_string()),
            );
        }
    };

    let device_map = match DeviceMap::parse(device_map_spec, num_layers, DeviceId(selected_device))
    {
        Ok(device_map) => device_map,
        Err(error) => {
            return error_report(
                INVALID_DEVICE_MAP_CLASSIFICATION,
                model_path,
                device_map_spec,
                selected_device,
                Some(num_layers),
                kernel_dir,
                false,
                Some(error.to_string()),
            );
        }
    };

    let plan = match ShardedModelPlan::from_device_map(device_map, num_layers) {
        Ok(plan) => plan,
        Err(error) => {
            return error_report(
                INVALID_DEVICE_MAP_CLASSIFICATION,
                model_path,
                device_map_spec,
                selected_device,
                Some(num_layers),
                kernel_dir,
                false,
                Some(error.to_string()),
            );
        }
    };

    if !construct_resources {
        let resource_plan = ShardedCudaResourcePlan::from_model_plan(&plan);
        let status = ShardedCudaResourceStatus::from_plan(&resource_plan);
        return render_report(
            SUCCESS_CLASSIFICATION,
            model_path,
            device_map_spec,
            selected_device,
            Some(num_layers),
            kernel_dir,
            false,
            true,
            &status,
            false,
            None,
        );
    }

    create_resource_status(&plan, kernel_dir)
        .map(|status| {
            render_report(
                SUCCESS_CLASSIFICATION,
                model_path,
                device_map_spec,
                selected_device,
                Some(num_layers),
                kernel_dir,
                true,
                true,
                &status,
                true,
                None,
            )
        })
        .unwrap_or_else(|(classification, error)| {
            let resource_plan = ShardedCudaResourcePlan::from_model_plan(&plan);
            let status = ShardedCudaResourceStatus::from_plan(&resource_plan);
            render_report(
                classification,
                model_path,
                device_map_spec,
                selected_device,
                Some(num_layers),
                kernel_dir,
                true,
                false,
                &status,
                false,
                Some(error),
            )
        })
}

fn read_num_layers(model_path: &Path) -> Result<usize> {
    let config_path = config_path_for_model(model_path);
    let text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
    json.get("num_hidden_layers")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .context("config.json missing numeric num_hidden_layers")
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
fn create_resource_status(
    plan: &ShardedModelPlan,
    kernel_dir: Option<&Path>,
) -> std::result::Result<ShardedCudaResourceStatus, (&'static str, String)> {
    use gpt_oss_model_runner::ShardedCudaResources;

    let resources = match kernel_dir {
        Some(kernel_dir) => ShardedCudaResources::create_for_plan_with_kernel_dir(plan, kernel_dir),
        None => ShardedCudaResources::create_for_plan(plan),
    };

    resources
        .map(|resources| resources.status())
        .map_err(|error| (RESOURCE_ERROR_CLASSIFICATION, error.to_string()))
}

#[cfg(not(feature = "cuda"))]
fn create_resource_status(
    _plan: &ShardedModelPlan,
    _kernel_dir: Option<&Path>,
) -> std::result::Result<ShardedCudaResourceStatus, (&'static str, String)> {
    Err((
        CUDA_UNAVAILABLE_CLASSIFICATION,
        "binary was built without the cuda feature".into(),
    ))
}

fn render_report(
    classification: &str,
    model_path: &Path,
    device_map_spec: &str,
    selected_device: usize,
    num_layers: Option<usize>,
    kernel_dir: Option<&Path>,
    resource_construction_attempted: bool,
    resource_construction_succeeded: bool,
    status: &ShardedCudaResourceStatus,
    resource_flags: bool,
    error: Option<String>,
) -> CudaResourceSmokeReport {
    CudaResourceSmokeReport {
        classification: classification.into(),
        model_path: model_path.display().to_string(),
        device_map_spec: device_map_spec.into(),
        selected_device,
        num_layers,
        cuda_feature_enabled: cfg!(feature = "cuda"),
        resource_construction_attempted,
        resource_construction_succeeded,
        kernel_dir: kernel_dir.map(|path| path.display().to_string()),
        shard_count: status.shards.len(),
        shards: status
            .shards
            .iter()
            .map(|shard| render_shard_report(shard, resource_flags))
            .collect(),
        omitted_allocations: OMITTED_ALLOCATIONS
            .iter()
            .map(|name| (*name).to_string())
            .collect(),
        error,
    }
}

fn render_shard_report(
    shard: &CudaShardResourceStatus,
    resource_flags: bool,
) -> CudaResourceSmokeShardReport {
    CudaResourceSmokeShardReport {
        device_id: shard.device_id.0,
        absolute_layers: shard.absolute_layers.clone(),
        owns_embeddings: shard.owns_embeddings,
        owns_final_head: shard.owns_final_head,
        context_created: resource_flags,
        stream_created: resource_flags,
        cublas_created: resource_flags,
        kernel_loader_created: resource_flags,
    }
}

fn error_report(
    classification: &str,
    model_path: &Path,
    device_map_spec: &str,
    selected_device: usize,
    num_layers: Option<usize>,
    kernel_dir: Option<&Path>,
    resource_construction_attempted: bool,
    error: Option<String>,
) -> CudaResourceSmokeReport {
    render_report(
        classification,
        model_path,
        device_map_spec,
        selected_device,
        num_layers,
        kernel_dir,
        resource_construction_attempted,
        false,
        &ShardedCudaResourceStatus { shards: Vec::new() },
        false,
        error,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_temp_model_dir(test_name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "multi_gpu_layer_sharding_cuda_resource_smoke_{test_name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_config(dir: &Path, num_layers: usize) {
        let config = serde_json::json!({
            "num_hidden_layers": num_layers,
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
    }

    fn plan_only_report(dir: &Path, device_map: &str) -> CudaResourceSmokeReport {
        build_cuda_resource_smoke_report(dir, device_map, 0, None, false)
    }

    fn shard(report: &CudaResourceSmokeReport, device_id: usize) -> &CudaResourceSmokeShardReport {
        report
            .shards
            .iter()
            .find(|shard| shard.device_id == device_id)
            .unwrap()
    }

    #[test]
    fn cuda_resource_smoke_status_builds_single_map_plan() {
        let dir = unique_temp_model_dir("single");
        write_config(&dir, 24);

        let report = plan_only_report(&dir, "single");

        assert_eq!(report.classification, SUCCESS_CLASSIFICATION);
        assert_eq!(report.num_layers, Some(24));
        assert_eq!(report.shard_count, 1);
        assert_eq!(report.shards[0].device_id, 0);
        assert_eq!(
            report.shards[0].absolute_layers,
            (0..24).collect::<Vec<_>>()
        );
        assert!(report.shards[0].owns_embeddings);
        assert!(report.shards[0].owns_final_head);
        assert!(!report.resource_construction_attempted);
    }

    #[test]
    fn cuda_resource_smoke_status_builds_split_map_plan() {
        let dir = unique_temp_model_dir("split");
        write_config(&dir, 24);

        let report = plan_only_report(&dir, "split:0-11@0,12-23@1");

        assert_eq!(report.classification, SUCCESS_CLASSIFICATION);
        assert_eq!(report.shard_count, 2);
        assert_eq!(
            shard(&report, 0).absolute_layers,
            (0..12).collect::<Vec<_>>()
        );
        assert!(shard(&report, 0).owns_embeddings);
        assert!(!shard(&report, 0).owns_final_head);
        assert_eq!(
            shard(&report, 1).absolute_layers,
            (12..24).collect::<Vec<_>>()
        );
        assert!(!shard(&report, 1).owns_embeddings);
        assert!(shard(&report, 1).owns_final_head);
    }

    #[test]
    fn cuda_resource_smoke_status_json_contains_success_classification() {
        let dir = unique_temp_model_dir("json");
        write_config(&dir, 24);

        let report = plan_only_report(&dir, "split:0-11@0,12-23@1");
        let json = serde_json::to_string(&report).unwrap();

        assert!(json.contains(SUCCESS_CLASSIFICATION));
    }

    #[test]
    fn omitted_allocations_cover_non_executing_boundary() {
        let dir = unique_temp_model_dir("omitted");
        write_config(&dir, 24);

        let report = plan_only_report(&dir, "single");

        for expected in [
            "model_tensor_upload",
            "u8_payload_upload",
            "kv_cache",
            "rope_tables",
            "metadata_buffers",
            "f16_scratch",
            "fused_weights",
            "moe_gpu_weights",
            "transformer_layers",
            "gpu_model_runner",
            "activation_transfer",
        ] {
            assert!(report.omitted_allocations.contains(&expected.to_string()));
        }
    }

    #[test]
    fn invalid_device_map_reports_invalid_device_map_classification() {
        let dir = unique_temp_model_dir("invalid_device_map");
        write_config(&dir, 24);

        let report = plan_only_report(&dir, "split:0-10@0,12-23@1");

        assert_eq!(report.classification, INVALID_DEVICE_MAP_CLASSIFICATION);
        assert_eq!(report.num_layers, Some(24));
        assert!(report.error.unwrap().contains("missing layer coverage"));
    }

    #[test]
    fn no_cuda_build_reports_cuda_unavailable_when_construction_is_requested() {
        let dir = unique_temp_model_dir("no_cuda");
        write_config(&dir, 24);

        let report = build_cuda_resource_smoke_report(&dir, "split:0-11@0,12-23@1", 0, None, true);

        if cfg!(feature = "cuda") {
            assert_ne!(report.classification, CUDA_UNAVAILABLE_CLASSIFICATION);
        } else {
            assert_eq!(report.classification, CUDA_UNAVAILABLE_CLASSIFICATION);
            assert!(report.resource_construction_attempted);
            assert!(!report.resource_construction_succeeded);
        }
    }
}
