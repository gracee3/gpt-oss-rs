use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Parser)]
#[command(
    name = "qkv_projection_policy_compare",
    about = "Validate Q/K/V projection policy comparison inputs and emit a skeleton status artifact"
)]
struct Cli {
    #[arg(long)]
    norm_input: PathBuf,

    #[arg(long)]
    q_weight: PathBuf,

    #[arg(long)]
    k_weight: PathBuf,

    #[arg(long)]
    v_weight: PathBuf,

    #[arg(long)]
    q_oracle: PathBuf,

    #[arg(long)]
    k_oracle: PathBuf,

    #[arg(long)]
    v_oracle: PathBuf,

    #[arg(long, default_value = "current")]
    policy: String,

    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Serialize)]
struct Status {
    mode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    projection_execution: bool,
    artifact_metadata_loaded: bool,
    policies_requested: Vec<String>,
    inputs: Inputs,
    shape_validation: ShapeValidation,
    artifacts: BTreeMap<&'static str, ArtifactStatus>,
    expected_shapes: ExpectedShapes,
    comparisons: Vec<serde_json::Value>,
    latency: BTreeMap<String, serde_json::Value>,
    warnings: Vec<String>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Serialize)]
struct Inputs {
    norm_input: String,
    q_weight: String,
    k_weight: String,
    v_weight: String,
    q_oracle: String,
    k_oracle: String,
    v_oracle: String,
}

#[derive(Debug, Serialize)]
struct ExpectedShapes {
    norm_input: [usize; 2],
    q: ProjectionShape,
    k: ProjectionShape,
    v: ProjectionShape,
}

#[derive(Debug, Serialize)]
struct ProjectionShape {
    activation: [usize; 2],
    weight: [usize; 2],
    output: [usize; 2],
}

#[derive(Debug, Serialize)]
struct ShapeValidation {
    all_available_shapes_match: bool,
    all_shapes_available: bool,
}

#[derive(Debug, Serialize)]
struct ArtifactStatus {
    path: String,
    json_loaded: bool,
    shape_available: bool,
    discovered_shape: Option<Vec<usize>>,
    expected_shape: [usize; 2],
    shape_matched: Option<bool>,
    dtype: Option<String>,
    digest: Option<String>,
    value_count: Option<usize>,
    warnings: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    validate_inputs(&cli)?;
    let artifacts = load_artifacts(&cli);
    let warnings = collect_warnings(&artifacts);
    let shape_validation = ShapeValidation {
        all_available_shapes_match: artifacts
            .values()
            .all(|artifact| artifact.shape_matched.unwrap_or(true)),
        all_shapes_available: artifacts.values().all(|artifact| artifact.shape_available),
    };
    let artifact_metadata_loaded = artifacts.values().all(|artifact| artifact.json_loaded);
    let classification = classify(&artifacts);

    let status = Status {
        mode: "qkv_projection_policy_compare",
        classification,
        implemented: false,
        runtime_behavior_changed: false,
        projection_execution: false,
        artifact_metadata_loaded,
        policies_requested: parse_policies(&cli.policy),
        inputs: Inputs {
            norm_input: path_string(&cli.norm_input),
            q_weight: path_string(&cli.q_weight),
            k_weight: path_string(&cli.k_weight),
            v_weight: path_string(&cli.v_weight),
            q_oracle: path_string(&cli.q_oracle),
            k_oracle: path_string(&cli.k_oracle),
            v_oracle: path_string(&cli.v_oracle),
        },
        shape_validation,
        artifacts,
        expected_shapes: ExpectedShapes {
            norm_input: [74, 2880],
            q: ProjectionShape {
                activation: [74, 2880],
                weight: [4096, 2880],
                output: [74, 4096],
            },
            k: ProjectionShape {
                activation: [74, 2880],
                weight: [512, 2880],
                output: [74, 512],
            },
            v: ProjectionShape {
                activation: [74, 2880],
                weight: [512, 2880],
                output: [74, 512],
            },
        },
        comparisons: Vec::new(),
        latency: BTreeMap::new(),
        warnings,
        next_bounded_step:
            "implement baseline current CUDA helper comparison against loaded artifacts",
    };

    write_status(&cli.output, &status)?;
    println!("{}", serde_json::to_string_pretty(&status)?);
    Ok(())
}

fn validate_inputs(cli: &Cli) -> Result<()> {
    let required = [
        ("norm_input", cli.norm_input.as_path()),
        ("q_weight", cli.q_weight.as_path()),
        ("k_weight", cli.k_weight.as_path()),
        ("v_weight", cli.v_weight.as_path()),
        ("q_oracle", cli.q_oracle.as_path()),
        ("k_oracle", cli.k_oracle.as_path()),
        ("v_oracle", cli.v_oracle.as_path()),
    ];

    let missing: Vec<_> = required
        .iter()
        .filter(|(_, path)| !path.exists())
        .map(|(name, path)| format!("{name}={}", path.display()))
        .collect();

    if !missing.is_empty() {
        bail!(
            "classification=qkv_projection_policy_compare_missing_inputs; missing required input path(s): {}",
            missing.join(", ")
        );
    }

    Ok(())
}

fn write_status(output: &Path, status: &Status) -> Result<()> {
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "classification=qkv_projection_policy_compare_output_write_failed; failed to create output directory {}",
                parent.display()
            )
        })?;
    }

    let bytes = serde_json::to_vec_pretty(status).with_context(|| {
        "classification=qkv_projection_policy_compare_output_write_failed; failed to serialize status"
    })?;
    std::fs::write(output, bytes).with_context(|| {
        format!(
            "classification=qkv_projection_policy_compare_output_write_failed; failed to write {}",
            output.display()
        )
    })?;
    Ok(())
}

fn load_artifacts(cli: &Cli) -> BTreeMap<&'static str, ArtifactStatus> {
    [
        ("norm_input", cli.norm_input.as_path(), [74, 2880]),
        ("q_weight", cli.q_weight.as_path(), [4096, 2880]),
        ("k_weight", cli.k_weight.as_path(), [512, 2880]),
        ("v_weight", cli.v_weight.as_path(), [512, 2880]),
        ("q_oracle", cli.q_oracle.as_path(), [74, 4096]),
        ("k_oracle", cli.k_oracle.as_path(), [74, 512]),
        ("v_oracle", cli.v_oracle.as_path(), [74, 512]),
    ]
    .into_iter()
    .map(|(name, path, expected_shape)| (name, load_artifact(path, expected_shape)))
    .collect()
}

fn load_artifact(path: &Path, expected_shape: [usize; 2]) -> ArtifactStatus {
    let mut warnings = Vec::new();
    let mut json_loaded = false;
    let mut shape = None;
    let mut dtype = None;
    let mut digest = None;
    let mut value_count = None;

    match std::fs::read_to_string(path) {
        Ok(text) if text.trim().is_empty() => {
            warnings.push("artifact is empty; metadata unavailable".to_string());
        }
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(value) => {
                json_loaded = true;
                let shape_candidates = collect_shapes(&value);
                shape = shape_candidates
                    .iter()
                    .find(|candidate| candidate.as_slice() == expected_shape)
                    .cloned()
                    .or_else(|| shape_candidates.into_iter().next());
                dtype = find_string_field(&value, &is_dtype_key);
                digest = find_string_field(&value, &is_digest_key);
                value_count = find_value_count(&value);
            }
            Err(err) => {
                warnings.push(format!("artifact is not valid JSON: {err}"));
            }
        },
        Err(err) => {
            warnings.push(format!("failed to read artifact: {err}"));
        }
    }

    if shape.is_none() {
        warnings.push("shape metadata not found".to_string());
    }
    if dtype.is_none() {
        warnings.push("dtype metadata not found".to_string());
    }
    if digest.is_none() {
        warnings.push("digest/checksum metadata not found".to_string());
    }

    let shape_matched = shape
        .as_ref()
        .map(|discovered| discovered.as_slice() == expected_shape);
    if shape_matched == Some(false) {
        warnings.push(format!(
            "shape mismatch: expected {:?}, discovered {:?}",
            expected_shape,
            shape.as_ref().unwrap()
        ));
    }

    ArtifactStatus {
        path: path_string(path),
        json_loaded,
        shape_available: shape.is_some(),
        discovered_shape: shape,
        expected_shape,
        shape_matched,
        dtype,
        digest,
        value_count,
        warnings,
    }
}

fn classify(artifacts: &BTreeMap<&'static str, ArtifactStatus>) -> &'static str {
    if artifacts
        .values()
        .any(|artifact| artifact.shape_matched == Some(false))
    {
        "qkv_projection_policy_compare_artifact_shape_mismatch"
    } else if !artifacts
        .values()
        .all(|artifact| artifact.json_loaded && artifact.shape_available)
    {
        "qkv_projection_policy_compare_artifact_metadata_incomplete"
    } else {
        "qkv_projection_policy_compare_artifact_metadata_loaded"
    }
}

fn collect_warnings(artifacts: &BTreeMap<&'static str, ArtifactStatus>) -> Vec<String> {
    artifacts
        .iter()
        .flat_map(|(name, artifact)| {
            artifact
                .warnings
                .iter()
                .map(move |warning| format!("{name}: {warning}"))
        })
        .collect()
}

fn collect_shapes(value: &Value) -> Vec<Vec<usize>> {
    let mut shapes = Vec::new();
    collect_shapes_inner(value, None, &mut shapes);
    shapes
}

fn collect_shapes_inner(value: &Value, key: Option<&str>, shapes: &mut Vec<Vec<usize>>) {
    if key.is_some_and(is_shape_key) {
        if let Some(shape) = as_usize_array(value) {
            shapes.push(shape);
        }
    }

    match value {
        Value::Object(map) => {
            for (child_key, child_value) in map {
                collect_shapes_inner(child_value, Some(child_key.as_str()), shapes);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_shapes_inner(item, None, shapes);
            }
        }
        _ => {}
    }
}

fn as_usize_array(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::Array(items) => {
            let shape: Option<Vec<_>> = items
                .iter()
                .map(|item| item.as_u64().and_then(|n| usize::try_from(n).ok()))
                .collect();
            shape.filter(|shape| !shape.is_empty())
        }
        _ => None,
    }
}

fn find_string_field(value: &Value, key_predicate: &dyn Fn(&str) -> bool) -> Option<String> {
    find_string_field_inner(value, None, key_predicate)
}

fn find_string_field_inner(
    value: &Value,
    key: Option<&str>,
    key_predicate: &dyn Fn(&str) -> bool,
) -> Option<String> {
    if key.is_some_and(key_predicate) {
        if let Some(text) = value.as_str() {
            return Some(text.to_string());
        }
    }

    match value {
        Value::Object(map) => map.iter().find_map(|(child_key, child_value)| {
            find_string_field_inner(child_value, Some(child_key.as_str()), key_predicate)
        }),
        Value::Array(items) => items
            .iter()
            .find_map(|item| find_string_field_inner(item, None, key_predicate)),
        _ => None,
    }
}

fn find_value_count(value: &Value) -> Option<usize> {
    find_value_count_inner(value, None)
}

fn find_value_count_inner(value: &Value, key: Option<&str>) -> Option<usize> {
    if key.is_some_and(is_count_key) {
        if let Some(count) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
            return Some(count);
        }
    }
    if key == Some("values") {
        if let Some(values) = value.as_array() {
            return Some(flat_value_count(values));
        }
    }

    match value {
        Value::Object(map) => map.iter().find_map(|(child_key, child_value)| {
            find_value_count_inner(child_value, Some(child_key.as_str()))
        }),
        Value::Array(items) => items
            .iter()
            .find_map(|item| find_value_count_inner(item, None)),
        _ => None,
    }
}

fn flat_value_count(values: &[Value]) -> usize {
    values
        .iter()
        .map(|value| match value {
            Value::Array(items) => flat_value_count(items),
            _ => 1,
        })
        .sum()
}

fn is_shape_key(key: &str) -> bool {
    matches!(key, "shape" | "tensor_shape" | "output_shape")
}

fn is_dtype_key(key: &str) -> bool {
    key == "dtype" || key == "serialization_dtype" || key.ends_with("_dtype")
}

fn is_digest_key(key: &str) -> bool {
    key == "digest" || key == "checksum" || key.ends_with("_digest")
}

fn is_count_key(key: &str) -> bool {
    matches!(key, "count" | "value_count" | "num_values")
}

fn parse_policies(policy: &str) -> Vec<String> {
    policy
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn path_string(path: &Path) -> String {
    path.display().to_string()
}
