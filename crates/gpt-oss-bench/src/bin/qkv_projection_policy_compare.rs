use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Serialize;

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
    policies_requested: Vec<String>,
    inputs: Inputs,
    expected_shapes: ExpectedShapes,
    comparisons: Vec<serde_json::Value>,
    latency: BTreeMap<String, serde_json::Value>,
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    validate_inputs(&cli)?;

    let status = Status {
        mode: "qkv_projection_policy_compare",
        classification: "qkv_projection_policy_compare_skeleton_ready",
        implemented: false,
        runtime_behavior_changed: false,
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
        expected_shapes: ExpectedShapes {
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
        next_bounded_step:
            "implement artifact shape/digest loading and baseline current CUDA policy comparison",
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
