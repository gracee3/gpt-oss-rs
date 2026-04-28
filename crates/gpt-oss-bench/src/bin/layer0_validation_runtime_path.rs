use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    name = "layer0_validation_runtime_path",
    about = "Emit the validation-only layer0 runtime-path skeleton status"
)]
struct Cli {
    /// Exact layer0 input/residual artifact for the validation case.
    #[arg(long)]
    layer0_input: PathBuf,

    /// Official layer0 final-token output artifact to compare against later.
    #[arg(long)]
    official_layer0_output: PathBuf,

    /// JSON status output path.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Serialize)]
struct Status {
    mode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    cuda_execution: bool,
    model_runner_routing_changed: bool,
    inputs: Inputs,
    target: Target,
    planned_stages: Vec<&'static str>,
    do_not_promote_boundaries: Vec<&'static str>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Serialize)]
struct Inputs {
    layer0_input: ArtifactPath,
    official_layer0_output: ArtifactPath,
}

#[derive(Debug, Serialize)]
struct ArtifactPath {
    path: String,
    exists: bool,
}

#[derive(Debug, Serialize)]
struct Target {
    exact_case: &'static str,
    output_boundary: &'static str,
    output_shape: [usize; 1],
    comparison: &'static str,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    validate_path(&cli.layer0_input, "layer0 input")?;
    validate_path(&cli.official_layer0_output, "official layer0 output")?;

    let status = Status {
        mode: "layer0_validation_runtime_path",
        classification: "layer0_validation_runtime_path_skeleton_ready",
        implemented: false,
        runtime_behavior_changed: false,
        validation_only: true,
        cuda_execution: false,
        model_runner_routing_changed: false,
        inputs: Inputs {
            layer0_input: artifact_path(&cli.layer0_input),
            official_layer0_output: artifact_path(&cli.official_layer0_output),
        },
        target: Target {
            exact_case: "developer-message-user-smoke",
            output_boundary: "layer0_final_token_hidden_state_after_mlp_residual_add",
            output_shape: [2880],
            comparison: "planned exact BF16-boundary comparison against official oracle",
        },
        planned_stages: vec![
            "skeleton_status_binary",
            "layer0_attention_only_validation",
            "layer0_mlp_validation",
            "full_layer0_final_token_validation",
            "layer_ladder_and_final_logits_later",
        ],
        do_not_promote_boundaries: vec![
            "no_production_runtime_routing",
            "no_default_model_runner_behavior_change",
            "no_cuda_kernel_replacement",
            "no_onednn_or_torch_runtime_dependency",
            "no_runtime_forward_proof_binary_import",
            "no_raw_artifacts_committed",
            "no_4097_work",
        ],
        next_bounded_step:
            "implement layer0 attention-only validation using runtime RoPE and validation-only projection policy",
    };

    if let Some(parent) = cli.output.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create output directory for {}",
                cli.output.display()
            )
        })?;
    }
    let payload = serde_json::to_vec_pretty(&status)?;
    fs::write(&cli.output, &payload)
        .with_context(|| format!("failed to write {}", cli.output.display()))?;
    println!("{}", String::from_utf8_lossy(&payload));
    Ok(())
}

fn validate_path(path: &Path, label: &str) -> Result<()> {
    anyhow::ensure!(
        path.exists(),
        "{} artifact does not exist: {}",
        label,
        path.display()
    );
    anyhow::ensure!(
        path.is_file(),
        "{} artifact is not a file: {}",
        label,
        path.display()
    );
    Ok(())
}

fn artifact_path(path: &Path) -> ArtifactPath {
    ArtifactPath {
        path: path.display().to_string(),
        exists: path.exists(),
    }
}
