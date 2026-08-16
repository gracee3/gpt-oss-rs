use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use gpt_oss_gpu::event::CorrelatedTimeline;
use gpt_oss_model_runner::heterogeneous::{
    ExpertOwner, GptOssExpertPlacementManifestV1, HeterogeneousControlRuntime,
    RelayPinnedPoolStats, SelectedExpertInjectedFault, CONSERVATIVE_OWNER_EXPERT_BYTES,
};
use gpt_oss_model_runner::model_loader::gpt_oss_native::GptOssCheckpointView;
use gpt_oss_model_runner::model_loader::owner_selective::OwnerSelectiveConstructor;
use gpt_oss_model_runner::CpuGptOssConfig;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FaultMode {
    Recoverable,
    Unproven,
}

#[derive(Parser)]
struct Cli {
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
    #[arg(long, value_enum)]
    mode: FaultMode,
    #[arg(long, default_value_t = 0)]
    successful_remote_submissions_before_fault: usize,
    #[arg(long)]
    force_all_remote_fixture: bool,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Deserialize)]
struct RetainedControl {
    prompt_token_ids: Vec<u32>,
}

#[derive(Serialize)]
struct FaultEvidence {
    schema: &'static str,
    mode: &'static str,
    binary_sha256: String,
    placement_sha256: String,
    placement_mode: &'static str,
    placement_owner_counts_cpu_gpu0_gpu1: [usize; 3],
    tokens_committed_before_fault: usize,
    fault_generation: u64,
    successful_remote_submissions_before_fault: usize,
    first_error: String,
    first_drain_proven: bool,
    runtime_poisoned_after: bool,
    model_quarantined_after: bool,
    shell_poisoned_after: bool,
    pinned_checked_out_after: usize,
    pinned_quarantined_after: u64,
    clean_retry_succeeded: bool,
    retry_rejected_after_unproven: bool,
    final_pinned_checked_out: usize,
    final_pinned_quarantined: u64,
    passed: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.owner_cache.as_path() != Path::new("/home/emmy/workspace/gpt-oss-rs-het-cache") {
        bail!("H7 fault probe requires the authorized project-scoped owner cache");
    }
    let retained: RetainedControl = serde_json::from_slice(&std::fs::read(&cli.retained_trace)?)?;
    if retained.prompt_token_ids.is_empty() {
        bail!("retained control has no prompt token");
    }
    let config = CpuGptOssConfig::from_snapshot(&cli.model)?;
    let source_placement_bytes = std::fs::read(&cli.placement)?;
    let mut manifest: GptOssExpertPlacementManifestV1 =
        serde_json::from_slice(&source_placement_bytes)?;
    if cli.force_all_remote_fixture {
        let remote = manifest.remote_worker.clone();
        for assignment in &mut manifest.assignments {
            assignment.owner = ExpertOwner::RemoteGpu {
                device: remote.clone(),
            };
        }
        let expert_count = u32::try_from(manifest.assignments.len())?;
        manifest.budgets.max_cpu_experts = 0;
        manifest.budgets.max_layer_owner_experts = 0;
        manifest.budgets.max_remote_gpu_experts = expert_count;
        manifest.budgets.max_host_owner_bytes = 0;
        manifest.budgets.max_layer_owner_bytes = 0;
        manifest.budgets.max_remote_gpu_bytes = u64::from(expert_count)
            .checked_mul(CONSERVATIVE_OWNER_EXPERT_BYTES)
            .context("all-remote fixture byte budget overflow")?;
        manifest.policy_seed = 0x4837_464f_4c4c_4f57;
        manifest.placement_epoch = manifest
            .placement_epoch
            .checked_add(1)
            .context("all-remote fixture placement epoch overflow")?;
    }
    let placement_owner_counts_cpu_gpu0_gpu1 =
        manifest
            .assignments
            .iter()
            .fold([0_usize; 3], |mut counts, assignment| {
                counts[match &assignment.owner {
                    ExpertOwner::Cpu { .. } => 0,
                    ExpertOwner::LayerOwnerGpu { .. } => 1,
                    ExpertOwner::RemoteGpu { .. } => 2,
                }] += 1;
                counts
            });
    let placement_bytes = serde_json::to_vec(&manifest)?;
    let checkpoint = GptOssCheckpointView::open(&cli.native_model)?;
    let constructor = OwnerSelectiveConstructor::new(&cli.owner_cache);
    let mut model = constructor.construct(checkpoint, &manifest, |_| Ok(()))?;
    let mut runtime = HeterogeneousControlRuntime::new(&mut model, &config)?;
    runtime.inject_remote_expert_failure_after_for_test(
        &mut model,
        match cli.mode {
            FaultMode::Recoverable => SelectedExpertInjectedFault::SubmitAfterInputEnqueue,
            FaultMode::Unproven => {
                SelectedExpertInjectedFault::SubmitAfterInputEnqueueAndFallbackDrainFailure
            }
        },
        cli.successful_remote_submissions_before_fault,
    )?;
    let mut fault = None;
    for (index, &input) in retained.prompt_token_ids.iter().enumerate() {
        let generation = u64::try_from(index + 1)?;
        match runtime.execute_step(
            &mut model,
            &config,
            input,
            generation,
            None,
            &CorrelatedTimeline::new(),
        ) {
            Ok(_) => runtime.commit_prepared_token()?,
            Err(failure) => {
                fault = Some((index, generation, input, failure));
                break;
            }
        }
    }
    let (tokens_committed_before_fault, fault_generation, input, failure) = fault
        .context("armed remote selected-expert fault was not reached in the retained prompt")?;
    let first_error = failure.error.to_string();
    let first_drain_proven = failure.drain_proven;
    let runtime_poisoned_after = runtime.is_poisoned_for_test();
    let model_quarantined_after = model.execution_quarantined_for_test();
    let shell_poisoned_after = runtime.shell_is_poisoned_for_test();
    let after = runtime.pinned_pool_stats();
    let pinned_checked_out_after = checked_out(&after);
    let pinned_quarantined_after = quarantined(&after);

    let (clean_retry_succeeded, retry_rejected_after_unproven) = match cli.mode {
        FaultMode::Recoverable => {
            let retry = runtime.execute_step(
                &mut model,
                &config,
                input,
                fault_generation
                    .checked_add(1)
                    .context("fault retry generation overflow")?,
                None,
                &CorrelatedTimeline::new(),
            );
            let succeeded = retry.is_ok();
            if succeeded {
                runtime.discard_prepared_token(&mut model)?;
                runtime.drain(&mut model)?;
            }
            (succeeded, false)
        }
        FaultMode::Unproven => {
            let retry = runtime.execute_step(
                &mut model,
                &config,
                input,
                fault_generation
                    .checked_add(1)
                    .context("fault retry generation overflow")?,
                None,
                &CorrelatedTimeline::new(),
            );
            (false, retry.is_err_and(|failure| !failure.drain_proven))
        }
    };
    let final_stats = runtime.pinned_pool_stats();
    let final_pinned_checked_out = checked_out(&final_stats);
    let final_pinned_quarantined = quarantined(&final_stats);
    let passed = match cli.mode {
        FaultMode::Recoverable => {
            first_drain_proven
                && !runtime_poisoned_after
                && !model_quarantined_after
                && !shell_poisoned_after
                && pinned_checked_out_after == 0
                && pinned_quarantined_after == 0
                && clean_retry_succeeded
                && final_pinned_checked_out == 0
                && final_pinned_quarantined == 0
        }
        FaultMode::Unproven => {
            !first_drain_proven
                && runtime_poisoned_after
                && model_quarantined_after
                && shell_poisoned_after
                && pinned_checked_out_after == 5
                && retry_rejected_after_unproven
                && final_pinned_checked_out == 5
        }
    };
    let evidence = FaultEvidence {
        schema: "gpt-oss-rs.heterogeneous-control-fault-h7-followup/v2",
        mode: match cli.mode {
            FaultMode::Recoverable => "recoverable_remote_post_enqueue_and_clean_retry",
            FaultMode::Unproven => "unproven_remote_post_enqueue_quarantine",
        },
        binary_sha256: sha256_file(&std::env::current_exe()?)?,
        placement_sha256: hash_bytes(&placement_bytes),
        placement_mode: if cli.force_all_remote_fixture {
            "deterministic_all_remote_multi_result_fixture"
        } else {
            "supplied_manifest"
        },
        placement_owner_counts_cpu_gpu0_gpu1,
        tokens_committed_before_fault,
        fault_generation,
        successful_remote_submissions_before_fault: cli.successful_remote_submissions_before_fault,
        first_error,
        first_drain_proven,
        runtime_poisoned_after,
        model_quarantined_after,
        shell_poisoned_after,
        pinned_checked_out_after,
        pinned_quarantined_after,
        clean_retry_succeeded,
        retry_rejected_after_unproven,
        final_pinned_checked_out,
        final_pinned_quarantined,
        passed,
    };
    if let Some(path) = cli.output {
        let mut bytes = serde_json::to_vec_pretty(&evidence)?;
        bytes.push(b'\n');
        std::fs::write(path, bytes)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&evidence)?);
    }
    if !passed {
        bail!("H7 control fault gate failed");
    }
    Ok(())
}

fn checked_out(stats: &RelayPinnedPoolStats) -> usize {
    stats.source_activation.checked_out
        + stats.route_descriptors.checked_out
        + stats.remote_gpu_input.checked_out
        + stats.remote_gpu_result.checked_out
        + stats.cpu_result.checked_out
}

fn quarantined(stats: &RelayPinnedPoolStats) -> u64 {
    stats.source_activation.quarantined
        + stats.route_descriptors.quarantined
        + stats.remote_gpu_input.quarantined
        + stats.remote_gpu_result.quarantined
        + stats.cpu_result.quarantined
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(hash_bytes(&std::fs::read(path)?))
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
