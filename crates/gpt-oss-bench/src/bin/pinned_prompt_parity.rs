use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use gpt_oss_bench::pinned_prompt_official_launcher::{
    preflight_official_python, run_distributed_helper, selected_visible_devices_from_env,
};
use gpt_oss_bench::pinned_prompt_parity::{
    compare_artifacts, compare_intermediate_artifacts, default_artifact_schema_version,
    default_intermediate_artifact_schema_version,
    default_official_intermediate_capture_input_schema_version, default_prompt_renderer,
    import_runtime_forward_intermediate_capture, validate_artifact, validate_intermediate_artifact,
    validate_intermediate_boundary_layer_idx, validate_manifest, CaptureSource,
    PinnedPromptArtifact, PinnedPromptCaptureProvenance, PinnedPromptCase,
    PinnedPromptCaseArtifact, PinnedPromptIntermediateArtifact, PinnedPromptIntermediateBoundary,
    PinnedPromptIntermediateCaseArtifact, PinnedPromptManifest, PinnedPromptOfficialCaptureInput,
    PinnedPromptOfficialCaptureOutput, PinnedPromptOfficialIntermediateCaptureInput,
    PinnedPromptOfficialIntermediateCaptureOutput, PinnedTopLogit,
    RuntimeForwardIntermediateCapture,
};
use gpt_oss_core::prelude::{BlockId, RequestId, SamplingParams, SequenceId, TokenId};
use gpt_oss_core::types::Dtype;
use gpt_oss_engine::worker::gpu_worker::GpuWorker;
use gpt_oss_engine::{RuntimeMode, SequenceData, SequenceGroupMetadata, WorkerConfig};
use gpt_oss_tokenizer::{HarmonyProtocol, Tokenizer};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(about = "Pinned prompt parity scaffold harness")]
struct Cli {
    #[command(subcommand)]
    command: Command,

    #[arg(long, default_value = "info", global = true)]
    log_level: String,
}

#[derive(Debug, Subcommand)]
enum Command {
    Capture(CaptureArgs),
    CaptureOfficial(CaptureOfficialArgs),
    Compare(CompareArgs),
    CaptureOfficialIntermediate(CaptureOfficialIntermediateArgs),
    CompareIntermediate(CompareIntermediateArgs),
    ImportRuntimeForwardIntermediate(ImportRuntimeForwardIntermediateArgs),
}

#[derive(Debug, Parser)]
struct CaptureArgs {
    #[arg(long)]
    manifest: PathBuf,

    #[arg(long)]
    model: String,

    #[arg(long)]
    output: PathBuf,

    #[arg(long, value_enum)]
    capture_source: Option<CaptureSource>,

    #[arg(long, default_value_t = 128)]
    max_model_len: usize,

    #[arg(long, default_value_t = 0.75)]
    gpu_memory_utilization: f32,
}

#[derive(Debug, Parser)]
struct CompareArgs {
    #[arg(long)]
    left: PathBuf,

    #[arg(long)]
    right: PathBuf,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Parser)]
struct CaptureOfficialArgs {
    #[arg(long)]
    manifest: PathBuf,

    #[arg(long)]
    model: String,

    #[arg(long)]
    official_model: String,

    #[arg(long)]
    output: PathBuf,

    #[arg(long)]
    case_id: Option<String>,

    #[arg(long)]
    python: Option<PathBuf>,

    #[arg(long)]
    official_checkout: Option<PathBuf>,
}

#[derive(Debug, Parser)]
struct CaptureOfficialIntermediateArgs {
    #[arg(long)]
    manifest: PathBuf,

    #[arg(long)]
    model: String,

    #[arg(long)]
    official_model: String,

    #[arg(long)]
    output: PathBuf,

    #[arg(long)]
    case_id: Option<String>,

    #[arg(long)]
    python: Option<PathBuf>,

    #[arg(long)]
    official_checkout: Option<PathBuf>,

    #[arg(long, value_enum)]
    boundary: PinnedPromptIntermediateBoundary,

    #[arg(long)]
    layer_idx: Option<usize>,
}

#[derive(Debug, Parser)]
struct CompareIntermediateArgs {
    #[arg(long)]
    left: PathBuf,

    #[arg(long)]
    right: PathBuf,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Parser)]
struct ImportRuntimeForwardIntermediateArgs {
    #[arg(long)]
    manifest: PathBuf,

    #[arg(long)]
    case_id: String,

    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    output: PathBuf,

    #[arg(long, default_value_t = 128)]
    max_model_len: usize,

    #[arg(long, default_value_t = 0.75)]
    gpu_memory_utilization: f32,
}

fn init_tracing(log_level: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);

    match cli.command {
        Command::Capture(args) => capture(args),
        Command::CaptureOfficial(args) => capture_official(args),
        Command::Compare(args) => compare(args),
        Command::CaptureOfficialIntermediate(args) => capture_official_intermediate(args),
        Command::CompareIntermediate(args) => compare_intermediate(args),
        Command::ImportRuntimeForwardIntermediate(args) => {
            import_runtime_forward_intermediate(args)
        }
    }
}

fn capture(args: CaptureArgs) -> Result<()> {
    let manifest = read_json::<PinnedPromptManifest>(&args.manifest)?;
    validate_manifest(&manifest)?;
    let capture_source = args
        .capture_source
        .unwrap_or(manifest.default_capture_source);
    if capture_source == CaptureSource::OfficialTorch {
        bail!("capture --capture-source official_torch is unsupported; use capture-official");
    }

    let tokenizer = Tokenizer::from_pretrained(&args.model)?;
    let protocol = HarmonyProtocol::gpt_oss()?;
    let model_path = Path::new(&args.model);
    let mut worker = build_worker(model_path, args.max_model_len, args.gpu_memory_utilization)?;

    let mut cases = Vec::with_capacity(manifest.cases.len());
    for case in &manifest.cases {
        let rendered =
            protocol.render_prompt(&case.messages, case.instructions.as_deref(), &case.tools)?;
        let metadata = build_single_sequence_metadata(&rendered.token_ids);
        let logits = match capture_source {
            CaptureSource::Runner => worker.debug_runner_logits(&metadata)?,
            CaptureSource::Worker => worker.debug_logits(&metadata)?,
            CaptureSource::OfficialTorch => unreachable!("validated above"),
        };
        let last_logits = last_token_logits(&logits, tokenizer.vocab_size())?;
        cases.push(build_case_artifact(
            case.id.clone(),
            rendered.token_ids,
            last_logits,
            case.effective_top_k(manifest.default_top_k),
            &tokenizer,
        )?);
    }

    let artifact = PinnedPromptArtifact {
        schema_version: default_artifact_schema_version(),
        suite_id: manifest.suite_id,
        provenance: PinnedPromptCaptureProvenance {
            model: args.model,
            capture_source,
            reference_kind: manifest.reference_kind,
            authority_level: manifest.authority_level,
            prompt_renderer: default_prompt_renderer(),
            visible_devices: std::env::var("CUDA_VISIBLE_DEVICES")
                .unwrap_or_else(|_| "<unset>".into()),
            max_model_len: args.max_model_len,
            gpu_memory_utilization: args.gpu_memory_utilization,
        },
        cases,
    };
    write_json(&args.output, &artifact)?;
    println!("{}", serde_json::to_string_pretty(&artifact)?);
    Ok(())
}

fn capture_official(args: CaptureOfficialArgs) -> Result<()> {
    let manifest = read_json::<PinnedPromptManifest>(&args.manifest)?;
    validate_manifest(&manifest)?;
    let case = select_case(&manifest, args.case_id.as_deref(), "capture-official")?;

    let tokenizer = Tokenizer::from_pretrained(&args.model)?;
    let protocol = HarmonyProtocol::gpt_oss()?;
    let rendered =
        protocol.render_prompt(&case.messages, case.instructions.as_deref(), &case.tools)?;
    let effective_top_k = case.effective_top_k(manifest.default_top_k);
    let official_checkout = resolve_official_checkout(args.official_checkout.as_deref())?;
    let python = resolve_python_interpreter(args.python.as_deref(), &official_checkout)?;
    let visible_devices = selected_visible_devices_from_env();
    preflight_official_python(&python, &official_checkout, &visible_devices)?;

    let helper_input = PinnedPromptOfficialCaptureInput {
        schema_version:
            gpt_oss_bench::pinned_prompt_parity::default_official_capture_input_schema_version(),
        suite_id: manifest.suite_id.clone(),
        case_id: case.id.clone(),
        prompt_renderer: default_prompt_renderer(),
        model: args.model.clone(),
        official_model: args.official_model.clone(),
        input_token_ids: rendered.token_ids.clone(),
        top_k: effective_top_k,
    };

    let helper_output = run_official_capture_helper(
        &python,
        &official_checkout,
        &visible_devices,
        &helper_input,
        &args.output,
    )?;
    validate_official_capture_output(&helper_output, &helper_input)?;

    let cases = vec![PinnedPromptCaseArtifact {
        id: helper_output.case_id,
        input_token_ids: helper_output.input_token_ids,
        argmax_token_id: helper_output.argmax_token_id,
        argmax_token_text: tokenizer
            .decode(&[helper_output.argmax_token_id])
            .unwrap_or_default(),
        final_position_top_k: helper_output
            .final_position_top_k
            .into_iter()
            .map(|entry| PinnedTopLogit {
                token_id: entry.token_id,
                token_text: tokenizer.decode(&[entry.token_id]).unwrap_or_default(),
                logit: entry.logit,
            })
            .collect(),
    }];

    let artifact = PinnedPromptArtifact {
        schema_version: default_artifact_schema_version(),
        suite_id: manifest.suite_id,
        provenance: PinnedPromptCaptureProvenance {
            model: args.official_model,
            capture_source: CaptureSource::OfficialTorch,
            reference_kind: gpt_oss_bench::pinned_prompt_parity::ReferenceKind::OfficialReference,
            authority_level: gpt_oss_bench::pinned_prompt_parity::AuthorityLevel::Authoritative,
            prompt_renderer: default_prompt_renderer(),
            visible_devices,
            max_model_len: 0,
            gpu_memory_utilization: 0.0,
        },
        cases,
    };
    validate_artifact(&artifact)?;
    write_json(&args.output, &artifact)?;
    println!("{}", serde_json::to_string_pretty(&artifact)?);
    Ok(())
}

fn capture_official_intermediate(args: CaptureOfficialIntermediateArgs) -> Result<()> {
    let manifest = read_json::<PinnedPromptManifest>(&args.manifest)?;
    validate_manifest(&manifest)?;
    validate_intermediate_boundary_layer_idx(
        args.boundary,
        args.layer_idx,
        "official intermediate capture input",
    )?;
    let case = select_case(
        &manifest,
        args.case_id.as_deref(),
        "capture-official-intermediate",
    )?;

    let protocol = HarmonyProtocol::gpt_oss()?;
    let rendered =
        protocol.render_prompt(&case.messages, case.instructions.as_deref(), &case.tools)?;
    let official_checkout = resolve_official_checkout(args.official_checkout.as_deref())?;
    let python = resolve_python_interpreter(args.python.as_deref(), &official_checkout)?;
    let visible_devices = selected_visible_devices_from_env();
    preflight_official_python(&python, &official_checkout, &visible_devices)?;

    let helper_input = PinnedPromptOfficialIntermediateCaptureInput {
        schema_version: default_official_intermediate_capture_input_schema_version(),
        suite_id: manifest.suite_id.clone(),
        case_id: case.id.clone(),
        prompt_renderer: default_prompt_renderer(),
        model: args.model.clone(),
        official_model: args.official_model.clone(),
        input_token_ids: rendered.token_ids.clone(),
        boundary: args.boundary,
        layer_idx: args.layer_idx,
    };

    let helper_output = run_official_intermediate_capture_helper(
        &python,
        &official_checkout,
        &visible_devices,
        &helper_input,
        &args.output,
    )?;
    validate_official_intermediate_capture_output(&helper_output, &helper_input)?;

    let artifact = PinnedPromptIntermediateArtifact {
        schema_version: default_intermediate_artifact_schema_version(),
        suite_id: manifest.suite_id,
        boundary: helper_output.boundary,
        layer_idx: helper_output.layer_idx,
        provenance: PinnedPromptCaptureProvenance {
            model: args.official_model,
            capture_source: CaptureSource::OfficialTorch,
            reference_kind: gpt_oss_bench::pinned_prompt_parity::ReferenceKind::OfficialReference,
            authority_level: gpt_oss_bench::pinned_prompt_parity::AuthorityLevel::Authoritative,
            prompt_renderer: default_prompt_renderer(),
            visible_devices,
            max_model_len: 0,
            gpu_memory_utilization: 0.0,
        },
        cases: vec![PinnedPromptIntermediateCaseArtifact {
            id: helper_output.case_id,
            input_token_ids: helper_output.input_token_ids,
            hidden_size: helper_output.hidden_size,
            final_token_hidden_f32: helper_output.final_token_hidden_f32,
        }],
    };
    validate_intermediate_artifact(&artifact)?;
    write_json(&args.output, &artifact)?;
    println!("{}", serde_json::to_string_pretty(&artifact)?);
    Ok(())
}

fn import_runtime_forward_intermediate(args: ImportRuntimeForwardIntermediateArgs) -> Result<()> {
    let manifest = read_json::<PinnedPromptManifest>(&args.manifest)?;
    validate_manifest(&manifest)?;
    let input = read_json::<RuntimeForwardIntermediateCapture>(&args.input)?;
    let artifact = import_runtime_forward_intermediate_capture(
        &manifest,
        &args.case_id,
        &input,
        args.max_model_len,
        args.gpu_memory_utilization,
    )?;
    write_json(&args.output, &artifact)?;
    println!("{}", serde_json::to_string_pretty(&artifact)?);
    Ok(())
}

fn compare(args: CompareArgs) -> Result<()> {
    let left = read_json::<PinnedPromptArtifact>(&args.left)?;
    let right = read_json::<PinnedPromptArtifact>(&args.right)?;
    validate_artifact(&left)?;
    validate_artifact(&right)?;
    let report = compare_artifacts(&left, &right);
    if let Some(path) = args.output.as_ref() {
        write_json(path, &report)?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn compare_intermediate(args: CompareIntermediateArgs) -> Result<()> {
    let left = read_json::<PinnedPromptIntermediateArtifact>(&args.left)?;
    let right = read_json::<PinnedPromptIntermediateArtifact>(&args.right)?;
    validate_intermediate_artifact(&left)?;
    validate_intermediate_artifact(&right)?;
    let report = compare_intermediate_artifacts(&left, &right);
    if let Some(path) = args.output.as_ref() {
        write_json(path, &report)?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?,
    )
    .with_context(|| format!("parse {}", path.display()))
}

fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("write {}", path.display()))
}

fn build_worker(
    model_path: &Path,
    max_model_len: usize,
    gpu_memory_utilization: f32,
) -> Result<GpuWorker> {
    let config = build_worker_config(model_path, max_model_len, gpu_memory_utilization)?;
    let mut worker = GpuWorker::new(config)?;
    worker.init_model(model_path)?;
    worker.load_weights(model_path)?;
    let (num_gpu_blocks, num_cpu_blocks) =
        worker.profile_num_available_blocks(gpu_memory_utilization)?;
    worker.init_cache(num_gpu_blocks, num_cpu_blocks)?;
    Ok(worker)
}

fn build_case_artifact(
    case_id: String,
    input_token_ids: Vec<TokenId>,
    last_logits: &[f32],
    top_k: usize,
    tokenizer: &Tokenizer,
) -> Result<PinnedPromptCaseArtifact> {
    let argmax_token_id = argmax_token(last_logits)?;
    let argmax_token_text = tokenizer.decode(&[argmax_token_id]).unwrap_or_default();
    let final_position_top_k = top_k_logits(last_logits, top_k, tokenizer)?;
    Ok(PinnedPromptCaseArtifact {
        id: case_id,
        input_token_ids,
        argmax_token_id,
        argmax_token_text,
        final_position_top_k,
    })
}

fn select_case<'a>(
    manifest: &'a PinnedPromptManifest,
    case_id: Option<&str>,
    command_name: &str,
) -> Result<&'a PinnedPromptCase> {
    if let Some(case_id) = case_id {
        return manifest
            .cases
            .iter()
            .find(|case| case.id == case_id)
            .with_context(|| format!("manifest missing case '{}'", case_id));
    }
    match manifest.cases.as_slice() {
        [case] => Ok(case),
        _ => bail!(
            "{command_name} requires exactly one selected case; pass --case-id for manifest '{}'",
            manifest.suite_id,
        ),
    }
}

fn resolve_official_checkout(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if path.join("gpt_oss").is_dir() {
            return Ok(path.to_path_buf());
        }
        bail!(
            "official checkout '{}' is missing gpt_oss/; pass the local openai/gpt-oss repo root",
            path.display()
        );
    }

    let repo_root = repo_root()?;
    let mut candidates = Vec::new();
    if let Some(parent) = repo_root.parent() {
        candidates.push(parent.join("gpt-oss"));
        if let Some(grandparent) = parent.parent() {
            candidates.push(grandparent.join("gpt-oss"));
        }
    }
    for candidate in candidates {
        if candidate.join("gpt_oss").is_dir() {
            return Ok(candidate);
        }
    }
    bail!("could not locate a sibling gpt-oss checkout; pass --official-checkout explicitly")
}

fn resolve_python_interpreter(
    explicit: Option<&Path>,
    official_checkout: &Path,
) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }
    let venv_python = official_checkout.join(".venv").join("bin").join("python");
    if venv_python.is_file() {
        return Ok(venv_python);
    }
    Ok(PathBuf::from("python3"))
}

fn run_official_capture_helper(
    python: &Path,
    official_checkout: &Path,
    visible_devices: &str,
    input: &PinnedPromptOfficialCaptureInput,
    output_path: &Path,
) -> Result<PinnedPromptOfficialCaptureOutput> {
    let helper_input_path = temporary_json_path(output_path, "official-input");
    let helper_output_path = temporary_json_path(output_path, "official-output");
    write_json(&helper_input_path, input)?;

    let output = run_distributed_helper(
        python,
        official_checkout,
        visible_devices,
        &helper_input_path,
        &helper_output_path,
    )?;

    let _ = std::fs::remove_file(&helper_input_path);

    if !output.status.success() {
        let _ = std::fs::remove_file(&helper_output_path);
        bail!(
            "official capture helper failed for python '{}' with CUDA_VISIBLE_DEVICES='{}'. stdout:\n{}\nstderr:\n{}",
            python.display(),
            visible_devices,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    if !helper_output_path.is_file() {
        bail!(
            "official capture helper exited successfully but did not produce rank-0 output at {}",
            helper_output_path.display()
        );
    }

    let helper_output = read_json::<PinnedPromptOfficialCaptureOutput>(&helper_output_path)?;
    let _ = std::fs::remove_file(&helper_output_path);
    Ok(helper_output)
}

fn run_official_intermediate_capture_helper(
    python: &Path,
    official_checkout: &Path,
    visible_devices: &str,
    input: &PinnedPromptOfficialIntermediateCaptureInput,
    output_path: &Path,
) -> Result<PinnedPromptOfficialIntermediateCaptureOutput> {
    let helper_input_path = temporary_json_path(output_path, "official-intermediate-input");
    let helper_output_path = temporary_json_path(output_path, "official-intermediate-output");
    write_json(&helper_input_path, input)?;

    let output = run_distributed_helper(
        python,
        official_checkout,
        visible_devices,
        &helper_input_path,
        &helper_output_path,
    )?;

    let _ = std::fs::remove_file(&helper_input_path);

    if !output.status.success() {
        let _ = std::fs::remove_file(&helper_output_path);
        bail!(
            "official intermediate capture helper failed for python '{}' with CUDA_VISIBLE_DEVICES='{}'. stdout:\n{}\nstderr:\n{}",
            python.display(),
            visible_devices,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    if !helper_output_path.is_file() {
        bail!(
            "official intermediate capture helper exited successfully but did not produce rank-0 output at {}",
            helper_output_path.display()
        );
    }

    let helper_output =
        read_json::<PinnedPromptOfficialIntermediateCaptureOutput>(&helper_output_path)?;
    let _ = std::fs::remove_file(&helper_output_path);
    Ok(helper_output)
}

fn validate_official_capture_output(
    output: &PinnedPromptOfficialCaptureOutput,
    input: &PinnedPromptOfficialCaptureInput,
) -> Result<()> {
    if output.suite_id != input.suite_id {
        bail!(
            "official capture output suite_id mismatch: expected '{}', got '{}'",
            input.suite_id,
            output.suite_id
        );
    }
    if output.case_id != input.case_id {
        bail!(
            "official capture output case_id mismatch: expected '{}', got '{}'",
            input.case_id,
            output.case_id
        );
    }
    if output.prompt_renderer != input.prompt_renderer {
        bail!(
            "official capture output prompt_renderer mismatch: expected '{}', got '{}'",
            input.prompt_renderer,
            output.prompt_renderer
        );
    }
    if output.official_model != input.official_model {
        bail!(
            "official capture output official_model mismatch: expected '{}', got '{}'",
            input.official_model,
            output.official_model
        );
    }
    if output.input_token_ids != input.input_token_ids {
        bail!("official capture output input_token_ids mismatch");
    }
    if output.backend != "official_torch" {
        bail!(
            "official capture output backend mismatch: expected 'official_torch', got '{}'",
            output.backend
        );
    }
    if output.final_position_top_k.is_empty() {
        bail!("official capture output final_position_top_k is empty");
    }
    Ok(())
}

fn validate_official_intermediate_capture_output(
    output: &PinnedPromptOfficialIntermediateCaptureOutput,
    input: &PinnedPromptOfficialIntermediateCaptureInput,
) -> Result<()> {
    if output.suite_id != input.suite_id {
        bail!(
            "official intermediate capture output suite_id mismatch: expected '{}', got '{}'",
            input.suite_id,
            output.suite_id
        );
    }
    if output.case_id != input.case_id {
        bail!(
            "official intermediate capture output case_id mismatch: expected '{}', got '{}'",
            input.case_id,
            output.case_id
        );
    }
    if output.prompt_renderer != input.prompt_renderer {
        bail!(
            "official intermediate capture output prompt_renderer mismatch: expected '{}', got '{}'",
            input.prompt_renderer,
            output.prompt_renderer
        );
    }
    if output.official_model != input.official_model {
        bail!(
            "official intermediate capture output official_model mismatch: expected '{}', got '{}'",
            input.official_model,
            output.official_model
        );
    }
    if output.input_token_ids != input.input_token_ids {
        bail!("official intermediate capture output input_token_ids mismatch");
    }
    if output.backend != "official_torch" {
        bail!(
            "official intermediate capture output backend mismatch: expected 'official_torch', got '{}'",
            output.backend
        );
    }
    if output.boundary != input.boundary {
        bail!(
            "official intermediate capture output boundary mismatch: expected '{:?}', got '{:?}'",
            input.boundary,
            output.boundary
        );
    }
    if output.layer_idx != input.layer_idx {
        bail!(
            "official intermediate capture output layer_idx mismatch: expected '{:?}', got '{:?}'",
            input.layer_idx,
            output.layer_idx
        );
    }
    validate_intermediate_boundary_layer_idx(
        output.boundary,
        output.layer_idx,
        "official intermediate capture output",
    )?;
    if output.hidden_size == 0 {
        bail!("official intermediate capture output hidden_size must be greater than 0");
    }
    if output.final_token_hidden_f32.len() != output.hidden_size {
        bail!(
            "official intermediate capture output hidden size mismatch: hidden_size={}, values={}",
            output.hidden_size,
            output.final_token_hidden_f32.len()
        );
    }
    Ok(())
}

fn temporary_json_path(base_output: &Path, label: &str) -> PathBuf {
    let stem = base_output
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("pinned-prompt-parity");
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before unix epoch")
        .as_nanos();
    base_output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{stem}.{label}.{}.json", nanos))
}

fn repo_root() -> Result<PathBuf> {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .context("failed to resolve repository root from CARGO_MANIFEST_DIR")
}

fn build_worker_config(
    model_path: &Path,
    max_model_len: usize,
    gpu_memory_utilization: f32,
) -> Result<WorkerConfig> {
    let config_path = model_path.join("config.json");
    let value: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
    let get_usize = |key: &str, default: usize| -> usize {
        value
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(default)
    };
    let get_f32 = |key: &str, default: f32| -> f32 {
        value
            .get(key)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(default)
    };
    let get_bool = |key: &str, default: bool| -> bool {
        value.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
    };
    let layer_types = value
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|vals| {
            vals.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let hidden_size = get_usize("hidden_size", 2880);
    let num_attention_heads = get_usize("num_attention_heads", 64);
    let head_dim = get_usize(
        "head_dim",
        hidden_size.checked_div(num_attention_heads).unwrap_or(64),
    );

    Ok(WorkerConfig {
        model_name: model_path.display().to_string(),
        runtime_mode: RuntimeMode::Trusted,
        device_id: 0,
        num_layers: get_usize("num_hidden_layers", 24),
        num_kv_heads: get_usize("num_key_value_heads", 8),
        head_dim,
        hidden_size,
        num_attention_heads,
        intermediate_size: get_usize("intermediate_size", 2880),
        vocab_size: get_usize("vocab_size", 201088),
        max_model_len: max_model_len.min(get_usize("max_position_embeddings", max_model_len)),
        rms_norm_eps: get_f32("rms_norm_eps", 1e-5),
        block_size: 16,
        gpu_memory_utilization,
        rank: 0,
        tensor_parallel_size: 1,
        pipeline_parallel_size: 1,
        architecture: value
            .get("architectures")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            .unwrap_or("GptOssForCausalLM")
            .to_string(),
        dtype: Dtype::Float16,
        rope_theta: get_f32("rope_theta", 150000.0),
        partial_rotary_factor: get_f32("partial_rotary_factor", 1.0),
        attn_logit_softcapping: get_f32("attn_logit_softcapping", 0.0),
        attention_bias: get_bool("attention_bias", false),
        sliding_window: value
            .get("sliding_window")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        layer_types,
        num_local_experts: get_usize("num_local_experts", 32),
        num_experts_per_tok: get_usize("num_experts_per_tok", 4),
        kv_cache_dtype: "auto".into(),
        enable_prefix_caching: false,
    })
}

fn build_single_sequence_metadata(prompt_token_ids: &[TokenId]) -> Vec<SequenceGroupMetadata> {
    let seq_id = SequenceId(1);
    let total_len = prompt_token_ids.len();
    let num_blocks = total_len.max(1).div_ceil(16);
    let mut seq_data = HashMap::new();
    seq_data.insert(
        seq_id,
        SequenceData {
            prompt_token_ids: prompt_token_ids.to_vec(),
            output_token_ids: Vec::new(),
            cumulative_logprob: 0.0,
        },
    );
    let mut block_tables = HashMap::new();
    block_tables.insert(seq_id, (0..num_blocks).map(|i| BlockId(i as u32)).collect());

    vec![SequenceGroupMetadata {
        request_id: RequestId(1),
        is_prompt: true,
        seq_data,
        sampling_params: SamplingParams {
            temperature: 0.0,
            max_tokens: 1,
            seed: Some(0),
            ..SamplingParams::default()
        },
        block_tables,
    }]
}

fn last_token_logits<'a>(logits: &'a [f32], vocab_size: usize) -> Result<&'a [f32]> {
    if logits.len() < vocab_size {
        bail!(
            "logits buffer too small: {} < vocab {}",
            logits.len(),
            vocab_size
        );
    }
    let num_tokens = logits.len() / vocab_size;
    let last_offset = (num_tokens - 1) * vocab_size;
    Ok(&logits[last_offset..last_offset + vocab_size])
}

fn argmax_token(logits: &[f32]) -> Result<TokenId> {
    logits
        .iter()
        .enumerate()
        .max_by(|(left_idx, left), (right_idx, right)| {
            left.total_cmp(right).then_with(|| right_idx.cmp(left_idx))
        })
        .map(|(idx, _)| idx as TokenId)
        .context("empty logits buffer")
}

fn top_k_logits(
    logits: &[f32],
    top_k: usize,
    tokenizer: &Tokenizer,
) -> Result<Vec<PinnedTopLogit>> {
    let mut indexed = logits
        .iter()
        .enumerate()
        .map(|(idx, logit)| (idx as TokenId, *logit))
        .collect::<Vec<_>>();
    indexed.sort_by(|(left_id, left_logit), (right_id, right_logit)| {
        right_logit
            .total_cmp(left_logit)
            .then_with(|| left_id.cmp(right_id))
    });
    Ok(indexed
        .into_iter()
        .take(top_k.max(1))
        .map(|(token_id, logit)| PinnedTopLogit {
            token_id,
            token_text: tokenizer.decode(&[token_id]).unwrap_or_default(),
            logit,
        })
        .collect())
}
