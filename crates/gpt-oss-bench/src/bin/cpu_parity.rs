use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use gpt_oss_cpu_kernels::{KernelPath, Mxfp4MatmulBackend};
use gpt_oss_evidence::{
    ArtifactRef, EvidenceStatus, ModelEvidence, OracleIdentityEvidence, RunManifestV1,
    SourceProvenance, WorkloadEvidence,
};
use gpt_oss_model_runner::{
    CpuDenseBoundaryProbe, CpuExpertProjection, CpuModelRunner, CpuModelRunnerOptions,
    CpuPrefillTrace, CpuXeAttachmentMode, CpuXeConfig,
};
use gpt_oss_tokenizer::{
    FunctionDefinition, HarmonyProtocol, ProtocolMessage, ToolDefinition, ToolParameterProperty,
    ToolParameters, HARMONY_CALL_TOKEN_ID, HARMONY_RETURN_TOKEN_ID,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const DEFAULT_FIXTURES: &str = "crates/gpt-oss-bench/fixtures/cpu_harmony_parity.json";

#[derive(Debug, Parser)]
#[command(about = "Pinned GPT-OSS CPU Harmony parity and prefill trace runner")]
struct Cli {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    repack_cache: PathBuf,

    #[arg(long, default_value = DEFAULT_FIXTURES)]
    fixtures: PathBuf,

    #[arg(long)]
    scenario: String,

    #[arg(long, default_value = "auto")]
    kernel: KernelPath,

    #[arg(long, default_value = "auto")]
    cpu_matmul_backend: Mxfp4MatmulBackend,

    #[arg(long, default_value = "residual-q8")]
    expert_projection: CpuExpertProjection,

    #[arg(long, default_value_t = 4)]
    threads: usize,

    #[arg(long, default_value_t = 8)]
    max_new_tokens: usize,

    /// Run the prompt in one transactional layer-major batch. This exposes
    /// real multi-row expert buckets for offline profiling.
    #[arg(long, default_value_t = false, conflicts_with = "trace_layers")]
    layer_major_prefill: bool,

    /// Profiling-only deterministic repeated-segment prompt. The pinned
    /// fixture remains the source template and official parity is disabled.
    #[arg(long, requires = "cpu_profile_output")]
    profile_repeated_segments: Option<usize>,

    #[arg(long, value_delimiter = ',')]
    trace_layers: Vec<usize>,

    /// Zero-based generated-token index whose selecting context/logits are
    /// captured. Step 0 is the final prefill token; step N>0 is the decode
    /// after generated token N-1.
    #[arg(long, requires = "trace_layers")]
    trace_step: Option<usize>,

    #[arg(long, default_value_t = 8)]
    top_k: usize,

    /// Attach the explicit CPU+Xe projection backend. Requires `--features xe`.
    #[arg(long, default_value_t = false)]
    xe: bool,

    #[arg(long, default_value_t = 128)]
    xe_max_resident_mib: usize,

    /// Forced-only immutable expert cache; legal only with --xe.
    #[arg(long, default_value_t = 0)]
    xe_expert_cache_mib: usize,

    /// Prime the explicit Xe cache with layer-major prefills in this process.
    #[arg(long, default_value_t = 0, requires = "xe")]
    xe_warmup_prefills: usize,

    #[arg(long)]
    cpu_profile_output: Option<PathBuf>,

    #[arg(long, requires = "cpu_profile_output")]
    cpu_profile_cap_mib: Option<usize>,
    /// Offline-only layer-0 dense boundary projection (`k` or `v`).
    #[arg(long, requires = "dense_boundary_output")]
    dense_boundary_projection: Option<String>,

    #[arg(long, requires = "dense_boundary_projection")]
    dense_boundary_output: Option<usize>,

    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Deserialize, Serialize)]
struct FixtureManifest {
    schema_version: u32,
    model: serde_json::Value,
    official_oracle: serde_json::Value,
    llama_cpp: serde_json::Value,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Scenario {
    id: String,
    #[serde(flatten)]
    input: ScenarioInput,
    expected_prompt_tokens: usize,
    prompt_text_sha256: String,
    prompt_token_ids_sha256: String,
    #[serde(default)]
    official_greedy_tokens: Option<Vec<u32>>,
    #[serde(default)]
    llama_ubatch_1_greedy_tokens: Option<Vec<u32>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ScenarioInput {
    UserText { content: String },
    RepeatedSegments { segments: usize },
    ToolHistory,
}

#[derive(Debug, Serialize)]
struct ParityCapture<'a> {
    schema_version: u32,
    scenario: &'a str,
    fixture_scenario: &'a str,
    profile_repeated_segments: Option<usize>,
    fixture_manifest: &'a Path,
    model_path: &'a Path,
    repack_cache: &'a Path,
    executable_sha256: String,
    kernel: String,
    cpu_matmul_backend: String,
    effective_cpu_kernel: String,
    effective_dispatch_plan: String,
    effective_m1_matrix_backend: String,
    effective_multirow_matrix_backend: String,
    layer_major_prefill: bool,
    expert_projection: CpuExpertProjection,
    xe: Option<serde_json::Value>,
    xe_residency: Option<serde_json::Value>,
    xe_warmup_prefills: usize,
    profile_measured_sequence_start: Option<usize>,
    xe_residency_before_request: Option<serde_json::Value>,
    prompt_text: String,
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    expected_official_greedy_tokens: Option<&'a [u32]>,
    pinned_llama_ubatch_1_greedy_tokens: Option<&'a [u32]>,
    startup_seconds: f64,
    prompt_seconds: f64,
    generation_seconds: f64,
    time_to_first_token_seconds: f64,
    full_request_seconds: f64,
    token_arrival_seconds: Vec<f64>,
    inter_token_seconds: Vec<f64>,
    trace: Option<CpuPrefillTrace>,
    dense_boundary_probe: Option<CpuDenseBoundaryProbe>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dense_boundary_probe_repetitions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dense_boundary_probe_repeat_identical: Option<bool>,
    pinned_model: &'a serde_json::Value,
    pinned_official_oracle: &'a serde_json::Value,
    pinned_llama_cpp: &'a serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    oracle_identity: Option<OracleIdentityEvidence>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Err(error) = run(&cli) {
        let failure = serde_json::json!({
            "schema_version": 1,
            "evidence_status": "incomplete",
            "worker": "cpu_parity",
            "scenario": cli.scenario,
            "requested_kernel": cli.kernel.to_string(),
            "requested_matrix_backend": cli.cpu_matmul_backend.to_string(),
            "error": format!("{error:#}"),
        });
        let file_name = cli
            .output
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("capture");
        let failure_path = cli
            .output
            .with_file_name(format!("{file_name}.failure.json"));
        if let Ok(bytes) = gpt_oss_evidence::stable_json(&failure) {
            let _ = gpt_oss_evidence::atomic_write_new(&failure_path, &bytes);
        }
        return Err(error);
    }
    Ok(())
}

fn run(cli: &Cli) -> Result<()> {
    if cli.xe_expert_cache_mib != 0 && !cli.xe {
        bail!("--xe-expert-cache-mib requires --xe");
    }
    if cli.xe_warmup_prefills != 0 && !cli.layer_major_prefill {
        bail!("--xe-warmup-prefills requires --layer-major-prefill");
    }
    let oracle_identity = oracle_identity_from_environment()?;
    validate_trace_step(cli.trace_step, cli.max_new_tokens)?;
    let manifest = load_manifest(&cli.fixtures)?;
    let scenario = manifest
        .scenarios
        .iter()
        .find(|scenario| scenario.id == cli.scenario)
        .with_context(|| format!("unknown fixture scenario '{}'", cli.scenario))?;
    if cli.profile_repeated_segments == Some(0) {
        bail!("--profile-repeated-segments must be positive");
    }
    if cli.profile_repeated_segments.is_some() && !cli.layer_major_prefill {
        bail!("--profile-repeated-segments requires --layer-major-prefill");
    }
    let rendered = if let Some(segments) = cli.profile_repeated_segments {
        render_repeated_segments(segments)?
    } else {
        let rendered = render_scenario(scenario)?;
        verify_rendered_fixture(scenario, &rendered.text, &rendered.token_ids)?;
        rendered
    };
    let workload_id = cli.profile_repeated_segments.map_or_else(
        || scenario.id.clone(),
        |segments| format!("{}_segments_{segments}", scenario.id),
    );
    let actual_prompt_sha256 = sha256(rendered.text.as_bytes());

    let context_cap = rendered
        .token_ids
        .len()
        .checked_add(cli.max_new_tokens)
        .context("context length overflow")?;
    let startup_start = Instant::now();
    let xe_max_resident_bytes = cli
        .xe_max_resident_mib
        .checked_mul(1024 * 1024)
        .context("--xe-max-resident-mib overflows bytes")?;
    let xe_expert_cache_bytes = cli
        .xe_expert_cache_mib
        .checked_mul(1024 * 1024)
        .context("--xe-expert-cache-mib overflows bytes")?;
    let profile_capacity_bytes = cli
        .cpu_profile_output
        .as_ref()
        .map(|_| {
            cli.cpu_profile_cap_mib
                .unwrap_or(16)
                .checked_mul(1024 * 1024)
                .context("--cpu-profile-cap-mib overflows bytes")
        })
        .transpose()?;
    let mut runner = CpuModelRunner::load_with_options(
        &cli.model,
        &cli.repack_cache,
        CpuModelRunnerOptions {
            kernel_path: cli.kernel,
            matmul_backend: cli.cpu_matmul_backend,
            threads: cli.threads,
            context_cap,
            expert_projection: cli.expert_projection,
            xe: cli.xe.then_some(CpuXeConfig {
                mode: CpuXeAttachmentMode::Explicit,
                max_resident_bytes: xe_max_resident_bytes,
                expert_cache_bytes: xe_expert_cache_bytes,
            }),
            profile_capacity_bytes,
        },
    )?;
    let startup_seconds = startup_start.elapsed().as_secs_f64();
    let xe_descriptor = runner
        .model()
        .xe_descriptor()
        .map(serde_json::to_value)
        .transpose()?;
    for _ in 0..cli.xe_warmup_prefills {
        let _ = runner.prefill_layer_major(&rendered.token_ids)?;
    }
    let profile_measured_sequence_start = runner.execution_profile_records_written();
    let xe_residency_before_request = runner
        .model()
        .xe_residency_stats()
        .map(serde_json::to_value)
        .transpose()?;

    let prompt_start = Instant::now();
    let target_trace_step = (!cli.trace_layers.is_empty()).then_some(cli.trace_step.unwrap_or(0));
    let (mut logits, mut trace) = if target_trace_step == Some(0) {
        let (logits, trace) =
            runner.prefill_trace(&rendered.token_ids, &cli.trace_layers, cli.top_k)?;
        (logits, Some(trace))
    } else if cli.layer_major_prefill {
        (runner.prefill_layer_major(&rendered.token_ids)?, None)
    } else {
        (runner.prefill(&rendered.token_ids)?, None)
    };
    let prompt_seconds = prompt_start.elapsed().as_secs_f64();

    let generation_start = Instant::now();
    let request_start = prompt_start;
    let mut generated_token_ids = Vec::with_capacity(cli.max_new_tokens);
    let mut token_arrival_seconds = Vec::with_capacity(cli.max_new_tokens);
    for step in 0..cli.max_new_tokens {
        let token_id = greedy_token(&logits)? as u32;
        generated_token_ids.push(token_id);
        token_arrival_seconds.push(request_start.elapsed().as_secs_f64());
        if matches!(token_id, HARMONY_RETURN_TOKEN_ID | HARMONY_CALL_TOKEN_ID) {
            break;
        }
        if target_trace_step == Some(step + 1) {
            let (next_logits, step_trace) =
                runner.decode_trace(token_id, &cli.trace_layers, cli.top_k, step + 1)?;
            logits = next_logits;
            trace = Some(step_trace);
        } else {
            logits = runner.decode(token_id)?;
        }
    }
    if target_trace_step.is_some() && trace.is_none() {
        bail!("generation stopped before the requested --trace-step");
    }
    let generation_seconds = generation_start.elapsed().as_secs_f64();
    let time_to_first_token_seconds = token_arrival_seconds
        .first()
        .copied()
        .unwrap_or(prompt_seconds);
    let full_request_seconds = request_start.elapsed().as_secs_f64();
    let inter_token_seconds = token_arrival_seconds
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect();
    let dense_boundary_probe = match (
        cli.dense_boundary_projection.as_deref(),
        cli.dense_boundary_output,
    ) {
        (Some(projection), Some(output_index)) => {
            let layer = trace
                .as_ref()
                .and_then(|trace| trace.layers.iter().find(|layer| layer.layer_index == 0))
                .context("dense boundary probe requires --trace-layers to include layer 0")?;
            let mut probes = Vec::with_capacity(5);
            for _ in 0..5 {
                probes.push(runner.dense_boundary_probe(layer, projection, output_index)?);
            }
            let encoded = probes
                .iter()
                .map(serde_json::to_vec)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if encoded.windows(2).any(|pair| pair[0] != pair[1]) {
                bail!("dense boundary probe was not repeat-identical");
            }
            probes.into_iter().next()
        }
        (None, None) => None,
        _ => bail!("dense boundary projection and output must be supplied together"),
    };
    let dense_boundary_probe_repetitions = dense_boundary_probe.as_ref().map(|_| 5);
    let dense_boundary_probe_repeat_identical = dense_boundary_probe.as_ref().map(|_| true);
    if cli.profile_repeated_segments.is_none() {
        if let Some(expected) = &scenario.official_greedy_tokens {
            if cli.max_new_tokens >= expected.len() && &generated_token_ids != expected {
                bail!(
                    "scenario {} generated {:?}, expected official tokens {:?}",
                    scenario.id,
                    generated_token_ids,
                    expected
                );
            }
        }
    }
    let (expected_official, expected_llama) = if cli.profile_repeated_segments.is_none() {
        (
            scenario.official_greedy_tokens.as_deref(),
            scenario.llama_ubatch_1_greedy_tokens.as_deref(),
        )
    } else {
        (None, None)
    };
    let xe_residency = runner
        .model()
        .xe_residency_stats()
        .map(serde_json::to_value)
        .transpose()?;

    let capture = ParityCapture {
        schema_version: manifest.schema_version,
        scenario: &workload_id,
        fixture_scenario: &scenario.id,
        profile_repeated_segments: cli.profile_repeated_segments,
        fixture_manifest: &cli.fixtures,
        model_path: &cli.model,
        repack_cache: &cli.repack_cache,
        executable_sha256: std::env::current_exe()
            .context("could not resolve parity runner executable")
            .and_then(|path| {
                std::fs::read(path).context("could not read parity runner executable for hashing")
            })
            .map(|bytes| sha256(&bytes))?,
        kernel: cli.kernel.to_string(),
        cpu_matmul_backend: runner.matmul_backend().to_string(),
        effective_cpu_kernel: runner.kernel_path().to_string(),
        effective_dispatch_plan: runner.kernel_dispatch_plan().to_string(),
        effective_m1_matrix_backend: runner.matmul_backend().resolved_for_rows(1).to_string(),
        effective_multirow_matrix_backend: runner.matmul_backend().resolved_for_rows(2).to_string(),
        layer_major_prefill: cli.layer_major_prefill,
        expert_projection: runner.expert_projection(),
        xe: xe_descriptor,
        xe_residency,
        xe_warmup_prefills: cli.xe_warmup_prefills,
        profile_measured_sequence_start,
        xe_residency_before_request,
        prompt_text: rendered.text,
        prompt_token_ids: rendered.token_ids,
        generated_token_ids,
        expected_official_greedy_tokens: expected_official,
        pinned_llama_ubatch_1_greedy_tokens: expected_llama,
        startup_seconds,
        prompt_seconds,
        generation_seconds,
        time_to_first_token_seconds,
        full_request_seconds,
        token_arrival_seconds,
        inter_token_seconds,
        trace,
        dense_boundary_probe,
        dense_boundary_probe_repetitions,
        dense_boundary_probe_repeat_identical,
        pinned_model: &manifest.model,
        pinned_official_oracle: &manifest.official_oracle,
        pinned_llama_cpp: &manifest.llama_cpp,
        oracle_identity: oracle_identity.clone(),
    };
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let encoded = serde_json::to_vec_pretty(&capture)?;
    gpt_oss_evidence::atomic_write_new(&cli.output, &encoded)?;
    write_evidence_sidecar(
        cli,
        &manifest,
        &workload_id,
        &actual_prompt_sha256,
        &encoded,
        oracle_identity,
    )?;
    if let Some(path) = &cli.cpu_profile_output {
        runner.write_execution_profile(path, Some(workload_id))?;
    }
    println!("{}", String::from_utf8(encoded)?);
    Ok(())
}

fn write_evidence_sidecar(
    cli: &Cli,
    fixture: &FixtureManifest,
    workload_id: &str,
    prompt_sha256: &str,
    raw_bytes: &[u8],
    oracle_identity: Option<OracleIdentityEvidence>,
) -> Result<()> {
    let artifact = ArtifactRef::from_path("raw-output", &cli.output)?;
    if artifact.sha256 != sha256(raw_bytes) {
        bail!("written parity output does not match its in-memory capture");
    }
    let mut evidence = RunManifestV1::new(
        format!("cpu-parity-{workload_id}"),
        "correctness",
        EvidenceStatus::InsufficientEvidence,
    );
    evidence.source = local_source_provenance();
    evidence.model = ModelEvidence {
        id: fixture
            .model
            .get("id")
            .or_else(|| fixture.model.get("model"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("cpu-parity-model")
            .to_string(),
        revision: fixture
            .model
            .get("revision")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("fixture-pinned")
            .to_string(),
        ..ModelEvidence::default()
    };
    evidence.command.argv_redacted = std::env::args()
        .map(|argument| {
            if argument == cli.model.to_string_lossy()
                || argument == cli.repack_cache.to_string_lossy()
            {
                "<redacted-local-path>".to_string()
            } else {
                argument
            }
        })
        .collect();
    evidence.workload = WorkloadEvidence {
        id: workload_id.into(),
        prompt_sha256: Some(prompt_sha256.into()),
        seed: 0,
        repetitions: 1,
    };
    evidence.artifacts.push(artifact);
    evidence.oracle_identity = oracle_identity.unwrap_or_default();
    evidence
        .limitations
        .push("single local CPU parity capture".into());
    let file_name = cli
        .output
        .file_name()
        .and_then(|name| name.to_str())
        .context("output has no UTF-8 file name")?;
    let sidecar = cli
        .output
        .with_file_name(format!("{file_name}.manifest.json"));
    evidence.write_atomic_new(sidecar)?;
    Ok(())
}

fn oracle_identity_from_environment() -> Result<Option<OracleIdentityEvidence>> {
    let Some(value) = std::env::var_os("GPT_OSS_ORACLE_IDENTITY_JSON") else {
        return Ok(None);
    };
    let identity: OracleIdentityEvidence = serde_json::from_str(&value.to_string_lossy())
        .context("GPT_OSS_ORACLE_IDENTITY_JSON is invalid")?;
    let mut validation = RunManifestV1::new(
        "oracle-identity-check",
        "identity",
        EvidenceStatus::Incomplete,
    );
    validation.workload.repetitions = 1;
    validation.oracle_identity = identity.clone();
    validation.validate()?;
    Ok(Some(identity))
}

fn local_source_provenance() -> SourceProvenance {
    let repository_commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| value.len() == 40)
        .unwrap_or_default();
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .is_some_and(|output| output.status.success() && !output.stdout.is_empty());
    let cargo_lock_sha256 = std::fs::read("Cargo.lock")
        .ok()
        .map(|bytes| sha256(&bytes))
        .unwrap_or_default();
    let toolchain = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    SourceProvenance {
        repository_commit,
        dirty,
        branch_role: "candidate".into(),
        cargo_lock_sha256,
        toolchain,
        profile: "release".into(),
        features: if cfg!(feature = "xe") {
            vec!["xe".into()]
        } else {
            Vec::new()
        },
    }
}

fn validate_trace_step(trace_step: Option<usize>, max_new_tokens: usize) -> Result<()> {
    if trace_step.is_some_and(|trace_step| trace_step >= max_new_tokens) {
        bail!("--trace-step must be smaller than --max-new-tokens");
    }
    Ok(())
}

fn load_manifest(path: &Path) -> Result<FixtureManifest> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read fixture manifest {}", path.display()))?;
    let manifest: FixtureManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse fixture manifest {}", path.display()))?;
    if manifest.schema_version != 1 {
        bail!(
            "unsupported CPU parity fixture schema {}",
            manifest.schema_version
        );
    }
    Ok(manifest)
}

fn render_scenario(scenario: &Scenario) -> Result<gpt_oss_tokenizer::RenderedPrompt> {
    let (messages, tools) = match &scenario.input {
        ScenarioInput::UserText { content } => {
            (vec![ProtocolMessage::new("user", content)], Vec::new())
        }
        ScenarioInput::RepeatedSegments { segments } => (
            vec![ProtocolMessage::new("user", repeated_content(*segments))],
            Vec::new(),
        ),
        ScenarioInput::ToolHistory => tool_history(),
    };
    HarmonyProtocol::gpt_oss()?
        .render_prompt(&messages, None, &tools)
        .map_err(Into::into)
}

fn repeated_content(segments: usize) -> String {
    (0..segments)
        .map(|index| format!("Segment {index:02}: alpha beta gamma delta epsilon zeta eta theta."))
        .collect::<Vec<_>>()
        .join(" ")
        + " Summarize the repeated pattern in one sentence."
}

fn render_repeated_segments(segments: usize) -> Result<gpt_oss_tokenizer::RenderedPrompt> {
    HarmonyProtocol::gpt_oss()?
        .render_prompt(
            &[ProtocolMessage::new("user", repeated_content(segments))],
            None,
            &[],
        )
        .map_err(Into::into)
}

fn tool_history() -> (Vec<ProtocolMessage>, Vec<ToolDefinition>) {
    let messages = vec![
        ProtocolMessage::new("user", "What is the weather in Boston?"),
        ProtocolMessage::new("assistant", "{\"city\":\"Boston\"}")
            .with_channel("commentary")
            .with_recipient("functions.get_weather"),
        ProtocolMessage::new(
            "tool",
            "{\"city\":\"Boston\",\"temperature_f\":72,\"conditions\":\"sunny\"}",
        )
        .with_author_name("functions.get_weather")
        .with_channel("commentary")
        .with_recipient("assistant"),
        ProtocolMessage::new(
            "user",
            "Given the tool result, answer the user's weather question briefly.",
        ),
    ];
    let mut properties = HashMap::new();
    properties.insert(
        "city".to_string(),
        ToolParameterProperty {
            param_type: "string".to_string(),
            description: Some("City to inspect".to_string()),
            enum_values: None,
        },
    );
    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_weather".to_string(),
            description: Some("Return current weather for a city".to_string()),
            parameters: Some(ToolParameters {
                schema_type: "object".to_string(),
                properties,
                required: vec!["city".to_string()],
            }),
        },
    }];
    (messages, tools)
}

fn verify_rendered_fixture(scenario: &Scenario, text: &str, token_ids: &[u32]) -> Result<()> {
    if token_ids.len() != scenario.expected_prompt_tokens {
        bail!(
            "fixture {} rendered {} tokens, expected {}",
            scenario.id,
            token_ids.len(),
            scenario.expected_prompt_tokens
        );
    }
    let token_text = token_ids
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let text_hash = sha256(text.as_bytes());
    let token_hash = sha256(token_text.as_bytes());
    if text_hash != scenario.prompt_text_sha256 || token_hash != scenario.prompt_token_ids_sha256 {
        bail!(
            "fixture {} rendering changed: text sha256 {}, token sha256 {}",
            scenario.id,
            text_hash,
            token_hash
        );
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn greedy_token(logits: &[f32]) -> Result<usize> {
    logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(left_index, left), (right_index, right)| {
            left.partial_cmp(right)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .context("model returned no finite logits")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_pinned_harmony_scenarios_render_exactly() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/cpu_harmony_parity.json");
        let manifest = load_manifest(&path).unwrap();
        assert_eq!(manifest.scenarios.len(), 7);
        for scenario in &manifest.scenarios {
            let rendered = render_scenario(scenario).unwrap();
            verify_rendered_fixture(scenario, &rendered.text, &rendered.token_ids).unwrap();
        }
    }

    #[test]
    fn trace_step_range_is_zero_based_and_bounded_by_generation() {
        validate_trace_step(Some(0), 8).unwrap();
        validate_trace_step(Some(6), 8).unwrap();
        assert!(validate_trace_step(Some(8), 8).is_err());
        assert!(validate_trace_step(Some(0), 0).is_err());
    }

    #[test]
    fn extended_repeated_workload_reaches_the_2048_token_neighborhood() {
        let rendered = render_repeated_segments(142).unwrap();
        assert!(rendered.token_ids.len().abs_diff(2048) <= 64);
    }
}
