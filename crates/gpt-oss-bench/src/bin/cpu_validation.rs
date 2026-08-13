use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use gpt_oss_evidence::{
    atomic_write_new, stable_json, ArtifactRef, BinaryEvidence, CampaignAttemptV1,
    CampaignIdentity, CampaignIndexV1, DispatchEvidence, EvidenceStatus, OracleIdentityEvidence,
    RunManifestV1, SourceProvenance, WorkloadEvidence,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const REQUIRED_ANCESTOR: &str = "f86674d6acf17484899f5d17e286dcb2c6d1f850";
const REQUIRED_BRANCH: &str = "agent/tiger-lake-optimization-foundation";
const OFFICIAL_REVISION: &str = "599476783c6f88508dab8577808b5ead5cbee8d2";
const LLAMA_REVISION: &str = "030ebb558a5820b444a8f836ed5cdd46c9b4bd7a";
const LLAMA_GGUF_SHA256: &str = "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901";
const FIXTURE_SHA256: &str = "9d8397acf8fe20268e5a5d96fd43b3a6cc2138830585971a0accaf7ef90878ee";
const INDEX_FILE: &str = "campaign-index.json";
const SCENARIOS: [&str; 7] = [
    "harmony_63",
    "harmony_122",
    "harmony_136",
    "harmony_262",
    "harmony_346",
    "harmony_444",
    "tool_history_180",
];
const COMPARISON_CELLS: [(&str, &str); 6] = [
    ("automatic", "auto"),
    ("scalar", "auto"),
    ("avx2", "auto"),
    ("avx512-vnni", "auto"),
    ("automatic", "avx2"),
    ("automatic", "avx512-vnni"),
];
const EXPECTED_OFFICIAL_COMPARISONS: usize = SCENARIOS.len() * COMPARISON_CELLS.len();

#[derive(Debug, Parser)]
#[command(about = "Resumable, hash-verified CPU validation campaign driver")]
struct Cli {
    #[arg(long)]
    root: PathBuf,

    #[command(subcommand)]
    command: CampaignCommand,
}

#[derive(Debug, Subcommand)]
enum CampaignCommand {
    Init(InitArgs),
    Run(RunArgs),
    Resume,
    Finalize(FinalizeArgs),
}

#[derive(Debug, Args)]
struct InitArgs {
    #[arg(long, default_value = "oracle/cpu-oracle.lock.json")]
    oracle_lock: PathBuf,
    #[arg(long)]
    oci_archive: PathBuf,
    #[arg(
        long,
        default_value = "crates/gpt-oss-bench/fixtures/cpu_harmony_parity.json"
    )]
    fixtures: PathBuf,
    #[arg(long, default_value = "/data/models/openai/gpt-oss-20b")]
    model: PathBuf,
    #[arg(
        long,
        default_value = "/home/emmy/src/cpu-runtime-research/llama.cpp-oracle-030ebb558"
    )]
    llama_source: PathBuf,
    #[arg(
        long,
        default_value = "/data/models/llama-cpp/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf"
    )]
    llama_model: PathBuf,
    #[arg(long, default_value_t = 40)]
    minimum_free_gib: u64,
    #[arg(long, default_value_t = 20)]
    reserve_gib: u64,
    #[arg(long)]
    parent_candidate_sha: Option<String>,
}

#[derive(Debug, Args)]
struct RunArgs {
    #[arg(long)]
    phase: String,
    #[arg(long)]
    scenario: String,
    #[arg(long, default_value = "automatic")]
    kernel: String,
    #[arg(long, default_value = "auto")]
    backend: String,
    #[arg(long)]
    effective_kernel: Option<String>,
    #[arg(long)]
    effective_backend: Option<String>,
    #[arg(long, default_value = "native", value_parser = ["native", "generic"])]
    execution_mode: String,
    #[arg(long, default_value_t = 20)]
    reserve_gib: u64,
    #[arg(last = true, required = true)]
    command: Vec<String>,
}

#[derive(Debug, Args)]
struct FinalizeArgs {
    /// Write an incomplete closeout rather than claiming campaign completion.
    #[arg(long)]
    allow_incomplete: bool,
}

#[derive(Debug, Serialize)]
struct PreflightSnapshot {
    schema: &'static str,
    candidate_sha: String,
    origin_main_sha: String,
    required_ancestor: &'static str,
    repository_clean: bool,
    branch: String,
    epoch_seconds: u64,
    free_bytes: u64,
    minimum_free_bytes: u64,
    reserve_bytes: u64,
    available_memory_bytes: u64,
    swap_total_bytes: u64,
    swap_used_bytes: u64,
    oracle_lock: OracleLockCheck,
    model_path: String,
    oci_archive: String,
    fixtures: FileCheck,
    llama_source: RevisionCheck,
    llama_model: FileCheck,
    cache_initial_entries: usize,
    cache_initial_sha256: String,
    commands: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
struct OracleLockCheck {
    path: String,
    sha256: Option<String>,
    valid: bool,
    reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct RevisionCheck {
    path: String,
    expected: &'static str,
    observed: Option<String>,
    clean: bool,
    valid: bool,
}

#[derive(Debug, Serialize)]
struct FileCheck {
    path: String,
    expected_sha256: &'static str,
    observed_sha256: Option<String>,
    bytes: Option<u64>,
    valid: bool,
}

#[derive(Debug, Serialize)]
struct FinalSummary {
    schema: &'static str,
    campaign_id: String,
    candidate_sha: String,
    complete: bool,
    accepted_c3: usize,
    official_comparisons: usize,
    official_comparison_matrix: Vec<String>,
    llama_captures: usize,
    service_cells: usize,
    performance_cells: usize,
    terminal_attempts: usize,
    artifact_set_sha256: String,
    limitations: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct OracleLockCoordinates {
    image_manifest_digest: String,
    image_config_digest: String,
    software_lock_sha256: String,
    official_source_revision: String,
    container_policy_sha256: String,
}

#[derive(Debug, Deserialize)]
struct OraclePreflight {
    status: String,
    lock_path: Option<PathBuf>,
    lock_sha256: Option<String>,
    probes: Option<BTreeMap<String, OracleProbeArtifact>>,
}

#[derive(Debug, Deserialize)]
struct OracleProbeArtifact {
    path: PathBuf,
    sha256: String,
    record: OracleProbeRecord,
}

#[derive(Debug, Deserialize)]
struct OracleProbeRecord {
    host_fingerprint: String,
}

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("cpu_validation: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn execute() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        CampaignCommand::Init(args) => init(&cli.root, &args),
        CampaignCommand::Run(args) => run_attempt(&cli.root, &args),
        CampaignCommand::Resume => resume(&cli.root),
        CampaignCommand::Finalize(args) => finalize(&cli.root, args.allow_incomplete),
    }
}

fn init(root: &Path, args: &InitArgs) -> Result<()> {
    if root.exists() {
        bail!("campaign root already exists: {}", root.display());
    }
    let parent = root.parent().context("campaign root has no parent")?;
    fs::create_dir_all(parent)?;
    let parent = fs::canonicalize(parent)?;
    if !parent.starts_with("/home") {
        bail!("fresh campaign roots must live on /home");
    }
    let free_bytes = filesystem_free_bytes(&parent)?;
    let minimum_free_bytes = gib(args.minimum_free_gib)?;
    let reserve_bytes = gib(args.reserve_gib)?;
    if free_bytes < minimum_free_bytes {
        bail!(
            "campaign filesystem has {} free bytes, below required {}",
            free_bytes,
            minimum_free_bytes
        );
    }

    let candidate_sha = command_text("git", &["rev-parse", "HEAD"])?;
    if root.file_name().and_then(|name| name.to_str()) != Some(candidate_sha.as_str()) {
        bail!("fresh campaign root must be named with the full candidate SHA");
    }
    let origin_main_sha = command_text("git", &["rev-parse", "origin/main"])?;
    let branch = command_text("git", &["branch", "--show-current"])?;
    let repository_clean = command_text("git", &["status", "--porcelain"])?.is_empty();
    if !repository_clean {
        bail!("candidate repository must be clean before campaign init");
    }
    if branch != REQUIRED_BRANCH {
        bail!("campaign must run from {REQUIRED_BRANCH}, observed {branch}");
    }
    command_ok(
        "git",
        &[
            "merge-base",
            "--is-ancestor",
            REQUIRED_ANCESTOR,
            &origin_main_sha,
        ],
    )?;

    let oracle_lock = oracle_lock_check(&args.oracle_lock, &args.oci_archive);
    let fixtures = file_check(&args.fixtures, FIXTURE_SHA256);
    let llama_source = revision_check(&args.llama_source, LLAMA_REVISION);
    let llama_model = file_check(&args.llama_model, LLAMA_GGUF_SHA256);
    let (available_memory_bytes, swap_total_bytes, swap_used_bytes) = memory_snapshot()?;
    let snapshot = PreflightSnapshot {
        schema: "gpt-oss-rs.cpu-validation-preflight/v1",
        candidate_sha: candidate_sha.clone(),
        origin_main_sha,
        required_ancestor: REQUIRED_ANCESTOR,
        repository_clean,
        branch,
        epoch_seconds: epoch_seconds(),
        free_bytes,
        minimum_free_bytes,
        reserve_bytes,
        available_memory_bytes,
        swap_total_bytes,
        swap_used_bytes,
        oracle_lock,
        model_path: args.model.display().to_string(),
        oci_archive: args.oci_archive.display().to_string(),
        fixtures,
        llama_source,
        llama_model,
        cache_initial_entries: 0,
        cache_initial_sha256: format!("{:x}", Sha256::digest([])),
        commands: tool_fingerprints(),
    };

    fs::create_dir(root)?;
    for directory in ["attempts", "cache", "build", "raw", "private", "publish"] {
        fs::create_dir(root.join(directory))?;
    }
    atomic_write_new(
        &root.join("private/preflight.json"),
        &stable_json(&snapshot)?,
    )?;
    let campaign_id = root
        .file_name()
        .and_then(|name| name.to_str())
        .context("campaign root has no UTF-8 name")?;
    let mut index = CampaignIndexV1::new(campaign_id, candidate_sha);
    if let Some(parent) = &args.parent_candidate_sha {
        if parent.len() != 40 || !parent.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            bail!("parent candidate SHA must contain exactly 40 hexadecimal characters");
        }
        index.parent_candidate_sha = Some(parent.clone());
    }
    index.write_atomic(root.join(INDEX_FILE))?;

    if !snapshot.oracle_lock.valid
        || !snapshot.fixtures.valid
        || !snapshot.llama_source.valid
        || !snapshot.llama_model.valid
    {
        write_preflight_terminal(
            root,
            index,
            EvidenceStatus::Unavailable,
            vec!["static oracle lock, fixture, or llama.cpp validation failed".into()],
        )?;
        bail!("pinned preflight dependency unavailable; recorded terminal unavailable attempt");
    }

    let repository = fs::canonicalize(".")?;
    let oracle_output = Command::new("python3")
        .args([
            "oracle/cpu_oracle.py",
            "preflight",
            "--lock",
            args.oracle_lock.to_string_lossy().as_ref(),
            "--repository",
            repository.to_string_lossy().as_ref(),
            "--model",
            args.model.to_string_lossy().as_ref(),
            "--llama-source",
            args.llama_source.to_string_lossy().as_ref(),
            "--oci-archive",
            args.oci_archive.to_string_lossy().as_ref(),
            "--attempt-directory",
            root.join("private/oracle-probes")
                .to_string_lossy()
                .as_ref(),
            "--output",
            root.join("private/oracle-preflight.json")
                .to_string_lossy()
                .as_ref(),
        ])
        .output()
        .context("failed to launch immutable oracle preflight")?;
    atomic_write_new(
        &root.join("private/oracle-preflight.stdout"),
        &oracle_output.stdout,
    )?;
    atomic_write_new(
        &root.join("private/oracle-preflight.stderr"),
        &oracle_output.stderr,
    )?;
    if !oracle_output.status.success() {
        write_preflight_terminal(
            root,
            index,
            EvidenceStatus::Unavailable,
            vec!["immutable image/model/probe qualification failed".into()],
        )?;
        bail!("oracle preflight failed; recorded terminal unavailable attempt");
    }
    let preflight: OraclePreflight =
        serde_json::from_slice(&fs::read(root.join("private/oracle-preflight.json"))?)?;
    if preflight.status != "pass" {
        bail!("oracle preflight exited successfully without pass status");
    }
    write_preflight_terminal(root, index, EvidenceStatus::Pass, Vec::new())?;
    println!("initialized campaign {}", root.display());
    Ok(())
}

fn write_preflight_terminal(
    root: &Path,
    mut index: CampaignIndexV1,
    status: EvidenceStatus,
    limitations: Vec<String>,
) -> Result<()> {
    let cell_key = CampaignIndexV1::stable_cell_key("preflight", "identity", "none", "none")?;
    let attempt_number = index.next_attempt(&cell_key);
    let attempt_id = attempt_id(
        &index.campaign_id,
        &index.candidate_sha,
        "preflight",
        "identity",
        "none",
        "none",
        attempt_number,
    );
    let directory = root.join("attempts").join(&attempt_id);
    fs::create_dir(&directory)?;
    let preflight = ArtifactRef::from_path("preflight", root.join("private/preflight.json"))?;
    let coordinates = ManifestCoordinates {
        attempt_id: &attempt_id,
        cell_key: &cell_key,
        attempt_number,
        phase: "preflight",
        scenario: "identity",
        kernel: "none",
        backend: "none",
    };
    let mut manifest = base_manifest(&index, &coordinates, status)?;
    manifest.artifacts.push(preflight);
    for file in [
        root.join("private/oracle-preflight.json"),
        root.join("private/oracle-preflight.stdout"),
        root.join("private/oracle-preflight.stderr"),
    ] {
        if file.is_file() {
            manifest
                .artifacts
                .push(ArtifactRef::from_path("oracle-preflight", file)?);
        }
    }
    let probe_directory = root.join("private/oracle-probes");
    if probe_directory.is_dir() {
        manifest
            .artifacts
            .extend(artifacts_from_tree("oracle-probe", &probe_directory, &[])?);
    }
    manifest.limitations.extend(limitations);
    if status == EvidenceStatus::Pass {
        manifest.oracle_identity = oracle_identity(root, "native")?;
    }
    write_manifest_pair(&directory, &manifest)?;
    let private =
        ArtifactRef::from_path("terminal-manifest", directory.join("private.manifest.json"))?;
    index.push(CampaignAttemptV1 {
        cell_key,
        attempt_id,
        attempt_number,
        status,
        terminal_manifest: Some(private),
    })?;
    index.write_atomic(root.join(INDEX_FILE))?;
    Ok(())
}

fn run_attempt(root: &Path, args: &RunArgs) -> Result<()> {
    ensure_reserve(root, args.reserve_gib)?;
    validate_command_policy(args)?;
    let mut index = CampaignIndexV1::read(root.join(INDEX_FILE))?;
    validate_campaign_context(&index)?;
    if args.phase == "performance" {
        require_correctness_and_service_gates(&index)?;
    }
    let cell_key =
        CampaignIndexV1::stable_cell_key(&args.phase, &args.scenario, &args.kernel, &args.backend)?;
    if let Some(valid) = valid_completed_attempt(&index, &cell_key)? {
        println!("skipping valid completed attempt {}", valid.attempt_id);
        return Ok(());
    }
    let attempt_number = index.next_attempt(&cell_key);
    let attempt_id = attempt_id(
        &index.campaign_id,
        &index.candidate_sha,
        &args.phase,
        &args.scenario,
        &args.kernel,
        &args.backend,
        attempt_number,
    );
    let directory = root.join("attempts").join(&attempt_id);
    fs::create_dir(&directory)?;
    let oracle_identity = oracle_identity(root, &args.execution_mode)?;
    let identity_json = serde_json::to_string(&oracle_identity)?;
    let started = Instant::now();
    let output = Command::new(&args.command[0])
        .args(&args.command[1..])
        .env("GPT_OSS_ORACLE_IDENTITY_JSON", &identity_json)
        .env("GPT_OSS_CAMPAIGN_ROOT", root)
        .env("GPT_OSS_ATTEMPT_DIR", &directory)
        .env("GPT_OSS_ORACLE_LOCK", campaign_lock_path(root)?)
        .output()
        .with_context(|| format!("failed to launch {}", args.command[0]))?;
    let elapsed = started.elapsed();
    atomic_write_new(&directory.join("stdout.raw"), &output.stdout)?;
    atomic_write_new(&directory.join("stderr.raw"), &output.stderr)?;

    let status = authoritative_status(&args.phase, output.status.code(), &args.execution_mode);
    let coordinates = ManifestCoordinates {
        attempt_id: &attempt_id,
        cell_key: &cell_key,
        attempt_number,
        phase: &args.phase,
        scenario: &args.scenario,
        kernel: &args.kernel,
        backend: &args.backend,
    };
    let mut manifest = base_manifest(&index, &coordinates, status)?;
    manifest.command.argv_redacted = args.command.iter().map(|arg| redact_cli_arg(arg)).collect();
    manifest.dispatch = DispatchEvidence {
        requested_kernel: args.kernel.clone(),
        effective_kernel: args
            .effective_kernel
            .clone()
            .unwrap_or_else(|| args.kernel.clone()),
        requested_matrix_backend: args.backend.clone(),
        effective_matrix_backend: args
            .effective_backend
            .clone()
            .unwrap_or_else(|| args.backend.clone()),
    };
    manifest.measured.wall_time_ns = elapsed.as_nanos().try_into().unwrap_or(u64::MAX);
    manifest.oracle_identity = oracle_identity;
    manifest.artifacts.extend([
        ArtifactRef::from_path("stdout", directory.join("stdout.raw"))?,
        ArtifactRef::from_path("stderr", directory.join("stderr.raw"))?,
    ]);
    manifest.artifacts.extend(artifacts_from_tree(
        "worker-output",
        &directory,
        &[
            Path::new("stdout.raw"),
            Path::new("stderr.raw"),
            Path::new("private.manifest.json"),
            Path::new("publish.manifest.json"),
        ],
    )?);
    if status == EvidenceStatus::InsufficientEvidence {
        manifest
            .limitations
            .push("standalone capture; comparator has not assigned parity".into());
    }
    write_manifest_pair(&directory, &manifest)?;
    let terminal =
        ArtifactRef::from_path("terminal-manifest", directory.join("private.manifest.json"))?;
    index.push(CampaignAttemptV1 {
        cell_key,
        attempt_id,
        attempt_number,
        status,
        terminal_manifest: Some(terminal),
    })?;
    index.write_atomic(root.join(INDEX_FILE))?;
    ensure_reserve(root, args.reserve_gib)?;
    if status == EvidenceStatus::Fail || status == EvidenceStatus::Invalid {
        bail!("attempt completed with {status:?}");
    }
    Ok(())
}

fn validate_command_policy(args: &RunArgs) -> Result<()> {
    if args.phase == "compare"
        && !args
            .command
            .iter()
            .any(|argument| argument.ends_with("run_cpu_comparison.py"))
    {
        bail!("authoritative compare cells must use run_cpu_comparison.py");
    }
    if args.phase == "official"
        && !(args
            .command
            .iter()
            .any(|argument| argument == "oracle/cpu_oracle.py")
            && args.command.iter().any(|argument| argument == "exec"))
    {
        bail!("official captures must execute through oracle/cpu_oracle.py exec");
    }
    if args.phase == "c3"
        && !args
            .command
            .iter()
            .any(|argument| argument.ends_with("run_c3_x001.py"))
    {
        bail!("C3-X-001 cells must use run_c3_x001.py");
    }
    if args.phase == "llama"
        && SCENARIOS.contains(&args.scenario.as_str())
        && !args
            .command
            .iter()
            .any(|argument| argument.ends_with("select_llama_cpu_capture.py"))
    {
        bail!("counted llama cells must use select_llama_cpu_capture.py");
    }
    let required_service_worker = match (args.phase.as_str(), args.scenario.as_str()) {
        ("service", "model-free-lifecycle-http") => Some("run_model_free_service_suite.py"),
        ("service", "bounded-20b") => Some("run_bounded_20b_service.py"),
        _ => None,
    };
    if let Some(worker) = required_service_worker {
        if !args
            .command
            .iter()
            .any(|argument| argument.ends_with(worker))
        {
            bail!("counted service cell must use {worker}");
        }
    }
    Ok(())
}

fn resume(root: &Path) -> Result<()> {
    let index = CampaignIndexV1::read(root.join(INDEX_FILE))?;
    validate_campaign_context(&index)?;
    let mut completed = BTreeSet::new();
    let mut retry = BTreeSet::new();
    for attempt in &index.attempts {
        if attempt.terminal_manifest.is_some() && attempt.status != EvidenceStatus::Incomplete {
            completed.insert(attempt.cell_key.clone());
        } else {
            retry.insert(attempt.cell_key.clone());
        }
    }
    println!("verified {} completed cells", completed.len());
    for cell in retry {
        println!("retry required: {cell}");
    }
    Ok(())
}

fn finalize(root: &Path, allow_incomplete: bool) -> Result<()> {
    let index = CampaignIndexV1::read(root.join(INDEX_FILE))?;
    validate_campaign_context(&index)?;
    let terminal = terminal_cells(&index)?;
    let accepted_c3 = usize::from(cell_accepted(
        &terminal,
        "c3",
        "c3-x-001",
        "automatic",
        "auto",
        &[EvidenceStatus::Pass, EvidenceStatus::InsufficientEvidence],
        true,
    )?);
    let official_comparisons = expected_comparison_cells(&terminal)?;
    let llama_captures = expected_llama_cells(&terminal)?;
    let service_cells = expected_service_cells(&terminal)?;
    let performance_cells = count_phase(
        &terminal,
        "performance",
        &[EvidenceStatus::Pass, EvidenceStatus::InsufficientEvidence],
    );
    let complete = accepted_c3 == 1
        && official_comparisons == EXPECTED_OFFICIAL_COMPARISONS
        && llama_captures == 7
        && service_cells == 2
        && performance_cells >= 1;
    let mut limitations = Vec::new();
    if !complete {
        limitations.push("campaign acceptance matrix is incomplete".into());
    }
    let summary = FinalSummary {
        schema: "gpt-oss-rs.cpu-validation-final/v2",
        campaign_id: index.campaign_id.clone(),
        candidate_sha: index.candidate_sha.clone(),
        complete,
        accepted_c3,
        official_comparisons,
        official_comparison_matrix: COMPARISON_CELLS
            .iter()
            .map(|(kernel, backend)| format!("{kernel}/{backend}"))
            .collect(),
        llama_captures,
        service_cells,
        performance_cells,
        terminal_attempts: terminal.len(),
        artifact_set_sha256: artifact_set_sha256(&terminal),
        limitations,
    };
    atomic_write_new(
        &root.join("private/final-summary.json"),
        &stable_json(&summary)?,
    )?;
    atomic_write_new(
        &root.join("publish/final-summary.json"),
        &stable_json(&summary)?,
    )?;
    if !complete && !allow_incomplete {
        bail!("campaign is incomplete; final summary was preserved without a completion claim");
    }
    Ok(())
}

struct ManifestCoordinates<'a> {
    attempt_id: &'a str,
    cell_key: &'a str,
    attempt_number: u32,
    phase: &'a str,
    scenario: &'a str,
    kernel: &'a str,
    backend: &'a str,
}

fn base_manifest(
    index: &CampaignIndexV1,
    coordinates: &ManifestCoordinates<'_>,
    status: EvidenceStatus,
) -> Result<RunManifestV1> {
    let mut manifest = RunManifestV1::new(coordinates.attempt_id, coordinates.phase, status);
    manifest.source = source_provenance(&index.candidate_sha);
    manifest.workload = WorkloadEvidence {
        id: coordinates.scenario.into(),
        prompt_sha256: None,
        seed: 0,
        repetitions: 1,
    };
    manifest.campaign = CampaignIdentity {
        campaign_id: index.campaign_id.clone(),
        candidate_sha: index.candidate_sha.clone(),
        phase: coordinates.phase.into(),
        scenario: coordinates.scenario.into(),
        requested_kernel: coordinates.kernel.into(),
        attempt_number: coordinates.attempt_number,
        attempt_id: coordinates.attempt_id.into(),
        cell_key: coordinates.cell_key.into(),
    };
    manifest.dispatch = DispatchEvidence {
        requested_kernel: coordinates.kernel.into(),
        effective_kernel: coordinates.kernel.into(),
        requested_matrix_backend: coordinates.backend.into(),
        effective_matrix_backend: coordinates.backend.into(),
    };
    let executable = std::env::current_exe()?;
    manifest.build_binaries.push(BinaryEvidence {
        role: "cpu-validation-driver".into(),
        sha256: gpt_oss_evidence::sha256_file(executable)?,
    });
    Ok(manifest)
}

fn write_manifest_pair(directory: &Path, manifest: &RunManifestV1) -> Result<()> {
    manifest.validate_campaign_complete()?;
    manifest.write_atomic_new(directory.join("private.manifest.json"))?;
    let redacted = manifest.redacted();
    debug_assert_eq!(
        manifest
            .artifacts
            .iter()
            .map(|artifact| (&artifact.role, &artifact.sha256))
            .collect::<Vec<_>>(),
        redacted
            .artifacts
            .iter()
            .map(|artifact| (&artifact.role, &artifact.sha256))
            .collect::<Vec<_>>()
    );
    redacted.write_atomic_new(directory.join("publish.manifest.json"))?;
    Ok(())
}

fn artifacts_from_tree(
    role_prefix: &str,
    root: &Path,
    excluded: &[&Path],
) -> Result<Vec<ArtifactRef>> {
    let mut files = Vec::new();
    collect_regular_files(root, root, &mut files)?;
    files.sort();
    files
        .into_iter()
        .filter(|path| {
            path.strip_prefix(root)
                .is_ok_and(|relative| !excluded.contains(&relative))
        })
        .map(|path| {
            let relative = path
                .strip_prefix(root)
                .expect("collected artifact belongs to its root");
            ArtifactRef::from_path(format!("{role_prefix}:{}", relative.display()), path)
                .map_err(anyhow::Error::from)
        })
        .collect()
}

fn collect_regular_files(root: &Path, directory: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            bail!(
                "artifact tree {} contains a symbolic link: {}",
                root.display(),
                path.display()
            );
        }
        if metadata.is_dir() {
            collect_regular_files(root, &path, files)?;
        } else if metadata.is_file() {
            files.push(path);
        } else {
            bail!(
                "artifact tree {} contains a non-regular entry: {}",
                root.display(),
                path.display()
            );
        }
    }
    Ok(())
}

fn authoritative_status(phase: &str, code: Option<i32>, execution_mode: &str) -> EvidenceStatus {
    match (phase, code, execution_mode) {
        ("compare" | "service", Some(0), "native") => EvidenceStatus::Pass,
        ("compare" | "service", Some(1), "native") => EvidenceStatus::Fail,
        (_, Some(0), _) => EvidenceStatus::InsufficientEvidence,
        (_, Some(_), _) => EvidenceStatus::Invalid,
        (_, None, _) => EvidenceStatus::Incomplete,
    }
}

fn valid_completed_attempt<'a>(
    index: &'a CampaignIndexV1,
    cell_key: &str,
) -> Result<Option<&'a CampaignAttemptV1>> {
    for attempt in index.attempts.iter().rev() {
        if attempt.cell_key != cell_key || attempt.status == EvidenceStatus::Incomplete {
            continue;
        }
        if let Some(terminal_manifest) = &attempt.terminal_manifest {
            terminal_manifest.verify()?;
            return Ok(Some(attempt));
        }
    }
    Ok(None)
}

fn count_phase(attempts: &[&CampaignAttemptV1], phase: &str, accepted: &[EvidenceStatus]) -> usize {
    attempts
        .iter()
        .filter(|attempt| {
            attempt.cell_key.starts_with(&format!("{phase}--"))
                && accepted.contains(&attempt.status)
        })
        .count()
}

fn cell_accepted(
    attempts: &[&CampaignAttemptV1],
    phase: &str,
    scenario: &str,
    kernel: &str,
    backend: &str,
    accepted: &[EvidenceStatus],
    require_native: bool,
) -> Result<bool> {
    let key = CampaignIndexV1::stable_cell_key(phase, scenario, kernel, backend)?;
    let Some(attempt) = attempts
        .iter()
        .find(|attempt| attempt.cell_key == key && accepted.contains(&attempt.status))
    else {
        return Ok(false);
    };
    if require_native {
        let reference = attempt
            .terminal_manifest
            .as_ref()
            .context("accepted attempt lacks terminal manifest")?;
        let manifest: RunManifestV1 = serde_json::from_slice(&fs::read(&reference.absolute_path)?)?;
        if manifest.oracle_identity.execution_mode.as_deref() != Some("native") {
            return Ok(false);
        }
    }
    Ok(true)
}

fn expected_comparison_cells(attempts: &[&CampaignAttemptV1]) -> Result<usize> {
    let mut count = 0;
    for scenario in SCENARIOS {
        for (kernel, backend) in COMPARISON_CELLS {
            count += usize::from(cell_accepted(
                attempts,
                "compare",
                scenario,
                kernel,
                backend,
                &[EvidenceStatus::Pass],
                true,
            )?);
        }
    }
    Ok(count)
}

fn expected_llama_cells(attempts: &[&CampaignAttemptV1]) -> Result<usize> {
    let mut count = 0;
    for scenario in SCENARIOS {
        count += usize::from(cell_accepted(
            attempts,
            "llama",
            scenario,
            "llama-cpp",
            "ubatch-1",
            &[EvidenceStatus::InsufficientEvidence],
            true,
        )?);
    }
    Ok(count)
}

fn expected_service_cells(attempts: &[&CampaignAttemptV1]) -> Result<usize> {
    let model_free = cell_accepted(
        attempts,
        "service",
        "model-free-lifecycle-http",
        "none",
        "test-suite",
        &[EvidenceStatus::Pass],
        true,
    )?;
    let bounded_model = cell_accepted(
        attempts,
        "service",
        "bounded-20b",
        "automatic",
        "auto",
        &[EvidenceStatus::Pass],
        true,
    )?;
    Ok(usize::from(model_free) + usize::from(bounded_model))
}

fn terminal_cells(index: &CampaignIndexV1) -> Result<Vec<&CampaignAttemptV1>> {
    let mut cells = BTreeMap::new();
    for attempt in &index.attempts {
        let Some(reference) = &attempt.terminal_manifest else {
            continue;
        };
        reference.verify()?;
        let manifest: RunManifestV1 = serde_json::from_slice(&fs::read(&reference.absolute_path)?)?;
        manifest.validate_campaign_complete()?;
        if manifest.campaign.campaign_id != index.campaign_id
            || manifest.campaign.candidate_sha != index.candidate_sha
            || manifest.campaign.cell_key != attempt.cell_key
            || manifest.campaign.attempt_id != attempt.attempt_id
            || manifest.status != attempt.status
        {
            bail!(
                "terminal attempt {} crosses campaign identity",
                attempt.attempt_id
            );
        }
        cells.insert(attempt.cell_key.as_str(), attempt);
    }
    Ok(cells.into_values().collect())
}

fn artifact_set_sha256(attempts: &[&CampaignAttemptV1]) -> String {
    let mut hashes = attempts
        .iter()
        .filter_map(|attempt| {
            attempt
                .terminal_manifest
                .as_ref()
                .map(|manifest| manifest.sha256.as_str())
        })
        .collect::<Vec<_>>();
    hashes.sort_unstable();
    format!("{:x}", Sha256::digest(hashes.join("\n")))
}

fn require_correctness_and_service_gates(index: &CampaignIndexV1) -> Result<()> {
    let terminal = terminal_cells(index)?;
    let c3 = cell_accepted(
        &terminal,
        "c3",
        "c3-x-001",
        "automatic",
        "auto",
        &[EvidenceStatus::Pass, EvidenceStatus::InsufficientEvidence],
        true,
    )?;
    let comparisons = expected_comparison_cells(&terminal)?;
    let llama = expected_llama_cells(&terminal)?;
    let service = expected_service_cells(&terminal)?;
    if !c3 || comparisons != EXPECTED_OFFICIAL_COMPARISONS || llama != 7 || service != 2 {
        bail!(
            "performance is locked until C3, 42 official comparisons, seven llama captures, and service gates pass"
        );
    }
    Ok(())
}

fn validate_campaign_context(index: &CampaignIndexV1) -> Result<()> {
    let head = command_text("git", &["rev-parse", "HEAD"])?;
    if head != index.candidate_sha {
        bail!(
            "campaign candidate {} does not match HEAD {head}",
            index.candidate_sha
        );
    }
    let branch = command_text("git", &["branch", "--show-current"])?;
    if branch != REQUIRED_BRANCH {
        bail!("campaign must remain on {REQUIRED_BRANCH}");
    }
    if !command_text("git", &["status", "--porcelain"])?.is_empty() {
        bail!("campaign execution requires a clean candidate repository");
    }
    if index.campaign_id != index.candidate_sha {
        bail!("fresh campaign ID must equal its full candidate SHA");
    }
    Ok(())
}

fn read_oracle_preflight(root: &Path) -> Result<OraclePreflight> {
    let preflight: OraclePreflight =
        serde_json::from_slice(&fs::read(root.join("private/oracle-preflight.json"))?)?;
    if preflight.status != "pass" {
        bail!("oracle preflight is not passing");
    }
    Ok(preflight)
}

fn oracle_identity(root: &Path, execution_mode: &str) -> Result<OracleIdentityEvidence> {
    let preflight = read_oracle_preflight(root)?;
    let lock_path = preflight
        .lock_path
        .context("oracle preflight lacks lock_path")?;
    let lock_bytes = fs::read(&lock_path)?;
    let observed_lock_hash = format!("{:x}", Sha256::digest(&lock_bytes));
    if preflight.lock_sha256.as_deref() != Some(observed_lock_hash.as_str()) {
        bail!("oracle lock changed after campaign preflight");
    }
    let lock: OracleLockCoordinates = serde_json::from_slice(&lock_bytes)?;
    if lock.official_source_revision != OFFICIAL_REVISION {
        bail!("oracle lock official source revision changed");
    }
    let probes = preflight.probes.context("oracle preflight lacks probes")?;
    let probe = probes
        .get(execution_mode)
        .with_context(|| format!("oracle preflight lacks {execution_mode} probe"))?;
    let observed_probe_hash = gpt_oss_evidence::sha256_file(&probe.path)?;
    if observed_probe_hash != probe.sha256 {
        bail!("{execution_mode} probe artifact changed after preflight");
    }
    Ok(OracleIdentityEvidence {
        image_manifest_digest: Some(lock.image_manifest_digest),
        image_config_digest: Some(lock.image_config_digest),
        software_lock_sha256: Some(lock.software_lock_sha256),
        official_source_revision: Some(lock.official_source_revision),
        execution_mode: Some(execution_mode.into()),
        host_fingerprint: Some(probe.record.host_fingerprint.clone()),
        container_policy_sha256: Some(lock.container_policy_sha256),
        probe_artifact_sha256: Some(probe.sha256.clone()),
    })
}

fn campaign_lock_path(root: &Path) -> Result<PathBuf> {
    read_oracle_preflight(root)?
        .lock_path
        .context("oracle preflight lacks lock_path")
}

fn attempt_id(
    campaign: &str,
    candidate: &str,
    phase: &str,
    scenario: &str,
    kernel: &str,
    backend: &str,
    number: u32,
) -> String {
    format!(
        "{campaign}--{}--{phase}--{scenario}--{kernel}--{backend}--{number}",
        &candidate[..12]
    )
}

fn source_provenance(candidate: &str) -> SourceProvenance {
    SourceProvenance {
        repository_commit: candidate.into(),
        dirty: false,
        branch_role: "candidate".into(),
        cargo_lock_sha256: fs::read("Cargo.lock")
            .ok()
            .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
            .unwrap_or_default(),
        toolchain: command_text("rustc", &["--version"]).unwrap_or_else(|_| "unknown".into()),
        profile: "release-locked".into(),
        features: Vec::new(),
    }
}

fn revision_check(path: &Path, expected: &'static str) -> RevisionCheck {
    let observed = command_text(
        "git",
        &["-C", path.to_string_lossy().as_ref(), "rev-parse", "HEAD"],
    )
    .ok();
    let clean = command_text(
        "git",
        &[
            "-C",
            path.to_string_lossy().as_ref(),
            "status",
            "--porcelain",
        ],
    )
    .is_ok_and(|value| value.is_empty());
    RevisionCheck {
        path: path.display().to_string(),
        expected,
        valid: observed.as_deref() == Some(expected) && clean,
        observed,
        clean,
    }
}

fn file_check(path: &Path, expected_sha256: &'static str) -> FileCheck {
    let metadata = fs::metadata(path)
        .ok()
        .filter(|metadata| metadata.is_file());
    let observed_sha256 = metadata
        .as_ref()
        .and_then(|_| gpt_oss_evidence::sha256_file(path).ok());
    FileCheck {
        path: path.display().to_string(),
        valid: observed_sha256.as_deref() == Some(expected_sha256),
        expected_sha256,
        observed_sha256,
        bytes: metadata.map(|metadata| metadata.len()),
    }
}

fn oracle_lock_check(lock: &Path, archive: &Path) -> OracleLockCheck {
    let output = Command::new("python3")
        .args([
            "oracle/cpu_oracle.py",
            "verify-lock",
            "--lock",
            lock.to_string_lossy().as_ref(),
            "--repository",
            ".",
            "--oci-archive",
            archive.to_string_lossy().as_ref(),
        ])
        .output();
    match output {
        Ok(output) if output.status.success() => OracleLockCheck {
            path: lock.display().to_string(),
            sha256: gpt_oss_evidence::sha256_file(lock).ok(),
            valid: true,
            reason: None,
        },
        Ok(output) => OracleLockCheck {
            path: lock.display().to_string(),
            sha256: gpt_oss_evidence::sha256_file(lock).ok(),
            valid: false,
            reason: Some(String::from_utf8_lossy(&output.stderr).trim().to_string()),
        },
        Err(error) => OracleLockCheck {
            path: lock.display().to_string(),
            sha256: None,
            valid: false,
            reason: Some(error.to_string()),
        },
    }
}

fn tool_fingerprints() -> BTreeMap<String, String> {
    [
        ("rustc", "rustc", vec!["--version", "--verbose"]),
        ("cargo", "cargo", vec!["--version", "--verbose"]),
        ("cmake", "cmake", vec!["--version"]),
        ("compiler", "cc", vec!["--version"]),
    ]
    .into_iter()
    .map(|(name, command, args)| {
        (
            name.into(),
            command_text(command, &args).unwrap_or_else(|error| format!("unavailable: {error}")),
        )
    })
    .collect()
}

fn memory_snapshot() -> Result<(u64, u64, u64)> {
    let text = fs::read_to_string("/proc/meminfo")?;
    let mut values = BTreeMap::new();
    for line in text.lines() {
        if let Some((key, rest)) = line.split_once(':') {
            let kib = rest
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
            values.insert(key, kib.saturating_mul(1024));
        }
    }
    let available = *values.get("MemAvailable").unwrap_or(&0);
    let total = *values.get("SwapTotal").unwrap_or(&0);
    let free = *values.get("SwapFree").unwrap_or(&0);
    Ok((available, total, total.saturating_sub(free)))
}

fn filesystem_free_bytes(path: &Path) -> Result<u64> {
    let output = Command::new("df")
        .args(["-Pk", path.to_string_lossy().as_ref()])
        .output()?;
    if !output.status.success() {
        bail!("df failed for {}", path.display());
    }
    let text = String::from_utf8(output.stdout)?;
    let fields = text
        .lines()
        .last()
        .context("df returned no filesystem row")?
        .split_whitespace()
        .collect::<Vec<_>>();
    let available_kib: u64 = fields
        .get(3)
        .context("df row has no available column")?
        .parse()?;
    Ok(available_kib.saturating_mul(1024))
}

fn ensure_reserve(root: &Path, reserve_gib: u64) -> Result<()> {
    let free = filesystem_free_bytes(root)?;
    let reserve = gib(reserve_gib)?;
    if free < reserve {
        bail!("disk reserve violated: {free} free bytes is below {reserve}");
    }
    Ok(())
}

fn command_text(command: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(command).args(args).output()?;
    if !output.status.success() {
        bail!("{command} exited with {}", output.status);
    }
    let stdout = String::from_utf8(output.stdout)?;
    Ok(stdout.trim().to_string())
}

fn command_ok(command: &str, args: &[&str]) -> Result<()> {
    let status = Command::new(command).args(args).status()?;
    if !status.success() {
        bail!("{command} exited with {status}");
    }
    Ok(())
}

fn redact_cli_arg(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    if Path::new(value).is_absolute()
        || lower.contains("token=")
        || lower.contains("secret=")
        || lower.contains("password=")
    {
        "<redacted>".into()
    } else {
        value.into()
    }
}

fn epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn gib(value: u64) -> Result<u64> {
    value
        .checked_mul(1024 * 1024 * 1024)
        .context("GiB byte count overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standalone_workers_never_assign_authoritative_pass() {
        assert_eq!(
            authoritative_status("native", Some(0), "native"),
            EvidenceStatus::InsufficientEvidence
        );
        assert_eq!(
            authoritative_status("official", Some(0), "native"),
            EvidenceStatus::InsufficientEvidence
        );
        assert_eq!(
            authoritative_status("llama", Some(0), "native"),
            EvidenceStatus::InsufficientEvidence
        );
        assert_eq!(
            authoritative_status("compare", Some(0), "native"),
            EvidenceStatus::Pass
        );
        assert_eq!(
            authoritative_status("compare", Some(0), "generic"),
            EvidenceStatus::InsufficientEvidence
        );
    }

    #[test]
    fn attempt_identity_contains_all_required_coordinates() {
        let id = attempt_id(
            "cpu-validation-1",
            &"a".repeat(40),
            "official",
            "harmony_262",
            "avx2",
            "auto",
            3,
        );
        assert_eq!(
            id,
            "cpu-validation-1--aaaaaaaaaaaa--official--harmony_262--avx2--auto--3"
        );
    }

    #[test]
    fn tiger_lake_matrix_has_six_unique_effective_controls() {
        assert_eq!(EXPECTED_OFFICIAL_COMPARISONS, 42);
        assert_eq!(
            COMPARISON_CELLS.into_iter().collect::<BTreeSet<_>>().len(),
            6
        );
        assert!(COMPARISON_CELLS.contains(&("automatic", "avx2")));
        assert!(COMPARISON_CELLS.contains(&("automatic", "avx512-vnni")));
    }
}
