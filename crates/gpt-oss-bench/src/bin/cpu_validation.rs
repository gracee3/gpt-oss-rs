use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use gpt_oss_evidence::{
    atomic_write_new, stable_json, ArtifactRef, BinaryEvidence, CampaignAttemptV1,
    CampaignIdentity, CampaignIndexV1, DispatchEvidence, EvidenceStatus, RunManifestV1,
    SourceProvenance, WorkloadEvidence,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

const REQUIRED_ANCESTOR: &str = "ac3eea350c2e926087f0b4eb67afa75ee5eecde1";
const OFFICIAL_REVISION: &str = "7802bf263f902efd4c7d18fcceff3ba72f941e80";
const LLAMA_REVISION: &str = "030ebb558a5820b444a8f836ed5cdd46c9b4bd7a";
const INDEX_FILE: &str = "campaign-index.json";

#[derive(Debug, Parser)]
#[command(about = "Resumable, hash-verified CPU validation campaign driver")]
struct Cli {
    #[arg(long, global = true)]
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
    #[arg(long, default_value = "/data/models/.venv-awq")]
    oracle_venv: PathBuf,
    #[arg(
        long,
        default_value = "/home/emmy/src/cpu-runtime-research/openai-gpt-oss-oracle-7802bf263"
    )]
    official_source: PathBuf,
    #[arg(
        long,
        default_value = "/home/emmy/src/cpu-runtime-research/llama.cpp-oracle-030ebb558"
    )]
    llama_source: PathBuf,
    #[arg(long, default_value_t = 40)]
    minimum_free_gib: u64,
    #[arg(long, default_value_t = 20)]
    reserve_gib: u64,
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
    oracle_venv: DependencyCheck,
    official_source: RevisionCheck,
    llama_source: RevisionCheck,
    commands: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
struct DependencyCheck {
    path: String,
    available: bool,
    python: Option<String>,
    fingerprint: Option<String>,
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
struct FinalSummary {
    schema: &'static str,
    campaign_id: String,
    candidate_sha: String,
    complete: bool,
    accepted_c3: usize,
    official_comparisons: usize,
    llama_captures: usize,
    service_cells: usize,
    performance_cells: usize,
    terminal_attempts: usize,
    limitations: Vec<String>,
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
    let free_bytes = filesystem_free_bytes(parent)?;
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
    let origin_main_sha = command_text("git", &["rev-parse", "origin/main"])?;
    let branch = command_text("git", &["branch", "--show-current"])?;
    let repository_clean = command_text("git", &["status", "--porcelain"])?.is_empty();
    if !repository_clean {
        bail!("candidate repository must be clean before campaign init");
    }
    if branch != "agent/cpu-validation-closure" {
        bail!("campaign must run from agent/cpu-validation-closure, observed {branch}");
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
        oracle_venv: dependency_check(&args.oracle_venv),
        official_source: revision_check(&args.official_source, OFFICIAL_REVISION),
        llama_source: revision_check(&args.llama_source, LLAMA_REVISION),
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
    let index = CampaignIndexV1::new(campaign_id, candidate_sha);
    index.write_atomic(root.join(INDEX_FILE))?;

    if !snapshot.oracle_venv.available
        || !snapshot.official_source.valid
        || !snapshot.llama_source.valid
    {
        write_preflight_unavailable(root, &snapshot, index)?;
        bail!("pinned preflight dependency unavailable; recorded terminal unavailable attempt");
    }
    println!("initialized campaign {}", root.display());
    Ok(())
}

fn write_preflight_unavailable(
    root: &Path,
    snapshot: &PreflightSnapshot,
    mut index: CampaignIndexV1,
) -> Result<()> {
    let cell_key = CampaignIndexV1::stable_cell_key("preflight", "identity", "none", "none")?;
    let attempt_number = index.next_attempt(&cell_key);
    let attempt_id = attempt_id(
        &index.campaign_id,
        &index.candidate_sha,
        "preflight",
        "identity",
        "none",
        attempt_number,
    );
    let directory = root.join("attempts").join(&attempt_id);
    fs::create_dir(&directory)?;
    let preflight = ArtifactRef::from_path("preflight", root.join("private/preflight.json"))?;
    let mut manifest = base_manifest(
        &index,
        &attempt_id,
        &cell_key,
        attempt_number,
        "preflight",
        "identity",
        "none",
        "none",
        EvidenceStatus::Unavailable,
    )?;
    manifest.artifacts.push(preflight);
    manifest.limitations.push(format!(
        "oracle environment available: {}",
        snapshot.oracle_venv.available
    ));
    write_manifest_pair(&directory, &manifest)?;
    let private =
        ArtifactRef::from_path("terminal-manifest", directory.join("private.manifest.json"))?;
    index.push(CampaignAttemptV1 {
        cell_key,
        attempt_id,
        attempt_number,
        status: EvidenceStatus::Unavailable,
        terminal_manifest: Some(private),
    })?;
    index.write_atomic(root.join(INDEX_FILE))?;
    Ok(())
}

fn run_attempt(root: &Path, args: &RunArgs) -> Result<()> {
    ensure_reserve(root, args.reserve_gib)?;
    let mut index = CampaignIndexV1::read(root.join(INDEX_FILE))?;
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
        attempt_number,
    );
    let directory = root.join("attempts").join(&attempt_id);
    fs::create_dir(&directory)?;
    let started = Instant::now();
    let output = Command::new(&args.command[0])
        .args(&args.command[1..])
        .output()
        .with_context(|| format!("failed to launch {}", args.command[0]))?;
    let elapsed = started.elapsed();
    atomic_write_new(&directory.join("stdout.raw"), &output.stdout)?;
    atomic_write_new(&directory.join("stderr.raw"), &output.stderr)?;

    let status = authoritative_status(&args.phase, output.status.code());
    let mut manifest = base_manifest(
        &index,
        &attempt_id,
        &cell_key,
        attempt_number,
        &args.phase,
        &args.scenario,
        &args.kernel,
        &args.backend,
        status,
    )?;
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
    manifest.artifacts.extend([
        ArtifactRef::from_path("stdout", directory.join("stdout.raw"))?,
        ArtifactRef::from_path("stderr", directory.join("stderr.raw"))?,
    ]);
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

fn resume(root: &Path) -> Result<()> {
    let index = CampaignIndexV1::read(root.join(INDEX_FILE))?;
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
    let terminal = index
        .attempts
        .iter()
        .filter(|attempt| attempt.terminal_manifest.is_some())
        .collect::<Vec<_>>();
    let accepted_c3 = count_phase(
        &terminal,
        "c3",
        &[EvidenceStatus::Pass, EvidenceStatus::InsufficientEvidence],
    );
    let official_comparisons = count_phase(&terminal, "compare", &[EvidenceStatus::Pass]);
    let llama_captures = count_phase(&terminal, "llama", &[EvidenceStatus::InsufficientEvidence]);
    let service_cells = count_phase(&terminal, "service", &[EvidenceStatus::Pass]);
    let performance_cells = count_phase(
        &terminal,
        "performance",
        &[EvidenceStatus::Pass, EvidenceStatus::InsufficientEvidence],
    );
    let complete = accepted_c3 >= 1
        && official_comparisons >= 28
        && llama_captures >= 7
        && service_cells >= 1
        && performance_cells >= 1;
    let mut limitations = Vec::new();
    if !complete {
        limitations.push("campaign acceptance matrix is incomplete".into());
    }
    let summary = FinalSummary {
        schema: "gpt-oss-rs.cpu-validation-final/v1",
        campaign_id: index.campaign_id,
        candidate_sha: index.candidate_sha,
        complete,
        accepted_c3,
        official_comparisons,
        llama_captures,
        service_cells,
        performance_cells,
        terminal_attempts: terminal.len(),
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

fn base_manifest(
    index: &CampaignIndexV1,
    attempt_id: &str,
    cell_key: &str,
    attempt_number: u32,
    phase: &str,
    scenario: &str,
    kernel: &str,
    backend: &str,
    status: EvidenceStatus,
) -> Result<RunManifestV1> {
    let mut manifest = RunManifestV1::new(attempt_id, phase, status);
    manifest.source = source_provenance(&index.candidate_sha);
    manifest.workload = WorkloadEvidence {
        id: scenario.into(),
        prompt_sha256: None,
        seed: 0,
        repetitions: 1,
    };
    manifest.campaign = CampaignIdentity {
        campaign_id: index.campaign_id.clone(),
        candidate_sha: index.candidate_sha.clone(),
        phase: phase.into(),
        scenario: scenario.into(),
        requested_kernel: kernel.into(),
        attempt_number,
        attempt_id: attempt_id.into(),
        cell_key: cell_key.into(),
    };
    manifest.dispatch = DispatchEvidence {
        requested_kernel: kernel.into(),
        effective_kernel: kernel.into(),
        requested_matrix_backend: backend.into(),
        effective_matrix_backend: backend.into(),
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

fn authoritative_status(phase: &str, code: Option<i32>) -> EvidenceStatus {
    match (phase, code) {
        ("compare" | "service", Some(0)) => EvidenceStatus::Pass,
        ("compare" | "service", Some(1)) => EvidenceStatus::Fail,
        (_, Some(0)) => EvidenceStatus::InsufficientEvidence,
        (_, Some(_)) => EvidenceStatus::Invalid,
        (_, None) => EvidenceStatus::Incomplete,
    }
}

fn valid_completed_attempt<'a>(
    index: &'a CampaignIndexV1,
    cell_key: &str,
) -> Result<Option<&'a CampaignAttemptV1>> {
    for attempt in index.attempts.iter().rev() {
        if attempt.cell_key == cell_key
            && attempt.status != EvidenceStatus::Incomplete
            && attempt.terminal_manifest.is_some()
        {
            attempt
                .terminal_manifest
                .as_ref()
                .expect("checked terminal")
                .verify()?;
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

fn attempt_id(
    campaign: &str,
    candidate: &str,
    phase: &str,
    scenario: &str,
    kernel: &str,
    number: u32,
) -> String {
    format!(
        "{campaign}--{}--{phase}--{scenario}--{kernel}--{number}",
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

fn dependency_check(venv: &Path) -> DependencyCheck {
    let python = venv.join("bin/python");
    if !python.is_file() {
        return DependencyCheck {
            path: venv.display().to_string(),
            available: false,
            python: None,
            fingerprint: None,
            reason: Some("required pinned interpreter is absent".into()),
        };
    }
    let script = "import hashlib,importlib.metadata,json,sys,torch; d={'python':sys.version,'packages':sorted((x.metadata['Name'],x.version) for x in importlib.metadata.distributions()),'torch':torch.__config__.show()}; print(hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest())";
    let output = Command::new(&python).args(["-c", script]).output();
    match output {
        Ok(output) if output.status.success() => DependencyCheck {
            path: venv.display().to_string(),
            available: true,
            python: command_text(python.to_string_lossy().as_ref(), &["--version"]).ok(),
            fingerprint: String::from_utf8(output.stdout)
                .ok()
                .map(|value| value.trim().to_string()),
            reason: None,
        },
        Ok(output) => DependencyCheck {
            path: venv.display().to_string(),
            available: false,
            python: None,
            fingerprint: None,
            reason: Some(String::from_utf8_lossy(&output.stderr).trim().to_string()),
        },
        Err(error) => DependencyCheck {
            path: venv.display().to_string(),
            available: false,
            python: None,
            fingerprint: None,
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
            authoritative_status("native", Some(0)),
            EvidenceStatus::InsufficientEvidence
        );
        assert_eq!(
            authoritative_status("official", Some(0)),
            EvidenceStatus::InsufficientEvidence
        );
        assert_eq!(
            authoritative_status("llama", Some(0)),
            EvidenceStatus::InsufficientEvidence
        );
        assert_eq!(
            authoritative_status("compare", Some(0)),
            EvidenceStatus::Pass
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
            3,
        );
        assert_eq!(
            id,
            "cpu-validation-1--aaaaaaaaaaaa--official--harmony_262--avx2--3"
        );
    }
}
