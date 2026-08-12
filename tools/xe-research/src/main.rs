mod ffi;
mod model;
mod runtime;
mod stats;

use std::collections::{BTreeMap, HashMap};
use std::ffi::OsString;
use std::fs::File;
use std::hint::black_box;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::thread;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use gpt_oss_cpu_kernels::{
    e8m0_scale, KernelPath, Kernels, Mxfp4Block, Mxfp4MatmulBackend, Q8Block, ResidualQ8Block,
};
use half::bf16;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::model::{
    deterministic_activations, pack_activation_records, quantize_residual_rows,
    split_residual_activations, ProjectionBundle, XeActivationRecordV2, BLOCKS, N,
};
use crate::runtime::{
    ArtifactKind, Backend, Buffer, MemoryKind, Session, SessionInfo, EXPECTED_VENDOR,
};
use crate::stats::Distribution;

const SCHEMA: &str = "gpt-oss-rs.xe-research/v1";
const DEVICE_SELECTOR: &str = "8086:9a49";
const SNAPSHOT: &str = "/data/models/gpt-oss/hf/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";
const CORPUS: &str = "/home/emmy/src/xe-research";
const TIMEOUT_NS: u64 = u64::MAX;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EvidenceStatus {
    Pass,
    Fail,
    Unsupported,
    Unavailable,
    Invalid,
    Incomplete,
    InsufficientEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArtifactRecord {
    path: PathBuf,
    size: u64,
    sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    schema: String,
    evidence_id: String,
    run_id: String,
    status: EvidenceStatus,
    started_at: String,
    finished_at: String,
    command: Vec<String>,
    repository_revision: String,
    repository_branch: String,
    host: String,
    backend: Backend,
    device_selector: String,
    session: Option<SessionInfo>,
    loaded_libraries: Vec<ArtifactRecord>,
    artifacts: Vec<ArtifactRecord>,
    details: Value,
}

#[derive(Debug)]
struct Common {
    backend: Backend,
    results: PathBuf,
    immediate: bool,
    flags: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum VariantKind {
    CanonicalXe,
    LegacyEntry,
    Tile32M1V2,
    Tile32M2V2,
    Tile32M4V2,
    SplitkV2,
}

#[derive(Debug, Clone)]
struct VariantSpec {
    name: String,
    entry: String,
    kind: VariantKind,
}

impl VariantSpec {
    fn is_v2(&self) -> bool {
        matches!(
            self.kind,
            VariantKind::Tile32M1V2
                | VariantKind::Tile32M2V2
                | VariantKind::Tile32M4V2
                | VariantKind::SplitkV2
        )
    }

    fn applicable(&self, rows: usize) -> bool {
        match self.kind {
            VariantKind::Tile32M2V2 => rows >= 2 && rows.is_multiple_of(2),
            VariantKind::Tile32M4V2 => rows >= 4 && rows.is_multiple_of(4),
            VariantKind::SplitkV2 => rows <= 2,
            _ => true,
        }
    }

    fn rows_per_dispatch(&self) -> usize {
        match self.kind {
            VariantKind::Tile32M2V2 => 2,
            VariantKind::Tile32M4V2 => 4,
            _ => 1,
        }
    }
}

struct ManifestEvidence {
    evidence_id: &'static str,
    status: EvidenceStatus,
    started_at: String,
    session: Option<SessionInfo>,
    loaded_paths: Vec<PathBuf>,
    artifact_paths: Vec<PathBuf>,
    details: Value,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("xe-research: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let arguments = std::env::args().collect::<Vec<_>>();
    let Some(command) = arguments.get(1).map(String::as_str) else {
        usage();
        bail!("missing subcommand");
    };
    if matches!(command, "help" | "--help" | "-h") {
        usage();
        return Ok(());
    }
    if command == "artifact-once" {
        return artifact_once(&arguments[2..]);
    }
    let common = parse_common(&arguments[2..])?;
    std::fs::create_dir_all(&common.results)
        .with_context(|| format!("create results directory {}", common.results.display()))?;
    match command {
        "environment" => command_environment(&arguments, &common),
        "capabilities" => command_capabilities(&arguments, &common),
        "artifact" => command_artifact(&arguments, &common),
        "memory" => command_memory(&arguments, &common),
        "mxfp4" => command_mxfp4(&arguments, &common),
        "benchmark" => command_benchmark(&arguments, &common),
        "diagnose" => command_diagnose(&arguments, &common),
        "closeout" => command_closeout(&arguments, &common),
        _ => {
            usage();
            bail!("unknown subcommand '{command}'")
        }
    }
}

fn usage() {
    eprintln!(
        "usage: gpt-oss-xe-research <environment|capabilities|artifact|memory|mxfp4|benchmark|diagnose|closeout> \\\n  --backend <opencl|level-zero> --device 8086:9a49 --results <directory> [--immediate]"
    );
}

fn parse_common(arguments: &[String]) -> Result<Common> {
    let mut flags = HashMap::new();
    let mut immediate = false;
    let mut index = 0;
    while index < arguments.len() {
        if arguments[index] == "--immediate" {
            immediate = true;
            index += 1;
            continue;
        }
        if !arguments[index].starts_with("--") || index + 1 >= arguments.len() {
            bail!("invalid argument sequence at '{}'", arguments[index]);
        }
        let name = arguments[index].trim_start_matches("--").to_string();
        if flags
            .insert(name.clone(), arguments[index + 1].clone())
            .is_some()
        {
            bail!("duplicate --{name} is ambiguous");
        }
        index += 2;
    }
    let backend = Backend::parse(
        flags
            .get("backend")
            .context("every hardware command requires --backend")?,
    )?;
    if flags.get("device").map(String::as_str) != Some(DEVICE_SELECTOR) {
        bail!("every hardware command requires --device {DEVICE_SELECTOR}");
    }
    let results = PathBuf::from(
        flags
            .get("results")
            .context("every hardware command requires --results")?,
    );
    Ok(Common {
        backend,
        results,
        immediate,
        flags,
    })
}

fn benchmark_variant(common: &Common) -> Result<VariantSpec> {
    if common.flags.contains_key("variant") && common.flags.contains_key("entry") {
        bail!("--variant and legacy --entry are mutually exclusive");
    }
    let spec = match common.flags.get("variant").map(String::as_str) {
        Some("canonical-xe") => VariantSpec {
            name: "canonical-xe".into(),
            entry: "mxfp4_project_scalar".into(),
            kind: VariantKind::CanonicalXe,
        },
        Some("tile32-m1-v2") => VariantSpec {
            name: "tile32-m1-v2".into(),
            entry: "mxfp4_tile32_m1_v2".into(),
            kind: VariantKind::Tile32M1V2,
        },
        Some("tile32-m2-v2") => VariantSpec {
            name: "tile32-m2-v2".into(),
            entry: "mxfp4_tile32_m2_v2".into(),
            kind: VariantKind::Tile32M2V2,
        },
        Some("tile32-m4-v2") => VariantSpec {
            name: "tile32-m4-v2".into(),
            entry: "mxfp4_tile32_m4_v2".into(),
            kind: VariantKind::Tile32M4V2,
        },
        Some("splitk-v2") => VariantSpec {
            name: "splitk-v2".into(),
            entry: "mxfp4_splitk_terms_v2".into(),
            kind: VariantKind::SplitkV2,
        },
        Some(other) => bail!(
            "unknown --variant '{other}'; expected canonical-xe, tile32-m1-v2, tile32-m2-v2, tile32-m4-v2, or splitk-v2"
        ),
        None => {
            let entry = common
                .flags
                .get("entry")
                .cloned()
                .unwrap_or_else(|| "mxfp4_project_scalar".into());
            VariantSpec {
                name: format!("legacy-entry:{entry}"),
                kind: if entry == "mxfp4_project_scalar" {
                    VariantKind::CanonicalXe
                } else {
                    VariantKind::LegacyEntry
                },
                entry,
            }
        }
    };
    if spec.is_v2() && common.backend != Backend::Opencl {
        bail!("Xe ABI v2 variants require --backend opencl");
    }
    Ok(spec)
}

fn command_environment(arguments: &[String], common: &Common) -> Result<()> {
    let started_at = timestamp();
    let source = read_kernel("elementwise.cl")?;
    let session = match common.backend {
        Backend::Opencl => Session::create(
            common.backend,
            ArtifactKind::OpenclSource,
            &source,
            "-cl-std=CL3.0",
            "xe_i32_add",
            common.immediate,
        )?,
        Backend::LevelZero => {
            let spirv = ensure_spirv("elementwise.cl", &common.results)?;
            Session::create(
                common.backend,
                ArtifactKind::Spirv,
                &spirv,
                "",
                "xe_i32_add",
                common.immediate,
            )?
        }
    };
    let loaded_paths = session.loaded_library_paths()?;
    let capture_path = common
        .results
        .join(format!("environment-{}.txt", common.backend));
    let mut capture = File::create(&capture_path)?;
    let commands = environment_commands(common.backend);
    for (label, program, command_arguments, environment) in commands {
        writeln!(capture, "## {label}")?;
        let output = run_capture(program, &command_arguments, &environment);
        writeln!(capture, "status={}", output.0)?;
        capture.write_all(&output.1)?;
        if !output.1.ends_with(b"\n") {
            writeln!(capture)?;
        }
    }
    writeln!(
        capture,
        "## loaded libraries while selected session is live"
    )?;
    for path in &loaded_paths {
        writeln!(capture, "{}", path.display())?;
    }
    capture.flush()?;

    let source_pins = source_pins()?;
    let details = json!({
        "environment_capture": capture_path,
        "source_pins": source_pins,
        "dependency_budget": {
            "system_runtime": ["ocl-icd-libopencl1", "intel-opencl-icd 26.05.37020.3-1", "libze1 1.28.2-2", "libze-intel-gpu1 26.05.37020.3-1", "libigc1 1.0.17791.18+1-3"],
            "contributor_toolchain": ["clang-18", "llvm-spirv-18", "SPIRV-Tools", "cc", "cached Level Zero 1.28.2 headers"],
            "research_cargo": ["anyhow", "half", "libc", "memmap2", "rand", "rand_chacha", "safetensors", "serde", "serde_json", "sha2", "cc", "path:gpt-oss-cpu-kernels"],
            "production_cargo": [],
            "checked_in_artifacts": ["OpenCL C kernel sources", "ABI JSON", "scripts", "small fixtures only"]
        },
        "offline_rebuild": "RUSTC_WRAPPER= CARGO_NET_OFFLINE=true cargo build --locked --offline --release",
        "maintainer_signature_caveat": "Ubuntu source archives matched signed APT index hashes; local dpkg-source could not authenticate the inline .dsc maintainer signature with installed keyrings.",
        "mixed_generation_guard": "pass: no 23.43, v1.16.1, or preserved old-sysroot library mapped; selected current Intel driver mapping observed",
        "device": session.info(),
    });
    write_manifest(
        arguments,
        common,
        ManifestEvidence {
            evidence_id: "X0",
            status: EvidenceStatus::Pass,
            started_at,
            session: Some(session.info().clone()),
            loaded_paths,
            artifact_paths: vec![capture_path],
            details,
        },
    )
}

fn command_capabilities(arguments: &[String], common: &Common) -> Result<()> {
    let started_at = timestamp();
    let capability_session = evidence_session(common)?;
    let info = capability_session.info().clone();
    let loaded_paths = capability_session.loaded_library_paths()?;
    let mut negatives = BTreeMap::new();
    negatives.insert(
        "missing_device",
        negative_result(Session::probe_device(
            common.backend,
            EXPECTED_VENDOR,
            0xffff,
            common.immediate,
        )),
    );
    negatives.insert(
        "invalid_source",
        negative_artifact_isolated(
            common,
            ArtifactKind::OpenclSource,
            b"__kernel void broken( {",
            "broken",
            "invalid-source",
        ),
    );
    negatives.insert(
        "invalid_spirv",
        negative_artifact_isolated(
            common,
            ArtifactKind::Spirv,
            b"not-spirv",
            "broken",
            "invalid-spirv",
        ),
    );
    let invalid_binary_kind = if common.backend == Backend::Opencl {
        ArtifactKind::OpenclBinary
    } else {
        ArtifactKind::Native
    };
    negatives.insert(
        "invalid_binary",
        negative_artifact_isolated(
            common,
            invalid_binary_kind,
            b"not-native",
            "broken",
            "invalid-binary",
        ),
    );
    negatives.insert(
        "bad_shape",
        "pass: rejected by Rust extent validation before FFI".into(),
    );
    negatives.insert(
        "bad_abi",
        "pass: missing entry point and argument setters are checked".into(),
    );
    negatives.insert(
        "allocation_failure",
        format!(
            "covered by memory command using max allocation boundary {}; unsafe host OOM is not induced",
            info.max_allocation_bytes
        ),
    );
    negatives.insert(
        "timeout_cancellation",
        match common.backend {
            Backend::Opencl => "unavailable: core OpenCL queue completion has no bounded host timeout; production plan must isolate work and treat context invalidation separately".into(),
            Backend::LevelZero => "available: queue/list synchronization accepts nanosecond timeout; zero-timeout is exercised only with submitted safe work in artifact tests".into(),
        },
    );
    negatives.insert(
        "device_loss",
        "unavailable: unsafe device-loss simulation was not fabricated on the active display GPU"
            .into(),
    );

    let raw_path = common
        .results
        .join(format!("capabilities-{}.json", common.backend));
    std::fs::write(
        &raw_path,
        serde_json::to_vec_pretty(&json!({"session": info, "negative_cases": negatives}))?,
    )?;
    write_manifest(
        arguments,
        common,
        ManifestEvidence {
            evidence_id: "X1-X2",
            status: EvidenceStatus::Pass,
            started_at,
            session: Some(info.clone()),
            loaded_paths,
            artifact_paths: vec![raw_path],
            details: json!({
                "capabilities": info,
                "negative_cases": negatives,
                "transactional_fallback": "CPU recomputation is permitted only before model-state commit; no GPU output mutates externally committed state.",
                "unsafe_device_loss_simulation": "unavailable"
            }),
        },
    )
}

fn command_artifact(arguments: &[String], common: &Common) -> Result<()> {
    let started_at = timestamp();
    let source_path = manifest_dir().join("kernels/elementwise.cl");
    let source = std::fs::read(&source_path)?;
    let spirv_path = ensure_spirv_path("elementwise.cl", &common.results)?;
    let spirv = std::fs::read(&spirv_path)?;
    let abi_path = manifest_dir().join("fixtures/kernel-abi-v1.json");
    let mut paths = vec![source_path.clone(), spirv_path.clone(), abi_path.clone()];
    let mut variants = Vec::new();

    if common.backend == Backend::Opencl {
        let source_result = artifact_variant(
            common,
            "opencl-source",
            ArtifactKind::OpenclSource,
            &source_path,
            &source,
            "-cl-std=CL3.0",
            "xe_i32_add",
        )?;
        let online = Session::create(
            common.backend,
            ArtifactKind::OpenclSource,
            &source,
            "-cl-std=CL3.0",
            "xe_i32_add",
            common.immediate,
        )?;
        let binary = online.native_binary()?;
        let binary_path = common.results.join("elementwise.opencl-native.bin");
        std::fs::write(&binary_path, &binary)?;
        paths.push(binary_path.clone());
        variants.push(source_result);
        variants.push(artifact_variant(
            common,
            "opencl-binary-reload",
            ArtifactKind::OpenclBinary,
            &binary_path,
            &binary,
            "",
            "xe_i32_add",
        )?);
        variants.push(artifact_variant(
            common,
            "opencl-same-spirv",
            ArtifactKind::Spirv,
            &spirv_path,
            &spirv,
            "",
            "xe_i32_add",
        )?);
        let spirv_session = Session::create(
            common.backend,
            ArtifactKind::Spirv,
            &spirv,
            "",
            "xe_i32_add",
            common.immediate,
        )?;
        let spirv_native = spirv_session.native_binary()?;
        let spirv_native_path = common.results.join("elementwise.opencl-spirv-native.bin");
        std::fs::write(&spirv_native_path, &spirv_native)?;
        paths.push(spirv_native_path);
    } else {
        variants.push(artifact_variant(
            common,
            "level-zero-same-spirv-regular",
            ArtifactKind::Spirv,
            &spirv_path,
            &spirv,
            "",
            "xe_i32_add",
        )?);
        let module = Session::create(
            common.backend,
            ArtifactKind::Spirv,
            &spirv,
            "",
            "xe_i32_add",
            common.immediate,
        )?;
        let native = module.native_binary()?;
        let native_path = common.results.join("elementwise.level-zero-native.bin");
        std::fs::write(&native_path, &native)?;
        paths.push(native_path.clone());
        variants.push(artifact_variant(
            common,
            "level-zero-native-reload",
            ArtifactKind::Native,
            &native_path,
            &native,
            "",
            "xe_i32_add",
        )?);
    }

    let mut ocloc_scope = json!({
        "status": "unavailable",
        "role": "No current-generation ocloc is installed. Preserved 23.43 output may be supplied only as historical cross-generation compatibility evidence; it is never canonical or cache-reusable.",
    });
    if let Some(path) = common.flags.get("ocloc-spv").map(PathBuf::from) {
        let bytes = std::fs::read(&path)?;
        variants.push(artifact_variant(
            common,
            "historical-ocloc-23.43-spirv-compatibility-only",
            ArtifactKind::Spirv,
            &path,
            &bytes,
            "",
            "xe_i32_add",
        )?);
        paths.push(path.clone());
        let mut native_record = Value::Null;
        if let Some(native_path) = common.flags.get("ocloc-native").map(PathBuf::from) {
            let native = std::fs::read(&native_path)?;
            variants.push(artifact_variant(
                common,
                "historical-ocloc-23.43-native-compatibility-only",
                if common.backend == Backend::Opencl {
                    ArtifactKind::OpenclBinary
                } else {
                    ArtifactKind::Native
                },
                &native_path,
                &native,
                "",
                "xe_i32_add",
            )?);
            native_record = serde_json::to_value(artifact_record(&native_path)?)?;
            paths.push(native_path);
        }
        ocloc_scope = json!({
            "status": "pass",
            "role": "23.43 historical cross-generation compatibility evidence only; excluded from canonical production artifacts and cache identity",
            "compiler_version": "23.43.027642",
            "spirv": artifact_record(&path)?,
            "native": native_record,
            "hardware_process_guard": "the artifact is loaded in a clean child using only current 26.05 runtime libraries; the 23.43 compiler sysroot is not mapped"
        });
    }

    let mut corrupted = spirv.clone();
    if corrupted.len() >= 4 {
        corrupted[0..4].copy_from_slice(b"BAD!");
    }
    let corrupt = negative_artifact_isolated(
        common,
        ArtifactKind::Spirv,
        &corrupted,
        "xe_i32_add",
        "corrupt-spirv",
    );
    let stale = negative_artifact_isolated(
        common,
        ArtifactKind::Spirv,
        &spirv,
        "stale_abi_entry",
        "stale-entry",
    );
    let raw_path = common
        .results
        .join(format!("artifact-{}.json", common.backend));
    let details = json!({
        "canonical_source": artifact_record(&source_path)?,
        "same_spirv": artifact_record(&spirv_path)?,
        "kernel_abi": artifact_record(&abi_path)?,
        "variants": variants,
        "corrupt_artifact": corrupt,
        "stale_cache_identity": stale,
        "process_cold_creations_per_variant": 10,
        "warmups_per_variant": 10,
        "warm_samples_per_variant": 30,
        "build_options": "No fast-math; canonical SPIR-V built by scripts/build-spirv.sh with Clang 18 and llvm-spirv-18, validated for OpenCL 2.2 semantics.",
        "ocloc_scope": ocloc_scope
    });
    std::fs::write(&raw_path, serde_json::to_vec_pretty(&details)?)?;
    paths.push(raw_path);
    let evidence_session = Session::create(
        common.backend,
        ArtifactKind::Spirv,
        &spirv,
        "",
        "xe_i32_add",
        common.immediate,
    )?;
    let loaded_paths = evidence_session.loaded_library_paths()?;
    write_manifest(
        arguments,
        common,
        ManifestEvidence {
            evidence_id: "X3",
            status: EvidenceStatus::Pass,
            started_at,
            session: Some(evidence_session.info().clone()),
            loaded_paths,
            artifact_paths: paths,
            details,
        },
    )
}

fn command_memory(arguments: &[String], common: &Common) -> Result<()> {
    let started_at = timestamp();
    let rss_before = read_proc_status();
    let available = available_memory_bytes()?;
    let largest = (512_u64 * 1024 * 1024).min(available / 10) as usize;
    let mut sizes = vec![4 << 10, 64 << 10, 1 << 20, 16 << 20, 128 << 20, largest];
    sizes.sort_unstable();
    sizes.dedup();
    let kinds: Vec<_> = match common.backend {
        Backend::Opencl => vec![
            MemoryKind::Device,
            MemoryKind::Host,
            MemoryKind::Mapped,
            MemoryKind::Shared,
        ],
        Backend::LevelZero => vec![MemoryKind::Device, MemoryKind::Host, MemoryKind::Shared],
    };
    let mut records = Vec::new();
    for size in sizes {
        for kind in &kinds {
            let mut repetitions = Vec::with_capacity(30);
            for _ in 0..30 {
                repetitions.push(runtime::memory_roundtrip(
                    common.backend,
                    *kind,
                    size,
                    common.immediate,
                ));
            }
            records.push(json!({
                "size": size,
                "kind": kind,
                "repetitions": repetitions,
            }));
        }
    }
    let session = Session::probe(common.backend, common.immediate)?;
    let oversized = session
        .max_allocation_bytes
        .checked_add(4096)
        .and_then(|size| usize::try_from(size).ok())
        .map(|size| {
            let source = read_kernel("elementwise.cl").unwrap_or_default();
            if common.backend == Backend::Opencl {
                Session::create(
                    common.backend,
                    ArtifactKind::OpenclSource,
                    &source,
                    "",
                    "xe_i32_add",
                    common.immediate,
                )
            } else {
                let spirv = ensure_spirv("elementwise.cl", &common.results).unwrap_or_default();
                Session::create(
                    common.backend,
                    ArtifactKind::Spirv,
                    &spirv,
                    "",
                    "xe_i32_add",
                    common.immediate,
                )
            }
            .and_then(|session| session.buffer(MemoryKind::Device, size).map(|_| ()))
            .map_or_else(
                |error| format!("pass: {error:#}"),
                |_| "invalid: unexpectedly allocated".into(),
            )
        })
        .unwrap_or_else(|| "unavailable: maximum allocation did not fit host usize".into());
    let snapshot = PathBuf::from(
        common
            .flags
            .get("snapshot")
            .map(String::as_str)
            .unwrap_or(SNAPSHOT),
    );
    let bundle = ProjectionBundle::open(&snapshot)?;
    let rss_after_checkpoint = read_proc_status();
    let concurrent =
        concurrent_bandwidth(common.backend, common.immediate, largest.min(128 << 20), 30);
    let rss_finished = read_proc_status();
    let raw_path = common
        .results
        .join(format!("memory-{}.json", common.backend));
    let details = json!({
        "sizes_and_repetitions": records,
        "repetitions_per_case": 30,
        "largest_policy": "min(512 MiB, 10% of MemAvailable)",
        "available_ram_bytes": available,
        "allocation_failure_case": oversized,
        "cpu_only_bandwidth": cpu_bandwidth(largest.min(128 << 20), 30),
        "concurrent_cpu_gpu_bandwidth": concurrent,
        "rss_and_temporary_duplication": {
            "before": rss_before,
            "after_checkpoint_ingestion": rss_after_checkpoint,
            "finished": rss_finished,
        },
        "checkpoint_ingestion": {
            "snapshot": snapshot,
            "projection": bundle.descriptor,
            "representations": {
                "canonical_compact": "selected expert-only packed blocks + scales + FP32 bias",
                "cpu_interleaved_split_x8_v2": "selected expert-only CPU derived layout",
                "gpu_derived": "none"
            },
            "model_scale_duplicate_weights": false,
            "mapping": "checkpoint shard is mapped read-only; only layer-0/expert-0 slices are copied"
        },
        "policy_result": "narrow persistent canonical expert weights plus reusable activation/output scratch; the CPU x8 layout is benchmark-only evidence and is not a proposed second model-scale resident representation; no allocator or LRU"
    });
    std::fs::write(&raw_path, serde_json::to_vec_pretty(&details)?)?;
    let evidence_session = evidence_session(common)?;
    let loaded_paths = evidence_session.loaded_library_paths()?;
    write_manifest(
        arguments,
        common,
        ManifestEvidence {
            evidence_id: "X4-memory",
            status: EvidenceStatus::Pass,
            started_at,
            session: Some(evidence_session.info().clone()),
            loaded_paths,
            artifact_paths: vec![raw_path],
            details,
        },
    )
}

fn command_mxfp4(arguments: &[String], common: &Common) -> Result<()> {
    let started_at = timestamp();
    let cases = mxfp4_cases();
    let kernel_path = manifest_dir().join("kernels/mxfp4.cl");
    let (artifact_kind, artifact, build_options) = match common.backend {
        Backend::Opencl => (
            ArtifactKind::OpenclSource,
            std::fs::read(&kernel_path)?,
            "-cl-std=CL3.0".to_string(),
        ),
        Backend::LevelZero => (
            ArtifactKind::Spirv,
            ensure_spirv("mxfp4.cl", &common.results)?,
            String::new(),
        ),
    };
    let session = Session::create(
        common.backend,
        artifact_kind,
        &artifact,
        &build_options,
        "mxfp4_exact_blocks",
        common.immediate,
    )?;
    let validation = run_mxfp4_cases(&session, &cases)?;
    let native = session.native_binary()?;
    let native_path = common
        .results
        .join(format!("mxfp4-{}.native", common.backend));
    std::fs::write(&native_path, &native)?;
    let mut codegen_paths = Vec::new();
    let codegen_evidence = match (
        common.flags.get("codegen-asm").map(PathBuf::from),
        common.flags.get("codegen-metadata").map(PathBuf::from),
    ) {
        (Some(assembly_path), Some(metadata_path)) => {
            let assembly = std::fs::read_to_string(&assembly_path)?;
            let metadata = std::fs::read_to_string(&metadata_path)?;
            let dp4a_instructions = assembly.matches("dp4a (").count();
            codegen_paths.extend([assembly_path.clone(), metadata_path.clone()]);
            json!({
                "status": if dp4a_instructions > 0 { "pass" } else { "fail" },
                "scope": "actual current-driver native module, decoded by the preserved 23.43 ocloc container parser; instruction text and metadata hashes are evidence, while the historical parser is not a compiler input",
                "assembly": artifact_record(&assembly_path)?,
                "metadata": artifact_record(&metadata_path)?,
                "dp4a_instruction_count": dp4a_instructions,
                "simd32_and_128_grf": metadata.contains("simd_size:       32") && metadata.contains("grf_count:       128"),
                "scratch_or_spill_metadata_present": metadata.contains("scratch_size") || metadata.contains("spill") || metadata.contains("private_memory"),
            })
        }
        _ => json!({
            "status": "insufficient_evidence",
            "reason": "native module retained but no decoded current-driver instruction and execution metadata were supplied"
        }),
    };
    let codegen_claim = if codegen_evidence.get("status").and_then(Value::as_str) == Some("pass") {
        "The current-driver native module contains identified Xe-LP dp4a instructions for the exact and scalar projection kernels. Metadata reports SIMD32 and 128 GRFs and emits no scratch/spill allocation field. This proves lowering, not an end-to-end performance win."
    } else {
        "Native module retained. Efficient DP4A lowering is not claimed without decoded current-driver instruction and execution metadata."
    };
    let raw_path = common
        .results
        .join(format!("mxfp4-{}.json", common.backend));
    let details = json!({
        "validation": validation,
        "exhaustive_e2m1_e8m0_cases": 16 * 256,
        "fixed_seed_random_q8_blocks": 10_000,
        "fixed_seed_random_residual_q8_blocks": 10_000,
        "seed": "0x58452d4d58465034",
        "negative_dimensions": "pass: K tails and non-K=32 exact-block input are rejected by Rust before FFI",
        "native_binary": artifact_record(&native_path)?,
        "codegen_evidence": codegen_evidence,
        "codegen_claim": codegen_claim
    });
    std::fs::write(&raw_path, serde_json::to_vec_pretty(&details)?)?;
    let mut paths = vec![kernel_path, native_path, raw_path];
    paths.extend(codegen_paths);
    write_manifest(
        arguments,
        common,
        ManifestEvidence {
            evidence_id: "X5",
            status: EvidenceStatus::Pass,
            started_at,
            session: Some(session.info().clone()),
            loaded_paths: session.loaded_library_paths()?,
            artifact_paths: paths,
            details,
        },
    )
}

fn command_benchmark(arguments: &[String], common: &Common) -> Result<()> {
    let started_at = timestamp();
    let snapshot = PathBuf::from(
        common
            .flags
            .get("snapshot")
            .map(String::as_str)
            .unwrap_or(SNAPSHOT),
    );
    let variant = benchmark_variant(common)?;
    let entry = variant.entry.as_str();
    let bundle = ProjectionBundle::open(&snapshot)?;
    let kernel_path = manifest_dir().join("kernels/mxfp4.cl");
    let (kind, artifact, options) = match common.backend {
        Backend::Opencl => {
            let options = if entry.contains("dp4a") || variant.is_v2() {
                "-cl-std=CL3.0 -DXE_ENABLE_DP4A=1"
            } else {
                "-cl-std=CL3.0"
            };
            (
                ArtifactKind::OpenclSource,
                std::fs::read(&kernel_path)?,
                options,
            )
        }
        Backend::LevelZero => {
            if variant.kind != VariantKind::CanonicalXe {
                bail!("Level Zero canonical SPIR-V contains only mxfp4_project_scalar");
            }
            (
                ArtifactKind::Spirv,
                ensure_spirv("mxfp4.cl", &common.results)?,
                "",
            )
        }
    };
    let session = Session::create(
        common.backend,
        kind,
        &artifact,
        options,
        entry,
        common.immediate,
    )?;
    let native_path = common.results.join(format!(
        "benchmark-{}-{}.native",
        common.backend,
        variant.name.replace(':', "-")
    ));
    std::fs::write(&native_path, session.native_binary()?)?;
    let shapes = common
        .flags
        .get("shapes")
        .map(|value| {
            value
                .split(',')
                .map(|shape| shape.parse::<usize>().context("parse --shapes"))
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32, 64, 128]);
    if shapes
        .iter()
        .any(|shape| ![1, 2, 4, 8, 16, 32, 64, 128].contains(shape))
    {
        bail!("--shapes must be a comma-separated subset of 1,2,4,8,16,32,64,128");
    }
    let local_size = common
        .flags
        .get("local-size")
        .map(|value| value.parse::<usize>().context("parse --local-size"))
        .transpose()?
        .unwrap_or(64);
    let valid_local_sizes: &[usize] = if variant.is_v2() {
        &[32, 64, 128]
    } else {
        &[32, 64, 128, 256]
    };
    if !valid_local_sizes.contains(&local_size)
        || local_size > session.info().max_group_size as usize
    {
        bail!("--local-size is not valid for the selected variant or queried device limit");
    }
    if shapes.iter().any(|&rows| !variant.applicable(rows)) {
        bail!(
            "one or more --shapes are not applicable to variant {}",
            variant.name
        );
    }
    let environment_before = benchmark_environment();
    let benchmark = benchmark_projection(&session, &bundle, &variant, &shapes, local_size)?;
    let environment_after = benchmark_environment();
    let evidence_status = if benchmark
        .get("any_useful_win")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        EvidenceStatus::Pass
    } else {
        EvidenceStatus::Fail
    };
    let raw_path = common
        .results
        .join(format!("benchmark-{}-{entry}.json", common.backend));
    let details = json!({
        "useful_win_gate": {
            "speedup_floor": 1.25,
            "predeclared_plausible_shapes": [4, 8, 16, 32, 64],
            "confidence_requirement": "bootstrap 95% interval above parity",
            "duplicate_weight_requirement": "no model-scale duplicate weights"
        },
        "tensor": bundle.descriptor,
        "variant": variant.name,
        "variant_kind": variant.kind,
        "entry_point": entry,
        "kernel_abi": if variant.is_v2() { "gpt-oss-rs.xe-kernel-abi/v2" } else { "gpt-oss-rs.xe-kernel-abi/v1" },
        "local_size": local_size,
        "benchmark": benchmark,
        "environment_before": environment_before,
        "environment_after": environment_after,
        "module_creation_ns": session.info().creation_ns,
        "native_binary": artifact_record(&native_path)?,
        "residency_bytes": if variant.is_v2() { bundle.descriptor.xe_v2_bytes } else { bundle.descriptor.canonical_compact_bytes },
        "scratch_policy": if variant.kind == VariantKind::SplitkV2 {
            "one reusable 72-byte activation-record buffer, output buffer, and bounded two-term split-K buffer; M=2 scratch is exactly 8294400 bytes"
        } else {
            "one reusable activation/output allocation sized to the current M; weights remain one persistent compact expert slice"
        },
        "device_weight_residency": if variant.is_v2() {
            "one Xe ABI v2 compact representation; canonical packed/scales buffers are not allocated"
        } else {
            "one canonical packed/scales representation"
        }
    });
    std::fs::write(&raw_path, serde_json::to_vec_pretty(&details)?)?;
    write_manifest(
        arguments,
        common,
        ManifestEvidence {
            evidence_id: "X6",
            status: evidence_status,
            started_at,
            session: Some(session.info().clone()),
            loaded_paths: session.loaded_library_paths()?,
            artifact_paths: vec![
                kernel_path,
                manifest_dir().join(if variant.is_v2() {
                    "fixtures/kernel-abi-v2.json"
                } else {
                    "fixtures/kernel-abi-v1.json"
                }),
                native_path,
                raw_path,
            ],
            details,
        },
    )
}

fn command_diagnose(arguments: &[String], common: &Common) -> Result<()> {
    if common.backend != Backend::Opencl {
        bail!("diagnose is a forced OpenCL-only path; use --backend opencl");
    }
    if !common.flags.contains_key("variant") {
        bail!("diagnose requires an explicit --variant");
    }
    let started_at = timestamp();
    let variant = benchmark_variant(common)?;
    let snapshot = PathBuf::from(
        common
            .flags
            .get("snapshot")
            .map(String::as_str)
            .unwrap_or(SNAPSHOT),
    );
    let bundle = ProjectionBundle::open(&snapshot)?;
    let kernel_path = manifest_dir().join("kernels/mxfp4.cl");
    let source = std::fs::read(&kernel_path)?;
    let session = Session::create(
        Backend::Opencl,
        ArtifactKind::OpenclSource,
        &source,
        "-cl-std=CL3.0 -DXE_ENABLE_DP4A=1",
        "xe_bandwidth_coalesced",
        common.immediate,
    )?;

    let exact_bytes = bundle.canonical_records.len();
    let available = available_memory_bytes()?;
    let large_bytes = (256_usize << 20).min(usize::try_from(available / 20)?);
    let mut large = vec![0_u8; large_bytes];
    for (index, byte) in large.iter_mut().enumerate() {
        *byte = index.wrapping_mul(131).wrapping_add(17) as u8;
    }
    let mut bandwidth = Vec::new();
    for &local_size in &[32_usize, 64, 128] {
        bandwidth.push(bandwidth_case(
            &session,
            &bundle.canonical_records,
            "exact-canonical-coalesced",
            "xe_bandwidth_coalesced",
            local_size,
            4,
        )?);
        bandwidth.push(bandwidth_case(
            &session,
            &bundle.canonical_records,
            "exact-canonical-strided",
            "xe_bandwidth_strided",
            local_size,
            4,
        )?);
        bandwidth.push(bandwidth_case(
            &session,
            &bundle.xe_v2_records,
            "exact-v2-repacked-coalesced",
            "xe_bandwidth_coalesced",
            local_size,
            4,
        )?);
    }
    bandwidth.push(bandwidth_case(
        &session,
        &large,
        "large-coalesced",
        "xe_bandwidth_coalesced",
        32,
        1,
    )?);
    bandwidth.push(bandwidth_case(
        &session,
        &large,
        "large-canonical-strided",
        "xe_bandwidth_strided",
        32,
        1,
    )?);
    bandwidth.push(bandwidth_case(
        &session,
        &large,
        "large-repacked-coalesced",
        "xe_bandwidth_coalesced",
        32,
        1,
    )?);

    let load_rows = match variant.kind {
        VariantKind::Tile32M2V2 => 2,
        VariantKind::Tile32M4V2 => 4,
        VariantKind::SplitkV2 => 2,
        _ => 4,
    };
    let executable = std::env::current_exe()?;
    let load_results = common.results.join("under-load-benchmark");
    std::fs::create_dir_all(&load_results)?;
    let mut child = Command::new(&executable);
    child
        .arg("benchmark")
        .arg("--backend")
        .arg("opencl")
        .arg("--device")
        .arg(DEVICE_SELECTOR)
        .arg("--results")
        .arg(&load_results)
        .arg("--variant")
        .arg(&variant.name)
        .arg("--shapes")
        .arg(load_rows.to_string())
        .arg("--local-size")
        .arg("32")
        .env("OCL_ICD_VENDORS", "/etc/OpenCL/vendors/intel.icd")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(snapshot) = common.flags.get("snapshot") {
        child.arg("--snapshot").arg(snapshot);
    }
    let mut child = child.spawn().context("start under-load benchmark")?;
    let mut clock_samples = Vec::new();
    loop {
        clock_samples.push(json!({
            "monotonic_ns": monotonic_now_ns(),
            "frequencies_mhz": gt_frequency_snapshot(),
        }));
        if child.try_wait()?.is_some() {
            break;
        }
        thread::sleep(std::time::Duration::from_millis(10));
    }
    let child_output = child.wait_with_output()?;
    if !child_output.status.success() {
        bail!(
            "under-load benchmark failed: {}{}",
            String::from_utf8_lossy(&child_output.stdout),
            String::from_utf8_lossy(&child_output.stderr)
        );
    }

    let pmu_capture = common.flags.get("pmu-capture").map(PathBuf::from);
    let pmu_status = match &pmu_capture {
        Some(path) if path.is_file() => json!({
            "status": "pass",
            "capture": artifact_record(path)?,
            "claim_boundary": "i915 engine-busy/frequency PMU measures engine utilization and frequency; it is not proof that every one of 80 EUs was active"
        }),
        Some(path) => json!({
            "status": "insufficient_evidence",
            "reason": format!("supplied PMU capture does not exist: {}", path.display()),
            "claim_boundary": "no EU occupancy claim"
        }),
        None => json!({
            "status": "insufficient_evidence",
            "reason": "noninteractive sudo authentication is unavailable and no --pmu-capture was supplied",
            "claim_boundary": "no EU occupancy claim"
        }),
    };
    let pmu_results = common.results.join("pmu-benchmark");
    let pmu_command = format!(
        "sudo perf stat -x, -e i915/rcs0-busy/,i915/actual-frequency/ -o {} -- env OCL_ICD_VENDORS=/etc/OpenCL/vendors/intel.icd {} benchmark --backend opencl --device {} --results {} --variant {} --shapes {} --local-size 32",
        common.results.join("i915-pmu.csv").display(),
        executable.display(),
        DEVICE_SELECTOR,
        pmu_results.display(),
        variant.name,
        load_rows,
    );
    let raw_path = common.results.join("diagnose-opencl.json");
    let details = json!({
        "variant": variant.name,
        "gt_frequency_discovery": {
            "path_policy": "/sys/bus/pci/devices/0000:00:02.0/drm/card*/gt/gt0/{rps_act_freq_mhz,rps_cur_freq_mhz,punit_req_freq_mhz}",
            "idle_snapshot": gt_frequency_snapshot(),
            "under_load_samples": clock_samples,
            "sampling_period_ms": 10,
            "load_command_status": child_output.status.code(),
            "load_stdout": String::from_utf8_lossy(&child_output.stdout),
            "load_stderr": String::from_utf8_lossy(&child_output.stderr),
        },
        "pmu": pmu_status,
        "pmu_user_command": pmu_command,
        "pmu_interpretation": "rcs0-busy is render/compute engine utilization evidence; neither it nor actual-frequency proves simultaneous activity of all 80 execution units",
        "bandwidth": {
            "exact_weight_bytes": exact_bytes,
            "exact_expert_resident_bytes_with_fp32_bias": bundle.descriptor.xe_v2_bytes,
            "large_bytes": large_bytes,
            "large_policy": "min(256 MiB, 5% MemAvailable)",
            "gpu_cases": bandwidth,
            "cpu_copy_exact": cpu_bandwidth(exact_bytes, 30),
            "cpu_copy_large": cpu_bandwidth(large_bytes, 30),
            "concurrent_exact": concurrent_bandwidth(Backend::Opencl, common.immediate, exact_bytes, 30),
            "concurrent_large": concurrent_bandwidth(Backend::Opencl, common.immediate, large_bytes, 30),
        },
        "workgroup_widths": [32, 64, 128],
        "system_mutations": []
    });
    std::fs::write(&raw_path, serde_json::to_vec_pretty(&details)?)?;
    let native_path = common.results.join("diagnose-opencl.native");
    std::fs::write(&native_path, session.native_binary()?)?;
    let mut paths = vec![
        kernel_path,
        manifest_dir().join("fixtures/kernel-abi-v2.json"),
        raw_path,
        native_path,
    ];
    if let Some(path) = pmu_capture.filter(|path| path.is_file()) {
        paths.push(path);
    }
    write_manifest(
        arguments,
        common,
        ManifestEvidence {
            evidence_id: "X8-diagnose",
            status: EvidenceStatus::Pass,
            started_at,
            session: Some(session.info().clone()),
            loaded_paths: session.loaded_library_paths()?,
            artifact_paths: paths,
            details,
        },
    )
}

fn bandwidth_case(
    session: &Session,
    input: &[u8],
    name: &str,
    entry: &str,
    local_size: usize,
    passes: u32,
) -> Result<Value> {
    const WORKERS: usize = 4096;
    session.select_kernel(entry)?;
    let input_buffer = session.buffer(MemoryKind::Device, input.len())?;
    let checksum_buffer = session.buffer(MemoryKind::Device, WORKERS * 4)?;
    input_buffer.write(input)?;
    session.set_buffer(0, &input_buffer)?;
    session.set_buffer(1, &checksum_buffer)?;
    let bytes = u32::try_from(input.len()).context("bandwidth input exceeds u32")?;
    session.set_scalar(2, &bytes)?;
    session.set_scalar(3, &passes)?;
    session.set_group_size(u32::try_from(local_size)?, 1, 1)?;
    for _ in 0..10 {
        session.run([WORKERS, 1, 1], [local_size, 1, 1], TIMEOUT_NS)?;
    }
    let mut host = Vec::with_capacity(30);
    let mut device = Vec::with_capacity(30);
    for _ in 0..30 {
        let timing = session.run([WORKERS, 1, 1], [local_size, 1, 1], TIMEOUT_NS)?;
        host.push(timing.host_ns);
        if let Some(value) = timing.device_ns {
            device.push(value);
        }
    }
    let mut checksums = vec![0_u32; WORKERS];
    checksum_buffer.read(&mut checksums)?;
    let actual = checksums.iter().map(|&value| u64::from(value)).sum::<u64>();
    let expected = input.iter().map(|&value| u64::from(value)).sum::<u64>() * u64::from(passes);
    if actual != expected {
        bail!("bandwidth kernel {name} checksum mismatch: {actual} != {expected}");
    }
    Ok(json!({
        "name": name,
        "entry_point": entry,
        "bytes": input.len(),
        "passes": passes,
        "workgroup_width": local_size,
        "host": Distribution::from_samples(&host, 0x5d00 + local_size as u64),
        "device": (!device.is_empty()).then(|| Distribution::from_samples(&device, 0x5e00 + local_size as u64)),
        "checksum": actual,
    }))
}

fn gt_frequency_snapshot() -> Value {
    let drm = Path::new("/sys/bus/pci/devices/0000:00:02.0/drm");
    let mut values = serde_json::Map::new();
    let mut cards = std::fs::read_dir(drm)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("card") && !name.contains('-'))
        })
        .collect::<Vec<_>>();
    cards.sort();
    for card in cards {
        let gt = card.join("gt/gt0");
        for name in [
            "rps_act_freq_mhz",
            "rps_cur_freq_mhz",
            "punit_req_freq_mhz",
            "rps_min_freq_mhz",
            "rps_max_freq_mhz",
        ] {
            let path = gt.join(name);
            if let Ok(value) = std::fs::read_to_string(&path) {
                values.insert(
                    path.display().to_string(),
                    Value::String(value.trim().to_string()),
                );
            }
        }
    }
    Value::Object(values)
}

fn monotonic_now_ns() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

fn command_closeout(arguments: &[String], common: &Common) -> Result<()> {
    if common.backend != Backend::Opencl {
        bail!("X7 counterfactual tie-breaker is OpenCL; invoke closeout with --backend opencl");
    }
    let started_at = timestamp();
    let evidence_root = PathBuf::from(
        common
            .flags
            .get("evidence-root")
            .context("closeout requires --evidence-root")?,
    );
    let evidence_paths = [
        evidence_root.join("opencl/x6-opencl.manifest.json"),
        evidence_root.join("level-zero-regular/x6-level-zero.manifest.json"),
        evidence_root.join("level-zero-immediate/x6-level-zero.manifest.json"),
        evidence_root.join("opencl/x5-opencl.manifest.json"),
        evidence_root.join("level-zero-regular/x5-level-zero.manifest.json"),
        evidence_root.join("opencl/x4-memory-opencl.manifest.json"),
        evidence_root.join("level-zero-regular/x4-memory-level-zero.manifest.json"),
    ];
    let mut evidence = Vec::new();
    let mut useful_win = false;
    for (index, path) in evidence_paths.iter().enumerate() {
        let value: Value = serde_json::from_slice(
            &std::fs::read(path).with_context(|| format!("read {}", path.display()))?,
        )?;
        if index < 3 {
            useful_win |= value
                .pointer("/details/benchmark/any_useful_win")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        }
        evidence.push(json!({
            "record": artifact_record(path)?,
            "evidence_id": value.get("evidence_id"),
            "status": value.get("status"),
            "repository_revision": value.get("repository_revision"),
        }));
    }
    if useful_win {
        bail!("closeout rejected: an X6 manifest reports a useful win");
    }
    let state_path = evidence_root.join("x6-system-state.txt");
    let session = evidence_session(common)?;
    let loaded_paths = session.loaded_library_paths()?;
    write_manifest(
        arguments,
        common,
        ManifestEvidence {
            evidence_id: "X7",
            status: EvidenceStatus::Fail,
            started_at,
            session: Some(session.info().clone()),
            loaded_paths,
            artifact_paths: evidence_paths
                .into_iter()
                .chain(state_path.exists().then_some(state_path))
                .collect(),
            details: json!({
                "terminal_result": "negative_closeout",
                "useful_win_gate": "fail: no Xe path reached 1.25x at M=4,8,16,32,or 64 with a bootstrap 95% interval above parity",
                "correctness": "pass for every measured path and output; the lane closes on end-to-end performance without weakening numerical or memory gates",
                "model_scale_duplicate_weights": false,
                "decisions": {
                    "host_api": "none selected for implementation; counterfactual forced-only tie-breaker is OpenCL because no plausible shape showed a >=10% API win and its audited symbol/lifecycle/error surface is smaller",
                    "kernel_delivery": "reproducible validated SPIR-V",
                    "native_caching": "optional atomic runtime cache only, invalidated by the full v1 cache key",
                    "memory_residency": "one compact persistent selected region or fixed bounded slab plus reusable activation/output scratch",
                    "submission": "counterfactual serialized in-order OpenCL queue with completion event; no automatic dispatch",
                    "integration_boundary": "forced experimental model attachment below model-state commit and above the existing CPU MXFP4 oracle",
                    "cpu_fallback": "discard failed or unvalidated GPU output and recompute on CPU only before commit"
                },
                "evidence": evidence,
                "scope": "T14 Tiger Lake-LP 8086:9a49 and the captured 26.05 Compute Runtime / Level Zero 1.28.2 stack only"
            }),
        },
    )
}

#[derive(Debug, Serialize, Deserialize)]
struct ArtifactVariant {
    name: String,
    creation: Distribution,
    warm_host: Distribution,
    warm_device: Option<Distribution>,
    validation: String,
    native_bytes: usize,
}

fn artifact_variant(
    common: &Common,
    name: &str,
    kind: ArtifactKind,
    artifact_path: &Path,
    artifact: &[u8],
    options: &str,
    entry: &str,
) -> Result<ArtifactVariant> {
    let creation_samples = process_cold_samples(common, kind, artifact_path, options, entry, 10)?;
    let session = Session::create(
        common.backend,
        kind,
        artifact,
        options,
        entry,
        common.immediate,
    )?;
    let (warm_host, warm_device) = run_elementwise_samples(&session, 10, 30)?;
    let native_bytes = session.native_binary()?.len();
    Ok(ArtifactVariant {
        name: name.to_string(),
        creation: Distribution::from_samples(&creation_samples, 0x5833),
        warm_host: Distribution::from_samples(&warm_host, 0x5834),
        warm_device: (!warm_device.is_empty())
            .then(|| Distribution::from_samples(&warm_device, 0x5835)),
        validation: "pass: 4096 exact i32 outputs".into(),
        native_bytes,
    })
}

fn process_cold_samples(
    common: &Common,
    kind: ArtifactKind,
    artifact_path: &Path,
    options: &str,
    entry: &str,
    count: usize,
) -> Result<Vec<u64>> {
    let executable = std::env::current_exe()?;
    let kind_name = match kind {
        ArtifactKind::OpenclSource => "source",
        ArtifactKind::Spirv => "spirv",
        ArtifactKind::Native => "native",
        ArtifactKind::OpenclBinary => "opencl-binary",
    };
    let mut samples = Vec::with_capacity(count);
    for _ in 0..count {
        let mut command = Command::new(&executable);
        command
            .arg("artifact-once")
            .arg("--backend")
            .arg(common.backend.as_str())
            .arg("--device")
            .arg(DEVICE_SELECTOR)
            .arg("--kind")
            .arg(kind_name)
            .arg("--artifact")
            .arg(artifact_path)
            .arg("--options")
            .arg(options)
            .arg("--entry")
            .arg(entry);
        if common.immediate {
            command.arg("--immediate");
        }
        let output = command
            .output()
            .context("run process-cold artifact child")?;
        if !output.status.success() {
            bail!(
                "artifact child failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        samples.push(
            String::from_utf8(output.stdout)?
                .trim()
                .parse::<u64>()
                .context("parse process-cold creation time")?,
        );
    }
    Ok(samples)
}

fn artifact_once(arguments: &[String]) -> Result<()> {
    let mut flags = HashMap::new();
    let mut immediate = false;
    let mut index = 0;
    while index < arguments.len() {
        if arguments[index] == "--immediate" {
            immediate = true;
            index += 1;
        } else {
            flags.insert(arguments[index].clone(), arguments[index + 1].clone());
            index += 2;
        }
    }
    let backend = Backend::parse(flags.get("--backend").context("child backend")?)?;
    if flags.get("--device").map(String::as_str) != Some(DEVICE_SELECTOR) {
        bail!("child exact device selector mismatch");
    }
    let kind = match flags.get("--kind").map(String::as_str) {
        Some("source") => ArtifactKind::OpenclSource,
        Some("spirv") => ArtifactKind::Spirv,
        Some("native") => ArtifactKind::Native,
        Some("opencl-binary") => ArtifactKind::OpenclBinary,
        _ => bail!("child artifact kind"),
    };
    let artifact = std::fs::read(flags.get("--artifact").context("child artifact")?)?;
    let session = Session::create(
        backend,
        kind,
        &artifact,
        flags.get("--options").map(String::as_str).unwrap_or(""),
        flags.get("--entry").context("child entry")?,
        immediate,
    )?;
    println!("{}", session.info().creation_ns);
    Ok(())
}

fn run_elementwise_samples(
    session: &Session,
    warmups: usize,
    samples: usize,
) -> Result<(Vec<u64>, Vec<u64>)> {
    let count = 4096_u32;
    let left = (0..count)
        .map(|value| value as i32 - 2048)
        .collect::<Vec<_>>();
    let right = (0..count)
        .map(|value| value as i32 * 3 - 7)
        .collect::<Vec<_>>();
    let expected = left
        .iter()
        .zip(&right)
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    let mut output = vec![0_i32; count as usize];
    let left_buffer = session.buffer(MemoryKind::Device, std::mem::size_of_val(left.as_slice()))?;
    let right_buffer =
        session.buffer(MemoryKind::Device, std::mem::size_of_val(right.as_slice()))?;
    let output_buffer =
        session.buffer(MemoryKind::Device, std::mem::size_of_val(output.as_slice()))?;
    left_buffer.write(&left)?;
    right_buffer.write(&right)?;
    session.set_buffer(0, &left_buffer)?;
    session.set_buffer(1, &right_buffer)?;
    session.set_buffer(2, &output_buffer)?;
    session.set_scalar(3, &count)?;
    session.set_group_size(64, 1, 1)?;
    for _ in 0..warmups {
        session.run([count as usize, 1, 1], [64, 1, 1], TIMEOUT_NS)?;
    }
    let mut host = Vec::with_capacity(samples);
    let mut device = Vec::with_capacity(samples);
    for _ in 0..samples {
        let timing = session.run([count as usize, 1, 1], [64, 1, 1], TIMEOUT_NS)?;
        host.push(timing.host_ns);
        if let Some(value) = timing.device_ns {
            device.push(value);
        }
    }
    output_buffer.read(&mut output)?;
    if output != expected {
        bail!("elementwise output validation failed");
    }
    Ok((host, device))
}

#[derive(Debug)]
struct Mxfp4Case {
    weight: Mxfp4Block,
    activation: ResidualQ8Block,
}

fn mxfp4_cases() -> Vec<Mxfp4Case> {
    let mut cases = Vec::with_capacity(16 * 256 + 10_000);
    for scale in 0_u8..=u8::MAX {
        for code in 0_u8..16 {
            let packed = code | (code << 4);
            cases.push(Mxfp4Case {
                weight: Mxfp4Block {
                    scale,
                    packed: [packed; 16],
                },
                activation: ResidualQ8Block {
                    primary: Q8Block {
                        scale: f32::from_bits(0x3e80_0000 + ((scale as u32 & 7) << 17)),
                        values: std::array::from_fn(|index| index as i8 * 7 - 101),
                    },
                    residual: Q8Block {
                        scale: f32::from_bits(0x3c80_0000 + ((code as u32 & 7) << 17)),
                        values: std::array::from_fn(|index| 63 - index as i8 * 3),
                    },
                },
            });
        }
    }
    let mut random = ChaCha8Rng::seed_from_u64(0x5845_2d4d_5846_5034);
    for _ in 0..10_000 {
        let mut packed = [0_u8; 16];
        let mut primary = [0_i8; 32];
        let mut residual = [0_i8; 32];
        random.fill_bytes(&mut packed);
        random.fill_bytes(unsafe {
            std::slice::from_raw_parts_mut(primary.as_mut_ptr().cast::<u8>(), primary.len())
        });
        random.fill_bytes(unsafe {
            std::slice::from_raw_parts_mut(residual.as_mut_ptr().cast::<u8>(), residual.len())
        });
        cases.push(Mxfp4Case {
            weight: Mxfp4Block {
                scale: random.gen(),
                packed,
            },
            activation: ResidualQ8Block {
                primary: Q8Block {
                    scale: random.gen_range(2.0_f32.powi(-12)..2.0_f32.powi(4)),
                    values: primary,
                },
                residual: Q8Block {
                    scale: random.gen_range(2.0_f32.powi(-18)..2.0_f32.powi(-2)),
                    values: residual,
                },
            },
        });
    }
    cases
}

fn run_mxfp4_cases(session: &Session, cases: &[Mxfp4Case]) -> Result<Value> {
    let count = u32::try_from(cases.len())?;
    let kernels = Kernels::new(KernelPath::Scalar)?;
    let mut packed = Vec::with_capacity(cases.len() * 16);
    let mut weight_scales = Vec::with_capacity(cases.len());
    let mut primary = Vec::with_capacity(cases.len() * 32);
    let mut residual = Vec::with_capacity(cases.len() * 32);
    let mut primary_scales = Vec::with_capacity(cases.len());
    let mut residual_scales = Vec::with_capacity(cases.len());
    let mut expected_primary_integer = Vec::with_capacity(cases.len());
    let mut expected_residual_integer = Vec::with_capacity(cases.len());
    let mut expected_q8 = Vec::with_capacity(cases.len());
    let mut expected_residual = Vec::with_capacity(cases.len());
    for case in cases {
        packed.extend_from_slice(&case.weight.packed);
        weight_scales.push(case.weight.scale);
        primary.extend_from_slice(&case.activation.primary.values);
        residual.extend_from_slice(&case.activation.residual.values);
        primary_scales.push(case.activation.primary.scale);
        residual_scales.push(case.activation.residual.scale);
        let dots = kernels.mxfp4_residual_q8_block_dot_i32(&case.weight, &case.activation);
        expected_primary_integer.push(dots[0]);
        expected_residual_integer.push(dots[1]);
        expected_q8.push(kernels.mxfp4_q8_dot(
            std::slice::from_ref(&case.weight),
            std::slice::from_ref(&case.activation.primary),
        )?);
        expected_residual.push(kernels.mxfp4_residual_q8_dot(
            std::slice::from_ref(&case.weight),
            std::slice::from_ref(&case.activation),
        )?);
    }
    let inputs: [&[u8]; 2] = [&packed, &weight_scales];
    let packed_buffer = session.buffer(MemoryKind::Device, packed.len())?;
    let scales_buffer = session.buffer(MemoryKind::Device, weight_scales.len())?;
    let primary_buffer = session.buffer(MemoryKind::Device, primary.len())?;
    let residual_buffer = session.buffer(MemoryKind::Device, residual.len())?;
    let primary_scale_buffer = session.buffer(
        MemoryKind::Device,
        std::mem::size_of_val(primary_scales.as_slice()),
    )?;
    let residual_scale_buffer = session.buffer(
        MemoryKind::Device,
        std::mem::size_of_val(residual_scales.as_slice()),
    )?;
    let primary_integer_buffer = session.buffer(MemoryKind::Device, cases.len() * 4)?;
    let residual_integer_buffer = session.buffer(MemoryKind::Device, cases.len() * 4)?;
    let q8_buffer = session.buffer(MemoryKind::Device, cases.len() * 4)?;
    let residual_q8_buffer = session.buffer(MemoryKind::Device, cases.len() * 4)?;
    packed_buffer.write(inputs[0])?;
    scales_buffer.write(inputs[1])?;
    primary_buffer.write(&primary)?;
    residual_buffer.write(&residual)?;
    primary_scale_buffer.write(&primary_scales)?;
    residual_scale_buffer.write(&residual_scales)?;
    for (index, buffer) in [
        &packed_buffer,
        &scales_buffer,
        &primary_buffer,
        &residual_buffer,
        &primary_scale_buffer,
        &residual_scale_buffer,
        &primary_integer_buffer,
        &residual_integer_buffer,
        &q8_buffer,
        &residual_q8_buffer,
    ]
    .into_iter()
    .enumerate()
    {
        session.set_buffer(index as u32, buffer)?;
    }
    session.set_scalar(10, &count)?;
    session.set_group_size(64, 1, 1)?;
    let global = round_up(cases.len(), 64);
    let timing = session.run([global, 1, 1], [64, 1, 1], TIMEOUT_NS)?;
    let mut actual_primary_integer = vec![0_i32; cases.len()];
    let mut actual_residual_integer = vec![0_i32; cases.len()];
    let mut actual_q8 = vec![0_f32; cases.len()];
    let mut actual_residual = vec![0_f32; cases.len()];
    primary_integer_buffer.read(&mut actual_primary_integer)?;
    residual_integer_buffer.read(&mut actual_residual_integer)?;
    q8_buffer.read(&mut actual_q8)?;
    residual_q8_buffer.read(&mut actual_residual)?;
    if actual_primary_integer != expected_primary_integer
        || actual_residual_integer != expected_residual_integer
    {
        bail!("MXFP4 exact integer intermediates differ from scalar oracle");
    }
    let mut finite_bit_mismatches = 0_usize;
    let mut bf16_mismatches = 0_usize;
    let mut nan_mismatches = 0_usize;
    for ((expected, actual), (expected_residual, actual_residual)) in expected_q8
        .iter()
        .zip(&actual_q8)
        .zip(expected_residual.iter().zip(&actual_residual))
    {
        if expected.is_nan() {
            nan_mismatches += usize::from(!actual.is_nan());
        } else {
            finite_bit_mismatches += usize::from(expected.to_bits() != actual.to_bits());
            bf16_mismatches += usize::from(
                bf16::from_f32(*expected).to_bits() != bf16::from_f32(*actual).to_bits(),
            );
        }
        if expected_residual.is_nan() {
            nan_mismatches += usize::from(!actual_residual.is_nan());
        } else {
            finite_bit_mismatches +=
                usize::from(expected_residual.to_bits() != actual_residual.to_bits());
            bf16_mismatches += usize::from(
                bf16::from_f32(*expected_residual).to_bits()
                    != bf16::from_f32(*actual_residual).to_bits(),
            );
        }
    }
    if finite_bit_mismatches != 0 || bf16_mismatches != 0 || nan_mismatches != 0 {
        bail!(
            "MXFP4 one-block mismatch: finite_bits={finite_bit_mismatches}, bf16={bf16_mismatches}, nan={nan_mismatches}"
        );
    }
    Ok(json!({
        "status": "pass",
        "cases": cases.len(),
        "integer_mismatches": 0,
        "finite_bit_mismatches": finite_bit_mismatches,
        "bf16_boundary_mismatches": bf16_mismatches,
        "nan_behavior_mismatches": nan_mismatches,
        "host_ns": timing.host_ns,
        "device_ns": timing.device_ns,
        "e8m0_zero_bits": format!("0x{:08x}", e8m0_scale(0).to_bits()),
        "e8m0_ff_is_nan": e8m0_scale(0xff).is_nan()
    }))
}

fn negative_artifact_isolated(
    common: &Common,
    kind: ArtifactKind,
    artifact: &[u8],
    entry: &str,
    label: &str,
) -> String {
    let path = common.results.join(format!("{label}.invalid-artifact"));
    if let Err(error) = std::fs::write(&path, artifact) {
        return format!("incomplete: could not preserve negative fixture: {error}");
    }
    let executable = match std::env::current_exe() {
        Ok(executable) => executable,
        Err(error) => return format!("incomplete: current executable unavailable: {error}"),
    };
    let kind_name = match kind {
        ArtifactKind::OpenclSource => "source",
        ArtifactKind::Spirv => "spirv",
        ArtifactKind::Native => "native",
        ArtifactKind::OpenclBinary => "opencl-binary",
    };
    let mut command = Command::new(executable);
    command
        .arg("artifact-once")
        .arg("--backend")
        .arg(common.backend.as_str())
        .arg("--device")
        .arg(DEVICE_SELECTOR)
        .arg("--kind")
        .arg(kind_name)
        .arg("--artifact")
        .arg(&path)
        .arg("--options")
        .arg("")
        .arg("--entry")
        .arg(entry);
    if common.immediate {
        command.arg("--immediate");
    }
    match command.output() {
        Ok(output) if !output.status.success() => format!(
            "pass: rejected safely in child process (exit={:?}): {}{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
        Ok(_) => "invalid: malformed artifact unexpectedly accepted".into(),
        Err(error) => format!("incomplete: negative child failed to start: {error}"),
    }
}

fn negative_result<T>(result: Result<T>) -> String {
    result.map_or_else(
        |error| format!("pass: rejected safely: {error:#}"),
        |_| "invalid: negative case unexpectedly succeeded".into(),
    )
}

fn ensure_spirv(kernel: &str, results: &Path) -> Result<Vec<u8>> {
    let path = ensure_spirv_path(kernel, results)?;
    std::fs::read(&path).with_context(|| format!("read generated SPIR-V {}", path.display()))
}

fn ensure_spirv_path(kernel: &str, results: &Path) -> Result<PathBuf> {
    let stem = kernel.trim_end_matches(".cl");
    let output = results.join(format!("{stem}.spv"));
    let source = manifest_dir().join("kernels").join(kernel);
    let script = manifest_dir().join("scripts/build-spirv.sh");
    let status = Command::new(&script)
        .arg(&source)
        .arg(&output)
        .status()
        .with_context(|| format!("run {}", script.display()))?;
    if !status.success() {
        bail!("SPIR-V build failed for {}", source.display());
    }
    Ok(output)
}

fn read_kernel(name: &str) -> Result<Vec<u8>> {
    std::fs::read(manifest_dir().join("kernels").join(name)).context("read kernel source")
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn round_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

fn write_manifest(arguments: &[String], common: &Common, evidence: ManifestEvidence) -> Result<()> {
    let manifest = Manifest {
        schema: SCHEMA.into(),
        evidence_id: evidence.evidence_id.into(),
        run_id: common
            .results
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("unnamed-run")
            .into(),
        status: evidence.status,
        started_at: evidence.started_at,
        finished_at: timestamp(),
        command: arguments.to_vec(),
        repository_revision: git_output(&["rev-parse", "HEAD"]),
        repository_branch: git_output(&["branch", "--show-current"]),
        host: hostname(),
        backend: common.backend,
        device_selector: DEVICE_SELECTOR.into(),
        session: evidence.session,
        loaded_libraries: evidence
            .loaded_paths
            .into_iter()
            .filter(|path| path.is_file())
            .map(|path| artifact_record(&path))
            .collect::<Result<Vec<_>>>()?,
        artifacts: evidence
            .artifact_paths
            .into_iter()
            .filter(|path| path.is_file())
            .map(|path| artifact_record(&path))
            .collect::<Result<Vec<_>>>()?,
        details: evidence.details,
    };
    let path = common.results.join(format!(
        "{}-{}.manifest.json",
        evidence.evidence_id.to_ascii_lowercase(),
        common.backend
    ));
    std::fs::write(&path, serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", path.display());
    Ok(())
}

fn artifact_record(path: &Path) -> Result<ArtifactRecord> {
    let mut file = File::open(path).with_context(|| format!("open artifact {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    let mut size = 0_u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
        size += read as u64;
    }
    Ok(ArtifactRecord {
        path: path.to_path_buf(),
        size,
        sha256: format!("{:x}", hash.finalize()),
    })
}

type CaptureCommand = (
    &'static str,
    &'static str,
    Vec<OsString>,
    Vec<(OsString, OsString)>,
);

fn environment_commands(backend: Backend) -> Vec<CaptureCommand> {
    let mut commands = vec![
        ("kernel", "uname", vec!["-a".into()], vec![]),
        (
            "pci",
            "lspci",
            vec!["-nnk".into(), "-s".into(), "00:02.0".into()],
            vec![],
        ),
        ("identity", "id", vec![], vec![]),
        (
            "render nodes",
            "ls",
            vec!["-l".into(), "/dev/dri".into()],
            vec![],
        ),
        (
            "render ACL",
            "getfacl",
            vec!["-p".into(), "/dev/dri/renderD128".into()],
            vec![],
        ),
        ("driver module", "modinfo", vec!["i915".into()], vec![]),
        (
            "firmware packages",
            "dpkg-query",
            vec![
                "-W".into(),
                "linux-firmware".into(),
                "intel-microcode".into(),
            ],
            vec![],
        ),
        (
            "runtime packages",
            "dpkg-query",
            vec![
                "-W".into(),
                "intel-opencl-icd".into(),
                "intel-opencl-icd-legacy".into(),
                "libze1".into(),
                "libze-intel-gpu1".into(),
                "libze-intel-gpu-legacy1-1".into(),
                "libigc1".into(),
                "libigdgmm12".into(),
                "ocl-icd-libopencl1".into(),
                "libdrm2".into(),
                "mesa-libgallium".into(),
            ],
            vec![],
        ),
        (
            "package candidates",
            "apt-cache",
            vec![
                "policy".into(),
                "intel-opencl-icd".into(),
                "libze1".into(),
                "libze-intel-gpu1".into(),
                "libigc1".into(),
            ],
            vec![],
        ),
        ("loader cache", "ldconfig", vec!["-p".into()], vec![]),
        (
            "rust",
            "rustc",
            vec!["--version".into(), "--verbose".into()],
            vec![],
        ),
        (
            "cargo",
            "cargo",
            vec!["--version".into(), "--verbose".into()],
            vec![],
        ),
    ];
    match backend {
        Backend::Opencl => commands.push((
            "OpenCL exact ICD",
            "clinfo",
            vec![],
            vec![(
                "OCL_ICD_VENDORS".into(),
                "/etc/OpenCL/vendors/intel.icd".into(),
            )],
        )),
        Backend::LevelZero => commands.push((
            "Level Zero exact post-upgrade probe",
            "/home/emmy/src/xe-research/build/ze-info-minimal/ze_info",
            vec![],
            vec![],
        )),
    }
    commands
}

fn run_capture(
    program: &str,
    arguments: &[OsString],
    environment: &[(OsString, OsString)],
) -> (i32, Vec<u8>) {
    let output = Command::new(program)
        .args(arguments)
        .envs(environment.iter().cloned())
        .stderr(Stdio::piped())
        .output();
    match output {
        Ok(output) => {
            let mut bytes = output.stdout;
            bytes.extend_from_slice(&output.stderr);
            (output.status.code().unwrap_or(-1), bytes)
        }
        Err(error) => (-1, format!("unavailable: {error}\n").into_bytes()),
    }
}

fn source_pins() -> Result<Value> {
    let roots = [
        (
            "compute-runtime",
            format!("{CORPUS}/runtime-matched/compute-runtime-26.05.37020.3"),
        ),
        (
            "level-zero",
            format!("{CORPUS}/runtime-matched/level-zero-v1.28.2"),
        ),
        ("llama.cpp", format!("{CORPUS}/llama.cpp")),
        ("level-zero-tests", format!("{CORPUS}/level-zero-tests")),
        ("pti-gpu", format!("{CORPUS}/pti-gpu")),
    ];
    let mut values = serde_json::Map::new();
    for (name, root) in roots {
        let revision = Command::new("git")
            .args(["-C", &root, "rev-parse", "HEAD"])
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_else(|_| "unavailable".into());
        let status = Command::new("git")
            .args(["-C", &root, "status", "--porcelain"])
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_else(|_| "unavailable".into());
        values.insert(
            name.into(),
            json!({"path": root, "revision": revision, "status": status}),
        );
    }
    Ok(Value::Object(values))
}

fn cpu_bandwidth(size: usize, repetitions: usize) -> Value {
    let source = vec![0xa5_u8; size];
    let mut destination = vec![0_u8; size];
    let mut samples = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let start = Instant::now();
        destination.copy_from_slice(&source);
        black_box(&destination);
        samples.push(start.elapsed().as_nanos() as u64);
    }
    json!({
        "bytes": size,
        "timing": Distribution::from_samples(&samples, 0x5844),
    })
}

fn concurrent_bandwidth(
    backend: Backend,
    immediate: bool,
    size: usize,
    repetitions: usize,
) -> Value {
    let mut cpu_samples = Vec::with_capacity(repetitions);
    let mut gpu_roundtrips = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let worker = thread::spawn(move || {
            let source = vec![0x5a_u8; size];
            let mut destination = vec![0_u8; size];
            let start = Instant::now();
            destination.copy_from_slice(&source);
            black_box(destination);
            start.elapsed().as_nanos() as u64
        });
        gpu_roundtrips.push(runtime::memory_roundtrip(
            backend,
            MemoryKind::Device,
            size,
            immediate,
        ));
        match worker.join() {
            Ok(sample) => cpu_samples.push(sample),
            Err(_) => {
                return json!({
                    "status": "invalid",
                    "reason": "CPU copy worker panicked",
                });
            }
        }
    }
    json!({
        "status": "pass",
        "bytes_per_side": size,
        "repetitions": repetitions,
        "cpu_copy_timing": Distribution::from_samples(&cpu_samples, 0x5844_434f),
        "gpu_roundtrips": gpu_roundtrips,
        "scope": "CPU memcpy and a device allocation/write/read/reuse/cleanup roundtrip begin concurrently; this is contention evidence, not an overlap throughput claim"
    })
}

fn evidence_session(common: &Common) -> Result<Session> {
    match common.backend {
        Backend::Opencl => Session::create(
            common.backend,
            ArtifactKind::OpenclSource,
            &read_kernel("elementwise.cl")?,
            "-cl-std=CL3.0",
            "xe_i32_add",
            common.immediate,
        ),
        Backend::LevelZero => Session::create(
            common.backend,
            ArtifactKind::Spirv,
            &ensure_spirv("elementwise.cl", &common.results)?,
            "",
            "xe_i32_add",
            common.immediate,
        ),
    }
}

fn available_memory_bytes() -> Result<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    let kib = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .context("parse MemAvailable")?;
    Ok(kib * 1024)
}

fn read_proc_status() -> Value {
    let text = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let fields = ["VmPeak", "VmSize", "VmHWM", "VmRSS"];
    let values = fields
        .into_iter()
        .filter_map(|field| {
            text.lines()
                .find(|line| line.starts_with(field))
                .map(|line| (field.to_string(), Value::String(line.to_string())))
        })
        .collect::<serde_json::Map<_, _>>();
    Value::Object(values)
}

fn benchmark_environment() -> Value {
    let commands = [
        ("power_profile", "powerprofilesctl", vec!["get"]),
        ("ac_online", "bash", vec!["-lc", "for f in /sys/class/power_supply/*/online; do printf '%s=' \"$f\"; cat \"$f\"; done"]),
        ("display", "loginctl", vec!["show-session", "auto", "-p", "Type", "-p", "Remote", "-p", "State"]),
        ("cpu_frequency", "bash", vec!["-lc", "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null"]),
        ("gpu_frequency", "bash", vec!["-lc", "find -L /sys/bus/pci/devices/0000:00:02.0/drm/card*/gt/gt0 -maxdepth 1 -type f \\( -name 'rps_*freq_mhz' -o -name 'punit_req_freq_mhz' \\) -print -exec cat {} \\; 2>/dev/null"]),
        ("thermals", "bash", vec!["-lc", "for f in /sys/class/thermal/thermal_zone*/temp; do printf '%s ' \"$f\"; cat \"$f\"; done 2>/dev/null"]),
    ];
    let mut values = serde_json::Map::new();
    for (name, program, arguments) in commands {
        let output = Command::new(program).args(arguments).output();
        values.insert(
            name.into(),
            Value::String(
                output
                    .map(|output| {
                        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
                        text.push_str(&String::from_utf8_lossy(&output.stderr));
                        text.trim().to_string()
                    })
                    .unwrap_or_else(|error| format!("unavailable: {error}")),
            ),
        );
    }
    Value::Object(values)
}

fn timestamp() -> String {
    Command::new("date")
        .arg("--iso-8601=seconds")
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|_| {
            format!(
                "unix:{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            )
        })
}

fn git_output(arguments: &[&str]) -> String {
    Command::new("git")
        .args(arguments)
        .current_dir(manifest_dir().join("../.."))
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|_| "unavailable".into())
}

fn hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .map(|value| value.trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

#[derive(Debug, Serialize)]
struct CorrectnessReport {
    rows: usize,
    scalar_vs_avx2: Comparison,
    scalar_vs_xe: Comparison,
    pass: bool,
}

#[derive(Debug, Default, Serialize)]
struct Comparison {
    values: usize,
    non_finite_mismatches: usize,
    tolerance_mismatches: usize,
    bf16_boundary_mismatches: usize,
    max_ulp: u32,
    max_absolute: f32,
}

#[derive(Debug, Serialize)]
struct TrialReport {
    trial: usize,
    method_order: Vec<&'static str>,
    scalar_oracle: Distribution,
    avx2: Distribution,
    xe: Distribution,
    scalar_oracle_phases: PhaseDistributions,
    avx2_phases: PhaseDistributions,
    xe_phases: PhaseDistributions,
}

#[derive(Debug, Serialize)]
struct ShapeReport {
    rows: usize,
    correctness: CorrectnessReport,
    trials: Vec<TrialReport>,
    combined_scalar_oracle: Option<Distribution>,
    combined_avx2: Option<Distribution>,
    combined_xe: Option<Distribution>,
    combined_scalar_oracle_phases: Option<PhaseDistributions>,
    combined_avx2_phases: Option<PhaseDistributions>,
    combined_xe_phases: Option<PhaseDistributions>,
    avx2_over_xe_speedup: Option<f64>,
    conservative_speedup_ci95: Option<[f64; 2]>,
    useful_win: bool,
    estimated_residency_break_even_requests: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize)]
struct PhaseTiming {
    total_request_ns: u64,
    quantization_ns: u64,
    activation_packing_ns: u64,
    upload_ns: u64,
    argument_setup_ns: u64,
    host_submission_ns: u64,
    host_wait_ns: u64,
    device_kernel_ns: Option<u64>,
    readback_ns: u64,
    bf16_conversion_ns: u64,
}

#[derive(Debug, Default)]
struct PhaseSamples {
    total_request_ns: Vec<u64>,
    quantization_ns: Vec<u64>,
    activation_packing_ns: Vec<u64>,
    upload_ns: Vec<u64>,
    argument_setup_ns: Vec<u64>,
    host_submission_ns: Vec<u64>,
    host_wait_ns: Vec<u64>,
    device_kernel_ns: Vec<u64>,
    readback_ns: Vec<u64>,
    bf16_conversion_ns: Vec<u64>,
}

impl PhaseSamples {
    fn push(&mut self, timing: PhaseTiming) {
        self.total_request_ns.push(timing.total_request_ns);
        self.quantization_ns.push(timing.quantization_ns);
        self.activation_packing_ns
            .push(timing.activation_packing_ns);
        self.upload_ns.push(timing.upload_ns);
        self.argument_setup_ns.push(timing.argument_setup_ns);
        self.host_submission_ns.push(timing.host_submission_ns);
        self.host_wait_ns.push(timing.host_wait_ns);
        if let Some(device) = timing.device_kernel_ns {
            self.device_kernel_ns.push(device);
        }
        self.readback_ns.push(timing.readback_ns);
        self.bf16_conversion_ns.push(timing.bf16_conversion_ns);
    }

    fn extend(&mut self, other: &Self) {
        self.total_request_ns
            .extend_from_slice(&other.total_request_ns);
        self.quantization_ns
            .extend_from_slice(&other.quantization_ns);
        self.activation_packing_ns
            .extend_from_slice(&other.activation_packing_ns);
        self.upload_ns.extend_from_slice(&other.upload_ns);
        self.argument_setup_ns
            .extend_from_slice(&other.argument_setup_ns);
        self.host_submission_ns
            .extend_from_slice(&other.host_submission_ns);
        self.host_wait_ns.extend_from_slice(&other.host_wait_ns);
        self.device_kernel_ns
            .extend_from_slice(&other.device_kernel_ns);
        self.readback_ns.extend_from_slice(&other.readback_ns);
        self.bf16_conversion_ns
            .extend_from_slice(&other.bf16_conversion_ns);
    }

    fn distributions(&self, seed: u64) -> PhaseDistributions {
        let distribution = |samples: &[u64], offset| {
            (!samples.is_empty()).then(|| Distribution::from_samples(samples, seed + offset))
        };
        PhaseDistributions {
            total_request_ns: distribution(&self.total_request_ns, 0),
            quantization_ns: distribution(&self.quantization_ns, 1),
            activation_packing_ns: distribution(&self.activation_packing_ns, 2),
            upload_ns: distribution(&self.upload_ns, 3),
            argument_setup_ns: distribution(&self.argument_setup_ns, 4),
            host_submission_ns: distribution(&self.host_submission_ns, 5),
            host_wait_ns: distribution(&self.host_wait_ns, 6),
            device_kernel_ns: distribution(&self.device_kernel_ns, 7),
            readback_ns: distribution(&self.readback_ns, 8),
            bf16_conversion_ns: distribution(&self.bf16_conversion_ns, 9),
        }
    }
}

#[derive(Debug, Serialize)]
struct PhaseDistributions {
    total_request_ns: Option<Distribution>,
    quantization_ns: Option<Distribution>,
    activation_packing_ns: Option<Distribution>,
    upload_ns: Option<Distribution>,
    argument_setup_ns: Option<Distribution>,
    host_submission_ns: Option<Distribution>,
    host_wait_ns: Option<Distribution>,
    device_kernel_ns: Option<Distribution>,
    readback_ns: Option<Distribution>,
    bf16_conversion_ns: Option<Distribution>,
}

enum GpuBuffers<'a> {
    V1 {
        primary: Buffer<'a>,
        residual: Buffer<'a>,
        primary_scales: Buffer<'a>,
        residual_scales: Buffer<'a>,
        output: Buffer<'a>,
    },
    V2 {
        activations: Buffer<'a>,
        output: Buffer<'a>,
        splitk_terms: Option<Buffer<'a>>,
    },
}

struct GpuRequestContext<'request, 'session> {
    session: &'request Session,
    gpu: &'request GpuBuffers<'session>,
    v2_weights: Option<&'request Buffer<'session>>,
    bias: &'request Buffer<'session>,
    variant: &'request VariantSpec,
    local_size: usize,
}

fn benchmark_projection(
    session: &Session,
    bundle: &ProjectionBundle,
    variant: &VariantSpec,
    shapes: &[usize],
    local_size: usize,
) -> Result<Value> {
    let max_rows = *shapes
        .iter()
        .max()
        .context("benchmark needs at least one shape")?;
    let residency_start = Instant::now();
    let legacy_packed = (!variant.is_v2())
        .then(|| session.buffer(MemoryKind::Device, bundle.packed.len()))
        .transpose()?;
    let legacy_scales = (!variant.is_v2())
        .then(|| session.buffer(MemoryKind::Device, bundle.scales.len()))
        .transpose()?;
    let v2_weights = variant
        .is_v2()
        .then(|| session.buffer(MemoryKind::Device, bundle.xe_v2_records.len()))
        .transpose()?;
    let bias_buffer = session.buffer(
        MemoryKind::Device,
        std::mem::size_of_val(bundle.bias.as_slice()),
    )?;
    let mut weight_write_ns = 0_u64;
    if let Some(buffer) = &legacy_packed {
        weight_write_ns += buffer.write(&bundle.packed)?.host_ns;
    }
    if let Some(buffer) = &legacy_scales {
        weight_write_ns += buffer.write(&bundle.scales)?.host_ns;
    }
    if let Some(buffer) = &v2_weights {
        weight_write_ns += buffer.write(&bundle.xe_v2_records)?.host_ns;
    }
    weight_write_ns += bias_buffer.write(&bundle.bias)?.host_ns;
    let residency_ns = residency_start.elapsed().as_nanos() as u64;

    let gpu = if variant.is_v2() {
        let splitk_bytes = max_rows
            .checked_mul(N)
            .and_then(|value| value.checked_mul(BLOCKS))
            .and_then(|value| value.checked_mul(2 * std::mem::size_of::<f32>()))
            .context("split-K scratch extent overflow")?;
        if variant.kind == VariantKind::SplitkV2 && splitk_bytes > 8_294_400 {
            bail!("split-K scratch exceeds the immutable 8294400-byte bound");
        }
        GpuBuffers::V2 {
            activations: session.buffer(
                MemoryKind::Device,
                max_rows * BLOCKS * std::mem::size_of::<XeActivationRecordV2>(),
            )?,
            output: session.buffer(MemoryKind::Device, max_rows * N * 4)?,
            splitk_terms: (variant.kind == VariantKind::SplitkV2)
                .then(|| session.buffer(MemoryKind::Device, splitk_bytes))
                .transpose()?,
        }
    } else {
        GpuBuffers::V1 {
            primary: session.buffer(MemoryKind::Device, max_rows * BLOCKS * 32)?,
            residual: session.buffer(MemoryKind::Device, max_rows * BLOCKS * 32)?,
            primary_scales: session.buffer(MemoryKind::Device, max_rows * BLOCKS * 4)?,
            residual_scales: session.buffer(MemoryKind::Device, max_rows * BLOCKS * 4)?,
            output: session.buffer(MemoryKind::Device, max_rows * N * 4)?,
        }
    };
    match &gpu {
        GpuBuffers::V1 {
            primary,
            residual,
            primary_scales,
            residual_scales,
            output,
        } => {
            let buffers = [
                legacy_packed
                    .as_ref()
                    .context("missing canonical packed buffer")?,
                legacy_scales
                    .as_ref()
                    .context("missing canonical scales buffer")?,
                primary,
                residual,
                primary_scales,
                residual_scales,
                &bias_buffer,
                output,
            ];
            for (index, buffer) in buffers.into_iter().enumerate() {
                session.set_buffer(index as u32, buffer)?;
            }
        }
        GpuBuffers::V2 {
            activations,
            output,
            ..
        } if variant.kind != VariantKind::SplitkV2 => {
            session.set_buffer(0, v2_weights.as_ref().context("missing v2 weights")?)?;
            session.set_buffer(1, activations)?;
            session.set_buffer(2, &bias_buffer)?;
            session.set_buffer(3, output)?;
        }
        GpuBuffers::V2 { .. } => {}
    }
    session.set_group_size(u32::try_from(local_size)?, 1, 1)?;
    let gpu_context = GpuRequestContext {
        session,
        gpu: &gpu,
        v2_weights: v2_weights.as_ref(),
        bias: &bias_buffer,
        variant,
        local_size,
    };

    let mut scratch_storage = vec![0_u8; (1 << 20) + 4096];
    let scratch_offset = scratch_storage.as_ptr().align_offset(4096);
    let scratch = &mut scratch_storage[scratch_offset..scratch_offset + (1 << 20)];
    let mut reports = Vec::new();
    let mut correctness_stopped = false;
    for &rows in shapes {
        let inputs = deterministic_activations(rows, 0x5845_5052_4546_494c);
        let prepared = quantize_residual_rows(&inputs)?;
        let mut scalar_output = vec![0.0_f32; rows * N];
        let mut avx2_output = vec![0.0_f32; rows * N];
        let mut xe_output = vec![0.0_f32; rows * N];
        bundle.cpu_projection_into(
            Mxfp4MatmulBackend::Scalar,
            &prepared,
            rows,
            &mut scalar_output,
            &mut [],
        )?;
        bundle.cpu_projection_into(
            Mxfp4MatmulBackend::Avx2,
            &prepared,
            rows,
            &mut avx2_output,
            scratch,
        )?;
        gpu_request(&gpu_context, &inputs, rows, &mut xe_output)?;
        let scalar_vs_avx2 = compare_projection(&scalar_output, &avx2_output);
        let scalar_vs_xe = compare_projection(&scalar_output, &xe_output);
        let pass = comparison_passes(&scalar_vs_avx2) && comparison_passes(&scalar_vs_xe);
        let correctness = CorrectnessReport {
            rows,
            scalar_vs_avx2,
            scalar_vs_xe,
            pass,
        };
        if !pass {
            correctness_stopped = true;
            reports.push(ShapeReport {
                rows,
                correctness,
                trials: Vec::new(),
                combined_scalar_oracle: None,
                combined_avx2: None,
                combined_xe: None,
                combined_scalar_oracle_phases: None,
                combined_avx2_phases: None,
                combined_xe_phases: None,
                avx2_over_xe_speedup: None,
                conservative_speedup_ci95: None,
                useful_win: false,
                estimated_residency_break_even_requests: None,
            });
            break;
        }

        let orders = [
            ["scalar-oracle", "avx2", "xe"],
            ["avx2", "xe", "scalar-oracle"],
            ["xe", "scalar-oracle", "avx2"],
        ];
        let mut trials = Vec::new();
        let mut all_scalar = PhaseSamples::default();
        let mut all_avx2 = PhaseSamples::default();
        let mut all_xe = PhaseSamples::default();
        for (trial, order) in orders.into_iter().enumerate() {
            let mut scalar = PhaseSamples::default();
            let mut avx2 = PhaseSamples::default();
            let mut xe = PhaseSamples::default();
            for method in order {
                for _ in 0..10 {
                    match method {
                        "scalar-oracle" => {
                            cpu_request(
                                bundle,
                                Mxfp4MatmulBackend::Scalar,
                                &inputs,
                                rows,
                                &mut scalar_output,
                                &mut [],
                            )?;
                        }
                        "avx2" => {
                            cpu_request(
                                bundle,
                                Mxfp4MatmulBackend::Avx2,
                                &inputs,
                                rows,
                                &mut avx2_output,
                                scratch,
                            )?;
                        }
                        "xe" => {
                            gpu_request(&gpu_context, &inputs, rows, &mut xe_output)?;
                        }
                        _ => unreachable!(),
                    }
                }
                for _ in 0..30 {
                    let timing = match method {
                        "scalar-oracle" => cpu_request(
                            bundle,
                            Mxfp4MatmulBackend::Scalar,
                            &inputs,
                            rows,
                            &mut scalar_output,
                            &mut [],
                        )?,
                        "avx2" => cpu_request(
                            bundle,
                            Mxfp4MatmulBackend::Avx2,
                            &inputs,
                            rows,
                            &mut avx2_output,
                            scratch,
                        )?,
                        "xe" => gpu_request(&gpu_context, &inputs, rows, &mut xe_output)?,
                        _ => unreachable!(),
                    };
                    match method {
                        "scalar-oracle" => scalar.push(timing),
                        "avx2" => avx2.push(timing),
                        "xe" => xe.push(timing),
                        _ => unreachable!(),
                    }
                }
            }
            all_scalar.extend(&scalar);
            all_avx2.extend(&avx2);
            all_xe.extend(&xe);
            trials.push(TrialReport {
                trial,
                method_order: order.to_vec(),
                scalar_oracle: Distribution::from_samples(
                    &scalar.total_request_ns,
                    0x5800 + trial as u64,
                ),
                avx2: Distribution::from_samples(&avx2.total_request_ns, 0x5810 + trial as u64),
                xe: Distribution::from_samples(&xe.total_request_ns, 0x5820 + trial as u64),
                scalar_oracle_phases: scalar.distributions(0x5860 + trial as u64 * 16),
                avx2_phases: avx2.distributions(0x58a0 + trial as u64 * 16),
                xe_phases: xe.distributions(0x58e0 + trial as u64 * 16),
            });
        }
        let scalar_distribution =
            Distribution::from_samples(&all_scalar.total_request_ns, 0x5830 + rows as u64);
        let avx2_distribution =
            Distribution::from_samples(&all_avx2.total_request_ns, 0x5840 + rows as u64);
        let xe_distribution =
            Distribution::from_samples(&all_xe.total_request_ns, 0x5850 + rows as u64);
        let speedup = avx2_distribution.median_ns as f64 / xe_distribution.median_ns as f64;
        let confidence = [
            avx2_distribution.bootstrap_median_ci95_ns[0] as f64
                / xe_distribution.bootstrap_median_ci95_ns[1] as f64,
            avx2_distribution.bootstrap_median_ci95_ns[1] as f64
                / xe_distribution.bootstrap_median_ci95_ns[0] as f64,
        ];
        let plausible = [4, 8, 16, 32, 64].contains(&rows);
        let useful_win = plausible && speedup >= 1.25 && confidence[0] > 1.0;
        let per_request_saved = avx2_distribution
            .median_ns
            .saturating_sub(xe_distribution.median_ns);
        let break_even = (per_request_saved != 0).then_some(
            (session.info().creation_ns + residency_ns) as f64 / per_request_saved as f64,
        );
        reports.push(ShapeReport {
            rows,
            correctness,
            trials,
            combined_scalar_oracle: Some(scalar_distribution),
            combined_avx2: Some(avx2_distribution),
            combined_xe: Some(xe_distribution),
            combined_scalar_oracle_phases: Some(
                all_scalar.distributions(0x5a00 + rows as u64 * 16),
            ),
            combined_avx2_phases: Some(all_avx2.distributions(0x5b00 + rows as u64 * 16)),
            combined_xe_phases: Some(all_xe.distributions(0x5c00 + rows as u64 * 16)),
            avx2_over_xe_speedup: Some(speedup),
            conservative_speedup_ci95: Some(confidence),
            useful_win,
            estimated_residency_break_even_requests: break_even,
        });
    }
    let any_useful_win = reports.iter().any(|report| report.useful_win);
    let weight_allocation_ns = legacy_packed.as_ref().map_or(0, Buffer::allocation_ns)
        + legacy_scales.as_ref().map_or(0, Buffer::allocation_ns)
        + v2_weights.as_ref().map_or(0, Buffer::allocation_ns)
        + bias_buffer.allocation_ns();
    Ok(json!({
        "status": if correctness_stopped { "fail" } else if any_useful_win { "pass" } else { "fail" },
        "correctness_stopped_performance": correctness_stopped,
        "any_useful_win": any_useful_win,
        "variant": variant.name,
        "shapes": reports,
        "trial_count": 3,
        "warmups_per_method_per_trial": 10,
        "samples_per_method_per_trial": 30,
        "total_samples_per_method_per_shape": 90,
        "weight_residency": {
            "allocation_ns": weight_allocation_ns,
            "staging_ns": residency_ns,
            "reported_write_ns": weight_write_ns,
            "bytes": if variant.is_v2() { bundle.descriptor.xe_v2_bytes } else { bundle.descriptor.canonical_compact_bytes },
            "canonical_and_v2_co_resident": false,
        },
        "request_path": "activation residual-Q8 quantization, activation packing, one or four uploads according to ABI, argument setup, submission, wait, device kernel, readback, and BF16 output conversion",
        "phase_semantics": {
            "host_submission_ns": "CPU projection execution for CPU methods; queue/list enqueue time for Xe",
            "host_wait_ns": "zero for CPU; completion wait for Xe",
            "device_kernel_ns": "driver event timestamp; absent for CPU",
            "total_request_ns": "legacy end-to-end request measurement retained unchanged in scope"
        },
        "one_time_excluded": ["module creation", "persistent compact weight allocation and staging", "reusable scratch allocation"],
    }))
}

fn cpu_request(
    bundle: &ProjectionBundle,
    backend: Mxfp4MatmulBackend,
    inputs: &[f32],
    rows: usize,
    output: &mut [f32],
    scratch: &mut [u8],
) -> Result<PhaseTiming> {
    let started = Instant::now();
    let phase = Instant::now();
    let prepared = quantize_residual_rows(inputs)?;
    let quantization_ns = phase.elapsed().as_nanos() as u64;
    let phase = Instant::now();
    bundle.cpu_projection_into(backend, &prepared, rows, output, scratch)?;
    let projection_ns = phase.elapsed().as_nanos() as u64;
    let phase = Instant::now();
    let boundary = output
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    black_box(boundary);
    let bf16_conversion_ns = phase.elapsed().as_nanos() as u64;
    Ok(PhaseTiming {
        total_request_ns: started.elapsed().as_nanos() as u64,
        quantization_ns,
        host_submission_ns: projection_ns,
        bf16_conversion_ns,
        ..PhaseTiming::default()
    })
}

fn gpu_request(
    context: &GpuRequestContext<'_, '_>,
    inputs: &[f32],
    rows: usize,
    output: &mut [f32],
) -> Result<PhaseTiming> {
    let GpuRequestContext {
        session,
        gpu,
        v2_weights,
        bias,
        variant,
        local_size,
    } = context;
    let started = Instant::now();
    let phase = Instant::now();
    let prepared = quantize_residual_rows(inputs)?;
    let quantization_ns = phase.elapsed().as_nanos() as u64;
    let phase = Instant::now();
    let mut v1_packed = None;
    let mut v2_packed = None;
    match gpu {
        GpuBuffers::V1 { .. } => {
            v1_packed = Some(split_residual_activations(&prepared));
        }
        GpuBuffers::V2 { .. } => {
            v2_packed = Some(pack_activation_records(&prepared));
        }
    }
    let activation_packing_ns = phase.elapsed().as_nanos() as u64;
    let phase = Instant::now();
    match (gpu, v1_packed.as_ref(), v2_packed.as_ref()) {
        (
            GpuBuffers::V1 {
                primary,
                residual,
                primary_scales,
                residual_scales,
                ..
            },
            Some((primary_values, residual_values, primary_scale_values, residual_scale_values)),
            None,
        ) => {
            primary.write(primary_values)?;
            residual.write(residual_values)?;
            primary_scales.write(primary_scale_values)?;
            residual_scales.write(residual_scale_values)?;
        }
        (GpuBuffers::V2 { activations, .. }, None, Some(records)) => {
            activations.write(records)?;
        }
        _ => bail!("activation packing did not match the selected GPU ABI"),
    }
    let upload_ns = phase.elapsed().as_nanos() as u64;
    let rows_u32 = u32::try_from(rows)?;
    let columns_u32 = N as u32;
    let blocks_u32 = BLOCKS as u32;
    let mut argument_setup_ns = 0_u64;
    let mut host_submission_ns = 0_u64;
    let mut host_wait_ns = 0_u64;
    let mut device_kernel_ns = 0_u64;
    match gpu {
        GpuBuffers::V1 {
            output: gpu_output, ..
        } => {
            let phase = Instant::now();
            session.set_scalar(8, &rows_u32)?;
            session.set_scalar(9, &columns_u32)?;
            session.set_scalar(10, &blocks_u32)?;
            argument_setup_ns = phase.elapsed().as_nanos() as u64;
            let global = round_up(rows * N, *local_size);
            let timing = session.run([global, 1, 1], [*local_size, 1, 1], TIMEOUT_NS)?;
            host_submission_ns = timing.submit_ns;
            host_wait_ns = timing.wait_ns;
            device_kernel_ns = timing.device_ns.unwrap_or(0);
            let phase = Instant::now();
            gpu_output.read(output)?;
            let readback_ns = phase.elapsed().as_nanos() as u64;
            let phase = Instant::now();
            let boundary = output
                .iter()
                .map(|value| bf16::from_f32(*value).to_bits())
                .collect::<Vec<_>>();
            black_box(boundary);
            return Ok(PhaseTiming {
                total_request_ns: started.elapsed().as_nanos() as u64,
                quantization_ns,
                activation_packing_ns,
                upload_ns,
                argument_setup_ns,
                host_submission_ns,
                host_wait_ns,
                device_kernel_ns: (device_kernel_ns != 0).then_some(device_kernel_ns),
                readback_ns,
                bf16_conversion_ns: phase.elapsed().as_nanos() as u64,
            });
        }
        GpuBuffers::V2 {
            activations,
            output: gpu_output,
            splitk_terms,
        } if variant.kind == VariantKind::SplitkV2 => {
            let weights = v2_weights.context("split-K requires v2 weights")?;
            let terms = splitk_terms
                .as_ref()
                .context("split-K terms buffer is absent")?;
            let phase = Instant::now();
            session.select_kernel("mxfp4_splitk_terms_v2")?;
            session.set_buffer(0, weights)?;
            session.set_buffer(1, activations)?;
            session.set_buffer(2, terms)?;
            session.set_scalar(3, &rows_u32)?;
            session.set_scalar(4, &columns_u32)?;
            session.set_scalar(5, &blocks_u32)?;
            argument_setup_ns += phase.elapsed().as_nanos() as u64;
            let timing = session.run(
                [round_up(N, *local_size), BLOCKS, rows],
                [*local_size, 1, 1],
                TIMEOUT_NS,
            )?;
            host_submission_ns += timing.submit_ns;
            host_wait_ns += timing.wait_ns;
            device_kernel_ns += timing.device_ns.unwrap_or(0);

            let phase = Instant::now();
            session.select_kernel("mxfp4_splitk_reduce_v2")?;
            session.set_buffer(0, terms)?;
            session.set_buffer(1, bias)?;
            session.set_buffer(2, gpu_output)?;
            session.set_scalar(3, &rows_u32)?;
            session.set_scalar(4, &columns_u32)?;
            session.set_scalar(5, &blocks_u32)?;
            argument_setup_ns += phase.elapsed().as_nanos() as u64;
            let timing = session.run(
                [round_up(N, *local_size), rows, 1],
                [*local_size, 1, 1],
                TIMEOUT_NS,
            )?;
            host_submission_ns += timing.submit_ns;
            host_wait_ns += timing.wait_ns;
            device_kernel_ns += timing.device_ns.unwrap_or(0);
        }
        GpuBuffers::V2 {
            output: gpu_output, ..
        } => {
            let phase = Instant::now();
            session.set_scalar(4, &rows_u32)?;
            session.set_scalar(5, &columns_u32)?;
            session.set_scalar(6, &blocks_u32)?;
            argument_setup_ns = phase.elapsed().as_nanos() as u64;
            let dispatch_rows = rows / variant.rows_per_dispatch();
            let timing = session.run(
                [round_up(N, *local_size), dispatch_rows, 1],
                [*local_size, 1, 1],
                TIMEOUT_NS,
            )?;
            host_submission_ns = timing.submit_ns;
            host_wait_ns = timing.wait_ns;
            device_kernel_ns = timing.device_ns.unwrap_or(0);
            let _ = gpu_output;
        }
    }
    let gpu_output = match gpu {
        GpuBuffers::V1 { output, .. } | GpuBuffers::V2 { output, .. } => output,
    };
    let phase = Instant::now();
    gpu_output.read(output)?;
    let readback_ns = phase.elapsed().as_nanos() as u64;
    let phase = Instant::now();
    let boundary = output
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    black_box(boundary);
    Ok(PhaseTiming {
        total_request_ns: started.elapsed().as_nanos() as u64,
        quantization_ns,
        activation_packing_ns,
        upload_ns,
        argument_setup_ns,
        host_submission_ns,
        host_wait_ns,
        device_kernel_ns: (device_kernel_ns != 0).then_some(device_kernel_ns),
        readback_ns,
        bf16_conversion_ns: phase.elapsed().as_nanos() as u64,
    })
}

fn compare_projection(expected: &[f32], actual: &[f32]) -> Comparison {
    let mut comparison = Comparison {
        values: expected.len(),
        ..Comparison::default()
    };
    if expected.len() != actual.len() {
        comparison.tolerance_mismatches = expected.len().max(actual.len());
        return comparison;
    }
    for (&expected, &actual) in expected.iter().zip(actual) {
        if !expected.is_finite() || !actual.is_finite() {
            comparison.non_finite_mismatches += usize::from(expected.to_bits() != actual.to_bits());
            continue;
        }
        let absolute = (expected - actual).abs();
        let ulp = ulp_distance(expected, actual);
        comparison.max_absolute = comparison.max_absolute.max(absolute);
        comparison.max_ulp = comparison.max_ulp.max(ulp);
        comparison.tolerance_mismatches += usize::from(absolute > 1.0e-6 && ulp > 4);
        comparison.bf16_boundary_mismatches +=
            usize::from(bf16::from_f32(expected).to_bits() != bf16::from_f32(actual).to_bits());
    }
    comparison
}

fn comparison_passes(comparison: &Comparison) -> bool {
    comparison.non_finite_mismatches == 0
        && comparison.tolerance_mismatches == 0
        && comparison.bf16_boundary_mismatches == 0
}

fn ulp_distance(left: f32, right: f32) -> u32 {
    fn ordered(value: f32) -> i32 {
        let bits = value.to_bits() as i32;
        if bits < 0 {
            i32::MIN - bits
        } else {
            bits
        }
    }
    ordered(left).abs_diff(ordered(right))
}
