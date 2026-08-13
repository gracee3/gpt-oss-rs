use std::collections::HashSet;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use gpt_oss_cpu_kernels::{
    mxfp4_adjacent_to_split, CpuFeatures, CpuHardwareIdentity, KernelPath, Kernels, Mxfp4Block,
    Mxfp4MatmulBackend, Mxfp4MatmulProblem, Mxfp4MatrixView, Mxfp4ScratchRequirement,
    Mxfp4WeightLayout, Q8Block, Q8MatrixView, ResidualQ8Block, ResidualQ8MatrixView,
    MXFP4_PACKED_BYTES, QUANT_BLOCK_SIZE,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

const SCHEMA: &str = "gpt-oss-rs.mxfp4-matrix-benchmark/v1";

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum ActivationKind {
    Q8,
    ResidualQ8,
}

#[derive(Debug, Parser)]
#[command(about = "Interleaved multi-method MXFP4 matrix benchmark")]
struct Cli {
    #[arg(long, value_delimiter = ',', required = true)]
    m_values: Vec<usize>,
    #[arg(long)]
    n: usize,
    #[arg(long)]
    k: usize,
    #[arg(long, value_enum, default_value = "residual-q8")]
    activation: ActivationKind,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "scalar,avx2,avx512-vnni,auto"
    )]
    methods: Vec<Mxfp4MatmulBackend>,
    #[arg(long, default_value_t = 3)]
    warmups: usize,
    #[arg(long, default_value_t = 7)]
    trials: usize,
    #[arg(long, default_value_t = 5)]
    samples_per_trial: usize,
    #[arg(long, default_value_t = 4)]
    thread_policy: usize,
    /// Maximum core temperature permitted immediately before each kernel.
    #[arg(long, default_value_t = 65.0)]
    thermal_start_gate_c: f64,
    /// Reject the attempt if a measured kernel ends above this temperature.
    #[arg(long, default_value_t = 95.0)]
    thermal_end_ceiling_c: f64,
    #[arg(long, default_value_t = 250)]
    thermal_poll_ms: u64,
    #[arg(long, default_value_t = 900)]
    thermal_max_wait_seconds: u64,
    /// Abort at the first measured thermal-throttle event.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    reject_throttled: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    bias: bool,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Serialize)]
struct BenchmarkDocument {
    schema: &'static str,
    repository_commit: String,
    repository_dirty: bool,
    executable_sha256: String,
    cpu_identity: serde_json::Value,
    cpu_features: serde_json::Value,
    timer: &'static str,
    warmups: usize,
    trials: usize,
    samples_per_trial: usize,
    thread_policy: usize,
    thermal_start_gate_c: f64,
    thermal_end_ceiling_c: f64,
    thermal_poll_ms: u64,
    thermal_max_wait_seconds: u64,
    reject_throttled: bool,
    n: usize,
    k: usize,
    activation: ActivationKind,
    bias: bool,
    methods: Vec<String>,
    correctness: Vec<CorrectnessRecord>,
    samples: Vec<SampleRecord>,
}

#[derive(Debug, Serialize)]
struct CorrectnessRecord {
    m: usize,
    method: String,
    output_sha256: String,
    scalar_exact: bool,
    scratch_bytes: usize,
    scratch_alignment: usize,
}

#[derive(Debug, Serialize)]
struct SampleRecord {
    m: usize,
    n: usize,
    k: usize,
    method: String,
    requested_backend: String,
    effective_backend: String,
    trial: usize,
    sample: usize,
    order: usize,
    duration_ns: u64,
    start_temperature_c: f64,
    end_temperature_c: f64,
    package_throttle_time_delta_ms: u64,
    core_throttle_time_delta_ms: u64,
    scratch_bytes: usize,
    scratch_alignment: usize,
    output_sha256: String,
}

struct AlignedScratch {
    storage: Vec<u8>,
    offset: usize,
    requirement: Mxfp4ScratchRequirement,
}

impl AlignedScratch {
    fn new(requirement: Mxfp4ScratchRequirement) -> Self {
        let mut storage = vec![0_u8; requirement.size + requirement.alignment];
        let offset = if requirement.size == 0 {
            0
        } else {
            (requirement.alignment - storage.as_ptr() as usize % requirement.alignment)
                % requirement.alignment
        };
        // Materialize the allocation before timing.
        black_box(&mut storage);
        Self {
            storage,
            offset,
            requirement,
        }
    }

    fn bytes(&mut self) -> &mut [u8] {
        &mut self.storage[self.offset..self.offset + self.requirement.size]
    }
}

#[derive(Clone, Copy)]
enum Activations<'a> {
    Q8(Q8MatrixView<'a>),
    ResidualQ8(ResidualQ8MatrixView<'a>),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    validate_cli(&cli)?;
    let (repository_commit, repository_dirty) = repository_identity();
    if repository_dirty {
        bail!("benchmark evidence requires a clean repository")
    }
    let executable = std::env::current_exe().context("resolve benchmark executable")?;
    let executable_sha256 = file_sha256(&executable)?;
    let identity = CpuHardwareIdentity::detect();
    let features = CpuFeatures::detect();
    let kernels = Kernels::new(KernelPath::Auto)?;
    let blocks = cli.k / QUANT_BLOCK_SIZE;
    let maximum_m = *cli.m_values.iter().max().unwrap();
    let canonical = make_weights(cli.n, blocks);
    let packed = pack_x8(&canonical, cli.n, blocks);
    let weights = Mxfp4MatrixView::new(
        &packed,
        cli.n,
        blocks,
        Mxfp4WeightLayout::InterleavedSplitX8V2,
    )?;
    let q8 = make_q8(maximum_m, blocks);
    let residual = make_residual(&q8);
    let bias = cli.bias.then(|| make_bias(cli.n));
    let bias = bias.as_deref();
    let mut correctness = Vec::new();
    let mut samples = Vec::new();

    for (shape_index, &m) in cli.m_values.iter().enumerate() {
        let activations = match cli.activation {
            ActivationKind::Q8 => Activations::Q8(Q8MatrixView::new(&q8, m, blocks, blocks)?),
            ActivationKind::ResidualQ8 => {
                Activations::ResidualQ8(ResidualQ8MatrixView::new(&residual, m, blocks, blocks)?)
            }
        };
        let mut scalar_output = vec![f32::NAN; m * cli.n];
        execute(
            kernels,
            Mxfp4MatmulBackend::Scalar,
            weights,
            activations,
            bias,
            &mut scalar_output,
            &mut [],
        )?;
        let scalar_bits = output_bits(&scalar_output);

        let mut method_state = Vec::with_capacity(cli.methods.len());
        for &method in &cli.methods {
            let mut output = vec![f32::NAN; m * cli.n];
            let requirement = requirement(method, weights, activations, bias, &mut output)?;
            let mut scratch = AlignedScratch::new(requirement);
            execute(
                kernels,
                method,
                weights,
                activations,
                bias,
                &mut output,
                scratch.bytes(),
            )?;
            let bits = output_bits(&output);
            if bits != scalar_bits {
                bail!(
                    "{} differed from scalar for M={m}, N={}, K={}",
                    method,
                    cli.n,
                    cli.k
                )
            }
            correctness.push(CorrectnessRecord {
                m,
                method: method.to_string(),
                output_sha256: bytes_sha256(&bits),
                scalar_exact: true,
                scratch_bytes: requirement.size,
                scratch_alignment: requirement.alignment,
            });
            method_state.push((method, output, scratch));
        }

        for warmup in 0..cli.warmups {
            let start = (shape_index + warmup) % method_state.len();
            for offset in 0..method_state.len() {
                let index = (start + offset) % method_state.len();
                let (method, output, scratch) = &mut method_state[index];
                wait_for_thermal_gate(&cli)?;
                execute(
                    kernels,
                    *method,
                    weights,
                    activations,
                    bias,
                    black_box(output),
                    black_box(scratch.bytes()),
                )?;
            }
        }

        for trial in 0..cli.trials {
            for sample in 0..cli.samples_per_trial {
                let start = (shape_index + trial + sample) % method_state.len();
                for order in 0..method_state.len() {
                    let index = (start + order) % method_state.len();
                    let (method, output, scratch) = &mut method_state[index];
                    let start_temperature_c = wait_for_thermal_gate(&cli)?;
                    let throttle_before = throttle_time_ms()?;
                    let timer = Instant::now();
                    execute(
                        kernels,
                        *method,
                        weights,
                        activations,
                        bias,
                        black_box(output),
                        black_box(scratch.bytes()),
                    )?;
                    let duration_ns = timer.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);
                    let throttle_after = throttle_time_ms()?;
                    let end_temperature_c = core_temperature_c()?;
                    let package_throttle_time_delta_ms =
                        throttle_after.0.saturating_sub(throttle_before.0);
                    let core_throttle_time_delta_ms =
                        throttle_after.1.saturating_sub(throttle_before.1);
                    if end_temperature_c > cli.thermal_end_ceiling_c {
                        bail!(
                            "thermal ceiling exceeded after M={m} method={method}: \
                             {end_temperature_c:.1} C > {:.1} C",
                            cli.thermal_end_ceiling_c
                        )
                    }
                    if cli.reject_throttled
                        && (package_throttle_time_delta_ms != 0 || core_throttle_time_delta_ms != 0)
                    {
                        bail!(
                            "thermal throttling during M={m} method={method}: package delta \
                             {package_throttle_time_delta_ms} ms, core delta \
                             {core_throttle_time_delta_ms} ms"
                        )
                    }
                    samples.push(SampleRecord {
                        m,
                        n: cli.n,
                        k: cli.k,
                        method: method.to_string(),
                        requested_backend: method.to_string(),
                        effective_backend: method.resolved_for_rows(m).to_string(),
                        trial,
                        sample,
                        order,
                        duration_ns,
                        start_temperature_c,
                        end_temperature_c,
                        package_throttle_time_delta_ms,
                        core_throttle_time_delta_ms,
                        scratch_bytes: scratch.requirement.size,
                        scratch_alignment: scratch.requirement.alignment,
                        output_sha256: bytes_sha256(&output_bits(output)),
                    });
                }
            }
        }
    }

    let document = BenchmarkDocument {
        schema: SCHEMA,
        repository_commit,
        repository_dirty,
        executable_sha256,
        cpu_identity: serde_json::json!({
            "vendor": identity.vendor,
            "family": identity.family,
            "model": identity.model,
            "stepping": identity.stepping,
            "xcr0": identity.xcr0,
            "osxsave": identity.osxsave,
            "physical_cores": identity.physical_cores,
            "logical_cpus": identity.logical_cpus,
            "microcode": identity.microcode.map(|value| format!("0x{value:x}")),
            "profile_key": identity.profile_key(),
        }),
        cpu_features: serde_json::to_value(features_json(features))?,
        timer: "std::time::Instant around kernel execution only",
        warmups: cli.warmups,
        trials: cli.trials,
        samples_per_trial: cli.samples_per_trial,
        thread_policy: cli.thread_policy,
        thermal_start_gate_c: cli.thermal_start_gate_c,
        thermal_end_ceiling_c: cli.thermal_end_ceiling_c,
        thermal_poll_ms: cli.thermal_poll_ms,
        thermal_max_wait_seconds: cli.thermal_max_wait_seconds,
        reject_throttled: cli.reject_throttled,
        n: cli.n,
        k: cli.k,
        activation: cli.activation,
        bias: cli.bias,
        methods: cli.methods.iter().map(ToString::to_string).collect(),
        correctness,
        samples,
    };
    let mut encoded = serde_json::to_vec_pretty(&document)?;
    encoded.push(b'\n');
    gpt_oss_evidence::atomic_write_new(&cli.output, &encoded)?;
    println!("{}", String::from_utf8(encoded)?);
    Ok(())
}

fn validate_cli(cli: &Cli) -> Result<()> {
    if cli.m_values.contains(&0)
        || cli.n == 0
        || cli.k == 0
        || !cli.k.is_multiple_of(QUANT_BLOCK_SIZE)
        || cli.methods.is_empty()
        || cli.warmups == 0
        || cli.trials < 7
        || cli.trials * cli.samples_per_trial < 30
        || cli.thread_policy == 0
        || !cli.thermal_start_gate_c.is_finite()
        || !cli.thermal_end_ceiling_c.is_finite()
        || cli.thermal_start_gate_c <= 0.0
        || cli.thermal_start_gate_c >= cli.thermal_end_ceiling_c
        || cli.thermal_poll_ms == 0
        || cli.thermal_max_wait_seconds == 0
    {
        bail!("invalid shape or protocol; require K%32=0, >=7 trials, and >=30 samples")
    }
    let methods = cli
        .methods
        .iter()
        .map(ToString::to_string)
        .collect::<HashSet<_>>();
    if methods.len() != cli.methods.len()
        || !methods.contains("scalar")
        || !methods.contains("avx2")
        || !methods.contains("avx512-vnni")
        || !methods.contains("auto")
    {
        bail!("methods must contain scalar, avx2, avx512-vnni, and auto exactly once")
    }
    Ok(())
}

fn core_temperature_c() -> Result<f64> {
    for entry in fs::read_dir("/sys/class/hwmon").context("read hwmon directory")? {
        let path = entry?.path();
        let Ok(name) = fs::read_to_string(path.join("name")) else {
            continue;
        };
        if name.trim() != "coretemp" {
            continue;
        }
        let mut temperatures = Vec::new();
        for sensor in fs::read_dir(&path).context("read coretemp sensors")? {
            let sensor = sensor?.path();
            let Some(file_name) = sensor.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if !file_name.starts_with("temp") || !file_name.ends_with("_input") {
                continue;
            }
            if let Ok(value) = fs::read_to_string(sensor) {
                if let Ok(value) = value.trim().parse::<f64>() {
                    temperatures.push(value / 1_000.0);
                }
            }
        }
        return temperatures
            .into_iter()
            .reduce(f64::max)
            .context("coretemp exposes no readable temperature sensors");
    }
    bail!("coretemp hwmon device is unavailable")
}

fn wait_for_thermal_gate(cli: &Cli) -> Result<f64> {
    let started = Instant::now();
    loop {
        let temperature = core_temperature_c()?;
        if temperature <= cli.thermal_start_gate_c {
            return Ok(temperature);
        }
        if started.elapsed() >= Duration::from_secs(cli.thermal_max_wait_seconds) {
            bail!(
                "thermal gate remained at {temperature:.1} C above {:.1} C for {} seconds",
                cli.thermal_start_gate_c,
                cli.thermal_max_wait_seconds
            )
        }
        thread::sleep(Duration::from_millis(cli.thermal_poll_ms));
    }
}

fn throttle_time_ms() -> Result<(u64, u64)> {
    let package = fs::read_to_string(
        "/sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_total_time_ms",
    )
    .context("read package throttle time")?
    .trim()
    .parse::<u64>()
    .context("parse package throttle time")?;
    let mut core = 0_u64;
    for cpu in 0..4 {
        let path = format!(
            "/sys/devices/system/cpu/cpu{cpu}/thermal_throttle/core_throttle_total_time_ms"
        );
        core = core.saturating_add(
            fs::read_to_string(&path)
                .with_context(|| format!("read {path}"))?
                .trim()
                .parse::<u64>()
                .with_context(|| format!("parse {path}"))?,
        );
    }
    Ok((package, core))
}

fn requirement(
    backend: Mxfp4MatmulBackend,
    weights: Mxfp4MatrixView<'_>,
    activations: Activations<'_>,
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> Result<Mxfp4ScratchRequirement> {
    let problem = problem(weights, activations, bias, output)?;
    Ok(backend.scratch_requirement(&problem)?)
}

fn execute(
    kernels: Kernels,
    backend: Mxfp4MatmulBackend,
    weights: Mxfp4MatrixView<'_>,
    activations: Activations<'_>,
    bias: Option<&[f32]>,
    output: &mut [f32],
    scratch: &mut [u8],
) -> Result<()> {
    let problem = problem(weights, activations, bias, output)?;
    kernels.mxfp4_matmul(backend, problem, scratch)?;
    Ok(())
}

fn problem<'a>(
    weights: Mxfp4MatrixView<'a>,
    activations: Activations<'a>,
    bias: Option<&'a [f32]>,
    output: &'a mut [f32],
) -> Result<Mxfp4MatmulProblem<'a>> {
    Ok(match activations {
        Activations::Q8(view) => {
            Mxfp4MatmulProblem::new_q8(weights, view, bias, output, weights.rows())?
        }
        Activations::ResidualQ8(view) => {
            Mxfp4MatmulProblem::new_residual_q8(weights, view, bias, output, weights.rows())?
        }
    })
}

fn make_weights(rows: usize, blocks: usize) -> Vec<Mxfp4Block> {
    (0..rows * blocks)
        .map(|index| Mxfp4Block {
            scale: if index % 127 == 0 {
                0
            } else {
                124 + (index % 7) as u8
            },
            packed: std::array::from_fn::<_, MXFP4_PACKED_BYTES, _>(|byte| {
                ((index * 13 + byte * 7) as u8 & 0x0f)
                    | (((index * 5 + byte * 11 + 3) as u8 & 0x0f) << 4)
            }),
        })
        .collect()
}

fn make_q8(rows: usize, blocks: usize) -> Vec<Q8Block> {
    (0..rows * blocks)
        .map(|index| Q8Block {
            scale: 0.0005 + (index % 251) as f32 * 0.000001,
            values: std::array::from_fn(|lane| {
                (((index * 31 + lane * 17) % 255) as i16 - 127) as i8
            }),
        })
        .collect()
}

fn make_residual(primary: &[Q8Block]) -> Vec<ResidualQ8Block> {
    primary
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, primary)| ResidualQ8Block {
            residual: Q8Block {
                scale: primary.scale / 127.0,
                values: std::array::from_fn(|lane| ((index + lane * 3) % 15) as i8 - 7),
            },
            primary,
        })
        .collect()
}

fn make_bias(rows: usize) -> Vec<f32> {
    (0..rows)
        .map(|index| index as f32 * 0.00003125 - 0.25)
        .collect()
}

fn pack_x8(weights: &[Mxfp4Block], rows: usize, blocks: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(weights.len() * 17);
    for group in 0..rows / 8 {
        for block in 0..blocks {
            for lane in 0..8 {
                data.push(weights[((group * 8 + lane) * blocks) + block].scale);
            }
            let split = std::array::from_fn::<_, 8, _>(|lane| {
                mxfp4_adjacent_to_split(weights[((group * 8 + lane) * blocks) + block].packed)
            });
            for chunk in 0..2 {
                for row in &split {
                    data.extend_from_slice(&row[chunk * 8..chunk * 8 + 8]);
                }
            }
        }
    }
    for row in rows / 8 * 8..rows {
        for block in 0..blocks {
            let weight = &weights[row * blocks + block];
            data.push(weight.scale);
            data.extend_from_slice(&weight.packed);
        }
    }
    data
}

fn output_bits(output: &[f32]) -> Vec<u8> {
    output
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn bytes_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_sha256(path: &std::path::Path) -> Result<String> {
    Ok(bytes_sha256(&std::fs::read(path)?))
}

fn repository_identity() -> (String, bool) {
    let output = |arguments: &[&str]| {
        Command::new("git")
            .args(arguments)
            .output()
            .ok()
            .filter(|result| result.status.success())
            .map(|result| String::from_utf8_lossy(&result.stdout).trim().to_owned())
    };
    (
        output(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into()),
        output(&["status", "--porcelain"]).is_none_or(|value| !value.is_empty()),
    )
}

fn features_json(features: CpuFeatures) -> serde_json::Value {
    serde_json::json!({
        "avx2": features.avx2,
        "fma": features.fma,
        "avx512_f": features.avx512_f,
        "avx512_bw": features.avx512_bw,
        "avx512_vl": features.avx512_vl,
        "avx512_vnni": features.avx512_vnni,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_cli() -> Cli {
        Cli {
            m_values: vec![1, 2, 3],
            n: 16,
            k: 32,
            activation: ActivationKind::ResidualQ8,
            methods: vec![
                Mxfp4MatmulBackend::Scalar,
                Mxfp4MatmulBackend::Avx2,
                Mxfp4MatmulBackend::Avx512Vnni,
                Mxfp4MatmulBackend::Auto,
            ],
            warmups: 3,
            trials: 7,
            samples_per_trial: 5,
            thread_policy: 4,
            bias: true,
            output: PathBuf::from("unused"),
        }
    }

    #[test]
    fn protocol_requires_all_controls_and_thirty_samples() {
        let mut cli = valid_cli();
        validate_cli(&cli).unwrap();
        cli.samples_per_trial = 4;
        assert!(validate_cli(&cli).is_err());
        cli = valid_cli();
        cli.methods.pop();
        assert!(validate_cli(&cli).is_err());
        cli = valid_cli();
        cli.k = 33;
        assert!(validate_cli(&cli).is_err());
    }

    #[test]
    fn deterministic_fixture_pack_round_trips_canonical_weights() {
        let canonical = make_weights(17, 3);
        let packed = pack_x8(&canonical, 17, 3);
        let view =
            Mxfp4MatrixView::new(&packed, 17, 3, Mxfp4WeightLayout::InterleavedSplitX8V2).unwrap();
        for row in 0..17 {
            for block in 0..3 {
                let expected = &canonical[row * 3 + block];
                let actual = view.block(row, block).unwrap();
                assert_eq!(actual.scale, expected.scale);
                assert_eq!(actual.packed, expected.packed);
            }
        }
    }
}
