use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use gpt_oss_cpu_kernels::{KernelPath, Mxfp4MatmulBackend};
use gpt_oss_model_runner::heterogeneous::{
    exact_rank_ordered_reduction_reference, exact_selected_expert_reference, pack_routes_bounded,
    CanonicalExpertContribution, CudaExactRouter, CudaLayerOwnerShell, CudaRankOrderedReducer,
    CudaResultRelay, ExactRouterWeightsView, GptOssExpertKey, NativeMxfp4ExpertView,
    PreparedRankOrderedReduction, RelayPinnedPools, DOWN_BIAS_VALUES, DOWN_BLOCK_BYTES,
    DOWN_SCALE_BYTES, GATE_UP_BIAS_VALUES, GATE_UP_BLOCK_BYTES, GATE_UP_SCALE_BYTES,
    GPT_OSS_REDUCTION_OUTPUT_BYTES, GPT_OSS_REDUCTION_TRACE_BYTES,
};
use gpt_oss_model_runner::model_loader::gpt_oss_native::GptOssCheckpointView;
use gpt_oss_model_runner::model_loader::owner_selective::OwnerSelectiveConstructor;
use gpt_oss_model_runner::{
    CpuExpertProjection, CpuLayerTrace, CpuModelRunner, CpuModelRunnerOptions, CpuTensorStore,
};
use half::bf16;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(feature = "heterogeneous-test-faults")]
use gpt_oss_model_runner::heterogeneous::LayerOwnerInjectedFault;

const EXPECTED_ROUTE: [usize; 4] = [31, 21, 22, 6];

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    native_model: PathBuf,
    #[arg(long)]
    cpu_repack_cache: PathBuf,
    #[arg(long)]
    owner_cache: PathBuf,
    #[arg(long)]
    placement: PathBuf,
    #[arg(long)]
    retained_trace: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[cfg(feature = "heterogeneous-test-faults")]
    #[arg(long)]
    exercise_shell_faults: bool,
}

#[derive(Deserialize)]
struct ControlDocument {
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
}

#[derive(Serialize)]
struct H6aEvidence {
    schema: &'static str,
    pre_moe_authority: &'static str,
    post_router_authority: &'static str,
    model_config_sha256: String,
    model_index_sha256: String,
    native_mapping_sha256: String,
    placement_sha256: String,
    retained_trace_sha256: String,
    layer_owner_pci_bus_id: String,
    token_id: u32,
    position: usize,
    prior_kv_rows: usize,
    prior_kv_bytes: usize,
    owner_shell_work_bytes: usize,
    owner_shell_host_staging_bytes: usize,
    fault_feature_enabled: bool,
    fault_exercise_requested: bool,
    shell_faults_drained: Vec<&'static str>,
    shell_fault_retries_passed: bool,
    owner_shell_kernel_elapsed_ms: [f32; 2],
    selected_experts: Vec<usize>,
    routing_weights_bf16_bits: Vec<u16>,
    router_logits_sha256: String,
    router_input_device_handoff_bytes: usize,
    router_input_execution_host_bytes: usize,
    router_evidence_activation_d2h_bytes: usize,
    router_descriptor_d2h_bytes: usize,
    cpu_authority_contribution_h2d_bytes: usize,
    exact_expert_output_sha256: Vec<String>,
    moe_output_sha256: String,
    reducer_output_device_handoff_bytes: usize,
    reducer_output_execution_host_bytes: usize,
    reducer_evidence_d2h_bytes: usize,
    layer_output_sha256: String,
    reduction_kernel_elapsed_ms: f32,
    boundaries: Vec<BoundaryEvidence>,
    owner_shell_prefix_repeat_exact: bool,
    passed: bool,
}

#[derive(Serialize)]
struct BoundaryEvidence {
    name: &'static str,
    values: usize,
    sha256: String,
    bit_exact: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let control: ControlDocument = serde_json::from_slice(&std::fs::read(&cli.retained_trace)?)?;
    let token_id = *control
        .generated_token_ids
        .first()
        .context("retained control has no generated token")?;
    if token_id != 200_005 || control.prompt_token_ids.len() != 63 {
        bail!("retained control is not the pinned 63-token/200005 fixture");
    }

    let mut cpu = CpuModelRunner::load_with_options(
        &cli.model,
        &cli.cpu_repack_cache,
        CpuModelRunnerOptions {
            kernel_path: KernelPath::Auto,
            matmul_backend: Mxfp4MatmulBackend::Auto,
            threads: 8,
            context_cap: 128,
            expert_projection: CpuExpertProjection::ResidualQ8,
            xe: None,
            profile_capacity_bytes: None,
        },
    )?;
    cpu.prefill(&control.prompt_token_ids)?;
    let cache = cpu
        .caches()
        .first()
        .context("CPU authority has no layer-0 cache")?
        .oracle_snapshot();
    let (_, trace) = cpu.decode_trace(token_id, &[0], 8, 1)?;
    let authority = trace.layers.first().context("CPU layer-0 trace missing")?;
    if authority.selected_experts != EXPECTED_ROUTE {
        bail!(
            "real CPU route changed: {:?} != {:?}",
            authority.selected_experts,
            EXPECTED_ROUTE
        );
    }
    let config = cpu.config().clone();
    drop(cpu);

    let store = CpuTensorStore::open(&cli.model)?;
    let embedding_tensor = store.tensor("model.embed_tokens.weight")?;
    let embeddings = embedding_tensor.bf16()?;
    let hidden_start = token_id as usize * config.hidden_size;
    let hidden = embeddings[hidden_start..hidden_start + config.hidden_size]
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    drop(store);

    let manifest = serde_json::from_slice(&std::fs::read(&cli.placement)?)?;
    let checkpoint = GptOssCheckpointView::open(&cli.native_model)?;
    let native_mapping_sha256 = checkpoint.mapping_sha256().to_owned();
    let router_weights = tensor_u16(
        &checkpoint,
        "model.layers.0.mlp.router.weight",
        config.num_local_experts * config.hidden_size,
    )?;
    let router_bias = tensor_u16(
        &checkpoint,
        "model.layers.0.mlp.router.bias",
        config.num_local_experts,
    )?;
    let constructor = OwnerSelectiveConstructor::new(&cli.owner_cache);
    let model = constructor.construct(checkpoint, &manifest, |_| Ok(()))?;
    let placement_sha256 = model.placement().manifest_hash().to_owned();
    let layer_owner_pci_bus_id = model
        .placement()
        .layer_owner()
        .stable_id
        .pci_bus_id
        .to_string();
    let mut shell = CudaLayerOwnerShell::new(&model, &config)?;
    #[allow(unused_mut)]
    let mut shell_faults_drained = Vec::with_capacity(5);
    #[allow(unused_mut)]
    let mut shell_fault_retries_passed = false;
    #[cfg(feature = "heterogeneous-test-faults")]
    let fault_exercise_requested = cli.exercise_shell_faults;
    #[cfg(not(feature = "heterogeneous-test-faults"))]
    let fault_exercise_requested = false;
    #[cfg(feature = "heterogeneous-test-faults")]
    if cli.exercise_shell_faults {
        for (fault, label) in [
            (
                LayerOwnerInjectedFault::SubmitAfterPriorKeyEnqueue,
                "submit_after_prior_key_enqueue",
            ),
            (
                LayerOwnerInjectedFault::TerminalDrain,
                "terminal_fallback_drain",
            ),
            (
                LayerOwnerInjectedFault::BoundaryDownloadAfterFirstEnqueue,
                "boundary_download_after_first_enqueue",
            ),
        ] {
            shell.inject_next_failure(fault)?;
            if shell
                .execute_layer0_decode(&model, &config, token_id, 63, &cache)
                .is_ok()
                || !shell.last_fault_drained()
                || shell.is_poisoned_for_test()
            {
                bail!("layer-owner fault {label} did not drain and remain safely reusable");
            }
            shell_faults_drained.push(label);
            shell.execute_layer0_decode(&model, &config, token_id, 63, &cache)?;
        }
    }
    let first = shell.execute_layer0_decode(&model, &config, token_id, 63, &cache)?;
    let second = shell.execute_layer0_decode(&model, &config, token_id, 63, &cache)?;

    let expected = ExpectedBoundaries::new(authority, hidden);
    let actual = ActualBoundaries::new(&first);
    let repeated = ActualBoundaries::new(&second);
    let mut boundaries = Vec::with_capacity(expected.values.len());
    for ((name, expected), (repeated_name, repeated)) in
        expected.values.iter().zip(repeated.values.iter())
    {
        let actual = actual
            .values
            .iter()
            .find(|(actual_name, _)| actual_name == name)
            .map(|(_, values)| values)
            .context("actual boundary missing")?;
        if name != repeated_name {
            bail!("repeat boundary ordering changed");
        }
        exact(name, expected, actual)?;
        exact(name, actual, repeated)?;
        boundaries.push(BoundaryEvidence {
            name,
            values: actual.len(),
            sha256: hash_u16(actual),
            bit_exact: true,
        });
    }

    const GENERATION: u64 = 6_001;
    let mut router = CudaExactRouter::new(
        model.placement().layer_owner().stable_id.clone(),
        1,
        ExactRouterWeightsView {
            experts: config.num_local_experts,
            weight_bf16_bits: &router_weights,
            bias_bf16_bits: &router_bias,
        },
    )?;
    let pools = RelayPinnedPools::warm_exact(&router, 1)?;
    let mut reservation = pools.try_reserve_all(GENERATION)?;
    let routed = shell.route_resident_decode(
        &mut router,
        0,
        model.placement().placement_epoch(),
        &mut reservation.source_activation,
        &mut reservation.route_descriptors,
        None,
    )?;
    exact(
        "router_input_evidence",
        &first.router_input_bf16_bits,
        &routed.batch.activation_bf16_bits,
    )?;
    exact(
        "router_logits",
        &bits(&authority.router_logits),
        &routed.router_logits_bf16_bits,
    )?;
    let routed_ids = routed
        .batch
        .routes
        .iter()
        .map(|route| usize::from(route.expert_id))
        .collect::<Vec<_>>();
    let routed_weights = routed
        .batch
        .routes
        .iter()
        .map(|route| route.weight_bf16_bits)
        .collect::<Vec<_>>();
    if routed_ids != authority.selected_experts
        || routed_weights != bits(&authority.routing_weights)
    {
        bail!("GPU-authored route IDs or BF16 weights diverged from CPU authority");
    }
    let plan = pack_routes_bounded(&routed.batch, model.placement())?;
    let prepared =
        PreparedRankOrderedReduction::prepare(&routed.batch, model.placement(), GENERATION)?;
    let descriptors = prepared.expected_results().to_vec();
    // The retained runner is ResidualQ8. Its dense/K/V/router boundaries are
    // authoritative here, but its expert/MoE/layer outputs are not the exact
    // H2 selected-expert contract. Recompute every contribution directly from
    // the native packed expert view using the real GPU-authored route.
    let authority_outputs = descriptors
        .iter()
        .map(|descriptor| {
            exact_expert_output(
                model.checkpoint(),
                0,
                descriptor.expert_id,
                &routed.batch.activation_bf16_bits,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let contributions = descriptors
        .iter()
        .cloned()
        .zip(authority_outputs.iter().cloned())
        .map(
            |(descriptor, output_bf16_bits)| CanonicalExpertContribution {
                descriptor,
                output_bf16_bits,
            },
        )
        .collect::<Vec<_>>();
    let exact_reduction =
        exact_rank_ordered_reduction_reference(&routed.batch, model.placement(), &contributions)?;
    let exact_expert_output_sha256 = authority_outputs
        .iter()
        .map(|output| hash_u16(output))
        .collect::<Vec<_>>();
    let mut relay = CudaResultRelay::new(&router, 1)?;
    relay.bind_decode_generation(GENERATION, &plan)?;
    let cpu_authority_contribution_h2d_bytes =
        relay.upload_cpu_authority_control(GENERATION, &descriptors, authority_outputs)?;
    let mut reducer = CudaRankOrderedReducer::new(&relay)?;
    let reduced = reducer.reduce_relay(&mut relay, prepared)?;
    exact(
        "moe_output",
        &exact_reduction.output_bf16_bits,
        &reduced.output_bf16_bits,
    )?;
    let expected_layer_output = exact_residual(
        &first.post_attention_residual_bf16_bits,
        &exact_reduction.output_bf16_bits,
    )?;
    #[cfg(feature = "heterogeneous-test-faults")]
    if cli.exercise_shell_faults {
        for (fault, label) in [
            (
                LayerOwnerInjectedFault::FinalResidualAfterD2dEnqueue,
                "final_residual_after_d2d_enqueue",
            ),
            (
                LayerOwnerInjectedFault::FinalOutputAfterD2hEnqueue,
                "final_output_after_d2h_enqueue",
            ),
        ] {
            shell.inject_next_failure(fault)?;
            if shell.finish_layer_residual_resident(&reducer).is_ok()
                || !shell.last_fault_drained()
                || shell.is_poisoned_for_test()
            {
                bail!(
                    "layer-owner residual fault {label} did not drain and remain safely reusable"
                );
            }
            shell_faults_drained.push(label);
            let retry = shell.finish_layer_residual_resident(&reducer)?;
            exact("layer_output_fault_retry", &expected_layer_output, &retry)?;
        }
        if shell_faults_drained.len() != 5 {
            bail!("not all five layer-owner lifecycle faults were exercised");
        }
        shell_fault_retries_passed = true;
    }
    let layer_output = shell.finish_layer_residual_resident(&reducer)?;
    exact("layer_output", &expected_layer_output, &layer_output)?;
    for (name, values) in [
        ("router_logits", routed.router_logits_bf16_bits.as_slice()),
        ("moe_output", reduced.output_bf16_bits.as_slice()),
        ("layer_output", layer_output.as_slice()),
    ] {
        boundaries.push(BoundaryEvidence {
            name,
            values: values.len(),
            sha256: hash_u16(values),
            bit_exact: true,
        });
    }
    reservation.release_drained()?;

    let evidence = H6aEvidence {
        schema: "gpt-oss-rs.heterogeneous-layer-oracle-h6a/v3",
        pre_moe_authority: "retained-residual-q8-dense-kv-router-only",
        post_router_authority: "native-mxfp4-exact-selected-expert-reference",
        model_config_sha256: hash_file(&cli.model.join("config.json"))?,
        model_index_sha256: hash_file(&cli.model.join("model.safetensors.index.json"))?,
        native_mapping_sha256,
        placement_sha256,
        retained_trace_sha256: hash_file(&cli.retained_trace)?,
        layer_owner_pci_bus_id,
        token_id,
        position: 63,
        prior_kv_rows: cache.len,
        prior_kv_bytes: (cache.keys_bf16_bits.len() + cache.values_bf16_bits.len()) * 2,
        owner_shell_work_bytes: shell.owned_device_bytes(),
        owner_shell_host_staging_bytes: shell.owned_host_staging_bytes(),
        fault_feature_enabled: cfg!(feature = "heterogeneous-test-faults"),
        fault_exercise_requested,
        shell_faults_drained,
        shell_fault_retries_passed,
        owner_shell_kernel_elapsed_ms: [first.kernel_elapsed_ms, second.kernel_elapsed_ms],
        selected_experts: authority.selected_experts.clone(),
        routing_weights_bf16_bits: routed_weights,
        router_logits_sha256: hash_u16(&routed.router_logits_bf16_bits),
        router_input_device_handoff_bytes: first.router_input_bf16_bits.len() * size_of::<u16>(),
        router_input_execution_host_bytes: 0,
        router_evidence_activation_d2h_bytes: routed.source_d2h_bytes,
        router_descriptor_d2h_bytes: routed.descriptor_d2h_bytes,
        cpu_authority_contribution_h2d_bytes,
        exact_expert_output_sha256,
        moe_output_sha256: hash_u16(&reduced.output_bf16_bits),
        reducer_output_device_handoff_bytes: GPT_OSS_REDUCTION_OUTPUT_BYTES,
        reducer_output_execution_host_bytes: 0,
        reducer_evidence_d2h_bytes: GPT_OSS_REDUCTION_OUTPUT_BYTES + GPT_OSS_REDUCTION_TRACE_BYTES,
        layer_output_sha256: hash_u16(&layer_output),
        reduction_kernel_elapsed_ms: reduced.kernel_elapsed_ms,
        boundaries,
        owner_shell_prefix_repeat_exact: true,
        passed: true,
    };
    shell.drain()?;
    router.drain()?;
    model.drain()?;
    write_json(&cli.output, &evidence)
}

struct ExpectedBoundaries {
    values: Vec<(&'static str, Vec<u16>)>,
}

impl ExpectedBoundaries {
    fn new(trace: &CpuLayerTrace, hidden: Vec<u16>) -> Self {
        Self {
            values: vec![
                ("hidden", hidden),
                ("input_norm", bits(&trace.input_norm)),
                ("query_after_rope", bits(&trace.query_after_rope)),
                ("key_after_rope", bits(&trace.key_after_rope)),
                ("value_projection", bits(&trace.value_projection)),
                ("attention_context", bits(&trace.attention_context)),
                ("attention_projection", bits(&trace.attention_projection)),
                (
                    "post_attention_residual",
                    bits(&trace.post_attention_residual),
                ),
                ("router_input", bits(&trace.router_input)),
            ],
        }
    }
}

struct ActualBoundaries<'a> {
    values: Vec<(&'static str, &'a [u16])>,
}

impl<'a> ActualBoundaries<'a> {
    fn new(execution: &'a gpt_oss_model_runner::heterogeneous::LayerOwnerShellExecution) -> Self {
        Self {
            values: vec![
                ("hidden", &execution.hidden_bf16_bits),
                ("input_norm", &execution.input_norm_bf16_bits),
                ("query_after_rope", &execution.query_after_rope_bf16_bits),
                ("key_after_rope", &execution.key_after_rope_bf16_bits),
                ("value_projection", &execution.value_projection_bf16_bits),
                ("attention_context", &execution.attention_context_bf16_bits),
                (
                    "attention_projection",
                    &execution.attention_projection_bf16_bits,
                ),
                (
                    "post_attention_residual",
                    &execution.post_attention_residual_bf16_bits,
                ),
                ("router_input", &execution.router_input_bf16_bits),
            ],
        }
    }
}

fn bits(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .copied()
        .map(bf16::from_f32)
        .map(bf16::to_bits)
        .collect()
}

fn exact(label: &str, expected: &[u16], actual: &[u16]) -> Result<()> {
    if expected.len() != actual.len() {
        bail!(
            "first divergence at {label}: expected length {}, actual {}",
            expected.len(),
            actual.len()
        );
    }
    if let Some((index, (&expected, &actual))) = expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (expected, actual))| expected != actual)
    {
        bail!(
            "first divergence at {label}[{index}]: expected bits=0x{expected:04x} value={} actual bits=0x{actual:04x} value={}",
            bf16::from_bits(expected).to_f32(),
            bf16::from_bits(actual).to_f32()
        );
    }
    Ok(())
}

fn exact_expert_output(
    checkpoint: &GptOssCheckpointView,
    layer: u16,
    expert: u16,
    input_bf16_bits: &[u16],
) -> Result<Vec<u16>> {
    let prefix = format!("model.layers.{layer}.mlp.experts");
    let gate_blocks = checkpoint.tensor(&format!("{prefix}.gate_up_proj_blocks"))?;
    let gate_scales = checkpoint.tensor(&format!("{prefix}.gate_up_proj_scales"))?;
    let gate_bias = checkpoint.tensor(&format!("{prefix}.gate_up_proj_bias"))?;
    let down_blocks = checkpoint.tensor(&format!("{prefix}.down_proj_blocks"))?;
    let down_scales = checkpoint.tensor(&format!("{prefix}.down_proj_scales"))?;
    let down_bias = checkpoint.tensor(&format!("{prefix}.down_proj_bias"))?;
    let expert = usize::from(expert);
    let gate_bias_bf16_bits = bytes_to_u16(expert_slice(
        gate_bias.bytes(),
        expert,
        GATE_UP_BIAS_VALUES * size_of::<u16>(),
    ))?;
    let down_bias_bf16_bits = bytes_to_u16(expert_slice(
        down_bias.bytes(),
        expert,
        DOWN_BIAS_VALUES * size_of::<u16>(),
    ))?;
    let gate_up_blocks = expert_slice(gate_blocks.bytes(), expert, GATE_UP_BLOCK_BYTES);
    let gate_up_scales = expert_slice(gate_scales.bytes(), expert, GATE_UP_SCALE_BYTES);
    let gate_up_bias_bytes = expert_slice(
        gate_bias.bytes(),
        expert,
        GATE_UP_BIAS_VALUES * size_of::<u16>(),
    );
    let down_projection_blocks = expert_slice(down_blocks.bytes(), expert, DOWN_BLOCK_BYTES);
    let down_projection_scales = expert_slice(down_scales.bytes(), expert, DOWN_SCALE_BYTES);
    let down_projection_bias_bytes = expert_slice(
        down_bias.bytes(),
        expert,
        DOWN_BIAS_VALUES * size_of::<u16>(),
    );
    let identity_sha256 = hash_surfaces(&[
        gate_up_blocks,
        gate_up_scales,
        gate_up_bias_bytes,
        down_projection_blocks,
        down_projection_scales,
        down_projection_bias_bytes,
    ]);
    let source = NativeMxfp4ExpertView {
        key: GptOssExpertKey {
            layer,
            expert: expert as u16,
        },
        gate_up_blocks,
        gate_up_scales,
        gate_up_bias_bf16_bits: &gate_bias_bf16_bits,
        down_blocks: down_projection_blocks,
        down_scales: down_projection_scales,
        down_bias_bf16_bits: &down_bias_bf16_bits,
        identity_sha256: &identity_sha256,
    };
    Ok(exact_selected_expert_reference(source, input_bf16_bits)?.down_bf16_bits)
}

fn exact_residual(residual: &[u16], update: &[u16]) -> Result<Vec<u16>> {
    if residual.len() != update.len() {
        bail!(
            "exact residual shape mismatch: residual={} update={}",
            residual.len(),
            update.len()
        );
    }
    Ok(residual
        .iter()
        .zip(update)
        .map(|(&residual, &update)| {
            bf16::from_f32(bf16::from_bits(residual).to_f32() + bf16::from_bits(update).to_f32())
                .to_bits()
        })
        .collect())
}

fn expert_slice<T>(values: &[T], expert: usize, stride: usize) -> &[T] {
    &values[expert * stride..(expert + 1) * stride]
}

fn bytes_to_u16(bytes: &[u8]) -> Result<Vec<u16>> {
    if !bytes.len().is_multiple_of(size_of::<u16>()) {
        bail!("BF16 byte extent is not u16-aligned");
    }
    Ok(bytes
        .chunks_exact(size_of::<u16>())
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect())
}

fn hash_u16(values: &[u16]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn hash_surfaces(surfaces: &[&[u8]]) -> String {
    let mut digest = Sha256::new();
    for surface in surfaces {
        digest.update(surface);
    }
    format!("{:x}", digest.finalize())
}

fn hash_file(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
}

fn tensor_u16(
    checkpoint: &GptOssCheckpointView,
    name: &str,
    expected_values: usize,
) -> Result<Vec<u16>> {
    let tensor = checkpoint.tensor(name)?;
    if tensor.bytes().len() != expected_values * size_of::<u16>() {
        bail!(
            "native tensor {name} has {} BF16 values, expected {expected_values}",
            tensor.bytes().len() / size_of::<u16>()
        );
    }
    Ok(tensor
        .bytes()
        .chunks_exact(size_of::<u16>())
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect())
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    std::fs::write(path, bytes)?;
    Ok(())
}
