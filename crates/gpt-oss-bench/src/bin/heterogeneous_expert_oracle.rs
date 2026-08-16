use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use gpt_oss_cpu_kernels::{KernelPath, Kernels};
use gpt_oss_gpu::device::{list_devices, StableCudaDeviceId};
use gpt_oss_gpu::kernel_loader::compiled_ptx_dir;
use gpt_oss_model_runner::heterogeneous::{
    exact_selected_expert_reference, selected_expert_device_memory_info,
    CudaSelectedExpertExecutor, ExpertOwner, GptOssExpertKey, GptOssPhase, GptOssRouteDescriptor,
    NativeMxfp4ExpertView, PackedRouteDescriptor, SelectedExpertCapture, DOWN_BIAS_VALUES,
    DOWN_BLOCK_BYTES, DOWN_SCALE_BYTES, GATE_UP_BIAS_VALUES, GATE_UP_BLOCK_BYTES,
    GATE_UP_SCALE_BYTES, GPT_OSS_SELECTED_EXPERT_DEVICE_WORK_BYTES,
    GPT_OSS_SELECTED_EXPERT_INPUT_BYTES, GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES,
    GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES, GPT_OSS_SELECTED_EXPERT_SCRATCH_BYTES,
    GPT_OSS_SELECTED_EXPERT_TRACE_BYTES, GPT_OSS_SELECTED_EXPERT_WORKSPACE_POOL_CLASS_BYTES,
    HIDDEN_SIZE,
};
use gpt_oss_model_runner::{CpuGptOssConfig, CpuTensorStore};
use half::bf16;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    retained_trace: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Deserialize)]
struct ControlDocument {
    trace: ControlTrace,
}

#[derive(Deserialize)]
struct ControlTrace {
    layers: Vec<ControlLayer>,
}

#[derive(Deserialize)]
struct ControlLayer {
    layer_index: usize,
    post_attention_residual: Vec<f32>,
    experts: Vec<ControlExpert>,
}

#[derive(Deserialize)]
struct ControlExpert {
    rank: usize,
    expert_index: usize,
}

#[derive(Serialize)]
struct OracleRecord {
    schema: &'static str,
    model_config_sha256: String,
    model_index_sha256: String,
    retained_trace_sha256: String,
    executable_sha256: String,
    ptx_sha256: String,
    payload_bytes: usize,
    logical_input_bytes: usize,
    logical_scratch_bytes: usize,
    logical_output_bytes: usize,
    logical_trace_bytes: usize,
    logical_device_work_bytes: usize,
    workspace_pool_class_bytes: usize,
    observed_allocator_reservation_high_water_bytes: usize,
    routes: Vec<RouteRecord>,
    exact: bool,
}

#[derive(Serialize)]
struct RouteRecord {
    device_pci_bus_id: String,
    runtime_ordinal: usize,
    route_rank: usize,
    expert_index: usize,
    expert_identity_sha256: String,
    gate_up_sha256: String,
    scaled_gate_sha256: String,
    sigmoid_sha256: String,
    glu_sha256: String,
    linear_sha256: String,
    swiglu_sha256: String,
    down_sha256: String,
    repeated_output_sha256: [String; 2],
    kernel_elapsed_ms: [f32; 2],
    free_bytes_before_executor: usize,
    free_bytes_after_executor: usize,
    free_bytes_after_result_slot: usize,
    free_bytes_after_upload: usize,
    free_bytes_after_first_execution: usize,
    free_bytes_after_repeats: usize,
    free_bytes_after_teardown: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let control: ControlDocument = serde_json::from_slice(&std::fs::read(&cli.retained_trace)?)?;
    let layer = control
        .trace
        .layers
        .first()
        .context("retained trace has no layers")?;
    if layer.layer_index != 0
        || layer
            .experts
            .iter()
            .map(|expert| expert.expert_index)
            .collect::<Vec<_>>()
            != [31, 21, 22, 6]
    {
        bail!("retained trace is not the pinned layer-0 route");
    }

    let config = CpuGptOssConfig::from_snapshot(&cli.model)?;
    let store = CpuTensorStore::open(&cli.model)?;
    let prefix = "model.layers.0.mlp.experts";
    let gate_blocks_tensor = store.tensor(&format!("{prefix}.gate_up_proj_blocks"))?;
    let gate_scales_tensor = store.tensor(&format!("{prefix}.gate_up_proj_scales"))?;
    let gate_bias_tensor = store.tensor(&format!("{prefix}.gate_up_proj_bias"))?;
    let down_blocks_tensor = store.tensor(&format!("{prefix}.down_proj_blocks"))?;
    let down_scales_tensor = store.tensor(&format!("{prefix}.down_proj_scales"))?;
    let down_bias_tensor = store.tensor(&format!("{prefix}.down_proj_bias"))?;

    let post_norm = store
        .tensor("model.layers.0.post_attention_layernorm.weight")?
        .bf16()?
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    if layer.post_attention_residual.len() != HIDDEN_SIZE {
        bail!("retained layer-0 residual has the wrong length");
    }
    let residual = layer
        .post_attention_residual
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect::<Vec<_>>();
    let mut normalized = vec![0.0_f32; HIDDEN_SIZE];
    Kernels::new(KernelPath::Auto)?.rms_norm(
        &residual,
        &post_norm,
        config.rms_norm_eps,
        &mut normalized,
    )?;
    let input = normalized
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();

    let devices = list_devices();
    if devices.len() != 2 {
        bail!("H2 oracle requires exactly two CUDA devices");
    }
    let mut routes = Vec::with_capacity(8);
    for expert in &layer.experts {
        let index = expert.expert_index;
        let gate_blocks = expert_slice(gate_blocks_tensor.u8()?, index, GATE_UP_BLOCK_BYTES);
        let gate_scales = expert_slice(gate_scales_tensor.u8()?, index, GATE_UP_SCALE_BYTES);
        let gate_bias = expert_slice(gate_bias_tensor.bf16()?, index, GATE_UP_BIAS_VALUES)
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let down_blocks = expert_slice(down_blocks_tensor.u8()?, index, DOWN_BLOCK_BYTES);
        let down_scales = expert_slice(down_scales_tensor.u8()?, index, DOWN_SCALE_BYTES);
        let down_bias = expert_slice(down_bias_tensor.bf16()?, index, DOWN_BIAS_VALUES)
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let expert_identity = hash_expert(
            gate_blocks,
            gate_scales,
            &gate_bias,
            down_blocks,
            down_scales,
            &down_bias,
        );
        let key = GptOssExpertKey {
            layer: 0,
            expert: index as u16,
        };
        let source = NativeMxfp4ExpertView {
            key,
            gate_up_blocks: gate_blocks,
            gate_up_scales: gate_scales,
            gate_up_bias_bf16_bits: &gate_bias,
            down_blocks,
            down_scales,
            down_bias_bf16_bits: &down_bias,
            identity_sha256: &expert_identity,
        };
        let expected = exact_selected_expert_reference(source, &input)?;
        for (position, device) in devices.iter().enumerate() {
            let stable = StableCudaDeviceId::from_device(device)?;
            let owner = if position == 0 {
                ExpertOwner::LayerOwnerGpu {
                    device: stable.clone(),
                }
            } else {
                ExpertOwner::RemoteGpu {
                    device: stable.clone(),
                }
            };
            let (free_before_executor, _) = selected_expert_device_memory_info(&stable)?;
            let mut executor = CudaSelectedExpertExecutor::new(stable.clone())?;
            let (free_after_executor, _) = executor.memory_info()?;
            let mut result_slot = executor.allocate_result_slot()?;
            let (free_after_result_slot, _) = executor.memory_info()?;
            let weights = executor.upload_expert(owner.clone(), source)?;
            let (free_after_upload, _) = executor.memory_info()?;
            let route = PackedRouteDescriptor {
                route: GptOssRouteDescriptor::new(0, expert.rank as u8, index as u16, 0.5, 0),
                owner,
                placement_epoch: 1,
                canonical_result_slot: expert.rank as u32,
                source_activation_slot: 0,
            };
            let mut output_hashes = Vec::with_capacity(2);
            let mut elapsed = Vec::with_capacity(2);
            let mut free_after_execution = Vec::with_capacity(2);
            for _ in 0..2 {
                let actual = executor.execute(
                    GptOssPhase::Decode,
                    &route,
                    &weights,
                    &input,
                    &mut result_slot,
                    SelectedExpertCapture::FirstDivergence,
                )?;
                let trace = actual.trace.context("first-divergence trace missing")?;
                exact(
                    "gate/up",
                    &expected.gate_up_bf16_bits,
                    &trace.gate_up_bf16_bits,
                )?;
                exact(
                    "scaled gate",
                    &expected.scaled_gate_bf16_bits,
                    &trace.scaled_gate_bf16_bits,
                )?;
                exact(
                    "sigmoid",
                    &expected.sigmoid_bf16_bits,
                    &trace.sigmoid_bf16_bits,
                )?;
                exact("GLU", &expected.glu_bf16_bits, &trace.glu_bf16_bits)?;
                exact(
                    "linear",
                    &expected.linear_bf16_bits,
                    &trace.linear_bf16_bits,
                )?;
                exact(
                    "SwiGLU",
                    &expected.swiglu_bf16_bits,
                    &trace.swiglu_bf16_bits,
                )?;
                exact("down", &expected.down_bf16_bits, &actual.output_bf16_bits)?;
                output_hashes.push(hash_u16(&actual.output_bf16_bits));
                elapsed.push(actual.kernel_elapsed_ms);
                free_after_execution.push(executor.memory_info()?.0);
            }
            let (free_after_repeats, _) = executor.memory_info()?;
            if free_after_repeats != free_after_upload || output_hashes[0] != output_hashes[1] {
                bail!("repeat execution changed device allocation or output identity");
            }
            let mut route_record = RouteRecord {
                device_pci_bus_id: stable.pci_bus_id.to_string(),
                runtime_ordinal: device.id,
                route_rank: expert.rank,
                expert_index: index,
                expert_identity_sha256: expert_identity.clone(),
                gate_up_sha256: hash_u16(&expected.gate_up_bf16_bits),
                scaled_gate_sha256: hash_u16(&expected.scaled_gate_bf16_bits),
                sigmoid_sha256: hash_u16(&expected.sigmoid_bf16_bits),
                glu_sha256: hash_u16(&expected.glu_bf16_bits),
                linear_sha256: hash_u16(&expected.linear_bf16_bits),
                swiglu_sha256: hash_u16(&expected.swiglu_bf16_bits),
                down_sha256: hash_u16(&expected.down_bf16_bits),
                repeated_output_sha256: output_hashes.try_into().expect("two repeats"),
                kernel_elapsed_ms: elapsed.try_into().expect("two repeats"),
                free_bytes_before_executor: free_before_executor,
                free_bytes_after_executor: free_after_executor,
                free_bytes_after_result_slot: free_after_result_slot,
                free_bytes_after_upload: free_after_upload,
                free_bytes_after_first_execution: free_after_execution[0],
                free_bytes_after_repeats: free_after_repeats,
                free_bytes_after_teardown: 0,
            };
            drop(weights);
            drop(result_slot);
            drop(executor);
            let (free_after_teardown, _) = selected_expert_device_memory_info(&stable)?;
            if free_after_teardown != free_before_executor {
                bail!(
                    "selected-expert teardown retained {} device bytes",
                    free_before_executor.saturating_sub(free_after_teardown)
                );
            }
            route_record.free_bytes_after_teardown = free_after_teardown;
            routes.push(route_record);
        }
    }

    let observed_allocator_reservation_high_water_bytes = routes
        .iter()
        .map(|route| {
            route.free_bytes_before_executor.saturating_sub(
                route
                    .free_bytes_after_executor
                    .min(route.free_bytes_after_result_slot)
                    .min(route.free_bytes_after_upload)
                    .min(route.free_bytes_after_first_execution)
                    .min(route.free_bytes_after_repeats),
            )
        })
        .max()
        .unwrap_or(0);

    let record = OracleRecord {
        schema: "gpt-oss-rs.heterogeneous-expert-oracle/v1",
        model_config_sha256: hash_file(&cli.model.join("config.json"))?,
        model_index_sha256: hash_file(&cli.model.join("model.safetensors.index.json"))?,
        retained_trace_sha256: hash_file(&cli.retained_trace)?,
        executable_sha256: hash_file(&std::env::current_exe()?)?,
        ptx_sha256: hash_file(&compiled_ptx_dir().join("gpt_oss_selected_expert.ptx"))?,
        payload_bytes: GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES,
        logical_input_bytes: GPT_OSS_SELECTED_EXPERT_INPUT_BYTES,
        logical_scratch_bytes: GPT_OSS_SELECTED_EXPERT_SCRATCH_BYTES,
        logical_output_bytes: GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES,
        logical_trace_bytes: GPT_OSS_SELECTED_EXPERT_TRACE_BYTES,
        logical_device_work_bytes: GPT_OSS_SELECTED_EXPERT_DEVICE_WORK_BYTES,
        workspace_pool_class_bytes: GPT_OSS_SELECTED_EXPERT_WORKSPACE_POOL_CLASS_BYTES,
        observed_allocator_reservation_high_water_bytes,
        routes,
        exact: true,
    };
    let mut encoded = serde_json::to_vec_pretty(&record)?;
    encoded.push(b'\n');
    std::fs::write(&cli.output, encoded)?;
    Ok(())
}

fn expert_slice<T>(values: &[T], expert: usize, stride: usize) -> &[T] {
    &values[expert * stride..(expert + 1) * stride]
}

fn exact(label: &str, expected: &[u16], actual: &[u16]) -> Result<()> {
    if expected.len() != actual.len() {
        bail!(
            "length mismatch at {label}: expected={} actual={}",
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
            "first divergence at {label}[{index}]: expected={} actual={}",
            bf16::from_bits(expected).to_f32(),
            bf16::from_bits(actual).to_f32()
        );
    }
    Ok(())
}

fn hash_many(parts: &[&[u8]]) -> String {
    let mut digest = Sha256::new();
    for part in parts {
        digest.update(part);
    }
    format!("{:x}", digest.finalize())
}

fn hash_bytes(bytes: &[u8]) -> String {
    hash_many(&[bytes])
}

fn hash_u16(values: &[u16]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn hash_expert(
    gate_blocks: &[u8],
    gate_scales: &[u8],
    gate_bias: &[u16],
    down_blocks: &[u8],
    down_scales: &[u8],
    down_bias: &[u16],
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"gpt-oss-rs-selected-expert-v1");
    digest.update(gate_blocks);
    digest.update(gate_scales);
    for value in gate_bias {
        digest.update(value.to_le_bytes());
    }
    digest.update(down_blocks);
    digest.update(down_scales);
    for value in down_bias {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn hash_file(path: &Path) -> Result<String> {
    Ok(hash_bytes(&std::fs::read(path).with_context(|| {
        format!("failed to read hash input {}", path.display())
    })?))
}
