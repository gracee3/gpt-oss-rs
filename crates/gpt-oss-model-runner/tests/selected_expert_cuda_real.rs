#![cfg(feature = "cuda")]

use std::path::PathBuf;

use gpt_oss_cpu_kernels::{KernelPath, Kernels};
use gpt_oss_gpu::device::{list_devices, StableCudaDeviceId};
use gpt_oss_model_runner::heterogeneous::{
    CudaSelectedExpertExecutor, ExpertOwner, GptOssExpertKey, GptOssPhase, GptOssRouteDescriptor,
    NativeMxfp4ExpertView, PackedRouteDescriptor, SelectedExpertCapture, DOWN_BIAS_VALUES,
    DOWN_BLOCK_BYTES, DOWN_SCALE_BYTES, GATE_UP_BIAS_VALUES, GATE_UP_BLOCK_BYTES,
    GATE_UP_SCALE_BYTES, HIDDEN_SIZE,
};
use gpt_oss_model_runner::{CpuGptOssConfig, CpuTensorStore};
use half::bf16;
use serde::Deserialize;
use sha2::{Digest, Sha256};

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

#[test]
fn real_20b_selected_expert_matches_retained_cpu_trace_on_both_gpus() {
    if std::env::var_os("GPT_OSS_RUN_SELECTED_EXPERT_REAL").is_none() {
        eprintln!("GPT_OSS_RUN_SELECTED_EXPERT_REAL is unset; skipping real-weight H2 gate");
        return;
    }
    let model = PathBuf::from(
        std::env::var_os("GPT_OSS_TEST_MODEL")
            .expect("GPT_OSS_TEST_MODEL must name the 20B snapshot"),
    );
    let trace_path = PathBuf::from(
        std::env::var_os("GPT_OSS_SELECTED_EXPERT_TRACE")
            .expect("GPT_OSS_SELECTED_EXPERT_TRACE must name the retained CPU trace"),
    );
    let control: ControlDocument =
        serde_json::from_slice(&std::fs::read(trace_path).unwrap()).unwrap();
    let layer = &control.trace.layers[0];
    assert_eq!(layer.layer_index, 0);
    assert_eq!(
        layer
            .experts
            .iter()
            .map(|expert| expert.expert_index)
            .collect::<Vec<_>>(),
        [31, 21, 22, 6]
    );

    let config = CpuGptOssConfig::from_snapshot(&model).unwrap();
    let store = CpuTensorStore::open(&model).unwrap();
    let prefix = format!("model.layers.{}.mlp.experts", layer.layer_index);
    let gate_blocks_tensor = store
        .tensor(&format!("{prefix}.gate_up_proj_blocks"))
        .unwrap();
    let gate_scales_tensor = store
        .tensor(&format!("{prefix}.gate_up_proj_scales"))
        .unwrap();
    let gate_bias_tensor = store
        .tensor(&format!("{prefix}.gate_up_proj_bias"))
        .unwrap();
    let down_blocks_tensor = store.tensor(&format!("{prefix}.down_proj_blocks")).unwrap();
    let down_scales_tensor = store.tensor(&format!("{prefix}.down_proj_scales")).unwrap();
    let down_bias_tensor = store.tensor(&format!("{prefix}.down_proj_bias")).unwrap();
    assert_eq!(gate_blocks_tensor.shape(), &[32, 5760, 90, 16]);
    assert_eq!(gate_scales_tensor.shape(), &[32, 5760, 90]);
    assert_eq!(gate_bias_tensor.shape(), &[32, 5760]);
    assert_eq!(down_blocks_tensor.shape(), &[32, 2880, 90, 16]);
    assert_eq!(down_scales_tensor.shape(), &[32, 2880, 90]);
    assert_eq!(down_bias_tensor.shape(), &[32, 2880]);

    let post_norm = store
        .tensor(&format!(
            "model.layers.{}.post_attention_layernorm.weight",
            layer.layer_index
        ))
        .unwrap();
    let post_norm = post_norm
        .bf16()
        .unwrap()
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    assert_eq!(layer.post_attention_residual.len(), HIDDEN_SIZE);
    let residual = layer
        .post_attention_residual
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect::<Vec<_>>();
    let mut normalized = vec![0.0_f32; HIDDEN_SIZE];
    // The retained control used the auto/AVX-512 RMS path. Reconstruct the
    // exact immutable activation it supplied to MoE, then run the deliberately
    // exact BF16 expert authority rather than comparing with that control's
    // residual-Q8 expert projection.
    Kernels::new(KernelPath::Auto)
        .unwrap()
        .rms_norm(&residual, &post_norm, config.rms_norm_eps, &mut normalized)
        .unwrap();
    let input = normalized
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();

    let devices = list_devices();
    assert_eq!(devices.len(), 2, "real H2 gate requires both local GPUs");
    for expert in &layer.experts {
        let expert_index = expert.expert_index;
        let gate_blocks = expert_slice(
            gate_blocks_tensor.u8().unwrap(),
            expert_index,
            GATE_UP_BLOCK_BYTES,
        );
        let gate_scales = expert_slice(
            gate_scales_tensor.u8().unwrap(),
            expert_index,
            GATE_UP_SCALE_BYTES,
        );
        let gate_bias = expert_slice(
            gate_bias_tensor.bf16().unwrap(),
            expert_index,
            GATE_UP_BIAS_VALUES,
        )
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
        let down_blocks = expert_slice(
            down_blocks_tensor.u8().unwrap(),
            expert_index,
            DOWN_BLOCK_BYTES,
        );
        let down_scales = expert_slice(
            down_scales_tensor.u8().unwrap(),
            expert_index,
            DOWN_SCALE_BYTES,
        );
        let down_bias = expert_slice(
            down_bias_tensor.bf16().unwrap(),
            expert_index,
            DOWN_BIAS_VALUES,
        )
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();

        let mut identity = Sha256::new();
        identity.update(b"gpt-oss-rs-selected-expert-v1");
        identity.update(gate_blocks);
        identity.update(gate_scales);
        identity.update(bytemuck::cast_slice(&gate_bias));
        identity.update(down_blocks);
        identity.update(down_scales);
        identity.update(bytemuck::cast_slice(&down_bias));
        let identity = format!("{:x}", identity.finalize());
        let key = GptOssExpertKey {
            layer: layer.layer_index as u16,
            expert: expert_index as u16,
        };
        let reference_source = NativeMxfp4ExpertView {
            key,
            gate_up_blocks: gate_blocks,
            gate_up_scales: gate_scales,
            gate_up_bias_bf16_bits: &gate_bias,
            down_blocks,
            down_scales,
            down_bias_bf16_bits: &down_bias,
            identity_sha256: &identity,
        };
        let expected = gpt_oss_model_runner::heterogeneous::exact_selected_expert_reference(
            reference_source,
            &input,
        )
        .unwrap();
        for (position, device) in devices.iter().enumerate() {
            let stable = StableCudaDeviceId::from_device(device).unwrap();
            let owner = if position == 0 {
                ExpertOwner::LayerOwnerGpu {
                    device: stable.clone(),
                }
            } else {
                ExpertOwner::RemoteGpu {
                    device: stable.clone(),
                }
            };
            let mut executor = CudaSelectedExpertExecutor::new(stable).unwrap();
            let mut result_slot = executor.allocate_result_slot().unwrap();
            let weights = executor
                .upload_expert(owner.clone(), reference_source)
                .unwrap();
            let route = PackedRouteDescriptor {
                route: GptOssRouteDescriptor::new(
                    0,
                    expert.rank as u8,
                    expert_index as u16,
                    0.5,
                    0,
                ),
                owner,
                placement_epoch: 1,
                canonical_result_slot: expert.rank as u32,
                source_activation_slot: 0,
            };
            let actual = executor
                .execute(
                    GptOssPhase::Decode,
                    &route,
                    &weights,
                    &input,
                    &mut result_slot,
                    SelectedExpertCapture::FirstDivergence,
                )
                .unwrap();
            let actual_trace = actual.trace.unwrap();
            assert_exact(
                "gate_up",
                &expected.gate_up_bf16_bits,
                &actual_trace.gate_up_bf16_bits,
            );
            assert_exact(
                "swiglu",
                &expected.swiglu_bf16_bits,
                &actual_trace.swiglu_bf16_bits,
            );
            assert_exact(
                "scaled_gate",
                &expected.scaled_gate_bf16_bits,
                &actual_trace.scaled_gate_bf16_bits,
            );
            assert_exact(
                "sigmoid",
                &expected.sigmoid_bf16_bits,
                &actual_trace.sigmoid_bf16_bits,
            );
            assert_exact("glu", &expected.glu_bf16_bits, &actual_trace.glu_bf16_bits);
            assert_exact(
                "linear",
                &expected.linear_bf16_bits,
                &actual_trace.linear_bf16_bits,
            );
            assert_exact(
                "down",
                &expected.down_bf16_bits,
                &actual_trace.down_bf16_bits,
            );
            assert_exact("output", &expected.down_bf16_bits, &actual.output_bf16_bits);
            eprintln!(
                "device={} rank={} expert={} kernel_elapsed_ms={:.6}",
                device.id, expert.rank, expert_index, actual.kernel_elapsed_ms
            );
        }
    }
}

fn expert_slice<T>(values: &[T], expert: usize, stride: usize) -> &[T] {
    &values[expert * stride..(expert + 1) * stride]
}

fn assert_exact(label: &str, expected: &[u16], actual: &[u16]) {
    assert_eq!(expected.len(), actual.len());
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            expected,
            actual,
            "first divergence at {label}[{index}]: expected={} actual={}",
            bf16::from_bits(expected).to_f32(),
            bf16::from_bits(actual).to_f32()
        );
    }
}
