#![cfg(feature = "cuda")]

use gpt_oss_cpu_kernels::MXFP4_PACKED_BYTES;
use gpt_oss_gpu::device::{list_devices, StableCudaDeviceId};
use gpt_oss_model_runner::heterogeneous::ExpertOwner;
use gpt_oss_model_runner::heterogeneous::{
    exact_selected_expert_reference, CudaSelectedExpertExecutor, GptOssExpertKey, GptOssPhase,
    GptOssRouteDescriptor, NativeMxfp4ExpertView, PackedRouteDescriptor, SelectedExpertCapture,
    DOWN_BIAS_VALUES, DOWN_BLOCK_BYTES, DOWN_SCALE_BYTES, GATE_UP_BIAS_VALUES, GATE_UP_BLOCK_BYTES,
    GATE_UP_SCALE_BYTES, GPT_OSS_SELECTED_EXPERT_DEVICE_WORK_BYTES,
    GPT_OSS_SELECTED_EXPERT_INPUT_BYTES, GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES,
    GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES, GPT_OSS_SELECTED_EXPERT_SCRATCH_BYTES,
    GPT_OSS_SELECTED_EXPERT_TRACE_BYTES, GPT_OSS_SELECTED_EXPERT_WORKSPACE_POOL_CLASS_BYTES,
    HIDDEN_SIZE, INPUT_BLOCKS,
};
use half::bf16;

#[cfg(feature = "heterogeneous-test-faults")]
use gpt_oss_model_runner::heterogeneous::SelectedExpertInjectedFault;

const IDENTITY: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[test]
fn selected_native_expert_is_exact_on_every_cuda_device() {
    assert_eq!(GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES, 13_236_480);
    assert_eq!(GPT_OSS_SELECTED_EXPERT_INPUT_BYTES, 5_760);
    assert_eq!(GPT_OSS_SELECTED_EXPERT_SCRATCH_BYTES, 17_280);
    assert_eq!(GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES, 5_760);
    assert_eq!(GPT_OSS_SELECTED_EXPERT_TRACE_BYTES, 23_040);
    assert_eq!(GPT_OSS_SELECTED_EXPERT_DEVICE_WORK_BYTES, 51_840);
    assert_eq!(GPT_OSS_SELECTED_EXPERT_WORKSPACE_POOL_CLASS_BYTES, 65_536);
    const {
        assert!(
            GPT_OSS_SELECTED_EXPERT_DEVICE_WORK_BYTES
                <= GPT_OSS_SELECTED_EXPERT_WORKSPACE_POOL_CLASS_BYTES
        );
    }

    let devices = list_devices();
    assert_eq!(devices.len(), 2, "H2 gate requires the two local RTX 3090s");

    let mut input = (0..HIDDEN_SIZE)
        .map(|index| bf16::from_f32(((index % 31) as f32 - 15.0) / 16.0).to_bits())
        .collect::<Vec<_>>();
    input[0] = bf16::from_f32(-0.0).to_bits();
    input[1] = bf16::MAX.to_bits();
    input[2] = bf16::MIN_POSITIVE.to_bits();
    let mut gate_blocks = vec![0_u8; GATE_UP_BLOCK_BYTES];
    let mut gate_scales = vec![127_u8; GATE_UP_SCALE_BYTES];
    let mut gate_bias = (0..GATE_UP_BIAS_VALUES)
        .map(|index| bf16::from_f32(((index % 9) as f32 - 4.0) / 64.0).to_bits())
        .collect::<Vec<_>>();
    let mut down_blocks = vec![0_u8; DOWN_BLOCK_BYTES];
    let mut down_scales = vec![127_u8; DOWN_SCALE_BYTES];
    let down_bias = (0..DOWN_BIAS_VALUES)
        .map(|index| bf16::from_f32(((index % 11) as f32 - 5.0) / 32.0).to_bits())
        .collect::<Vec<_>>();

    // Six rows cover all 16 E2M1 codes and E8M0 finite/special edges without
    // changing the production fixed shape. Remaining rows validate bias-only
    // behavior and make accidental all-expert indexing visible.
    for row in 0..6 {
        let start = row * INPUT_BLOCKS * MXFP4_PACKED_BYTES;
        for pair in 0..MXFP4_PACKED_BYTES {
            gate_blocks[start + pair] = (pair as u8) | (((15 - pair) as u8) << 4);
        }
        gate_scales[row * INPUT_BLOCKS] = [0, 1, 126, 127, 254, 255][row];
        gate_bias[row] = bf16::from_f32((row as f32 - 2.0) / 8.0).to_bits();
    }
    for row in 0..6 {
        let start = row * INPUT_BLOCKS * MXFP4_PACKED_BYTES;
        for pair in 0..MXFP4_PACKED_BYTES {
            down_blocks[start + pair] =
                ((pair + row) as u8 & 0x0f) | ((((15 - pair + row) as u8) & 0x0f) << 4);
        }
        down_scales[row * INPUT_BLOCKS] = [0, 1, 126, 127, 254, 255][row];
    }
    // Bias-only rows force signed zero, both clamp boundaries, adjacent BF16
    // values, and finite extrema through the exact GPT SwiGLU sequence.
    for (pair, (gate, up)) in [
        (-0.0, -0.0),
        (6.96875, -7.03125),
        (7.0, -7.0),
        (7.03125, -6.96875),
        (bf16::MAX.to_f32(), bf16::MAX.to_f32()),
    ]
    .into_iter()
    .enumerate()
    {
        let index = 16 + pair * 2;
        gate_bias[index] = bf16::from_f32(gate).to_bits();
        gate_bias[index + 1] = bf16::from_f32(up).to_bits();
    }

    let reference_source = NativeMxfp4ExpertView {
        key: GptOssExpertKey {
            layer: 0,
            expert: 0,
        },
        gate_up_blocks: &gate_blocks,
        gate_up_scales: &gate_scales,
        gate_up_bias_bf16_bits: &gate_bias,
        down_blocks: &down_blocks,
        down_scales: &down_scales,
        down_bias_bf16_bits: &down_bias,
        identity_sha256: IDENTITY,
    };
    let expected = exact_selected_expert_reference(reference_source, &input).unwrap();

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
        let key = GptOssExpertKey {
            layer: 0,
            expert: position as u16,
        };
        let source = NativeMxfp4ExpertView {
            key,
            gate_up_blocks: &gate_blocks,
            gate_up_scales: &gate_scales,
            gate_up_bias_bf16_bits: &gate_bias,
            down_blocks: &down_blocks,
            down_scales: &down_scales,
            down_bias_bf16_bits: &down_bias,
            identity_sha256: IDENTITY,
        };
        let mut executor = CudaSelectedExpertExecutor::new(stable).unwrap();
        let mut result_slot = executor.allocate_result_slot().unwrap();
        assert!(executor
            .upload_expert(
                ExpertOwner::Cpu {
                    pool: gpt_oss_model_runner::heterogeneous::CpuPoolId(0),
                },
                source,
            )
            .is_err());
        let weights = executor.upload_expert(owner.clone(), source).unwrap();
        let route = PackedRouteDescriptor {
            route: GptOssRouteDescriptor::new(0, 0, key.expert, 0.5, 0),
            owner,
            placement_epoch: 7,
            canonical_result_slot: 0,
            source_activation_slot: 0,
        };

        let foreign_stable = StableCudaDeviceId::from_device(&devices[1 - position]).unwrap();
        let foreign_executor = CudaSelectedExpertExecutor::new(foreign_stable).unwrap();
        let mut foreign_result_slot = foreign_executor.allocate_result_slot().unwrap();
        assert!(executor
            .prepare(
                GptOssPhase::Decode,
                &route,
                &weights,
                &input,
                &mut foreign_result_slot,
            )
            .is_err());
        drop(foreign_result_slot);
        drop(foreign_executor);

        assert!(executor
            .prepare(
                GptOssPhase::Prefill,
                &route,
                &weights,
                &input,
                &mut result_slot,
            )
            .is_err());
        let mut non_finite = input.clone();
        non_finite[17] = bf16::NAN.to_bits();
        assert!(executor
            .prepare(
                GptOssPhase::Decode,
                &route,
                &weights,
                &non_finite,
                &mut result_slot,
            )
            .is_err());
        let mut wrong_route = route.clone();
        wrong_route.route.expert_id += 1;
        assert!(executor
            .prepare(
                GptOssPhase::Decode,
                &wrong_route,
                &weights,
                &input,
                &mut result_slot,
            )
            .is_err());

        let canceled = executor
            .prepare(
                GptOssPhase::Decode,
                &route,
                &weights,
                &input,
                &mut result_slot,
            )
            .unwrap()
            .submit()
            .unwrap()
            .cancel()
            .unwrap();
        assert_eq!(canceled.expert_id, key.expert);

        #[cfg(feature = "heterogeneous-test-faults")]
        {
            executor
                .inject_next_failure(SelectedExpertInjectedFault::SubmitBeforeEnqueue)
                .unwrap();
            assert!(executor
                .prepare(
                    GptOssPhase::Decode,
                    &route,
                    &weights,
                    &input,
                    &mut result_slot,
                )
                .unwrap()
                .submit()
                .is_err());

            executor
                .inject_next_failure(SelectedExpertInjectedFault::SubmitAfterInputEnqueue)
                .unwrap();
            assert!(executor
                .prepare(
                    GptOssPhase::Decode,
                    &route,
                    &weights,
                    &input,
                    &mut result_slot,
                )
                .unwrap()
                .submit()
                .is_err());
            assert!(executor.last_post_enqueue_fault_drained());

            executor
                .inject_next_failure(SelectedExpertInjectedFault::Drain)
                .unwrap();
            assert!(executor
                .prepare(
                    GptOssPhase::Decode,
                    &route,
                    &weights,
                    &input,
                    &mut result_slot,
                )
                .unwrap()
                .submit()
                .unwrap()
                .drain(SelectedExpertCapture::OutputOnly)
                .is_err());
        }

        let mut first_output = None;
        for _ in 0..2 {
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
            let trace = actual.trace.unwrap();
            assert_bf16_boundary(
                "gate_up",
                &expected.gate_up_bf16_bits,
                &trace.gate_up_bf16_bits,
            );
            assert_bf16_boundary(
                "swiglu",
                &expected.swiglu_bf16_bits,
                &trace.swiglu_bf16_bits,
            );
            assert_bf16_boundary(
                "scaled_gate",
                &expected.scaled_gate_bf16_bits,
                &trace.scaled_gate_bf16_bits,
            );
            assert_bf16_boundary(
                "sigmoid",
                &expected.sigmoid_bf16_bits,
                &trace.sigmoid_bf16_bits,
            );
            assert_bf16_boundary("glu", &expected.glu_bf16_bits, &trace.glu_bf16_bits);
            assert_bf16_boundary(
                "linear",
                &expected.linear_bf16_bits,
                &trace.linear_bf16_bits,
            );
            assert_bf16_boundary("down", &expected.down_bf16_bits, &trace.down_bf16_bits);
            assert_bf16_boundary("output", &expected.down_bf16_bits, &actual.output_bf16_bits);
            if let Some(first) = &first_output {
                assert_eq!(first, &actual.output_bf16_bits);
            } else {
                first_output = Some(actual.output_bf16_bits.clone());
            }
            assert_eq!(actual.result.expert_id, key.expert);
            assert_eq!(actual.result.route_rank, 0);
            assert!(actual.kernel_elapsed_ms.is_finite());
        }
    }
}

fn assert_bf16_boundary(label: &str, expected: &[u16], actual: &[u16]) {
    assert_eq!(expected.len(), actual.len());
    let mut mismatch_count = 0_usize;
    let mut first = Vec::new();
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        if expected != actual {
            mismatch_count += 1;
            if first.len() < 8 {
                first.push(format!(
                    "{index}: expected=0x{expected:04x} actual=0x{actual:04x}"
                ));
            }
        }
    }
    assert_eq!(
        mismatch_count,
        0,
        "{label} has {mismatch_count} BF16 bit mismatches; first: {}",
        first.join(", ")
    );
}
