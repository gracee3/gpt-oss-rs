#![cfg(feature = "cuda")]

use cudarc::driver::CudaContext;
use gpt_oss_gpu::device::{list_devices, StableCudaDeviceId};
use gpt_oss_gpu::event::CorrelatedTimeline;
use gpt_oss_gpu::kernel_loader::compiled_ptx_dir;
use gpt_oss_model_runner::heterogeneous::{
    exact_router_reference, CudaExactRouter, ExactRouterWeightsView, GptOssPhase, RelayPinnedPools,
    ResidentExactRouterWeights, GPT_OSS_ROUTER_MAX_ROWS, HIDDEN_SIZE,
};
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};

use gpt_oss_model_runner::model_loader::gpt_oss_native::GptOssCheckpointView;
use gpt_oss_model_runner::model_loader::owner_selective::ResidentExactRouterSources;

#[cfg(feature = "heterogeneous-test-faults")]
use gpt_oss_model_runner::heterogeneous::{
    resident_router_handoff_quarantines_for_test, ResidentRouterHandoffInjectedFault,
};
#[cfg(feature = "heterogeneous-test-faults")]
use gpt_oss_model_runner::heterogeneous::{ExactRouterInjectedFault, ResultRelayInjectedFault};

fn fixture(experts: usize, rows: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let activation = (0..rows * HIDDEN_SIZE)
        .map(|index| bf16::from_f32((index as f32 % 31.0 - 15.0) / 16.0).to_bits())
        .collect::<Vec<_>>();
    let weights = (0..experts * HIDDEN_SIZE)
        .map(|index| bf16::from_f32((index as f32 % 17.0 - 8.0) / 64.0).to_bits())
        .collect::<Vec<_>>();
    let bias = (0..experts)
        .map(|expert| bf16::from_f32((expert % 7) as f32 / 16.0).to_bits())
        .collect::<Vec<_>>();
    (activation, weights, bias)
}

fn layer_owner() -> StableCudaDeviceId {
    let devices = list_devices();
    assert_eq!(devices.len(), 2, "H4 gate requires both local RTX 3090s");
    StableCudaDeviceId::from_device(&devices[0]).unwrap()
}

fn resident_source(
    stable_device: StableCudaDeviceId,
    experts: usize,
    weights: &[u16],
    bias: &[u16],
) -> ResidentExactRouterWeights {
    let ordinal = list_devices()
        .into_iter()
        .find(|device| device.pci_bus_id == Some(stable_device.pci_bus_id))
        .unwrap()
        .id;
    let context = CudaContext::new(ordinal).unwrap();
    let stream = context.new_stream().unwrap();
    let weight_bytes: &[u8] = bytemuck::cast_slice(weights);
    let bias_bytes: &[u8] = bytemuck::cast_slice(bias);
    let device_weights = stream.clone_htod(weight_bytes).unwrap();
    let device_bias = stream.clone_htod(bias_bytes).unwrap();
    stream.synchronize().unwrap();
    ResidentExactRouterWeights::new(stable_device, experts, device_weights, device_bias).unwrap()
}

#[test]
fn resident_router_handoff_matches_host_constructor_on_both_gpus() {
    let devices = list_devices();
    assert_eq!(devices.len(), 2, "resident handoff gate requires two GPUs");
    for (device_index, experts) in [32_usize, 128].into_iter().enumerate() {
        let stable = StableCudaDeviceId::from_device(&devices[device_index]).unwrap();
        let (activation, weights, bias) = fixture(experts, 1);
        let view = ExactRouterWeightsView {
            experts,
            weight_bf16_bits: &weights,
            bias_bf16_bits: &bias,
        };
        let mut host_router = CudaExactRouter::new(stable.clone(), 1, view).unwrap();
        let mut resident_router = CudaExactRouter::from_resident_weights(
            1,
            resident_source(stable, experts, &weights, &bias),
        )
        .unwrap();
        let host_pools = RelayPinnedPools::warm_exact(&host_router, 1).unwrap();
        let resident_pools = RelayPinnedPools::warm_exact(&resident_router, 1).unwrap();
        let mut host_reservation = host_pools.try_reserve_all(70).unwrap();
        let mut resident_reservation = resident_pools.try_reserve_all(71).unwrap();
        let host = host_router
            .execute_and_download(
                0,
                GptOssPhase::Decode,
                9,
                1,
                &activation,
                &mut host_reservation.source_activation,
                &mut host_reservation.route_descriptors,
                None,
            )
            .unwrap();
        let resident = resident_router
            .execute_and_download(
                0,
                GptOssPhase::Decode,
                9,
                1,
                &activation,
                &mut resident_reservation.source_activation,
                &mut resident_reservation.route_descriptors,
                None,
            )
            .unwrap();
        assert_eq!(
            resident.router_logits_bf16_bits,
            host.router_logits_bf16_bits
        );
        assert_eq!(resident.batch, host.batch);
        assert_eq!(resident_router.stable_device(), host_router.stable_device());
        assert_eq!(
            resident_router.owned_device_bytes().unwrap(),
            host_router.owned_device_bytes().unwrap()
        );
        host_reservation.release_drained().unwrap();
        resident_reservation.release_drained().unwrap();
        host_router.drain().unwrap();
        resident_router.drain().unwrap();
    }
}

#[test]
fn resident_router_handoff_rejects_wrong_device_context_and_shapes() {
    let devices = list_devices();
    assert_eq!(devices.len(), 2, "resident handoff gate requires two GPUs");
    let stable0 = StableCudaDeviceId::from_device(&devices[0]).unwrap();
    let stable1 = StableCudaDeviceId::from_device(&devices[1]).unwrap();
    let (_, weights, bias) = fixture(32, 1);

    let ordinal0 = devices[0].id;
    let ordinal1 = devices[1].id;
    let context0 = CudaContext::new(ordinal0).unwrap();
    let context1 = CudaContext::new(ordinal1).unwrap();
    let stream0 = context0.new_stream().unwrap();
    let stream1 = context1.new_stream().unwrap();
    let weight0 = stream0
        .clone_htod::<u8, _>(bytemuck::cast_slice(&weights))
        .unwrap();
    let bias0 = stream0
        .clone_htod::<u8, _>(bytemuck::cast_slice(&bias))
        .unwrap();
    stream0.synchronize().unwrap();
    assert!(ResidentExactRouterWeights::new(stable1, 32, weight0, bias0).is_err());

    let weight0 = stream0
        .clone_htod::<u8, _>(bytemuck::cast_slice(&weights))
        .unwrap();
    let bias1 = stream1
        .clone_htod::<u8, _>(bytemuck::cast_slice(&bias))
        .unwrap();
    stream0.synchronize().unwrap();
    stream1.synchronize().unwrap();
    assert!(ResidentExactRouterWeights::new(stable0.clone(), 32, weight0, bias1).is_err());

    let short_weights = stream0
        .clone_htod::<u8, _>(&bytemuck::cast_slice::<u16, u8>(&weights)[..16])
        .unwrap();
    let bias0 = stream0
        .clone_htod::<u8, _>(bytemuck::cast_slice(&bias))
        .unwrap();
    stream0.synchronize().unwrap();
    assert!(ResidentExactRouterWeights::new(stable0.clone(), 32, short_weights, bias0).is_err());

    let weight0 = stream0
        .clone_htod::<u8, _>(bytemuck::cast_slice(&weights))
        .unwrap();
    let short_bias = stream0
        .clone_htod::<u8, _>(&bytemuck::cast_slice::<u16, u8>(&bias)[..2])
        .unwrap();
    stream0.synchronize().unwrap();
    assert!(ResidentExactRouterWeights::new(stable0.clone(), 32, weight0, short_bias).is_err());

    let weight0 = stream0
        .clone_htod::<u8, _>(bytemuck::cast_slice(&weights))
        .unwrap();
    let bias0 = stream0
        .clone_htod::<u8, _>(bytemuck::cast_slice(&bias))
        .unwrap();
    stream0.synchronize().unwrap();
    assert!(ResidentExactRouterWeights::new(stable0.clone(), 64, weight0, bias0).is_err());

    assert!(CudaExactRouter::from_resident_weights(
        0,
        resident_source(stable0.clone(), 32, &weights, &bias),
    )
    .is_err());
    assert!(CudaExactRouter::from_resident_weights(
        GPT_OSS_ROUTER_MAX_ROWS + 1,
        resident_source(stable0, 32, &weights, &bias),
    )
    .is_err());
}

#[test]
fn resident_router_sources_are_ordered_and_single_consumer() {
    let devices = list_devices();
    assert!(!devices.is_empty(), "resident source gate requires one GPU");
    let stable = StableCudaDeviceId::from_device(&devices[0]).unwrap();
    let (_, weights, bias) = fixture(32, 1);

    let reordered = vec![
        (1, resident_source(stable.clone(), 32, &weights, &bias)),
        (0, resident_source(stable.clone(), 32, &weights, &bias)),
    ];
    assert!(ResidentExactRouterSources::new(2, 32, stable.clone(), reordered).is_err());
    assert!(ResidentExactRouterSources::new(0, 32, stable.clone(), Vec::new()).is_err());

    let source = resident_source(stable.clone(), 32, &weights, &bias);
    let expected_bytes = source.device_bytes().unwrap();
    let mut sources = ResidentExactRouterSources::new(1, 32, stable, vec![(0, source)]).unwrap();
    assert_eq!(sources.available_layers(), 1);
    assert_eq!(sources.source_tensor_count().unwrap(), 2);
    assert_eq!(sources.device_bytes().unwrap(), expected_bytes);
    let mut taken = sources.take_ordered().unwrap();
    assert_eq!(taken.len(), 1);
    assert_eq!(sources.available_layers(), 0);
    assert_eq!(sources.source_tensor_count().unwrap(), 0);
    assert_eq!(sources.device_bytes().unwrap(), 0);
    assert!(sources.take_ordered().is_err());

    let mut router = CudaExactRouter::from_resident_weights(1, taken.pop().unwrap()).unwrap();
    router.drain().unwrap();
}

#[cfg(feature = "heterogeneous-test-faults")]
#[test]
fn resident_router_source_drops_only_after_terminal_and_unproven_state_is_quarantined() {
    use std::sync::{atomic::AtomicBool, atomic::Ordering, Arc};

    let devices = list_devices();
    let stable = StableCudaDeviceId::from_device(&devices[0]).unwrap();
    let (_, weights, bias) = fixture(32, 1);

    let success_drop = Arc::new(AtomicBool::new(false));
    let mut success_source = resident_source(stable.clone(), 32, &weights, &bias);
    success_source.set_drop_probe_for_test(Arc::clone(&success_drop));
    let mut router = CudaExactRouter::from_resident_weights(1, success_source).unwrap();
    assert!(success_drop.load(Ordering::Acquire));
    router.drain().unwrap();

    let recoverable_drop = Arc::new(AtomicBool::new(false));
    let mut recoverable_source = resident_source(stable.clone(), 32, &weights, &bias);
    recoverable_source.set_drop_probe_for_test(Arc::clone(&recoverable_drop));
    recoverable_source
        .inject_handoff_failure(ResidentRouterHandoffInjectedFault::AfterWeightCopyEnqueue)
        .unwrap();
    assert!(CudaExactRouter::from_resident_weights(1, recoverable_source).is_err());
    assert!(recoverable_drop.load(Ordering::Acquire));
    let mut retry = CudaExactRouter::from_resident_weights(
        1,
        resident_source(stable.clone(), 32, &weights, &bias),
    )
    .unwrap();
    retry.drain().unwrap();

    let quarantined_before = resident_router_handoff_quarantines_for_test();
    let failure_drop = Arc::new(AtomicBool::new(false));
    let mut failure_source = resident_source(stable, 32, &weights, &bias);
    failure_source.set_drop_probe_for_test(Arc::clone(&failure_drop));
    failure_source
        .inject_handoff_failure(
            ResidentRouterHandoffInjectedFault::AfterWeightCopyEnqueueAndFallbackDrainFailure,
        )
        .unwrap();
    let error = match CudaExactRouter::from_resident_weights(1, failure_source) {
        Ok(_) => panic!("unproven resident handoff unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("source, destination, stream, and context are quarantined"));
    assert!(!failure_drop.load(Ordering::Acquire));
    assert_eq!(
        resident_router_handoff_quarantines_for_test(),
        quarantined_before + 1
    );
}

#[test]
fn gpu0_router_is_bit_exact_for_e32_e128_and_fixed_pools_reuse() {
    for (experts, rows, phase) in [(32, 1, GptOssPhase::Decode), (128, 3, GptOssPhase::Prefill)] {
        let (activation, weights, bias) = fixture(experts, rows);
        let view = ExactRouterWeightsView {
            experts,
            weight_bf16_bits: &weights,
            bias_bf16_bits: &bias,
        };
        let expected = exact_router_reference(0, phase, 17, rows, &activation, view).unwrap();
        let mut router = CudaExactRouter::new(layer_owner(), rows, view).unwrap();
        let pools = RelayPinnedPools::warm_exact(&router, rows).unwrap();
        let stats = pools.stats();
        assert_eq!(
            stats.raw_capacity_bytes,
            if rows == 1 { 74_944 } else { 224_832 }
        );
        assert!(stats.raw_capacity_bytes <= stats.hard_cap_bytes);

        let mut reservation = pools.try_reserve_all(19).unwrap();
        assert!(pools.try_reserve_all(20).is_err());
        let timeline = CorrelatedTimeline::new();
        let actual = router
            .execute_and_download(
                0,
                phase,
                17,
                rows,
                &activation,
                &mut reservation.source_activation,
                &mut reservation.route_descriptors,
                Some(&timeline),
            )
            .unwrap();
        assert_eq!(
            actual.router_logits_bf16_bits,
            expected.router_logits_bf16_bits
        );
        assert_eq!(actual.batch.routes, expected.batch.routes);
        assert_eq!(
            &reservation.source_activation.as_slice()[..activation.len()],
            activation
        );
        assert!(timeline
            .points()
            .iter()
            .any(|point| point.label == "source_d2h_end"));
        reservation.release_drained().unwrap();
        let reused = pools.try_reserve_all(21).unwrap();
        reused.release_drained().unwrap();
        let final_stats = pools.stats();
        for pool in [
            final_stats.source_activation,
            final_stats.route_descriptors,
            final_stats.remote_gpu_input,
            final_stats.remote_gpu_result,
            final_stats.cpu_result,
        ] {
            assert_eq!(pool.fixed_allocations, 1);
            assert_eq!(pool.available, 1);
            assert_eq!(pool.checked_out, 0);
            assert_eq!(pool.high_water, 1);
            assert_eq!(pool.quarantined, 0);
        }
    }
}

#[test]
fn gpu0_router_preserves_lower_id_ties_and_rejects_non_finite_logits() {
    let (activation, mut weights, mut bias) = fixture(32, 1);
    weights.fill(bf16::from_f32(0.0).to_bits());
    bias.fill(bf16::from_f32(0.0).to_bits());
    let view = ExactRouterWeightsView {
        experts: 32,
        weight_bf16_bits: &weights,
        bias_bf16_bits: &bias,
    };
    let mut router = CudaExactRouter::new(layer_owner(), 1, view).unwrap();
    let pools = RelayPinnedPools::warm_exact(&router, 1).unwrap();
    let mut reservation = pools.try_reserve_all(30).unwrap();
    let actual = router
        .execute_and_download(
            0,
            GptOssPhase::Decode,
            1,
            1,
            &activation,
            &mut reservation.source_activation,
            &mut reservation.route_descriptors,
            None,
        )
        .unwrap();
    assert_eq!(
        actual
            .batch
            .routes
            .iter()
            .map(|route| route.expert_id)
            .collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    assert!(actual
        .batch
        .routes
        .iter()
        .all(|route| { route.weight_bf16_bits == bf16::from_f32(0.25).to_bits() }));
    reservation.release_drained().unwrap();

    bias[7] = bf16::NAN.to_bits();
    let non_finite_view = ExactRouterWeightsView {
        experts: 32,
        weight_bf16_bits: &weights,
        bias_bf16_bits: &bias,
    };
    let mut router = CudaExactRouter::new(layer_owner(), 1, non_finite_view).unwrap();
    let pools = RelayPinnedPools::warm_exact(&router, 1).unwrap();
    let mut reservation = pools.try_reserve_all(31).unwrap();
    assert!(router
        .execute_and_download(
            0,
            GptOssPhase::Decode,
            1,
            1,
            &activation,
            &mut reservation.source_activation,
            &mut reservation.route_descriptors,
            None,
        )
        .is_err());
    reservation.release_drained().unwrap();
}

#[cfg(feature = "heterogeneous-test-faults")]
#[test]
fn router_post_enqueue_faults_drain_before_pinned_reuse() {
    let (activation, weights, bias) = fixture(32, 1);
    let view = ExactRouterWeightsView {
        experts: 32,
        weight_bf16_bits: &weights,
        bias_bf16_bits: &bias,
    };
    let mut router = CudaExactRouter::new(layer_owner(), 1, view).unwrap();
    let pools = RelayPinnedPools::warm_exact(&router, 1).unwrap();
    for (generation, fault) in [
        (40, ExactRouterInjectedFault::SubmitAfterInputEnqueue),
        (41, ExactRouterInjectedFault::RelayAfterSourceEnqueue),
    ] {
        let mut reservation = pools.try_reserve_all(generation).unwrap();
        router.inject_next_failure(fault).unwrap();
        assert!(router
            .execute_and_download(
                0,
                GptOssPhase::Decode,
                1,
                1,
                &activation,
                &mut reservation.source_activation,
                &mut reservation.route_descriptors,
                None,
            )
            .is_err());
        assert!(router.last_fault_drained());
        reservation.release_drained().unwrap();
    }
    let reservation = pools.try_reserve_all(42).unwrap();
    reservation.release_drained().unwrap();
    assert!(pools.stats().source_activation.quarantined == 0);

    // Keep this import live alongside the relay fault feature; the actual
    // post-result-enqueue fault is exercised by the detached relay test.
    let _ = ResultRelayInjectedFault::AfterFirstResultEnqueue;
}

#[cfg(feature = "heterogeneous-test-faults")]
#[test]
fn router_unproven_fallback_drain_quarantines_cuda_and_pinned_state() {
    let (activation, weights, bias) = fixture(32, 1);
    let view = ExactRouterWeightsView {
        experts: 32,
        weight_bf16_bits: &weights,
        bias_bf16_bits: &bias,
    };
    let mut router = CudaExactRouter::new(layer_owner(), 1, view).unwrap();
    let pools = RelayPinnedPools::warm_exact(&router, 1).unwrap();
    let mut reservation = pools.try_reserve_all(43).unwrap();
    router
        .inject_next_failure(
            ExactRouterInjectedFault::SubmitAfterInputEnqueueAndFallbackDrainFailure,
        )
        .unwrap();
    let error = router
        .execute_and_download(
            0,
            GptOssPhase::Decode,
            1,
            1,
            &activation,
            &mut reservation.source_activation,
            &mut reservation.route_descriptors,
            None,
        )
        .unwrap_err();
    assert!(error.to_string().contains("caller-owned pinned leases"));
    assert!(router.device_state_quarantined_for_test());
    assert!(router.drain().is_err());
    assert!(router
        .execute_and_download(
            0,
            GptOssPhase::Decode,
            1,
            1,
            &activation,
            &mut reservation.source_activation,
            &mut reservation.route_descriptors,
            None,
        )
        .is_err());
    // The injected unproven drain deliberately retains every lease which the
    // router's async H2D/D2H may still reference.
    std::mem::forget(reservation);
    drop(router);
    let stats = pools.stats();
    assert_eq!(stats.source_activation.checked_out, 1);
    assert_eq!(stats.route_descriptors.checked_out, 1);
    assert_eq!(stats.remote_gpu_input.checked_out, 1);
    assert_eq!(stats.remote_gpu_result.checked_out, 1);
    assert_eq!(stats.cpu_result.checked_out, 1);
}

const _: () = assert!(GPT_OSS_ROUTER_MAX_ROWS == 64);

#[derive(Serialize)]
struct NativeRouterEvidence {
    schema: &'static str,
    captured_unix_seconds: u64,
    repository_head: String,
    source_fingerprint_sha256: String,
    executable_sha256: String,
    router_ptx_sha256: String,
    cargo_profile: String,
    cuda_arch: String,
    layer_owner_pci_bus_id: String,
    cases: Vec<NativeRouterCase>,
    passed: bool,
}

#[derive(Serialize)]
struct NativeRouterCase {
    experts: usize,
    revision: String,
    mapping_sha256: String,
    router_weight_sha256: String,
    router_bias_sha256: String,
    input_sha256: String,
    logits_sha256: String,
    selected_ids: Vec<u16>,
    selected_weights_bf16_bits: Vec<u16>,
    source_d2h_bytes: usize,
    descriptor_d2h_bytes: usize,
    fixed_pinned_bytes: usize,
    router_elapsed_ms: f32,
}

#[test]
fn native_e32_e128_router_weights_are_bit_exact_on_layer_owner_gpu() {
    if std::env::var_os("GPT_OSS_RUN_H4_NATIVE_ROUTER").is_none() {
        eprintln!("GPT_OSS_RUN_H4_NATIVE_ROUTER is unset; skipping native E=32/E=128 router gate");
        return;
    }
    let layer_owner = layer_owner();
    let mut cases = Vec::new();
    for (path, experts) in [
        ("/data/models/openai/gpt-oss-20b/original", 32),
        ("/data/models/openai/gpt-oss-120b/original", 128),
    ] {
        let checkpoint = GptOssCheckpointView::open(path).unwrap();
        assert_eq!(checkpoint.config().num_experts, experts);
        let weight = checkpoint
            .tensor("model.layers.0.mlp.router.weight")
            .unwrap();
        let bias = checkpoint.tensor("model.layers.0.mlp.router.bias").unwrap();
        let weight_bits: &[u16] = bytemuck::try_cast_slice(weight.bytes()).unwrap();
        let bias_bits: &[u16] = bytemuck::try_cast_slice(bias.bytes()).unwrap();
        assert_eq!(weight_bits.len(), experts * HIDDEN_SIZE);
        assert_eq!(bias_bits.len(), experts);
        let activation = (0..HIDDEN_SIZE)
            .map(|index| {
                bf16::from_f32((((index * 13 + experts) % 97) as f32 - 48.0) / 64.0).to_bits()
            })
            .collect::<Vec<_>>();
        let view = ExactRouterWeightsView {
            experts,
            weight_bf16_bits: weight_bits,
            bias_bf16_bits: bias_bits,
        };
        let expected =
            exact_router_reference(0, GptOssPhase::Decode, 73, 1, &activation, view).unwrap();
        let mut router = CudaExactRouter::new(layer_owner.clone(), 1, view).unwrap();
        let pools = RelayPinnedPools::warm_exact(&router, 1).unwrap();
        let mut reservation = pools.try_reserve_all(74).unwrap();
        let actual = router
            .execute_and_download(
                0,
                GptOssPhase::Decode,
                73,
                1,
                &activation,
                &mut reservation.source_activation,
                &mut reservation.route_descriptors,
                None,
            )
            .unwrap();
        assert_eq!(
            actual.router_logits_bf16_bits,
            expected.router_logits_bf16_bits
        );
        assert_eq!(actual.batch.routes, expected.batch.routes);
        assert_eq!(actual.source_d2h_bytes, 5_760);
        assert_eq!(actual.descriptor_d2h_bytes, 64);
        let stats = pools.stats();
        assert_eq!(stats.raw_capacity_bytes, 74_944);
        cases.push(NativeRouterCase {
            experts,
            revision: checkpoint.revision().to_string(),
            mapping_sha256: checkpoint.mapping_sha256().to_string(),
            router_weight_sha256: hash(weight.bytes()),
            router_bias_sha256: hash(bias.bytes()),
            input_sha256: hash(bytemuck::cast_slice(&activation)),
            logits_sha256: hash(bytemuck::cast_slice(&actual.router_logits_bf16_bits)),
            selected_ids: actual
                .batch
                .routes
                .iter()
                .map(|route| route.expert_id)
                .collect(),
            selected_weights_bf16_bits: actual
                .batch
                .routes
                .iter()
                .map(|route| route.weight_bf16_bits)
                .collect(),
            source_d2h_bytes: actual.source_d2h_bytes,
            descriptor_d2h_bytes: actual.descriptor_d2h_bytes,
            fixed_pinned_bytes: stats.raw_capacity_bytes,
            router_elapsed_ms: actual.router_elapsed_ms,
        });
        reservation.release_drained().unwrap();
    }
    let evidence = NativeRouterEvidence {
        schema: "gpt-oss-rs.heterogeneous-h4-native-router/v1",
        captured_unix_seconds: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        repository_head: required_evidence_env("GPT_OSS_H4_REPO_HEAD"),
        source_fingerprint_sha256: required_evidence_env("GPT_OSS_H4_SOURCE_FINGERPRINT"),
        executable_sha256: hash_file(&std::env::current_exe().unwrap()),
        router_ptx_sha256: hash_file(&compiled_ptx_dir().join("gpt_oss_router.ptx")),
        cargo_profile: required_evidence_env("GPT_OSS_H4_CARGO_PROFILE"),
        cuda_arch: required_evidence_env("CUDA_ARCH"),
        layer_owner_pci_bus_id: layer_owner.pci_bus_id.to_string(),
        cases,
        passed: true,
    };
    if let Some(path) = std::env::var_os("GPT_OSS_H4_ROUTER_EVIDENCE") {
        let path = std::path::PathBuf::from(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut bytes = serde_json::to_vec_pretty(&evidence).unwrap();
        bytes.push(b'\n');
        std::fs::write(path, bytes).unwrap();
    }
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn required_evidence_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required when writing H4 evidence"))
}

fn hash_file(path: &std::path::Path) -> String {
    use std::io::Read;

    let mut file = std::fs::File::open(path).unwrap();
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).unwrap();
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    format!("{:x}", hasher.finalize())
}
