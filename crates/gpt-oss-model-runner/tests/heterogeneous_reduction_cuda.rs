#![cfg(all(feature = "cuda", feature = "heterogeneous-test-faults"))]

use std::path::Path;

use gpt_oss_gpu::device::{list_devices, StableCudaDeviceId};
use gpt_oss_gpu::kernel_loader::compiled_ptx_dir;
use gpt_oss_model_runner::cpu_repack::CpuOwnerRepackCache;
use gpt_oss_model_runner::heterogeneous::{
    exact_rank_ordered_reduction_reference, exact_selected_expert_reference, pack_remote_inputs,
    pack_routes_bounded, CanonicalExpertContribution, CanonicalRouteContract,
    CpuX8SelectedExpertWorker, CudaExactRouter, CudaRankOrderedReducer, CudaResultRelay,
    CudaSelectedExpertExecutor, ExactRouterWeightsView, GptOssExpertKey,
    GptOssExpertPlacementManifestV1, GptOssPhase, NativeMxfp4ExpertView, PackedDispatchPlan,
    PreparedRankOrderedReduction, RankReductionConstructionFault, RankReductionInjectedFault,
    RelayPinnedPools, RelayPinnedReservation, DOWN_BIAS_VALUES, DOWN_BLOCK_BYTES, DOWN_SCALE_BYTES,
    GATE_UP_BIAS_VALUES, GATE_UP_BLOCK_BYTES, GATE_UP_SCALE_BYTES,
    GPT_OSS_REDUCER_OWNED_DEVICE_BYTES, GPT_OSS_REDUCTION_CONTRIBUTION_BYTES,
    GPT_OSS_REDUCTION_DEVICE_WORK_BYTES, GPT_OSS_REDUCTION_WORKSPACE_CLASS_BYTES, HIDDEN_SIZE,
};
use gpt_oss_model_runner::model_loader::gpt_oss_native::GptOssCheckpointView;
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};

const MODEL: &str = "/data/models/openai/gpt-oss-20b/original";
const PLACEMENT: &str = "docs/het/evidence/implementation-2026-08/h3/placement-20b.json";
const CACHE: &str = "/home/emmy/workspace/gpt-oss-rs-het-cache";
const IDENTITY: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const GENERATION: u64 = 101;

#[derive(Serialize)]
struct H5ReductionEvidence<'a> {
    schema: &'static str,
    captured_unix_seconds: u64,
    repository_head: String,
    source_fingerprint_sha256: String,
    executable_sha256: String,
    reducer_ptx_sha256: String,
    cargo_profile: String,
    cuda_arch: String,
    model_mapping_sha256: &'a str,
    placement_sha256: &'a str,
    layer_owner_pci_bus_id: String,
    remote_worker_pci_bus_id: String,
    transaction_generation: u64,
    selected_ids: Vec<u16>,
    expected_result_descriptors: usize,
    local_before_clear_rejected: bool,
    stale_local_generation_rejected: bool,
    stale_reducer_generation_rejected: bool,
    premature_rebind_rejected: bool,
    drained_abandon_then_higher_generation: bool,
    successful_reduction_closed_generation: bool,
    reduction_faults_drained: Vec<&'static str>,
    construction_faults_drained: Vec<&'static str>,
    identity_negatives: Vec<&'static str>,
    contribution_arena_bytes: usize,
    reducer_owned_device_bytes: usize,
    measured_driver_free_delta_bytes: usize,
    driver_allocator_reuse_observed: bool,
    reducer_drop_recovered_within_bytes: usize,
    pipeline_device_bytes: usize,
    workspace_class_bytes: usize,
    pinned_high_water: [usize; 5],
    output_sha256: String,
    weighted_trace_sha256: String,
    accumulator_trace_sha256: String,
    reducer_kernel_elapsed_ms: f32,
    passed: bool,
}

#[test]
fn reducer_unproven_fallback_drain_quarantines_reducer_and_shared_relay_state() {
    let devices = list_devices();
    assert_eq!(devices.len(), 2);
    let manifest: GptOssExpertPlacementManifestV1 =
        serde_json::from_slice(&std::fs::read(repo_root().join(PLACEMENT)).unwrap()).unwrap();
    let placement = manifest.validate(&devices).unwrap();
    let router_weights = vec![bf16::from_f32(0.0).to_bits(); 32 * HIDDEN_SIZE];
    let mut router_bias = vec![bf16::from_f32(0.0).to_bits(); 32];
    for (expert, logit) in [(31, 4.0), (21, 3.0), (22, 2.0), (6, 1.0)] {
        router_bias[expert] = bf16::from_f32(logit).to_bits();
    }
    let mut router = CudaExactRouter::new(
        placement.layer_owner().stable_id.clone(),
        1,
        ExactRouterWeightsView {
            experts: 32,
            weight_bf16_bits: &router_weights,
            bias_bf16_bits: &router_bias,
        },
    )
    .unwrap();
    let pools = RelayPinnedPools::warm_exact(&router, 1).unwrap();
    let mut reservation = pools.try_reserve_all(902).unwrap();
    let activation = vec![bf16::from_f32(0.0).to_bits(); HIDDEN_SIZE];
    let routed = router
        .execute_and_download(
            0,
            GptOssPhase::Decode,
            placement.placement_epoch(),
            1,
            &activation,
            &mut reservation.source_activation,
            &mut reservation.route_descriptors,
            None,
        )
        .unwrap();
    let plan = pack_routes_bounded(&routed.batch, &placement).unwrap();
    let prepared = PreparedRankOrderedReduction::prepare(&routed.batch, &placement, 902).unwrap();
    let descriptors = prepared.expected_results().to_vec();
    let outputs = (0..4)
        .map(|rank| vec![bf16::from_f32(rank as f32).to_bits(); HIDDEN_SIZE])
        .collect::<Vec<_>>();
    let mut relay = CudaResultRelay::new(&router, 1).unwrap();
    relay.bind_decode_generation(902, &plan).unwrap();
    relay
        .upload_cpu_authority_control(902, &descriptors, outputs)
        .unwrap();
    let mut reducer = CudaRankOrderedReducer::new(&relay).unwrap();
    reducer
        .inject_next_failure(RankReductionInjectedFault::AfterKernelLaunchAndFallbackDrainFailure)
        .unwrap();
    let error = reducer.reduce_relay(&mut relay, prepared).unwrap_err();
    assert!(error
        .to_string()
        .contains("all CUDA and host D2H state quarantined"));
    assert!(reducer.device_state_quarantined_for_test());
    assert!(relay.device_state_quarantined_for_test());
    drop(reducer);
    drop(relay);
    reservation.release_drained().unwrap();
}

#[test]
fn real_three_owner_rank_reduction_uses_h4_arena_and_is_generation_bound() {
    if std::env::var_os("GPT_OSS_RUN_H5_REAL").is_none() {
        eprintln!("GPT_OSS_RUN_H5_REAL is unset; skipping real H5 reduction gate");
        return;
    }
    let devices = list_devices();
    assert_eq!(
        devices.len(),
        2,
        "H5 reduction gate requires both local GPUs"
    );
    let manifest: GptOssExpertPlacementManifestV1 =
        serde_json::from_slice(&std::fs::read(repo_root().join(PLACEMENT)).unwrap()).unwrap();
    let placement = manifest.validate(&devices).unwrap();
    let layer_owner = placement.layer_owner().stable_id.clone();
    let remote_worker = placement.remote_worker().stable_id.clone();
    assert_eq!(
        layer_owner,
        StableCudaDeviceId::from_device(&devices[0]).unwrap()
    );
    assert_eq!(
        remote_worker,
        StableCudaDeviceId::from_device(&devices[1]).unwrap()
    );

    let checkpoint = GptOssCheckpointView::open(MODEL).unwrap();
    let cache = CpuOwnerRepackCache::new(
        CACHE,
        checkpoint.revision(),
        checkpoint.mapping_sha256(),
        placement.manifest_hash(),
        64 * 1024 * 1024,
    )
    .unwrap();
    let cpu_record = cache.open_or_create_layer(&checkpoint, 0, &[21]).unwrap();
    assert_eq!(cpu_record.expert_ids(), [21]);

    let activation = (0..HIDDEN_SIZE)
        .map(|index| bf16::from_f32(((index % 31) as f32 - 15.0) / 16.0).to_bits())
        .collect::<Vec<_>>();
    let router_weights = vec![bf16::from_f32(0.0).to_bits(); 32 * HIDDEN_SIZE];
    let mut router_bias = vec![bf16::from_f32(0.0).to_bits(); 32];
    for (expert, logit) in [(31, 4.0), (21, 3.0), (22, 2.0), (6, 1.0)] {
        router_bias[expert] = bf16::from_f32(logit).to_bits();
    }
    let router_view = ExactRouterWeightsView {
        experts: 32,
        weight_bf16_bits: &router_weights,
        bias_bf16_bits: &router_bias,
    };
    let mut router = CudaExactRouter::new(layer_owner.clone(), 1, router_view).unwrap();
    let pools = RelayPinnedPools::warm_exact(&router, 1).unwrap();
    let mut reservation = pools.try_reserve_all(GENERATION).unwrap();
    let routed = router
        .execute_and_download(
            0,
            GptOssPhase::Decode,
            placement.placement_epoch(),
            1,
            &activation,
            &mut reservation.source_activation,
            &mut reservation.route_descriptors,
            None,
        )
        .unwrap();
    let selected_ids = routed
        .batch
        .routes
        .iter()
        .map(|route| route.expert_id)
        .collect::<Vec<_>>();
    assert_eq!(selected_ids, [31, 21, 22, 6]);
    let plan = pack_routes_bounded(&routed.batch, &placement).unwrap();
    pack_remote_inputs(
        &plan,
        &reservation.source_activation,
        &mut reservation.remote_gpu_input,
    )
    .unwrap();

    // Every allocation/canonical descriptor/trace is prepared before any
    // selected-expert work is submitted.
    let prepared_clean =
        PreparedRankOrderedReduction::prepare(&routed.batch, &placement, GENERATION).unwrap();
    let expected_descriptors = prepared_clean.expected_results().to_vec();
    let prepared_stale =
        PreparedRankOrderedReduction::prepare(&routed.batch, &placement, GENERATION - 1).unwrap();
    let prepared_fault_weight =
        PreparedRankOrderedReduction::prepare(&routed.batch, &placement, GENERATION).unwrap();
    let prepared_fault_kernel =
        PreparedRankOrderedReduction::prepare(&routed.batch, &placement, GENERATION).unwrap();
    let prepared_fault_evidence =
        PreparedRankOrderedReduction::prepare(&routed.batch, &placement, GENERATION).unwrap();
    let local_routes = plan
        .local_gpu
        .iter()
        .flat_map(|owner| owner.routes.iter())
        .collect::<Vec<_>>();
    assert_eq!(local_routes.len(), 2);
    let cpu_route = &plan.cpu[0].routes[0];
    let remote_route = &plan.remote_gpu[0].routes[0];

    let local_sources = local_routes
        .iter()
        .map(|route| native_expert(&checkpoint, 0, route.descriptor.route.expert_id))
        .collect::<Vec<_>>();
    let remote_source = native_expert(&checkpoint, 0, remote_route.descriptor.route.expert_id);
    let cpu_source = native_expert(&checkpoint, 0, cpu_route.descriptor.route.expert_id);
    let expected_outputs = expected_descriptors
        .iter()
        .map(|descriptor| {
            let source = native_expert(&checkpoint, 0, descriptor.expert_id);
            exact_selected_expert_reference(source, &activation)
                .unwrap()
                .down_bf16_bits
        })
        .collect::<Vec<_>>();
    let contributions = expected_descriptors
        .iter()
        .cloned()
        .zip(expected_outputs.iter().cloned())
        .map(
            |(descriptor, output_bf16_bits)| CanonicalExpertContribution {
                descriptor,
                output_bf16_bits,
            },
        )
        .collect::<Vec<_>>();
    let oracle =
        exact_rank_ordered_reduction_reference(&routed.batch, &placement, &contributions).unwrap();
    let expected_nonlocal = plan
        .cpu
        .iter()
        .chain(&plan.remote_gpu)
        .flat_map(|owner| owner.routes.iter())
        .map(|route| CanonicalRouteContract::from_packed_route(&route.descriptor))
        .collect::<Vec<_>>();
    assert_eq!(expected_nonlocal.len(), 2);
    let missing_completion = vec![expected_nonlocal[0]];
    let duplicate_completion = vec![expected_nonlocal[0], expected_nonlocal[0]];
    let mut wrong_expert = expected_nonlocal.clone();
    wrong_expert[0].expert_id ^= 1;
    let mut wrong_weight = expected_nonlocal.clone();
    wrong_weight[0].weight_bf16_bits ^= 1;
    let mut wrong_owner = expected_nonlocal.clone();
    wrong_owner[0].owner = expected_nonlocal[1].owner;
    let mut wrong_slot = expected_nonlocal.clone();
    wrong_slot[0].result_slot = (wrong_slot[0].result_slot + 1) % 4;
    let mut wrong_plan = plan.clone();
    wrong_plan.cpu[0].routes[0].descriptor.route.expert_id ^= 1;

    let mut relay = CudaResultRelay::new(&router, 1).unwrap();
    relay.bind_decode_generation(GENERATION - 1, &plan).unwrap();
    assert!(relay.bind_decode_generation(GENERATION, &plan).is_err());
    assert!(relay
        .abandon_decode_generation(GENERATION - 1, false)
        .is_err());
    relay
        .abandon_decode_generation(GENERATION - 1, true)
        .unwrap();
    relay.bind_decode_generation(GENERATION, &plan).unwrap();
    assert!(relay.bind_decode_generation(GENERATION - 1, &plan).is_err());
    let (free_before_reducer, _) = relay.memory_info().unwrap();
    let mut construction_faults_drained = Vec::with_capacity(4);
    for (fault, label) in [
        (
            RankReductionConstructionFault::AfterWeights,
            "after_weights",
        ),
        (RankReductionConstructionFault::AfterOutput, "after_output"),
        (
            RankReductionConstructionFault::AfterWeightedTrace,
            "after_weighted_trace",
        ),
        (
            RankReductionConstructionFault::AfterAccumulatorTrace,
            "after_accumulator_trace",
        ),
    ] {
        let (before, _) = relay.memory_info().unwrap();
        assert!(CudaRankOrderedReducer::new_with_construction_fault(&relay, fault).is_err());
        let (after, _) = relay.memory_info().unwrap();
        assert!(before.abs_diff(after) <= 4 * 1024);
        construction_faults_drained.push(label);
    }
    let mut reducer = CudaRankOrderedReducer::new(&relay).unwrap();
    let (free_after_reducer, _) = relay.memory_info().unwrap();
    let measured_driver_free_delta_bytes = free_before_reducer.saturating_sub(free_after_reducer);
    assert!(
        measured_driver_free_delta_bytes <= GPT_OSS_REDUCTION_WORKSPACE_CLASS_BYTES,
        "CUDA free-memory delta {measured_driver_free_delta_bytes} exceeds workspace class {}",
        GPT_OSS_REDUCTION_WORKSPACE_CLASS_BYTES,
    );
    let driver_allocator_reuse_observed =
        measured_driver_free_delta_bytes < GPT_OSS_REDUCER_OWNED_DEVICE_BYTES;
    assert_eq!(reducer.stable_device(), &layer_owner);
    assert_eq!(
        reducer.owned_device_bytes(),
        GPT_OSS_REDUCER_OWNED_DEVICE_BYTES
    );
    assert_eq!(
        reducer.pipeline_device_bytes(),
        GPT_OSS_REDUCTION_DEVICE_WORK_BYTES
    );

    let mut local_executor = CudaSelectedExpertExecutor::new(layer_owner.clone()).unwrap();
    let local_weights = local_routes
        .iter()
        .zip(local_sources.iter().copied())
        .map(|(route, source)| {
            local_executor
                .upload_expert(route.descriptor.owner.clone(), source)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut local_slots = (0..local_routes.len())
        .map(|index| {
            local_executor
                .allocate_result_slot_for_route(GENERATION, &local_routes[index].descriptor)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let stale_slot = local_executor
        .allocate_result_slot_for_route(GENERATION - 1, &local_routes[0].descriptor)
        .unwrap();
    let mut remote_executor = CudaSelectedExpertExecutor::new(remote_worker).unwrap();
    let remote_weights = remote_executor
        .upload_expert(remote_route.descriptor.owner.clone(), remote_source)
        .unwrap();
    let mut remote_slot = remote_executor
        .allocate_result_slot_for_route(GENERATION, &remote_route.descriptor)
        .unwrap();
    let mut cpu_worker = CpuX8SelectedExpertWorker::new();
    let mut local_completions = Vec::with_capacity(2);
    let mut nonlocal_completions = Vec::with_capacity(2);

    // Dispatch starts here. Completion APIs below perform no descriptor or
    // output allocation; they fill the pre-reserved slots/leases.
    for ((route, weights), result_slot) in local_routes
        .iter()
        .zip(local_weights.iter())
        .zip(local_slots.iter_mut())
    {
        let execution = local_executor
            .prepare(
                GptOssPhase::Decode,
                &route.descriptor,
                weights,
                &activation,
                result_slot,
            )
            .unwrap()
            .submit()
            .unwrap()
            .drain_device_only()
            .unwrap();
        local_completions.push(execution.route_contract);
    }
    let remote_start = remote_route.owner_route_slot as usize * HIDDEN_SIZE;
    let remote_execution = remote_executor
        .prepare(
            GptOssPhase::Decode,
            &remote_route.descriptor,
            &remote_weights,
            &reservation.remote_gpu_input.as_slice()[remote_start..remote_start + HIDDEN_SIZE],
            &mut remote_slot,
        )
        .unwrap()
        .submit()
        .unwrap()
        .drain_into_pinned_device_only(&mut reservation.remote_gpu_result, None)
        .unwrap();
    nonlocal_completions.push(remote_execution.route_contract);
    let cpu_execution = cpu_worker
        .execute_into_pinned_device_only(
            0,
            &cpu_route.descriptor,
            cpu_route.owner_route_slot,
            cpu_record.expert_view(21).unwrap(),
            &reservation.source_activation.as_slice()[..HIDDEN_SIZE],
            &mut reservation.cpu_result,
            None,
        )
        .unwrap();
    nonlocal_completions.push(cpu_execution.route_contract);
    assert_eq!(local_completions.len(), 2);
    for actual_completion in &local_completions {
        assert!(plan
            .local_gpu
            .iter()
            .flat_map(|owner| owner.routes.iter())
            .map(|route| CanonicalRouteContract::from_packed_route(&route.descriptor))
            .any(|expected| expected == *actual_completion));
    }

    let first_local_descriptor =
        &expected_descriptors[local_routes[0].descriptor.canonical_result_slot as usize];
    let early_slot = local_slots.remove(0);
    let early_failure = relay
        .upload_local_device_result(GENERATION, first_local_descriptor, early_slot)
        .unwrap_err();
    assert!(early_failure
        .error
        .to_string()
        .contains("completed arena clear"));
    let early_slot = early_failure
        .result_slot
        .expect("pre-clear rejection must return the unsubmitted local slot");

    // Every identity failure is rejected before arena clear/H2D and returns
    // the exact reservation, allowing the valid generation to continue.
    reservation =
        expect_bound_upload_rejection(&mut relay, &plan, reservation, &missing_completion);
    reservation =
        expect_bound_upload_rejection(&mut relay, &plan, reservation, &duplicate_completion);
    reservation = expect_bound_upload_rejection(&mut relay, &plan, reservation, &wrong_expert);
    reservation = expect_bound_upload_rejection(&mut relay, &plan, reservation, &wrong_weight);
    reservation = expect_bound_upload_rejection(&mut relay, &plan, reservation, &wrong_owner);
    reservation = expect_bound_upload_rejection(&mut relay, &plan, reservation, &wrong_slot);
    reservation =
        expect_bound_upload_rejection(&mut relay, &wrong_plan, reservation, &expected_nonlocal);
    for actual_completion in &nonlocal_completions {
        assert!(expected_nonlocal.contains(actual_completion));
    }

    let completed_relay = relay
        .upload_results_bound(&plan, reservation, &nonlocal_completions, None)
        .unwrap();
    let reservation = completed_relay.reservation;
    assert_eq!(completed_relay.execution.arena_generation, GENERATION);
    assert!(relay.bind_decode_generation(GENERATION + 1, &plan).is_err());

    let stale_failure = relay
        .upload_local_device_result(GENERATION, first_local_descriptor, stale_slot)
        .unwrap_err();
    assert!(stale_failure.result_slot.is_some());
    assert!(stale_failure
        .error
        .to_string()
        .contains("identity mismatch"));
    let second_local_route = local_routes[1];
    let second_descriptor =
        &expected_descriptors[second_local_route.descriptor.canonical_result_slot as usize];

    // A computed rank-0 slot cannot be relabeled as rank 3 even though both
    // routes have the same local owner device.
    let relabeled_failure = relay
        .upload_local_device_result(GENERATION, second_descriptor, early_slot)
        .unwrap_err();
    assert!(relabeled_failure.result_slot.is_some());
    let early_slot = relabeled_failure.result_slot.unwrap();

    relay
        .upload_local_device_result(GENERATION, first_local_descriptor, early_slot)
        .unwrap();
    relay
        .upload_local_device_result(GENERATION, second_descriptor, local_slots.remove(0))
        .unwrap();

    let stale_error = reducer
        .reduce_relay(&mut relay, prepared_stale)
        .unwrap_err();
    assert!(stale_error.to_string().contains("generation"));

    let mut reduction_faults_drained = Vec::with_capacity(3);
    for (fault, prepared, label) in [
        (
            RankReductionInjectedFault::AfterWeightEnqueue,
            prepared_fault_weight,
            "after_weight_enqueue",
        ),
        (
            RankReductionInjectedFault::AfterKernelLaunch,
            prepared_fault_kernel,
            "after_kernel_launch",
        ),
        (
            RankReductionInjectedFault::AfterEvidenceEnqueue,
            prepared_fault_evidence,
            "after_evidence_enqueue",
        ),
    ] {
        reducer.inject_next_failure(fault).unwrap();
        assert!(reducer.reduce_relay(&mut relay, prepared).is_err());
        assert!(reducer.last_fault_drained());
        reduction_faults_drained.push(label);
    }
    let actual = reducer.reduce_relay(&mut relay, prepared_clean).unwrap();
    assert_eq!(actual.output_bf16_bits, oracle.output_bf16_bits);
    assert_eq!(
        actual.trace.weighted_f32_bits,
        oracle.trace.weighted_f32_bits
    );
    assert_eq!(
        actual.trace.accumulator_f32_bits,
        oracle.trace.accumulator_f32_bits
    );
    drop(reducer);
    let (free_after_reducer_drop, _) = relay.memory_info().unwrap();
    let reducer_drop_recovered_within_bytes = free_before_reducer.abs_diff(free_after_reducer_drop);
    assert!(reducer_drop_recovered_within_bytes <= 4 * 1024);
    relay.bind_decode_generation(GENERATION + 1, &plan).unwrap();
    relay
        .abandon_decode_generation(GENERATION + 1, true)
        .unwrap();

    reservation.release_drained().unwrap();
    let after = pools.stats();
    let pool_stats = [
        after.source_activation,
        after.route_descriptors,
        after.remote_gpu_input,
        after.remote_gpu_result,
        after.cpu_result,
    ];
    for pool in pool_stats {
        assert_eq!(pool.available, 1);
        assert_eq!(pool.checked_out, 0);
        assert_eq!(pool.quarantined, 0);
        assert_eq!(pool.fixed_allocations, 1);
        assert_eq!(pool.high_water, 1);
    }

    if let Some(path) = std::env::var_os("GPT_OSS_H5_REDUCTION_EVIDENCE") {
        let evidence = H5ReductionEvidence {
            schema: "gpt-oss-rs.heterogeneous-h5-reduction/v1",
            captured_unix_seconds: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            repository_head: required_evidence_env("GPT_OSS_H5_REPO_HEAD"),
            source_fingerprint_sha256: required_evidence_env("GPT_OSS_H5_SOURCE_FINGERPRINT"),
            executable_sha256: hash_file(&std::env::current_exe().unwrap()),
            reducer_ptx_sha256: hash_file(&compiled_ptx_dir().join("gpt_oss_rank_reduction.ptx")),
            cargo_profile: required_evidence_env("GPT_OSS_H5_CARGO_PROFILE"),
            cuda_arch: required_evidence_env("CUDA_ARCH"),
            model_mapping_sha256: checkpoint.mapping_sha256(),
            placement_sha256: placement.manifest_hash(),
            layer_owner_pci_bus_id: placement.layer_owner().stable_id.pci_bus_id.to_string(),
            remote_worker_pci_bus_id: placement.remote_worker().stable_id.pci_bus_id.to_string(),
            transaction_generation: GENERATION,
            selected_ids,
            expected_result_descriptors: expected_descriptors.len(),
            local_before_clear_rejected: true,
            stale_local_generation_rejected: true,
            stale_reducer_generation_rejected: true,
            premature_rebind_rejected: true,
            drained_abandon_then_higher_generation: true,
            successful_reduction_closed_generation: true,
            reduction_faults_drained,
            construction_faults_drained,
            identity_negatives: vec![
                "missing",
                "duplicate",
                "wrong_expert",
                "wrong_weight",
                "wrong_owner",
                "wrong_slot",
                "wrong_plan",
                "relabelled_local_slot",
                "stale_generation",
            ],
            contribution_arena_bytes: GPT_OSS_REDUCTION_CONTRIBUTION_BYTES,
            reducer_owned_device_bytes: GPT_OSS_REDUCER_OWNED_DEVICE_BYTES,
            measured_driver_free_delta_bytes,
            driver_allocator_reuse_observed,
            reducer_drop_recovered_within_bytes,
            pipeline_device_bytes: GPT_OSS_REDUCTION_DEVICE_WORK_BYTES,
            workspace_class_bytes: GPT_OSS_REDUCTION_WORKSPACE_CLASS_BYTES,
            pinned_high_water: pool_stats.map(|pool| pool.high_water),
            output_sha256: hash_u16(&actual.output_bf16_bits),
            weighted_trace_sha256: hash_u32(&actual.trace.weighted_f32_bits),
            accumulator_trace_sha256: hash_u32(&actual.trace.accumulator_f32_bits),
            reducer_kernel_elapsed_ms: actual.kernel_elapsed_ms,
            passed: true,
        };
        write_json(Path::new(&path), &evidence);
    }

    // Keep the native source lifetime checks explicit and prevent an unused
    // fixture from hiding a wrong CPU expert in this test.
    assert_eq!(cpu_source.key.expert, 21);
}

fn native_expert(
    checkpoint: &GptOssCheckpointView,
    layer: u16,
    expert: u16,
) -> NativeMxfp4ExpertView<'_> {
    let prefix = format!("model.layers.{layer}.mlp.experts");
    let gate_blocks = checkpoint
        .tensor(&format!("{prefix}.gate_up_proj_blocks"))
        .unwrap();
    let gate_scales = checkpoint
        .tensor(&format!("{prefix}.gate_up_proj_scales"))
        .unwrap();
    let gate_bias = checkpoint
        .tensor(&format!("{prefix}.gate_up_proj_bias"))
        .unwrap();
    let down_blocks = checkpoint
        .tensor(&format!("{prefix}.down_proj_blocks"))
        .unwrap();
    let down_scales = checkpoint
        .tensor(&format!("{prefix}.down_proj_scales"))
        .unwrap();
    let down_bias = checkpoint
        .tensor(&format!("{prefix}.down_proj_bias"))
        .unwrap();
    let expert_index = usize::from(expert);
    NativeMxfp4ExpertView {
        key: GptOssExpertKey { layer, expert },
        gate_up_blocks: expert_slice(gate_blocks.bytes(), expert_index, GATE_UP_BLOCK_BYTES),
        gate_up_scales: expert_slice(gate_scales.bytes(), expert_index, GATE_UP_SCALE_BYTES),
        gate_up_bias_bf16_bits: expert_slice(
            bytemuck::try_cast_slice(gate_bias.bytes()).unwrap(),
            expert_index,
            GATE_UP_BIAS_VALUES,
        ),
        down_blocks: expert_slice(down_blocks.bytes(), expert_index, DOWN_BLOCK_BYTES),
        down_scales: expert_slice(down_scales.bytes(), expert_index, DOWN_SCALE_BYTES),
        down_bias_bf16_bits: expert_slice(
            bytemuck::try_cast_slice(down_bias.bytes()).unwrap(),
            expert_index,
            DOWN_BIAS_VALUES,
        ),
        identity_sha256: IDENTITY,
    }
}

fn expect_bound_upload_rejection(
    relay: &mut CudaResultRelay,
    plan: &PackedDispatchPlan,
    reservation: RelayPinnedReservation,
    completions: &[CanonicalRouteContract],
) -> RelayPinnedReservation {
    let failure = relay
        .upload_results_bound(plan, reservation, completions, None)
        .unwrap_err();
    assert!(failure.error.to_string().contains("identity"));
    failure
        .reservation
        .expect("pre-enqueue identity rejection must return the pinned reservation")
}

fn expert_slice<T>(values: &[T], expert: usize, stride: usize) -> &[T] {
    &values[expert * stride..(expert + 1) * stride]
}

fn hash_u16(values: &[u16]) -> String {
    hash_bytes(bytemuck::cast_slice(values))
}

fn hash_u32(values: &[u32]) -> String {
    hash_bytes(bytemuck::cast_slice(values))
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn write_json(path: &Path, value: &impl Serialize) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut bytes = serde_json::to_vec_pretty(value).unwrap();
    bytes.push(b'\n');
    std::fs::write(path, bytes).unwrap();
}

fn required_evidence_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required when writing H5 evidence"))
}

fn hash_file(path: &Path) -> String {
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

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
