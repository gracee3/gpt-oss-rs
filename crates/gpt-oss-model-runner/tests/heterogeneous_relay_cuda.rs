#![cfg(all(feature = "cuda", feature = "heterogeneous-test-faults"))]

use std::path::Path;

use gpt_oss_gpu::device::{list_devices, StableCudaDeviceId};
use gpt_oss_gpu::event::{CorrelatedTimeline, TimelinePoint};
use gpt_oss_model_runner::cpu_repack::CpuOwnerRepackCache;
use gpt_oss_model_runner::heterogeneous::{
    exact_selected_expert_reference, pack_remote_inputs, pack_routes_bounded,
    CpuX8SelectedExpertWorker, CudaExactRouter, CudaResultRelay, CudaSelectedExpertExecutor,
    ExactRouterWeightsView, GptOssExpertKey, GptOssExpertPlacementManifestV1, GptOssPhase,
    NativeMxfp4ExpertView, PreparedRankOrderedReduction, RelayPinnedPools,
    ResultRelayInjectedFault, SelectedExpertCapture, DOWN_BIAS_VALUES, DOWN_BLOCK_BYTES,
    DOWN_SCALE_BYTES, GATE_UP_BIAS_VALUES, GATE_UP_BLOCK_BYTES, GATE_UP_SCALE_BYTES, HIDDEN_SIZE,
};
use gpt_oss_model_runner::model_loader::gpt_oss_native::GptOssCheckpointView;
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};

const MODEL: &str = "/data/models/openai/gpt-oss-20b/original";
const PLACEMENT: &str = "docs/het/evidence/implementation-2026-08/h3/placement-20b.json";
const CACHE: &str = "/home/emmy/workspace/gpt-oss-rs-het-cache";
const IDENTITY: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Serialize)]
struct RelayEvidence<'a> {
    schema: &'static str,
    captured_unix_seconds: u64,
    repository_head: String,
    source_fingerprint_sha256: String,
    executable_sha256: String,
    cargo_profile: String,
    cuda_arch: String,
    model_mapping_sha256: &'a str,
    placement_sha256: &'a str,
    layer_owner_pci_bus_id: String,
    remote_worker_pci_bus_id: String,
    cpu_record: String,
    selected_ids: Vec<u16>,
    source_d2h_bytes: usize,
    descriptor_d2h_bytes: usize,
    gpu1_h2d_bytes: usize,
    gpu1_d2h_bytes: usize,
    cpu_result_bytes: usize,
    gpu0_result_h2d_bytes: usize,
    fixed_pinned_bytes: usize,
    fixed_pool_allocations: [usize; 5],
    pool_high_water: [usize; 5],
    pool_exhaustions: [u64; 5],
    pool_quarantined: [u64; 5],
    pool_available_after_release: [usize; 5],
    cpu_x8_scratch_bytes: usize,
    cpu_x8_high_water_jobs: usize,
    result_fault_drained: bool,
    cpu_gpu_overlap: bool,
    transfer_legs: Vec<TransferLegEvidence>,
    points: Vec<TimelinePoint>,
    passed: bool,
}

#[derive(Serialize)]
struct TransferLegEvidence {
    actor: &'static str,
    direction: &'static str,
    begin_label: &'static str,
    end_label: &'static str,
    bytes: usize,
}

#[test]
fn cpu_authority_post_enqueue_fault_drains_without_publication_and_retries() {
    let devices = list_devices();
    assert_eq!(
        devices.len(),
        2,
        "CPU-authority relay fault gate requires both local GPUs"
    );
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
    const GENERATION: u64 = 58;
    let mut reservation = pools.try_reserve_all(GENERATION).unwrap();
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
    let prepared =
        PreparedRankOrderedReduction::prepare(&routed.batch, &placement, GENERATION).unwrap();
    let descriptors = prepared.expected_results().to_vec();
    let outputs = (0..4)
        .map(|rank| vec![bf16::from_f32(rank as f32).to_bits(); HIDDEN_SIZE])
        .collect::<Vec<_>>();
    let mut relay = CudaResultRelay::new(&router, 1).unwrap();
    relay.bind_decode_generation(GENERATION, &plan).unwrap();
    relay
        .inject_next_failure(ResultRelayInjectedFault::CpuAuthorityAfterFirstEnqueue)
        .unwrap();
    assert!(relay
        .upload_cpu_authority_control(GENERATION, &descriptors, outputs.clone())
        .is_err());
    assert!(relay.last_fault_drained());
    assert_eq!(relay.published_arena_generation_for_test(), 0);
    assert!(relay.has_active_generation_for_test());

    assert_eq!(
        relay
            .upload_cpu_authority_control(GENERATION, &descriptors, outputs)
            .unwrap(),
        4 * HIDDEN_SIZE * size_of::<u16>()
    );
    assert_eq!(relay.published_arena_generation_for_test(), GENERATION);
    relay.abandon_decode_generation(GENERATION, true).unwrap();
    assert!(!relay.has_active_generation_for_test());
    relay.bind_decode_generation(GENERATION + 1, &plan).unwrap();
    relay
        .abandon_decode_generation(GENERATION + 1, true)
        .unwrap();
    assert!(!relay.has_active_generation_for_test());
    reservation.release_drained().unwrap();
    let stats = pools.stats();
    for pool in [
        stats.source_activation,
        stats.route_descriptors,
        stats.remote_gpu_input,
        stats.remote_gpu_result,
        stats.cpu_result,
    ] {
        assert_eq!(pool.available, 1);
        assert_eq!(pool.checked_out, 0);
        assert_eq!(pool.quarantined, 0);
    }
}

#[test]
fn real_x8_three_owner_relay_is_bounded_correlated_and_drained() {
    if std::env::var_os("GPT_OSS_RUN_H4_REAL").is_none() {
        eprintln!("GPT_OSS_RUN_H4_REAL is unset; skipping real x8 H4 relay gate");
        return;
    }
    let devices = list_devices();
    assert_eq!(devices.len(), 2, "H4 relay gate requires both local GPUs");
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

    // Exhaust the second pool so try_reserve_all must return the source lease
    // acquired immediately before it. This is an actual pool-exhaustion path,
    // not a synthetic error injected outside the reservation implementation.
    let held_descriptors = pools.hold_route_descriptors_for_test(54).unwrap();
    assert!(pools.try_reserve_all(55).is_err());
    let partial_exhaustion = pools.stats();
    assert_eq!(partial_exhaustion.source_activation.available, 1);
    assert_eq!(partial_exhaustion.source_activation.checked_out, 0);
    assert_eq!(partial_exhaustion.route_descriptors.available, 0);
    assert_eq!(partial_exhaustion.route_descriptors.checked_out, 1);
    for pool in [
        partial_exhaustion.remote_gpu_input,
        partial_exhaustion.remote_gpu_result,
        partial_exhaustion.cpu_result,
    ] {
        assert_eq!(pool.available, 1);
        assert_eq!(pool.checked_out, 0);
    }
    held_descriptors.release_drained().unwrap();

    let mut reservation = pools.try_reserve_all(56).unwrap();
    assert!(pools.try_reserve_all(57).is_err());
    let timeline = CorrelatedTimeline::new();
    let routed = router
        .execute_and_download(
            0,
            GptOssPhase::Decode,
            placement.placement_epoch(),
            1,
            &activation,
            &mut reservation.source_activation,
            &mut reservation.route_descriptors,
            Some(&timeline),
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
    assert_eq!(plan.bytes.source_activation_d2h, 5_760);
    assert_eq!(plan.bytes.route_descriptor_d2h, 64);
    assert_eq!(routed.source_d2h_bytes, plan.bytes.source_activation_d2h);
    assert_eq!(routed.descriptor_d2h_bytes, plan.bytes.route_descriptor_d2h);
    assert_eq!(plan.bytes.remote_gpu_h2d, 5_760);
    assert_eq!(plan.bytes.remote_gpu_d2h, 5_760);
    assert_eq!(plan.bytes.cpu_result_bytes, 5_760);
    assert_eq!(plan.bytes.raw_pinned_bytes, 74_944);
    assert_eq!(plan.local_route_count(), 2);
    assert_eq!(plan.cpu_route_count(), 1);
    assert_eq!(plan.remote_gpu_route_count(), 1);

    pack_remote_inputs(
        &plan,
        &reservation.source_activation,
        &mut reservation.remote_gpu_input,
    )
    .unwrap();
    assert_eq!(
        &reservation.remote_gpu_input.as_slice()[..HIDDEN_SIZE],
        activation
    );

    let local_route = plan
        .local_gpu
        .iter()
        .flat_map(|owner| &owner.routes)
        .find(|route| route.descriptor.route.expert_id == 31)
        .unwrap();
    let cpu_route = &plan.cpu[0].routes[0];
    let remote_route = &plan.remote_gpu[0].routes[0];
    assert_eq!(cpu_route.descriptor.route.expert_id, 21);
    assert_eq!(remote_route.descriptor.route.expert_id, 22);

    let local_source = native_expert(&checkpoint, 0, 31);
    let remote_source = native_expert(&checkpoint, 0, 22);
    let cpu_native_source = native_expert(&checkpoint, 0, 21);
    let cpu_expected = exact_selected_expert_reference(cpu_native_source, &activation).unwrap();

    let mut local_executor = CudaSelectedExpertExecutor::new(layer_owner).unwrap();
    let local_weights = local_executor
        .upload_expert(local_route.descriptor.owner.clone(), local_source)
        .unwrap();
    let mut local_result_slot = local_executor.allocate_result_slot().unwrap();
    let mut remote_executor = CudaSelectedExpertExecutor::new(remote_worker).unwrap();
    let remote_weights = remote_executor
        .upload_expert(remote_route.descriptor.owner.clone(), remote_source)
        .unwrap();
    let mut remote_result_slot = remote_executor.allocate_result_slot().unwrap();

    let local_pending = local_executor
        .prepare(
            GptOssPhase::Decode,
            &local_route.descriptor,
            &local_weights,
            &activation,
            &mut local_result_slot,
        )
        .unwrap()
        .submit_with_timeline(&timeline, "gpu0_local_expert")
        .unwrap();
    let remote_input_start = remote_route.owner_route_slot as usize * HIDDEN_SIZE;
    let remote_pending = remote_executor
        .prepare(
            GptOssPhase::Decode,
            &remote_route.descriptor,
            &remote_weights,
            &reservation.remote_gpu_input.as_slice()
                [remote_input_start..remote_input_start + HIDDEN_SIZE],
            &mut remote_result_slot,
        )
        .unwrap()
        .submit_with_timeline(&timeline, "gpu1_expert")
        .unwrap();

    let mut cpu_worker = CpuX8SelectedExpertWorker::new();
    assert_eq!(cpu_worker.scratch_bytes(), 17_280);
    let cpu_actual = cpu_worker
        .execute_into_pinned(
            0,
            &cpu_route.descriptor,
            cpu_route.owner_route_slot,
            cpu_record.expert_view(21).unwrap(),
            &reservation.source_activation.as_slice()[..HIDDEN_SIZE],
            &mut reservation.cpu_result,
            Some(&timeline),
        )
        .unwrap();
    assert_eq!(cpu_actual.output_bytes, 5_760);
    assert_eq!(cpu_worker.high_water_jobs(), 1);
    let cpu_start = cpu_route.owner_route_slot as usize * HIDDEN_SIZE;
    assert_eq!(
        &reservation.cpu_result.as_slice()[cpu_start..cpu_start + HIDDEN_SIZE],
        cpu_expected.down_bf16_bits
    );

    let local_actual = local_pending
        .drain(SelectedExpertCapture::OutputOnly)
        .unwrap();
    assert_eq!(
        local_actual.output_bf16_bits,
        exact_selected_expert_reference(local_source, &activation)
            .unwrap()
            .down_bf16_bits
    );
    let remote_actual = remote_pending
        .drain_into_pinned(
            &mut reservation.remote_gpu_result,
            Some((&timeline, "gpu1_expert")),
        )
        .unwrap();
    assert_eq!(remote_actual.output_bytes, plan.bytes.remote_gpu_d2h);
    let remote_expected = exact_selected_expert_reference(remote_source, &activation).unwrap();
    assert_eq!(
        &reservation.remote_gpu_result.as_slice()[..HIDDEN_SIZE],
        remote_expected.down_bf16_bits
    );

    let mut relay = CudaResultRelay::new(&router, 1).unwrap();
    relay
        .inject_next_failure(ResultRelayInjectedFault::AfterFirstResultEnqueue)
        .unwrap();
    reservation = relay
        .upload_results(&plan, reservation, None)
        .unwrap_err()
        .reservation
        .expect("injected relay fault drained, so reservation is recoverable");
    assert!(relay.last_fault_drained());
    let completed_relay = relay
        .upload_results(&plan, reservation, Some(&timeline))
        .unwrap();
    let relay_result = completed_relay.execution;
    let reservation = completed_relay.reservation;
    assert_eq!(relay_result.cpu_h2d_bytes, plan.bytes.cpu_result_bytes);
    assert_eq!(relay_result.remote_gpu_h2d_bytes, plan.bytes.remote_gpu_d2h);
    assert_eq!(
        relay_result.cpu_h2d_bytes + relay_result.remote_gpu_h2d_bytes,
        11_520
    );

    let points = timeline.points();
    let source_relay_d2h = interval(&points, "gpu0_relay", "source_d2h_begin", "source_d2h_end");
    let cpu_interval = interval(&points, "cpu_expert", "compute_begin", "compute_end");
    let gpu0_interval = interval(&points, "gpu0_local_expert", "compute_begin", "compute_end");
    let gpu1_interval = interval(&points, "gpu1_expert", "compute_begin", "compute_end");
    let gpu1_input_h2d = interval(&points, "gpu1_expert", "input_h2d_begin", "input_h2d_end");
    assert!(
        gpu1_input_h2d.1 <= gpu1_interval.0,
        "GPU1 input H2D did not complete before compute: {points:?}"
    );
    assert_eq!(
        plan.bytes.remote_gpu_h2d,
        HIDDEN_SIZE * size_of::<u16>(),
        "GPU1 input H2D interval must describe exactly one packed route row"
    );
    let gpu1_result_d2h = interval(&points, "gpu1_expert", "result_d2h_begin", "result_d2h_end");
    let cpu_gpu_overlap =
        overlaps(cpu_interval, gpu0_interval) || overlaps(cpu_interval, gpu1_interval);
    assert!(
        cpu_gpu_overlap,
        "globally correlated CPU interval did not overlap GPU0 or GPU1 compute: {points:?}"
    );
    let relay_interval = interval(&points, "gpu0_relay", "result_h2d_begin", "result_h2d_end");
    assert!(relay_interval.0 <= relay_interval.1);
    let transfer_legs = vec![
        TransferLegEvidence {
            actor: "gpu0_relay",
            direction: "D2H",
            begin_label: "source_d2h_begin",
            end_label: "source_d2h_end",
            bytes: routed.source_d2h_bytes + routed.descriptor_d2h_bytes,
        },
        TransferLegEvidence {
            actor: "gpu1_expert",
            direction: "H2D",
            begin_label: "input_h2d_begin",
            end_label: "input_h2d_end",
            bytes: plan.bytes.remote_gpu_h2d,
        },
        TransferLegEvidence {
            actor: "gpu1_expert",
            direction: "D2H",
            begin_label: "result_d2h_begin",
            end_label: "result_d2h_end",
            bytes: remote_actual.output_bytes,
        },
        TransferLegEvidence {
            actor: "gpu0_relay",
            direction: "H2D",
            begin_label: "result_h2d_begin",
            end_label: "result_h2d_end",
            bytes: relay_result.cpu_h2d_bytes + relay_result.remote_gpu_h2d_bytes,
        },
    ];
    assert!(source_relay_d2h.0 <= source_relay_d2h.1);
    assert!(gpu1_result_d2h.0 <= gpu1_result_d2h.1);

    reservation.release_drained().unwrap();
    let after = pools.stats();
    assert_eq!(after.raw_capacity_bytes, 74_944);
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

    if let Some(path) = std::env::var_os("GPT_OSS_H4_RELAY_EVIDENCE") {
        let evidence = RelayEvidence {
            schema: "gpt-oss-rs.heterogeneous-h4-relay/v1",
            captured_unix_seconds: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            repository_head: required_evidence_env("GPT_OSS_H4_REPO_HEAD"),
            source_fingerprint_sha256: required_evidence_env("GPT_OSS_H4_SOURCE_FINGERPRINT"),
            executable_sha256: hash_file(&std::env::current_exe().unwrap()),
            cargo_profile: required_evidence_env("GPT_OSS_H4_CARGO_PROFILE"),
            cuda_arch: required_evidence_env("CUDA_ARCH"),
            model_mapping_sha256: checkpoint.mapping_sha256(),
            placement_sha256: placement.manifest_hash(),
            layer_owner_pci_bus_id: placement.layer_owner().stable_id.pci_bus_id.to_string(),
            remote_worker_pci_bus_id: placement.remote_worker().stable_id.pci_bus_id.to_string(),
            cpu_record: cpu_record.path().display().to_string(),
            selected_ids,
            source_d2h_bytes: routed.source_d2h_bytes,
            descriptor_d2h_bytes: routed.descriptor_d2h_bytes,
            gpu1_h2d_bytes: plan.bytes.remote_gpu_h2d,
            gpu1_d2h_bytes: remote_actual.output_bytes,
            cpu_result_bytes: cpu_actual.output_bytes,
            gpu0_result_h2d_bytes: relay_result.cpu_h2d_bytes + relay_result.remote_gpu_h2d_bytes,
            fixed_pinned_bytes: plan.bytes.raw_pinned_bytes,
            fixed_pool_allocations: pool_stats.map(|pool| pool.fixed_allocations),
            pool_high_water: pool_stats.map(|pool| pool.high_water),
            pool_exhaustions: pool_stats.map(|pool| pool.exhaustions),
            pool_quarantined: pool_stats.map(|pool| pool.quarantined),
            pool_available_after_release: pool_stats.map(|pool| pool.available),
            cpu_x8_scratch_bytes: cpu_worker.scratch_bytes(),
            cpu_x8_high_water_jobs: cpu_worker.high_water_jobs(),
            result_fault_drained: relay.last_fault_drained(),
            cpu_gpu_overlap,
            transfer_legs,
            points,
            passed: true,
        };
        write_json(Path::new(&path), &evidence);
    }
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
    let expert = usize::from(expert);
    NativeMxfp4ExpertView {
        key: GptOssExpertKey {
            layer,
            expert: expert as u16,
        },
        gate_up_blocks: expert_slice(gate_blocks.bytes(), expert, GATE_UP_BLOCK_BYTES),
        gate_up_scales: expert_slice(gate_scales.bytes(), expert, GATE_UP_SCALE_BYTES),
        gate_up_bias_bf16_bits: expert_slice(
            bytemuck::try_cast_slice(gate_bias.bytes()).unwrap(),
            expert,
            GATE_UP_BIAS_VALUES,
        ),
        down_blocks: expert_slice(down_blocks.bytes(), expert, DOWN_BLOCK_BYTES),
        down_scales: expert_slice(down_scales.bytes(), expert, DOWN_SCALE_BYTES),
        down_bias_bf16_bits: expert_slice(
            bytemuck::try_cast_slice(down_bias.bytes()).unwrap(),
            expert,
            DOWN_BIAS_VALUES,
        ),
        identity_sha256: IDENTITY,
    }
}

fn expert_slice<T>(values: &[T], expert: usize, stride: usize) -> &[T] {
    &values[expert * stride..(expert + 1) * stride]
}

fn interval(points: &[TimelinePoint], actor: &str, begin: &str, end: &str) -> (u64, u64) {
    let starts = points
        .iter()
        .filter(|point| point.actor == actor && point.label == begin)
        .collect::<Vec<_>>();
    let finishes = points
        .iter()
        .filter(|point| point.actor == actor && point.label == end)
        .collect::<Vec<_>>();
    assert_eq!(starts.len(), 1, "expected exactly one {actor}/{begin}");
    assert_eq!(finishes.len(), 1, "expected exactly one {actor}/{end}");
    let start = starts[0].monotonic_ns;
    let finish = finishes[0].monotonic_ns;
    assert!(start <= finish, "reversed interval for {actor}");
    (start, finish)
}

fn overlaps(left: (u64, u64), right: (u64, u64)) -> bool {
    left.0 < right.1 && right.0 < left.1
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
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required when writing H4 evidence"))
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
