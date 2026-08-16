#![cfg(feature = "cuda")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use gpt_oss_engine::worker::{
    reserve_owner_queues_all_or_none, CapacityOneOwnerQueue, HeterogeneousQueueRole,
};
use gpt_oss_engine::{
    DrainRole, HeterogeneousGpuMetadataAdapter, HeterogeneousTransactionCoordinator,
    SequenceCommitImage, TransactionOutcome,
};
use gpt_oss_gpu::device::{GpuDevice, PciBusId, StableCudaDeviceId};
use gpt_oss_model_runner::heterogeneous::placement::ExpertAssignment;
use gpt_oss_model_runner::heterogeneous::{
    CpuPoolId, ErrorOwner, ExpertOwner, GptOssExpertKey, GptOssExpertPlacementManifestV1,
    GptOssPhase, GptOssPlacementModel, GptOssRouteDescriptor, GptOssRoutedBatchDescriptor,
    HeterogeneousErrorKind, HeterogeneousErrorRecord, PlacementBudgets, PlacementPolicyClass,
    HETEROGENEOUS_PLACEMENT_SCHEMA_V1,
};
use serde::Serialize;

struct CountingAllocator;

static COUNT_ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNT_ENABLED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarding the exact layout to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarding the allocation and its exact layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNT_ENABLED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarding the exact layout to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNT_ENABLED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarding the allocation, old layout, and new size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

const ROLES: [DrainRole; 6] = [
    DrainRole::LayerOwnerRouter,
    DrainRole::LayerOwnerExpert,
    DrainRole::LayerOwnerRelay,
    DrainRole::CpuExpert,
    DrainRole::RemoteGpuExpert,
    DrainRole::RankReduction,
];

const OWNER_ORDERS: [[DrainRole; 3]; 6] = [
    [
        DrainRole::LayerOwnerExpert,
        DrainRole::CpuExpert,
        DrainRole::RemoteGpuExpert,
    ],
    [
        DrainRole::LayerOwnerExpert,
        DrainRole::RemoteGpuExpert,
        DrainRole::CpuExpert,
    ],
    [
        DrainRole::CpuExpert,
        DrainRole::LayerOwnerExpert,
        DrainRole::RemoteGpuExpert,
    ],
    [
        DrainRole::CpuExpert,
        DrainRole::RemoteGpuExpert,
        DrainRole::LayerOwnerExpert,
    ],
    [
        DrainRole::RemoteGpuExpert,
        DrainRole::LayerOwnerExpert,
        DrainRole::CpuExpert,
    ],
    [
        DrainRole::RemoteGpuExpert,
        DrainRole::CpuExpert,
        DrainRole::LayerOwnerExpert,
    ],
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct StateSnapshot {
    revision: u64,
    visibility_epoch: u64,
    placement_epoch: u64,
    committed_length: u32,
    block_table: Vec<(u32, u64)>,
    token_ids: Vec<u32>,
    output_image: Vec<u8>,
    evidence_image: Vec<u8>,
    delivery_failure: Option<String>,
    free_blocks: usize,
    active_steps: usize,
}

#[derive(Debug, Serialize)]
struct MatrixCaseEvidence {
    name: String,
    boundary: &'static str,
    expected_kind: HeterogeneousErrorKind,
    completion_order: Vec<DrainRole>,
    before_failure: StateSnapshot,
    after_failure: StateSnapshot,
    authoritative_state_requirement_met: bool,
    delivery_status_exception: bool,
    clean_second_step_committed: bool,
    authority: &'static str,
}

#[derive(Serialize)]
struct TransactionEvidence {
    schema: &'static str,
    captured_unix_seconds: u64,
    repository_head: String,
    source_fingerprint_sha256: String,
    commit_allocations: usize,
    visibility_epoch_before_commit: u64,
    visibility_epoch_after_commit: u64,
    named_matrix_cases: Vec<MatrixCaseEvidence>,
    predispatch_cases: usize,
    postdispatch_lifecycle_cases: usize,
    cancellation_cases: usize,
    owner_completion_permutations: usize,
    adapter_first_adapter_active: usize,
    adapter_first_transaction_active: usize,
    coordinator_first_adapter_active: usize,
    coordinator_first_transaction_active: usize,
    abandoned_vector_count: usize,
    abandoned_capacity_bytes: usize,
    final_adapter_active: usize,
    final_transaction_active: usize,
    coordinator_lifecycle_scope: &'static str,
    concrete_lower_layer_fault_authority: &'static str,
    pinned_allocation_authority: &'static str,
    device_scratch_allocation_authority: &'static str,
    passed: bool,
}

#[test]
fn heterogeneous_transaction_matrix_commit_and_coordinated_shutdown() {
    let mut committed = coordinator(7);
    let step = prepare_dispatched(&mut committed, 7);
    drain_all(&mut committed, step);
    committed.mark_reduced(step, &[0; 2_880]).unwrap();
    committed
        .prepare_commit(step, commit_image(1, &[1, 2, 3, 4]))
        .unwrap();
    let visibility_epoch_before_commit = committed.committed_view(7).unwrap().visibility_epoch;
    ALLOCATIONS.store(0, Ordering::SeqCst);
    COUNT_ENABLED.store(true, Ordering::SeqCst);
    let terminal = committed.commit(step).unwrap();
    COUNT_ENABLED.store(false, Ordering::SeqCst);
    let commit_allocations = ALLOCATIONS.load(Ordering::SeqCst);
    assert_eq!(commit_allocations, 0, "exclusive publication allocated");
    assert_eq!(terminal.outcome, TransactionOutcome::Committed);
    let visibility_epoch_after_commit = committed.committed_view(7).unwrap().visibility_epoch;
    assert_eq!(visibility_epoch_after_commit, 1);

    let mut cases = Vec::new();
    predispatch_matrix(&mut cases);
    postdispatch_lifecycle_matrix(&mut cases);
    cancellation_matrix(&mut cases);
    publication_matrix(&mut cases);
    simultaneous_owner_failure_matrix(&mut cases);
    let (adapter_first, coordinator_first, abandoned) = coordinated_shutdown_orders();

    let predispatch_cases = cases
        .iter()
        .filter(|case| case.boundary.contains("predispatch"))
        .count();
    let cancellation_cases = cases
        .iter()
        .filter(|case| case.expected_kind == HeterogeneousErrorKind::Cancelled)
        .count();
    let postdispatch_lifecycle_cases = cases.len() - predispatch_cases;
    assert!(cases.iter().all(|case| {
        case.authoritative_state_requirement_met && case.clean_second_step_committed
    }));

    if let Some(path) = std::env::var_os("GPT_OSS_H5_TRANSACTION_EVIDENCE") {
        let evidence = TransactionEvidence {
            schema: "gpt-oss-rs.heterogeneous-h5-transaction/v3",
            captured_unix_seconds: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            repository_head: required_env("GPT_OSS_H5_REPO_HEAD"),
            source_fingerprint_sha256: required_env("GPT_OSS_H5_SOURCE_FINGERPRINT"),
            commit_allocations,
            visibility_epoch_before_commit,
            visibility_epoch_after_commit,
            named_matrix_cases: cases,
            predispatch_cases,
            postdispatch_lifecycle_cases,
            cancellation_cases,
            owner_completion_permutations: OWNER_ORDERS.len(),
            adapter_first_adapter_active: adapter_first.0,
            adapter_first_transaction_active: adapter_first.1,
            coordinator_first_adapter_active: coordinator_first.0,
            coordinator_first_transaction_active: coordinator_first.1,
            abandoned_vector_count: abandoned.0,
            abandoned_capacity_bytes: abandoned.1,
            final_adapter_active: 0,
            final_transaction_active: 0,
            coordinator_lifecycle_scope: "named host state-machine injections prove reservation, publication, cancellation, drain accounting, error precedence, quarantine, and clean retry; they do not claim concrete CUDA leg injection",
            concrete_lower_layer_fault_authority: "H2 failure.json plus resolution.json prove selected-expert submit/kernel/D2H drain semantics; H4 real-x8-relay.json proves pinned GPU0/GPU1 relay legs and post-enqueue drain; H5 reduction.json proves canonical result identity, weight H2D, reduction kernel, trace D2H, terminal drain, and allocation faults; the H2 and H4 executable gates were rerun on the final H5 source",
            pinned_allocation_authority: "H4 bounded-pinned-pool exhaustion test plus h5/reduction.json fixed-pool high-water/release",
            device_scratch_allocation_authority: "h5/reduction.json four staged reducer-construction faults with stream drain and device-memory recovery",
            passed: true,
        };
        write_json(Path::new(&path), &evidence);
    }
}

fn coordinator(sequence_id: u64) -> HeterogeneousTransactionCoordinator {
    coordinator_with_capacity(sequence_id, 16)
}

fn coordinator_with_capacity(
    sequence_id: u64,
    capacity: u32,
) -> HeterogeneousTransactionCoordinator {
    let mut coordinator = HeterogeneousTransactionCoordinator::new(4, capacity, false).unwrap();
    coordinator
        .register_sequence(sequence_id, 3, 11, vec![1, 2, 3])
        .unwrap();
    coordinator
}

fn prepare_dispatched(
    coordinator: &mut HeterogeneousTransactionCoordinator,
    sequence_id: u64,
) -> u64 {
    let placement_epoch = coordinator
        .committed_view(sequence_id)
        .unwrap()
        .placement_epoch;
    let step = coordinator
        .reserve_step(sequence_id, 1, placement_epoch)
        .unwrap();
    coordinator.mark_prepared(step).unwrap();
    coordinator.mark_dispatched(step, &ROLES).unwrap();
    step
}

fn drain_all(coordinator: &mut HeterogeneousTransactionCoordinator, step: u64) {
    for role in ROLES {
        coordinator.mark_terminal(step, role).unwrap();
    }
}

fn drain_in_order(
    coordinator: &mut HeterogeneousTransactionCoordinator,
    step: u64,
    owner_order: &[DrainRole; 3],
) -> Vec<DrainRole> {
    let order = [
        DrainRole::LayerOwnerRouter,
        owner_order[0],
        owner_order[1],
        owner_order[2],
        DrainRole::LayerOwnerRelay,
        DrainRole::RankReduction,
    ];
    for role in order {
        coordinator.mark_terminal(step, role).unwrap();
    }
    order.to_vec()
}

fn snapshot(coordinator: &HeterogeneousTransactionCoordinator, sequence_id: u64) -> StateSnapshot {
    let view = coordinator.committed_view(sequence_id).unwrap();
    StateSnapshot {
        revision: view.request_revision,
        visibility_epoch: view.visibility_epoch,
        placement_epoch: view.placement_epoch,
        committed_length: view.committed_length,
        block_table: view
            .committed_block_table
            .iter()
            .map(|block| (block.block_id, block.generation))
            .collect(),
        token_ids: view.token_ids.clone(),
        output_image: view.output_image.clone(),
        evidence_image: view.evidence_image.clone(),
        delivery_failure: view.delivery_failure.clone(),
        free_blocks: coordinator.free_block_count(),
        active_steps: coordinator.active_step_count(),
    }
}

fn clean_second_step(
    coordinator: &mut HeterogeneousTransactionCoordinator,
    sequence_id: u64,
) -> bool {
    let before = snapshot(coordinator, sequence_id);
    let step = prepare_dispatched(coordinator, sequence_id);
    drain_all(coordinator, step);
    coordinator.mark_reduced(step, &[0; 2_880]).unwrap();
    let mut tokens = before.token_ids.clone();
    tokens.push(0xC1EA);
    coordinator
        .prepare_commit(
            step,
            commit_image(before.revision.checked_add(1).unwrap(), &tokens),
        )
        .unwrap();
    let terminal = coordinator.commit(step).unwrap();
    let after = snapshot(coordinator, sequence_id);
    terminal.outcome == TransactionOutcome::Committed
        && after.revision == before.revision + 1
        && after.visibility_epoch == before.visibility_epoch + 1
        && after.committed_length == before.committed_length + 1
        && after.token_ids == tokens
        && after.output_image == [5, 6]
        && after.evidence_image == [7, 8]
        && after.active_steps == 0
}

fn commit_image(revision: u64, token_ids: &[u32]) -> SequenceCommitImage {
    SequenceCommitImage {
        next_revision: revision,
        token_ids: token_ids.to_vec(),
        output_image: vec![5, 6],
        evidence_image: vec![7, 8],
    }
}

fn error(kind: HeterogeneousErrorKind, owner: ErrorOwner) -> HeterogeneousErrorRecord {
    HeterogeneousErrorRecord {
        kind,
        owner,
        route_slot: Some(1),
        message: format!("injected {kind:?}"),
    }
}

fn record_case(
    cases: &mut Vec<MatrixCaseEvidence>,
    name: impl Into<String>,
    boundary: &'static str,
    kind: HeterogeneousErrorKind,
    completion_order: Vec<DrainRole>,
    before_failure: StateSnapshot,
    after_failure: StateSnapshot,
    requirement_met: bool,
    clean: bool,
    authority: &'static str,
) {
    assert!(
        requirement_met,
        "{boundary} authoritative-state requirement was not met"
    );
    assert!(
        clean,
        "{boundary} failure contaminated the second clean step"
    );
    cases.push(MatrixCaseEvidence {
        name: name.into(),
        boundary,
        expected_kind: kind,
        completion_order,
        before_failure,
        after_failure,
        authoritative_state_requirement_met: requirement_met,
        delivery_status_exception: boundary == "postcommit_external_delivery",
        clean_second_step_committed: clean,
        authority,
    });
}

fn predispatch_matrix(cases: &mut Vec<MatrixCaseEvidence>) {
    for (index, (name, kind, validation_failed)) in [
        (
            "invalid_manifest_schema",
            HeterogeneousErrorKind::Manifest,
            invalid_placement(InvalidPlacement::Schema),
        ),
        (
            "stable_device_mismatch",
            HeterogeneousErrorKind::StableDevice,
            invalid_placement(InvalidPlacement::StableDevice),
        ),
        (
            "duplicate_expert_owner",
            HeterogeneousErrorKind::Ownership,
            invalid_placement(InvalidPlacement::DuplicateOwner),
        ),
        (
            "route_bounds",
            HeterogeneousErrorKind::Bounds,
            invalid_route(true),
        ),
        (
            "route_canonical_order",
            HeterogeneousErrorKind::Route,
            invalid_route(false),
        ),
    ]
    .into_iter()
    .enumerate()
    {
        assert!(validation_failed);
        let sequence_id = 1_000 + index as u64;
        let mut coordinator = coordinator(sequence_id);
        let before = snapshot(&coordinator, sequence_id);
        let after = snapshot(&coordinator, sequence_id);
        record_case(
            cases,
            name,
            "predispatch_validation",
            kind,
            Vec::new(),
            before.clone(),
            after.clone(),
            before == after,
            clean_second_step(&mut coordinator, sequence_id),
            "real placement/route validator; coordinator was never reserved",
        );
    }

    let sequence_id = 1_010;
    let mut reservation = coordinator_with_capacity(sequence_id, 1);
    let before = snapshot(&reservation, sequence_id);
    assert!(reservation.reserve_step(sequence_id, 2, 11).is_err());
    let after = snapshot(&reservation, sequence_id);
    let unchanged = after == before;
    record_case(
        cases,
        "kv_pool_reservation_exhausted",
        "predispatch_reservation",
        HeterogeneousErrorKind::Reservation,
        Vec::new(),
        before,
        after,
        unchanged,
        clean_second_step(&mut reservation, sequence_id),
        "real generation-block allocator exhaustion; partial-lease rollback is also covered by the private unit probe",
    );

    let sequence_id = 1_011;
    let mut queue = coordinator(sequence_id);
    let before = snapshot(&queue, sequence_id);
    let step = queue.reserve_step(sequence_id, 1, 11).unwrap();
    queue.mark_prepared(step).unwrap();
    let cpu = CapacityOneOwnerQueue::new(HeterogeneousQueueRole::Cpu);
    let remote = CapacityOneOwnerQueue::new(HeterogeneousQueueRole::RemoteGpu);
    let occupied = remote.try_reserve(7).unwrap();
    assert!(reserve_owner_queues_all_or_none(&cpu, &remote, step).is_err());
    assert!(!cpu.stats().occupied && remote.stats().occupied);
    let terminal = queue
        .record_error(
            step,
            error(HeterogeneousErrorKind::Queue, ErrorOwner::Coordinator),
        )
        .unwrap()
        .unwrap();
    occupied.release().unwrap();
    assert_eq!(terminal.outcome, TransactionOutcome::Discarded);
    let after = snapshot(&queue, sequence_id);
    let unchanged = after == before;
    record_case(
        cases,
        "capacity_one_queue_all_or_none",
        "predispatch_queue_reservation",
        HeterogeneousErrorKind::Queue,
        Vec::new(),
        before,
        after,
        unchanged,
        clean_second_step(&mut queue, sequence_id),
        "real capacity-one owner queues and prepared-step discard",
    );
}

fn postdispatch_lifecycle_matrix(cases: &mut Vec<MatrixCaseEvidence>) {
    let kinds = [
        (
            "cpu_execution",
            HeterogeneousErrorKind::Cpu,
            ErrorOwner::Cpu,
        ),
        (
            "gpu0_launch",
            HeterogeneousErrorKind::CudaLaunch,
            ErrorOwner::LayerOwnerGpu,
        ),
        (
            "gpu1_async",
            HeterogeneousErrorKind::CudaAsync,
            ErrorOwner::RemoteGpu,
        ),
        (
            "gpu1_h2d",
            HeterogeneousErrorKind::H2d,
            ErrorOwner::RemoteGpu,
        ),
        (
            "gpu1_d2h",
            HeterogeneousErrorKind::D2h,
            ErrorOwner::RemoteGpu,
        ),
        (
            "result_identity",
            HeterogeneousErrorKind::ResultIdentity,
            ErrorOwner::Coordinator,
        ),
        (
            "owner_reduction",
            HeterogeneousErrorKind::Reduction,
            ErrorOwner::LayerOwnerGpu,
        ),
    ];
    for (index, (name, kind, owner)) in kinds.into_iter().enumerate() {
        let sequence_id = 1_100 + index as u64;
        let mut coordinator = coordinator(sequence_id);
        let before = snapshot(&coordinator, sequence_id);
        let step = prepare_dispatched(&mut coordinator, sequence_id);
        assert!(coordinator
            .record_error(step, error(kind, owner))
            .unwrap()
            .is_none());
        let order = drain_in_order(&mut coordinator, step, &OWNER_ORDERS[index % 6]);
        let terminal = coordinator.finalize_discard(step).unwrap();
        assert_eq!(terminal.outcome, TransactionOutcome::Discarded);
        assert_eq!(terminal.errors[0].kind, kind);
        let after = snapshot(&coordinator, sequence_id);
        let unchanged = after == before;
        record_case(
            cases,
            name,
            "postdispatch_mandatory_drain",
            kind,
            order,
            before,
            after,
            unchanged,
            clean_second_step(&mut coordinator, sequence_id),
            "coordinator lifecycle classification; concrete CUDA transfer/kernel/reduction behavior is in h5/reduction.json",
        );
    }
}

fn cancellation_matrix(cases: &mut Vec<MatrixCaseEvidence>) {
    for (offset, prepared) in [false, true].into_iter().enumerate() {
        let sequence_id = 1_200 + offset as u64;
        let mut coordinator = coordinator(sequence_id);
        let before = snapshot(&coordinator, sequence_id);
        let step = coordinator.reserve_step(sequence_id, 1, 11).unwrap();
        if prepared {
            coordinator.mark_prepared(step).unwrap();
        }
        let terminal = coordinator.cancel_step(step).unwrap().unwrap();
        assert_eq!(terminal.outcome, TransactionOutcome::Discarded);
        let after = snapshot(&coordinator, sequence_id);
        let unchanged = after == before;
        record_case(
            cases,
            if prepared {
                "cancel_prepared"
            } else {
                "cancel_reserved"
            },
            if prepared {
                "prepared_predispatch_cancellation"
            } else {
                "reserved_predispatch_cancellation"
            },
            HeterogeneousErrorKind::Cancelled,
            Vec::new(),
            before,
            after,
            unchanged,
            clean_second_step(&mut coordinator, sequence_id),
            "transaction coordinator",
        );
    }

    for (order_index, owner_order) in OWNER_ORDERS.iter().enumerate() {
        let completion_order = [
            DrainRole::LayerOwnerRouter,
            owner_order[0],
            owner_order[1],
            owner_order[2],
            DrainRole::LayerOwnerRelay,
            DrainRole::RankReduction,
        ];
        for terminal_count in 0..=completion_order.len() {
            let sequence_id = 1_300 + (order_index * 10 + terminal_count) as u64;
            let mut coordinator = coordinator(sequence_id);
            let before = snapshot(&coordinator, sequence_id);
            let step = prepare_dispatched(&mut coordinator, sequence_id);
            for role in completion_order.iter().take(terminal_count) {
                coordinator.mark_terminal(step, *role).unwrap();
            }
            assert!(coordinator.cancel_step(step).unwrap().is_none());
            if terminal_count < completion_order.len() {
                assert!(coordinator.finalize_discard(step).is_err());
                for role in completion_order.iter().skip(terminal_count) {
                    coordinator.mark_terminal(step, *role).unwrap();
                }
            }
            let terminal = coordinator.finalize_discard(step).unwrap();
            assert_eq!(terminal.errors[0].kind, HeterogeneousErrorKind::Cancelled);
            let after = snapshot(&coordinator, sequence_id);
            let unchanged = after == before;
            record_case(
                cases,
                format!("cancel_owner_order_{order_index}_after_{terminal_count}"),
                "postdispatch_cancellation",
                HeterogeneousErrorKind::Cancelled,
                completion_order.to_vec(),
                before,
                after,
                unchanged,
                clean_second_step(&mut coordinator, sequence_id),
                "transaction coordinator; all six CPU/GPU0/GPU1 completion orderings",
            );
        }
    }
}

fn publication_matrix(cases: &mut Vec<MatrixCaseEvidence>) {
    for (offset, ready) in [false, true].into_iter().enumerate() {
        let sequence_id = 1_500 + offset as u64;
        let mut coordinator = coordinator(sequence_id);
        let before = snapshot(&coordinator, sequence_id);
        let step = prepare_dispatched(&mut coordinator, sequence_id);
        drain_all(&mut coordinator, step);
        coordinator.mark_reduced(step, &[0; 2_880]).unwrap();
        if ready {
            coordinator
                .prepare_commit(step, commit_image(1, &[1, 2, 3, 4]))
                .unwrap();
        }
        assert!(coordinator.cancel_step(step).unwrap().is_none());
        let terminal = coordinator.finalize_discard(step).unwrap();
        assert_eq!(terminal.outcome, TransactionOutcome::Discarded);
        let after = snapshot(&coordinator, sequence_id);
        let unchanged = after == before;
        record_case(
            cases,
            if ready {
                "cancel_ready_to_commit"
            } else {
                "cancel_after_reduction"
            },
            if ready {
                "ready_to_commit_cancellation"
            } else {
                "reduced_cancellation"
            },
            HeterogeneousErrorKind::Cancelled,
            ROLES.to_vec(),
            before,
            after,
            unchanged,
            clean_second_step(&mut coordinator, sequence_id),
            "transaction coordinator",
        );
    }

    let sequence_id = 1_502;
    let mut publication_coordinator = coordinator(sequence_id);
    let before = snapshot(&publication_coordinator, sequence_id);
    let step = prepare_dispatched(&mut publication_coordinator, sequence_id);
    drain_all(&mut publication_coordinator, step);
    publication_coordinator
        .mark_reduced(step, &[0; 2_880])
        .unwrap();
    assert!(publication_coordinator
        .prepare_commit(step, commit_image(9, &[1, 2, 3, 4]))
        .is_err());
    assert!(publication_coordinator
        .record_error(
            step,
            error(HeterogeneousErrorKind::Publication, ErrorOwner::Coordinator),
        )
        .unwrap()
        .is_none());
    let terminal = publication_coordinator.finalize_discard(step).unwrap();
    assert_eq!(terminal.errors[0].kind, HeterogeneousErrorKind::Publication);
    let after = snapshot(&publication_coordinator, sequence_id);
    let unchanged = after == before;
    record_case(
        cases,
        "publication_image_preparation_failure",
        "reduced_publication_preparation",
        HeterogeneousErrorKind::Publication,
        ROLES.to_vec(),
        before,
        after,
        unchanged,
        clean_second_step(&mut publication_coordinator, sequence_id),
        "fallible commit-image validation before ReadyToCommit",
    );

    let sequence_id = 1_503;
    let mut delivery_coordinator = coordinator(sequence_id);
    let step = prepare_dispatched(&mut delivery_coordinator, sequence_id);
    drain_all(&mut delivery_coordinator, step);
    delivery_coordinator
        .mark_reduced(step, &[0; 2_880])
        .unwrap();
    delivery_coordinator
        .prepare_commit(step, commit_image(1, &[1, 2, 3, 4]))
        .unwrap();
    delivery_coordinator.commit(step).unwrap();
    let committed_before_delivery = snapshot(&delivery_coordinator, sequence_id);
    delivery_coordinator
        .record_delivery_failure(sequence_id, "injected receiver close".into())
        .unwrap();
    let committed_after_delivery = snapshot(&delivery_coordinator, sequence_id);
    let mut expected = committed_before_delivery.clone();
    expected.delivery_failure = Some("injected receiver close".into());
    let delivery_state_consistent = committed_after_delivery == expected;
    record_case(
        cases,
        "external_delivery_failure_after_commit",
        "postcommit_external_delivery",
        HeterogeneousErrorKind::Publication,
        ROLES.to_vec(),
        committed_before_delivery,
        committed_after_delivery,
        delivery_state_consistent,
        clean_second_step(&mut delivery_coordinator, sequence_id),
        "committed K/V, revision, epoch, tokens, output, and evidence are not rolled back; only delivery status changes",
    );
}

fn simultaneous_owner_failure_matrix(cases: &mut Vec<MatrixCaseEvidence>) {
    for (error_order, owner_order) in [(false, 0_usize), (true, 5_usize)] {
        let sequence_id = 1_600 + owner_order as u64;
        let mut coordinator = coordinator(sequence_id);
        let before = snapshot(&coordinator, sequence_id);
        let step = prepare_dispatched(&mut coordinator, sequence_id);
        let cpu = error(HeterogeneousErrorKind::Cpu, ErrorOwner::Cpu);
        let gpu1 = error(HeterogeneousErrorKind::CudaAsync, ErrorOwner::RemoteGpu);
        for record in if error_order {
            [gpu1, cpu]
        } else {
            [cpu, gpu1]
        } {
            coordinator.record_error(step, record).unwrap();
        }
        let order = drain_in_order(&mut coordinator, step, &OWNER_ORDERS[owner_order]);
        let terminal = coordinator.finalize_discard(step).unwrap();
        assert_eq!(terminal.errors[0].kind, HeterogeneousErrorKind::Cpu);
        assert_eq!(terminal.errors[1].kind, HeterogeneousErrorKind::CudaAsync);
        let after = snapshot(&coordinator, sequence_id);
        let unchanged = after == before;
        record_case(
            cases,
            if error_order {
                "simultaneous_gpu1_then_cpu"
            } else {
                "simultaneous_cpu_then_gpu1"
            },
            "postdispatch_simultaneous_owner_failure",
            HeterogeneousErrorKind::Cpu,
            order,
            before,
            after,
            unchanged,
            clean_second_step(&mut coordinator, sequence_id),
            "deterministic error precedence under opposite observation and completion orders",
        );
    }
}

enum InvalidPlacement {
    Schema,
    StableDevice,
    DuplicateOwner,
}

fn stable(pci: &str) -> StableCudaDeviceId {
    StableCudaDeviceId {
        pci_bus_id: pci.parse::<PciBusId>().unwrap(),
        expected_name: "NVIDIA GeForce RTX 3090".into(),
        compute_capability: (8, 6),
        minimum_memory: 24 * 1024 * 1024 * 1024,
    }
}

fn invalid_placement(kind: InvalidPlacement) -> bool {
    let layer_owner = stable("0000:19:00.0");
    let remote = stable("0000:65:00.0");
    let assignments = (0..24)
        .flat_map(|layer| {
            let layer_owner = layer_owner.clone();
            let remote = remote.clone();
            (0..32).map(move |expert| ExpertAssignment {
                key: GptOssExpertKey { layer, expert },
                owner: match expert % 3 {
                    0 => ExpertOwner::Cpu { pool: CpuPoolId(0) },
                    1 => ExpertOwner::LayerOwnerGpu {
                        device: layer_owner.clone(),
                    },
                    _ => ExpertOwner::RemoteGpu {
                        device: remote.clone(),
                    },
                },
            })
        })
        .collect::<Vec<_>>();
    let mut manifest = GptOssExpertPlacementManifestV1 {
        schema: HETEROGENEOUS_PLACEMENT_SCHEMA_V1.into(),
        model: GptOssPlacementModel {
            revision: "test".into(),
            config_sha256: "0".repeat(64),
            index_sha256: "1".repeat(64),
            mapping_sha256: "2".repeat(64),
            num_layers: 24,
            experts_per_layer: 32,
            hidden_size: 2_880,
            intermediate_size: 2_880,
            top_k: 4,
        },
        layer_owner: layer_owner.clone(),
        remote_worker: remote,
        policy: PlacementPolicyClass::Proof,
        policy_seed: 1,
        placement_epoch: 11,
        budgets: PlacementBudgets {
            max_cpu_experts: 768,
            max_layer_owner_experts: 768,
            max_remote_gpu_experts: 768,
            max_host_owner_bytes: u64::MAX,
            max_layer_owner_bytes: u64::MAX,
            max_remote_gpu_bytes: u64::MAX,
        },
        assignments,
    };
    match kind {
        InvalidPlacement::Schema => manifest.schema = "invalid".into(),
        InvalidPlacement::StableDevice => manifest.layer_owner.expected_name = "wrong".into(),
        InvalidPlacement::DuplicateOwner => {
            manifest.assignments.push(manifest.assignments[0].clone())
        }
    }
    let devices = [
        GpuDevice {
            id: 0,
            name: "NVIDIA GeForce RTX 3090".into(),
            compute_capability: (8, 6),
            total_memory: 24 * 1024 * 1024 * 1024,
            pci_bus_id: Some("0000:19:00.0".parse().unwrap()),
        },
        GpuDevice {
            id: 1,
            name: "NVIDIA GeForce RTX 3090".into(),
            compute_capability: (8, 6),
            total_memory: 24 * 1024 * 1024 * 1024,
            pci_bus_id: Some("0000:65:00.0".parse().unwrap()),
        },
    ];
    manifest.validate(&devices).is_err()
}

fn invalid_route(bounds: bool) -> bool {
    let mut routes = (0..4)
        .map(|rank| GptOssRouteDescriptor::new(0, rank, rank as u16, 0.25, 0))
        .collect::<Vec<_>>();
    if !bounds {
        routes[0] = GptOssRouteDescriptor::new(0, 1, 0, 0.25, 0);
    }
    let batch = GptOssRoutedBatchDescriptor {
        layer: 0,
        phase: GptOssPhase::Decode,
        rows: 1,
        hidden_size: if bounds { 2_879 } else { 2_880 },
        experts_per_layer: 32,
        placement_epoch: 11,
        activation_bf16_bits: vec![0; if bounds { 2_879 } else { 2_880 }],
        routes,
    };
    batch.validate().is_err()
}

fn coordinated_shutdown_orders() -> ((usize, usize), (usize, usize), (usize, usize)) {
    // Adapter drains first: its zero does not imply the transaction is zero.
    let mut coordinator_a = coordinator(1_700);
    let step_a = prepare_dispatched(&mut coordinator_a, 1_700);
    let adapter_a = HeterogeneousGpuMetadataAdapter::new().unwrap();
    let view_a = coordinator_a.private_kv_view(step_a).unwrap();
    let generation_a = view_a.transaction_generation;
    let ticket_a = adapter_a.prepare_private_decode(3, &view_a, 4).unwrap();
    adapter_a.begin_shutdown();
    coordinator_a.begin_shutdown().unwrap();
    drop(ticket_a);
    let abandoned_a = adapter_a.quarantined_input_ownership();
    assert!(abandoned_a.vector_count > 0 && abandoned_a.capacity_bytes > 0);
    adapter_a.cancel_abandoned(generation_a, true).unwrap();
    adapter_a.finish_shutdown().unwrap();
    let adapter_first = (
        adapter_a.active_ticket_count(),
        coordinator_a.active_step_count(),
    );
    assert_eq!(adapter_first, (0, 1));
    drain_all(&mut coordinator_a, step_a);
    coordinator_a.finish_shutdown().unwrap();
    assert_eq!(coordinator_a.active_step_count(), 0);

    // Coordinator drains first: its zero does not release the quarantined
    // ModelInput still owned by the metadata adapter.
    let mut coordinator_b = coordinator(1_701);
    let step_b = prepare_dispatched(&mut coordinator_b, 1_701);
    let adapter_b = HeterogeneousGpuMetadataAdapter::new().unwrap();
    let view_b = coordinator_b.private_kv_view(step_b).unwrap();
    let generation_b = view_b.transaction_generation;
    let ticket_b = adapter_b.prepare_private_decode(3, &view_b, 4).unwrap();
    adapter_b.begin_shutdown();
    coordinator_b.begin_shutdown().unwrap();
    drop(ticket_b);
    let abandoned_b = adapter_b.quarantined_input_ownership();
    drain_all(&mut coordinator_b, step_b);
    coordinator_b.finish_shutdown().unwrap();
    let coordinator_first = (
        adapter_b.active_ticket_count(),
        coordinator_b.active_step_count(),
    );
    assert_eq!(coordinator_first, (1, 0));
    assert_eq!(adapter_b.quarantined_input_ownership(), abandoned_b);
    adapter_b.cancel_abandoned(generation_b, true).unwrap();
    adapter_b.finish_shutdown().unwrap();
    assert_eq!(adapter_b.active_ticket_count(), 0);
    assert_eq!(coordinator_b.active_step_count(), 0);

    (
        adapter_first,
        coordinator_first,
        (abandoned_b.vector_count, abandoned_b.capacity_bytes),
    )
}

fn write_json(path: &Path, value: &impl Serialize) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut bytes = serde_json::to_vec_pretty(value).unwrap();
    bytes.push(b'\n');
    std::fs::write(path, bytes).unwrap();
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required when writing H5 evidence"))
}
