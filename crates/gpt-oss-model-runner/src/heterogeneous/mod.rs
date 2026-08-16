//! Narrow contracts for exact static heterogeneous GPT-OSS expert execution.
//!
//! H1 intentionally defines data and validation only. No existing model path
//! constructs these types or changes execution behavior yet.

pub mod contract;
#[cfg(feature = "cuda")]
pub mod cpu_expert;
#[cfg(feature = "cuda")]
pub mod cuda_expert;
#[cfg(feature = "cuda")]
pub mod layer;
pub mod packing;
pub mod placement;
#[cfg(feature = "cuda")]
pub mod reduction;
#[cfg(feature = "cuda")]
pub mod relay;
#[cfg(feature = "cuda")]
pub mod router;

#[cfg(feature = "cuda")]
pub use cpu_expert::{
    CpuX8SelectedExpertDeviceExecution, CpuX8SelectedExpertExecution, CpuX8SelectedExpertWorker,
};
#[cfg(feature = "cuda")]
pub use cuda_expert::{
    exact_selected_expert_reference, selected_expert_device_memory_info,
    CudaSelectedExpertExecutor, CudaSelectedExpertResultSlot, CudaSelectedExpertWeights,
    NativeMxfp4ExpertView, PendingSelectedExpert, PreparedSelectedExpert, SelectedExpertCapture,
    SelectedExpertDeviceExecution, SelectedExpertExecution, SelectedExpertFirstDivergenceTrace,
    SelectedExpertPinnedExecution, DOWN_BIAS_VALUES, DOWN_BLOCK_BYTES, DOWN_SCALE_BYTES,
    GATE_UP_BIAS_VALUES, GATE_UP_BLOCK_BYTES, GATE_UP_SCALE_BYTES,
    GPT_OSS_SELECTED_EXPERT_DEVICE_WORK_BYTES, GPT_OSS_SELECTED_EXPERT_INPUT_BYTES,
    GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES, GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES,
    GPT_OSS_SELECTED_EXPERT_SCRATCH_BYTES, GPT_OSS_SELECTED_EXPERT_TRACE_BYTES,
    GPT_OSS_SELECTED_EXPERT_WORKSPACE_POOL_CLASS_BYTES, HIDDEN_SIZE, INPUT_BLOCKS,
    INTERMEDIATE_SIZE,
};
#[cfg(all(feature = "cuda", feature = "heterogeneous-test-faults"))]
pub use layer::LayerOwnerInjectedFault;
#[cfg(feature = "cuda")]
pub use layer::{
    CudaLayerOwnerShell, LayerOwnerShellExecution, GPT_OSS_LAYER_OWNER_HOST_STAGING_BYTES,
    GPT_OSS_LAYER_OWNER_WORK_BYTES,
};
#[cfg(feature = "cuda")]
pub use reduction::{
    exact_rank_ordered_reduction_reference, CanonicalExpertContribution, CudaRankOrderedReducer,
    PreparedRankOrderedReduction, RankOrderedReductionExecution, RankOrderedReductionTrace,
    GPT_OSS_REDUCER_OWNED_DEVICE_BYTES, GPT_OSS_REDUCTION_CONTRIBUTION_BYTES,
    GPT_OSS_REDUCTION_DEVICE_WORK_BYTES, GPT_OSS_REDUCTION_OUTPUT_BYTES,
    GPT_OSS_REDUCTION_TRACE_BYTES, GPT_OSS_REDUCTION_WEIGHT_BYTES,
    GPT_OSS_REDUCTION_WORKSPACE_CLASS_BYTES,
};
#[cfg(all(feature = "cuda", feature = "heterogeneous-test-faults"))]
pub use reduction::{RankReductionConstructionFault, RankReductionInjectedFault};
#[cfg(all(feature = "cuda", feature = "heterogeneous-test-faults"))]
pub use relay::ResultRelayInjectedFault;
#[cfg(feature = "cuda")]
pub use relay::{
    fixed_relay_byte_plan, pack_remote_inputs, CompletedLocalResultRelay, CompletedResultRelay,
    CudaResultRelay, LocalResultRelayFailure, RelayPinnedPoolStats, RelayPinnedPools,
    RelayPinnedReservation, ResultRelayExecution, ResultRelayFailure,
};
#[cfg(all(feature = "cuda", feature = "heterogeneous-test-faults"))]
pub use router::ExactRouterInjectedFault;
#[cfg(feature = "cuda")]
pub use router::{
    exact_router_reference, CudaExactRouter, ExactRouterExecution, ExactRouterReference,
    ExactRouterWeightsView, GPT_OSS_ROUTER_DESCRIPTOR_BYTES_PER_ROW, GPT_OSS_ROUTER_MAX_ROWS,
};

#[cfg(all(feature = "cuda", feature = "heterogeneous-test-faults"))]
pub use cuda_expert::SelectedExpertInjectedFault;

pub use contract::{
    group_routes_stably, sort_errors_by_precedence, CanonicalCudaDevice, CanonicalExpertOwner,
    CanonicalRouteContract, CompletionDescriptor, ContractError, ErrorOwner,
    ExpertRepresentationTag, ExpertResultDescriptor, ExpertWeightDescriptor, GptOssPhase,
    GptOssRouteDescriptor, GptOssRouteWireV1, GptOssRoutedBatchDescriptor, HeterogeneousErrorKind,
    HeterogeneousErrorRecord, PackedRouteDescriptor, PreparedStepDescriptor, PreparedStepState,
    GPT_OSS_ROUTE_WIRE_V1_BYTES,
};
pub use packing::{
    pack_routes_bounded, PackedDispatchPlan, PackedDispatchRoute, PackedOwnerDispatch,
    RelayBytePlan, H4_DECODE_PINNED_CAP_BYTES, H4_PREFILL_MAX_ROWS, H4_PREFILL_PINNED_CAP_BYTES,
    H4_ROUTE_DESCRIPTOR_MAX_BYTES, H4_ROUTE_DESCRIPTOR_TRANSFER_BYTES,
};
pub use placement::{
    CpuPoolId, ExpertOwner, GptOssExpertKey, GptOssExpertPlacementManifestV1, GptOssPlacementModel,
    PlacementBudgets, PlacementError, PlacementPolicyClass, ResolvedExpertPlacement,
    CONSERVATIVE_OWNER_EXPERT_BYTES, HETEROGENEOUS_PLACEMENT_SCHEMA_V1,
};
