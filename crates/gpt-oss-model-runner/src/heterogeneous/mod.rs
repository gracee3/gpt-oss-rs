//! Narrow contracts for exact static heterogeneous GPT-OSS expert execution.
//!
//! H1 intentionally defines data and validation only. No existing model path
//! constructs these types or changes execution behavior yet.

pub mod contract;
#[cfg(feature = "cuda")]
pub mod cuda_expert;
pub mod placement;

#[cfg(feature = "cuda")]
pub use cuda_expert::{
    exact_selected_expert_reference, selected_expert_device_memory_info,
    CudaSelectedExpertExecutor, CudaSelectedExpertResultSlot, CudaSelectedExpertWeights,
    NativeMxfp4ExpertView, PendingSelectedExpert, PreparedSelectedExpert, SelectedExpertCapture,
    SelectedExpertExecution, SelectedExpertFirstDivergenceTrace, DOWN_BIAS_VALUES,
    DOWN_BLOCK_BYTES, DOWN_SCALE_BYTES, GATE_UP_BIAS_VALUES, GATE_UP_BLOCK_BYTES,
    GATE_UP_SCALE_BYTES, GPT_OSS_SELECTED_EXPERT_DEVICE_WORK_BYTES,
    GPT_OSS_SELECTED_EXPERT_INPUT_BYTES, GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES,
    GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES, GPT_OSS_SELECTED_EXPERT_SCRATCH_BYTES,
    GPT_OSS_SELECTED_EXPERT_TRACE_BYTES, GPT_OSS_SELECTED_EXPERT_WORKSPACE_POOL_CLASS_BYTES,
    HIDDEN_SIZE, INPUT_BLOCKS, INTERMEDIATE_SIZE,
};

#[cfg(all(feature = "cuda", feature = "heterogeneous-test-faults"))]
pub use cuda_expert::SelectedExpertInjectedFault;

pub use contract::{
    group_routes_stably, sort_errors_by_precedence, CompletionDescriptor, ContractError,
    ErrorOwner, ExpertRepresentationTag, ExpertResultDescriptor, ExpertWeightDescriptor,
    GptOssPhase, GptOssRouteDescriptor, GptOssRoutedBatchDescriptor, HeterogeneousErrorKind,
    HeterogeneousErrorRecord, PackedRouteDescriptor, PreparedStepDescriptor, PreparedStepState,
};
pub use placement::{
    CpuPoolId, ExpertOwner, GptOssExpertKey, GptOssExpertPlacementManifestV1, GptOssPlacementModel,
    PlacementBudgets, PlacementError, PlacementPolicyClass, ResolvedExpertPlacement,
    CONSERVATIVE_OWNER_EXPERT_BYTES, HETEROGENEOUS_PLACEMENT_SCHEMA_V1,
};
