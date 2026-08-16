//! Narrow contracts for exact static heterogeneous GPT-OSS expert execution.
//!
//! H1 intentionally defines data and validation only. No existing model path
//! constructs these types or changes execution behavior yet.

pub mod contract;
pub mod placement;

pub use contract::{
    group_routes_stably, sort_errors_by_precedence, CompletionDescriptor, ContractError,
    ErrorOwner, ExpertRepresentationTag, ExpertResultDescriptor, ExpertWeightDescriptor,
    GptOssPhase, GptOssRouteDescriptor, GptOssRoutedBatchDescriptor, HeterogeneousErrorKind,
    HeterogeneousErrorRecord, PackedRouteDescriptor, PreparedStepDescriptor, PreparedStepState,
};
pub use placement::{
    CpuPoolId, ExpertOwner, GptOssExpertKey, GptOssExpertPlacementManifestV1, GptOssPlacementModel,
    PlacementBudgets, PlacementError, PlacementPolicyClass, ResolvedExpertPlacement,
    HETEROGENEOUS_PLACEMENT_SCHEMA_V1,
};
