//! Exact rank-bearing routed-expert data contracts.

use half::bf16;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use gpt_oss_gpu::device::StableCudaDeviceId;

use super::placement::{ExpertOwner, GptOssExpertKey, ResolvedExpertPlacement};

pub const GPT_OSS_HIDDEN_SIZE: usize = 2_880;
pub const GPT_OSS_TOP_K: usize = 4;
pub const GPT_OSS_ROUTE_WIRE_V1_BYTES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GptOssPhase {
    Prefill,
    Decode,
}

/// Canonical route identity. The selected weight remains in BF16 bit form so
/// packing cannot silently widen and reround it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GptOssRouteDescriptor {
    pub source_row: u32,
    pub route_rank: u8,
    pub expert_id: u16,
    pub weight_bf16_bits: u16,
    pub activation_slot: u32,
}

impl GptOssRouteDescriptor {
    pub fn new(
        source_row: u32,
        route_rank: u8,
        expert_id: u16,
        weight: f32,
        activation_slot: u32,
    ) -> Self {
        Self {
            source_row,
            route_rank,
            expert_id,
            weight_bf16_bits: bf16::from_f32(weight).to_bits(),
            activation_slot,
        }
    }

    pub fn weight(self) -> f32 {
        bf16::from_bits(self.weight_bf16_bits).to_f32()
    }

    pub fn canonical_result_slot(self) -> u32 {
        self.source_row * GPT_OSS_TOP_K as u32 + u32::from(self.route_rank)
    }
}

/// GPU-authored canonical route record transferred from the layer owner.
///
/// The explicit reserved bytes make this a stable 16-byte wire shape. A host
/// consumer validates, but never reconstructs, row/rank/activation identity.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GptOssRouteWireV1 {
    pub source_row: u32,
    pub activation_slot: u32,
    pub expert_id: u16,
    pub weight_bf16_bits: u16,
    pub route_rank: u8,
    pub reserved: [u8; 3],
}

impl GptOssRouteWireV1 {
    pub fn into_descriptor(self) -> Result<GptOssRouteDescriptor, ContractError> {
        if self.reserved != [0; 3] {
            return Err(ContractError::RouteWireReserved(self.reserved));
        }
        Ok(GptOssRouteDescriptor {
            source_row: self.source_row,
            route_rank: self.route_rank,
            expert_id: self.expert_id,
            weight_bf16_bits: self.weight_bf16_bits,
            activation_slot: self.activation_slot,
        })
    }
}

const _: () = assert!(size_of::<GptOssRouteWireV1>() == GPT_OSS_ROUTE_WIRE_V1_BYTES);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GptOssRoutedBatchDescriptor {
    pub layer: u16,
    pub phase: GptOssPhase,
    pub rows: u32,
    pub hidden_size: u16,
    pub experts_per_layer: u16,
    pub placement_epoch: u64,
    pub activation_bf16_bits: Vec<u16>,
    pub routes: Vec<GptOssRouteDescriptor>,
}

impl GptOssRoutedBatchDescriptor {
    pub fn validate(&self) -> Result<(), ContractError> {
        if usize::from(self.hidden_size) != GPT_OSS_HIDDEN_SIZE
            || !matches!(self.experts_per_layer, 32 | 128)
        {
            return Err(ContractError::UnsupportedDimensions {
                hidden_size: self.hidden_size,
                experts_per_layer: self.experts_per_layer,
            });
        }
        let activation_len = self.rows as usize * GPT_OSS_HIDDEN_SIZE;
        if self.activation_bf16_bits.len() != activation_len {
            return Err(ContractError::ActivationLength {
                expected: activation_len,
                observed: self.activation_bf16_bits.len(),
            });
        }
        let expected_routes = self.rows as usize * GPT_OSS_TOP_K;
        if self.routes.len() != expected_routes {
            return Err(ContractError::RouteCount {
                expected: expected_routes,
                observed: self.routes.len(),
            });
        }
        for (slot, route) in self.routes.iter().copied().enumerate() {
            let expected_row = (slot / GPT_OSS_TOP_K) as u32;
            let expected_rank = (slot % GPT_OSS_TOP_K) as u8;
            if route.source_row != expected_row || route.route_rank != expected_rank {
                return Err(ContractError::CanonicalOrder {
                    slot: slot as u32,
                    source_row: route.source_row,
                    route_rank: route.route_rank,
                });
            }
            if route.activation_slot >= self.rows {
                return Err(ContractError::ActivationSlot(route.activation_slot));
            }
            if route.expert_id >= self.experts_per_layer {
                return Err(ContractError::ExpertOutOfRange(route.expert_id));
            }
            if !route.weight().is_finite() {
                return Err(ContractError::NonFiniteWeight { slot: slot as u32 });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedRouteDescriptor {
    pub route: GptOssRouteDescriptor,
    pub owner: ExpertOwner,
    pub placement_epoch: u64,
    pub canonical_result_slot: u32,
    pub source_activation_slot: u32,
}

/// Stable-group canonical routes by owner and expert while carrying their
/// original result slot. No caller may reconstruct rank after this operation.
pub fn group_routes_stably(
    batch: &GptOssRoutedBatchDescriptor,
    placement: &ResolvedExpertPlacement,
) -> Result<Vec<PackedRouteDescriptor>, ContractError> {
    batch.validate()?;
    if batch.placement_epoch != placement.placement_epoch() {
        return Err(ContractError::PlacementEpoch {
            expected: placement.placement_epoch(),
            observed: batch.placement_epoch,
        });
    }
    let mut packed = Vec::with_capacity(batch.routes.len());
    for route in &batch.routes {
        let key = GptOssExpertKey {
            layer: batch.layer,
            expert: route.expert_id,
        };
        let owner = placement
            .owner(key)
            .ok_or(ContractError::MissingOwner {
                layer: key.layer,
                expert: key.expert,
            })?
            .clone();
        packed.push(PackedRouteDescriptor {
            route: *route,
            owner,
            placement_epoch: batch.placement_epoch,
            canonical_result_slot: route.canonical_result_slot(),
            source_activation_slot: route.activation_slot,
        });
    }
    packed.sort_by(|left, right| {
        (&left.owner, left.route.expert_id).cmp(&(&right.owner, right.route.expert_id))
    });
    Ok(packed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertRepresentationTag {
    CpuMxfp4InterleavedX8V2,
    CudaNativeMxfp4BlocksScalesV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertWeightDescriptor {
    pub key: GptOssExpertKey,
    pub owner: ExpertOwner,
    pub representation: ExpertRepresentationTag,
    pub payload_bytes: u64,
    pub identity_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertResultDescriptor {
    pub source_row: u32,
    pub route_rank: u8,
    pub expert_id: u16,
    pub weight_bf16_bits: u16,
    pub owner: ExpertOwner,
    pub placement_epoch: u64,
    pub result_slot: u32,
}

/// Allocation-free identity for one canonical route/result slot. Stable GPU
/// ownership includes every admission field; the device name is represented
/// by a fixed hash so the contract remains `Copy` and bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalRouteContract {
    pub source_row: u32,
    pub activation_slot: u32,
    pub source_activation_slot: u32,
    pub route_rank: u8,
    pub expert_id: u16,
    pub weight_bf16_bits: u16,
    pub owner: CanonicalExpertOwner,
    pub placement_epoch: u64,
    pub result_slot: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalExpertOwner {
    Cpu { pool: u16 },
    LayerOwnerGpu { device: CanonicalCudaDevice },
    RemoteGpu { device: CanonicalCudaDevice },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalCudaDevice {
    pub pci_bus_id: gpt_oss_gpu::device::PciBusId,
    pub expected_name_sha256: [u8; 32],
    pub compute_capability: (u32, u32),
    pub minimum_memory: u64,
}

impl CanonicalRouteContract {
    pub fn from_packed_route(route: &PackedRouteDescriptor) -> Self {
        Self {
            source_row: route.route.source_row,
            activation_slot: route.route.activation_slot,
            source_activation_slot: route.source_activation_slot,
            route_rank: route.route.route_rank,
            expert_id: route.route.expert_id,
            weight_bf16_bits: route.route.weight_bf16_bits,
            owner: CanonicalExpertOwner::from_owner(&route.owner),
            placement_epoch: route.placement_epoch,
            result_slot: route.canonical_result_slot,
        }
    }

    pub fn validate_result(&self, result: &ExpertResultDescriptor) -> Result<(), ContractError> {
        if self.source_row == result.source_row
            && self.route_rank == result.route_rank
            && self.expert_id == result.expert_id
            && self.weight_bf16_bits == result.weight_bf16_bits
            && self.owner == CanonicalExpertOwner::from_owner(&result.owner)
            && self.placement_epoch == result.placement_epoch
            && self.result_slot == result.result_slot
        {
            Ok(())
        } else {
            Err(ContractError::ResultIdentity {
                result_slot: result.result_slot,
            })
        }
    }
}

impl CanonicalExpertOwner {
    fn from_owner(owner: &ExpertOwner) -> Self {
        match owner {
            ExpertOwner::Cpu { pool } => Self::Cpu { pool: pool.0 },
            ExpertOwner::LayerOwnerGpu { device } => Self::LayerOwnerGpu {
                device: CanonicalCudaDevice::from_stable(device),
            },
            ExpertOwner::RemoteGpu { device } => Self::RemoteGpu {
                device: CanonicalCudaDevice::from_stable(device),
            },
        }
    }
}

impl CanonicalCudaDevice {
    fn from_stable(device: &gpt_oss_gpu::device::StableCudaDeviceId) -> Self {
        let expected_name_sha256: [u8; 32] = Sha256::digest(device.expected_name.as_bytes()).into();
        Self {
            pci_bus_id: device.pci_bus_id,
            expected_name_sha256,
            compute_capability: device.compute_capability,
            minimum_memory: device.minimum_memory,
        }
    }
}

impl ExpertResultDescriptor {
    pub fn from_packed_route(route: &PackedRouteDescriptor) -> Self {
        Self {
            source_row: route.route.source_row,
            route_rank: route.route.route_rank,
            expert_id: route.route.expert_id,
            weight_bf16_bits: route.route.weight_bf16_bits,
            owner: route.owner.clone(),
            placement_epoch: route.placement_epoch,
            result_slot: route.canonical_result_slot,
        }
    }

    pub fn validate_against(&self, route: &PackedRouteDescriptor) -> Result<(), ContractError> {
        if self == &Self::from_packed_route(route) {
            Ok(())
        } else {
            Err(ContractError::ResultIdentity {
                result_slot: self.result_slot,
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CompletionDescriptor {
    CpuJoin {
        worker: u16,
        generation: u64,
    },
    CudaEvent {
        device: StableCudaDeviceId,
        stream_role: String,
        generation: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreparedStepState {
    Reserved,
    Prepared,
    Dispatched,
    PartiallyComplete,
    Draining,
    Reduced,
    ReadyToCommit,
    Invalidated,
    Committed,
    Discarded,
}

impl PreparedStepState {
    pub const fn allows(self, next: Self) -> bool {
        use PreparedStepState as S;
        matches!(
            (self, next),
            (S::Reserved, S::Prepared | S::Discarded)
                | (S::Prepared, S::Dispatched | S::Discarded)
                | (S::Dispatched, S::PartiallyComplete | S::Draining)
                | (
                    S::PartiallyComplete,
                    S::PartiallyComplete | S::Reduced | S::Draining
                )
                | (S::Reduced, S::ReadyToCommit | S::Draining)
                | (S::ReadyToCommit, S::Committed | S::Draining)
                | (S::Draining, S::Invalidated)
                | (S::Invalidated, S::Discarded)
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedStepDescriptor {
    pub step_id: String,
    pub sequence_id: u64,
    pub expected_revision: u64,
    pub expected_visibility_epoch: u64,
    pub placement_epoch: u64,
    pub generation: u64,
    pub state: PreparedStepState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeterogeneousErrorKind {
    Manifest,
    StableDevice,
    Ownership,
    Bounds,
    Route,
    Reservation,
    Queue,
    Cpu,
    CudaLaunch,
    CudaAsync,
    H2d,
    D2h,
    ResultIdentity,
    Reduction,
    StaleRevision,
    Cancelled,
    Publication,
    Drain,
    Cleanup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorOwner {
    LayerOwnerGpu,
    Cpu,
    RemoteGpu,
    Coordinator,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeterogeneousErrorRecord {
    pub kind: HeterogeneousErrorKind,
    pub owner: ErrorOwner,
    pub route_slot: Option<u32>,
    pub message: String,
}

impl HeterogeneousErrorRecord {
    pub fn precedence_key(&self) -> (u8, u8, u8, u8, u32) {
        let (class, stage, detail) = match self.kind {
            HeterogeneousErrorKind::Manifest => (0, 0, 0),
            HeterogeneousErrorKind::StableDevice => (0, 1, 0),
            HeterogeneousErrorKind::Ownership => (0, 2, 0),
            HeterogeneousErrorKind::Bounds => (0, 3, 0),
            HeterogeneousErrorKind::Route => (0, 4, 0),
            HeterogeneousErrorKind::ResultIdentity => (0, 5, 0),
            HeterogeneousErrorKind::Reservation => (1, 0, 0),
            HeterogeneousErrorKind::Queue => (1, 2, 0),
            HeterogeneousErrorKind::H2d => (1, 3, 0),
            HeterogeneousErrorKind::CudaLaunch => (1, 4, 0),
            HeterogeneousErrorKind::Cpu => (1, 4, 1),
            HeterogeneousErrorKind::CudaAsync => (1, 4, 2),
            HeterogeneousErrorKind::D2h => (1, 5, 0),
            HeterogeneousErrorKind::Reduction => (1, 6, 0),
            HeterogeneousErrorKind::StaleRevision => (1, 7, 0),
            HeterogeneousErrorKind::Publication => (1, 7, 1),
            HeterogeneousErrorKind::Cancelled => (2, 0, 0),
            HeterogeneousErrorKind::Drain => (3, 0, 0),
            HeterogeneousErrorKind::Cleanup => (3, 1, 0),
        };
        let owner = match self.owner {
            ErrorOwner::LayerOwnerGpu => 0,
            ErrorOwner::Cpu => 1,
            ErrorOwner::RemoteGpu => 2,
            ErrorOwner::Coordinator => 3,
        };
        (
            class,
            stage,
            owner,
            detail,
            self.route_slot.unwrap_or(u32::MAX),
        )
    }
}

pub fn sort_errors_by_precedence(errors: &mut [HeterogeneousErrorRecord]) {
    errors.sort_by_key(HeterogeneousErrorRecord::precedence_key);
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ContractError {
    #[error(
        "unsupported routed-batch dimensions hidden={hidden_size}, experts={experts_per_layer}"
    )]
    UnsupportedDimensions {
        hidden_size: u16,
        experts_per_layer: u16,
    },
    #[error("activation contains {observed} BF16 values, expected {expected}")]
    ActivationLength { expected: usize, observed: usize },
    #[error("route arena contains {observed} records, expected {expected}")]
    RouteCount { expected: usize, observed: usize },
    #[error(
        "route slot {slot} is not canonical row/rank order (row={source_row}, rank={route_rank})"
    )]
    CanonicalOrder {
        slot: u32,
        source_row: u32,
        route_rank: u8,
    },
    #[error("activation slot {0} is outside the routed batch")]
    ActivationSlot(u32),
    #[error("expert {0} is outside the routed layer")]
    ExpertOutOfRange(u16),
    #[error("route slot {slot} has a non-finite BF16 selected weight")]
    NonFiniteWeight { slot: u32 },
    #[error("GPU-authored route wire has nonzero reserved bytes {0:?}")]
    RouteWireReserved([u8; 3]),
    #[error("placement epoch mismatch: expected {expected}, observed {observed}")]
    PlacementEpoch { expected: u64, observed: u64 },
    #[error("no owner for layer {layer} expert {expert}")]
    MissingOwner { layer: u16, expert: u16 },
    #[error("returned expert result does not match route identity for slot {result_slot}")]
    ResultIdentity { result_slot: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heterogeneous::placement::{
        CpuPoolId, ExpertAssignment, GptOssExpertPlacementManifestV1, GptOssPlacementModel,
        PlacementBudgets, PlacementPolicyClass, HETEROGENEOUS_PLACEMENT_SCHEMA_V1,
    };
    use gpt_oss_gpu::device::{GpuDevice, StableCudaDeviceId};

    fn stable(pci: &str) -> StableCudaDeviceId {
        StableCudaDeviceId {
            pci_bus_id: pci.parse().unwrap(),
            expected_name: "NVIDIA GeForce RTX 3090".into(),
            compute_capability: (8, 6),
            minimum_memory: 24 * 1024 * 1024 * 1024,
        }
    }

    fn placement() -> ResolvedExpertPlacement {
        let layer_owner = stable("0000:19:00.0");
        let remote_worker = stable("0000:65:00.0");
        let devices = [
            GpuDevice {
                id: 0,
                name: layer_owner.expected_name.clone(),
                compute_capability: (8, 6),
                total_memory: 24 * 1024 * 1024 * 1024,
                pci_bus_id: Some(layer_owner.pci_bus_id),
            },
            GpuDevice {
                id: 1,
                name: remote_worker.expected_name.clone(),
                compute_capability: (8, 6),
                total_memory: 24 * 1024 * 1024 * 1024,
                pci_bus_id: Some(remote_worker.pci_bus_id),
            },
        ];
        let assignments = (0..24)
            .flat_map(|layer| {
                let layer_owner = layer_owner.clone();
                let remote_worker = remote_worker.clone();
                (0..32).map(move |expert| ExpertAssignment {
                    key: GptOssExpertKey { layer, expert },
                    owner: match expert % 3 {
                        0 => ExpertOwner::Cpu { pool: CpuPoolId(0) },
                        1 => ExpertOwner::LayerOwnerGpu {
                            device: layer_owner.clone(),
                        },
                        _ => ExpertOwner::RemoteGpu {
                            device: remote_worker.clone(),
                        },
                    },
                })
            })
            .collect();
        GptOssExpertPlacementManifestV1 {
            schema: HETEROGENEOUS_PLACEMENT_SCHEMA_V1.into(),
            model: GptOssPlacementModel {
                revision: "revision".into(),
                config_sha256: "1".repeat(64),
                index_sha256: "2".repeat(64),
                mapping_sha256: "3".repeat(64),
                num_layers: 24,
                experts_per_layer: 32,
                hidden_size: 2_880,
                intermediate_size: 2_880,
                top_k: 4,
            },
            layer_owner,
            remote_worker,
            policy: PlacementPolicyClass::Proof,
            policy_seed: 0,
            placement_epoch: 9,
            budgets: PlacementBudgets {
                max_cpu_experts: u32::MAX,
                max_layer_owner_experts: u32::MAX,
                max_remote_gpu_experts: u32::MAX,
                max_host_owner_bytes: u64::MAX,
                max_layer_owner_bytes: u64::MAX,
                max_remote_gpu_bytes: u64::MAX,
            },
            assignments,
        }
        .validate(&devices)
        .unwrap()
    }

    fn batch() -> GptOssRoutedBatchDescriptor {
        GptOssRoutedBatchDescriptor {
            layer: 0,
            phase: GptOssPhase::Decode,
            rows: 1,
            hidden_size: 2_880,
            experts_per_layer: 32,
            placement_epoch: 9,
            activation_bf16_bits: vec![bf16::from_f32(0.5).to_bits(); 2_880],
            routes: [31_u16, 21, 22, 6]
                .into_iter()
                .enumerate()
                .map(|(rank, expert_id)| {
                    GptOssRouteDescriptor::new(0, rank as u8, expert_id, 0.25, 0)
                })
                .collect(),
        }
    }

    #[test]
    fn descriptor_preserves_bf16_weight_bits() {
        let route = GptOssRouteDescriptor::new(3, 2, 17, 0.333_251_95, 3);
        assert_eq!(
            bf16::from_f32(route.weight()).to_bits(),
            route.weight_bf16_bits
        );
        let json = serde_json::to_vec(&route).unwrap();
        let decoded: GptOssRouteDescriptor = serde_json::from_slice(&json).unwrap();
        assert_eq!(route, decoded);
    }

    #[test]
    fn route_wire_v1_is_exactly_16_bytes_and_rejects_reserved_bits() {
        assert_eq!(size_of::<GptOssRouteWireV1>(), 16);
        let wire = GptOssRouteWireV1 {
            source_row: 7,
            activation_slot: 7,
            expert_id: 23,
            weight_bf16_bits: bf16::from_f32(0.375).to_bits(),
            route_rank: 2,
            reserved: [0; 3],
        };
        assert_eq!(
            wire.into_descriptor(),
            Ok(GptOssRouteDescriptor {
                source_row: 7,
                activation_slot: 7,
                expert_id: 23,
                weight_bf16_bits: bf16::from_f32(0.375).to_bits(),
                route_rank: 2,
            })
        );
        assert!(GptOssRouteWireV1 {
            reserved: [0, 1, 0],
            ..wire
        }
        .into_descriptor()
        .is_err());
    }

    #[test]
    fn packing_preserves_canonical_slots_and_stable_group_order() {
        let batch = batch();
        let packed = group_routes_stably(&batch, &placement()).unwrap();
        let mut slots = packed
            .iter()
            .map(|route| route.canonical_result_slot)
            .collect::<Vec<_>>();
        slots.sort_unstable();
        assert_eq!(slots, vec![0, 1, 2, 3]);
        for route in &packed {
            assert_eq!(route.canonical_result_slot, route.route.route_rank as u32);
            assert_eq!(route.source_activation_slot, 0);
            assert_eq!(route.placement_epoch, 9);
            assert_eq!(route.route.weight_bf16_bits, bf16::from_f32(0.25).to_bits());
        }
        let cpu = packed
            .iter()
            .filter(|route| matches!(route.owner, ExpertOwner::Cpu { .. }))
            .collect::<Vec<_>>();
        assert_eq!(cpu.len(), 2);
        assert_eq!(cpu[0].route.route_rank, 3);
        assert_eq!(cpu[1].route.route_rank, 1);
    }

    #[test]
    fn grouping_preserves_canonical_order_within_a_repeated_expert() {
        let mut batch = batch();
        batch.rows = 2;
        batch
            .activation_bf16_bits
            .extend(vec![bf16::from_f32(0.75).to_bits(); GPT_OSS_HIDDEN_SIZE]);
        batch
            .routes
            .extend(
                [6_u16, 22, 21, 31]
                    .into_iter()
                    .enumerate()
                    .map(|(rank, expert_id)| {
                        GptOssRouteDescriptor::new(1, rank as u8, expert_id, 0.25, 1)
                    }),
            );
        let packed = group_routes_stably(&batch, &placement()).unwrap();
        let repeated_slots = packed
            .iter()
            .filter(|route| route.route.expert_id == 6)
            .map(|route| route.canonical_result_slot)
            .collect::<Vec<_>>();
        assert_eq!(repeated_slots, vec![3, 4]);
    }

    #[test]
    fn returned_result_identity_includes_weight_owner_epoch_and_rank() {
        let packed = group_routes_stably(&batch(), &placement()).unwrap();
        let route = &packed[0];
        let result = ExpertResultDescriptor::from_packed_route(route);
        result.validate_against(route).unwrap();
        let mut wrong = result;
        wrong.weight_bf16_bits ^= 1;
        assert!(matches!(
            wrong.validate_against(route),
            Err(ContractError::ResultIdentity { .. })
        ));
    }

    #[test]
    fn batch_rejects_rank_reconstruction_or_missing_route() {
        let mut wrong = batch();
        wrong.routes.swap(0, 1);
        assert!(matches!(
            wrong.validate(),
            Err(ContractError::CanonicalOrder { .. })
        ));
        let mut missing = batch();
        missing.routes.pop();
        assert!(matches!(
            missing.validate(),
            Err(ContractError::RouteCount { .. })
        ));
    }

    #[test]
    fn error_precedence_is_independent_of_completion_order() {
        let mut errors = vec![
            HeterogeneousErrorRecord {
                kind: HeterogeneousErrorKind::Cancelled,
                owner: ErrorOwner::Coordinator,
                route_slot: None,
                message: "cancel".into(),
            },
            HeterogeneousErrorRecord {
                kind: HeterogeneousErrorKind::CudaAsync,
                owner: ErrorOwner::RemoteGpu,
                route_slot: Some(2),
                message: "worker".into(),
            },
            HeterogeneousErrorRecord {
                kind: HeterogeneousErrorKind::Route,
                owner: ErrorOwner::Coordinator,
                route_slot: Some(1),
                message: "route".into(),
            },
            HeterogeneousErrorRecord {
                kind: HeterogeneousErrorKind::Cpu,
                owner: ErrorOwner::Cpu,
                route_slot: Some(1),
                message: "cpu".into(),
            },
        ];
        sort_errors_by_precedence(&mut errors);
        assert_eq!(errors[0].message, "route");
        assert_eq!(errors[1].message, "cpu");
        assert_eq!(errors[2].message, "worker");
        assert_eq!(errors[3].message, "cancel");
    }

    #[test]
    fn error_precedence_breaks_same_owner_stage_ties() {
        let mut errors = vec![
            HeterogeneousErrorRecord {
                kind: HeterogeneousErrorKind::CudaAsync,
                owner: ErrorOwner::LayerOwnerGpu,
                route_slot: Some(0),
                message: "async".into(),
            },
            HeterogeneousErrorRecord {
                kind: HeterogeneousErrorKind::CudaLaunch,
                owner: ErrorOwner::LayerOwnerGpu,
                route_slot: Some(0),
                message: "launch".into(),
            },
        ];
        sort_errors_by_precedence(&mut errors);
        assert_eq!(errors[0].message, "launch");
        assert_eq!(errors[1].message, "async");
    }

    #[test]
    fn prepared_state_machine_excludes_publication_shortcuts() {
        assert!(PreparedStepState::Reserved.allows(PreparedStepState::Prepared));
        assert!(PreparedStepState::ReadyToCommit.allows(PreparedStepState::Committed));
        assert!(!PreparedStepState::Dispatched.allows(PreparedStepState::Committed));
        assert!(!PreparedStepState::Draining.allows(PreparedStepState::ReadyToCommit));
    }
}
