//! Bounded, rank-preserving route packing by static expert owner.

use std::collections::{BTreeMap, BTreeSet};

use gpt_oss_core::error::{LLMError, Result};
use serde::{Deserialize, Serialize};

use super::contract::{
    group_routes_stably, GptOssPhase, GptOssRoutedBatchDescriptor, PackedRouteDescriptor,
    GPT_OSS_HIDDEN_SIZE, GPT_OSS_TOP_K,
};
use super::placement::{ExpertOwner, ResolvedExpertPlacement};

pub const H4_DECODE_PINNED_CAP_BYTES: usize = 128 * 1024;
pub const H4_PREFILL_PINNED_CAP_BYTES: usize = 8 * 1024 * 1024;
pub const H4_PREFILL_MAX_ROWS: usize = 64;
pub const H4_ROUTE_DESCRIPTOR_MAX_BYTES: usize = 16;
pub const H4_ROUTE_DESCRIPTOR_TRANSFER_BYTES: usize = H4_ROUTE_DESCRIPTOR_MAX_BYTES;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedDispatchRoute {
    pub descriptor: PackedRouteDescriptor,
    /// Canonical row in the one full GPU0-downloaded source activation arena.
    /// This does not compact around local-only rows.
    pub relay_activation_slot: u32,
    /// Stable position within this owner's packed input/result arena.
    pub owner_route_slot: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedOwnerDispatch {
    pub owner: ExpertOwner,
    pub routes: Vec<PackedDispatchRoute>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelayBytePlan {
    /// Actual bytes copied for this route mix.
    pub source_activation_d2h: usize,
    pub route_descriptor_d2h: usize,
    pub remote_gpu_h2d: usize,
    pub remote_gpu_d2h: usize,
    pub cpu_result_bytes: usize,
    /// Fixed, prewarmed reservation capacities. These do not shrink with the
    /// observed owner mix.
    pub source_activation_capacity: usize,
    pub route_descriptor_capacity: usize,
    pub remote_gpu_input_capacity: usize,
    pub remote_gpu_result_capacity: usize,
    pub cpu_result_capacity: usize,
    pub raw_pinned_bytes: usize,
    pub hard_cap_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedDispatchPlan {
    pub layer: u16,
    pub phase: GptOssPhase,
    pub rows: u32,
    pub placement_epoch: u64,
    pub nonlocal_source_rows: Vec<u32>,
    pub local_gpu: Vec<PackedOwnerDispatch>,
    pub cpu: Vec<PackedOwnerDispatch>,
    pub remote_gpu: Vec<PackedOwnerDispatch>,
    pub bytes: RelayBytePlan,
}

impl PackedDispatchPlan {
    pub fn all_routes(&self) -> impl Iterator<Item = &PackedDispatchRoute> {
        self.local_gpu
            .iter()
            .chain(&self.cpu)
            .chain(&self.remote_gpu)
            .flat_map(|owner| owner.routes.iter())
    }

    pub fn local_route_count(&self) -> usize {
        self.local_gpu.iter().map(|owner| owner.routes.len()).sum()
    }

    pub fn cpu_route_count(&self) -> usize {
        self.cpu.iter().map(|owner| owner.routes.len()).sum()
    }

    pub fn remote_gpu_route_count(&self) -> usize {
        self.remote_gpu.iter().map(|owner| owner.routes.len()).sum()
    }

    pub fn validate_round_trip(&self) -> Result<()> {
        let expected = self.rows as usize * GPT_OSS_TOP_K;
        let mut slots = BTreeSet::new();
        for route in self.all_routes() {
            if route.descriptor.placement_epoch != self.placement_epoch
                || route.descriptor.canonical_result_slot
                    != route.descriptor.route.canonical_result_slot()
                || route.descriptor.owner
                    != owner_for_plan_route(self, route).ok_or_else(|| {
                        LLMError::ModelError("packed route is outside every owner bucket".into())
                    })?
            {
                return Err(LLMError::ModelError(
                    "packed route identity changed during compaction".into(),
                ));
            }
            if route.relay_activation_slot != route.descriptor.route.source_row
                || route.relay_activation_slot >= self.rows
            {
                return Err(LLMError::ModelError(
                    "packed route changed its canonical source row".into(),
                ));
            }
            if !slots.insert(route.descriptor.canonical_result_slot) {
                return Err(LLMError::ModelError(format!(
                    "duplicate packed canonical result slot {}",
                    route.descriptor.canonical_result_slot
                )));
            }
        }
        if slots.len() != expected
            || slots
                .iter()
                .copied()
                .ne((0..expected).map(|slot| slot as u32))
        {
            return Err(LLMError::ModelError(
                "packed routes do not cover every canonical row/rank slot".into(),
            ));
        }
        validate_category_slots("layer-owner GPU", &self.local_gpu)?;
        validate_category_slots("CPU", &self.cpu)?;
        validate_category_slots("remote GPU", &self.remote_gpu)?;
        Ok(())
    }
}

fn validate_category_slots(category: &str, owners: &[PackedOwnerDispatch]) -> Result<()> {
    let expected = owners.iter().map(|owner| owner.routes.len()).sum::<usize>();
    let mut slots = BTreeSet::new();
    for route in owners.iter().flat_map(|owner| &owner.routes) {
        if !slots.insert(route.owner_route_slot) {
            return Err(LLMError::ModelError(format!(
                "duplicate {category} packed arena slot {}",
                route.owner_route_slot
            )));
        }
    }
    if slots.len() != expected
        || slots
            .iter()
            .copied()
            .ne((0..expected).map(|slot| slot as u32))
    {
        return Err(LLMError::ModelError(format!(
            "{category} packed arena slots are not contiguous"
        )));
    }
    Ok(())
}

fn owner_for_plan_route(
    plan: &PackedDispatchPlan,
    route: &PackedDispatchRoute,
) -> Option<ExpertOwner> {
    plan.local_gpu
        .iter()
        .chain(&plan.cpu)
        .chain(&plan.remote_gpu)
        .find(|owner| {
            owner.routes.iter().any(|candidate| {
                candidate.descriptor.canonical_result_slot == route.descriptor.canonical_result_slot
            })
        })
        .map(|owner| owner.owner.clone())
}

/// Stable pack by `(owner, expert_id)` while retaining GPU-authored rank,
/// selected BF16 weight bits, and canonical result slot.
pub fn pack_routes_bounded(
    batch: &GptOssRoutedBatchDescriptor,
    placement: &ResolvedExpertPlacement,
) -> Result<PackedDispatchPlan> {
    batch
        .validate()
        .map_err(|error| LLMError::ModelError(format!("route pack input: {error}")))?;
    let rows = batch.rows as usize;
    match batch.phase {
        GptOssPhase::Decode if rows != 1 => {
            return Err(LLMError::ModelError(
                "H4 decode packing supports M=1 only".into(),
            ));
        }
        GptOssPhase::Prefill if rows > H4_PREFILL_MAX_ROWS => {
            return Err(LLMError::ModelError(format!(
                "H4 prefill chunk rows {rows} exceed {H4_PREFILL_MAX_ROWS}"
            )));
        }
        _ => {}
    }
    let packed = group_routes_stably(batch, placement)
        .map_err(|error| LLMError::ModelError(format!("stable route grouping: {error}")))?;
    let nonlocal_source_rows = packed
        .iter()
        .filter(|route| !matches!(route.owner, ExpertOwner::LayerOwnerGpu { .. }))
        .map(|route| route.route.source_row)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut owners = BTreeMap::<ExpertOwner, Vec<PackedRouteDescriptor>>::new();
    for route in packed {
        owners.entry(route.owner.clone()).or_default().push(route);
    }

    let mut local_gpu = Vec::new();
    let mut cpu = Vec::new();
    let mut remote_gpu = Vec::new();
    // Each category has one shared input/result arena. Slots therefore advance
    // across all owner buckets in that category, including multiple CPU pools.
    let mut next_local_slot = 0_u32;
    let mut next_cpu_slot = 0_u32;
    let mut next_remote_slot = 0_u32;
    for (owner, routes) in owners {
        match &owner {
            ExpertOwner::LayerOwnerGpu { .. } => {
                local_gpu.push(pack_owner_routes(owner, routes, &mut next_local_slot))
            }
            ExpertOwner::Cpu { .. } => {
                cpu.push(pack_owner_routes(owner, routes, &mut next_cpu_slot))
            }
            ExpertOwner::RemoteGpu { .. } => {
                remote_gpu.push(pack_owner_routes(owner, routes, &mut next_remote_slot))
            }
        }
    }
    let remote_routes = remote_gpu
        .iter()
        .map(|owner| owner.routes.len())
        .sum::<usize>();
    let cpu_routes = cpu.iter().map(|owner| owner.routes.len()).sum::<usize>();
    let row_bytes = GPT_OSS_HIDDEN_SIZE * size_of::<u16>();
    let source_activation_d2h = rows * row_bytes;
    let route_descriptor_d2h = rows * GPT_OSS_TOP_K * H4_ROUTE_DESCRIPTOR_TRANSFER_BYTES;
    let remote_gpu_h2d = remote_routes * row_bytes;
    let remote_gpu_d2h = remote_routes * row_bytes;
    let cpu_result_bytes = cpu_routes * row_bytes;
    let source_activation_capacity = rows * row_bytes;
    let route_descriptor_capacity = rows * GPT_OSS_TOP_K * H4_ROUTE_DESCRIPTOR_MAX_BYTES;
    let remote_gpu_input_capacity = rows * GPT_OSS_TOP_K * row_bytes;
    let remote_gpu_result_capacity = rows * GPT_OSS_TOP_K * row_bytes;
    let cpu_result_capacity = rows * GPT_OSS_TOP_K * row_bytes;
    let raw_pinned_bytes = source_activation_capacity
        .checked_add(route_descriptor_capacity)
        .and_then(|bytes| bytes.checked_add(remote_gpu_input_capacity))
        .and_then(|bytes| bytes.checked_add(remote_gpu_result_capacity))
        .and_then(|bytes| bytes.checked_add(cpu_result_capacity))
        .ok_or_else(|| LLMError::ModelError("relay byte plan overflows".into()))?;
    let hard_cap_bytes = match batch.phase {
        GptOssPhase::Decode => H4_DECODE_PINNED_CAP_BYTES,
        GptOssPhase::Prefill => H4_PREFILL_PINNED_CAP_BYTES,
    };
    if raw_pinned_bytes > hard_cap_bytes {
        return Err(LLMError::MemoryError(format!(
            "relay pinned requirement {raw_pinned_bytes} exceeds hard cap {hard_cap_bytes}"
        )));
    }
    let plan = PackedDispatchPlan {
        layer: batch.layer,
        phase: batch.phase,
        rows: batch.rows,
        placement_epoch: batch.placement_epoch,
        nonlocal_source_rows,
        local_gpu,
        cpu,
        remote_gpu,
        bytes: RelayBytePlan {
            source_activation_d2h,
            route_descriptor_d2h,
            remote_gpu_h2d,
            remote_gpu_d2h,
            cpu_result_bytes,
            source_activation_capacity,
            route_descriptor_capacity,
            remote_gpu_input_capacity,
            remote_gpu_result_capacity,
            cpu_result_capacity,
            raw_pinned_bytes,
            hard_cap_bytes,
        },
    };
    plan.validate_round_trip()?;
    Ok(plan)
}

fn pack_owner_routes(
    owner: ExpertOwner,
    routes: Vec<PackedRouteDescriptor>,
    next_category_slot: &mut u32,
) -> PackedOwnerDispatch {
    let routes = routes
        .into_iter()
        .map(|descriptor| {
            let owner_route_slot = *next_category_slot;
            *next_category_slot += 1;
            // The router downloads the complete row-major activation arena
            // once. Preserve canonical row slots so a local-only preceding row
            // cannot shift a later nonlocal route.
            PackedDispatchRoute {
                relay_activation_slot: descriptor.route.source_row,
                descriptor,
                owner_route_slot,
            }
        })
        .collect();
    PackedOwnerDispatch { owner, routes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heterogeneous::contract::GptOssRouteDescriptor;
    use crate::heterogeneous::placement::{
        CpuPoolId, ExpertAssignment, GptOssExpertKey, GptOssExpertPlacementManifestV1,
        GptOssPlacementModel, PlacementBudgets, PlacementPolicyClass,
        HETEROGENEOUS_PLACEMENT_SCHEMA_V1,
    };
    use gpt_oss_gpu::device::{GpuDevice, StableCudaDeviceId};
    use half::bf16;

    fn stable(pci: &str) -> StableCudaDeviceId {
        StableCudaDeviceId {
            pci_bus_id: pci.parse().unwrap(),
            expected_name: "NVIDIA GeForce RTX 3090".into(),
            compute_capability: (8, 6),
            minimum_memory: 24 * 1024 * 1024 * 1024,
        }
    }

    fn placement(experts: u16) -> ResolvedExpertPlacement {
        placement_with_cpu_pool_split(experts, false)
    }

    fn placement_with_cpu_pool_split(
        experts: u16,
        split_cpu_pools: bool,
    ) -> ResolvedExpertPlacement {
        let layers = if experts == 32 { 24 } else { 36 };
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
        let assignments = (0..layers)
            .flat_map(|layer| {
                let layer_owner = layer_owner.clone();
                let remote_worker = remote_worker.clone();
                (0..experts).map(move |expert| ExpertAssignment {
                    key: GptOssExpertKey { layer, expert },
                    owner: match expert % 3 {
                        0 => ExpertOwner::LayerOwnerGpu {
                            device: layer_owner.clone(),
                        },
                        1 => ExpertOwner::Cpu {
                            pool: CpuPoolId(if split_cpu_pools { (expert / 3) % 2 } else { 0 }),
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
                revision: "fixture".into(),
                config_sha256: "1".repeat(64),
                index_sha256: "2".repeat(64),
                mapping_sha256: "3".repeat(64),
                num_layers: layers,
                experts_per_layer: experts,
                hidden_size: GPT_OSS_HIDDEN_SIZE as u16,
                intermediate_size: GPT_OSS_HIDDEN_SIZE as u16,
                top_k: GPT_OSS_TOP_K as u8,
            },
            layer_owner,
            remote_worker,
            policy: PlacementPolicyClass::Proof,
            policy_seed: 0,
            placement_epoch: 4,
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

    fn batch(rows: usize, experts: u16, phase: GptOssPhase) -> GptOssRoutedBatchDescriptor {
        let routes = (0..rows)
            .flat_map(|row| {
                (0..GPT_OSS_TOP_K).map(move |rank| {
                    let expert = ((row * 11 + rank * 7) % experts as usize) as u16;
                    GptOssRouteDescriptor {
                        source_row: row as u32,
                        route_rank: rank as u8,
                        expert_id: expert,
                        weight_bf16_bits: bf16::from_f32(0.25).to_bits(),
                        activation_slot: row as u32,
                    }
                })
            })
            .collect();
        GptOssRoutedBatchDescriptor {
            layer: 0,
            phase,
            rows: rows as u32,
            hidden_size: GPT_OSS_HIDDEN_SIZE as u16,
            experts_per_layer: experts,
            placement_epoch: 4,
            activation_bf16_bits: vec![0; rows * GPT_OSS_HIDDEN_SIZE],
            routes,
        }
    }

    #[test]
    fn packing_round_trips_rank_and_weight_for_decode_and_prefill() {
        for (rows, phase) in [(1, GptOssPhase::Decode), (64, GptOssPhase::Prefill)] {
            let batch = batch(rows, 128, phase);
            let plan = pack_routes_bounded(&batch, &placement(128)).unwrap();
            plan.validate_round_trip().unwrap();
            assert_eq!(plan.all_routes().count(), rows * GPT_OSS_TOP_K);
            for route in plan.all_routes() {
                let canonical = &batch.routes[route.descriptor.canonical_result_slot as usize];
                assert_eq!(&route.descriptor.route, canonical);
                assert_eq!(
                    route.relay_activation_slot,
                    route.descriptor.route.source_row
                );
            }
            assert!(plan.bytes.raw_pinned_bytes <= plan.bytes.hard_cap_bytes);
        }
    }

    #[test]
    fn decode_and_prefill_caps_match_phase_plan() {
        let decode =
            pack_routes_bounded(&batch(1, 32, GptOssPhase::Decode), &placement(32)).unwrap();
        assert_eq!(decode.bytes.hard_cap_bytes, H4_DECODE_PINNED_CAP_BYTES);
        assert_eq!(decode.bytes.raw_pinned_bytes, 74_944);
        let prefill =
            pack_routes_bounded(&batch(64, 32, GptOssPhase::Prefill), &placement(32)).unwrap();
        assert_eq!(prefill.bytes.hard_cap_bytes, H4_PREFILL_PINNED_CAP_BYTES);
        assert_eq!(prefill.bytes.raw_pinned_bytes, 4_796_416);
        assert!(pack_routes_bounded(&batch(65, 32, GptOssPhase::Prefill), &placement(32)).is_err());
    }

    #[test]
    fn local_only_preceding_row_does_not_shift_nonlocal_source_slot() {
        let placement = placement(32);
        let mut batch = batch(2, 32, GptOssPhase::Prefill);
        // Expert 0 is local. Expert 1 is CPU and expert 2 is remote GPU.
        for route in &mut batch.routes[..GPT_OSS_TOP_K] {
            route.expert_id = 0;
        }
        for (rank, route) in batch.routes[GPT_OSS_TOP_K..].iter_mut().enumerate() {
            route.expert_id = if rank % 2 == 0 { 1 } else { 2 };
        }
        let plan = pack_routes_bounded(&batch, &placement).unwrap();
        assert_eq!(plan.nonlocal_source_rows, vec![1]);
        assert!(plan
            .cpu
            .iter()
            .chain(&plan.remote_gpu)
            .flat_map(|owner| &owner.routes)
            .all(|route| route.relay_activation_slot == 1));
        assert_eq!(plan.bytes.source_activation_d2h, 2 * 5_760);
        assert_eq!(plan.bytes.source_activation_capacity, 2 * 5_760);
    }

    #[test]
    fn deterministic_route_pattern_sweep_preserves_every_rank_and_weight() {
        let mut state = 0x6a09_e667_f3bc_c909_u64;
        for experts in [32_u16, 128] {
            let resolved = placement(experts);
            for case in 0..96_usize {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let (rows, phase) = if case == 0 {
                    (1, GptOssPhase::Decode)
                } else {
                    (
                        1 + (state as usize % H4_PREFILL_MAX_ROWS),
                        GptOssPhase::Prefill,
                    )
                };
                let mut routed = batch(rows, experts, phase);
                for route in &mut routed.routes {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    route.expert_id = (state % u64::from(experts)) as u16;
                    route.weight_bf16_bits =
                        bf16::from_f32(((state >> 17) & 0xff) as f32 / 255.0).to_bits();
                }
                let plan = pack_routes_bounded(&routed, &resolved).unwrap();
                plan.validate_round_trip().unwrap();
                let mut unpacked = plan
                    .all_routes()
                    .map(|route| route.descriptor.clone())
                    .collect::<Vec<_>>();
                unpacked.sort_by_key(|route| route.canonical_result_slot);
                assert_eq!(unpacked.len(), routed.routes.len());
                for (slot, descriptor) in unpacked.iter().enumerate() {
                    assert_eq!(descriptor.route, routed.routes[slot]);
                    assert_eq!(descriptor.canonical_result_slot, slot as u32);
                    assert_eq!(
                        descriptor.source_activation_slot,
                        descriptor.route.activation_slot
                    );
                    assert_eq!(descriptor.placement_epoch, routed.placement_epoch);
                }
            }
        }
    }

    #[test]
    fn category_global_slots_do_not_collide_across_cpu_owner_groups() {
        let resolved = placement_with_cpu_pool_split(32, true);
        let mut routed = batch(2, 32, GptOssPhase::Prefill);
        for (route, expert) in routed.routes.iter_mut().zip([1_u16, 1, 4, 2, 4, 1, 2, 0]) {
            route.expert_id = expert;
        }
        let plan = pack_routes_bounded(&routed, &resolved).unwrap();
        plan.validate_round_trip().unwrap();
        assert_eq!(plan.cpu.len(), 2, "fixture must cover both CPU pool owners");

        let mut cpu_slots = plan
            .cpu
            .iter()
            .flat_map(|owner| owner.routes.iter().map(|route| route.owner_route_slot))
            .collect::<Vec<_>>();
        cpu_slots.sort_unstable();
        assert_eq!(cpu_slots, [0, 1, 2, 3, 4]);
        let mut remote_slots = plan
            .remote_gpu
            .iter()
            .flat_map(|owner| owner.routes.iter().map(|route| route.owner_route_slot))
            .collect::<Vec<_>>();
        remote_slots.sort_unstable();
        assert_eq!(remote_slots, [0, 1]);
        assert_eq!(plan.local_gpu[0].routes[0].owner_route_slot, 0);

        let mut unpacked = plan
            .all_routes()
            .map(|route| route.descriptor.clone())
            .collect::<Vec<_>>();
        unpacked.sort_by_key(|route| route.canonical_result_slot);
        assert_eq!(
            unpacked.iter().map(|route| route.route).collect::<Vec<_>>(),
            routed.routes
        );
    }
}
