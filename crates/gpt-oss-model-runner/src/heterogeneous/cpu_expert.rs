//! Bounded exact selected-expert execution from H3 owner-filtered x8 records.

use std::time::Instant;

use gpt_oss_core::error::{LLMError, Result};
use gpt_oss_cpu_kernels::{accumulate_mxfp4_bf16_block, Mxfp4MatrixView, QUANT_BLOCK_SIZE};
use gpt_oss_gpu::event::CorrelatedTimeline;
use gpt_oss_gpu::pinned_memory::BoundedPinnedLease;
use half::bf16;

use crate::cpu_repack::CpuOwnerExpertView;

use super::contract::{CanonicalRouteContract, ExpertResultDescriptor, PackedRouteDescriptor};
use super::cuda_expert::SelectedExpertTraceStorage;
use super::placement::ExpertOwner;
use super::{HIDDEN_SIZE, INPUT_BLOCKS, INTERMEDIATE_SIZE};

const SWIGLU_ALPHA: f32 = 1.702;
const SWIGLU_LIMIT: f32 = 7.0;
const GATE_UP_ROWS: usize = INTERMEDIATE_SIZE * 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuX8SelectedExpertExecution {
    pub result: ExpertResultDescriptor,
    pub output_bytes: usize,
    pub elapsed_ns: u64,
}

/// Allocation-free completion used by a prepared heterogeneous step. The
/// step already owns the canonical result descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuX8SelectedExpertDeviceExecution {
    pub route_contract: CanonicalRouteContract,
    pub output_bytes: usize,
    pub elapsed_ns: u64,
}

/// One capacity-one CPU worker. Scratch is allocated once at construction and
/// reused; execute performs no heap or pinned allocation.
pub struct CpuX8SelectedExpertWorker {
    gate_up_bf16_bits: Vec<u16>,
    swiglu_bf16_bits: Vec<u16>,
    high_water_jobs: usize,
}

impl CpuX8SelectedExpertWorker {
    pub fn new() -> Self {
        Self {
            gate_up_bf16_bits: vec![0; GATE_UP_ROWS],
            swiglu_bf16_bits: vec![0; INTERMEDIATE_SIZE],
            high_water_jobs: 0,
        }
    }

    pub const fn scratch_bytes(&self) -> usize {
        (GATE_UP_ROWS + INTERMEDIATE_SIZE) * size_of::<u16>()
    }

    pub const fn high_water_jobs(&self) -> usize {
        self.high_water_jobs
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_into_pinned(
        &mut self,
        layer: u16,
        route: &PackedRouteDescriptor,
        owner_route_slot: u32,
        expert: CpuOwnerExpertView<'_>,
        input_bf16_bits: &[u16],
        output: &mut BoundedPinnedLease<u16>,
        timeline: Option<&CorrelatedTimeline>,
    ) -> Result<CpuX8SelectedExpertExecution> {
        let result = ExpertResultDescriptor::from_packed_route(route);
        let execution = self.execute_into_pinned_device_only(
            layer,
            route,
            owner_route_slot,
            expert,
            input_bf16_bits,
            output,
            timeline,
        )?;
        Ok(CpuX8SelectedExpertExecution {
            result,
            output_bytes: execution.output_bytes,
            elapsed_ns: execution.elapsed_ns,
        })
    }

    /// Execute into a fixed pinned result arena without allocating or cloning
    /// route identity after dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_into_pinned_device_only(
        &mut self,
        layer: u16,
        route: &PackedRouteDescriptor,
        owner_route_slot: u32,
        expert: CpuOwnerExpertView<'_>,
        input_bf16_bits: &[u16],
        output: &mut BoundedPinnedLease<u16>,
        timeline: Option<&CorrelatedTimeline>,
    ) -> Result<CpuX8SelectedExpertDeviceExecution> {
        self.execute_into_pinned_inner(
            layer,
            route,
            owner_route_slot,
            expert,
            input_bf16_bits,
            output,
            None,
            timeline,
        )
    }

    /// H6 exact worker path with all first-divergence boundaries written into
    /// storage that was allocated before dispatch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_into_pinned_with_trace(
        &mut self,
        layer: u16,
        route: &PackedRouteDescriptor,
        owner_route_slot: u32,
        expert: CpuOwnerExpertView<'_>,
        input_bf16_bits: &[u16],
        output: &mut BoundedPinnedLease<u16>,
        trace: &mut SelectedExpertTraceStorage,
        timeline: Option<&CorrelatedTimeline>,
    ) -> Result<CpuX8SelectedExpertDeviceExecution> {
        self.execute_into_pinned_inner(
            layer,
            route,
            owner_route_slot,
            expert,
            input_bf16_bits,
            output,
            Some(trace),
            timeline,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_into_pinned_inner(
        &mut self,
        layer: u16,
        route: &PackedRouteDescriptor,
        owner_route_slot: u32,
        expert: CpuOwnerExpertView<'_>,
        input_bf16_bits: &[u16],
        output: &mut BoundedPinnedLease<u16>,
        mut trace: Option<&mut SelectedExpertTraceStorage>,
        timeline: Option<&CorrelatedTimeline>,
    ) -> Result<CpuX8SelectedExpertDeviceExecution> {
        if !matches!(route.owner, ExpertOwner::Cpu { .. })
            || route.route.expert_id != expert.expert_id
            || route.canonical_result_slot != route.route.canonical_result_slot()
            || route.source_activation_slot != route.route.activation_slot
            || usize::from(route.route.route_rank) >= super::contract::GPT_OSS_TOP_K
        {
            return Err(LLMError::ModelError(
                "CPU x8 selected-expert descriptor/owner mismatch".into(),
            ));
        }
        if expert.layer != layer {
            return Err(LLMError::ModelError(
                "CPU x8 selected-expert layer/expert mismatch".into(),
            ));
        }
        if input_bf16_bits.len() != HIDDEN_SIZE
            || input_bf16_bits
                .iter()
                .copied()
                .map(bf16::from_bits)
                .any(|value| !value.to_f32().is_finite())
        {
            return Err(LLMError::ModelError(
                "CPU x8 selected-expert input is invalid".into(),
            ));
        }
        let output_start = owner_route_slot as usize * HIDDEN_SIZE;
        if output_start
            .checked_add(HIDDEN_SIZE)
            .is_none_or(|end| end > output.as_slice().len())
        {
            return Err(LLMError::MemoryError(
                "CPU x8 selected-expert result lease is undersized".into(),
            ));
        }
        self.high_water_jobs = 1;
        if let Some(timeline) = timeline {
            timeline.record_host("cpu_expert", "compute_begin");
        }
        let started = Instant::now();
        exact_x8_gemv_into(
            expert.gate_up,
            expert.gate_up_bias,
            input_bf16_bits,
            &mut self.gate_up_bf16_bits,
        )?;
        if let Some(trace) = trace.as_deref_mut() {
            trace
                .gate_up_bf16_bits
                .copy_from_slice(&self.gate_up_bf16_bits);
            exact_swiglu_with_trace(&self.gate_up_bf16_bits, &mut self.swiglu_bf16_bits, trace);
        } else {
            exact_swiglu_into(&self.gate_up_bf16_bits, &mut self.swiglu_bf16_bits);
        }
        exact_x8_gemv_into(
            expert.down,
            expert.down_bias,
            &self.swiglu_bf16_bits,
            &mut output.as_mut_slice()[output_start..output_start + HIDDEN_SIZE],
        )?;
        if let Some(trace) = trace {
            trace
                .down_bf16_bits
                .copy_from_slice(&output.as_slice()[output_start..output_start + HIDDEN_SIZE]);
        }
        let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        if let Some(timeline) = timeline {
            timeline.record_host("cpu_expert", "compute_end");
        }
        Ok(CpuX8SelectedExpertDeviceExecution {
            route_contract: CanonicalRouteContract::from_packed_route(route),
            output_bytes: HIDDEN_SIZE * size_of::<u16>(),
            elapsed_ns,
        })
    }
}

impl Default for CpuX8SelectedExpertWorker {
    fn default() -> Self {
        Self::new()
    }
}

fn exact_x8_gemv_into(
    weights: Mxfp4MatrixView<'_>,
    bias: &[f32],
    input_bf16_bits: &[u16],
    output_bf16_bits: &mut [u16],
) -> Result<()> {
    if weights.blocks() != INPUT_BLOCKS
        || bias.len() != weights.rows()
        || output_bf16_bits.len() != weights.rows()
        || input_bf16_bits.len() != INPUT_BLOCKS * QUANT_BLOCK_SIZE
    {
        return Err(LLMError::ModelError(
            "invalid CPU x8 exact GEMV dimensions".into(),
        ));
    }
    for (row, output) in output_bf16_bits.iter_mut().enumerate() {
        let mut lanes = [0.0_f32; 16];
        for block_index in 0..INPUT_BLOCKS {
            let weight = weights
                .block(row, block_index)
                .map_err(|error| LLMError::ModelError(error.to_string()))?;
            let start = block_index * QUANT_BLOCK_SIZE;
            let activation =
                std::array::from_fn(|offset| bf16::from_bits(input_bf16_bits[start + offset]));
            accumulate_mxfp4_bf16_block(&weight, &activation, &mut lanes);
        }
        let total = bias[row] + lanes.into_iter().sum::<f32>();
        *output = bf16::from_f32(total).to_bits();
    }
    Ok(())
}

fn exact_swiglu_into(gate_up: &[u16], output: &mut [u16]) {
    debug_assert_eq!(gate_up.len(), GATE_UP_ROWS);
    debug_assert_eq!(output.len(), INTERMEDIATE_SIZE);
    for (index, output) in output.iter_mut().enumerate() {
        let gate = bf16::from_bits(gate_up[index * 2])
            .to_f32()
            .min(SWIGLU_LIMIT);
        let up = bf16::from_bits(gate_up[index * 2 + 1])
            .to_f32()
            .clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
        let scaled_gate = bf16::from_f32(gate * SWIGLU_ALPHA).to_f32();
        let sigmoid = bf16::from_f32(1.0 / (1.0 + (-scaled_gate).exp())).to_f32();
        let glu = bf16::from_f32(gate * sigmoid).to_f32();
        let linear = bf16::from_f32(up + 1.0).to_f32();
        *output = bf16::from_f32(glu * linear).to_bits();
    }
}

fn exact_swiglu_with_trace(
    gate_up: &[u16],
    output: &mut [u16],
    trace: &mut SelectedExpertTraceStorage,
) {
    debug_assert_eq!(gate_up.len(), GATE_UP_ROWS);
    debug_assert_eq!(output.len(), INTERMEDIATE_SIZE);
    for (index, output) in output.iter_mut().enumerate() {
        let gate = bf16::from_bits(gate_up[index * 2])
            .to_f32()
            .min(SWIGLU_LIMIT);
        let up = bf16::from_bits(gate_up[index * 2 + 1])
            .to_f32()
            .clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
        let scaled_gate = bf16::from_f32(gate * SWIGLU_ALPHA);
        let sigmoid = bf16::from_f32(1.0 / (1.0 + (-scaled_gate.to_f32()).exp()));
        let glu = bf16::from_f32(gate * sigmoid.to_f32());
        let linear = bf16::from_f32(up + 1.0);
        let swiglu = bf16::from_f32(glu.to_f32() * linear.to_f32());
        trace.scaled_gate_bf16_bits[index] = scaled_gate.to_bits();
        trace.sigmoid_bf16_bits[index] = sigmoid.to_bits();
        trace.glu_bf16_bits[index] = glu.to_bits();
        trace.linear_bf16_bits[index] = linear.to_bits();
        trace.swiglu_bf16_bits[index] = swiglu.to_bits();
        *output = swiglu.to_bits();
    }
}
