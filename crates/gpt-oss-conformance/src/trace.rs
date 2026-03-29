use gpt_oss_reference::ReferenceTrace;
use gpt_oss_runtime_plan::ExecutionPlan;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceEvent {
    pub stage: String,
    pub payload: String,
}

impl TraceEvent {
    pub fn new(stage: impl Into<String>, payload: impl Into<String>) -> Self {
        Self {
            stage: stage.into(),
            payload: payload.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceFrame {
    pub label: String,
    pub events: Vec<TraceEvent>,
}

impl TraceFrame {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            events: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceSummary {
    pub label: String,
    pub frames: Vec<TraceFrame>,
}

impl TraceSummary {
    pub fn synthetic(label: impl Into<String>, inputs: Vec<u32>) -> Self {
        let label = label.into();
        let mut frame = TraceFrame::new(label.clone());
        frame
            .events
            .push(TraceEvent::new("inputs", format!("{inputs:?}")));
        Self {
            label,
            frames: vec![frame],
        }
    }

    pub fn from_observed_case(label: impl Into<String>, is_prefill: bool, seq_start_pos: u32) -> Self {
        let label = label.into();
        let mut frame = TraceFrame::new(format!("{label}:observed"));
        frame.events.push(TraceEvent::new(
            "reference_phase",
            if is_prefill { "Prefill" } else { "Decode" },
        ));
        frame.events.push(TraceEvent::new("seq_start_pos", seq_start_pos.to_string()));
        Self {
            label,
            frames: vec![frame],
        }
    }

    pub fn from_observed_case_with_plan(
        label: impl Into<String>,
        plan: &ExecutionPlan,
        is_prefill: bool,
        seq_start_pos: u32,
        token_count: usize,
        block_size: usize,
        layer_types: &[String],
        traced_num_local_experts: usize,
        traced_num_experts_per_tok: usize,
        traced_moe_layer_indices: &[usize],
        traced_selected_experts: Option<&[usize]>,
        num_layers: usize,
    ) -> Self {
        let label = label.into();
        let mut frames = Vec::with_capacity(num_layers.saturating_add(1));

        let mut plan_frame = TraceFrame::new(format!("{label}:plan"));
        plan_frame.events.push(TraceEvent::new(
            "runtime_mode",
            format!("{:?}", plan.runtime_mode),
        ));
        plan_frame.events.push(TraceEvent::new(
            "backend_path",
            format!("{:?}", plan.backend_path),
        ));
        plan_frame.events.push(TraceEvent::new(
            "request_kind",
            format!("{:?}", plan.request_kind),
        ));
        plan_frame.events.push(TraceEvent::new(
            "reference_phase",
            if is_prefill { "Prefill" } else { "Decode" },
        ));
        plan_frame.events.push(TraceEvent::new(
            "seq_start_pos",
            seq_start_pos.to_string(),
        ));
        plan_frame.events.push(TraceEvent::new(
            "graph_policy",
            format!("{:?}", plan.graph_policy),
        ));
        plan_frame.events.push(TraceEvent::new(
            "output_policy",
            format!("{:?}", plan.output_policy),
        ));
        plan_frame
            .events
            .push(TraceEvent::new("reason", plan.reason.clone()));
        frames.push(plan_frame);

        let position_ids = (0..token_count)
            .map(|idx| seq_start_pos + idx as u32)
            .collect::<Vec<_>>();
        let visible_tokens = (0..(seq_start_pos as usize + token_count)).collect::<Vec<_>>();
        let blocks_touched = if token_count == 0 {
            0
        } else {
            ((seq_start_pos as usize + token_count - 1) / block_size)
                .saturating_sub(seq_start_pos as usize / block_size)
                + 1
        };
        for layer_index in 0..num_layers {
            let mut frame = TraceFrame::new(format!("{label}:layer-{layer_index}"));
            frame.events.push(TraceEvent::new(
                "layer",
                format!(
                    "{}:{}->{} positions={:?}",
                    layer_index, token_count, token_count, position_ids
                ),
            ));
            let layer_type = layer_types
                .get(layer_index)
                .map(String::as_str)
                .unwrap_or("full_attention");
            if matches!(layer_type, "full_attention" | "global_attention") {
                frame.events.push(TraceEvent::new(
                    "attention",
                    format!("Full/{} visible={:?}", visible_tokens.len(), visible_tokens),
                ));
                frame.events.push(TraceEvent::new(
                    "cache",
                    format!(
                        "block_size={} visibility=Full blocks={}",
                        block_size, blocks_touched
                    ),
                ));
            }
            let effective_top_k = traced_num_experts_per_tok.min(traced_num_local_experts);
            if effective_top_k > 0 && traced_moe_layer_indices.contains(&layer_index) {
                let selected = traced_selected_experts
                    .map(|selected| selected.to_vec())
                    .unwrap_or_else(|| (0..effective_top_k).collect::<Vec<_>>());
                frame.events.push(TraceEvent::new(
                    "moe",
                    format!(
                        "SparseTopK/{} selected={:?}",
                        token_count * effective_top_k,
                        vec![selected; token_count]
                    ),
                ));
            } else {
                frame.events.push(TraceEvent::new("moe", "DenseOnly/0 selected=[]"));
            }
            frames.push(frame);
        }

        Self { label, frames }
    }

    pub fn from_reference(
        label: impl Into<String>,
        plan: &ExecutionPlan,
        trace: &ReferenceTrace,
    ) -> Self {
        let label = label.into();
        let mut frames = Vec::with_capacity(trace.layers.len() + 1);

        let mut plan_frame = TraceFrame::new(format!("{label}:plan"));
        plan_frame.events.push(TraceEvent::new(
            "runtime_mode",
            format!("{:?}", plan.runtime_mode),
        ));
        plan_frame.events.push(TraceEvent::new(
            "backend_path",
            format!("{:?}", plan.backend_path),
        ));
        plan_frame.events.push(TraceEvent::new(
            "request_kind",
            format!("{:?}", plan.request_kind),
        ));
        plan_frame.events.push(TraceEvent::new(
            "reference_phase",
            format!("{:?}", trace.phase),
        ));
        plan_frame.events.push(TraceEvent::new(
            "seq_start_pos",
            trace.seq_start_pos.to_string(),
        ));
        plan_frame.events.push(TraceEvent::new(
            "graph_policy",
            format!("{:?}", plan.graph_policy),
        ));
        plan_frame.events.push(TraceEvent::new(
            "output_policy",
            format!("{:?}", plan.output_policy),
        ));
        plan_frame
            .events
            .push(TraceEvent::new("reason", plan.reason.clone()));
        frames.push(plan_frame);

        for layer in &trace.layers {
            let mut frame = TraceFrame::new(format!("{label}:layer-{}", layer.layer_index));
            frame.events.push(TraceEvent::new(
                "layer",
                format!(
                    "{}:{}->{} positions={:?}",
                    layer.layer_index, layer.input_tokens, layer.output_tokens, layer.position_ids
                ),
            ));
            frame.events.push(TraceEvent::new(
                "attention",
                format!(
                    "{:?}/{} visible={:?}",
                    layer.attention.mode, layer.attention.attended_tokens, layer.attention.visible_tokens
                ),
            ));
            frame.events.push(TraceEvent::new(
                "moe",
                format!(
                    "{:?}/{} selected={:?}",
                    layer.moe.mode, layer.moe.experts_invoked, layer.moe.selected_experts
                ),
            ));
            frame.events.push(TraceEvent::new(
                "cache",
                format!(
                    "block_size={} visibility={:?} blocks={}",
                    trace.cache_layout.block_size,
                    trace.cache_layout.visibility,
                    trace.cache.blocks.len()
                ),
            ));
            frames.push(frame);
        }

        Self { label, frames }
    }

    pub fn find_event_payload(&self, stage: &str) -> Option<&str> {
        self.frames
            .iter()
            .flat_map(|frame| frame.events.iter())
            .find(|event| event.stage == stage)
            .map(|event| event.payload.as_str())
    }
}
