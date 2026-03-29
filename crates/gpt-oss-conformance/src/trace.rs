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
        for layer_index in 0..num_layers {
            let mut frame = TraceFrame::new(format!("{label}:layer-{layer_index}"));
            frame.events.push(TraceEvent::new(
                "layer",
                format!(
                    "{}:{}->{} positions={:?}",
                    layer_index, token_count, token_count, position_ids
                ),
            ));
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
