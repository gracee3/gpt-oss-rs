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
                    "{}:{}->{}",
                    layer.layer_index, layer.input_tokens, layer.output_tokens
                ),
            ));
            frame.events.push(TraceEvent::new(
                "attention",
                format!(
                    "{:?}/{}",
                    trace.attention.mode, trace.attention.attended_tokens
                ),
            ));
            frame.events.push(TraceEvent::new(
                "moe",
                format!("{:?}/{}", trace.moe.mode, trace.moe.experts_invoked),
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
}
