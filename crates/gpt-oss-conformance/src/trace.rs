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
}
