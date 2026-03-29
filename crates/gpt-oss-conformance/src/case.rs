use crate::trace::TraceSummary;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceCase {
    pub name: String,
    pub inputs: Vec<u32>,
}

impl ConformanceCase {
    pub fn synthetic(name: impl Into<String>, inputs: Vec<u32>) -> Self {
        Self {
            name: name.into(),
            inputs,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionSample {
    pub logits: Vec<f32>,
    pub tokens: Vec<u32>,
    pub trace: TraceSummary,
}

#[derive(Clone, Debug)]
pub struct PlaceholderBackend {
    trace_suffix: String,
}

impl PlaceholderBackend {
    pub fn new(_name: impl Into<String>) -> Self {
        Self {
            trace_suffix: String::new(),
        }
    }

    pub fn with_trace_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.trace_suffix = suffix.into();
        self
    }

    pub fn run(&self, case: &ConformanceCase) -> ExecutionSample {
        let logits = case
            .inputs
            .iter()
            .enumerate()
            .map(|(idx, value)| (*value as f32) + (idx as f32) * 0.25)
            .collect::<Vec<_>>();
        let tokens = case.inputs.iter().map(|value| value % 11).collect::<Vec<_>>();
        let trace = TraceSummary::synthetic(
            format!("{}{}", case.name, self.trace_suffix),
            case.inputs.clone(),
        );

        ExecutionSample {
            logits,
            tokens,
            trace,
        }
    }
}
