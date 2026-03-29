use gpt_oss_moe_semantics::{ExpertStorageKind, MoeTrace};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionSample {
    pub backend: String,
    pub case_name: String,
    pub trace: TraceSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceSummary {
    pub events: Vec<TraceEvent>,
    pub moe: Option<MoeTraceSummary>,
}

impl TraceSummary {
    pub fn empty() -> Self {
        Self {
            events: Vec::new(),
            moe: None,
        }
    }

    pub fn from_moe_trace(trace: &MoeTrace) -> Self {
        let moe = MoeTraceSummary::from(trace);
        let events = moe
            .tokens
            .iter()
            .map(|token| TraceEvent::MoeRouter {
                token_index: token.token_index,
                selected_experts: token.selected_experts.clone(),
                normalized_weights: token.normalized_weights.clone(),
            })
            .collect();
        Self {
            events,
            moe: Some(moe),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TraceEvent {
    MoeRouter {
        token_index: usize,
        selected_experts: Vec<usize>,
        normalized_weights: Vec<f32>,
    },
    Note(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeTraceSummary {
    pub storage: ExpertStorageKind,
    pub contract: gpt_oss_moe_semantics::MoeContract,
    pub tokens: Vec<MoeTokenSummary>,
}

impl From<&MoeTrace> for MoeTraceSummary {
    fn from(trace: &MoeTrace) -> Self {
        Self {
            storage: trace.contract.storage,
            contract: trace.contract.clone(),
            tokens: trace.tokens.iter().map(MoeTokenSummary::from).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeTokenSummary {
    pub token_index: usize,
    pub router_logits: Vec<f32>,
    pub selected_experts: Vec<usize>,
    pub normalized_weights: Vec<f32>,
    pub weight_sum: f32,
    pub routed_experts: Vec<RoutedExpertSummary>,
}

impl From<&gpt_oss_moe_semantics::MoeTokenTrace> for MoeTokenSummary {
    fn from(token: &gpt_oss_moe_semantics::MoeTokenTrace) -> Self {
        Self {
            token_index: token.token_index,
            router_logits: token.router_logits.clone(),
            selected_experts: token.selected_experts.clone(),
            normalized_weights: token.normalized_weights.clone(),
            weight_sum: token.weight_sum,
            routed_experts: token
                .routed_experts
                .iter()
                .map(RoutedExpertSummary::from)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutedExpertSummary {
    pub expert_index: usize,
    pub rank: usize,
    pub weight: f32,
}

impl From<&gpt_oss_moe_semantics::RoutedExpert> for RoutedExpertSummary {
    fn from(expert: &gpt_oss_moe_semantics::RoutedExpert) -> Self {
        Self {
            expert_index: expert.expert_index,
            rank: expert.rank,
            weight: expert.weight,
        }
    }
}
