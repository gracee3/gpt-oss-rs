use gpt_oss_moe_semantics::MoeContract;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConformanceCase {
    pub name: String,
    pub moe: Option<MoeCaseMetadata>,
}

impl ConformanceCase {
    pub fn moe(
        name: impl Into<String>,
        contract: MoeContract,
        router_logits: Vec<Vec<f32>>,
    ) -> Self {
        Self {
            name: name.into(),
            moe: Some(MoeCaseMetadata {
                contract,
                router_logits,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeCaseMetadata {
    pub contract: MoeContract,
    pub router_logits: Vec<Vec<f32>>,
}
