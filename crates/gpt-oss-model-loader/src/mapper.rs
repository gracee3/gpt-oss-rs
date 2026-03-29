/// Maps HuggingFace weight names for this GPT-OSS-only fork.
///
/// The runtime codepaths consume raw HuggingFace tensor names (`model.layers.*`,
/// `model.embed_tokens.*`, `lm_head.*`), so the mapper is now intentionally a
/// no-op. It remains as a small seam in case GPT-OSS-specific remapping is
/// needed later.
#[derive(Debug)]
pub struct WeightMapper;

impl WeightMapper {
    pub fn new(_model_name: &str) -> Self {
        Self
    }

    pub fn map_name(&self, hf_name: &str) -> String {
        hf_name.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_huggingface_weight_names() {
        let mapper = WeightMapper::new("openai/gpt-oss-20b");
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(mapper.map_name("lm_head.weight"), "lm_head.weight");
    }
}
