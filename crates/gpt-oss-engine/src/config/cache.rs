//! KV-cache configuration.

use serde::{Deserialize, Serialize};

use gpt_oss_kv_model::CacheLayoutSpec;

/// Configuration controlling KV-cache block allocation and memory budget.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheConfigImpl {
    /// Number of tokens per cache block.
    pub block_size: usize,
    /// Fraction of GPU memory to use for KV cache (0.0..1.0].
    pub gpu_memory_utilization: f32,
    /// CPU swap space budget in GiB.
    pub swap_space_gb: f32,
    /// Override: fixed number of GPU blocks (computed at runtime if None).
    pub num_gpu_blocks: Option<usize>,
    /// Override: fixed number of CPU blocks (computed at runtime if None).
    pub num_cpu_blocks: Option<usize>,
    /// Enable prefix caching to reuse KV blocks for shared prompt prefixes.
    #[serde(default)]
    pub enable_prefix_caching: bool,
    /// KV cache data type: "auto" (fp16), "fp8", "fp8_e4m3".
    #[serde(default = "default_kv_cache_dtype")]
    pub kv_cache_dtype: String,
}

fn default_kv_cache_dtype() -> String {
    "auto".into()
}

impl Default for CacheConfigImpl {
    fn default() -> Self {
        Self {
            block_size: 16,
            gpu_memory_utilization: 0.90,
            swap_space_gb: 4.0,
            num_gpu_blocks: None,
            num_cpu_blocks: None,
            enable_prefix_caching: false,
            kv_cache_dtype: "auto".into(),
        }
    }
}

impl CacheConfigImpl {
    /// Create a new builder for tests and programmatic construction.
    pub fn builder() -> CacheConfigBuilder {
        CacheConfigBuilder::default()
    }

    /// Convert runtime cache config into backend-independent cache semantics.
    ///
    /// This keeps the current runtime behavior unchanged while giving later
    /// planner work a stable cache-layout shape to depend on.
    pub fn semantic_layout(&self, num_layers: usize) -> CacheLayoutSpec {
        CacheLayoutSpec::new(num_layers, self.block_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpt_oss_kv_model::CacheVisibility;

    #[test]
    fn semantic_layout_uses_runtime_block_size() {
        let cfg = CacheConfigImpl::builder().block_size(32).build();
        let layout = cfg.semantic_layout(4);

        assert_eq!(layout.num_layers, 4);
        assert_eq!(layout.block_size, 32);
        assert!(matches!(
            layout.layer(0).unwrap().visibility,
            CacheVisibility::Full
        ));
    }
}

/// Builder for [`CacheConfigImpl`].
#[derive(Debug, Default)]
pub struct CacheConfigBuilder(CacheConfigImpl);

impl CacheConfigBuilder {
    /// Set block size.
    pub fn block_size(mut self, v: usize) -> Self {
        self.0.block_size = v;
        self
    }

    /// Set GPU memory utilization fraction.
    pub fn gpu_memory_utilization(mut self, v: f32) -> Self {
        self.0.gpu_memory_utilization = v;
        self
    }

    /// Set swap space in GiB.
    pub fn swap_space_gb(mut self, v: f32) -> Self {
        self.0.swap_space_gb = v;
        self
    }

    /// Set fixed number of GPU blocks.
    pub fn num_gpu_blocks(mut self, v: usize) -> Self {
        self.0.num_gpu_blocks = Some(v);
        self
    }

    /// Set fixed number of CPU blocks.
    pub fn num_cpu_blocks(mut self, v: usize) -> Self {
        self.0.num_cpu_blocks = Some(v);
        self
    }

    /// Enable prefix caching.
    pub fn enable_prefix_caching(mut self, v: bool) -> Self {
        self.0.enable_prefix_caching = v;
        self
    }

    /// Set KV cache data type.
    pub fn kv_cache_dtype(mut self, v: impl Into<String>) -> Self {
        self.0.kv_cache_dtype = v.into();
        self
    }

    /// Consume the builder and return the config.
    pub fn build(self) -> CacheConfigImpl {
        self.0
    }
}
