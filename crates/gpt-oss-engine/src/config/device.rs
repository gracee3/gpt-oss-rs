//! Device configuration.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

fn default_device() -> String {
    "auto".into()
}

fn default_cpu_kernel() -> String {
    "auto".into()
}

fn default_cpu_matmul_backend() -> String {
    "auto".into()
}

fn default_cpu_threads() -> usize {
    num_cpus::get_physical().max(1)
}

fn default_cpu_repack_cache() -> PathBuf {
    if let Some(path) = std::env::var_os("GPT_OSS_RS_CACHE") {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(path).join("gpt-oss-rs");
    }
    if let Some(path) = std::env::var_os("HOME") {
        return PathBuf::from(path).join(".cache/gpt-oss-rs");
    }
    PathBuf::from(".cache/gpt-oss-rs")
}

/// Which device family to target.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeviceConfig {
    /// Device string: "auto", "cpu", "cuda", or explicit test-only "mock".
    #[serde(default = "default_device")]
    pub device: String,
    /// Native CPU kernel dispatch path.
    #[serde(default = "default_cpu_kernel")]
    pub cpu_kernel: String,
    /// MXFP4 matrix backend used for multi-row CPU execution.
    #[serde(default = "default_cpu_matmul_backend")]
    pub cpu_matmul_backend: String,
    /// Rayon worker threads used by the batch-one CPU runner.
    #[serde(default = "default_cpu_threads")]
    pub cpu_threads: usize,
    /// Root for versioned, memory-mapped MXFP4 repacks.
    #[serde(default = "default_cpu_repack_cache")]
    pub cpu_repack_cache: PathBuf,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device: default_device(),
            cpu_kernel: default_cpu_kernel(),
            cpu_matmul_backend: default_cpu_matmul_backend(),
            cpu_threads: default_cpu_threads(),
            cpu_repack_cache: default_cpu_repack_cache(),
        }
    }
}

impl DeviceConfig {
    /// Returns true when targeting a GPU device.
    pub fn is_gpu(&self) -> bool {
        self.device == "cuda"
    }

    /// Create a new builder for tests and programmatic construction.
    pub fn builder() -> DeviceConfigBuilder {
        DeviceConfigBuilder::default()
    }
}

/// Builder for [`DeviceConfig`].
#[derive(Debug, Default)]
pub struct DeviceConfigBuilder(DeviceConfig);

impl DeviceConfigBuilder {
    /// Set device string.
    pub fn device(mut self, v: impl Into<String>) -> Self {
        self.0.device = v.into();
        self
    }

    /// Select the native CPU kernel path.
    pub fn cpu_kernel(mut self, v: impl Into<String>) -> Self {
        self.0.cpu_kernel = v.into();
        self
    }

    /// Select the MXFP4 matrix backend.
    pub fn cpu_matmul_backend(mut self, v: impl Into<String>) -> Self {
        self.0.cpu_matmul_backend = v.into();
        self
    }

    /// Set the physical CPU worker count.
    pub fn cpu_threads(mut self, v: usize) -> Self {
        self.0.cpu_threads = v;
        self
    }

    /// Set the MXFP4 repack cache root.
    pub fn cpu_repack_cache(mut self, v: impl Into<PathBuf>) -> Self {
        self.0.cpu_repack_cache = v.into();
        self
    }

    /// Consume the builder and return the config.
    pub fn build(self) -> DeviceConfig {
        self.0
    }
}
