//! Device configuration.

use serde::{Deserialize, Serialize};

use gpt_oss_model_runner::{DeviceId, DeviceMap};

/// Which device family to target.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeviceConfig {
    /// Device string: "cuda", "cpu", "metal", etc.
    pub device: String,
    /// Inert multi-GPU layer-sharding placement spec.
    ///
    /// Only `single` is executable today. Split maps are parsed at startup to
    /// validate placement intent, then rejected before CUDA allocation.
    #[serde(default = "default_device_map")]
    pub device_map: String,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device: "cuda".into(),
            device_map: default_device_map(),
        }
    }
}

impl DeviceConfig {
    /// Returns true when targeting a GPU device.
    pub fn is_gpu(&self) -> bool {
        matches!(self.device.as_str(), "cuda" | "metal")
    }

    /// Create a new builder for tests and programmatic construction.
    pub fn builder() -> DeviceConfigBuilder {
        DeviceConfigBuilder::default()
    }
}

fn default_device_map() -> String {
    "single".into()
}

/// Validate the startup device-map spec before CUDA resources are allocated.
pub fn validate_device_map_spec_for_cuda_startup(
    spec: Option<&str>,
    num_layers: usize,
    selected_device_id: usize,
) -> Result<DeviceMap, String> {
    let spec = spec.unwrap_or("single").trim();
    let selected_device = DeviceId(selected_device_id);

    if spec.is_empty() || spec == "single" {
        return DeviceMap::single(num_layers, selected_device).map_err(|e| e.to_string());
    }

    let map = DeviceMap::parse(spec, num_layers, selected_device).map_err(|e| e.to_string())?;

    if spec.starts_with("split:") && !map.is_single_device() {
        return Err("split device maps are parsed but not executable yet".into());
    }

    map.validate_single_device_executable(num_layers)
        .map_err(|e| e.to_string())?;

    let expected = selected_device;
    if map.devices.first().copied() != Some(expected)
        || map.embedding_device != expected
        || map.final_device != expected
        || map.layer_device.iter().any(|&device| device != expected)
    {
        return Err(format!(
            "single executable device map must match selected CUDA device {selected_device_id}"
        ));
    }

    Ok(map)
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

    /// Set inert device-map placement spec.
    pub fn device_map(mut self, v: impl Into<String>) -> Self {
        self.0.device_map = v.into();
        self
    }

    /// Consume the builder and return the config.
    pub fn build(self) -> DeviceConfig {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_device_map_resolves_to_selected_single_device() {
        let map = validate_device_map_spec_for_cuda_startup(None, 24, 3).unwrap();

        assert_eq!(map.devices, vec![DeviceId(3)]);
        assert_eq!(map.embedding_device, DeviceId(3));
        assert_eq!(map.final_device, DeviceId(3));
        assert!(map.layer_device.iter().all(|&device| device == DeviceId(3)));
    }

    #[test]
    fn explicit_single_device_map_resolves_to_selected_single_device() {
        let map = validate_device_map_spec_for_cuda_startup(Some("single"), 24, 2).unwrap();

        assert_eq!(map.devices, vec![DeviceId(2)]);
        assert!(map.layer_device.iter().all(|&device| device == DeviceId(2)));
    }

    #[test]
    fn valid_split_device_map_is_rejected_as_non_executable() {
        let err = validate_device_map_spec_for_cuda_startup(Some("split:0-11@0,12-23@1"), 24, 0)
            .unwrap_err();

        assert!(
            err.contains("split device maps are parsed but not executable yet"),
            "got: {err}"
        );
    }

    #[test]
    fn invalid_device_map_is_rejected_before_runtime_construction() {
        let err = validate_device_map_spec_for_cuda_startup(Some("split:0-10@0,12-23@1"), 24, 0)
            .unwrap_err();

        assert!(err.contains("missing layer coverage"), "got: {err}");
    }

    #[test]
    fn single_device_map_for_wrong_device_is_rejected() {
        let err =
            validate_device_map_spec_for_cuda_startup(Some("split:0-23@1"), 24, 0).unwrap_err();

        assert!(
            err.contains("must match selected CUDA device 0"),
            "got: {err}"
        );
    }
}
