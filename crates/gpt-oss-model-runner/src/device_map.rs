//! Inert device-map parsing for future multi-GPU layer sharding.
//!
//! This module is deliberately CUDA-free. It validates placement intent without
//! allocating devices, uploading tensors, or changing runtime execution.

use std::collections::BTreeSet;
use std::fmt;

/// CUDA device identifier used by an inert [`DeviceMap`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceId(pub usize);

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Parsed layer placement for future layer-sharded execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceMap {
    pub devices: Vec<DeviceId>,
    pub layer_device: Vec<DeviceId>,
    pub embedding_device: DeviceId,
    pub final_device: DeviceId,
}

impl DeviceMap {
    /// Build the default single-device map.
    pub fn single(num_layers: usize, current_device: DeviceId) -> Result<Self, DeviceMapError> {
        if num_layers == 0 {
            return Err(DeviceMapError::InvalidSpec(
                "num_layers must be greater than 0".into(),
            ));
        }

        Ok(Self {
            devices: vec![current_device],
            layer_device: vec![current_device; num_layers],
            embedding_device: current_device,
            final_device: current_device,
        })
    }

    /// Parse `single` or `split:<inclusive-layer-range>@<device>,...`.
    pub fn parse(
        spec: &str,
        num_layers: usize,
        current_device: DeviceId,
    ) -> Result<Self, DeviceMapError> {
        let spec = spec.trim();
        if spec.is_empty() {
            return Err(DeviceMapError::InvalidSpec(
                "device map spec must not be empty".into(),
            ));
        }

        if spec == "single" {
            return Self::single(num_layers, current_device);
        }

        let split_spec = spec.strip_prefix("split:").ok_or_else(|| {
            DeviceMapError::InvalidSpec(format!(
                "unknown device map spec '{spec}'; expected 'single' or 'split:<range>@<device>,...'"
            ))
        })?;

        Self::parse_split(split_spec, num_layers)
    }

    fn parse_split(split_spec: &str, num_layers: usize) -> Result<Self, DeviceMapError> {
        if num_layers == 0 {
            return Err(DeviceMapError::InvalidSpec(
                "num_layers must be greater than 0".into(),
            ));
        }
        if split_spec.trim().is_empty() {
            return Err(DeviceMapError::InvalidSpec(
                "split device map must include at least one range".into(),
            ));
        }

        let mut layer_device = vec![None; num_layers];
        let mut devices = BTreeSet::new();
        let mut previous_end: Option<usize> = None;

        for raw_entry in split_spec.split(',') {
            let entry = raw_entry.trim();
            if entry.is_empty() {
                return Err(DeviceMapError::InvalidSpec(
                    "split device map contains an empty range entry".into(),
                ));
            }

            let (range, device) = parse_entry(entry)?;

            if range.start > range.end {
                return Err(DeviceMapError::InvalidSpec(format!(
                    "reversed layer range '{}': start {} is greater than end {}",
                    range.raw, range.start, range.end
                )));
            }

            if range.end >= num_layers {
                return Err(DeviceMapError::InvalidSpec(format!(
                    "layer range '{}' is out of bounds for num_layers={num_layers}",
                    range.raw
                )));
            }

            if let Some(prev_end) = previous_end {
                if range.start <= prev_end {
                    return Err(DeviceMapError::InvalidSpec(format!(
                        "layer ranges must be ordered, non-overlapping, and without duplicate assignments: range '{}' starts at {} after previous end {}",
                        range.raw, range.start, prev_end
                    )));
                }
            }

            for layer in range.start..=range.end {
                if layer_device[layer].is_some() {
                    return Err(DeviceMapError::InvalidSpec(format!(
                        "duplicate layer assignment for layer {layer}"
                    )));
                }
                layer_device[layer] = Some(device);
            }

            devices.insert(device);
            previous_end = Some(range.end);
        }

        let mut assigned = Vec::with_capacity(num_layers);
        for (layer, device) in layer_device.into_iter().enumerate() {
            let device = device.ok_or_else(|| {
                DeviceMapError::InvalidSpec(format!(
                    "missing layer coverage: layer {layer} is not assigned"
                ))
            })?;
            assigned.push(device);
        }

        Ok(Self {
            devices: devices.into_iter().collect(),
            embedding_device: assigned[0],
            final_device: assigned[num_layers - 1],
            layer_device: assigned,
        })
    }

    /// Returns true for the only executable map in this slice.
    pub fn is_single_device(&self) -> bool {
        self.devices.len() == 1
    }
}

/// Device-map parse error with user-facing detail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceMapError {
    InvalidSpec(String),
}

impl fmt::Display for DeviceMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceMapError::InvalidSpec(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for DeviceMapError {}

#[derive(Debug)]
struct LayerRange<'a> {
    raw: &'a str,
    start: usize,
    end: usize,
}

fn parse_entry(entry: &str) -> Result<(LayerRange<'_>, DeviceId), DeviceMapError> {
    if entry.matches('@').count() != 1 {
        return Err(DeviceMapError::InvalidSpec(format!(
            "malformed range entry '{entry}'; expected '<start>-<end>@<device>'"
        )));
    }

    let (range_spec, device_spec) = entry.split_once('@').expect("checked one @");
    let range = parse_range(range_spec)?;
    let device = parse_device(device_spec)?;
    Ok((range, device))
}

fn parse_range(range_spec: &str) -> Result<LayerRange<'_>, DeviceMapError> {
    if range_spec.matches('-').count() != 1 {
        return Err(DeviceMapError::InvalidSpec(format!(
            "malformed layer range '{range_spec}'; expected '<start>-<end>'"
        )));
    }

    let (start_spec, end_spec) = range_spec.split_once('-').expect("checked one -");
    let start = parse_usize("layer start", start_spec)?;
    let end = parse_usize("layer end", end_spec)?;
    Ok(LayerRange {
        raw: range_spec,
        start,
        end,
    })
}

fn parse_device(device_spec: &str) -> Result<DeviceId, DeviceMapError> {
    parse_usize("device id", device_spec).map(DeviceId)
}

fn parse_usize(label: &str, value: &str) -> Result<usize, DeviceMapError> {
    if value.trim().is_empty() {
        return Err(DeviceMapError::InvalidSpec(format!(
            "{label} must not be empty"
        )));
    }
    if value.trim_start().starts_with('-') {
        return Err(DeviceMapError::InvalidSpec(format!(
            "{label} must be non-negative, got '{value}'"
        )));
    }
    value
        .parse::<usize>()
        .map_err(|_| DeviceMapError::InvalidSpec(format!("{label} must be numeric, got '{value}'")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(spec: &str) -> Result<DeviceMap, DeviceMapError> {
        DeviceMap::parse(spec, 24, DeviceId(0))
    }

    fn assert_invalid_contains(spec: &str, expected: &str) {
        let err = parse(spec).expect_err("spec should be invalid");
        let message = err.to_string();
        assert!(
            message.contains(expected),
            "expected error containing '{expected}', got '{message}'"
        );
    }

    #[test]
    fn device_map_single_assigns_all_layers_to_current_device() {
        let map = DeviceMap::parse("single", 24, DeviceId(7)).unwrap();

        assert_eq!(map.devices, vec![DeviceId(7)]);
        assert_eq!(map.layer_device.len(), 24);
        assert!(map.layer_device.iter().all(|&device| device == DeviceId(7)));
        assert_eq!(map.embedding_device, DeviceId(7));
        assert_eq!(map.final_device, DeviceId(7));
        assert!(map.is_single_device());
    }

    #[test]
    fn device_map_split_assigns_expected_layer_ranges() {
        let map = parse("split:0-11@0,12-23@1").unwrap();

        assert_eq!(map.devices, vec![DeviceId(0), DeviceId(1)]);
        assert_eq!(map.layer_device[0], DeviceId(0));
        assert_eq!(map.layer_device[11], DeviceId(0));
        assert_eq!(map.layer_device[12], DeviceId(1));
        assert_eq!(map.layer_device[23], DeviceId(1));
        assert_eq!(map.embedding_device, DeviceId(0));
        assert_eq!(map.final_device, DeviceId(1));
        assert!(!map.is_single_device());
    }

    #[test]
    fn device_map_rejects_empty_spec() {
        assert_invalid_contains("", "must not be empty");
    }

    #[test]
    fn device_map_rejects_unknown_prefix() {
        assert_invalid_contains("other:0-23@0", "unknown device map spec");
    }

    #[test]
    fn device_map_rejects_malformed_range() {
        assert_invalid_contains("split:0@0", "malformed layer range");
    }

    #[test]
    fn device_map_rejects_reversed_range() {
        assert_invalid_contains("split:11-0@0,12-23@1", "reversed layer range");
    }

    #[test]
    fn device_map_rejects_overlapping_ranges() {
        assert_invalid_contains("split:0-12@0,12-23@1", "ordered, non-overlapping");
    }

    #[test]
    fn device_map_rejects_missing_layer_coverage() {
        assert_invalid_contains("split:0-10@0,12-23@1", "missing layer coverage");
    }

    #[test]
    fn device_map_rejects_out_of_bounds_layer_index() {
        assert_invalid_contains("split:0-11@0,12-24@1", "out of bounds");
    }

    #[test]
    fn device_map_rejects_duplicate_layer_assignment() {
        assert_invalid_contains("split:0-11@0,11-11@1,12-23@1", "duplicate assignments");
    }

    #[test]
    fn device_map_rejects_unordered_ranges() {
        assert_invalid_contains("split:12-23@1,0-11@0", "ordered, non-overlapping");
    }

    #[test]
    fn device_map_rejects_non_numeric_device_id() {
        assert_invalid_contains("split:0-11@gpu0,12-23@1", "device id must be numeric");
    }

    #[test]
    fn device_map_rejects_negative_layer_values() {
        assert_invalid_contains("split:-1-11@0,12-23@1", "malformed layer range");
    }

    #[test]
    fn device_map_rejects_negative_device_values() {
        assert_invalid_contains("split:0-11@-1,12-23@1", "device id must be non-negative");
    }
}
