//! GPU device descriptor, stable PCI identity, and enumeration.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Normalized PCI domain:bus:device.function identity.
///
/// CUDA ordinals are process-local and may change across boots or environment
/// configuration. This value is the durable identity used by placement data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct PciBusId {
    domain: u16,
    bus: u8,
    device: u8,
    function: u8,
}

impl PciBusId {
    pub const fn new(domain: u16, bus: u8, device: u8, function: u8) -> Self {
        Self {
            domain,
            bus,
            device,
            function,
        }
    }

    pub const fn domain(self) -> u16 {
        self.domain
    }

    pub const fn bus(self) -> u8 {
        self.bus
    }

    pub const fn device(self) -> u8 {
        self.device
    }

    pub const fn function(self) -> u8 {
        self.function
    }
}

impl fmt::Display for PciBusId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

impl FromStr for PciBusId {
    type Err = StableDeviceError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (domain_bus, device_function) = value
            .trim()
            .rsplit_once(':')
            .ok_or_else(|| StableDeviceError::InvalidPciBusId(value.to_owned()))?;
        let (domain, bus) = domain_bus
            .rsplit_once(':')
            .ok_or_else(|| StableDeviceError::InvalidPciBusId(value.to_owned()))?;
        let (device, function) = device_function
            .split_once('.')
            .ok_or_else(|| StableDeviceError::InvalidPciBusId(value.to_owned()))?;
        let parse = |component: &str| {
            u32::from_str_radix(component, 16)
                .map_err(|_| StableDeviceError::InvalidPciBusId(value.to_owned()))
        };
        let domain = parse(domain)?;
        let bus = parse(bus)?;
        let device = parse(device)?;
        let function = parse(function)?;
        if domain > u16::MAX as u32 || bus > u8::MAX as u32 || device > 0x1f || function > 0x7 {
            return Err(StableDeviceError::InvalidPciBusId(value.to_owned()));
        }
        Ok(Self::new(
            domain as u16,
            bus as u8,
            device as u8,
            function as u8,
        ))
    }
}

impl TryFrom<String> for PciBusId {
    type Error = StableDeviceError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<PciBusId> for String {
    fn from(value: PciBusId) -> Self {
        value.to_string()
    }
}

/// Durable placement identity and minimum admission requirements for one GPU.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StableCudaDeviceId {
    pub pci_bus_id: PciBusId,
    pub expected_name: String,
    pub compute_capability: (u32, u32),
    pub minimum_memory: u64,
}

/// A durable identity resolved to this process's transient CUDA ordinal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCudaDevice {
    pub stable_id: StableCudaDeviceId,
    pub transient_ordinal: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StableDeviceError {
    #[error("invalid PCI bus identity '{0}'")]
    InvalidPciBusId(String),
    #[error("device {0} has no stable PCI identity")]
    MissingPciIdentity(usize),
    #[error("no CUDA device matches stable identity {0}")]
    NotFound(PciBusId),
    #[error("more than one CUDA device matches stable identity {0}")]
    Duplicate(PciBusId),
    #[error(
        "stable device {pci_bus_id} name mismatch: expected '{expected}', observed '{observed}'"
    )]
    NameMismatch {
        pci_bus_id: PciBusId,
        expected: String,
        observed: String,
    },
    #[error("stable device {pci_bus_id} compute capability mismatch: expected {expected:?}, observed {observed:?}")]
    ComputeCapabilityMismatch {
        pci_bus_id: PciBusId,
        expected: (u32, u32),
        observed: (u32, u32),
    },
    #[error("stable device {pci_bus_id} has {observed} bytes, below required {minimum} bytes")]
    InsufficientMemory {
        pci_bus_id: PciBusId,
        minimum: u64,
        observed: u64,
    },
}

/// Memory usage snapshot for a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryInfo {
    pub total: usize,
    pub free: usize,
    pub used: usize,
}

/// Static descriptor for a GPU device.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub id: usize,
    pub name: String,
    pub compute_capability: (u32, u32),
    pub total_memory: usize,
    pub pci_bus_id: Option<PciBusId>,
}

impl StableCudaDeviceId {
    pub fn from_device(device: &GpuDevice) -> Result<Self, StableDeviceError> {
        Ok(Self {
            pci_bus_id: device
                .pci_bus_id
                .ok_or(StableDeviceError::MissingPciIdentity(device.id))?,
            expected_name: device.name.clone(),
            compute_capability: device.compute_capability,
            minimum_memory: device.total_memory as u64,
        })
    }
}

/// Resolve a durable identity without trusting the manifest's prior ordinal.
pub fn resolve_stable_device(
    expected: &StableCudaDeviceId,
    devices: &[GpuDevice],
) -> Result<ResolvedCudaDevice, StableDeviceError> {
    let mut matches = devices
        .iter()
        .filter(|device| device.pci_bus_id == Some(expected.pci_bus_id));
    let device = matches
        .next()
        .ok_or(StableDeviceError::NotFound(expected.pci_bus_id))?;
    if matches.next().is_some() {
        return Err(StableDeviceError::Duplicate(expected.pci_bus_id));
    }
    if device.name != expected.expected_name {
        return Err(StableDeviceError::NameMismatch {
            pci_bus_id: expected.pci_bus_id,
            expected: expected.expected_name.clone(),
            observed: device.name.clone(),
        });
    }
    if device.compute_capability != expected.compute_capability {
        return Err(StableDeviceError::ComputeCapabilityMismatch {
            pci_bus_id: expected.pci_bus_id,
            expected: expected.compute_capability,
            observed: device.compute_capability,
        });
    }
    if (device.total_memory as u64) < expected.minimum_memory {
        return Err(StableDeviceError::InsufficientMemory {
            pci_bus_id: expected.pci_bus_id,
            minimum: expected.minimum_memory,
            observed: device.total_memory as u64,
        });
    }
    Ok(ResolvedCudaDevice {
        stable_id: expected.clone(),
        transient_ordinal: device.id,
    })
}

/// Enumerate available GPU devices.
///
/// Under `mock-gpu` this returns a single virtual device.
/// Under `cuda` this queries the CUDA driver for real devices.
pub fn list_devices() -> Vec<GpuDevice> {
    #[cfg(feature = "cuda")]
    {
        cuda_list_devices()
    }
    #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
    {
        vec![GpuDevice {
            id: 0,
            name: "MockGPU-0".into(),
            compute_capability: (8, 0),
            total_memory: 16 * 1024 * 1024 * 1024, // 16 GiB
            pci_bus_id: None,
        }]
    }
    #[cfg(not(any(feature = "mock-gpu", feature = "cuda")))]
    {
        Vec::new()
    }
}

#[cfg(feature = "cuda")]
fn cuda_list_devices() -> Vec<GpuDevice> {
    use cudarc::driver::CudaContext;

    let count = match CudaContext::device_count() {
        Ok(n) => n as usize,
        Err(e) => {
            tracing::warn!("Failed to query CUDA device count: {e}");
            return Vec::new();
        }
    };

    let mut devices = Vec::with_capacity(count);
    for id in 0..count {
        let ctx = match CudaContext::new(id) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(id, "Failed to init CUDA device: {e}");
                continue;
            }
        };

        let name = ctx.name().unwrap_or_else(|_| format!("CUDA Device {id}"));

        let (major, minor) = ctx.compute_capability().unwrap_or((0, 0));

        let total_memory = ctx.total_mem().unwrap_or(0);
        let pci_bus_id = cuda_pci_bus_id(&ctx, id);

        devices.push(GpuDevice {
            id,
            name,
            compute_capability: (major as u32, minor as u32),
            total_memory,
            pci_bus_id,
        });
    }

    devices
}

#[cfg(feature = "cuda")]
fn cuda_pci_bus_id(context: &cudarc::driver::CudaContext, id: usize) -> Option<PciBusId> {
    use std::ffi::CStr;

    let mut buffer = [0_i8; 32];
    // SAFETY: `buffer` is writable for the supplied length and `cu_device`
    // remains owned by the live context. CUDA writes a NUL-terminated ID.
    let result = unsafe {
        cudarc::driver::sys::cuDeviceGetPCIBusId(
            buffer.as_mut_ptr(),
            buffer.len() as i32,
            context.cu_device(),
        )
    };
    if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        tracing::warn!(id, ?result, "Failed to query CUDA PCI bus identity");
        return None;
    }
    // SAFETY: CUDA succeeded and guarantees a NUL-terminated string within
    // the provided buffer.
    let value = unsafe { CStr::from_ptr(buffer.as_ptr()) };
    match value.to_str().ok().and_then(|value| value.parse().ok()) {
        Some(identity) => Some(identity),
        None => {
            tracing::warn!(id, "CUDA returned an invalid PCI bus identity");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
    fn list_devices_returns_mock() {
        let devs = list_devices();
        assert_eq!(devs.len(), 1);
        assert_eq!(devs[0].id, 0);
        assert!(devs[0].name.contains("Mock"));
    }

    #[test]
    fn memory_info_eq() {
        let a = MemoryInfo {
            total: 100,
            free: 60,
            used: 40,
        };
        let b = MemoryInfo {
            total: 100,
            free: 60,
            used: 40,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn pci_bus_id_normalizes_cuda_domain_width() {
        let short: PciBusId = "0000:19:00.0".parse().unwrap();
        let long: PciBusId = "00000000:19:00.0".parse().unwrap();
        assert_eq!(short, long);
        assert_eq!(short.to_string(), "0000:19:00.0");
        assert!("0000:19:20.0".parse::<PciBusId>().is_err());
        assert!("0000:19:00.8".parse::<PciBusId>().is_err());
    }

    fn test_device(id: usize, pci: &str) -> GpuDevice {
        GpuDevice {
            id,
            name: "NVIDIA GeForce RTX 3090".into(),
            compute_capability: (8, 6),
            total_memory: 24 * 1024 * 1024 * 1024,
            pci_bus_id: Some(pci.parse().unwrap()),
        }
    }

    #[test]
    fn stable_resolution_ignores_ordinal_order() {
        let target = StableCudaDeviceId::from_device(&test_device(0, "0000:19:00.0")).unwrap();
        let devices = [
            test_device(0, "0000:65:00.0"),
            test_device(1, "0000:19:00.0"),
        ];
        let resolved = resolve_stable_device(&target, &devices).unwrap();
        assert_eq!(resolved.transient_ordinal, 1);
        assert_eq!(resolved.stable_id.pci_bus_id, target.pci_bus_id);
    }

    #[test]
    fn stable_resolution_rejects_duplicate_and_mismatch() {
        let target = StableCudaDeviceId::from_device(&test_device(0, "0000:19:00.0")).unwrap();
        let duplicate = [
            test_device(0, "0000:19:00.0"),
            test_device(1, "0000:19:00.0"),
        ];
        assert!(matches!(
            resolve_stable_device(&target, &duplicate),
            Err(StableDeviceError::Duplicate(_))
        ));

        let mut wrong = test_device(0, "0000:19:00.0");
        wrong.compute_capability = (9, 0);
        assert!(matches!(
            resolve_stable_device(&target, &[wrong]),
            Err(StableDeviceError::ComputeCapabilityMismatch { .. })
        ));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_devices_publish_resolvable_stable_pci_identities() {
        let devices = list_devices();
        assert!(!devices.is_empty(), "CUDA feature test requires a device");
        for device in &devices {
            let stable = StableCudaDeviceId::from_device(device).unwrap();
            let resolved = resolve_stable_device(&stable, &devices).unwrap();
            assert_eq!(resolved.transient_ordinal, device.id);
            assert_eq!(resolved.stable_id.pci_bus_id, device.pci_bus_id.unwrap());
        }
    }
}
