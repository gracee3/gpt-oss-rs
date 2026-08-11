//! AMX-INT8 capability, Linux XSTATE, and process-permission diagnostics.

use std::fmt;
use std::sync::OnceLock;

use thiserror::Error;

#[cfg(all(feature = "amx-int8", target_os = "linux", target_arch = "x86_64"))]
const XFEATURE_XTILECFG: u32 = 17;
#[cfg(all(feature = "amx-int8", target_os = "linux", target_arch = "x86_64"))]
const XFEATURE_XTILEDATA: u32 = 18;
#[cfg(all(feature = "amx-int8", target_os = "linux", target_arch = "x86_64"))]
const XFEATURE_XTILECFG_MASK: u64 = 1 << XFEATURE_XTILECFG;
#[cfg(all(feature = "amx-int8", target_os = "linux", target_arch = "x86_64"))]
const XFEATURE_XTILEDATA_MASK: u64 = 1 << XFEATURE_XTILEDATA;
#[cfg(all(feature = "amx-int8", target_os = "linux", target_arch = "x86_64"))]
const XFEATURE_AMX_MASK: u64 = XFEATURE_XTILECFG_MASK | XFEATURE_XTILEDATA_MASK;

/// Independent gates for explicit AMX-INT8 execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmxRuntimeStatus {
    pub build_support: bool,
    pub supported_target: bool,
    pub hardware_tile: bool,
    pub hardware_int8: bool,
    /// `None` means the Linux XSTATE query itself failed or was unavailable.
    pub kernel_xstate_support: Option<bool>,
    /// `None` means the process-permission query itself failed or was unavailable.
    pub tile_data_permission: Option<bool>,
}

impl AmxRuntimeStatus {
    pub fn detect() -> Self {
        SystemAmxProbe.status()
    }

    pub const fn is_ready(self) -> bool {
        self.build_support
            && self.supported_target
            && self.hardware_tile
            && self.hardware_int8
            && matches!(self.kernel_xstate_support, Some(true))
            && matches!(self.tile_data_permission, Some(true))
    }
}

impl fmt::Display for AmxRuntimeStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "build={}, target={}, cpuid_tile={}, cpuid_int8={}, kernel_xstate={}, permission={}",
            self.build_support,
            self.supported_target,
            self.hardware_tile,
            self.hardware_int8,
            optional_gate(self.kernel_xstate_support),
            optional_gate(self.tile_data_permission)
        )
    }
}

fn optional_gate(gate: Option<bool>) -> &'static str {
    match gate {
        Some(true) => "available",
        Some(false) => "unavailable",
        None => "query-failed",
    }
}

/// Precise unavailable gate for forced AMX initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AmxRuntimeError {
    #[error("the amx-int8 Cargo feature is not enabled")]
    BuildFeatureAbsent,
    #[error("AMX-INT8 native execution requires Linux x86-64")]
    UnsupportedTarget,
    #[error("CPUID does not report AMX-TILE")]
    HardwareTileAbsent,
    #[error("CPUID does not report AMX-INT8")]
    HardwareInt8Absent,
    #[error("Linux does not expose AMX tile state through XSTATE")]
    KernelXstateAbsent,
    #[error("the Linux AMX XSTATE support query failed")]
    KernelXstateQueryFailed,
    #[error("the Linux AMX tile-data permission request was denied")]
    PermissionDenied,
    #[error("the Linux AMX tile-data permission query failed")]
    PermissionQueryFailed,
}

impl AmxRuntimeError {
    pub const fn reason(self) -> &'static str {
        match self {
            Self::BuildFeatureAbsent => "the amx-int8 Cargo feature is not enabled",
            Self::UnsupportedTarget => "AMX-INT8 native execution requires Linux x86-64",
            Self::HardwareTileAbsent => "CPUID does not report AMX-TILE",
            Self::HardwareInt8Absent => "CPUID does not report AMX-INT8",
            Self::KernelXstateAbsent => "Linux does not expose AMX tile state through XSTATE",
            Self::KernelXstateQueryFailed => "the Linux AMX XSTATE support query failed",
            Self::PermissionDenied => "the Linux AMX tile-data permission request was denied",
            Self::PermissionQueryFailed => "the Linux AMX tile-data permission query failed",
        }
    }
}

trait AmxProbe {
    fn status(&self) -> AmxRuntimeStatus;
    fn request_tile_data_permission(&self) -> bool;
}

struct SystemAmxProbe;

impl AmxProbe for SystemAmxProbe {
    fn status(&self) -> AmxRuntimeStatus {
        let (hardware_tile, hardware_int8) = raw_amx_cpuid();
        let (kernel_xstate_support, tile_data_permission) = linux_xstate_status();
        AmxRuntimeStatus {
            build_support: cfg!(feature = "amx-int8"),
            supported_target: cfg!(all(target_os = "linux", target_arch = "x86_64")),
            hardware_tile,
            hardware_int8,
            kernel_xstate_support,
            tile_data_permission,
        }
    }

    fn request_tile_data_permission(&self) -> bool {
        request_linux_tile_data_permission()
    }
}

/// Request process AMX tile-data permission after checking every earlier gate.
/// Callers must invoke this before constructing worker threads.
pub fn initialize_amx_int8() -> Result<AmxRuntimeStatus, AmxRuntimeError> {
    static INITIALIZED: OnceLock<Result<AmxRuntimeStatus, AmxRuntimeError>> = OnceLock::new();
    *INITIALIZED.get_or_init(|| initialize_with(&SystemAmxProbe))
}

#[cfg(all(feature = "amx-int8", target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn execute_amx_int8_tile(
    rows: usize,
    a_panel: &[u8],
    b_panel: &[u8],
    c_tile: &mut [u8],
) -> Result<(), crate::KernelError> {
    if rows == 0
        || rows > 16
        || a_panel.len() < 16 * 32
        || b_panel.len() < 8 * 64
        || c_tile.len() < 16 * 16 * 4
        || !(a_panel.as_ptr() as usize).is_multiple_of(64)
        || !(b_panel.as_ptr() as usize).is_multiple_of(64)
        || !(c_tile.as_ptr() as usize).is_multiple_of(64)
    {
        return Err(crate::KernelError::InvalidDimensions(
            "invalid AMX-INT8 native tile buffers".into(),
        ));
    }

    unsafe extern "C" {
        fn gpt_oss_amx_int8_tile(a: *const i8, b: *const i8, c: *mut i32, rows: u32) -> i32;
    }

    // SAFETY: the checks above establish the fixed A/B/C extents, alignment,
    // and row bound. Explicit runtime initialization establishes CPUID,
    // kernel XSTATE, and process permission before this function is reached.
    let status = unsafe {
        gpt_oss_amx_int8_tile(
            a_panel.as_ptr().cast::<i8>(),
            b_panel.as_ptr().cast::<i8>(),
            c_tile.as_mut_ptr().cast::<i32>(),
            rows as u32,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(crate::KernelError::AmxShim(status))
    }
}

#[cfg(not(all(feature = "amx-int8", target_os = "linux", target_arch = "x86_64")))]
pub(crate) fn execute_amx_int8_tile(
    _rows: usize,
    _a_panel: &[u8],
    _b_panel: &[u8],
    _c_tile: &mut [u8],
) -> Result<(), crate::KernelError> {
    Err(crate::KernelError::UnavailableMatmulBackend {
        backend: crate::Mxfp4MatmulBackend::AmxInt8,
        reason: "AMX-INT8 native execution requires Linux x86-64",
    })
}

fn initialize_with(probe: &impl AmxProbe) -> Result<AmxRuntimeStatus, AmxRuntimeError> {
    let status = probe.status();
    validate_pre_permission(status)?;
    match status.tile_data_permission {
        Some(true) => return Ok(status),
        None => return Err(AmxRuntimeError::PermissionQueryFailed),
        Some(false) => {}
    }
    if !probe.request_tile_data_permission() {
        return Err(AmxRuntimeError::PermissionDenied);
    }
    let status = probe.status();
    validate_pre_permission(status)?;
    match status.tile_data_permission {
        Some(true) => Ok(status),
        Some(false) => Err(AmxRuntimeError::PermissionDenied),
        None => Err(AmxRuntimeError::PermissionQueryFailed),
    }
}

fn validate_pre_permission(status: AmxRuntimeStatus) -> Result<(), AmxRuntimeError> {
    if !status.build_support {
        return Err(AmxRuntimeError::BuildFeatureAbsent);
    }
    if !status.supported_target {
        return Err(AmxRuntimeError::UnsupportedTarget);
    }
    if !status.hardware_tile {
        return Err(AmxRuntimeError::HardwareTileAbsent);
    }
    if !status.hardware_int8 {
        return Err(AmxRuntimeError::HardwareInt8Absent);
    }
    match status.kernel_xstate_support {
        Some(true) => Ok(()),
        Some(false) => Err(AmxRuntimeError::KernelXstateAbsent),
        None => Err(AmxRuntimeError::KernelXstateQueryFailed),
    }
}

#[cfg(target_arch = "x86_64")]
fn raw_amx_cpuid() -> (bool, bool) {
    use std::arch::x86_64::{__cpuid, __cpuid_count};

    let maximum = __cpuid(0).eax;
    if maximum < 7 {
        return (false, false);
    }
    let leaf = __cpuid_count(7, 0);
    (leaf.edx & (1 << 24) != 0, leaf.edx & (1 << 25) != 0)
}

#[cfg(not(target_arch = "x86_64"))]
const fn raw_amx_cpuid() -> (bool, bool) {
    (false, false)
}

#[cfg(all(feature = "amx-int8", target_os = "linux", target_arch = "x86_64"))]
fn linux_xstate_status() -> (Option<bool>, Option<bool>) {
    const ARCH_GET_XCOMP_SUPP: libc::c_ulong = 0x1021;
    const ARCH_GET_XCOMP_PERM: libc::c_ulong = 0x1022;

    let support = arch_prctl_mask(ARCH_GET_XCOMP_SUPP);
    let permission = arch_prctl_mask(ARCH_GET_XCOMP_PERM);
    (
        support.map(|mask| mask & XFEATURE_AMX_MASK == XFEATURE_AMX_MASK),
        permission.map(|mask| mask & XFEATURE_XTILEDATA_MASK != 0),
    )
}

#[cfg(not(all(feature = "amx-int8", target_os = "linux", target_arch = "x86_64")))]
const fn linux_xstate_status() -> (Option<bool>, Option<bool>) {
    (None, None)
}

#[cfg(all(feature = "amx-int8", target_os = "linux", target_arch = "x86_64"))]
fn arch_prctl_mask(code: libc::c_ulong) -> Option<u64> {
    let mut mask = 0_u64;
    // SAFETY: Linux arch_prctl GET operations write one u64 to the supplied
    // valid pointer and do not retain it.
    let result = unsafe { libc::syscall(libc::SYS_arch_prctl, code, std::ptr::addr_of_mut!(mask)) };
    (result == 0).then_some(mask)
}

#[cfg(all(feature = "amx-int8", target_os = "linux", target_arch = "x86_64"))]
fn request_linux_tile_data_permission() -> bool {
    const ARCH_REQ_XCOMP_PERM: libc::c_ulong = 0x1023;
    // SAFETY: Linux interprets the third syscall argument as the XFEATURE
    // number for this request and retains no pointer.
    unsafe {
        libc::syscall(
            libc::SYS_arch_prctl,
            ARCH_REQ_XCOMP_PERM,
            XFEATURE_XTILEDATA,
        ) == 0
    }
}

#[cfg(not(all(feature = "amx-int8", target_os = "linux", target_arch = "x86_64")))]
const fn request_linux_tile_data_permission() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    struct FakeProbe {
        status: Cell<AmxRuntimeStatus>,
        requested_status: AmxRuntimeStatus,
        request_succeeds: bool,
        requests: Cell<usize>,
    }

    impl FakeProbe {
        fn new(status: AmxRuntimeStatus) -> Self {
            Self {
                status: Cell::new(status),
                requested_status: status,
                request_succeeds: true,
                requests: Cell::new(0),
            }
        }
    }

    impl AmxProbe for FakeProbe {
        fn status(&self) -> AmxRuntimeStatus {
            self.status.get()
        }

        fn request_tile_data_permission(&self) -> bool {
            self.requests.set(self.requests.get() + 1);
            if self.request_succeeds {
                self.status.set(self.requested_status);
            }
            self.request_succeeds
        }
    }

    const READY: AmxRuntimeStatus = AmxRuntimeStatus {
        build_support: true,
        supported_target: true,
        hardware_tile: true,
        hardware_int8: true,
        kernel_xstate_support: Some(true),
        tile_data_permission: Some(true),
    };

    #[test]
    fn diagnostics_reject_each_gate_before_requesting_permission() {
        let cases = [
            (
                AmxRuntimeStatus {
                    build_support: false,
                    ..READY
                },
                AmxRuntimeError::BuildFeatureAbsent,
            ),
            (
                AmxRuntimeStatus {
                    hardware_tile: false,
                    ..READY
                },
                AmxRuntimeError::HardwareTileAbsent,
            ),
            (
                AmxRuntimeStatus {
                    hardware_int8: false,
                    ..READY
                },
                AmxRuntimeError::HardwareInt8Absent,
            ),
            (
                AmxRuntimeStatus {
                    kernel_xstate_support: Some(false),
                    ..READY
                },
                AmxRuntimeError::KernelXstateAbsent,
            ),
            (
                AmxRuntimeStatus {
                    kernel_xstate_support: None,
                    ..READY
                },
                AmxRuntimeError::KernelXstateQueryFailed,
            ),
        ];
        for (status, expected) in cases {
            let probe = FakeProbe::new(status);
            assert_eq!(initialize_with(&probe), Err(expected));
            assert_eq!(probe.requests.get(), 0);
        }
    }

    #[test]
    fn permission_is_requested_once_and_rechecked() {
        let mut probe = FakeProbe::new(AmxRuntimeStatus {
            tile_data_permission: Some(false),
            ..READY
        });
        probe.requested_status = READY;
        assert_eq!(initialize_with(&probe), Ok(READY));
        assert_eq!(probe.requests.get(), 1);
    }

    #[test]
    fn permission_denial_and_query_failure_are_distinct() {
        let mut denied = FakeProbe::new(AmxRuntimeStatus {
            tile_data_permission: Some(false),
            ..READY
        });
        denied.request_succeeds = false;
        assert_eq!(
            initialize_with(&denied),
            Err(AmxRuntimeError::PermissionDenied)
        );

        let failed = FakeProbe::new(AmxRuntimeStatus {
            tile_data_permission: None,
            ..READY
        });
        assert_eq!(
            initialize_with(&failed),
            Err(AmxRuntimeError::PermissionQueryFailed)
        );
        assert_eq!(failed.requests.get(), 0);
    }

    #[test]
    fn automatic_diagnostics_never_request_permission() {
        let status = AmxRuntimeStatus::detect();
        assert_eq!(status.build_support, cfg!(feature = "amx-int8"));
        assert!(status.to_string().contains("permission="));
    }
}
