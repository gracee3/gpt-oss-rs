//! Capability-level CPU feature detection and kernel requirements.
//!
//! Detection is cached once per process.  The public description deliberately
//! names instruction-set capabilities rather than processor generations so
//! dispatch remains valid on future and non-Intel implementations.

use std::fmt;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub fma: bool,
    pub avx_vnni: bool,
    pub avx512_f: bool,
    pub avx512_dq: bool,
    pub avx512_ifma: bool,
    pub avx512_cd: bool,
    pub avx512_bw: bool,
    pub avx512_vl: bool,
    pub avx512_vbmi: bool,
    pub avx512_vbmi2: bool,
    pub avx512_vnni: bool,
    pub avx512_bitalg: bool,
    pub avx512_vpopcntdq: bool,
    pub avx512_bf16: bool,
    pub avx512_fp16: bool,
    pub amx_tile: bool,
    pub amx_int8: bool,
    pub amx_bf16: bool,
    pub amx_fp16: bool,
    pub amx_complex: bool,
    pub avx10_1: bool,
    pub avx10_2: bool,
    pub avx10_128: bool,
    pub avx10_256: bool,
    pub avx10_512: bool,
}

impl CpuFeatures {
    pub const NONE: Self = Self {
        avx2: false,
        fma: false,
        avx_vnni: false,
        avx512_f: false,
        avx512_dq: false,
        avx512_ifma: false,
        avx512_cd: false,
        avx512_bw: false,
        avx512_vl: false,
        avx512_vbmi: false,
        avx512_vbmi2: false,
        avx512_vnni: false,
        avx512_bitalg: false,
        avx512_vpopcntdq: false,
        avx512_bf16: false,
        avx512_fp16: false,
        amx_tile: false,
        amx_int8: false,
        amx_bf16: false,
        amx_fp16: false,
        amx_complex: false,
        avx10_1: false,
        avx10_2: false,
        avx10_128: false,
        avx10_256: false,
        avx10_512: false,
    };

    /// Return the process-wide feature snapshot.
    pub fn detect() -> Self {
        static FEATURES: OnceLock<CpuFeatures> = OnceLock::new();
        *FEATURES.get_or_init(detect_features)
    }

    pub const fn supports(self, requirements: KernelRequirements) -> bool {
        requirements.is_satisfied_by(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KernelRequirements(u64);

impl KernelRequirements {
    pub const NONE: Self = Self(0);
    pub const AVX2: Self = Self(1 << 0);
    pub const FMA: Self = Self(1 << 1);
    pub const AVX_VNNI: Self = Self(1 << 2);
    pub const AVX512_F: Self = Self(1 << 3);
    pub const AVX512_DQ: Self = Self(1 << 4);
    pub const AVX512_IFMA: Self = Self(1 << 5);
    pub const AVX512_CD: Self = Self(1 << 6);
    pub const AVX512_BW: Self = Self(1 << 7);
    pub const AVX512_VL: Self = Self(1 << 8);
    pub const AVX512_VBMI: Self = Self(1 << 9);
    pub const AVX512_VBMI2: Self = Self(1 << 10);
    pub const AVX512_VNNI: Self = Self(1 << 11);
    pub const AVX512_BITALG: Self = Self(1 << 12);
    pub const AVX512_VPOPCNTDQ: Self = Self(1 << 13);
    pub const AVX512_BF16: Self = Self(1 << 14);
    pub const AVX512_FP16: Self = Self(1 << 15);
    pub const AMX_TILE: Self = Self(1 << 16);
    pub const AMX_INT8: Self = Self(1 << 17);
    pub const AMX_BF16: Self = Self(1 << 18);
    pub const AMX_FP16: Self = Self(1 << 19);
    pub const AMX_COMPLEX: Self = Self(1 << 20);
    pub const AVX10_1: Self = Self(1 << 21);
    pub const AVX10_2: Self = Self(1 << 22);
    pub const AVX10_128: Self = Self(1 << 23);
    pub const AVX10_256: Self = Self(1 << 24);
    pub const AVX10_512: Self = Self(1 << 25);

    pub const AVX2_FMA: Self = Self(Self::AVX2.0 | Self::FMA.0);
    pub const AVX2_MXFP4: Self = Self::AVX2;
    pub const AVX512_BF16_MATVEC: Self = Self(Self::AVX512_F.0 | Self::AVX512_BW.0);
    pub const AVX512_QUANTIZE_Q8: Self = Self::AVX512_F;
    pub const AVX512_MXFP4_VNNI: Self =
        Self(Self::AVX2.0 | Self::AVX512_VL.0 | Self::AVX512_VNNI.0);
    pub const AVX512_RMS_NORM: Self = Self::AVX512_F;
    pub const AVX512_VNNI_PATH: Self = Self(
        Self::AVX2.0
            | Self::AVX512_F.0
            | Self::AVX512_BW.0
            | Self::AVX512_VL.0
            | Self::AVX512_VNNI.0,
    );

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn is_satisfied_by(self, features: CpuFeatures) -> bool {
        (self.0 & Self::AVX2.0 == 0 || features.avx2)
            && (self.0 & Self::FMA.0 == 0 || features.fma)
            && (self.0 & Self::AVX_VNNI.0 == 0 || features.avx_vnni)
            && (self.0 & Self::AVX512_F.0 == 0 || features.avx512_f)
            && (self.0 & Self::AVX512_DQ.0 == 0 || features.avx512_dq)
            && (self.0 & Self::AVX512_IFMA.0 == 0 || features.avx512_ifma)
            && (self.0 & Self::AVX512_CD.0 == 0 || features.avx512_cd)
            && (self.0 & Self::AVX512_BW.0 == 0 || features.avx512_bw)
            && (self.0 & Self::AVX512_VL.0 == 0 || features.avx512_vl)
            && (self.0 & Self::AVX512_VBMI.0 == 0 || features.avx512_vbmi)
            && (self.0 & Self::AVX512_VBMI2.0 == 0 || features.avx512_vbmi2)
            && (self.0 & Self::AVX512_VNNI.0 == 0 || features.avx512_vnni)
            && (self.0 & Self::AVX512_BITALG.0 == 0 || features.avx512_bitalg)
            && (self.0 & Self::AVX512_VPOPCNTDQ.0 == 0 || features.avx512_vpopcntdq)
            && (self.0 & Self::AVX512_BF16.0 == 0 || features.avx512_bf16)
            && (self.0 & Self::AVX512_FP16.0 == 0 || features.avx512_fp16)
            && (self.0 & Self::AMX_TILE.0 == 0 || features.amx_tile)
            && (self.0 & Self::AMX_INT8.0 == 0 || features.amx_int8)
            && (self.0 & Self::AMX_BF16.0 == 0 || features.amx_bf16)
            && (self.0 & Self::AMX_FP16.0 == 0 || features.amx_fp16)
            && (self.0 & Self::AMX_COMPLEX.0 == 0 || features.amx_complex)
            && (self.0 & Self::AVX10_1.0 == 0 || features.avx10_1)
            && (self.0 & Self::AVX10_2.0 == 0 || features.avx10_2)
            && (self.0 & Self::AVX10_128.0 == 0 || features.avx10_128)
            && (self.0 & Self::AVX10_256.0 == 0 || features.avx10_256)
            && (self.0 & Self::AVX10_512.0 == 0 || features.avx10_512)
    }
}

impl fmt::Display for KernelRequirements {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 0 {
            return formatter.write_str("scalar");
        }
        let capabilities = [
            (Self::AVX2, "avx2"),
            (Self::FMA, "fma"),
            (Self::AVX_VNNI, "avx-vnni"),
            (Self::AVX512_F, "avx512f"),
            (Self::AVX512_DQ, "avx512dq"),
            (Self::AVX512_IFMA, "avx512ifma"),
            (Self::AVX512_CD, "avx512cd"),
            (Self::AVX512_BW, "avx512bw"),
            (Self::AVX512_VL, "avx512vl"),
            (Self::AVX512_VBMI, "avx512vbmi"),
            (Self::AVX512_VBMI2, "avx512vbmi2"),
            (Self::AVX512_VNNI, "avx512vnni"),
            (Self::AVX512_BITALG, "avx512bitalg"),
            (Self::AVX512_VPOPCNTDQ, "avx512vpopcntdq"),
            (Self::AVX512_BF16, "avx512bf16"),
            (Self::AVX512_FP16, "avx512fp16"),
            (Self::AMX_TILE, "amx-tile"),
            (Self::AMX_INT8, "amx-int8"),
            (Self::AMX_BF16, "amx-bf16"),
            (Self::AMX_FP16, "amx-fp16"),
            (Self::AMX_COMPLEX, "amx-complex"),
            (Self::AVX10_1, "avx10.1"),
            (Self::AVX10_2, "avx10.2"),
            (Self::AVX10_128, "avx10-128"),
            (Self::AVX10_256, "avx10-256"),
            (Self::AVX10_512, "avx10-512"),
        ];
        let mut separator = "";
        for (bit, name) in capabilities {
            if self.0 & bit.0 != 0 {
                formatter.write_str(separator)?;
                formatter.write_str(name)?;
                separator = "+";
            }
        }
        Ok(())
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_features() -> CpuFeatures {
    use std::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};

    // CPUID is available on x86-64. XGETBV is executed only when the OSXSAVE
    // bit says the instruction and OS-managed XCR0 are present.
    let leaf0 = __cpuid(0);
    let leaf1 = __cpuid(1);
    let osxsave = leaf1.ecx & (1 << 27) != 0;
    let xcr0 = if osxsave {
        // SAFETY: guarded by OSXSAVE above.
        unsafe { _xgetbv(0) }
    } else {
        0
    };
    let avx_state = leaf1.ecx & (1 << 28) != 0 && xcr0 & 0b110 == 0b110;
    let avx512_state = avx_state && xcr0 & 0b1110_0000 == 0b1110_0000;
    let amx_state = xcr0 & ((1 << 17) | (1 << 18)) == ((1 << 17) | (1 << 18));

    if leaf0.eax < 7 {
        return CpuFeatures {
            fma: avx_state && leaf1.ecx & (1 << 12) != 0,
            ..CpuFeatures::NONE
        };
    }

    let leaf7_0 = __cpuid_count(7, 0);
    let leaf7_1 = if leaf7_0.eax >= 1 {
        Some(__cpuid_count(7, 1))
    } else {
        None
    };
    let avx10_present = leaf7_1.is_some_and(|leaf| leaf.edx & (1 << 19) != 0);
    let avx10 = if avx10_present && leaf0.eax >= 0x24 {
        Some(__cpuid_count(0x24, 0))
    } else {
        None
    };
    let avx10_version = avx10.map_or(0, |leaf| leaf.ebx & 0xff);

    CpuFeatures {
        avx2: avx_state && leaf7_0.ebx & (1 << 5) != 0,
        fma: avx_state && leaf1.ecx & (1 << 12) != 0,
        avx_vnni: avx_state && leaf7_1.is_some_and(|leaf| leaf.eax & (1 << 4) != 0),
        avx512_f: avx512_state && leaf7_0.ebx & (1 << 16) != 0,
        avx512_dq: avx512_state && leaf7_0.ebx & (1 << 17) != 0,
        avx512_ifma: avx512_state && leaf7_0.ebx & (1 << 21) != 0,
        avx512_cd: avx512_state && leaf7_0.ebx & (1 << 28) != 0,
        avx512_bw: avx512_state && leaf7_0.ebx & (1 << 30) != 0,
        avx512_vl: avx512_state && leaf7_0.ebx & (1 << 31) != 0,
        avx512_vbmi: avx512_state && leaf7_0.ecx & (1 << 1) != 0,
        avx512_vbmi2: avx512_state && leaf7_0.ecx & (1 << 6) != 0,
        avx512_vnni: avx512_state && leaf7_0.ecx & (1 << 11) != 0,
        avx512_bitalg: avx512_state && leaf7_0.ecx & (1 << 12) != 0,
        avx512_vpopcntdq: avx512_state && leaf7_0.ecx & (1 << 14) != 0,
        avx512_bf16: avx512_state && leaf7_1.is_some_and(|leaf| leaf.eax & (1 << 5) != 0),
        avx512_fp16: avx512_state && leaf7_0.edx & (1 << 23) != 0,
        amx_tile: amx_state && leaf7_0.edx & (1 << 24) != 0,
        amx_int8: amx_state && leaf7_0.edx & (1 << 25) != 0,
        amx_bf16: amx_state && leaf7_0.edx & (1 << 22) != 0,
        amx_fp16: amx_state && leaf7_1.is_some_and(|leaf| leaf.eax & (1 << 21) != 0),
        amx_complex: amx_state && leaf7_1.is_some_and(|leaf| leaf.edx & (1 << 8) != 0),
        avx10_1: avx_state && avx10_version >= 1,
        avx10_2: avx_state && avx10_version >= 2,
        avx10_128: avx_state && avx10.is_some_and(|leaf| leaf.ebx & (1 << 16) != 0),
        avx10_256: avx_state && avx10.is_some_and(|leaf| leaf.ebx & (1 << 17) != 0),
        avx10_512: avx512_state && avx10.is_some_and(|leaf| leaf.ebx & (1 << 18) != 0),
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_features() -> CpuFeatures {
    CpuFeatures::NONE
}
