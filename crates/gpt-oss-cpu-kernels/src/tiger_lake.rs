use crate::{CpuFeatures, CpuHardwareIdentity, Mxfp4MatmulBackend};

/// Immutable, candidate-scoped Tiger Lake matrix promotion record.
///
/// The two records cover the exact GPT-OSS residual-Q8 gate/up and down
/// projection shapes. Everything outside these records resolves to scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mxfp4PromotionRegion {
    pub activation: &'static str,
    pub m_start: usize,
    pub m_end: usize,
    pub n: usize,
    pub k: usize,
    pub backend: Mxfp4MatmulBackend,
}

pub const TIGER_LAKE_PROFILE_KEY: &str =
    "GenuineIntel-family6-model140-stepping1-cores4-logical8-microcodebe-xcr02e7";
pub const TIGER_LAKE_THREAD_POLICY: usize = 4;
pub const TIGER_LAKE_MXFP4_PROMOTION_BENCHMARK_COMMIT: &str =
    "ea72bb19cc3dd3b6ebb4c810110553d167a61a4f";
pub const TIGER_LAKE_MXFP4_PROMOTION_EVIDENCE_SHA256: [&str; 5] = [
    "bdd877e5342e56eeed44cf156c9c775a6247acb38e0c3d78051f06ce072235eb",
    "bfe14062c2e08784304cec0926911ba3c4b974da8ed19fa5098861ae221ee2c6",
    "ed6913d3d36e6ecf97e251d644efc697c3661a49a8f5839629ec4ac9a12c2373",
    "e7b2293f73cc05f67c1feb13ec126ee748386098426b2fdad9a215907bd309ef",
    "f682dfd772900e76e7cce7118732e24cbd006f2f233ec00b2e747288cf0d6885",
];
pub const TIGER_LAKE_MXFP4_PROMOTION_REGIONS: [Mxfp4PromotionRegion; 2] = [
    Mxfp4PromotionRegion {
        activation: "residual-q8",
        m_start: 3,
        m_end: 3,
        n: 5_760,
        k: 2_880,
        backend: Mxfp4MatmulBackend::Avx2,
    },
    Mxfp4PromotionRegion {
        activation: "residual-q8",
        m_start: 3,
        m_end: 3,
        n: 2_880,
        k: 2_880,
        backend: Mxfp4MatmulBackend::Avx2,
    },
];

pub fn tiger_lake_profile_matches(
    identity: &CpuHardwareIdentity,
    features: CpuFeatures,
    threads: usize,
) -> bool {
    threads == TIGER_LAKE_THREAD_POLICY
        && identity.profile_key() == TIGER_LAKE_PROFILE_KEY
        && features.avx2
        && features.fma
}

pub fn tiger_lake_auto_matmul_backend(
    profile_matches: bool,
    residual_q8: bool,
    m: usize,
    n: usize,
    k: usize,
) -> Mxfp4MatmulBackend {
    if m == 1 {
        return Mxfp4MatmulBackend::Auto;
    }
    if profile_matches && residual_q8 {
        for region in TIGER_LAKE_MXFP4_PROMOTION_REGIONS {
            if m >= region.m_start && m <= region.m_end && n == region.n && k == region.k {
                return region.backend;
            }
        }
    }
    Mxfp4MatmulBackend::Scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> CpuHardwareIdentity {
        CpuHardwareIdentity {
            vendor: "GenuineIntel".into(),
            family: 6,
            model: 140,
            stepping: 1,
            xcr0: 0x2e7,
            osxsave: true,
            physical_cores: 4,
            logical_cpus: 8,
            microcode: Some(0xbe),
        }
    }

    #[test]
    fn profile_matching_is_normalized_and_exact() {
        let features = CpuFeatures {
            avx2: true,
            fma: true,
            ..CpuFeatures::NONE
        };
        assert!(tiger_lake_profile_matches(&identity(), features, 4));
        assert!(!tiger_lake_profile_matches(&identity(), features, 3));
        let mut wrong = identity();
        wrong.microcode = Some(0xbd);
        assert!(!tiger_lake_profile_matches(&wrong, features, 4));
    }

    #[test]
    fn promotion_is_narrow_and_scalar_everywhere_else() {
        for n in [2_880, 5_760] {
            assert_eq!(
                tiger_lake_auto_matmul_backend(true, true, 3, n, 2_880),
                Mxfp4MatmulBackend::Avx2
            );
        }
        assert_eq!(
            tiger_lake_auto_matmul_backend(true, true, 1, 5_760, 2_880),
            Mxfp4MatmulBackend::Auto
        );
        for (matched, residual, m, n, k) in [
            (false, true, 3, 5_760, 2_880),
            (true, false, 3, 5_760, 2_880),
            (true, true, 2, 5_760, 2_880),
            (true, true, 4, 5_760, 2_880),
            (true, true, 3, 5_759, 2_880),
            (true, true, 3, 5_760, 2_912),
        ] {
            assert_eq!(
                tiger_lake_auto_matmul_backend(matched, residual, m, n, k),
                Mxfp4MatmulBackend::Scalar
            );
        }
    }
}
