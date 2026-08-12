use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Distribution {
    pub samples: usize,
    pub median_ns: u64,
    pub p95_ns: u64,
    pub mad_ns: u64,
    pub bootstrap_median_ci95_ns: [u64; 2],
    pub min_ns: u64,
    pub max_ns: u64,
}

impl Distribution {
    pub fn from_samples(samples: &[u64], seed: u64) -> Self {
        assert!(!samples.is_empty());
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let median = percentile(&sorted, 0.5);
        let deviations = sorted
            .iter()
            .map(|sample| sample.abs_diff(median))
            .collect::<Vec<_>>();
        let mut deviations = deviations;
        deviations.sort_unstable();

        let mut random = ChaCha8Rng::seed_from_u64(seed);
        let mut medians = Vec::with_capacity(10_000);
        let mut resample = vec![0_u64; samples.len()];
        for _ in 0..10_000 {
            for value in &mut resample {
                *value = samples[random.gen_range(0..samples.len())];
            }
            resample.sort_unstable();
            medians.push(percentile(&resample, 0.5));
        }
        medians.sort_unstable();
        Self {
            samples: samples.len(),
            median_ns: median,
            p95_ns: percentile(&sorted, 0.95),
            mad_ns: percentile(&deviations, 0.5),
            bootstrap_median_ci95_ns: [percentile(&medians, 0.025), percentile(&medians, 0.975)],
            min_ns: sorted[0],
            max_ns: *sorted.last().expect("non-empty"),
        }
    }
}

fn percentile(sorted: &[u64], percentile: f64) -> u64 {
    let index = ((sorted.len() - 1) as f64 * percentile).ceil() as usize;
    sorted[index.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_is_deterministic_and_bounded() {
        let values = [10, 11, 12, 13, 100];
        let first = Distribution::from_samples(&values, 42);
        let second = Distribution::from_samples(&values, 42);
        assert_eq!(first.median_ns, 12);
        assert_eq!(first.p95_ns, 100);
        assert_eq!(
            first.bootstrap_median_ci95_ns,
            second.bootstrap_median_ci95_ns
        );
        assert!(first.bootstrap_median_ci95_ns[0] <= first.median_ns);
        assert!(first.bootstrap_median_ci95_ns[1] >= first.median_ns);
    }
}
