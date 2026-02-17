//! Random sampling utilities.

use rand::prelude::*;
use rand::seq::SliceRandom;

/// Random sampler for selecting indices.
pub struct RandomSampler {
    rng: StdRng,
}

impl RandomSampler {
    /// Create a new sampler with a random seed.
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
        }
    }

    /// Create a new sampler with a specific seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample k unique indices from [0, n).
    pub fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        if k >= n {
            return (0..n).collect();
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.partial_shuffle(&mut self.rng, k);
        indices.truncate(k);
        indices
    }

    /// Sample with replacement.
    pub fn sample_with_replacement(&mut self, n: usize, k: usize) -> Vec<usize> {
        (0..k).map(|_| self.rng.gen_range(0..n)).collect()
    }

    /// Get a random float in [0, 1).
    pub fn random_f32(&mut self) -> f32 {
        self.rng.gen()
    }

    /// Get a random float in [low, high).
    pub fn random_range(&mut self, low: f32, high: f32) -> f32 {
        self.rng.gen_range(low..high)
    }
}

impl Default for RandomSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Sample k unique indices from [0, n) using a random seed.
pub fn sample_indices(n: usize, k: usize, seed: Option<u64>) -> Vec<usize> {
    let mut sampler = match seed {
        Some(s) => RandomSampler::with_seed(s),
        None => RandomSampler::new(),
    };
    sampler.sample_indices(n, k)
}

/// Reservoir sampling for streaming data.
pub struct ReservoirSampler<T> {
    samples: Vec<T>,
    capacity: usize,
    count: usize,
    rng: StdRng,
}

impl<T: Clone> ReservoirSampler<T> {
    /// Create a new reservoir sampler.
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
            count: 0,
            rng: StdRng::from_entropy(),
        }
    }

    /// Create a sampler with a specific seed.
    pub fn with_seed(capacity: usize, seed: u64) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
            count: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Add an item to the sample.
    pub fn add(&mut self, item: T) {
        if self.samples.len() < self.capacity {
            self.samples.push(item);
        } else {
            let j = self.rng.gen_range(0..=self.count);
            if j < self.capacity {
                self.samples[j] = item;
            }
        }
        self.count += 1;
    }

    /// Get the current samples.
    pub fn samples(&self) -> &[T] {
        &self.samples
    }

    /// Take the samples.
    pub fn take(self) -> Vec<T> {
        self.samples
    }

    /// Get the number of items seen.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Clear the sampler.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_indices() {
        let mut sampler = RandomSampler::with_seed(42);
        let indices = sampler.sample_indices(100, 10);

        assert_eq!(indices.len(), 10);
        // Check uniqueness
        let mut sorted = indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 10);
    }

    #[test]
    fn test_sample_indices_k_greater_than_n() {
        let mut sampler = RandomSampler::with_seed(42);
        let indices = sampler.sample_indices(5, 10);

        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_reservoir_sampler() {
        let mut sampler = ReservoirSampler::with_seed(10, 42);

        for i in 0..100 {
            sampler.add(i);
        }

        assert_eq!(sampler.samples().len(), 10);
        assert_eq!(sampler.count(), 100);
    }

    #[test]
    fn test_sample_with_replacement() {
        let mut sampler = RandomSampler::with_seed(42);
        let indices = sampler.sample_with_replacement(10, 100);

        assert_eq!(indices.len(), 100);
        for &i in &indices {
            assert!(i < 10);
        }
    }
}
