//! Configuration types for ScaNN.
//!
//! This module provides configuration structures equivalent to the protobuf
//! definitions in the C++ implementation.

use serde::{Deserialize, Serialize};
use crate::distance_measures::DistanceMeasure;

/// Main configuration for ScaNN searcher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScannConfig {
    /// Number of neighbors to return.
    pub num_neighbors: u32,

    /// Distance measure to use.
    pub distance_measure: DistanceMeasure,

    /// Brute force configuration (optional).
    pub brute_force: Option<BruteForceConfig>,

    /// Partitioning configuration (optional).
    pub partitioning: Option<PartitioningConfig>,

    /// Hashing configuration (optional).
    pub hash: Option<HashConfig>,

    /// Exact reordering configuration (optional).
    pub exact_reordering: Option<ExactReorderingConfig>,
}

impl Default for ScannConfig {
    fn default() -> Self {
        Self {
            num_neighbors: 10,
            distance_measure: DistanceMeasure::SquaredL2,
            brute_force: None,
            partitioning: None,
            hash: None,
            exact_reordering: None,
        }
    }
}

impl ScannConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of neighbors to return.
    pub fn with_num_neighbors(mut self, k: u32) -> Self {
        self.num_neighbors = k;
        self
    }

    /// Set the distance measure.
    pub fn with_distance_measure(mut self, measure: DistanceMeasure) -> Self {
        self.distance_measure = measure;
        self
    }

    /// Configure for brute-force search.
    pub fn with_brute_force(mut self) -> Self {
        self.brute_force = Some(BruteForceConfig::default());
        self.partitioning = None;
        self.hash = None;
        self
    }

    /// Configure partitioning.
    pub fn with_partitioning(mut self, config: PartitioningConfig) -> Self {
        self.partitioning = Some(config);
        self
    }

    /// Configure hashing.
    pub fn with_hash(mut self, config: HashConfig) -> Self {
        self.hash = Some(config);
        self
    }

    /// Configure exact reordering.
    pub fn with_exact_reordering(mut self, config: ExactReorderingConfig) -> Self {
        self.exact_reordering = Some(config);
        self
    }

    /// Check if this is a brute-force configuration.
    pub fn is_brute_force(&self) -> bool {
        self.brute_force.is_some() && self.partitioning.is_none() && self.hash.is_none()
    }

    /// Check if partitioning is enabled.
    pub fn has_partitioning(&self) -> bool {
        self.partitioning.is_some()
    }

    /// Check if hashing is enabled.
    pub fn has_hashing(&self) -> bool {
        self.hash.is_some()
    }

    /// Check if exact reordering is enabled.
    pub fn has_reordering(&self) -> bool {
        self.exact_reordering.is_some()
    }
}

/// Configuration for brute-force search.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BruteForceConfig {
    /// Enable scalar quantization for memory efficiency.
    pub scalar_quantization: bool,

    /// Quantization bits (if scalar quantization is enabled).
    pub quantization_bits: u8,
}

impl BruteForceConfig {
    /// Create a new brute-force configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable scalar quantization.
    pub fn with_scalar_quantization(mut self, bits: u8) -> Self {
        self.scalar_quantization = true;
        self.quantization_bits = bits;
        self
    }
}

/// Configuration for partitioning (tree-based search).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitioningConfig {
    /// Number of partitions (K in K-means).
    pub num_partitions: u32,

    /// Number of partitions to search per query.
    pub num_partitions_to_search: u32,

    /// Maximum training iterations for K-means.
    pub max_training_iterations: u32,

    /// Convergence threshold for K-means.
    pub convergence_threshold: f32,

    /// Number of tree levels (1 = flat, >1 = hierarchical).
    pub num_levels: u32,

    /// Enable spilling (search multiple partitions).
    pub spilling: bool,

    /// Spilling threshold (fraction of extra partitions to search).
    pub spilling_threshold: f32,
}

impl Default for PartitioningConfig {
    fn default() -> Self {
        Self {
            num_partitions: 100,
            num_partitions_to_search: 10,
            max_training_iterations: 100,
            convergence_threshold: 1e-5,
            num_levels: 1,
            spilling: false,
            spilling_threshold: 0.0,
        }
    }
}

impl PartitioningConfig {
    /// Create a new partitioning configuration.
    pub fn new(num_partitions: u32) -> Self {
        Self {
            num_partitions,
            ..Default::default()
        }
    }

    /// Set the number of partitions to search.
    pub fn with_partitions_to_search(mut self, n: u32) -> Self {
        self.num_partitions_to_search = n;
        self
    }

    /// Enable spilling.
    pub fn with_spilling(mut self, threshold: f32) -> Self {
        self.spilling = true;
        self.spilling_threshold = threshold;
        self
    }

    /// Set the number of tree levels.
    pub fn with_levels(mut self, levels: u32) -> Self {
        self.num_levels = levels;
        self
    }
}

/// Configuration for asymmetric hashing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashConfig {
    /// Type of asymmetric hashing.
    pub hash_type: HashType,

    /// Number of hash buckets per dimension block.
    pub num_buckets: u32,

    /// Number of dimension blocks (chunks).
    pub num_blocks: u32,

    /// Lookup table format.
    pub lut_format: LutFormat,

    /// Training sample size.
    pub training_sample_size: usize,
}

impl Default for HashConfig {
    fn default() -> Self {
        Self {
            hash_type: HashType::AsymmetricHashing,
            num_buckets: 256,
            num_blocks: 16,
            lut_format: LutFormat::Int8,
            training_sample_size: 100_000,
        }
    }
}

impl HashConfig {
    /// Create a new hash configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the hash type.
    pub fn with_type(mut self, hash_type: HashType) -> Self {
        self.hash_type = hash_type;
        self
    }

    /// Set the number of buckets.
    pub fn with_buckets(mut self, buckets: u32) -> Self {
        self.num_buckets = buckets;
        self
    }

    /// Set the number of blocks.
    pub fn with_blocks(mut self, blocks: u32) -> Self {
        self.num_blocks = blocks;
        self
    }

    /// Set the LUT format.
    pub fn with_lut_format(mut self, format: LutFormat) -> Self {
        self.lut_format = format;
        self
    }
}

/// Type of hashing algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HashType {
    /// Standard asymmetric hashing.
    AsymmetricHashing,

    /// Product quantization.
    ProductQuantization,
}

/// Lookup table format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LutFormat {
    /// 8-bit integer lookup tables.
    Int8,

    /// 16-bit integer lookup tables.
    Int16,

    /// 32-bit float lookup tables.
    Float,
}

/// Configuration for exact reordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactReorderingConfig {
    /// Number of candidates to reorder.
    pub num_candidates: u32,

    /// Use quantized reordering.
    pub quantized: bool,
}

impl Default for ExactReorderingConfig {
    fn default() -> Self {
        Self {
            num_candidates: 100,
            quantized: false,
        }
    }
}

impl ExactReorderingConfig {
    /// Create a new reordering configuration.
    pub fn new(num_candidates: u32) -> Self {
        Self {
            num_candidates,
            ..Default::default()
        }
    }

    /// Enable quantized reordering.
    pub fn with_quantized(mut self) -> Self {
        self.quantized = true;
        self
    }
}

/// Search parameters that can be adjusted per-query.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct QueryConfig {
    /// Number of neighbors to return.
    pub num_neighbors: Option<u32>,

    /// Number of partitions to search (overrides default).
    pub num_partitions_to_search: Option<u32>,

    /// Number of candidates for reordering (overrides default).
    pub reordering_num_candidates: Option<u32>,

    /// Epsilon for approximate search.
    pub epsilon: Option<f32>,
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ScannConfig::default();
        assert_eq!(config.num_neighbors, 10);
        assert_eq!(config.distance_measure, DistanceMeasure::SquaredL2);
        assert!(!config.is_brute_force());
    }

    #[test]
    fn test_brute_force_config() {
        let config = ScannConfig::new()
            .with_num_neighbors(5)
            .with_brute_force();
        assert!(config.is_brute_force());
        assert!(!config.has_partitioning());
    }

    #[test]
    fn test_partitioning_config() {
        let part_config = PartitioningConfig::new(100)
            .with_partitions_to_search(10)
            .with_spilling(0.2);

        assert_eq!(part_config.num_partitions, 100);
        assert_eq!(part_config.num_partitions_to_search, 10);
        assert!(part_config.spilling);
        assert_eq!(part_config.spilling_threshold, 0.2);
    }

    #[test]
    fn test_config_serialization() {
        let config = ScannConfig::new()
            .with_num_neighbors(20)
            .with_distance_measure(DistanceMeasure::DotProduct);

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ScannConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.num_neighbors, 20);
        assert_eq!(deserialized.distance_measure, DistanceMeasure::DotProduct);
    }
}
