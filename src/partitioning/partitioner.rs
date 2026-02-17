//! Partitioner trait and types.
//!
//! This module defines the partitioner interface for approximate search.

use crate::data_format::DatapointPtr;
use crate::error::Result;
use crate::types::{DatapointIndex, DatapointValue};
use serde::{Deserialize, Serialize};

/// Result of partitioning a query.
#[derive(Debug, Clone)]
pub struct PartitionResult {
    /// Token (partition) indices to search.
    pub tokens: Vec<u32>,

    /// Distances to partition centers.
    pub distances: Vec<f32>,

    /// Number of datapoints in each partition.
    pub partition_sizes: Vec<usize>,
}

impl PartitionResult {
    /// Create a new partition result.
    pub fn new(tokens: Vec<u32>, distances: Vec<f32>) -> Self {
        Self {
            tokens,
            distances,
            partition_sizes: Vec::new(),
        }
    }

    /// Create a result with partition sizes.
    pub fn with_sizes(tokens: Vec<u32>, distances: Vec<f32>, sizes: Vec<usize>) -> Self {
        Self {
            tokens,
            distances,
            partition_sizes: sizes,
        }
    }

    /// Get the number of partitions.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Get the top partition.
    pub fn top(&self) -> Option<(u32, f32)> {
        if self.tokens.is_empty() {
            None
        } else {
            Some((self.tokens[0], self.distances[0]))
        }
    }
}

/// Configuration for partitioner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionerConfig {
    /// Number of partitions.
    pub num_partitions: usize,

    /// Number of partitions to search per query.
    pub num_partitions_to_search: usize,

    /// Enable spilling (search multiple related partitions).
    pub spilling_enabled: bool,

    /// Spilling threshold (ratio of extra partitions).
    pub spilling_threshold: f32,

    /// Maximum spilling factor.
    pub max_spill_factor: f32,
}

impl Default for PartitionerConfig {
    fn default() -> Self {
        Self {
            num_partitions: 100,
            num_partitions_to_search: 10,
            spilling_enabled: false,
            spilling_threshold: 0.0,
            max_spill_factor: 2.0,
        }
    }
}

impl PartitionerConfig {
    /// Create a new configuration.
    pub fn new(num_partitions: usize) -> Self {
        Self {
            num_partitions,
            ..Default::default()
        }
    }

    /// Set the number of partitions to search.
    pub fn with_partitions_to_search(mut self, n: usize) -> Self {
        self.num_partitions_to_search = n;
        self
    }

    /// Enable spilling.
    pub fn with_spilling(mut self, threshold: f32) -> Self {
        self.spilling_enabled = true;
        self.spilling_threshold = threshold;
        self
    }
}

/// Trait for partitioners.
pub trait Partitioner<T: DatapointValue>: Send + Sync {
    /// Partition a query point and return the partitions to search.
    fn partition(&self, query: &DatapointPtr<'_, T>, num_partitions: usize) -> Result<PartitionResult>;

    /// Get the datapoint indices in a partition.
    fn partition_indices(&self, partition_id: u32) -> Option<&[DatapointIndex]>;

    /// Get the total number of partitions.
    fn num_partitions(&self) -> usize;

    /// Get the number of datapoints.
    fn num_datapoints(&self) -> usize;
}

/// Database tokenization result (partition assignment for dataset).
#[derive(Debug, Clone)]
pub struct DatabaseTokenization {
    /// Partition assignment for each datapoint.
    pub assignments: Vec<u32>,

    /// Datapoint indices in each partition.
    pub partition_to_indices: Vec<Vec<DatapointIndex>>,

    /// Partition centers.
    pub centers: Vec<Vec<f32>>,
}

impl DatabaseTokenization {
    /// Create a new tokenization.
    pub fn new(
        assignments: Vec<u32>,
        partition_to_indices: Vec<Vec<DatapointIndex>>,
        centers: Vec<Vec<f32>>,
    ) -> Self {
        Self {
            assignments,
            partition_to_indices,
            centers,
        }
    }

    /// Get the number of partitions.
    pub fn num_partitions(&self) -> usize {
        self.partition_to_indices.len()
    }

    /// Get the number of datapoints.
    pub fn num_datapoints(&self) -> usize {
        self.assignments.len()
    }

    /// Get the partition for a datapoint.
    pub fn partition_for(&self, index: DatapointIndex) -> Option<u32> {
        self.assignments.get(index as usize).copied()
    }

    /// Get the indices in a partition.
    pub fn indices_in_partition(&self, partition_id: u32) -> Option<&[DatapointIndex]> {
        self.partition_to_indices.get(partition_id as usize).map(|v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_result() {
        let result = PartitionResult::new(
            vec![0, 1, 2],
            vec![0.1, 0.2, 0.3],
        );

        assert_eq!(result.len(), 3);
        assert_eq!(result.top(), Some((0, 0.1)));
    }

    #[test]
    fn test_partitioner_config() {
        let config = PartitionerConfig::new(100)
            .with_partitions_to_search(10)
            .with_spilling(0.2);

        assert_eq!(config.num_partitions, 100);
        assert_eq!(config.num_partitions_to_search, 10);
        assert!(config.spilling_enabled);
        assert_eq!(config.spilling_threshold, 0.2);
    }
}
