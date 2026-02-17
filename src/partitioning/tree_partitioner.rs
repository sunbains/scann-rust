//! Tree-based partitioner using K-means trees.

use crate::data_format::{Dataset, DenseDataset, DatapointPtr};
use crate::error::{Result, ScannError};
use crate::partitioning::partitioner::{DatabaseTokenization, PartitionResult, Partitioner, PartitionerConfig};
use crate::trees::kmeans::{KMeans, KMeansConfig};
use crate::trees::kmeans_tree::{KMeansTree, KMeansTreeConfig};
use crate::types::{DatapointIndex, DatapointValue};
use ordered_float::OrderedFloat;

/// Helper to convert to f32 without ambiguity with NumCast::to_f32
#[inline(always)]
fn val_to_f32<T: DatapointValue>(v: T) -> f32 {
    DatapointValue::to_f32(v)
}

/// Tree-based partitioner using K-means.
pub struct TreePartitioner {
    /// K-means tree for hierarchical partitioning.
    tree: Option<KMeansTree>,

    /// Database tokenization (partition assignments).
    tokenization: Option<DatabaseTokenization>,

    /// Configuration.
    config: PartitionerConfig,

    /// Centers for flat partitioning.
    centers: Vec<Vec<f32>>,

    /// Dimensionality.
    dimensionality: usize,
}

impl TreePartitioner {
    /// Create a new tree partitioner.
    pub fn new(config: PartitionerConfig) -> Self {
        Self {
            tree: None,
            tokenization: None,
            config,
            centers: Vec::new(),
            dimensionality: 0,
        }
    }

    /// Build the partitioner from a dataset.
    pub fn build<T: DatapointValue + Sync>(&mut self, dataset: &DenseDataset<T>) -> Result<()> {
        if dataset.is_empty() {
            return Err(ScannError::invalid_argument("Cannot partition empty dataset"));
        }

        self.dimensionality = dataset.dimensionality() as usize;
        let n = dataset.size();
        let k = self.config.num_partitions.min(n);

        // Convert dataset to f32
        let data: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                dataset
                    .get(i as DatapointIndex)
                    .unwrap()
                    .values()
                    .iter()
                    .map(|&v| val_to_f32(v))
                    .collect()
            })
            .collect();

        // Run K-means clustering
        let kmeans_config = KMeansConfig::new(k)
            .with_max_iterations(100)
            .with_convergence_threshold(1e-5)
            .with_seed(42);

        let kmeans = KMeans::new(kmeans_config);
        let subset_dataset = DenseDataset::from_vecs(data.clone());
        let result = kmeans.fit(&subset_dataset)?;

        // Store centers
        self.centers = result.centers;

        // Build partition indices
        let mut partition_to_indices: Vec<Vec<DatapointIndex>> = vec![Vec::new(); k];
        let assignments: Vec<u32> = result.assignments.iter().map(|&a| a as u32).collect();

        for (i, &assignment) in assignments.iter().enumerate() {
            partition_to_indices[assignment as usize].push(i as DatapointIndex);
        }

        self.tokenization = Some(DatabaseTokenization::new(
            assignments,
            partition_to_indices,
            self.centers.clone(),
        ));

        Ok(())
    }

    /// Build with a K-means tree for hierarchical partitioning.
    pub fn build_hierarchical<T: DatapointValue + Sync>(
        &mut self,
        dataset: &DenseDataset<T>,
        tree_config: KMeansTreeConfig,
    ) -> Result<()> {
        if dataset.is_empty() {
            return Err(ScannError::invalid_argument("Cannot partition empty dataset"));
        }

        self.dimensionality = dataset.dimensionality() as usize;

        let mut tree = KMeansTree::new(tree_config);
        tree.build(dataset)?;

        // Build tokenization from tree leaves
        let leaves = tree.leaves();
        let num_partitions = leaves.len();

        let mut assignments = vec![0u32; dataset.size()];
        let mut partition_to_indices: Vec<Vec<DatapointIndex>> = vec![Vec::new(); num_partitions];
        let mut centers = Vec::with_capacity(num_partitions);

        for (partition_id, leaf) in leaves.iter().enumerate() {
            centers.push(leaf.center.clone());
            for &idx in &leaf.datapoint_indices {
                assignments[idx as usize] = partition_id as u32;
                partition_to_indices[partition_id].push(idx);
            }
        }

        self.centers = centers.clone();
        self.tokenization = Some(DatabaseTokenization::new(
            assignments,
            partition_to_indices,
            centers,
        ));
        self.tree = Some(tree);

        Ok(())
    }

    /// Get the tokenization.
    pub fn tokenization(&self) -> Option<&DatabaseTokenization> {
        self.tokenization.as_ref()
    }

    /// Get the centers.
    pub fn centers(&self) -> &[Vec<f32>] {
        &self.centers
    }

    /// Get the number of partitions.
    pub fn num_partitions(&self) -> usize {
        self.centers.len()
    }

    /// Get the number of datapoints.
    pub fn num_datapoints(&self) -> usize {
        self.tokenization.as_ref().map(|t| t.num_datapoints()).unwrap_or(0)
    }

    /// Get the centroid of a specific partition.
    pub fn partition_centroid(&self, partition_id: u32) -> Option<Vec<f32>> {
        self.centers.get(partition_id as usize).cloned()
    }

    /// Get the indices of datapoints in a specific partition.
    pub fn partition_indices(&self, partition_id: u32) -> Option<&[DatapointIndex]> {
        self.tokenization.as_ref()
            .and_then(|t| t.partition_to_indices.get(partition_id as usize))
            .map(|v| v.as_slice())
    }

    /// Compute distances to all centers.
    fn compute_center_distances(&self, query: &[f32]) -> Vec<f32> {
        self.centers
            .iter()
            .map(|center| Self::squared_distance(query, center))
            .collect()
    }

    /// Squared Euclidean distance.
    #[inline]
    fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }
}

impl<T: DatapointValue> Partitioner<T> for TreePartitioner {
    fn partition(&self, query: &DatapointPtr<'_, T>, num_partitions: usize) -> Result<PartitionResult> {
        let tokenization = self.tokenization.as_ref()
            .ok_or_else(|| ScannError::failed_precondition("Partitioner not built"))?;

        let query_vec: Vec<f32> = query.values().iter().map(|&v| val_to_f32(v)).collect();

        // Compute distances to all centers
        let distances = self.compute_center_distances(&query_vec);

        // Get top-k partitions by distance
        let mut indexed_distances: Vec<(u32, f32)> = distances
            .iter()
            .enumerate()
            .map(|(i, &d)| (i as u32, d))
            .collect();

        indexed_distances.sort_by_key(|&(_, d)| OrderedFloat(d));

        let num_to_return = num_partitions.min(indexed_distances.len());
        let tokens: Vec<u32> = indexed_distances[..num_to_return]
            .iter()
            .map(|&(t, _)| t)
            .collect();
        let result_distances: Vec<f32> = indexed_distances[..num_to_return]
            .iter()
            .map(|&(_, d)| d)
            .collect();
        let sizes: Vec<usize> = tokens
            .iter()
            .map(|&t| tokenization.partition_to_indices[t as usize].len())
            .collect();

        Ok(PartitionResult::with_sizes(tokens, result_distances, sizes))
    }

    fn partition_indices(&self, partition_id: u32) -> Option<&[DatapointIndex]> {
        self.tokenization
            .as_ref()
            .and_then(|t| t.indices_in_partition(partition_id))
    }

    fn num_partitions(&self) -> usize {
        self.tokenization
            .as_ref()
            .map(|t| t.num_partitions())
            .unwrap_or(0)
    }

    fn num_datapoints(&self) -> usize {
        self.tokenization
            .as_ref()
            .map(|t| t.num_datapoints())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_format::Datapoint;

    fn create_clustered_data() -> DenseDataset<f32> {
        let mut data = Vec::new();

        // Cluster around (0, 0)
        for i in 0..20 {
            data.push(vec![0.0 + (i as f32) * 0.1, 0.0 + (i as f32) * 0.05]);
        }

        // Cluster around (10, 10)
        for i in 0..20 {
            data.push(vec![10.0 + (i as f32) * 0.1, 10.0 + (i as f32) * 0.05]);
        }

        // Cluster around (0, 10)
        for i in 0..20 {
            data.push(vec![0.0 + (i as f32) * 0.1, 10.0 + (i as f32) * 0.05]);
        }

        DenseDataset::from_vecs(data)
    }

    #[test]
    fn test_tree_partitioner_build() {
        let dataset = create_clustered_data();
        let config = PartitionerConfig::new(3);
        let mut partitioner = TreePartitioner::new(config);
        partitioner.build(&dataset).unwrap();

        assert_eq!(partitioner.num_partitions(), 3);
        assert_eq!(partitioner.num_datapoints(), 60);
    }

    #[test]
    fn test_tree_partitioner_partition() {
        let dataset = create_clustered_data();
        let config = PartitionerConfig::new(3).with_partitions_to_search(2);
        let mut partitioner = TreePartitioner::new(config);
        partitioner.build(&dataset).unwrap();

        let query = Datapoint::dense(vec![0.0f32, 0.0]);
        let result = partitioner.partition(&query.as_ptr(), 2).unwrap();

        assert_eq!(result.len(), 2);
        // Results should be sorted by distance
        for i in 1..result.len() {
            assert!(result.distances[i] >= result.distances[i - 1]);
        }
    }

    #[test]
    fn test_tree_partitioner_indices() {
        let dataset = create_clustered_data();
        let config = PartitionerConfig::new(3);
        let mut partitioner = TreePartitioner::new(config);
        partitioner.build(&dataset).unwrap();

        // Check that all partitions have indices
        let mut total_indices = 0;
        for i in 0..3 {
            if let Some(indices) = partitioner.partition_indices(i as u32) {
                total_indices += indices.len();
            }
        }
        assert_eq!(total_indices, 60);
    }

    #[test]
    fn test_tree_partitioner_hierarchical() {
        let dataset = create_clustered_data();
        let config = PartitionerConfig::new(3);
        let tree_config = KMeansTreeConfig::new(2)
            .with_max_depth(2)
            .with_min_leaf_size(10)
            .with_seed(42);

        let mut partitioner = TreePartitioner::new(config);
        partitioner.build_hierarchical(&dataset, tree_config).unwrap();

        assert!(partitioner.num_partitions() > 0);
        assert_eq!(partitioner.num_datapoints(), 60);
    }
}
