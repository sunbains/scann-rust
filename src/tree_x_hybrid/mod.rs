//! Tree-X-Hybrid searcher implementation.
//!
//! This module provides advanced hybrid search algorithms that combine:
//! - Tree-based partitioning for coarse filtering
//! - Asymmetric hashing for fast approximate scoring
//! - Optional residual-based refinement

use crate::brute_force::FastTopNeighbors;
use crate::data_format::{DenseDataset, DatapointPtr, Dataset};
use crate::distance_measures::DistanceMeasure;
use crate::error::{Result, ScannError};
use crate::hashes::{AsymmetricHasherConfig, Codebook, CodebookConfig, LookupTable};
use crate::partitioning::{Partitioner, PartitionerConfig, TreePartitioner};
use crate::restricts::RestrictFilter;
use crate::searcher::{NNResult, SearchParameters, SearchResult, Searcher};
use crate::types::{DatapointIndex, NNResultsVector};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for Tree-X-Hybrid searcher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeXHybridConfig {
    /// Number of partitions (tree leaves).
    pub num_partitions: usize,
    /// Number of partitions to search.
    pub partitions_to_search: usize,
    /// Asymmetric hashing configuration.
    pub hash_config: AsymmetricHasherConfig,
    /// Whether to use residual-based AH.
    pub use_residuals: bool,
    /// Pre-reordering multiplier.
    pub pre_reorder_multiplier: f32,
    /// Whether to enable parallel partition search.
    pub parallel_partition_search: bool,
}

impl Default for TreeXHybridConfig {
    fn default() -> Self {
        Self {
            num_partitions: 100,
            partitions_to_search: 10,
            hash_config: AsymmetricHasherConfig::default(),
            use_residuals: true,
            pre_reorder_multiplier: 3.0,
            parallel_partition_search: true,
        }
    }
}

impl TreeXHybridConfig {
    /// Create a new configuration.
    pub fn new(num_partitions: usize, partitions_to_search: usize) -> Self {
        Self {
            num_partitions,
            partitions_to_search,
            ..Default::default()
        }
    }

    /// Set hash configuration.
    pub fn with_hash(mut self, config: AsymmetricHasherConfig) -> Self {
        self.hash_config = config;
        self
    }

    /// Enable/disable residual-based AH.
    pub fn with_residuals(mut self, use_residuals: bool) -> Self {
        self.use_residuals = use_residuals;
        self
    }

    /// Set pre-reorder multiplier.
    pub fn with_pre_reorder(mut self, multiplier: f32) -> Self {
        self.pre_reorder_multiplier = multiplier;
        self
    }
}

/// Per-partition data for Tree-AH.
struct PartitionData {
    /// Indices of datapoints in this partition.
    indices: Vec<DatapointIndex>,
    /// Encoded data for this partition.
    encoded: Vec<Vec<u8>>,
    /// Codebook (may be shared or per-partition).
    codebook: Arc<Codebook>,
    /// Partition centroid.
    centroid: Vec<f32>,
}

/// Tree-X-Hybrid searcher combining tree partitioning with asymmetric hashing.
pub struct TreeXHybridSearcher {
    /// Configuration.
    config: TreeXHybridConfig,
    /// Tree partitioner.
    partitioner: TreePartitioner,
    /// Per-partition data.
    partitions: Vec<PartitionData>,
    /// Original dataset (for reordering).
    dataset: Arc<DenseDataset<f32>>,
    /// Global codebook (if not using per-partition).
    global_codebook: Option<Codebook>,
    /// Distance measure.
    distance_measure: DistanceMeasure,
    /// Dimensionality.
    dimensionality: usize,
    /// Total number of datapoints.
    num_datapoints: usize,
}

impl TreeXHybridSearcher {
    /// Create a new Tree-X-Hybrid searcher.
    pub fn new(config: TreeXHybridConfig) -> Self {
        Self {
            config: config.clone(),
            partitioner: TreePartitioner::new(
                PartitionerConfig::new(config.num_partitions)
                    .with_partitions_to_search(config.partitions_to_search),
            ),
            partitions: Vec::new(),
            dataset: Arc::new(DenseDataset::new()),
            global_codebook: None,
            distance_measure: DistanceMeasure::SquaredL2,
            dimensionality: 0,
            num_datapoints: 0,
        }
    }

    /// Build the searcher from a dataset.
    pub fn build(&mut self, dataset: DenseDataset<f32>) -> Result<()> {
        if dataset.is_empty() {
            return Err(ScannError::invalid_argument("Cannot build from empty dataset"));
        }

        self.dimensionality = dataset.dimensionality() as usize;
        self.num_datapoints = dataset.size();
        self.dataset = Arc::new(dataset);

        // Build partitioner
        self.partitioner.build(&*self.dataset)?;

        // Train global codebook
        let codebook_config = CodebookConfig::new(
            self.config.hash_config.num_codes,
            self.config.hash_config.num_subspaces,
        ).with_seed(self.config.hash_config.seed.unwrap_or(42));

        let mut global_codebook = Codebook::new(codebook_config);

        if self.config.use_residuals {
            // Train on residuals
            let residuals = self.compute_residuals()?;
            let residual_dataset = DenseDataset::from_vecs(residuals);
            global_codebook.train(&residual_dataset)?;
        } else {
            global_codebook.train(&self.dataset)?;
        }

        let global_codebook = Arc::new(global_codebook);

        // Build per-partition data
        self.partitions.clear();
        let num_partitions = self.partitioner.num_partitions();

        for partition_id in 0..num_partitions {
            let indices = self.partitioner
                .partition_indices(partition_id as u32)
                .map(|s| s.to_vec())
                .unwrap_or_default();

            let centroid = self.partitioner
                .partition_centroid(partition_id as u32)
                .unwrap_or_else(|| vec![0.0; self.dimensionality]);

            // Encode partition data
            let encoded: Vec<Vec<u8>> = if self.config.use_residuals {
                indices.iter()
                    .filter_map(|&idx| {
                        self.dataset.get(idx).map(|dp| {
                            let residual: Vec<f32> = dp.values()
                                .iter()
                                .zip(centroid.iter())
                                .map(|(&v, &c)| v - c)
                                .collect();
                            global_codebook.encode(&residual)
                        })
                    })
                    .collect()
            } else {
                indices.iter()
                    .filter_map(|&idx| {
                        self.dataset.get(idx).map(|dp| global_codebook.encode(dp.values()))
                    })
                    .collect()
            };

            self.partitions.push(PartitionData {
                indices,
                encoded,
                codebook: Arc::clone(&global_codebook),
                centroid,
            });
        }

        self.global_codebook = Some(Arc::try_unwrap(global_codebook).unwrap_or_else(|arc| (*arc).clone()));

        Ok(())
    }

    /// Compute residuals (vector - partition centroid).
    fn compute_residuals(&self) -> Result<Vec<Vec<f32>>> {
        let mut residuals = Vec::with_capacity(self.num_datapoints);

        for i in 0..self.num_datapoints {
            if let Some(dp) = self.dataset.get(i as u32) {
                // Find which partition this point belongs to
                let partition_result = self.partitioner.partition(&DatapointPtr::dense(dp.values()), 1)?;

                if let Some(&partition_id) = partition_result.tokens.first() {
                    if let Some(centroid) = self.partitioner.partition_centroid(partition_id) {
                        let residual: Vec<f32> = dp.values()
                            .iter()
                            .zip(centroid.iter())
                            .map(|(&v, &c)| v - c)
                            .collect();
                        residuals.push(residual);
                        continue;
                    }
                }
                // Fallback: use original vector
                residuals.push(dp.values().to_vec());
            }
        }

        Ok(residuals)
    }

    /// Search for nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<NNResultsVector> {
        self.search_with_filter(query, k, None)
    }

    /// Search with optional filter.
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&dyn RestrictFilter>,
    ) -> Result<NNResultsVector> {
        if query.len() != self.dimensionality {
            return Err(ScannError::invalid_argument("Query dimensionality mismatch"));
        }

        let query_ptr = DatapointPtr::dense(query);

        // Find partitions to search
        let partition_result = self.partitioner.partition(
            &query_ptr,
            self.config.partitions_to_search,
        )?;

        let pre_reorder_k = (k as f32 * self.config.pre_reorder_multiplier) as usize;

        // Search partitions
        let partition_results: Vec<NNResultsVector> = if self.config.parallel_partition_search {
            partition_result.tokens
                .par_iter()
                .filter_map(|&partition_id| {
                    self.search_partition(query, partition_id, pre_reorder_k, filter).ok()
                })
                .collect()
        } else {
            partition_result.tokens
                .iter()
                .filter_map(|&partition_id| {
                    self.search_partition(query, partition_id, pre_reorder_k, filter).ok()
                })
                .collect()
        };

        // Merge results
        let mut all_results: Vec<(DatapointIndex, f32)> = partition_results
            .into_iter()
            .flatten()
            .collect();

        // Sort by approximate distance
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(pre_reorder_k);

        // Reorder with exact distances
        self.reorder_results(query, all_results, k)
    }

    /// Search within a single partition.
    fn search_partition(
        &self,
        query: &[f32],
        partition_id: u32,
        k: usize,
        filter: Option<&dyn RestrictFilter>,
    ) -> Result<NNResultsVector> {
        let partition = self.partitions
            .get(partition_id as usize)
            .ok_or_else(|| ScannError::out_of_range("Invalid partition ID"))?;

        // Compute query residual if using residual-based AH
        let query_for_lut: Vec<f32> = if self.config.use_residuals {
            query.iter()
                .zip(partition.centroid.iter())
                .map(|(&q, &c)| q - c)
                .collect()
        } else {
            query.to_vec()
        };

        // Build lookup table
        let lut = LookupTable::from_query(&partition.codebook, &query_for_lut);

        // Compute approximate distances
        let mut top_k = FastTopNeighbors::new(k);

        for (i, codes) in partition.encoded.iter().enumerate() {
            let idx = partition.indices[i];

            // Apply filter
            if let Some(f) = filter {
                if !f.is_allowed(idx) {
                    continue;
                }
            }

            let dist = lut.compute_distance(codes);
            top_k.push(idx, dist);
        }

        Ok(top_k.results())
    }

    /// Reorder results using exact distances.
    fn reorder_results(
        &self,
        query: &[f32],
        candidates: Vec<(DatapointIndex, f32)>,
        k: usize,
    ) -> Result<NNResultsVector> {
        let query_ptr = DatapointPtr::dense(query);

        let mut exact_results: Vec<(DatapointIndex, f32)> = candidates
            .iter()
            .filter_map(|&(idx, _)| {
                self.dataset.get(idx).map(|dp| {
                    let exact_dist = self.distance_measure.distance(&query_ptr, &dp);
                    (idx, exact_dist)
                })
            })
            .collect();

        exact_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        exact_results.truncate(k);

        Ok(exact_results)
    }

    /// Get the number of partitions.
    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// Get the number of datapoints.
    pub fn num_datapoints(&self) -> usize {
        self.num_datapoints
    }

    /// Get configuration.
    pub fn config(&self) -> &TreeXHybridConfig {
        &self.config
    }
}

impl Searcher<f32> for TreeXHybridSearcher {
    fn search_with_params(
        &self,
        query: &DatapointPtr<'_, f32>,
        params: &SearchParameters,
    ) -> Result<SearchResult> {
        let k = params.num_neighbors.unwrap_or(10) as usize;
        let results = self.search(query.values(), k)?;

        Ok(SearchResult {
            neighbors: results
                .into_iter()
                .map(|(idx, dist)| NNResult::new(idx, dist))
                .collect(),
        })
    }

    fn search_batched_with_params(
        &self,
        queries: &[DatapointPtr<'_, f32>],
        params: &[SearchParameters],
    ) -> Result<Vec<SearchResult>> {
        queries
            .par_iter()
            .zip(params.par_iter())
            .map(|(q, p)| self.search_with_params(q, p))
            .collect()
    }

    fn dataset_size(&self) -> usize {
        self.num_datapoints
    }

    fn dimensionality(&self) -> u64 {
        self.dimensionality as u64
    }
}

/// Tree-AH with residual-based scoring.
pub type TreeAHHybridResidual = TreeXHybridSearcher;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dataset() -> DenseDataset<f32> {
        let data: Vec<Vec<f32>> = (0..500)
            .map(|i| {
                (0..32).map(|j| ((i * j) as f32 / 100.0).sin()).collect()
            })
            .collect();
        DenseDataset::from_vecs(data)
    }

    #[test]
    fn test_tree_x_hybrid_build() {
        let dataset = create_test_dataset();
        let config = TreeXHybridConfig::new(10, 3)
            .with_hash(AsymmetricHasherConfig::new(16, 8).with_seed(42));

        let mut searcher = TreeXHybridSearcher::new(config);
        searcher.build(dataset).unwrap();

        assert_eq!(searcher.num_partitions(), 10);
        assert_eq!(searcher.num_datapoints(), 500);
    }

    #[test]
    fn test_tree_x_hybrid_search() {
        let dataset = create_test_dataset();
        let config = TreeXHybridConfig::new(10, 3)
            .with_hash(AsymmetricHasherConfig::new(16, 8).with_seed(42))
            .with_residuals(true);

        let mut searcher = TreeXHybridSearcher::new(config);
        searcher.build(dataset).unwrap();

        let query: Vec<f32> = (0..32).map(|i| (i as f32 / 10.0).sin()).collect();
        let results = searcher.search(&query, 10).unwrap();

        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }
}
