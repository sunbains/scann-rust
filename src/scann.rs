//! Main ScaNN interface.
//!
//! This module provides the unified ScaNN searcher that combines
//! partitioning, hashing, and reordering for high-performance search.

use crate::brute_force::BruteForceSearcher;
use crate::config::ScannConfig;
use crate::data_format::{Dataset, DenseDataset, DatapointPtr};
use crate::distance_measures::DistanceMeasure;
use crate::error::{Result, ScannError};
use crate::hashes::{AsymmetricHasher, AsymmetricHasherConfig};
use crate::partitioning::{Partitioner, PartitionerConfig, TreePartitioner};
use crate::searcher::{SearchParameters, SearchResult, Searcher};
use crate::types::{DatapointIndex, NNResultsVector};
use crate::utils::ReorderingHelper;
use std::sync::Arc;

/// Search mode for ScaNN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Brute-force exact search.
    BruteForce,
    /// Partitioned search (tree-based).
    Partitioned,
    /// Asymmetric hashing.
    Hashed,
    /// Combined partitioning + hashing.
    TreeAH,
}

/// Main ScaNN searcher interface.
///
/// This is the primary entry point for using ScaNN. It supports multiple
/// search configurations and automatically selects the best algorithm.
pub struct Scann {
    /// The underlying dataset.
    dataset: Arc<DenseDataset<f32>>,

    /// Brute-force searcher (always available).
    brute_force: BruteForceSearcher<f32>,

    /// Optional partitioner.
    partitioner: Option<TreePartitioner>,

    /// Optional asymmetric hasher.
    hasher: Option<AsymmetricHasher>,

    /// Configuration.
    config: ScannConfig,

    /// Current search mode.
    search_mode: SearchMode,

    /// Reordering helper.
    reordering_helper: ReorderingHelper<f32>,
}

impl Scann {
    /// Create a new ScaNN searcher with default configuration.
    pub fn new(dataset: DenseDataset<f32>) -> Result<Self> {
        Self::with_config(dataset, ScannConfig::default())
    }

    /// Create a ScaNN searcher with the given configuration.
    pub fn with_config(dataset: DenseDataset<f32>, config: ScannConfig) -> Result<Self> {
        if dataset.is_empty() {
            return Err(ScannError::invalid_argument("Dataset cannot be empty"));
        }

        let dataset = Arc::new(dataset);
        let brute_force = BruteForceSearcher::with_shared_dataset(
            Arc::clone(&dataset),
            config.distance_measure,
        );

        let reordering_helper = ReorderingHelper::new(config.distance_measure);

        let mut scann = Self {
            dataset,
            brute_force,
            partitioner: None,
            hasher: None,
            config,
            search_mode: SearchMode::BruteForce,
            reordering_helper,
        };

        // Initialize based on configuration
        if scann.config.brute_force.is_some() {
            scann.search_mode = SearchMode::BruteForce;
        } else if scann.config.partitioning.is_some() && scann.config.hash.is_some() {
            scann.init_tree_ah()?;
            scann.search_mode = SearchMode::TreeAH;
        } else if scann.config.partitioning.is_some() {
            scann.init_partitioning()?;
            scann.search_mode = SearchMode::Partitioned;
        } else if scann.config.hash.is_some() {
            scann.init_hashing()?;
            scann.search_mode = SearchMode::Hashed;
        }

        Ok(scann)
    }

    /// Create a brute-force searcher.
    pub fn brute_force(dataset: DenseDataset<f32>) -> Result<Self> {
        let config = ScannConfig::default().with_brute_force();
        Self::with_config(dataset, config)
    }

    /// Create a partitioned searcher.
    pub fn partitioned(
        dataset: DenseDataset<f32>,
        num_partitions: u32,
        partitions_to_search: u32,
    ) -> Result<Self> {
        let mut config = ScannConfig::default();
        config.partitioning = Some(crate::config::PartitioningConfig {
            num_partitions,
            num_partitions_to_search: partitions_to_search,
            ..Default::default()
        });
        Self::with_config(dataset, config)
    }

    /// Create a hasher-based searcher.
    pub fn hashed(
        dataset: DenseDataset<f32>,
        num_blocks: u32,
    ) -> Result<Self> {
        let mut config = ScannConfig::default();
        config.hash = Some(crate::config::HashConfig {
            num_blocks,
            ..Default::default()
        });
        Self::with_config(dataset, config)
    }

    /// Initialize partitioning.
    fn init_partitioning(&mut self) -> Result<()> {
        let part_config = self.config.partitioning.as_ref().unwrap();
        let config = PartitionerConfig::new(part_config.num_partitions as usize)
            .with_partitions_to_search(part_config.num_partitions_to_search as usize);

        let mut partitioner = TreePartitioner::new(config);
        partitioner.build(&*self.dataset)?;
        self.partitioner = Some(partitioner);

        Ok(())
    }

    /// Initialize hashing.
    fn init_hashing(&mut self) -> Result<()> {
        let hash_config = self.config.hash.as_ref().unwrap();
        let config = AsymmetricHasherConfig::new(
            hash_config.num_buckets as usize,
            hash_config.num_blocks as usize,
        );

        let mut hasher = AsymmetricHasher::new(config);
        hasher.build((*self.dataset).clone())?;
        self.hasher = Some(hasher);

        Ok(())
    }

    /// Initialize tree + asymmetric hashing.
    fn init_tree_ah(&mut self) -> Result<()> {
        self.init_partitioning()?;
        self.init_hashing()?;
        Ok(())
    }

    /// Search for the k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<NNResultsVector> {
        let query_ptr = DatapointPtr::dense(query);
        self.search_impl(&query_ptr, k)
    }

    /// Internal search implementation.
    fn search_impl(&self, query: &DatapointPtr<'_, f32>, k: usize) -> Result<NNResultsVector> {
        let results = match self.search_mode {
            SearchMode::BruteForce => {
                self.brute_force.search(query.values(), k)?
            }
            SearchMode::Partitioned => {
                self.search_partitioned(query, k)?
            }
            SearchMode::Hashed => {
                let hasher = self.hasher.as_ref().unwrap();
                hasher.search(query.values(), k)?
            }
            SearchMode::TreeAH => {
                self.search_tree_ah(query, k)?
            }
        };

        // Apply reordering if configured
        if let Some(reorder_config) = &self.config.exact_reordering {
            let num_candidates = reorder_config.num_candidates as usize;
            if num_candidates > k {
                return Ok(self.reordering_helper.reorder(
                    query,
                    &results,
                    &*self.dataset,
                    k,
                ));
            }
        }

        Ok(results)
    }

    /// Partitioned search.
    fn search_partitioned(
        &self,
        query: &DatapointPtr<'_, f32>,
        k: usize,
    ) -> Result<NNResultsVector> {
        let partitioner = self.partitioner.as_ref().unwrap();
        let part_config = self.config.partitioning.as_ref().unwrap();

        // Get partitions to search
        let partition_result = partitioner.partition(
            query,
            part_config.num_partitions_to_search as usize,
        )?;

        // Collect all candidate indices
        let mut candidates: Vec<DatapointIndex> = Vec::new();
        for token in &partition_result.tokens {
            if let Some(indices) = partitioner.partition_indices(*token) {
                candidates.extend_from_slice(indices);
            }
        }

        // Compute distances to candidates
        let distance_measure = self.config.distance_measure;
        let mut results: Vec<_> = candidates
            .iter()
            .filter_map(|&idx| {
                self.dataset.get(idx).map(|dp| {
                    let dist = distance_measure.distance(query, &dp);
                    (idx, dist)
                })
            })
            .collect();

        // Sort and return top-k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        Ok(results)
    }

    /// Tree + AH search.
    fn search_tree_ah(
        &self,
        query: &DatapointPtr<'_, f32>,
        k: usize,
    ) -> Result<NNResultsVector> {
        let partitioner = self.partitioner.as_ref().unwrap();
        let hasher = self.hasher.as_ref().unwrap();
        let part_config = self.config.partitioning.as_ref().unwrap();

        // Get partitions to search
        let partition_result = partitioner.partition(
            query,
            part_config.num_partitions_to_search as usize,
        )?;

        // Search within selected partitions using hashing
        let mut all_results: Vec<(DatapointIndex, f32)> = Vec::new();

        for token in &partition_result.tokens {
            if let Some(indices) = partitioner.partition_indices(*token) {
                // Use hasher's lookup table
                let lut = crate::hashes::LookupTable::from_query(
                    hasher.codebook(),
                    query.values(),
                );

                for &idx in indices {
                    let codes = &hasher.encoded_database()[idx as usize];
                    let dist = lut.compute_distance(codes);
                    all_results.push((idx, dist));
                }
            }
        }

        // Sort and return top-k
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(k);
        Ok(all_results)
    }

    /// Batched search for multiple queries.
    pub fn search_batched(
        &self,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<NNResultsVector>> {
        queries.iter().map(|q| self.search(q, k)).collect()
    }

    /// Get the search mode.
    pub fn search_mode(&self) -> SearchMode {
        self.search_mode
    }

    /// Get the dataset size.
    pub fn size(&self) -> usize {
        self.dataset.size()
    }

    /// Get the dimensionality.
    pub fn dimensionality(&self) -> u64 {
        self.dataset.dimensionality()
    }

    /// Get the distance measure.
    pub fn distance_measure(&self) -> DistanceMeasure {
        self.config.distance_measure
    }

    /// Get the configuration.
    pub fn config(&self) -> &ScannConfig {
        &self.config
    }
}

impl Searcher<f32> for Scann {
    fn search_with_params(
        &self,
        query: &DatapointPtr<'_, f32>,
        params: &SearchParameters,
    ) -> Result<SearchResult> {
        let k = params.num_neighbors.unwrap_or(self.config.num_neighbors) as usize;
        let results = self.search_impl(query, k)?;
        Ok(SearchResult::from_pairs(results))
    }

    fn search_batched_with_params(
        &self,
        queries: &[DatapointPtr<'_, f32>],
        params: &[SearchParameters],
    ) -> Result<Vec<SearchResult>> {
        queries
            .iter()
            .zip(params.iter())
            .map(|(q, p)| self.search_with_params(q, p))
            .collect()
    }

    fn dataset_size(&self) -> usize {
        self.size()
    }

    fn dimensionality(&self) -> u64 {
        self.dimensionality()
    }
}

/// Builder for creating ScaNN searchers.
pub struct ScannBuilder {
    config: ScannConfig,
}

impl ScannBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: ScannConfig::default(),
        }
    }

    /// Set the number of neighbors to return.
    pub fn num_neighbors(mut self, k: u32) -> Self {
        self.config.num_neighbors = k;
        self
    }

    /// Set the distance measure.
    pub fn distance_measure(mut self, measure: DistanceMeasure) -> Self {
        self.config.distance_measure = measure;
        self
    }

    /// Configure for brute-force search.
    pub fn brute_force(mut self) -> Self {
        self.config = self.config.with_brute_force();
        self
    }

    /// Configure partitioning.
    pub fn tree(mut self, num_partitions: u32, partitions_to_search: u32) -> Self {
        self.config.partitioning = Some(crate::config::PartitioningConfig {
            num_partitions,
            num_partitions_to_search: partitions_to_search,
            ..Default::default()
        });
        self
    }

    /// Configure hashing.
    pub fn hash(mut self, num_blocks: u32) -> Self {
        self.config.hash = Some(crate::config::HashConfig {
            num_blocks,
            ..Default::default()
        });
        self
    }

    /// Configure reordering.
    pub fn reorder(mut self, num_candidates: u32) -> Self {
        self.config.exact_reordering = Some(crate::config::ExactReorderingConfig {
            num_candidates,
            ..Default::default()
        });
        self
    }

    /// Build the ScaNN searcher.
    pub fn build(self, dataset: DenseDataset<f32>) -> Result<Scann> {
        Scann::with_config(dataset, self.config)
    }
}

impl Default for ScannBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dataset() -> DenseDataset<f32> {
        let mut data = Vec::new();
        for i in 0..100 {
            let mut vec = Vec::with_capacity(16);
            for j in 0..16 {
                vec.push((i as f32 + j as f32 * 0.1).sin());
            }
            data.push(vec);
        }
        DenseDataset::from_vecs(data)
    }

    #[test]
    fn test_scann_brute_force() {
        let dataset = create_test_dataset();
        let scann = Scann::brute_force(dataset).unwrap();

        assert_eq!(scann.search_mode(), SearchMode::BruteForce);
        assert_eq!(scann.size(), 100);

        let query = vec![0.5f32; 16];
        let results = scann.search(&query, 10).unwrap();

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_scann_builder() {
        let dataset = create_test_dataset();
        let scann = ScannBuilder::new()
            .num_neighbors(5)
            .distance_measure(DistanceMeasure::SquaredL2)
            .brute_force()
            .build(dataset)
            .unwrap();

        assert_eq!(scann.config().num_neighbors, 5);
        assert_eq!(scann.distance_measure(), DistanceMeasure::SquaredL2);
    }

    #[test]
    fn test_scann_batched() {
        let dataset = create_test_dataset();
        let scann = Scann::brute_force(dataset).unwrap();

        let queries = vec![
            vec![0.5f32; 16],
            vec![0.3f32; 16],
            vec![0.7f32; 16],
        ];
        let results = scann.search_batched(&queries, 5).unwrap();

        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.len(), 5);
        }
    }
}
