//! Asymmetric hasher for approximate distance computation.
//!
//! This module provides the main asymmetric hashing implementation
//! for fast approximate nearest neighbor search.

use crate::brute_force::FastTopNeighbors;
use crate::data_format::{Dataset, DenseDataset, DatapointPtr};
use crate::distance_measures::DistanceMeasure;
use crate::error::{Result, ScannError};
use crate::hashes::codebook::{Codebook, CodebookConfig};
use crate::hashes::lut::{LookupTable, LutFormat};
use crate::searcher::{NNResult, SearchParameters, SearchResult, Searcher};
use crate::types::{DatapointIndex, NNResultsVector};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for asymmetric hasher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsymmetricHasherConfig {
    /// Number of codes per subspace.
    pub num_codes: usize,

    /// Number of subspaces.
    pub num_subspaces: usize,

    /// Lookup table format.
    pub lut_format: LutFormat,

    /// Training sample size.
    pub training_sample_size: usize,

    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for AsymmetricHasherConfig {
    fn default() -> Self {
        Self {
            num_codes: 256,
            num_subspaces: 8,
            lut_format: LutFormat::Float32,
            training_sample_size: 100_000,
            seed: None,
        }
    }
}

impl AsymmetricHasherConfig {
    /// Create a new configuration.
    pub fn new(num_codes: usize, num_subspaces: usize) -> Self {
        Self {
            num_codes,
            num_subspaces,
            ..Default::default()
        }
    }

    /// Set the LUT format.
    pub fn with_lut_format(mut self, format: LutFormat) -> Self {
        self.lut_format = format;
        self
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Asymmetric hasher for approximate nearest neighbor search.
///
/// Uses product quantization to encode database vectors and computes
/// approximate distances using lookup tables.
pub struct AsymmetricHasher {
    /// Trained codebook.
    codebook: Codebook,

    /// Encoded database.
    encoded_database: Vec<Vec<u8>>,

    /// Original dataset (for exact recomputation).
    dataset: Option<Arc<DenseDataset<f32>>>,

    /// Configuration.
    config: AsymmetricHasherConfig,

    /// Number of datapoints.
    num_datapoints: usize,

    /// Dimensionality.
    dimensionality: usize,
}

impl AsymmetricHasher {
    /// Create a new asymmetric hasher with the given configuration.
    pub fn new(config: AsymmetricHasherConfig) -> Self {
        Self {
            codebook: Codebook::new(CodebookConfig::default()),
            encoded_database: Vec::new(),
            dataset: None,
            config,
            num_datapoints: 0,
            dimensionality: 0,
        }
    }

    /// Build the hasher from a dataset.
    pub fn build(&mut self, dataset: DenseDataset<f32>) -> Result<()> {
        if dataset.is_empty() {
            return Err(ScannError::invalid_argument("Cannot build from empty dataset"));
        }

        self.dimensionality = dataset.dimensionality() as usize;
        self.num_datapoints = dataset.size();

        // Train codebook
        let codebook_config = CodebookConfig::new(
            self.config.num_codes,
            self.config.num_subspaces,
        )
        .with_seed(self.config.seed.unwrap_or(42));

        self.codebook = Codebook::new(codebook_config);
        self.codebook.train(&dataset)?;

        // Encode database
        self.encoded_database = self.codebook.encode_dataset(&dataset);

        // Store dataset for reordering
        self.dataset = Some(Arc::new(dataset));

        Ok(())
    }

    /// Build without storing the original dataset.
    pub fn build_no_store(&mut self, dataset: &DenseDataset<f32>) -> Result<()> {
        if dataset.is_empty() {
            return Err(ScannError::invalid_argument("Cannot build from empty dataset"));
        }

        self.dimensionality = dataset.dimensionality() as usize;
        self.num_datapoints = dataset.size();

        // Train codebook
        let codebook_config = CodebookConfig::new(
            self.config.num_codes,
            self.config.num_subspaces,
        )
        .with_seed(self.config.seed.unwrap_or(42));

        self.codebook = Codebook::new(codebook_config);
        self.codebook.train(dataset)?;

        // Encode database
        self.encoded_database = self.codebook.encode_dataset(dataset);

        Ok(())
    }

    /// Search for nearest neighbors using asymmetric hashing.
    pub fn search(&self, query: &[f32], k: usize) -> Result<NNResultsVector> {
        if self.encoded_database.is_empty() {
            return Ok(Vec::new());
        }

        if query.len() != self.dimensionality {
            return Err(ScannError::invalid_argument(
                "Query dimensionality does not match",
            ));
        }

        // Build lookup table for query
        let lut = LookupTable::from_query(&self.codebook, query);

        // Compute approximate distances
        let mut top_k = FastTopNeighbors::new(k);

        for (i, codes) in self.encoded_database.iter().enumerate() {
            let dist = lut.compute_distance(codes);
            top_k.push(i as DatapointIndex, dist);
        }

        Ok(top_k.results())
    }

    /// Search with pre-reordering.
    pub fn search_with_reordering(
        &self,
        query: &[f32],
        k: usize,
        pre_reorder_k: usize,
    ) -> Result<NNResultsVector> {
        let dataset = self
            .dataset
            .as_ref()
            .ok_or_else(|| ScannError::failed_precondition("Dataset not stored"))?;

        // Get candidates with approximate search
        let candidates = self.search(query, pre_reorder_k)?;

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Recompute exact distances for candidates
        let query_ptr = DatapointPtr::dense(query);
        let distance_measure = DistanceMeasure::SquaredL2;

        let mut results: Vec<_> = candidates
            .iter()
            .filter_map(|(idx, _)| {
                dataset.get(*idx).map(|dp| {
                    let exact_dist = distance_measure.distance(&query_ptr, &dp);
                    (*idx, exact_dist)
                })
            })
            .collect();

        // Sort by exact distance
        results.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top-k
        results.truncate(k);
        Ok(results)
    }

    /// Batched search.
    pub fn search_batched(
        &self,
        queries: &[&[f32]],
        k: usize,
    ) -> Result<Vec<NNResultsVector>> {
        queries.iter().map(|q| self.search(q, k)).collect()
    }

    /// Get the codebook.
    pub fn codebook(&self) -> &Codebook {
        &self.codebook
    }

    /// Get the encoded database.
    pub fn encoded_database(&self) -> &[Vec<u8>] {
        &self.encoded_database
    }

    /// Get the number of datapoints.
    pub fn num_datapoints(&self) -> usize {
        self.num_datapoints
    }

    /// Get the dimensionality.
    pub fn dimensionality(&self) -> usize {
        self.dimensionality
    }
}

impl Searcher<f32> for AsymmetricHasher {
    fn search_with_params(
        &self,
        query: &DatapointPtr<'_, f32>,
        params: &SearchParameters,
    ) -> Result<SearchResult> {
        let k = params.num_neighbors.unwrap_or(10) as usize;
        let pre_reorder_k = params.pre_reordering_num_neighbors.map(|n| n as usize);

        let results = if let Some(pre_k) = pre_reorder_k {
            self.search_with_reordering(query.values(), k, pre_k)?
        } else {
            self.search(query.values(), k)?
        };

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
            .iter()
            .zip(params.iter())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_format::Datapoint;

    fn create_test_dataset() -> DenseDataset<f32> {
        let mut data = Vec::new();
        for i in 0..200 {
            let mut vec = Vec::with_capacity(32);
            for j in 0..32 {
                vec.push(((i * j) as f32 / 100.0).sin());
            }
            data.push(vec);
        }
        DenseDataset::from_vecs(data)
    }

    #[test]
    fn test_asymmetric_hasher_build() {
        let dataset = create_test_dataset();
        let config = AsymmetricHasherConfig::new(16, 8).with_seed(42);
        let mut hasher = AsymmetricHasher::new(config);
        hasher.build(dataset).unwrap();

        assert_eq!(hasher.num_datapoints(), 200);
        assert_eq!(hasher.dimensionality(), 32);
        assert_eq!(hasher.encoded_database().len(), 200);
    }

    #[test]
    fn test_asymmetric_hasher_search() {
        let dataset = create_test_dataset();
        let config = AsymmetricHasherConfig::new(16, 8).with_seed(42);
        let mut hasher = AsymmetricHasher::new(config);
        hasher.build(dataset).unwrap();

        let query = vec![0.5f32; 32];
        let results = hasher.search(&query, 10).unwrap();

        assert_eq!(results.len(), 10);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_asymmetric_hasher_reordering() {
        let dataset = create_test_dataset();
        let config = AsymmetricHasherConfig::new(16, 8).with_seed(42);
        let mut hasher = AsymmetricHasher::new(config);
        hasher.build(dataset).unwrap();

        let query = vec![0.5f32; 32];
        let results = hasher.search_with_reordering(&query, 10, 50).unwrap();

        assert_eq!(results.len(), 10);
        // Results should be sorted by exact distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_asymmetric_hasher_searcher_trait() {
        let dataset = create_test_dataset();
        let config = AsymmetricHasherConfig::new(16, 8).with_seed(42);
        let mut hasher = AsymmetricHasher::new(config);
        hasher.build(dataset).unwrap();

        let query = Datapoint::dense(vec![0.5f32; 32]);
        let params = SearchParameters::new().with_num_neighbors(5);
        let results = hasher.search_with_params(&query.as_ptr(), &params).unwrap();

        assert_eq!(results.len(), 5);
    }
}
