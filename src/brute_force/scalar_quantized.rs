//! Scalar quantized brute-force searcher.
//!
//! This module provides a brute-force searcher that uses scalar quantized (Int8)
//! data for memory efficiency and improved cache utilization, while maintaining
//! high search accuracy through asymmetric distance computation.
//!
//! The search uses float query vectors against quantized database vectors,
//! which provides a good trade-off between memory usage and accuracy.

use crate::brute_force::top_k::TopK;
use crate::data_format::{DatapointPtr, DenseDataset};
use crate::distance_measures::{
    DistanceMeasure,
    one_to_many_int8_float_dot_product,
    one_to_many_int8_float_squared_l2,
};
use crate::error::{Result, ScannError};
use crate::quantization::{ScalarQuantizer, ScalarQuantizerConfig, QuantizedDataset, Quantizer};
use crate::searcher::{NNResult, SearchParameters, SearchResult, Searcher};
use crate::types::{DatapointIndex, NNResultsVector};
use rayon::prelude::*;
use std::sync::Arc;

/// Configuration for scalar quantized brute force searcher.
#[derive(Debug, Clone)]
pub struct ScalarQuantizedConfig {
    /// Scalar quantizer configuration.
    pub quantizer_config: ScalarQuantizerConfig,
    /// Distance measure to use.
    pub distance_measure: DistanceMeasure,
    /// Enable parallel search.
    pub parallel: bool,
    /// Minimum batch size for parallel search.
    pub parallel_batch_threshold: usize,
}

impl Default for ScalarQuantizedConfig {
    fn default() -> Self {
        Self {
            quantizer_config: ScalarQuantizerConfig::default(),
            distance_measure: DistanceMeasure::SquaredL2,
            parallel: true,
            parallel_batch_threshold: 100,
        }
    }
}

impl ScalarQuantizedConfig {
    /// Create config for squared L2 distance.
    pub fn squared_l2() -> Self {
        Self {
            distance_measure: DistanceMeasure::SquaredL2,
            ..Default::default()
        }
    }

    /// Create config for dot product.
    pub fn dot_product() -> Self {
        Self {
            distance_measure: DistanceMeasure::DotProduct,
            ..Default::default()
        }
    }

    /// Set distance measure.
    pub fn with_distance(mut self, distance: DistanceMeasure) -> Self {
        self.distance_measure = distance;
        self
    }

    /// Set parallel execution.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
}

/// Brute-force searcher with scalar quantized (Int8) data.
///
/// This searcher stores the database in Int8 format (4x memory reduction)
/// and uses asymmetric distance computation where queries remain as floats.
pub struct ScalarQuantizedBruteForceSearcher {
    /// Quantized dataset.
    quantized_dataset: Arc<QuantizedDataset>,
    /// Precomputed squared L2 norms (for dot product -> L2 conversion if needed).
    squared_l2_norms: Vec<f32>,
    /// Distance measure.
    distance_measure: DistanceMeasure,
    /// Enable parallel search.
    parallel: bool,
    /// Minimum batch size for parallel search.
    parallel_batch_threshold: usize,
}

impl ScalarQuantizedBruteForceSearcher {
    /// Create a new scalar quantized searcher from a float dataset.
    ///
    /// This quantizes the dataset to Int8 format.
    pub fn new(dataset: &DenseDataset<f32>, config: ScalarQuantizedConfig) -> Result<Self> {
        let quantizer = ScalarQuantizer::new(config.quantizer_config.clone());
        let quantized_dataset = QuantizedDataset::from_dataset(dataset, quantizer)?;

        // Precompute squared L2 norms
        let squared_l2_norms = Self::compute_squared_norms(&quantized_dataset);

        Ok(Self {
            quantized_dataset: Arc::new(quantized_dataset),
            squared_l2_norms,
            distance_measure: config.distance_measure,
            parallel: config.parallel,
            parallel_batch_threshold: config.parallel_batch_threshold,
        })
    }

    /// Create from already quantized data.
    pub fn from_quantized(
        quantized_dataset: QuantizedDataset,
        distance_measure: DistanceMeasure,
    ) -> Self {
        let squared_l2_norms = Self::compute_squared_norms(&quantized_dataset);

        Self {
            quantized_dataset: Arc::new(quantized_dataset),
            squared_l2_norms,
            distance_measure,
            parallel: true,
            parallel_batch_threshold: 100,
        }
    }

    /// Compute squared L2 norms for all database points.
    fn compute_squared_norms(dataset: &QuantizedDataset) -> Vec<f32> {
        let num_points = dataset.size();
        let quantizer = dataset.quantizer();

        let mut norms = Vec::with_capacity(num_points);

        for i in 0..num_points {
            if let Some(data) = dataset.get_quantized(i as DatapointIndex) {
                let mut norm = 0.0f32;
                for &val in data.iter() {
                    let f = quantizer.dequantize_value(val);
                    norm += f * f;
                }
                norms.push(norm);
            }
        }

        norms
    }

    /// Get the quantized dataset.
    pub fn quantized_dataset(&self) -> &QuantizedDataset {
        &self.quantized_dataset
    }

    /// Get the distance measure.
    pub fn distance_measure(&self) -> DistanceMeasure {
        self.distance_measure
    }

    /// Set whether to use parallel search.
    pub fn set_parallel(&mut self, parallel: bool) {
        self.parallel = parallel;
    }

    /// Search for the k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<NNResultsVector> {
        if self.quantized_dataset.size() == 0 {
            return Ok(Vec::new());
        }

        let dim = self.quantized_dataset.dimensionality();
        if query.len() != dim {
            return Err(ScannError::invalid_argument(format!(
                "Query dimensionality {} does not match dataset dimensionality {}",
                query.len(),
                dim
            )));
        }

        let k = k.min(self.quantized_dataset.size());
        self.search_impl(query, k)
    }

    /// Internal search implementation.
    fn search_impl(&self, query: &[f32], k: usize) -> Result<NNResultsVector> {
        let num_points = self.quantized_dataset.size();

        // Compute all distances
        let mut distances = vec![0.0f32; num_points];
        self.compute_distances(query, &mut distances);

        // Select top-k
        let mut top_k = TopK::new(k);
        for (i, &dist) in distances.iter().enumerate() {
            top_k.push(i as DatapointIndex, dist);
        }

        Ok(top_k.drain_sorted())
    }

    /// Compute distances from query to all datapoints.
    fn compute_distances(&self, query: &[f32], distances: &mut [f32]) {
        let raw_data = self.quantized_dataset.raw_data();
        let num_points = self.quantized_dataset.size();
        let dim = self.quantized_dataset.dimensionality();
        let quantizer = self.quantized_dataset.quantizer();

        // The inv_scale is used for dequantization: dequant = quant * scale + min
        // For asymmetric computation we need to account for the scaling
        let scale = quantizer.scale();

        match self.distance_measure {
            DistanceMeasure::SquaredL2 | DistanceMeasure::L2 => {
                // Use SIMD asymmetric distance
                one_to_many_int8_float_squared_l2(
                    query,
                    raw_data,
                    scale,
                    dim,  // stride = dimensionality
                    num_points,
                    distances,
                );
                if self.distance_measure == DistanceMeasure::L2 {
                    for d in distances.iter_mut() {
                        *d = d.sqrt();
                    }
                }
            }
            DistanceMeasure::DotProduct => {
                one_to_many_int8_float_dot_product(
                    query,
                    raw_data,
                    scale,
                    dim,
                    num_points,
                    distances,
                );
            }
            _ => {
                // Fallback: dequantize and compute
                self.compute_distances_fallback(query, distances);
            }
        }
    }

    /// Fallback for unsupported distance measures.
    fn compute_distances_fallback(&self, query: &[f32], distances: &mut [f32]) {
        for i in 0..self.quantized_dataset.size() {
            if let Some(dequantized) = self.quantized_dataset.get_dequantized(i as DatapointIndex) {
                let query_ptr = DatapointPtr::dense(query);
                let db_ptr = DatapointPtr::dense(&dequantized);
                distances[i] = self.distance_measure.distance(&query_ptr, &db_ptr);
            }
        }
    }

    /// Search for neighbors within a given radius.
    pub fn search_radius(&self, query: &[f32], radius: f32) -> Result<NNResultsVector> {
        if self.quantized_dataset.size() == 0 {
            return Ok(Vec::new());
        }

        let dim = self.quantized_dataset.dimensionality();
        if query.len() != dim {
            return Err(ScannError::invalid_argument(
                "Query dimensionality does not match dataset",
            ));
        }

        let num_points = self.quantized_dataset.size();
        let mut distances = vec![0.0f32; num_points];
        self.compute_distances(query, &mut distances);

        let mut results: Vec<_> = distances
            .iter()
            .enumerate()
            .filter(|(_, &d)| d <= radius)
            .map(|(i, &d)| (i as DatapointIndex, d))
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Batched search for multiple queries.
    pub fn search_batched(
        &self,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<NNResultsVector>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        if self.parallel && queries.len() >= self.parallel_batch_threshold {
            self.search_batched_parallel(queries, k)
        } else {
            self.search_batched_sequential(queries, k)
        }
    }

    /// Sequential batched search.
    fn search_batched_sequential(
        &self,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<NNResultsVector>> {
        queries
            .iter()
            .map(|q| self.search(q, k))
            .collect()
    }

    /// Parallel batched search.
    fn search_batched_parallel(
        &self,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<NNResultsVector>> {
        queries
            .par_iter()
            .map(|q| self.search(q, k))
            .collect()
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let data_size = self.quantized_dataset.raw_data().len();
        let norms_size = self.squared_l2_norms.len() * std::mem::size_of::<f32>();
        data_size + norms_size
    }

    /// Get compression ratio compared to float32 storage.
    pub fn compression_ratio(&self) -> f32 {
        let float_size = self.quantized_dataset.size()
            * self.quantized_dataset.dimensionality()
            * std::mem::size_of::<f32>();
        let quantized_size = self.quantized_dataset.raw_data().len();

        if quantized_size > 0 {
            float_size as f32 / quantized_size as f32
        } else {
            0.0
        }
    }
}

impl Searcher<f32> for ScalarQuantizedBruteForceSearcher {
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
                .map(|(idx, dist)| NNResult {
                    index: idx,
                    distance: dist,
                    docid: None,
                })
                .collect(),
        })
    }

    fn search_batched_with_params(
        &self,
        queries: &[DatapointPtr<'_, f32>],
        params: &[SearchParameters],
    ) -> Result<Vec<SearchResult>> {
        if queries.len() != params.len() {
            return Err(ScannError::invalid_argument(
                "Number of queries must match number of parameter sets",
            ));
        }

        queries
            .iter()
            .zip(params.iter())
            .map(|(q, p)| self.search_with_params(q, p))
            .collect()
    }

    fn dataset_size(&self) -> usize {
        self.quantized_dataset.size()
    }

    fn dimensionality(&self) -> u64 {
        self.quantized_dataset.dimensionality() as u64
    }
}

impl Clone for ScalarQuantizedBruteForceSearcher {
    fn clone(&self) -> Self {
        Self {
            quantized_dataset: Arc::clone(&self.quantized_dataset),
            squared_l2_norms: self.squared_l2_norms.clone(),
            distance_measure: self.distance_measure,
            parallel: self.parallel,
            parallel_batch_threshold: self.parallel_batch_threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_format::DenseDataset;

    fn create_test_dataset() -> DenseDataset<f32> {
        DenseDataset::from_vecs(vec![
            vec![0.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
            vec![0.0, 10.0, 0.0],
            vec![0.0, 0.0, 10.0],
            vec![10.0, 10.0, 10.0],
        ])
    }

    #[test]
    fn test_scalar_quantized_search() {
        let dataset = create_test_dataset();
        let config = ScalarQuantizedConfig::squared_l2();
        let searcher = ScalarQuantizedBruteForceSearcher::new(&dataset, config).unwrap();

        let query = vec![0.0, 0.0, 0.0];
        let results = searcher.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 0); // Exact match (closest)
    }

    #[test]
    fn test_scalar_quantized_search_dot_product() {
        let dataset = DenseDataset::from_vecs(vec![
            vec![10.0, 0.0],
            vec![0.0, 10.0],
            vec![10.0, 10.0],
        ]);
        let config = ScalarQuantizedConfig::dot_product();
        let searcher = ScalarQuantizedBruteForceSearcher::new(&dataset, config).unwrap();

        let query = vec![1.0, 0.0];
        let results = searcher.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_scalar_quantized_batched() {
        let dataset = create_test_dataset();
        let config = ScalarQuantizedConfig::squared_l2().with_parallel(false);
        let searcher = ScalarQuantizedBruteForceSearcher::new(&dataset, config).unwrap();

        let queries = vec![
            vec![0.0, 0.0, 0.0],
            vec![10.0, 10.0, 10.0],
        ];
        let results = searcher.search_batched(&queries, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[1].len(), 2);
    }

    #[test]
    fn test_compression_ratio() {
        let dataset = DenseDataset::from_vecs(vec![
            vec![1.0; 128], // 128 dimensions
            vec![2.0; 128],
            vec![3.0; 128],
        ]);
        let config = ScalarQuantizedConfig::squared_l2();
        let searcher = ScalarQuantizedBruteForceSearcher::new(&dataset, config).unwrap();

        // Int8 is 1 byte vs f32's 4 bytes, so ~4x compression
        let ratio = searcher.compression_ratio();
        assert!(ratio > 3.0 && ratio < 5.0, "Compression ratio: {}", ratio);
    }

    #[test]
    fn test_accuracy_vs_float() {
        // Test that quantized search results are similar to float results
        let dataset = DenseDataset::from_vecs(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![1.1, 2.1, 3.1],
            vec![10.0, 0.0, 0.0],
        ]);

        // Float searcher
        let float_searcher = crate::brute_force::BruteForceSearcher::new(
            dataset.clone(),
            DistanceMeasure::SquaredL2,
        );

        // Quantized searcher
        let config = ScalarQuantizedConfig::squared_l2();
        let quantized_searcher = ScalarQuantizedBruteForceSearcher::new(&dataset, config).unwrap();

        let query = vec![1.0, 2.0, 3.0];

        let float_results = float_searcher.search(&query, 3).unwrap();
        let quant_results = quantized_searcher.search(&query, 3).unwrap();

        // Top result should be the same (or very close)
        assert_eq!(float_results[0].0, quant_results[0].0);
    }
}
