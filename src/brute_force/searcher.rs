//! Brute-force searcher implementation.
//!
//! This module provides the main brute-force nearest neighbor searcher.

use crate::brute_force::top_k::TopK;
use crate::data_format::{Dataset, DenseDataset, DatapointPtr};
use crate::distance_measures::{DistanceMeasure, one_to_many_squared_l2_strided, one_to_many_dot_product_strided};
use crate::error::{Result, ScannError};
use crate::searcher::{Searcher, SearchParameters, SearchResult, NNResult};
use crate::types::{DatapointIndex, DatapointValue, NNResultsVector};
use rayon::prelude::*;
use std::sync::Arc;

/// Brute-force nearest neighbor searcher.
///
/// This searcher computes exact distances to all datapoints and returns
/// the k nearest neighbors. It's optimized with SIMD for dense datasets.
pub struct BruteForceSearcher<T: DatapointValue> {
    /// The dataset to search.
    dataset: Arc<DenseDataset<T>>,

    /// Distance measure to use.
    distance_measure: DistanceMeasure,

    /// Enable parallel search.
    parallel: bool,

    /// Minimum batch size for parallel search.
    parallel_batch_threshold: usize,
}

impl<T: DatapointValue> BruteForceSearcher<T> {
    /// Create a new brute-force searcher.
    pub fn new(dataset: DenseDataset<T>, distance_measure: DistanceMeasure) -> Self {
        Self {
            dataset: Arc::new(dataset),
            distance_measure,
            parallel: true,
            parallel_batch_threshold: 100,
        }
    }

    /// Create a searcher with shared dataset.
    pub fn with_shared_dataset(
        dataset: Arc<DenseDataset<T>>,
        distance_measure: DistanceMeasure,
    ) -> Self {
        Self {
            dataset,
            distance_measure,
            parallel: true,
            parallel_batch_threshold: 100,
        }
    }

    /// Set whether to use parallel search.
    pub fn set_parallel(&mut self, parallel: bool) {
        self.parallel = parallel;
    }

    /// Set the minimum batch size for parallel search.
    pub fn set_parallel_batch_threshold(&mut self, threshold: usize) {
        self.parallel_batch_threshold = threshold;
    }

    /// Get the dataset.
    pub fn dataset(&self) -> &DenseDataset<T> {
        &self.dataset
    }

    /// Get the distance measure.
    pub fn distance_measure(&self) -> DistanceMeasure {
        self.distance_measure
    }

    /// Search for the k nearest neighbors.
    pub fn search(&self, query: &[T], k: usize) -> Result<NNResultsVector> {
        if self.dataset.is_empty() {
            return Ok(Vec::new());
        }

        let query_ptr = DatapointPtr::dense(query);
        if query_ptr.dimensionality() != self.dataset.dimensionality() {
            return Err(ScannError::invalid_argument(format!(
                "Query dimensionality {} does not match dataset dimensionality {}",
                query_ptr.dimensionality(),
                self.dataset.dimensionality()
            )));
        }

        let k = k.min(self.dataset.size());
        self.search_impl(&query_ptr, k)
    }

    /// Internal search implementation.
    fn search_impl(&self, query: &DatapointPtr<'_, T>, k: usize) -> Result<NNResultsVector> {
        let num_points = self.dataset.size();

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
    fn compute_distances(&self, query: &DatapointPtr<'_, T>, distances: &mut [f32]) {
        let raw_data = self.dataset.raw_data();
        let stride = self.dataset.stride();
        let num_points = self.dataset.size();
        let query_values = query.values();

        match self.distance_measure {
            DistanceMeasure::SquaredL2 | DistanceMeasure::L2 => {
                one_to_many_squared_l2_strided(query_values, raw_data, stride, num_points, distances);
                if self.distance_measure == DistanceMeasure::L2 {
                    for d in distances.iter_mut() {
                        *d = d.sqrt();
                    }
                }
            }
            DistanceMeasure::DotProduct => {
                one_to_many_dot_product_strided(query_values, raw_data, stride, num_points, distances);
            }
            _ => {
                // Fallback to one-by-one
                for i in 0..num_points {
                    let db_ptr = self.dataset.get(i as DatapointIndex).unwrap();
                    distances[i] = self.distance_measure.distance(query, &db_ptr);
                }
            }
        }
    }

    /// Search for neighbors within a given radius.
    pub fn search_radius(&self, query: &[T], radius: f32) -> Result<NNResultsVector> {
        if self.dataset.is_empty() {
            return Ok(Vec::new());
        }

        let query_ptr = DatapointPtr::dense(query);
        if query_ptr.dimensionality() != self.dataset.dimensionality() {
            return Err(ScannError::invalid_argument(
                "Query dimensionality does not match dataset",
            ));
        }

        let num_points = self.dataset.size();
        let mut distances = vec![0.0f32; num_points];
        self.compute_distances(&query_ptr, &mut distances);

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
        queries: &[Vec<T>],
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
        queries: &[Vec<T>],
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
        queries: &[Vec<T>],
        k: usize,
    ) -> Result<Vec<NNResultsVector>> {
        queries
            .par_iter()
            .map(|q| self.search(q, k))
            .collect()
    }
}

impl<T: DatapointValue> Searcher<T> for BruteForceSearcher<T> {
    fn search_with_params(
        &self,
        query: &DatapointPtr<'_, T>,
        params: &SearchParameters,
    ) -> Result<SearchResult> {
        let k = params.num_neighbors.unwrap_or(10) as usize;
        let results = self.search_impl(query, k)?;
        Ok(SearchResult {
            neighbors: results.into_iter().map(|(idx, dist)| NNResult {
                index: idx,
                distance: dist,
                docid: None,
            }).collect(),
        })
    }

    fn search_batched_with_params(
        &self,
        queries: &[DatapointPtr<'_, T>],
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
        self.dataset.size()
    }

    fn dimensionality(&self) -> u64 {
        self.dataset.dimensionality()
    }
}

impl<T: DatapointValue> Clone for BruteForceSearcher<T> {
    fn clone(&self) -> Self {
        Self {
            dataset: Arc::clone(&self.dataset),
            distance_measure: self.distance_measure,
            parallel: self.parallel,
            parallel_batch_threshold: self.parallel_batch_threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dataset() -> DenseDataset<f32> {
        DenseDataset::from_vecs(vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ])
    }

    #[test]
    fn test_brute_force_search() {
        let dataset = create_test_dataset();
        let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

        let query = vec![0.0, 0.0, 0.0];
        let results = searcher.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 0); // Exact match
        assert!((results[0].1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_brute_force_search_all() {
        let dataset = create_test_dataset();
        let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

        let query = vec![0.5, 0.5, 0.5];
        let results = searcher.search(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_brute_force_search_dot_product() {
        let dataset = DenseDataset::from_vecs(vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ]);
        let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::DotProduct);

        let query = vec![1.0, 0.0];
        let results = searcher.search(&query, 3).unwrap();

        // Dot product distances are negative, so higher dot product = lower distance
        assert_eq!(results.len(), 3);
        // [1,0] dot [1,0] = 1, [1,0] dot [1,1] = 1, [1,0] dot [0,1] = 0
        // Negative: -1, -1, 0
        assert!(results[0].1 <= results[1].1);
    }

    #[test]
    fn test_brute_force_radius_search() {
        let dataset = create_test_dataset();
        let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

        let query = vec![0.0, 0.0, 0.0];
        let results = searcher.search_radius(&query, 1.5).unwrap();

        // Should include points with squared distance <= 1.5
        // Point 0: 0.0, Points 1,2,3: 1.0 each
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_brute_force_batched() {
        let dataset = create_test_dataset();
        let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

        let queries = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ];
        let results = searcher.search_batched(&queries, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[1].len(), 2);
    }

    #[test]
    fn test_brute_force_empty_dataset() {
        let dataset = DenseDataset::<f32>::new();
        let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

        let query = vec![1.0, 2.0, 3.0];
        let results = searcher.search(&query, 5).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_brute_force_dimension_mismatch() {
        let dataset = create_test_dataset();
        let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

        let query = vec![1.0, 2.0]; // Wrong dimension
        let result = searcher.search(&query, 5);

        assert!(result.is_err());
    }
}
