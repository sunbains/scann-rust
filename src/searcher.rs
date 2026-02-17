//! Searcher traits and common types.
//!
//! This module defines the core searcher interface used by all search implementations.

use crate::data_format::{DatapointPtr, DocId};
use crate::error::Result;
use crate::types::{DatapointIndex, DatapointValue, NNResultsVector};
use serde::{Deserialize, Serialize};

/// Parameters for a single search query.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchParameters {
    /// Number of neighbors to return.
    pub num_neighbors: Option<u32>,

    /// Pre-reordering number of neighbors (for approximate search).
    pub pre_reordering_num_neighbors: Option<u32>,

    /// Epsilon for approximate search (0 = exact).
    pub pre_reordering_epsilon: Option<f32>,

    /// Post-reordering epsilon.
    pub post_reordering_epsilon: Option<f32>,

    /// Number of leaves to search (for tree-based search).
    pub num_leaves_to_search: Option<u32>,

    /// Enable crowding diversity.
    pub crowding_enabled: Option<bool>,
}

impl SearchParameters {
    /// Create new parameters with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of neighbors to return.
    pub fn with_num_neighbors(mut self, k: u32) -> Self {
        self.num_neighbors = Some(k);
        self
    }

    /// Set the pre-reordering number of neighbors.
    pub fn with_pre_reordering_neighbors(mut self, k: u32) -> Self {
        self.pre_reordering_num_neighbors = Some(k);
        self
    }

    /// Set the number of leaves to search.
    pub fn with_leaves_to_search(mut self, n: u32) -> Self {
        self.num_leaves_to_search = Some(n);
        self
    }

    /// Set the epsilon for approximate search.
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.pre_reordering_epsilon = Some(epsilon);
        self
    }
}

/// A single nearest neighbor result.
#[derive(Debug, Clone)]
pub struct NNResult {
    /// Index of the datapoint in the dataset.
    pub index: DatapointIndex,

    /// Distance to the query point.
    pub distance: f32,

    /// Optional document ID.
    pub docid: Option<DocId>,
}

impl NNResult {
    /// Create a new result.
    pub fn new(index: DatapointIndex, distance: f32) -> Self {
        Self {
            index,
            distance,
            docid: None,
        }
    }

    /// Create a new result with a document ID.
    pub fn with_docid(index: DatapointIndex, distance: f32, docid: DocId) -> Self {
        Self {
            index,
            distance,
            docid: Some(docid),
        }
    }
}

/// Result of a search query.
#[derive(Debug, Clone, Default)]
pub struct SearchResult {
    /// Nearest neighbors sorted by distance.
    pub neighbors: Vec<NNResult>,
}

impl SearchResult {
    /// Create an empty result.
    pub fn new() -> Self {
        Self {
            neighbors: Vec::new(),
        }
    }

    /// Create a result from neighbor pairs.
    pub fn from_pairs(pairs: NNResultsVector) -> Self {
        Self {
            neighbors: pairs
                .into_iter()
                .map(|(idx, dist)| NNResult::new(idx, dist))
                .collect(),
        }
    }

    /// Get the number of results.
    pub fn len(&self) -> usize {
        self.neighbors.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.neighbors.is_empty()
    }

    /// Get the indices of the neighbors.
    pub fn indices(&self) -> Vec<DatapointIndex> {
        self.neighbors.iter().map(|r| r.index).collect()
    }

    /// Get the distances to the neighbors.
    pub fn distances(&self) -> Vec<f32> {
        self.neighbors.iter().map(|r| r.distance).collect()
    }

    /// Get the top result if available.
    pub fn top(&self) -> Option<&NNResult> {
        self.neighbors.first()
    }
}

/// Trait for nearest neighbor searchers.
pub trait Searcher<T: DatapointValue>: Send + Sync {
    /// Search for nearest neighbors with the given parameters.
    fn search_with_params(
        &self,
        query: &DatapointPtr<'_, T>,
        params: &SearchParameters,
    ) -> Result<SearchResult>;

    /// Search for nearest neighbors with default parameters.
    fn search(&self, query: &DatapointPtr<'_, T>, k: u32) -> Result<SearchResult> {
        let params = SearchParameters::new().with_num_neighbors(k);
        self.search_with_params(query, &params)
    }

    /// Batched search for multiple queries.
    fn search_batched_with_params(
        &self,
        queries: &[DatapointPtr<'_, T>],
        params: &[SearchParameters],
    ) -> Result<Vec<SearchResult>>;

    /// Batched search with default parameters.
    fn search_batched(
        &self,
        queries: &[DatapointPtr<'_, T>],
        k: u32,
    ) -> Result<Vec<SearchResult>> {
        let params: Vec<_> = (0..queries.len())
            .map(|_| SearchParameters::new().with_num_neighbors(k))
            .collect();
        self.search_batched_with_params(queries, &params)
    }

    /// Get the size of the dataset.
    fn dataset_size(&self) -> usize;

    /// Get the dimensionality of the dataset.
    fn dimensionality(&self) -> u64;
}

/// Builder for creating searchers.
pub struct SearcherBuilder<T: DatapointValue> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DatapointValue> SearcherBuilder<T> {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DatapointValue> Default for SearcherBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_parameters() {
        let params = SearchParameters::new()
            .with_num_neighbors(10)
            .with_pre_reordering_neighbors(100)
            .with_epsilon(0.1);

        assert_eq!(params.num_neighbors, Some(10));
        assert_eq!(params.pre_reordering_num_neighbors, Some(100));
        assert_eq!(params.pre_reordering_epsilon, Some(0.1));
    }

    #[test]
    fn test_nn_result() {
        let result = NNResult::new(42, 1.5);
        assert_eq!(result.index, 42);
        assert_eq!(result.distance, 1.5);
        assert!(result.docid.is_none());

        let result_with_docid = NNResult::with_docid(42, 1.5, DocId::string("doc42"));
        assert!(result_with_docid.docid.is_some());
    }

    #[test]
    fn test_search_result() {
        let pairs = vec![(0, 1.0), (1, 2.0), (2, 3.0)];
        let result = SearchResult::from_pairs(pairs);

        assert_eq!(result.len(), 3);
        assert_eq!(result.indices(), vec![0, 1, 2]);
        assert_eq!(result.distances(), vec![1.0, 2.0, 3.0]);
        assert_eq!(result.top().unwrap().index, 0);
    }
}
