//! Reordering utilities for exact distance recomputation.

use crate::data_format::{Dataset, DatapointPtr};
use crate::distance_measures::DistanceMeasure;
use crate::types::{DatapointValue, NNResultsVector};

/// Helper for reordering search results with exact distances.
pub struct ReorderingHelper<T: DatapointValue> {
    distance_measure: DistanceMeasure,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DatapointValue> ReorderingHelper<T> {
    /// Create a new reordering helper.
    pub fn new(distance_measure: DistanceMeasure) -> Self {
        Self {
            distance_measure,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Reorder candidates by computing exact distances.
    pub fn reorder<D: Dataset<T>>(
        &self,
        query: &DatapointPtr<'_, T>,
        candidates: &NNResultsVector,
        dataset: &D,
        final_k: usize,
    ) -> NNResultsVector {
        if candidates.is_empty() {
            return Vec::new();
        }

        // Compute exact distances for all candidates
        let mut results: Vec<_> = candidates
            .iter()
            .filter_map(|(idx, _)| {
                dataset.get(*idx).map(|dp| {
                    let exact_dist = self.distance_measure.distance(query, &dp);
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
        results.truncate(final_k);
        results
    }

    /// Reorder candidates with parallel distance computation.
    pub fn reorder_parallel<D: Dataset<T> + Sync>(
        &self,
        query: &DatapointPtr<'_, T>,
        candidates: &NNResultsVector,
        dataset: &D,
        final_k: usize,
    ) -> NNResultsVector
    where
        T: Sync,
    {
        use rayon::prelude::*;

        if candidates.len() < 100 {
            return self.reorder(query, candidates, dataset, final_k);
        }

        // Compute exact distances in parallel
        let query_owned = query.to_owned();
        let mut results: Vec<_> = candidates
            .par_iter()
            .filter_map(|(idx, _)| {
                dataset.get(*idx).map(|dp| {
                    let exact_dist = self.distance_measure.distance(&query_owned.as_ptr(), &dp);
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
        results.truncate(final_k);
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_format::DenseDataset;

    #[test]
    fn test_reordering() {
        let dataset = DenseDataset::from_vecs(vec![
            vec![0.0f32, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
        ]);

        let query = crate::data_format::Datapoint::dense(vec![0.0f32, 0.0]);
        let helper = ReorderingHelper::new(DistanceMeasure::SquaredL2);

        // Candidates in wrong order
        let candidates = vec![(2, 0.0), (1, 0.0), (3, 0.0), (0, 0.0)];
        let results = helper.reorder(&query.as_ptr(), &candidates, &dataset, 3);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 0); // Closest
        assert_eq!(results[1].0, 1);
        assert_eq!(results[2].0, 2);
    }
}
