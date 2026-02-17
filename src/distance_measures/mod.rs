//! Distance measures for ScaNN.
//!
//! This module provides various distance measures for comparing datapoints,
//! with SIMD-optimized implementations for high performance.

mod one_to_one;
mod one_to_many;
pub mod one_to_many_asymmetric;
pub mod many_to_many;
mod sparse;

pub use one_to_one::*;
pub use one_to_many::*;
pub use one_to_many_asymmetric::{
    one_to_many_int8_float_dot_product,
    one_to_many_int8_float_squared_l2,
    one_to_many_bf16_float_dot_product,
    one_to_many_bf16_float_squared_l2,
    one_to_many_fp8_float_dot_product,
    one_to_many_fp8_float_squared_l2,
};
pub use many_to_many::*;
pub use sparse::*;

use crate::data_format::DatapointPtr;
use crate::types::DatapointValue;
use serde::{Deserialize, Serialize};

/// Enum representing the available distance measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Default)]
pub enum DistanceMeasure {
    /// L1 (Manhattan) distance: sum of absolute differences.
    L1,

    /// L2 (Euclidean) distance: square root of sum of squared differences.
    L2,

    /// Squared L2 distance: sum of squared differences (faster than L2).
    #[default]
    SquaredL2,

    /// Cosine distance: 1 - cosine_similarity.
    Cosine,

    /// Dot product (negative for similarity search).
    DotProduct,

    /// Hamming distance for binary vectors.
    Hamming,

    /// Limited inner product.
    LimitedInnerProduct,

    /// General inner product.
    GeneralInnerProduct,

    /// Jaccard distance for set similarity.
    Jaccard,

    /// Non-zero intersection (counts overlapping non-zero dimensions).
    NonZeroIntersect,

    /// Dice coefficient distance.
    Dice,
}

impl DistanceMeasure {
    /// Compute the distance between two dense datapoints.
    pub fn distance<T: DatapointValue>(
        &self,
        a: &DatapointPtr<'_, T>,
        b: &DatapointPtr<'_, T>,
    ) -> f32 {
        match self {
            DistanceMeasure::L1 => l1_distance(a, b),
            DistanceMeasure::L2 => l2_distance(a, b),
            DistanceMeasure::SquaredL2 => squared_l2_distance(a, b),
            DistanceMeasure::Cosine => cosine_distance(a, b),
            DistanceMeasure::DotProduct => negative_dot_product(a, b),
            DistanceMeasure::Hamming => hamming_distance(a, b),
            DistanceMeasure::LimitedInnerProduct => limited_inner_product(a, b),
            DistanceMeasure::GeneralInnerProduct => general_inner_product(a, b),
            DistanceMeasure::Jaccard => {
                // For Jaccard, we need sparse indices
                if a.is_sparse() && b.is_sparse() {
                    jaccard_distance_sparse(a.indices().unwrap(), b.indices().unwrap())
                } else {
                    // Fall back to L2 for dense vectors
                    squared_l2_distance(a, b)
                }
            }
            DistanceMeasure::NonZeroIntersect => {
                if a.is_sparse() && b.is_sparse() {
                    non_zero_intersect_distance(
                        a.values(),
                        a.indices().unwrap(),
                        b.values(),
                        b.indices().unwrap(),
                    )
                } else {
                    // Count non-zero overlaps for dense
                    -(a.values().iter().zip(b.values().iter())
                        .filter(|(&x, &y)| x.to_f32() != 0.0 && y.to_f32() != 0.0)
                        .count() as f32)
                }
            }
            DistanceMeasure::Dice => {
                if a.is_sparse() && b.is_sparse() {
                    dice_distance_sparse(a.indices().unwrap(), b.indices().unwrap())
                } else {
                    squared_l2_distance(a, b)
                }
            }
        }
    }

    /// Compute distances from one query to many database points.
    pub fn one_to_many<T: DatapointValue>(
        &self,
        query: &DatapointPtr<'_, T>,
        database: &[&[T]],
        results: &mut [f32],
    ) {
        match self {
            DistanceMeasure::SquaredL2 => {
                one_to_many_squared_l2(query, database, results);
            }
            DistanceMeasure::DotProduct => {
                one_to_many_dot_product(query, database, results);
            }
            _ => {
                // Fallback to one-to-one for other distances
                for (i, db_point) in database.iter().enumerate() {
                    let db_ptr = DatapointPtr::dense(db_point);
                    results[i] = self.distance(query, &db_ptr);
                }
            }
        }
    }

    /// Check if lower values indicate more similar points.
    pub fn is_lower_better(&self) -> bool {
        // All distance measures use lower = more similar convention
        true
    }

    /// Check if this distance measure requires sparse vectors.
    pub fn requires_sparse(&self) -> bool {
        matches!(self, DistanceMeasure::Jaccard | DistanceMeasure::Dice)
    }

    /// Get the name of this distance measure.
    pub fn name(&self) -> &'static str {
        match self {
            DistanceMeasure::L1 => "L1",
            DistanceMeasure::L2 => "L2",
            DistanceMeasure::SquaredL2 => "SquaredL2",
            DistanceMeasure::Cosine => "Cosine",
            DistanceMeasure::DotProduct => "DotProduct",
            DistanceMeasure::Hamming => "Hamming",
            DistanceMeasure::LimitedInnerProduct => "LimitedInnerProduct",
            DistanceMeasure::GeneralInnerProduct => "GeneralInnerProduct",
            DistanceMeasure::Jaccard => "Jaccard",
            DistanceMeasure::NonZeroIntersect => "NonZeroIntersect",
            DistanceMeasure::Dice => "Dice",
        }
    }
}


impl std::fmt::Display for DistanceMeasure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_format::Datapoint;

    #[test]
    fn test_distance_measure_enum() {
        let dp1 = Datapoint::dense(vec![1.0f32, 0.0, 0.0]);
        let dp2 = Datapoint::dense(vec![0.0f32, 1.0, 0.0]);

        let l2 = DistanceMeasure::L2.distance(&dp1.as_ptr(), &dp2.as_ptr());
        assert!((l2 - 2.0_f32.sqrt()).abs() < 1e-6);

        let sq_l2 = DistanceMeasure::SquaredL2.distance(&dp1.as_ptr(), &dp2.as_ptr());
        assert!((sq_l2 - 2.0).abs() < 1e-6);
    }
}
