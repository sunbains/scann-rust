//! Datapoint representation.
//!
//! This module provides types for representing individual datapoints,
//! supporting both dense and sparse representations.

use crate::types::{DatapointValue, DimensionIndex};

/// A borrowed view of a datapoint.
///
/// This is analogous to the C++ DatapointPtr, providing a non-owning
/// view into datapoint data that can be either dense or sparse.
#[derive(Debug, Clone, Copy)]
pub struct DatapointPtr<'a, T: DatapointValue> {
    /// Values (all values for dense, non-zero values for sparse).
    values: &'a [T],

    /// Indices of non-zero entries (None for dense datapoints).
    indices: Option<&'a [DimensionIndex]>,

    /// Total dimensionality of the datapoint.
    dimensionality: DimensionIndex,
}

impl<'a, T: DatapointValue> DatapointPtr<'a, T> {
    /// Create a dense datapoint pointer.
    #[inline]
    pub fn dense(values: &'a [T]) -> Self {
        let dim = values.len() as DimensionIndex;
        Self {
            values,
            indices: None,
            dimensionality: dim,
        }
    }

    /// Create a dense datapoint pointer with explicit dimensionality.
    #[inline]
    pub fn dense_with_dim(values: &'a [T], dimensionality: DimensionIndex) -> Self {
        Self {
            values,
            indices: None,
            dimensionality,
        }
    }

    /// Create a sparse datapoint pointer.
    #[inline]
    pub fn sparse(
        values: &'a [T],
        indices: &'a [DimensionIndex],
        dimensionality: DimensionIndex,
    ) -> Self {
        debug_assert_eq!(
            values.len(),
            indices.len(),
            "Sparse datapoint must have equal number of values and indices"
        );
        Self {
            values,
            indices: Some(indices),
            dimensionality,
        }
    }

    /// Check if this datapoint is dense.
    #[inline]
    pub fn is_dense(&self) -> bool {
        self.indices.is_none()
    }

    /// Check if this datapoint is sparse.
    #[inline]
    pub fn is_sparse(&self) -> bool {
        self.indices.is_some()
    }

    /// Get the values slice.
    #[inline]
    pub fn values(&self) -> &[T] {
        self.values
    }

    /// Get the indices slice (None for dense datapoints).
    #[inline]
    pub fn indices(&self) -> Option<&[DimensionIndex]> {
        self.indices
    }

    /// Get the total dimensionality.
    #[inline]
    pub fn dimensionality(&self) -> DimensionIndex {
        self.dimensionality
    }

    /// Get the number of non-zero entries.
    /// For dense datapoints, this equals dimensionality.
    #[inline]
    pub fn nonzero_entries(&self) -> usize {
        self.values.len()
    }

    /// Get a value by dimension index.
    ///
    /// For dense datapoints, this is O(1).
    /// For sparse datapoints, this performs a binary search, O(log n).
    pub fn get(&self, dim: DimensionIndex) -> T {
        if self.is_dense() {
            if (dim as usize) < self.values.len() {
                self.values[dim as usize]
            } else {
                T::default()
            }
        } else {
            // Binary search for sparse
            let indices = self.indices.unwrap();
            match indices.binary_search(&dim) {
                Ok(idx) => self.values[idx],
                Err(_) => T::default(),
            }
        }
    }

    /// Convert to an owned Datapoint.
    pub fn to_owned(&self) -> Datapoint<T> {
        if self.is_dense() {
            Datapoint::dense(self.values.to_vec())
        } else {
            Datapoint::sparse(
                self.values.to_vec(),
                self.indices.unwrap().to_vec(),
                self.dimensionality,
            )
        }
    }

    /// Compute the L2 norm (Euclidean length).
    pub fn l2_norm(&self) -> f32 {
        let sum_sq: f32 = self.values.iter().map(|&v| {
            let f = v.to_f32();
            f * f
        }).sum();
        sum_sq.sqrt()
    }

    /// Compute the squared L2 norm.
    pub fn squared_l2_norm(&self) -> f32 {
        self.values.iter().map(|&v| {
            let f = v.to_f32();
            f * f
        }).sum()
    }
}

/// An owned datapoint.
///
/// This is analogous to the C++ Datapoint class, providing owned storage
/// for datapoint data that can be either dense or sparse.
#[derive(Debug, Clone)]
pub struct Datapoint<T: DatapointValue> {
    /// Values (all values for dense, non-zero values for sparse).
    values: Vec<T>,

    /// Indices of non-zero entries (empty for dense datapoints).
    indices: Vec<DimensionIndex>,

    /// Total dimensionality of the datapoint.
    dimensionality: DimensionIndex,

    /// Whether this is a sparse datapoint.
    is_sparse: bool,
}

impl<T: DatapointValue> Datapoint<T> {
    /// Create a dense datapoint from values.
    pub fn dense(values: Vec<T>) -> Self {
        let dim = values.len() as DimensionIndex;
        Self {
            values,
            indices: Vec::new(),
            dimensionality: dim,
            is_sparse: false,
        }
    }

    /// Create a dense datapoint with explicit dimensionality.
    pub fn dense_with_dim(values: Vec<T>, dimensionality: DimensionIndex) -> Self {
        Self {
            values,
            indices: Vec::new(),
            dimensionality,
            is_sparse: false,
        }
    }

    /// Create a sparse datapoint.
    pub fn sparse(
        values: Vec<T>,
        indices: Vec<DimensionIndex>,
        dimensionality: DimensionIndex,
    ) -> Self {
        debug_assert_eq!(
            values.len(),
            indices.len(),
            "Sparse datapoint must have equal number of values and indices"
        );
        Self {
            values,
            indices,
            dimensionality,
            is_sparse: true,
        }
    }

    /// Create an empty dense datapoint with the given dimensionality.
    pub fn zeros(dimensionality: DimensionIndex) -> Self {
        Self {
            values: vec![T::default(); dimensionality as usize],
            indices: Vec::new(),
            dimensionality,
            is_sparse: false,
        }
    }

    /// Check if this datapoint is dense.
    #[inline]
    pub fn is_dense(&self) -> bool {
        !self.is_sparse
    }

    /// Check if this datapoint is sparse.
    #[inline]
    pub fn is_sparse(&self) -> bool {
        self.is_sparse
    }

    /// Get a borrowed view of this datapoint.
    #[inline]
    pub fn as_ptr(&self) -> DatapointPtr<'_, T> {
        if self.is_sparse {
            DatapointPtr::sparse(&self.values, &self.indices, self.dimensionality)
        } else {
            DatapointPtr::dense_with_dim(&self.values, self.dimensionality)
        }
    }

    /// Get the values slice.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Get mutable access to values.
    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Get the indices slice (empty for dense datapoints).
    #[inline]
    pub fn indices(&self) -> &[DimensionIndex] {
        &self.indices
    }

    /// Get the total dimensionality.
    #[inline]
    pub fn dimensionality(&self) -> DimensionIndex {
        self.dimensionality
    }

    /// Get the number of non-zero entries.
    #[inline]
    pub fn nonzero_entries(&self) -> usize {
        self.values.len()
    }

    /// Get a value by dimension index.
    pub fn get(&self, dim: DimensionIndex) -> T {
        self.as_ptr().get(dim)
    }

    /// Set a value by dimension index (only for dense datapoints).
    pub fn set(&mut self, dim: DimensionIndex, value: T) {
        if self.is_dense() && (dim as usize) < self.values.len() {
            self.values[dim as usize] = value;
        }
    }

    /// Compute the L2 norm.
    pub fn l2_norm(&self) -> f32 {
        self.as_ptr().l2_norm()
    }

    /// Compute the squared L2 norm.
    pub fn squared_l2_norm(&self) -> f32 {
        self.as_ptr().squared_l2_norm()
    }

    /// Normalize this datapoint to unit length.
    pub fn normalize(&mut self) {
        let norm = self.l2_norm();
        if norm > 0.0 {
            for v in &mut self.values {
                *v = T::from_f32(v.to_f32() / norm);
            }
        }
    }

    /// Convert this datapoint to a dense representation.
    pub fn to_dense(&self) -> Self {
        if self.is_dense() {
            self.clone()
        } else {
            let mut values = vec![T::default(); self.dimensionality as usize];
            for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
                values[idx as usize] = val;
            }
            Self::dense_with_dim(values, self.dimensionality)
        }
    }
}

impl<T: DatapointValue> Default for Datapoint<T> {
    fn default() -> Self {
        Self::dense(Vec::new())
    }
}

impl<T: DatapointValue> From<Vec<T>> for Datapoint<T> {
    fn from(values: Vec<T>) -> Self {
        Self::dense(values)
    }
}

impl<T: DatapointValue> From<&[T]> for Datapoint<T> {
    fn from(values: &[T]) -> Self {
        Self::dense(values.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_datapoint() {
        let dp = Datapoint::dense(vec![1.0f32, 2.0, 3.0]);
        assert!(dp.is_dense());
        assert!(!dp.is_sparse());
        assert_eq!(dp.dimensionality(), 3);
        assert_eq!(dp.nonzero_entries(), 3);
        assert_eq!(dp.get(0), 1.0);
        assert_eq!(dp.get(1), 2.0);
        assert_eq!(dp.get(2), 3.0);
    }

    #[test]
    fn test_sparse_datapoint() {
        let dp = Datapoint::sparse(
            vec![1.0f32, 3.0],
            vec![0, 2],
            3,
        );
        assert!(dp.is_sparse());
        assert_eq!(dp.dimensionality(), 3);
        assert_eq!(dp.nonzero_entries(), 2);
        assert_eq!(dp.get(0), 1.0);
        assert_eq!(dp.get(1), 0.0);
        assert_eq!(dp.get(2), 3.0);
    }

    #[test]
    fn test_datapoint_ptr() {
        let values = vec![1.0f32, 2.0, 3.0];
        let ptr = DatapointPtr::dense(&values);
        assert!(ptr.is_dense());
        assert_eq!(ptr.dimensionality(), 3);
        assert_eq!(ptr.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_l2_norm() {
        let dp = Datapoint::dense(vec![3.0f32, 4.0]);
        assert!((dp.l2_norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut dp = Datapoint::dense(vec![3.0f32, 4.0]);
        dp.normalize();
        assert!((dp.l2_norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_to_dense() {
        let sparse = Datapoint::sparse(
            vec![1.0f32, 3.0],
            vec![0, 2],
            3,
        );
        let dense = sparse.to_dense();
        assert!(dense.is_dense());
        assert_eq!(dense.values(), &[1.0, 0.0, 3.0]);
    }
}
