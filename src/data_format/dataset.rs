//! Dataset types for ScaNN.
//!
//! This module provides dataset abstractions for storing collections
//! of datapoints, supporting both dense and sparse representations.

use crate::types::{DatapointIndex, DatapointValue, DimensionIndex, SIMD_ALIGNMENT, align_up};
use crate::data_format::datapoint::DatapointPtr;
use crate::data_format::docid::{DocId, DocIdCollection};
use crate::error::{Result, ScannError};
use aligned_vec::{AVec, ConstAlign};

/// Trait for dataset types.
pub trait Dataset<T: DatapointValue>: Send + Sync {
    /// Get the number of datapoints in the dataset.
    fn size(&self) -> usize;

    /// Check if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Get the dimensionality of datapoints.
    fn dimensionality(&self) -> DimensionIndex;

    /// Get a datapoint by index.
    fn get(&self, index: DatapointIndex) -> Option<DatapointPtr<'_, T>>;

    /// Check if datapoints are dense.
    fn is_dense(&self) -> bool;

    /// Check if datapoints are sparse.
    fn is_sparse(&self) -> bool {
        !self.is_dense()
    }

    /// Get a datapoint, panicking if out of bounds.
    fn get_unchecked(&self, index: DatapointIndex) -> DatapointPtr<'_, T> {
        self.get(index).expect("Index out of bounds")
    }
}

/// A dense dataset storing datapoints in contiguous memory.
///
/// This is analogous to the C++ DenseDataset, with support for
/// memory alignment and strided access.
pub struct DenseDataset<T: DatapointValue> {
    /// Contiguous storage for all values, aligned for SIMD.
    data: AVec<T, ConstAlign<64>>,

    /// Number of datapoints.
    num_points: usize,

    /// Dimensionality of each datapoint.
    dimensionality: DimensionIndex,

    /// Stride between datapoints (may be > dimensionality for alignment).
    stride: usize,

    /// Document IDs (optional).
    docids: Option<DocIdCollection>,
}

impl<T: DatapointValue> DenseDataset<T> {
    /// Create an empty dense dataset.
    pub fn new() -> Self {
        Self {
            data: AVec::new(64),
            num_points: 0,
            dimensionality: 0,
            stride: 0,
            docids: None,
        }
    }

    /// Create a dense dataset with the given capacity.
    pub fn with_capacity(capacity: usize, dimensionality: DimensionIndex) -> Self {
        let stride = Self::compute_stride(dimensionality);
        let mut data = AVec::new(64);
        data.reserve(capacity * stride);
        Self {
            data,
            num_points: 0,
            dimensionality,
            stride,
            docids: None,
        }
    }

    /// Compute the stride for a given dimensionality.
    fn compute_stride(dimensionality: DimensionIndex) -> usize {
        let dim = dimensionality as usize;
        // Align to cache line boundary (64 bytes)
        let elem_size = std::mem::size_of::<T>();
        let elems_per_line = SIMD_ALIGNMENT / elem_size;
        align_up(dim, elems_per_line)
    }

    /// Create a dense dataset from a vector of vectors.
    pub fn from_vecs(vecs: Vec<Vec<T>>) -> Self {
        if vecs.is_empty() {
            return Self::new();
        }

        let dimensionality = vecs[0].len() as DimensionIndex;
        let num_points = vecs.len();
        let stride = Self::compute_stride(dimensionality);

        let mut data: AVec<T, ConstAlign<64>> = AVec::new(64);
        data.resize(num_points * stride, T::default());

        for (i, vec) in vecs.iter().enumerate() {
            let offset = i * stride;
            data[offset..offset + vec.len()].copy_from_slice(vec);
        }

        Self {
            data,
            num_points,
            dimensionality,
            stride,
            docids: None,
        }
    }

    /// Create a dense dataset from a flat array.
    pub fn from_flat(data: Vec<T>, dimensionality: DimensionIndex) -> Result<Self> {
        let dim = dimensionality as usize;
        if dim == 0 {
            return Err(ScannError::invalid_argument("Dimensionality cannot be 0"));
        }
        if data.len() % dim != 0 {
            return Err(ScannError::invalid_argument(
                "Data length must be a multiple of dimensionality",
            ));
        }

        let num_points = data.len() / dim;
        let stride = Self::compute_stride(dimensionality);

        // If stride equals dimensionality, we can use the data directly
        if stride == dim {
            let mut aligned_data: AVec<T, ConstAlign<64>> = AVec::new(64);
            aligned_data.reserve(data.len());
            for val in data {
                aligned_data.push(val);
            }
            return Ok(Self {
                data: aligned_data,
                num_points,
                dimensionality,
                stride,
                docids: None,
            });
        }

        // Otherwise, we need to copy with stride
        let mut aligned_data: AVec<T, ConstAlign<64>> = AVec::new(64);
        aligned_data.resize(num_points * stride, T::default());

        for i in 0..num_points {
            let src_offset = i * dim;
            let dst_offset = i * stride;
            aligned_data[dst_offset..dst_offset + dim]
                .copy_from_slice(&data[src_offset..src_offset + dim]);
        }

        Ok(Self {
            data: aligned_data,
            num_points,
            dimensionality,
            stride,
            docids: None,
        })
    }

    /// Get the stride between datapoints.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Append a datapoint to the dataset.
    pub fn append(&mut self, datapoint: &DatapointPtr<'_, T>) -> Result<()> {
        if self.num_points == 0 {
            self.dimensionality = datapoint.dimensionality();
            self.stride = Self::compute_stride(self.dimensionality);
        } else if datapoint.dimensionality() != self.dimensionality {
            return Err(ScannError::invalid_argument(format!(
                "Datapoint dimensionality {} does not match dataset dimensionality {}",
                datapoint.dimensionality(),
                self.dimensionality
            )));
        }

        let old_len = self.data.len();
        self.data.resize(old_len + self.stride, T::default());

        if datapoint.is_dense() {
            let values = datapoint.values();
            self.data[old_len..old_len + values.len()].copy_from_slice(values);
        } else {
            // Convert sparse to dense
            let indices = datapoint.indices().unwrap();
            let values = datapoint.values();
            for (&idx, &val) in indices.iter().zip(values.iter()) {
                self.data[old_len + idx as usize] = val;
            }
        }

        self.num_points += 1;
        Ok(())
    }

    /// Append a datapoint with a document ID.
    pub fn append_with_docid(
        &mut self,
        datapoint: &DatapointPtr<'_, T>,
        docid: DocId,
    ) -> Result<()> {
        self.append(datapoint)?;
        if self.docids.is_none() {
            self.docids = Some(DocIdCollection::new());
        }
        self.docids.as_mut().unwrap().push(docid);
        Ok(())
    }

    /// Get raw access to the underlying data.
    pub fn raw_data(&self) -> &[T] {
        &self.data
    }

    /// Get the raw data pointer at a given index.
    pub fn raw_ptr(&self, index: DatapointIndex) -> *const T {
        debug_assert!((index as usize) < self.num_points);
        unsafe { self.data.as_ptr().add(index as usize * self.stride) }
    }

    /// Get document IDs.
    pub fn docids(&self) -> Option<&DocIdCollection> {
        self.docids.as_ref()
    }

    /// Clear the dataset.
    pub fn clear(&mut self) {
        self.data.clear();
        self.num_points = 0;
        self.docids = None;
    }

    /// Reserve capacity for additional datapoints.
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional * self.stride);
    }
}

impl<T: DatapointValue> Dataset<T> for DenseDataset<T> {
    fn size(&self) -> usize {
        self.num_points
    }

    fn dimensionality(&self) -> DimensionIndex {
        self.dimensionality
    }

    fn get(&self, index: DatapointIndex) -> Option<DatapointPtr<'_, T>> {
        let idx = index as usize;
        if idx >= self.num_points {
            return None;
        }

        let offset = idx * self.stride;
        let dim = self.dimensionality as usize;
        let values = &self.data[offset..offset + dim];
        Some(DatapointPtr::dense_with_dim(values, self.dimensionality))
    }

    fn is_dense(&self) -> bool {
        true
    }
}

impl<T: DatapointValue> Default for DenseDataset<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DatapointValue> Clone for DenseDataset<T> {
    fn clone(&self) -> Self {
        let mut data: AVec<T, ConstAlign<64>> = AVec::new(64);
        data.reserve(self.data.len());
        for &val in self.data.iter() {
            data.push(val);
        }
        Self {
            data,
            num_points: self.num_points,
            dimensionality: self.dimensionality,
            stride: self.stride,
            docids: self.docids.clone(),
        }
    }
}

/// A sparse dataset storing datapoints with explicit indices.
pub struct SparseDataset<T: DatapointValue> {
    /// Values for each datapoint.
    values: Vec<Vec<T>>,

    /// Indices for each datapoint.
    indices: Vec<Vec<DimensionIndex>>,

    /// Dimensionality of the sparse space.
    dimensionality: DimensionIndex,

    /// Document IDs (optional).
    docids: Option<DocIdCollection>,
}

impl<T: DatapointValue> SparseDataset<T> {
    /// Create an empty sparse dataset.
    pub fn new(dimensionality: DimensionIndex) -> Self {
        Self {
            values: Vec::new(),
            indices: Vec::new(),
            dimensionality,
            docids: None,
        }
    }

    /// Create a sparse dataset with the given capacity.
    pub fn with_capacity(capacity: usize, dimensionality: DimensionIndex) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            indices: Vec::with_capacity(capacity),
            dimensionality,
            docids: None,
        }
    }

    /// Append a sparse datapoint.
    pub fn append(
        &mut self,
        values: Vec<T>,
        indices: Vec<DimensionIndex>,
    ) -> Result<()> {
        if values.len() != indices.len() {
            return Err(ScannError::invalid_argument(
                "Values and indices must have the same length",
            ));
        }
        self.values.push(values);
        self.indices.push(indices);
        Ok(())
    }

    /// Append a datapoint from a DatapointPtr.
    pub fn append_datapoint(&mut self, datapoint: &DatapointPtr<'_, T>) -> Result<()> {
        if datapoint.is_sparse() {
            self.values.push(datapoint.values().to_vec());
            self.indices.push(datapoint.indices().unwrap().to_vec());
        } else {
            // Convert dense to sparse (only non-zero values)
            let mut values = Vec::new();
            let mut indices = Vec::new();
            for (i, &v) in datapoint.values().iter().enumerate() {
                if v.to_f32() != 0.0 {
                    values.push(v);
                    indices.push(i as DimensionIndex);
                }
            }
            self.values.push(values);
            self.indices.push(indices);
        }
        Ok(())
    }

    /// Get the number of non-zero entries for a datapoint.
    pub fn nnz(&self, index: DatapointIndex) -> usize {
        self.values.get(index as usize).map(|v| v.len()).unwrap_or(0)
    }

    /// Clear the dataset.
    pub fn clear(&mut self) {
        self.values.clear();
        self.indices.clear();
        self.docids = None;
    }
}

impl<T: DatapointValue> Dataset<T> for SparseDataset<T> {
    fn size(&self) -> usize {
        self.values.len()
    }

    fn dimensionality(&self) -> DimensionIndex {
        self.dimensionality
    }

    fn get(&self, index: DatapointIndex) -> Option<DatapointPtr<'_, T>> {
        let idx = index as usize;
        if idx >= self.values.len() {
            return None;
        }

        Some(DatapointPtr::sparse(
            &self.values[idx],
            &self.indices[idx],
            self.dimensionality,
        ))
    }

    fn is_dense(&self) -> bool {
        false
    }
}

impl<T: DatapointValue> Clone for SparseDataset<T> {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
            indices: self.indices.clone(),
            dimensionality: self.dimensionality,
            docids: self.docids.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_format::datapoint::Datapoint;

    #[test]
    fn test_dense_dataset_from_vecs() {
        let data = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let dataset = DenseDataset::from_vecs(data);

        assert_eq!(dataset.size(), 3);
        assert_eq!(dataset.dimensionality(), 3);
        assert!(dataset.is_dense());

        let dp0 = dataset.get(0).unwrap();
        assert_eq!(dp0.values(), &[1.0, 2.0, 3.0]);

        let dp1 = dataset.get(1).unwrap();
        assert_eq!(dp1.values(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_dense_dataset_from_flat() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dataset = DenseDataset::from_flat(data, 3).unwrap();

        assert_eq!(dataset.size(), 2);
        assert_eq!(dataset.dimensionality(), 3);

        let dp0 = dataset.get(0).unwrap();
        assert_eq!(dp0.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dense_dataset_append() {
        let mut dataset = DenseDataset::<f32>::new();

        let dp1 = Datapoint::dense(vec![1.0, 2.0, 3.0]);
        dataset.append(&dp1.as_ptr()).unwrap();

        let dp2 = Datapoint::dense(vec![4.0, 5.0, 6.0]);
        dataset.append(&dp2.as_ptr()).unwrap();

        assert_eq!(dataset.size(), 2);
        assert_eq!(dataset.get(0).unwrap().values(), &[1.0, 2.0, 3.0]);
        assert_eq!(dataset.get(1).unwrap().values(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sparse_dataset() {
        let mut dataset = SparseDataset::<f32>::new(5);

        dataset.append(vec![1.0, 2.0], vec![0, 3]).unwrap();
        dataset.append(vec![3.0], vec![2]).unwrap();

        assert_eq!(dataset.size(), 2);
        assert_eq!(dataset.dimensionality(), 5);
        assert!(!dataset.is_dense());

        let dp0 = dataset.get(0).unwrap();
        assert_eq!(dp0.values(), &[1.0, 2.0]);
        assert_eq!(dp0.indices().unwrap(), &[0, 3]);
    }

    #[test]
    fn test_dataset_out_of_bounds() {
        let dataset = DenseDataset::from_vecs(vec![vec![1.0f32, 2.0]]);
        assert!(dataset.get(0).is_some());
        assert!(dataset.get(1).is_none());
    }
}
