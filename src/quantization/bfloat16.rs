//! BFloat16 (Brain Floating Point) support.
//!
//! BFloat16 is a 16-bit floating point format that maintains the same
//! exponent range as float32 but with reduced mantissa precision.

use crate::data_format::{Dataset, DenseDataset};
use crate::error::{Result, ScannError};
use crate::types::DatapointIndex;
use half::bf16;

/// Convert f32 to bf16.
#[inline]
pub fn f32_to_bf16(value: f32) -> bf16 {
    bf16::from_f32(value)
}

/// Convert bf16 to f32.
#[inline]
pub fn bf16_to_f32(value: bf16) -> f32 {
    value.to_f32()
}

/// Convert a slice of f32 to bf16.
pub fn f32_slice_to_bf16(input: &[f32]) -> Vec<bf16> {
    input.iter().map(|&v| f32_to_bf16(v)).collect()
}

/// Convert a slice of bf16 to f32.
pub fn bf16_slice_to_f32(input: &[bf16]) -> Vec<f32> {
    input.iter().map(|&v| bf16_to_f32(v)).collect()
}

/// A dataset stored in BFloat16 format.
#[derive(Clone)]
pub struct BFloat16Dataset {
    /// BFloat16 data storage.
    data: Vec<bf16>,
    /// Number of datapoints.
    num_points: usize,
    /// Dimensionality.
    dimensionality: usize,
    /// Stride between datapoints.
    stride: usize,
}

impl BFloat16Dataset {
    /// Create an empty BFloat16 dataset.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            num_points: 0,
            dimensionality: 0,
            stride: 0,
        }
    }

    /// Create from a dense f32 dataset.
    pub fn from_f32_dataset(dataset: &DenseDataset<f32>) -> Self {
        if dataset.is_empty() {
            return Self::new();
        }

        let num_points = dataset.size();
        let dimensionality = dataset.dimensionality() as usize;
        let stride = dimensionality;

        let mut data = Vec::with_capacity(num_points * stride);

        for i in 0..num_points {
            if let Some(dp) = dataset.get(i as u32) {
                for &val in dp.values() {
                    data.push(f32_to_bf16(val));
                }
            }
        }

        Self {
            data,
            num_points,
            dimensionality,
            stride,
        }
    }

    /// Create from vectors.
    pub fn from_vecs(vecs: Vec<Vec<f32>>) -> Self {
        if vecs.is_empty() {
            return Self::new();
        }

        let dimensionality = vecs[0].len();
        let num_points = vecs.len();
        let stride = dimensionality;

        let mut data = Vec::with_capacity(num_points * stride);

        for vec in &vecs {
            for &val in vec {
                data.push(f32_to_bf16(val));
            }
        }

        Self {
            data,
            num_points,
            dimensionality,
            stride,
        }
    }

    /// Get a datapoint as BFloat16 slice.
    pub fn get_bf16(&self, index: DatapointIndex) -> Option<&[bf16]> {
        let idx = index as usize;
        if idx >= self.num_points {
            return None;
        }

        let offset = idx * self.stride;
        Some(&self.data[offset..offset + self.dimensionality])
    }

    /// Get a datapoint converted to f32.
    pub fn get_f32(&self, index: DatapointIndex) -> Option<Vec<f32>> {
        self.get_bf16(index).map(bf16_slice_to_f32)
    }

    /// Get the number of datapoints.
    pub fn size(&self) -> usize {
        self.num_points
    }

    /// Get the dimensionality.
    pub fn dimensionality(&self) -> usize {
        self.dimensionality
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.num_points == 0
    }

    /// Convert to f32 dense dataset.
    pub fn to_f32_dataset(&self) -> DenseDataset<f32> {
        let mut vecs = Vec::with_capacity(self.num_points);
        for i in 0..self.num_points {
            if let Some(f32_vec) = self.get_f32(i as u32) {
                vecs.push(f32_vec);
            }
        }
        DenseDataset::from_vecs(vecs)
    }

    /// Get raw data.
    pub fn raw_data(&self) -> &[bf16] {
        &self.data
    }

    /// Compute squared L2 distance between a f32 query and a bf16 datapoint.
    pub fn squared_l2_distance(&self, query: &[f32], index: DatapointIndex) -> Option<f32> {
        let bf16_data = self.get_bf16(index)?;

        let mut sum = 0.0f32;
        for (i, &bf16_val) in bf16_data.iter().enumerate() {
            let f32_val = bf16_to_f32(bf16_val);
            let diff = query.get(i).copied().unwrap_or(0.0) - f32_val;
            sum += diff * diff;
        }

        Some(sum)
    }

    /// Compute dot product between a f32 query and a bf16 datapoint.
    pub fn dot_product(&self, query: &[f32], index: DatapointIndex) -> Option<f32> {
        let bf16_data = self.get_bf16(index)?;

        let mut sum = 0.0f32;
        for (i, &bf16_val) in bf16_data.iter().enumerate() {
            let f32_val = bf16_to_f32(bf16_val);
            sum += query.get(i).copied().unwrap_or(0.0) * f32_val;
        }

        Some(sum)
    }

    /// Append a datapoint.
    pub fn append(&mut self, values: &[f32]) -> Result<()> {
        if self.num_points == 0 {
            self.dimensionality = values.len();
            self.stride = self.dimensionality;
        } else if values.len() != self.dimensionality {
            return Err(ScannError::invalid_argument(
                "Datapoint dimensionality mismatch",
            ));
        }

        for &val in values {
            self.data.push(f32_to_bf16(val));
        }
        self.num_points += 1;

        Ok(())
    }

    /// Reserve capacity.
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional * self.stride);
    }

    /// Clear the dataset.
    pub fn clear(&mut self) {
        self.data.clear();
        self.num_points = 0;
    }
}

impl Default for BFloat16Dataset {
    fn default() -> Self {
        Self::new()
    }
}

/// SIMD-accelerated operations for BFloat16.
pub mod simd {
    use super::*;

    /// Compute dot product between f32 query and bf16 database vector.
    #[cfg(feature = "simd")]
    pub fn dot_product_f32_bf16(query: &[f32], database: &[bf16]) -> f32 {
        use wide::f32x8;

        let len = query.len().min(database.len());
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum = f32x8::ZERO;

        for i in 0..chunks {
            let offset = i * 8;
            let q = f32x8::new([
                query[offset],
                query[offset + 1],
                query[offset + 2],
                query[offset + 3],
                query[offset + 4],
                query[offset + 5],
                query[offset + 6],
                query[offset + 7],
            ]);
            let db = f32x8::new([
                bf16_to_f32(database[offset]),
                bf16_to_f32(database[offset + 1]),
                bf16_to_f32(database[offset + 2]),
                bf16_to_f32(database[offset + 3]),
                bf16_to_f32(database[offset + 4]),
                bf16_to_f32(database[offset + 5]),
                bf16_to_f32(database[offset + 6]),
                bf16_to_f32(database[offset + 7]),
            ]);
            sum += q * db;
        }

        let mut result: f32 = sum.reduce_add();

        for i in (len - remainder)..len {
            result += query[i] * bf16_to_f32(database[i]);
        }

        result
    }

    #[cfg(not(feature = "simd"))]
    pub fn dot_product_f32_bf16(query: &[f32], database: &[bf16]) -> f32 {
        query
            .iter()
            .zip(database.iter())
            .map(|(&q, &d)| q * bf16_to_f32(d))
            .sum()
    }

    /// Compute squared L2 distance between f32 query and bf16 database vector.
    #[cfg(feature = "simd")]
    pub fn squared_l2_f32_bf16(query: &[f32], database: &[bf16]) -> f32 {
        use wide::f32x8;

        let len = query.len().min(database.len());
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum = f32x8::ZERO;

        for i in 0..chunks {
            let offset = i * 8;
            let q = f32x8::new([
                query[offset],
                query[offset + 1],
                query[offset + 2],
                query[offset + 3],
                query[offset + 4],
                query[offset + 5],
                query[offset + 6],
                query[offset + 7],
            ]);
            let db = f32x8::new([
                bf16_to_f32(database[offset]),
                bf16_to_f32(database[offset + 1]),
                bf16_to_f32(database[offset + 2]),
                bf16_to_f32(database[offset + 3]),
                bf16_to_f32(database[offset + 4]),
                bf16_to_f32(database[offset + 5]),
                bf16_to_f32(database[offset + 6]),
                bf16_to_f32(database[offset + 7]),
            ]);
            let diff = q - db;
            sum += diff * diff;
        }

        let mut result: f32 = sum.reduce_add();

        for i in (len - remainder)..len {
            let diff = query[i] - bf16_to_f32(database[i]);
            result += diff * diff;
        }

        result
    }

    #[cfg(not(feature = "simd"))]
    pub fn squared_l2_f32_bf16(query: &[f32], database: &[bf16]) -> f32 {
        query
            .iter()
            .zip(database.iter())
            .map(|(&q, &d)| {
                let diff = q - bf16_to_f32(d);
                diff * diff
            })
            .sum()
    }
}

/// Float16 (IEEE 754 half-precision) support.
pub mod float16 {
    use half::f16;

    /// Convert f32 to f16.
    #[inline]
    pub fn f32_to_f16(value: f32) -> f16 {
        f16::from_f32(value)
    }

    /// Convert f16 to f32.
    #[inline]
    pub fn f16_to_f32(value: f16) -> f32 {
        value.to_f32()
    }

    /// Convert a slice of f32 to f16.
    pub fn f32_slice_to_f16(input: &[f32]) -> Vec<f16> {
        input.iter().map(|&v| f32_to_f16(v)).collect()
    }

    /// Convert a slice of f16 to f32.
    pub fn f16_slice_to_f32(input: &[f16]) -> Vec<f32> {
        input.iter().map(|&v| f16_to_f32(v)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_conversion() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 100.0, -0.001, 3.14159];

        for &val in &values {
            let bf16_val = f32_to_bf16(val);
            let recovered = bf16_to_f32(bf16_val);

            // BFloat16 has less precision, allow some error
            if val != 0.0 {
                let rel_error = ((val - recovered) / val).abs();
                assert!(rel_error < 0.01, "val={}, recovered={}", val, recovered);
            }
        }
    }

    #[test]
    fn test_bf16_dataset() {
        let data = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![-1.0, 0.0, 1.0],
        ];

        let dataset = BFloat16Dataset::from_vecs(data.clone());

        assert_eq!(dataset.size(), 3);
        assert_eq!(dataset.dimensionality(), 3);

        // Check conversion back
        let f32_vec = dataset.get_f32(1).unwrap();
        assert!((f32_vec[0] - 4.0).abs() < 0.01);
        assert!((f32_vec[1] - 5.0).abs() < 0.01);
        assert!((f32_vec[2] - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_bf16_distance() {
        let data = vec![
            vec![1.0f32, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];

        let dataset = BFloat16Dataset::from_vecs(data);

        let query = vec![1.0f32, 0.0, 0.0];
        let dist0 = dataset.squared_l2_distance(&query, 0).unwrap();
        let dist1 = dataset.squared_l2_distance(&query, 1).unwrap();

        assert!(dist0 < 0.01); // Same point
        assert!((dist1 - 2.0).abs() < 0.1); // Distance should be ~2
    }

    #[test]
    fn test_bf16_dot_product() {
        let data = vec![vec![1.0f32, 2.0, 3.0]];
        let dataset = BFloat16Dataset::from_vecs(data);

        let query = vec![1.0f32, 2.0, 3.0];
        let dot = dataset.dot_product(&query, 0).unwrap();

        assert!((dot - 14.0).abs() < 0.1); // 1*1 + 2*2 + 3*3 = 14
    }

    #[test]
    fn test_simd_operations() {
        let query = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let database: Vec<bf16> = query.iter().map(|&v| f32_to_bf16(v)).collect();

        let dot = simd::dot_product_f32_bf16(&query, &database);
        let expected: f32 = query.iter().map(|x| x * x).sum();

        assert!((dot - expected).abs() < 1.0);

        let dist = simd::squared_l2_f32_bf16(&query, &database);
        // Distance to self should be small
        assert!(dist < 1.0);
    }
}
