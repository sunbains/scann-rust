//! Lookup tables for fast distance approximation.
//!
//! This module provides lookup tables that precompute distances
//! from query subvectors to all centroids.

use crate::hashes::codebook::Codebook;
use serde::{Deserialize, Serialize};

/// Format for lookup table values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum LutFormat {
    /// 32-bit float values.
    #[default]
    Float32,
    /// 16-bit float values (half precision).
    Float16,
    /// 8-bit integer values (quantized).
    Int8,
    /// 16-bit integer values (quantized).
    Int16,
}


/// Lookup table for a single query.
///
/// Contains precomputed distances from the query to all centroids
/// in each subspace.
#[derive(Debug, Clone)]
pub struct LookupTable {
    /// Distance values organized by subspace.
    /// Shape: [num_subspaces][num_codes]
    distances: Vec<Vec<f32>>,

    /// Number of subspaces.
    num_subspaces: usize,

    /// Number of codes per subspace.
    num_codes: usize,

    /// Format of the LUT.
    format: LutFormat,
}

impl LookupTable {
    /// Create a new lookup table from a codebook and query.
    pub fn from_query(codebook: &Codebook, query: &[f32]) -> Self {
        let num_subspaces = codebook.num_subspaces();
        let num_codes = codebook.num_codes();
        let dims_per_subspace = codebook.dims_per_subspace;

        let distances: Vec<Vec<f32>> = codebook
            .subspaces
            .iter()
            .enumerate()
            .map(|(s, subspace_codebook)| {
                let start = s * dims_per_subspace;
                let end = start + dims_per_subspace;
                let query_subvector = &query[start..end];
                subspace_codebook.compute_distances(query_subvector)
            })
            .collect();

        Self {
            distances,
            num_subspaces,
            num_codes,
            format: LutFormat::Float32,
        }
    }

    /// Compute the approximate distance for a set of codes.
    #[inline]
    pub fn compute_distance(&self, codes: &[u8]) -> f32 {
        debug_assert_eq!(codes.len(), self.num_subspaces);

        let mut sum = 0.0f32;
        for (s, &code) in codes.iter().enumerate() {
            sum += self.distances[s][code as usize];
        }
        sum
    }

    /// Compute distances for multiple code vectors.
    pub fn compute_distances_batch(&self, codes_batch: &[Vec<u8>], results: &mut [f32]) {
        debug_assert_eq!(codes_batch.len(), results.len());

        for (i, codes) in codes_batch.iter().enumerate() {
            results[i] = self.compute_distance(codes);
        }
    }

    /// Get raw distance values for a subspace.
    pub fn subspace_distances(&self, subspace: usize) -> &[f32] {
        &self.distances[subspace]
    }

    /// Get the number of subspaces.
    pub fn num_subspaces(&self) -> usize {
        self.num_subspaces
    }

    /// Get the number of codes.
    pub fn num_codes(&self) -> usize {
        self.num_codes
    }

    /// Get the format.
    pub fn format(&self) -> LutFormat {
        self.format
    }

    /// Convert to int8 format for faster computation.
    pub fn to_int8(&self) -> LookupTableInt8 {
        // Find min and max values
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for subspace in &self.distances {
            for &d in subspace {
                min_val = min_val.min(d);
                max_val = max_val.max(d);
            }
        }

        let scale = if max_val > min_val {
            255.0 / (max_val - min_val)
        } else {
            1.0
        };

        let distances: Vec<Vec<u8>> = self
            .distances
            .iter()
            .map(|subspace| {
                subspace
                    .iter()
                    .map(|&d| ((d - min_val) * scale).round() as u8)
                    .collect()
            })
            .collect();

        LookupTableInt8 {
            distances,
            num_subspaces: self.num_subspaces,
            num_codes: self.num_codes,
            scale,
            offset: min_val,
        }
    }
}

/// Quantized lookup table with int8 values.
#[derive(Debug, Clone)]
pub struct LookupTableInt8 {
    /// Quantized distance values.
    distances: Vec<Vec<u8>>,

    /// Number of subspaces.
    num_subspaces: usize,

    /// Number of codes per subspace.
    num_codes: usize,

    /// Scale factor for dequantization.
    scale: f32,

    /// Offset for dequantization.
    offset: f32,
}

impl LookupTableInt8 {
    /// Compute the approximate distance for a set of codes.
    #[inline]
    pub fn compute_distance(&self, codes: &[u8]) -> f32 {
        debug_assert_eq!(codes.len(), self.num_subspaces);

        let mut sum = 0u32;
        for (s, &code) in codes.iter().enumerate() {
            sum += self.distances[s][code as usize] as u32;
        }

        // Dequantize
        (sum as f32) / self.scale + self.offset * (self.num_subspaces as f32)
    }

    /// Compute raw quantized distance (without dequantization).
    #[inline]
    pub fn compute_distance_raw(&self, codes: &[u8]) -> u32 {
        let mut sum = 0u32;
        for (s, &code) in codes.iter().enumerate() {
            sum += self.distances[s][code as usize] as u32;
        }
        sum
    }
}

/// Batch lookup table for multiple queries.
#[derive(Debug, Clone)]
pub struct BatchLookupTable {
    /// Individual lookup tables.
    tables: Vec<LookupTable>,
}

impl BatchLookupTable {
    /// Create batch lookup tables for multiple queries.
    pub fn from_queries(codebook: &Codebook, queries: &[&[f32]]) -> Self {
        let tables = queries
            .iter()
            .map(|q| LookupTable::from_query(codebook, q))
            .collect();

        Self { tables }
    }

    /// Get the number of queries.
    pub fn num_queries(&self) -> usize {
        self.tables.len()
    }

    /// Get a specific lookup table.
    pub fn get(&self, index: usize) -> Option<&LookupTable> {
        self.tables.get(index)
    }

    /// Compute distances for all queries to a single code vector.
    pub fn compute_distances(&self, codes: &[u8], results: &mut [f32]) {
        debug_assert_eq!(self.tables.len(), results.len());

        for (i, table) in self.tables.iter().enumerate() {
            results[i] = table.compute_distance(codes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_format::DenseDataset;
    use crate::hashes::codebook::CodebookConfig;

    fn create_trained_codebook() -> Codebook {
        let mut data = Vec::new();
        for i in 0..100 {
            let mut vec = Vec::with_capacity(16);
            for j in 0..16 {
                vec.push((i * j) as f32 / 100.0);
            }
            data.push(vec);
        }
        let dataset = DenseDataset::from_vecs(data);

        let config = CodebookConfig::new(8, 4).with_seed(42);
        let mut codebook = Codebook::new(config);
        codebook.train(&dataset).unwrap();
        codebook
    }

    #[test]
    fn test_lookup_table_creation() {
        let codebook = create_trained_codebook();
        let query = vec![0.5f32; 16];
        let lut = LookupTable::from_query(&codebook, &query);

        assert_eq!(lut.num_subspaces(), 4);
        assert!(lut.num_codes() <= 8);
    }

    #[test]
    fn test_lookup_table_compute_distance() {
        let codebook = create_trained_codebook();
        let query = vec![0.5f32; 16];
        let lut = LookupTable::from_query(&codebook, &query);

        let codes = vec![0u8, 1, 2, 3];
        let dist = lut.compute_distance(&codes);

        assert!(dist >= 0.0);
    }

    #[test]
    fn test_lookup_table_int8() {
        let codebook = create_trained_codebook();
        let query = vec![0.5f32; 16];
        let lut = LookupTable::from_query(&codebook, &query);
        let lut_int8 = lut.to_int8();

        let codes = vec![0u8, 1, 2, 3];
        let dist_f32 = lut.compute_distance(&codes);
        let dist_int8 = lut_int8.compute_distance(&codes);

        // Int8 quantization introduces error; allow reasonable tolerance
        assert!((dist_f32 - dist_int8).abs() < 2.0, "dist_f32={}, dist_int8={}", dist_f32, dist_int8);
    }

    #[test]
    fn test_batch_lookup_table() {
        let codebook = create_trained_codebook();
        let queries: Vec<Vec<f32>> = (0..5)
            .map(|i| vec![0.1 * i as f32; 16])
            .collect();
        let query_refs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();

        let batch_lut = BatchLookupTable::from_queries(&codebook, &query_refs);

        assert_eq!(batch_lut.num_queries(), 5);

        let codes = vec![0u8, 1, 2, 3];
        let mut results = vec![0.0f32; 5];
        batch_lut.compute_distances(&codes, &mut results);

        for &r in &results {
            assert!(r >= 0.0);
        }
    }
}
