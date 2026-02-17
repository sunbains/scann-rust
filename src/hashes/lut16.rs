//! LUT16 (4-bit lookup table) optimizations for asymmetric hashing.
//!
//! LUT16 uses 4-bit codes and 16-entry lookup tables for efficient
//! SIMD-accelerated distance computation.

use serde::{Deserialize, Serialize};

/// LUT16 configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lut16Config {
    /// Number of subspaces.
    pub num_subspaces: usize,
    /// Number of codes per subspace (16 for LUT16).
    pub num_codes: usize,
    /// Dimensions per subspace.
    pub dims_per_subspace: usize,
}

impl Lut16Config {
    /// Create a new LUT16 configuration.
    pub fn new(num_subspaces: usize, dims_per_subspace: usize) -> Self {
        Self {
            num_subspaces,
            num_codes: 16, // Always 16 for LUT16
            dims_per_subspace,
        }
    }
}

/// Packed 4-bit codes (2 codes per byte).
#[derive(Clone)]
pub struct PackedCodes4Bit {
    /// Packed data (2 codes per byte).
    data: Vec<u8>,
    /// Number of subspaces.
    num_subspaces: usize,
    /// Number of datapoints.
    num_datapoints: usize,
}

impl PackedCodes4Bit {
    /// Create from 8-bit codes.
    pub fn from_codes(codes: &[Vec<u8>], num_subspaces: usize) -> Self {
        let num_datapoints = codes.len();
        let bytes_per_point = (num_subspaces + 1) / 2;
        let mut data = Vec::with_capacity(num_datapoints * bytes_per_point);

        for point_codes in codes {
            for chunk in point_codes.chunks(2) {
                let lo = chunk[0] & 0x0F;
                let hi = if chunk.len() > 1 { (chunk[1] & 0x0F) << 4 } else { 0 };
                data.push(lo | hi);
            }
        }

        Self {
            data,
            num_subspaces,
            num_datapoints,
        }
    }

    /// Get codes for a datapoint.
    pub fn get_codes(&self, index: usize) -> Vec<u8> {
        let bytes_per_point = (self.num_subspaces + 1) / 2;
        let offset = index * bytes_per_point;
        let packed = &self.data[offset..offset + bytes_per_point];

        let mut codes = Vec::with_capacity(self.num_subspaces);
        for (i, &byte) in packed.iter().enumerate() {
            codes.push(byte & 0x0F);
            if i * 2 + 1 < self.num_subspaces {
                codes.push((byte >> 4) & 0x0F);
            }
        }
        codes
    }

    /// Number of datapoints.
    pub fn len(&self) -> usize {
        self.num_datapoints
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.num_datapoints == 0
    }

    /// Get raw packed data bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get number of subspaces.
    pub fn num_subspaces(&self) -> usize {
        self.num_subspaces
    }
}

/// 16-entry lookup table for a single subspace.
#[derive(Clone)]
pub struct Lut16Table {
    /// Distance values for 16 codes (SIMD-aligned).
    values: [f32; 16],
}

impl Lut16Table {
    /// Create a new LUT16 table.
    pub fn new(values: [f32; 16]) -> Self {
        Self { values }
    }

    /// Create from a distance function.
    pub fn from_distances<F>(distance_fn: F) -> Self
    where
        F: Fn(u8) -> f32,
    {
        let mut values = [0.0f32; 16];
        for (i, val) in values.iter_mut().enumerate() {
            *val = distance_fn(i as u8);
        }
        Self { values }
    }

    /// Lookup a distance value.
    #[inline]
    pub fn lookup(&self, code: u8) -> f32 {
        self.values[(code & 0x0F) as usize]
    }

    /// Get the raw values array.
    pub fn values(&self) -> &[f32; 16] {
        &self.values
    }
}

/// Complete LUT16 lookup tables for all subspaces.
#[derive(Clone)]
pub struct Lut16LookupTables {
    /// One 16-entry table per subspace.
    tables: Vec<Lut16Table>,
}

impl Lut16LookupTables {
    /// Create lookup tables from distance values.
    pub fn new(tables: Vec<Lut16Table>) -> Self {
        Self { tables }
    }

    /// Create from a query and codebook centroids.
    pub fn from_query(
        query: &[f32],
        centroids: &[Vec<Vec<f32>>], // [subspace][code][dim]
        dims_per_subspace: usize,
    ) -> Self {
        let num_subspaces = centroids.len();
        let mut tables = Vec::with_capacity(num_subspaces);

        for (s, subspace_centroids) in centroids.iter().enumerate() {
            let query_start = s * dims_per_subspace;
            let query_end = (query_start + dims_per_subspace).min(query.len());
            let query_subspace = &query[query_start..query_end];

            let table = Lut16Table::from_distances(|code| {
                let centroid = &subspace_centroids[code as usize];
                squared_l2_distance_slice(query_subspace, centroid)
            });

            tables.push(table);
        }

        Self { tables }
    }

    /// Compute approximate distance using lookup tables.
    #[inline]
    pub fn compute_distance(&self, codes: &[u8]) -> f32 {
        self.tables
            .iter()
            .zip(codes.iter())
            .map(|(table, &code)| table.lookup(code))
            .sum()
    }

    /// Compute distance from packed 4-bit codes.
    #[inline]
    pub fn compute_distance_packed(&self, packed: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        let mut table_idx = 0;

        for &byte in packed {
            if table_idx < self.tables.len() {
                sum += self.tables[table_idx].lookup(byte & 0x0F);
                table_idx += 1;
            }
            if table_idx < self.tables.len() {
                sum += self.tables[table_idx].lookup((byte >> 4) & 0x0F);
                table_idx += 1;
            }
        }

        sum
    }

    /// Number of subspaces.
    pub fn num_subspaces(&self) -> usize {
        self.tables.len()
    }

    /// Convert to SIMD-optimized tables for fast batch processing.
    ///
    /// The returned tables use quantized u8 values for PSHUFB-based
    /// parallel lookups, which is significantly faster for batch operations.
    pub fn to_simd_tables(&self) -> crate::hashes::lut16_simd::Lut16SimdTables {
        let float_tables: Vec<[f32; 16]> = self.tables
            .iter()
            .map(|t| *t.values())
            .collect();

        let table_refs: Vec<&[f32; 16]> = float_tables.iter().collect();
        crate::hashes::lut16_simd::Lut16SimdTables::from_float_tables(&table_refs)
    }

    /// Compute distances for multiple datapoints using SIMD acceleration.
    ///
    /// This method automatically uses AVX2 when available, falling back
    /// to portable SIMD otherwise.
    ///
    /// # Arguments
    /// * `packed_codes` - Packed codes from `PackedCodes4Bit`
    /// * `results` - Output buffer for distances
    pub fn compute_distances_batch_simd(
        &self,
        packed_codes: &PackedCodes4Bit,
        results: &mut [f32],
    ) {
        let simd_tables = self.to_simd_tables();
        simd_tables.compute_distances_batch(
            packed_codes.data(),
            packed_codes.len(),
            results,
        );
    }
}

#[inline]
fn squared_l2_distance_slice(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

/// SIMD-optimized LUT16 distance computation.
#[cfg(feature = "simd")]
pub mod simd {
    use super::*;
    

    /// Compute distances for 8 datapoints at once using SIMD.
    pub fn compute_distances_8(
        tables: &Lut16LookupTables,
        packed_codes: &PackedCodes4Bit,
        start_idx: usize,
    ) -> [f32; 8] {
        let mut results = [0.0f32; 8];

        for i in 0..8 {
            let idx = start_idx + i;
            if idx < packed_codes.len() {
                let codes = packed_codes.get_codes(idx);
                results[i] = tables.compute_distance(&codes);
            }
        }

        results
    }

    /// Batch distance computation with SIMD accumulation.
    pub fn compute_distances_batch(
        tables: &Lut16LookupTables,
        packed_codes: &PackedCodes4Bit,
    ) -> Vec<f32> {
        let n = packed_codes.len();
        let mut results = vec![0.0f32; n];

        // Process in chunks of 8
        let chunks = n / 8;
        for chunk_idx in 0..chunks {
            let start = chunk_idx * 8;
            let batch = compute_distances_8(tables, packed_codes, start);
            results[start..start + 8].copy_from_slice(&batch);
        }

        // Handle remainder
        for i in (chunks * 8)..n {
            let codes = packed_codes.get_codes(i);
            results[i] = tables.compute_distance(&codes);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_codes() {
        let codes = vec![
            vec![0u8, 1, 2, 3],
            vec![4, 5, 6, 7],
            vec![8, 9, 10, 11],
        ];

        let packed = PackedCodes4Bit::from_codes(&codes, 4);

        assert_eq!(packed.len(), 3);

        for (i, original) in codes.iter().enumerate() {
            let unpacked = packed.get_codes(i);
            assert_eq!(&unpacked, original);
        }
    }

    #[test]
    fn test_lut16_table() {
        let table = Lut16Table::from_distances(|code| code as f32 * 0.5);

        assert_eq!(table.lookup(0), 0.0);
        assert_eq!(table.lookup(1), 0.5);
        assert_eq!(table.lookup(10), 5.0);
    }

    #[test]
    fn test_lookup_tables() {
        // Create simple centroids: 2 subspaces with 2 dims each (4D total)
        let centroids = vec![
            // Subspace 0: 16 2D centroids, centroid i = [i, 0]
            (0..16).map(|i| vec![i as f32, 0.0]).collect(),
            // Subspace 1: 16 2D centroids, centroid i = [0, i]
            (0..16).map(|i| vec![0.0, i as f32]).collect(),
        ];

        // Query that exactly matches centroid 5 in both subspaces:
        // Subspace 0: query[0..2] = [5.0, 0.0] matches centroid 5 = [5.0, 0.0]
        // Subspace 1: query[2..4] = [0.0, 5.0] matches centroid 5 = [0.0, 5.0]
        let query = vec![5.0, 0.0, 0.0, 5.0];
        let tables = Lut16LookupTables::from_query(&query, &centroids, 2);

        assert_eq!(tables.num_subspaces(), 2);

        // Code [5, 5] should have distance 0 (query matches centroid 5 exactly)
        let dist = tables.compute_distance(&[5, 5]);
        assert!(dist < 0.01, "expected ~0, got {}", dist);

        // Code [0, 0] should have distance 50 (25 + 25)
        // Subspace 0: [5.0, 0.0] vs [0.0, 0.0] → distance = 25
        // Subspace 1: [0.0, 5.0] vs [0.0, 0.0] → distance = 25
        let dist = tables.compute_distance(&[0, 0]);
        assert!((dist - 50.0).abs() < 0.01, "expected ~50, got {}", dist);
    }
}
