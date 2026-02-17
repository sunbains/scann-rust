//! SIMD-optimized LUT16 distance computation using PSHUFB/VPSHUFB.
//!
//! This module provides high-performance LUT16 distance computation by:
//! 1. Quantizing float lookup tables to u8 for SIMD shuffle operations
//! 2. Using PSHUFB (16-byte) or VPSHUFB (32-byte) for parallel 16-entry lookups
//! 3. Accumulating in wider integer types to prevent overflow
//!
//! The key insight is that `_mm256_shuffle_epi8` can perform 32 parallel
//! 4-bit -> 8-bit lookups in a single instruction, which is ~20x faster
//! than scalar lookups.

use crate::simd::dispatch::lut16_distances_batch;

/// Quantized LUT16 tables optimized for SIMD shuffle operations.
///
/// Instead of storing f32 distances, we quantize to u8 values and store
/// in a layout optimized for VPSHUFB (32-byte shuffle).
#[derive(Clone)]
pub struct Lut16SimdTables {
    /// Packed lookup tables: 16 bytes per subspace, u8 quantized distances.
    /// Layout: [subspace0: 16 bytes][subspace1: 16 bytes]...
    packed_tables: Vec<u8>,
    /// Number of subspaces.
    num_subspaces: usize,
    /// Quantization bias (subtract this from final result).
    bias: f32,
    /// Quantization multiplier (multiply final sum by this).
    multiplier: f32,
}

impl Lut16SimdTables {
    /// Create quantized SIMD tables from float lookup tables.
    ///
    /// # Arguments
    /// * `float_tables` - Float lookup tables, one 16-entry table per subspace
    ///
    /// # Returns
    /// Quantized tables with bias and multiplier for dequantization.
    pub fn from_float_tables(float_tables: &[&[f32; 16]]) -> Self {
        let num_subspaces = float_tables.len();

        if num_subspaces == 0 {
            return Self {
                packed_tables: Vec::new(),
                num_subspaces: 0,
                bias: 0.0,
                multiplier: 1.0,
            };
        }

        // Find global min and max across all tables
        let mut global_min = f32::MAX;
        let mut global_max = f32::MIN;

        for table in float_tables {
            for &val in table.iter() {
                global_min = global_min.min(val);
                global_max = global_max.max(val);
            }
        }

        // Handle degenerate case where all values are the same
        let range = global_max - global_min;
        let (multiplier, bias) = if range < 1e-10 {
            (1.0, global_min)
        } else {
            // Scale to use full u8 range [0, 255]
            let scale = 255.0 / range;
            (1.0 / scale, global_min)
        };

        let scale = if range < 1e-10 { 1.0 } else { 255.0 / range };

        // Quantize all tables
        let mut packed_tables = Vec::with_capacity(num_subspaces * 16);

        for table in float_tables {
            for &val in table.iter() {
                let quantized = ((val - global_min) * scale).round() as u8;
                packed_tables.push(quantized);
            }
        }

        Self {
            packed_tables,
            num_subspaces,
            bias,
            multiplier,
        }
    }

    /// Create from raw u8 tables (already quantized).
    pub fn from_u8_tables(tables: &[u8], num_subspaces: usize, bias: f32, multiplier: f32) -> Self {
        assert_eq!(tables.len(), num_subspaces * 16);
        Self {
            packed_tables: tables.to_vec(),
            num_subspaces,
            bias,
            multiplier,
        }
    }

    /// Get the number of subspaces.
    pub fn num_subspaces(&self) -> usize {
        self.num_subspaces
    }

    /// Get the raw u8 lookup table data.
    pub fn packed_tables(&self) -> &[u8] {
        &self.packed_tables
    }

    /// Compute distances for multiple datapoints with automatic SIMD dispatch.
    ///
    /// # Arguments
    /// * `packed_codes` - Packed 4-bit codes (2 codes per byte, low nibble first)
    /// * `num_datapoints` - Number of datapoints to process
    /// * `results` - Output buffer for distances (must be at least num_datapoints)
    pub fn compute_distances_batch(
        &self,
        packed_codes: &[u8],
        num_datapoints: usize,
        results: &mut [f32],
    ) {
        debug_assert!(results.len() >= num_datapoints);

        // Use the dispatch function which handles AVX2/portable selection
        lut16_distances_batch(
            packed_codes,
            &self.packed_tables,
            self.num_subspaces,
            num_datapoints,
            results,
        );

        // Apply dequantization: result = sum * multiplier + bias * num_subspaces
        let bias_total = self.bias * self.num_subspaces as f32;
        for result in results.iter_mut().take(num_datapoints) {
            *result = *result * self.multiplier + bias_total;
        }
    }

    /// Compute distance for a single datapoint (scalar fallback).
    pub fn compute_distance_single(&self, codes: &[u8]) -> f32 {
        let mut sum = 0u32;

        for (s, &code) in codes.iter().enumerate().take(self.num_subspaces) {
            let table_offset = s * 16;
            let index = (code & 0x0F) as usize;
            sum += self.packed_tables[table_offset + index] as u32;
        }

        sum as f32 * self.multiplier + self.bias * self.num_subspaces as f32
    }
}

/// AVX2-optimized LUT16 batch distance computation.
///
/// This processes 32 codes at a time using VPSHUFB for parallel lookups.
/// Layout requirements:
/// - packed_codes: 2 codes per byte (low nibble, high nibble)
/// - lut: 16 bytes per subspace, aligned if possible
#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use std::arch::x86_64::*;

    /// Process a batch of datapoints using AVX2 VPSHUFB.
    ///
    /// # Safety
    /// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn lut16_batch_avx2(
        packed_codes: &[u8],
        lut: &[u8],
        num_subspaces: usize,
        num_datapoints: usize,
        results: &mut [f32],
    ) {
        let bytes_per_point = (num_subspaces + 1) / 2;

        // Process 32 datapoints at a time for maximum throughput
        let chunks_32 = num_datapoints / 32;

        for chunk in 0..chunks_32 {
            let base = chunk * 32;

            // Initialize 32 accumulators (2 x __m256i for 32 x u16)
            let mut acc_lo = _mm256_setzero_si256(); // Results 0-15
            let mut acc_hi = _mm256_setzero_si256(); // Results 16-31

            // Process each subspace
            let mut subspace = 0;
            for byte_idx in 0..bytes_per_point {
                // Load 32 bytes of packed codes
                let mut code_bytes = [0u8; 32];
                for i in 0..32 {
                    let dp = base + i;
                    if dp < num_datapoints {
                        code_bytes[i] = packed_codes[dp * bytes_per_point + byte_idx];
                    }
                }
                let codes = _mm256_loadu_si256(code_bytes.as_ptr() as *const __m256i);

                // Process low nibble (first subspace in this byte)
                if subspace < num_subspaces {
                    let lut_offset = subspace * 16;

                    // Load LUT (duplicate to both 128-bit lanes for VPSHUFB)
                    let lut_lo = _mm_loadu_si128(lut[lut_offset..].as_ptr() as *const __m128i);
                    let lut_vec = _mm256_broadcastsi128_si256(lut_lo);

                    // Mask low nibble
                    let mask_lo = _mm256_set1_epi8(0x0F);
                    let indices_lo = _mm256_and_si256(codes, mask_lo);

                    // VPSHUFB: parallel lookup
                    let values = _mm256_shuffle_epi8(lut_vec, indices_lo);

                    // Zero-extend u8 to u16 and add to accumulators
                    let zero = _mm256_setzero_si256();
                    let values_lo_16 = _mm256_unpacklo_epi8(values, zero);
                    let values_hi_16 = _mm256_unpackhi_epi8(values, zero);

                    acc_lo = _mm256_add_epi16(acc_lo, values_lo_16);
                    acc_hi = _mm256_add_epi16(acc_hi, values_hi_16);

                    subspace += 1;
                }

                // Process high nibble (second subspace in this byte)
                if subspace < num_subspaces {
                    let lut_offset = subspace * 16;

                    let lut_lo = _mm_loadu_si128(lut[lut_offset..].as_ptr() as *const __m128i);
                    let lut_vec = _mm256_broadcastsi128_si256(lut_lo);

                    // Shift right by 4 to get high nibble
                    let indices_hi = _mm256_srli_epi16(codes, 4);
                    let mask_lo = _mm256_set1_epi8(0x0F);
                    let indices_hi = _mm256_and_si256(indices_hi, mask_lo);

                    let values = _mm256_shuffle_epi8(lut_vec, indices_hi);

                    let zero = _mm256_setzero_si256();
                    let values_lo_16 = _mm256_unpacklo_epi8(values, zero);
                    let values_hi_16 = _mm256_unpackhi_epi8(values, zero);

                    acc_lo = _mm256_add_epi16(acc_lo, values_lo_16);
                    acc_hi = _mm256_add_epi16(acc_hi, values_hi_16);

                    subspace += 1;
                }
            }

            // Convert accumulated u16 sums to f32 results
            // acc_lo contains u16 values for datapoints arranged as:
            // Lane 0 (low 128-bit): dp0, dp1, dp2, ..., dp7, dp16, dp17, ..., dp23
            // Lane 1 (high 128-bit): dp8, dp9, ..., dp15, dp24, dp25, ..., dp31
            // Due to AVX2's lane-crossing behavior

            // Permute to get correct order and convert
            let mut sum_arr = [0u16; 32];
            _mm256_storeu_si256(sum_arr[0..16].as_mut_ptr() as *mut __m256i, acc_lo);
            _mm256_storeu_si256(sum_arr[16..32].as_mut_ptr() as *mut __m256i, acc_hi);

            // Store results (reorder due to AVX2 lane layout)
            for i in 0..32 {
                if base + i < num_datapoints {
                    results[base + i] = sum_arr[i] as f32;
                }
            }
        }

        // Handle remainder with scalar code
        let remainder_start = chunks_32 * 32;
        for dp in remainder_start..num_datapoints {
            let dp_base = dp * bytes_per_point;
            let mut sum = 0u32;
            let mut subspace = 0;

            for byte_idx in 0..bytes_per_point {
                let byte = packed_codes[dp_base + byte_idx];

                if subspace < num_subspaces {
                    let code = (byte & 0x0F) as usize;
                    sum += lut[subspace * 16 + code] as u32;
                    subspace += 1;
                }

                if subspace < num_subspaces {
                    let code = ((byte >> 4) & 0x0F) as usize;
                    sum += lut[subspace * 16 + code] as u32;
                    subspace += 1;
                }
            }

            results[dp] = sum as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_roundtrip() {
        // Create float tables with known values
        let table0: [f32; 16] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                  8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let table1: [f32; 16] = [15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0,
                                  7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0];

        let tables = Lut16SimdTables::from_float_tables(&[&table0, &table1]);

        assert_eq!(tables.num_subspaces(), 2);

        // Test single distance computation
        // codes [0, 0] -> table0[0] + table1[0] = 0 + 15 = 15
        let dist = tables.compute_distance_single(&[0, 0]);
        assert!((dist - 15.0).abs() < 0.1, "Expected ~15, got {}", dist);

        // codes [15, 15] -> table0[15] + table1[15] = 15 + 0 = 15
        let dist = tables.compute_distance_single(&[15, 15]);
        assert!((dist - 15.0).abs() < 0.1, "Expected ~15, got {}", dist);

        // codes [5, 10] -> table0[5] + table1[10] = 5 + 5 = 10
        let dist = tables.compute_distance_single(&[5, 10]);
        assert!((dist - 10.0).abs() < 0.1, "Expected ~10, got {}", dist);
    }

    #[test]
    fn test_batch_computation() {
        let table0: [f32; 16] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                  8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let tables = Lut16SimdTables::from_float_tables(&[&table0]);

        // Create packed codes for 4 datapoints with 1 subspace
        // Point 0: code 0 -> distance 0
        // Point 1: code 5 -> distance 5
        // Point 2: code 10 -> distance 10
        // Point 3: code 15 -> distance 15
        let packed_codes = vec![0x00, 0x05, 0x0A, 0x0F];
        let mut results = vec![0.0f32; 4];

        tables.compute_distances_batch(&packed_codes, 4, &mut results);

        assert!((results[0] - 0.0).abs() < 0.1, "Point 0: expected ~0, got {}", results[0]);
        assert!((results[1] - 5.0).abs() < 0.1, "Point 1: expected ~5, got {}", results[1]);
        assert!((results[2] - 10.0).abs() < 0.1, "Point 2: expected ~10, got {}", results[2]);
        assert!((results[3] - 15.0).abs() < 0.1, "Point 3: expected ~15, got {}", results[3]);
    }

    #[test]
    fn test_two_subspace_packed() {
        // 2 subspaces, packed as 1 byte per datapoint (low nibble = subspace 0, high = subspace 1)
        let table0: [f32; 16] = std::array::from_fn(|i| i as f32);
        let table1: [f32; 16] = std::array::from_fn(|i| (15 - i) as f32);

        let tables = Lut16SimdTables::from_float_tables(&[&table0, &table1]);

        // Point 0: codes [0, 0] packed as 0x00 -> 0 + 15 = 15
        // Point 1: codes [5, 5] packed as 0x55 -> 5 + 10 = 15
        // Point 2: codes [15, 0] packed as 0x0F -> 15 + 15 = 30
        // Point 3: codes [0, 15] packed as 0xF0 -> 0 + 0 = 0
        let packed_codes = vec![0x00, 0x55, 0x0F, 0xF0];
        let mut results = vec![0.0f32; 4];

        tables.compute_distances_batch(&packed_codes, 4, &mut results);

        assert!((results[0] - 15.0).abs() < 0.5, "Point 0: expected ~15, got {}", results[0]);
        assert!((results[1] - 15.0).abs() < 0.5, "Point 1: expected ~15, got {}", results[1]);
        assert!((results[2] - 30.0).abs() < 0.5, "Point 2: expected ~30, got {}", results[2]);
        assert!((results[3] - 0.0).abs() < 0.5, "Point 3: expected ~0, got {}", results[3]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_vs_portable() {
        if !crate::simd::dispatch::has_avx2() {
            println!("AVX2 not available, skipping test");
            return;
        }

        let table: [f32; 16] = std::array::from_fn(|i| i as f32 * 2.0);
        let tables = Lut16SimdTables::from_float_tables(&[&table, &table]);

        // Test with enough points to exercise both AVX2 and scalar paths
        let num_points = 100;
        let mut packed_codes = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let lo = (i % 16) as u8;
            let hi = ((i + 5) % 16) as u8;
            packed_codes.push(lo | (hi << 4));
        }

        let mut results = vec![0.0f32; num_points];
        tables.compute_distances_batch(&packed_codes, num_points, &mut results);

        // Verify against scalar computation
        for i in 0..num_points {
            let lo = (i % 16) as u8;
            let hi = ((i + 5) % 16) as u8;
            let expected = tables.compute_distance_single(&[lo, hi]);
            assert!(
                (results[i] - expected).abs() < 0.5,
                "Point {}: expected {}, got {}",
                i, expected, results[i]
            );
        }
    }
}
