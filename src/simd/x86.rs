//! x86_64-specific SIMD implementations using AVX2 intrinsics.
//!
//! These implementations provide maximum performance on modern x86_64 CPUs.
//! They require the AVX2 instruction set.
//!
//! Note: These functions are unsafe and require that AVX2 is supported.
//! Use the dispatch module for safe access with automatic feature detection.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// Check if AVX2 is supported at runtime.
#[inline]
pub fn is_avx2_supported() -> bool {
    is_x86_feature_detected!("avx2")
}

/// Check if FMA is supported at runtime.
#[inline]
pub fn is_fma_supported() -> bool {
    is_x86_feature_detected!("fma")
}

/// Compute horizontal sum of an AVX2 f32x8 register.
///
/// # Safety
/// Caller must ensure AVX2 is supported.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn horizontal_sum_f32_avx2(v: __m256) -> f32 {
    // Sum within 128-bit lanes
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);

    // Horizontal add within 128-bit register
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf = _mm_movehl_ps(sums, sums);
    let sums = _mm_add_ss(sums, shuf);

    _mm_cvtss_f32(sums)
}

/// Compute horizontal sum of an AVX2 i32x8 register.
///
/// # Safety
/// Caller must ensure AVX2 is supported.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn horizontal_sum_i32_avx2(v: __m256i) -> i32 {
    // Extract high and low 128-bit lanes
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo, hi);

    // Horizontal add within 128-bit
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 0b01);
    let sum32 = _mm_add_epi32(sum64, hi32);

    _mm_cvtsi128_si32(sum32)
}

/// Compute dot product of two f32 slices using AVX2.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are supported.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    let mut result = horizontal_sum_f32_avx2(sum);

    // Handle remainder
    for i in (len - remainder)..len {
        result += a[i] * b[i];
    }

    result
}

/// Compute L1 (Manhattan) distance of two f32 slices using AVX2.
///
/// # Safety
/// Caller must ensure AVX2 is supported.
#[target_feature(enable = "avx2")]
pub unsafe fn l1_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    // Use integer sign-bit manipulation for abs: clear the sign bit
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let diff = _mm256_sub_ps(va, vb);
        // abs: clear sign bit using andnot
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);
        sum = _mm256_add_ps(sum, abs_diff);
    }

    let mut result = horizontal_sum_f32_avx2(sum);

    // Handle remainder
    for i in (len - remainder)..len {
        result += (a[i] - b[i]).abs();
    }

    result
}

/// Compute squared L2 distance of two f32 slices using AVX2.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are supported.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn squared_l2_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    let mut result = horizontal_sum_f32_avx2(sum);

    // Handle remainder
    for i in (len - remainder)..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result
}

/// Perform LUT16 lookup using AVX2 VPSHUFB.
///
/// This performs 32 parallel 4-bit lookups using the AVX2 shuffle instruction.
///
/// # Safety
/// Caller must ensure AVX2 is supported.
#[target_feature(enable = "avx2")]
pub unsafe fn lut16_shuffle_avx2(lut: &[u8; 32], codes: &[u8; 32]) -> [u8; 32] {
    let lut_vec = _mm256_loadu_si256(lut.as_ptr() as *const __m256i);
    let codes_vec = _mm256_loadu_si256(codes.as_ptr() as *const __m256i);
    let mask = _mm256_set1_epi8(0x0F);

    // Mask to low 4 bits
    let masked = _mm256_and_si256(codes_vec, mask);

    // Perform shuffle lookup
    let result = _mm256_shuffle_epi8(lut_vec, masked);

    let mut output = [0u8; 32];
    _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
    output
}

/// Compute one-to-many dot products using AVX2 with 3-way unrolling.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are supported.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn one_to_many_dot_product_avx2(
    query: &[f32],
    database: &[f32],
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    let dim = query.len();
    let chunks = dim / 8;
    let remainder = dim % 8;

    // Preload query into registers
    let mut query_vecs: Vec<__m256> = Vec::with_capacity(chunks);
    for i in 0..chunks {
        let offset = i * 8;
        query_vecs.push(_mm256_loadu_ps(query.as_ptr().add(offset)));
    }

    // Process 3 datapoints at a time for better ILP
    let triple_batches = num_points / 3;

    for batch in 0..triple_batches {
        let base = batch * 3;

        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();

        for (i, &q_vec) in query_vecs.iter().enumerate() {
            let offset = i * 8;

            let db0 = _mm256_loadu_ps(database.as_ptr().add(base * stride + offset));
            let db1 = _mm256_loadu_ps(database.as_ptr().add((base + 1) * stride + offset));
            let db2 = _mm256_loadu_ps(database.as_ptr().add((base + 2) * stride + offset));

            sum0 = _mm256_fmadd_ps(q_vec, db0, sum0);
            sum1 = _mm256_fmadd_ps(q_vec, db1, sum1);
            sum2 = _mm256_fmadd_ps(q_vec, db2, sum2);
        }

        let mut r0 = horizontal_sum_f32_avx2(sum0);
        let mut r1 = horizontal_sum_f32_avx2(sum1);
        let mut r2 = horizontal_sum_f32_avx2(sum2);

        // Handle remainder dimensions
        for j in (dim - remainder)..dim {
            let q = query[j];
            r0 += q * database[base * stride + j];
            r1 += q * database[(base + 1) * stride + j];
            r2 += q * database[(base + 2) * stride + j];
        }

        // Store negated results (lower is better for distance)
        results[base] = -r0;
        results[base + 1] = -r1;
        results[base + 2] = -r2;
    }

    // Handle remaining datapoints
    for i in (triple_batches * 3)..num_points {
        let db_start = i * stride;
        results[i] = -dot_product_avx2(query, &database[db_start..db_start + dim]);
    }
}

/// Compute one-to-many squared L2 distances using AVX2.
///
/// Uses 4-way instruction-level parallelism for better throughput.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are supported.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn one_to_many_squared_l2_avx2(
    query: &[f32],
    database: &[f32],
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    let dim = query.len();
    let chunks = dim / 8;
    let remainder = dim % 8;

    // Preload query into registers for reuse
    let mut query_vecs: Vec<__m256> = Vec::with_capacity(chunks);
    for i in 0..chunks {
        let offset = i * 8;
        query_vecs.push(_mm256_loadu_ps(query.as_ptr().add(offset)));
    }

    // Process 4 datapoints at a time for better ILP
    let quad_batches = num_points / 4;

    for batch in 0..quad_batches {
        let base = batch * 4;

        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        for (i, &q_vec) in query_vecs.iter().enumerate() {
            let offset = i * 8;

            let db0 = _mm256_loadu_ps(database.as_ptr().add(base * stride + offset));
            let db1 = _mm256_loadu_ps(database.as_ptr().add((base + 1) * stride + offset));
            let db2 = _mm256_loadu_ps(database.as_ptr().add((base + 2) * stride + offset));
            let db3 = _mm256_loadu_ps(database.as_ptr().add((base + 3) * stride + offset));

            // Compute differences
            let diff0 = _mm256_sub_ps(q_vec, db0);
            let diff1 = _mm256_sub_ps(q_vec, db1);
            let diff2 = _mm256_sub_ps(q_vec, db2);
            let diff3 = _mm256_sub_ps(q_vec, db3);

            // Accumulate squared differences using FMA: sum += diff * diff
            sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
            sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
            sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
            sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
        }

        let mut r0 = horizontal_sum_f32_avx2(sum0);
        let mut r1 = horizontal_sum_f32_avx2(sum1);
        let mut r2 = horizontal_sum_f32_avx2(sum2);
        let mut r3 = horizontal_sum_f32_avx2(sum3);

        // Handle remainder dimensions
        for j in (dim - remainder)..dim {
            let q = query[j];
            let d0 = q - database[base * stride + j];
            let d1 = q - database[(base + 1) * stride + j];
            let d2 = q - database[(base + 2) * stride + j];
            let d3 = q - database[(base + 3) * stride + j];
            r0 += d0 * d0;
            r1 += d1 * d1;
            r2 += d2 * d2;
            r3 += d3 * d3;
        }

        results[base] = r0;
        results[base + 1] = r1;
        results[base + 2] = r2;
        results[base + 3] = r3;
    }

    // Handle remaining datapoints
    for i in (quad_batches * 4)..num_points {
        let db_start = i * stride;
        results[i] = squared_l2_avx2(query, &database[db_start..db_start + dim]);
    }
}

/// Compute Int8 to Float32 dot product using AVX2.
///
/// This is useful for asymmetric distance computation with quantized databases.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are supported.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_product_i8_f32_avx2(query_f32: &[f32], database_i8: &[i8]) -> f32 {
    debug_assert_eq!(query_f32.len(), database_i8.len());

    let len = query_f32.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 bytes of i8 data
        let i8_data = _mm_loadl_epi64(database_i8.as_ptr().add(offset) as *const __m128i);

        // Expand i8 to i32: cvtepi8_epi32 expands low 4 bytes
        let i32_lo = _mm256_cvtepi8_epi32(i8_data);

        // Convert i32 to f32
        let f32_data = _mm256_cvtepi32_ps(i32_lo);

        // Load f32 query
        let q_data = _mm256_loadu_ps(query_f32.as_ptr().add(offset));

        // FMA
        sum = _mm256_fmadd_ps(q_data, f32_data, sum);
    }

    let mut result = horizontal_sum_f32_avx2(sum);

    // Handle remainder
    for i in (len - remainder)..len {
        result += query_f32[i] * (database_i8[i] as f32);
    }

    result
}

/// LUT16 batch distance computation structure.
///
/// This processes multiple datapoints using SIMD shuffle operations.
pub struct Lut16Avx2 {
    /// Quantized lookup tables (16 bytes per subspace, duplicated for AVX2 lanes)
    packed_tables: Vec<u8>,
    /// Number of subspaces
    num_subspaces: usize,
}

impl Lut16Avx2 {
    /// Create from float lookup tables.
    ///
    /// Converts float distances to quantized u8 values for SIMD lookup.
    pub fn from_float_tables(tables: &[f32], num_subspaces: usize) -> Self {
        debug_assert_eq!(tables.len(), num_subspaces * 16);

        // Find min/max for quantization
        let min_val = tables.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = tables.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(1e-10);
        let scale = 255.0 / range;

        // Quantize and pack for AVX2 (duplicate for both 128-bit lanes)
        let mut packed = Vec::with_capacity(num_subspaces * 32);
        for s in 0..num_subspaces {
            let base = s * 16;
            // First 128-bit lane
            for i in 0..16 {
                let val = ((tables[base + i] - min_val) * scale).round() as u8;
                packed.push(val);
            }
            // Second 128-bit lane (duplicate)
            for i in 0..16 {
                let val = ((tables[base + i] - min_val) * scale).round() as u8;
                packed.push(val);
            }
        }

        Self {
            packed_tables: packed,
            num_subspaces,
        }
    }

    /// Compute distances for a batch of datapoints.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn compute_distances(
        &self,
        packed_codes: &[u8],
        num_datapoints: usize,
        results: &mut [f32],
    ) {
        let bytes_per_point = (self.num_subspaces + 1) / 2;

        // Process one datapoint at a time for now
        // A more optimized version would process 32 at a time with swizzled layout
        for dp in 0..num_datapoints {
            let mut sum = 0u32;
            let base = dp * bytes_per_point;

            for s in 0..self.num_subspaces {
                let byte_idx = s / 2;
                let packed_byte = packed_codes[base + byte_idx];
                let code = if s % 2 == 0 {
                    packed_byte & 0x0F
                } else {
                    (packed_byte >> 4) & 0x0F
                };

                let lut_offset = s * 32; // 32 bytes per subspace (duplicated)
                sum += self.packed_tables[lut_offset + code as usize] as u32;
            }

            results[dp] = sum as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_detection() {
        let has_avx2 = is_avx2_supported();
        println!("AVX2 supported: {}", has_avx2);
    }

    #[test]
    fn test_dot_product_avx2() {
        if !is_avx2_supported() || !is_fma_supported() {
            return;
        }

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let result = unsafe { dot_product_avx2(&a, &b) };
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_squared_l2_avx2() {
        if !is_avx2_supported() || !is_fma_supported() {
            return;
        }

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = unsafe { squared_l2_avx2(&a, &b) };
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();

        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_lut16_shuffle_avx2() {
        if !is_avx2_supported() {
            return;
        }

        let lut: [u8; 32] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        ];
        let codes: [u8; 32] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        ];

        let result = unsafe { lut16_shuffle_avx2(&lut, &codes) };

        // First lane: identity
        for i in 0..16 {
            assert_eq!(result[i], i as u8);
        }
        // Second lane: reversed
        for i in 0..16 {
            assert_eq!(result[16 + i], (15 - i) as u8);
        }
    }
}
