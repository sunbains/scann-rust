//! One-to-many asymmetric distance computations.
//!
//! This module provides SIMD-optimized distance computations between
//! a float query and quantized database vectors (Int8, BFloat16, FP8).
//!
//! Asymmetric here means the query has full float precision while the
//! database is stored in a more compact quantized format for memory
//! efficiency and bandwidth reduction.

use half::bf16;

// ============================================================================
// Int8 asymmetric operations
// ============================================================================

/// Compute dot products between a float query and Int8 database vectors.
///
/// # Arguments
/// * `query` - Float query vector
/// * `database` - Packed Int8 database (contiguous storage)
/// * `inv_multiplier` - Inverse of the quantization multiplier for dequantization
/// * `stride` - Number of elements per database vector
/// * `num_points` - Number of database vectors
/// * `results` - Output buffer for negative dot products (as distances)
pub fn one_to_many_int8_float_dot_product(
    query: &[f32],
    database: &[i8],
    inv_multiplier: f32,
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= num_points);
    debug_assert!(database.len() >= num_points * stride);

    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::dispatch::has_avx2() && crate::simd::dispatch::has_fma() {
            unsafe {
                one_to_many_int8_float_dot_product_avx2(
                    query, database, inv_multiplier, stride, num_points, results
                );
            }
            return;
        }
    }

    one_to_many_int8_float_dot_product_portable(
        query, database, inv_multiplier, stride, num_points, results
    );
}

fn one_to_many_int8_float_dot_product_portable(
    query: &[f32],
    database: &[i8],
    inv_multiplier: f32,
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    let dim = query.len();

    for i in 0..num_points {
        let base = i * stride;
        let mut sum = 0.0f32;

        for j in 0..dim {
            let db_val = database[base + j] as f32 * inv_multiplier;
            sum += query[j] * db_val;
        }

        // Negate for distance (higher dot product = closer = lower distance)
        results[i] = -sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn one_to_many_int8_float_dot_product_avx2(
    query: &[f32],
    database: &[i8],
    inv_multiplier: f32,
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    use std::arch::x86_64::*;

    let dim = query.len();
    let chunks = dim / 8;
    let remainder = dim % 8;

    let inv_mul_vec = _mm256_set1_ps(inv_multiplier);

    for i in 0..num_points {
        let base = i * stride;
        let mut acc = _mm256_setzero_ps();

        // Process 8 elements at a time
        for c in 0..chunks {
            let q_offset = c * 8;
            let db_offset = base + q_offset;

            // Load query (8 floats)
            let q_vec = _mm256_loadu_ps(query[q_offset..].as_ptr());

            // Load 8 int8 values and convert to float
            // Load as i64 to get 8 bytes, then unpack
            let db_bytes = _mm_loadl_epi64(database[db_offset..].as_ptr() as *const __m128i);

            // Sign-extend i8 to i16
            let db_i16 = _mm_cvtepi8_epi16(db_bytes);

            // Sign-extend i16 to i32 (low 4)
            let db_i32_lo = _mm_cvtepi16_epi32(db_i16);
            // Sign-extend i16 to i32 (high 4)
            let db_i16_hi = _mm_shuffle_epi32(db_i16, 0b11_10_11_10);
            let db_i32_hi = _mm_cvtepi16_epi32(db_i16_hi);

            // Combine into 256-bit
            let db_i32 = _mm256_setr_m128i(db_i32_lo, db_i32_hi);

            // Convert to float
            let db_f32 = _mm256_cvtepi32_ps(db_i32);

            // Scale by inverse multiplier
            let db_scaled = _mm256_mul_ps(db_f32, inv_mul_vec);

            // FMA: acc += query * db
            acc = _mm256_fmadd_ps(q_vec, db_scaled, acc);
        }

        // Horizontal sum
        let mut result = horizontal_sum_avx(acc);

        // Handle remainder
        for j in (dim - remainder)..dim {
            let db_val = database[base + j] as f32 * inv_multiplier;
            result += query[j] * db_val;
        }

        results[i] = -result;
    }
}

/// Compute squared L2 distances between a float query and Int8 database vectors.
///
/// # Arguments
/// * `query` - Float query vector
/// * `database` - Packed Int8 database (contiguous storage)
/// * `inv_multiplier` - Inverse of the quantization multiplier for dequantization
/// * `stride` - Number of elements per database vector
/// * `num_points` - Number of database vectors
/// * `results` - Output buffer for squared L2 distances
pub fn one_to_many_int8_float_squared_l2(
    query: &[f32],
    database: &[i8],
    inv_multiplier: f32,
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= num_points);
    debug_assert!(database.len() >= num_points * stride);

    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::dispatch::has_avx2() && crate::simd::dispatch::has_fma() {
            unsafe {
                one_to_many_int8_float_squared_l2_avx2(
                    query, database, inv_multiplier, stride, num_points, results
                );
            }
            return;
        }
    }

    one_to_many_int8_float_squared_l2_portable(
        query, database, inv_multiplier, stride, num_points, results
    );
}

fn one_to_many_int8_float_squared_l2_portable(
    query: &[f32],
    database: &[i8],
    inv_multiplier: f32,
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    let dim = query.len();

    for i in 0..num_points {
        let base = i * stride;
        let mut sum = 0.0f32;

        for j in 0..dim {
            let db_val = database[base + j] as f32 * inv_multiplier;
            let diff = query[j] - db_val;
            sum += diff * diff;
        }

        results[i] = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn one_to_many_int8_float_squared_l2_avx2(
    query: &[f32],
    database: &[i8],
    inv_multiplier: f32,
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    use std::arch::x86_64::*;

    let dim = query.len();
    let chunks = dim / 8;
    let remainder = dim % 8;

    let inv_mul_vec = _mm256_set1_ps(inv_multiplier);

    for i in 0..num_points {
        let base = i * stride;
        let mut acc = _mm256_setzero_ps();

        for c in 0..chunks {
            let q_offset = c * 8;
            let db_offset = base + q_offset;

            let q_vec = _mm256_loadu_ps(query[q_offset..].as_ptr());

            let db_bytes = _mm_loadl_epi64(database[db_offset..].as_ptr() as *const __m128i);
            let db_i16 = _mm_cvtepi8_epi16(db_bytes);
            let db_i32_lo = _mm_cvtepi16_epi32(db_i16);
            let db_i16_hi = _mm_shuffle_epi32(db_i16, 0b11_10_11_10);
            let db_i32_hi = _mm_cvtepi16_epi32(db_i16_hi);
            let db_i32 = _mm256_setr_m128i(db_i32_lo, db_i32_hi);
            let db_f32 = _mm256_cvtepi32_ps(db_i32);
            let db_scaled = _mm256_mul_ps(db_f32, inv_mul_vec);

            // diff = query - db
            let diff = _mm256_sub_ps(q_vec, db_scaled);

            // acc += diff * diff
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }

        let mut result = horizontal_sum_avx(acc);

        for j in (dim - remainder)..dim {
            let db_val = database[base + j] as f32 * inv_multiplier;
            let diff = query[j] - db_val;
            result += diff * diff;
        }

        results[i] = result;
    }
}

// ============================================================================
// BFloat16 asymmetric operations
// ============================================================================

/// Compute dot products between a float query and BFloat16 database vectors.
pub fn one_to_many_bf16_float_dot_product(
    query: &[f32],
    database: &[bf16],
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= num_points);
    debug_assert!(database.len() >= num_points * stride);

    let dim = query.len();

    for i in 0..num_points {
        let base = i * stride;
        let mut sum = 0.0f32;

        for j in 0..dim {
            sum += query[j] * database[base + j].to_f32();
        }

        results[i] = -sum;
    }
}

/// Compute squared L2 distances between a float query and BFloat16 database vectors.
pub fn one_to_many_bf16_float_squared_l2(
    query: &[f32],
    database: &[bf16],
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= num_points);
    debug_assert!(database.len() >= num_points * stride);

    let dim = query.len();

    for i in 0..num_points {
        let base = i * stride;
        let mut sum = 0.0f32;

        for j in 0..dim {
            let diff = query[j] - database[base + j].to_f32();
            sum += diff * diff;
        }

        results[i] = sum;
    }
}

// ============================================================================
// FP8 asymmetric operations
// ============================================================================

use crate::quantization::Fp8Value;

/// Compute dot products between a float query and FP8 database vectors.
///
/// Uses E4M3 format by default.
pub fn one_to_many_fp8_float_dot_product(
    query: &[f32],
    database: &[Fp8Value],
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= num_points);
    debug_assert!(database.len() >= num_points * stride);

    let dim = query.len();

    for i in 0..num_points {
        let base = i * stride;
        let mut sum = 0.0f32;

        for j in 0..dim {
            sum += query[j] * database[base + j].to_f32_e4m3();
        }

        results[i] = -sum;
    }
}

/// Compute squared L2 distances between a float query and FP8 database vectors.
///
/// Uses E4M3 format by default.
pub fn one_to_many_fp8_float_squared_l2(
    query: &[f32],
    database: &[Fp8Value],
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= num_points);
    debug_assert!(database.len() >= num_points * stride);

    let dim = query.len();

    for i in 0..num_points {
        let base = i * stride;
        let mut sum = 0.0f32;

        for j in 0..dim {
            let diff = query[j] - database[base + j].to_f32_e4m3();
            sum += diff * diff;
        }

        results[i] = sum;
    }
}

// ============================================================================
// Helper functions
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    // Add high 128 bits to low 128 bits
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(low, high);

    // Horizontal add within 128 bits
    let sum64 = _mm_hadd_ps(sum128, sum128);
    let sum32 = _mm_hadd_ps(sum64, sum64);

    _mm_cvtss_f32(sum32)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-3;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_int8_dot_product() {
        let query = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let inv_multiplier = 1.0 / 127.0; // Standard int8 scale

        // Database: 2 vectors
        let database: Vec<i8> = vec![
            127, 127, 127, 127, 127, 127, 127, 127, // ~[1, 1, 1, 1, 1, 1, 1, 1] when scaled
            0, 0, 0, 0, 0, 0, 0, 0,                 // [0, 0, 0, 0, 0, 0, 0, 0]
        ];
        let mut results = vec![0.0f32; 2];

        one_to_many_int8_float_dot_product(&query, &database, inv_multiplier, 8, 2, &mut results);

        // Point 0: dot(query, ones) ≈ -36 (negated)
        assert!(approx_eq(results[0], -36.0), "Point 0: expected ~-36, got {}", results[0]);
        // Point 1: dot(query, zeros) = 0
        assert!(approx_eq(results[1], 0.0), "Point 1: expected ~0, got {}", results[1]);
    }

    #[test]
    fn test_int8_squared_l2() {
        let query = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let inv_multiplier = 1.0 / 127.0;

        // Database: point 0 is ~same as query scaled, point 1 is zeros
        let database: Vec<i8> = vec![
            127, 127, 127, 127, 127, 127, 127, 127, // ~ones
            0, 0, 0, 0, 0, 0, 0, 0,                 // zeros
        ];
        let mut results = vec![0.0f32; 2];

        one_to_many_int8_float_squared_l2(&query, &database, inv_multiplier, 8, 2, &mut results);

        // Point 1: distance to zeros = sum of squares = 1+4+9+16+25+36+49+64 = 204
        assert!(
            (results[1] - 204.0).abs() < 0.5,
            "Point 1: expected ~204, got {}",
            results[1]
        );
    }

    #[test]
    fn test_bf16_dot_product() {
        let query = vec![1.0f32, 2.0, 3.0, 4.0];
        let database: Vec<bf16> = vec![
            bf16::from_f32(1.0), bf16::from_f32(1.0), bf16::from_f32(1.0), bf16::from_f32(1.0),
            bf16::from_f32(2.0), bf16::from_f32(2.0), bf16::from_f32(2.0), bf16::from_f32(2.0),
        ];
        let mut results = vec![0.0f32; 2];

        one_to_many_bf16_float_dot_product(&query, &database, 4, 2, &mut results);

        // Point 0: dot = 1+2+3+4 = 10, negated = -10
        assert!(approx_eq(results[0], -10.0), "Point 0: expected -10, got {}", results[0]);
        // Point 1: dot = 2+4+6+8 = 20, negated = -20
        assert!(approx_eq(results[1], -20.0), "Point 1: expected -20, got {}", results[1]);
    }

    #[test]
    fn test_bf16_squared_l2() {
        let query = vec![1.0f32, 2.0, 3.0, 4.0];
        let database: Vec<bf16> = vec![
            bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0), bf16::from_f32(4.0),
            bf16::from_f32(2.0), bf16::from_f32(3.0), bf16::from_f32(4.0), bf16::from_f32(5.0),
        ];
        let mut results = vec![0.0f32; 2];

        one_to_many_bf16_float_squared_l2(&query, &database, 4, 2, &mut results);

        // Point 0: same as query, distance = 0
        assert!(approx_eq(results[0], 0.0), "Point 0: expected 0, got {}", results[0]);
        // Point 1: each dim differs by 1, distance = 4
        assert!(approx_eq(results[1], 4.0), "Point 1: expected 4, got {}", results[1]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_vs_portable_int8() {
        if !crate::simd::dispatch::has_avx2() || !crate::simd::dispatch::has_fma() {
            println!("AVX2/FMA not available, skipping");
            return;
        }

        let dim = 128;
        let num_points = 50;
        let inv_multiplier = 1.0 / 64.0;

        // Generate test data
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
        let database: Vec<i8> = (0..num_points * dim)
            .map(|i| (i % 128) as i8 - 64)
            .collect();

        let mut results_avx = vec![0.0f32; num_points];
        let mut results_portable = vec![0.0f32; num_points];

        // Test dot product
        unsafe {
            one_to_many_int8_float_dot_product_avx2(
                &query, &database, inv_multiplier, dim, num_points, &mut results_avx
            );
        }
        one_to_many_int8_float_dot_product_portable(
            &query, &database, inv_multiplier, dim, num_points, &mut results_portable
        );

        for i in 0..num_points {
            assert!(
                (results_avx[i] - results_portable[i]).abs() < 0.1,
                "Dot product mismatch at {}: avx={}, portable={}",
                i, results_avx[i], results_portable[i]
            );
        }

        // Test squared L2
        unsafe {
            one_to_many_int8_float_squared_l2_avx2(
                &query, &database, inv_multiplier, dim, num_points, &mut results_avx
            );
        }
        one_to_many_int8_float_squared_l2_portable(
            &query, &database, inv_multiplier, dim, num_points, &mut results_portable
        );

        for i in 0..num_points {
            assert!(
                (results_avx[i] - results_portable[i]).abs() < 0.5,
                "Squared L2 mismatch at {}: avx={}, portable={}",
                i, results_avx[i], results_portable[i]
            );
        }
    }
}
