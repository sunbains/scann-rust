//! Runtime CPU feature detection and dispatch.
//!
//! This module automatically selects the best SIMD implementation based on
//! the runtime CPU capabilities.

use crate::simd::portable::PortableF32x8;
use crate::simd::traits::*;

/// Enumeration of SIMD support levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdSupportLevel {
    /// No SIMD support (scalar fallback).
    Scalar,
    /// Portable SIMD via `wide` crate.
    Portable,
    /// SSE4.1 support (x86_64).
    #[cfg(target_arch = "x86_64")]
    Sse4,
    /// AVX2 support (x86_64).
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AVX-512 support (x86_64).
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

/// Detect the highest supported SIMD level at runtime.
pub fn simd_support_level() -> SimdSupportLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            SimdSupportLevel::Avx512
        } else if is_x86_feature_detected!("avx2") {
            SimdSupportLevel::Avx2
        } else if is_x86_feature_detected!("sse4.1") {
            SimdSupportLevel::Sse4
        } else {
            SimdSupportLevel::Portable
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        SimdSupportLevel::Portable
    }
}

/// Check if AVX2 is available.
#[inline]
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Check if FMA is available.
#[inline]
pub fn has_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ============================================================================
// Dispatch functions for common operations
// ============================================================================

/// Compute dot product with automatic dispatch.
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            // SAFETY: We checked for AVX2+FMA support
            return unsafe { crate::simd::x86::dot_product_avx2(a, b) };
        }
    }

    // Portable fallback
    dot_product_portable(a, b)
}

/// Compute L1 (Manhattan) distance with automatic dispatch.
pub fn l1_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: We checked for AVX2 support
            return unsafe { crate::simd::x86::l1_distance_avx2(a, b) };
        }
    }

    // Portable fallback
    l1_distance_portable(a, b)
}

/// Compute squared L2 distance with automatic dispatch.
pub fn squared_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            // SAFETY: We checked for AVX2+FMA support
            return unsafe { crate::simd::x86::squared_l2_avx2(a, b) };
        }
    }

    // Portable fallback
    squared_l2_portable(a, b)
}

/// Compute horizontal sum with automatic dispatch.
pub fn horizontal_sum_f32(values: &[f32]) -> f32 {
    horizontal_sum_portable(values)
}

// ============================================================================
// Portable implementations
// ============================================================================

fn dot_product_portable(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = PortableF32x8::zero();

    for i in 0..chunks {
        let offset = i * 8;
        let va = PortableF32x8::load(&a[offset..]);
        let vb = PortableF32x8::load(&b[offset..]);
        sum = sum.add(va.mul(vb));
    }

    let mut result = sum.horizontal_sum();

    // Handle remainder
    for i in (len - remainder)..len {
        result += a[i] * b[i];
    }

    result
}

fn squared_l2_portable(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = PortableF32x8::zero();

    for i in 0..chunks {
        let offset = i * 8;
        let va = PortableF32x8::load(&a[offset..]);
        let vb = PortableF32x8::load(&b[offset..]);
        let diff = va.sub(vb);
        sum = sum.add(diff.mul(diff));
    }

    let mut result = sum.horizontal_sum();

    // Handle remainder
    for i in (len - remainder)..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result
}

fn horizontal_sum_portable(values: &[f32]) -> f32 {
    let len = values.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = PortableF32x8::zero();

    for i in 0..chunks {
        let offset = i * 8;
        let v = PortableF32x8::load(&values[offset..]);
        sum = sum.add(v);
    }

    let mut result = sum.horizontal_sum();

    for i in (len - remainder)..len {
        result += values[i];
    }

    result
}

fn l1_distance_portable(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = PortableF32x8::zero();

    for i in 0..chunks {
        let offset = i * 8;
        let va = PortableF32x8::load(&a[offset..]);
        let vb = PortableF32x8::load(&b[offset..]);
        let diff = va.sub(vb);
        sum = sum.add(diff.abs());
    }

    let mut result = sum.horizontal_sum();

    // Handle remainder
    for i in (len - remainder)..len {
        result += (a[i] - b[i]).abs();
    }

    result
}

// ============================================================================
// LUT16 dispatch functions
// ============================================================================

/// Perform LUT16 batch distance computation with automatic dispatch.
///
/// This is the critical function for asymmetric hashing performance.
///
/// # Arguments
/// * `packed_codes` - Packed 4-bit codes (2 codes per byte)
/// * `lut` - 16-entry lookup table per subspace
/// * `num_subspaces` - Number of subspaces (number of lookups per datapoint)
/// * `num_datapoints` - Number of datapoints to process
/// * `results` - Output distances
pub fn lut16_distances_batch(
    packed_codes: &[u8],
    lut: &[u8],
    num_subspaces: usize,
    num_datapoints: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= num_datapoints);

    // Always use portable for now - optimized AVX2 version needs swizzled layout
    lut16_distances_batch_portable(packed_codes, lut, num_subspaces, num_datapoints, results);
}

fn lut16_distances_batch_portable(
    packed_codes: &[u8],
    lut: &[u8],
    num_subspaces: usize,
    num_datapoints: usize,
    results: &mut [f32],
) {
    let bytes_per_point = (num_subspaces + 1) / 2;

    for dp in 0..num_datapoints {
        let base = dp * bytes_per_point;
        let mut sum = 0u32;
        let mut subspace = 0;

        for byte_idx in 0..bytes_per_point {
            let byte = packed_codes[base + byte_idx];

            // Low nibble
            if subspace < num_subspaces {
                let code = (byte & 0x0F) as usize;
                let lut_offset = subspace * 16;
                sum += lut[lut_offset + code] as u32;
                subspace += 1;
            }

            // High nibble
            if subspace < num_subspaces {
                let code = ((byte >> 4) & 0x0F) as usize;
                let lut_offset = subspace * 16;
                sum += lut[lut_offset + code] as u32;
                subspace += 1;
            }
        }

        results[dp] = sum as f32;
    }
}

// ============================================================================
// One-to-many distance dispatch
// ============================================================================

/// Compute one-to-many dot products with automatic dispatch.
pub fn one_to_many_dot_product_f32(
    query: &[f32],
    database: &[f32],
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= num_points);
    debug_assert!(database.len() >= num_points * stride);

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            // SAFETY: We checked for AVX2+FMA support
            unsafe {
                crate::simd::x86::one_to_many_dot_product_avx2(query, database, stride, num_points, results);
            }
            return;
        }
    }

    // Portable fallback
    one_to_many_dot_product_portable(query, database, stride, num_points, results);
}

fn one_to_many_dot_product_portable(
    query: &[f32],
    database: &[f32],
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    let dim = query.len();

    for i in 0..num_points {
        let db_start = i * stride;
        let db_slice = &database[db_start..db_start + dim];
        results[i] = -dot_product_portable(query, db_slice);
    }
}

/// Compute one-to-many squared L2 distances with automatic dispatch.
pub fn one_to_many_squared_l2_f32(
    query: &[f32],
    database: &[f32],
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= num_points);
    debug_assert!(database.len() >= num_points * stride);

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            // SAFETY: We checked for AVX2+FMA support
            unsafe {
                crate::simd::x86::one_to_many_squared_l2_avx2(query, database, stride, num_points, results);
            }
            return;
        }
    }

    // Portable fallback
    one_to_many_squared_l2_portable(query, database, stride, num_points, results);
}

fn one_to_many_squared_l2_portable(
    query: &[f32],
    database: &[f32],
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    let dim = query.len();

    for i in 0..num_points {
        let db_start = i * stride;
        let db_slice = &database[db_start..db_start + dim];
        results[i] = squared_l2_portable(query, db_slice);
    }
}
