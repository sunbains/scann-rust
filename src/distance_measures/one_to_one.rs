//! One-to-one distance computations.
//!
//! This module provides optimized distance computations between pairs of datapoints.

#[cfg(feature = "simd")]
use std::any::TypeId;
use crate::data_format::DatapointPtr;
use crate::types::DatapointValue;

/// Compute L1 (Manhattan) distance between two datapoints.
#[inline]
pub fn l1_distance<T: DatapointValue>(a: &DatapointPtr<'_, T>, b: &DatapointPtr<'_, T>) -> f32 {
    if a.is_dense() && b.is_dense() {
        l1_distance_dense(a.values(), b.values())
    } else {
        l1_distance_sparse(a, b)
    }
}

/// L1 distance for dense vectors.
#[inline]
fn l1_distance_dense<T: DatapointValue>(a: &[T], b: &[T]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "simd")]
    {
        l1_distance_dense_simd(a, b)
    }

    #[cfg(not(feature = "simd"))]
    {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x.to_f32() - y.to_f32()).abs())
            .sum()
    }
}

#[cfg(feature = "simd")]
fn l1_distance_dense_simd<T: DatapointValue>(a: &[T], b: &[T]) -> f32 {
    // Fast path for f32 using direct memory loads
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // Safety: We've verified T is f32
        let a_f32 = unsafe { &*(a as *const [T] as *const [f32]) };
        let b_f32 = unsafe { &*(b as *const [T] as *const [f32]) };
        return l1_distance_f32_simd(a_f32, b_f32);
    }

    // Generic path for other types
    l1_distance_generic_simd(a, b)
}

#[cfg(feature = "simd")]
#[inline]
fn l1_distance_f32_simd(a: &[f32], b: &[f32]) -> f32 {
    // Use the dispatch module which auto-selects AVX2 when available
    crate::simd::l1_distance_f32(a, b)
}

#[cfg(feature = "simd")]
fn l1_distance_generic_simd<T: DatapointValue>(a: &[T], b: &[T]) -> f32 {
    use wide::f32x8;

    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = f32x8::ZERO;

    for i in 0..chunks {
        let offset = i * 8;
        let va = f32x8::new([
            a[offset].to_f32(),
            a[offset + 1].to_f32(),
            a[offset + 2].to_f32(),
            a[offset + 3].to_f32(),
            a[offset + 4].to_f32(),
            a[offset + 5].to_f32(),
            a[offset + 6].to_f32(),
            a[offset + 7].to_f32(),
        ]);
        let vb = f32x8::new([
            b[offset].to_f32(),
            b[offset + 1].to_f32(),
            b[offset + 2].to_f32(),
            b[offset + 3].to_f32(),
            b[offset + 4].to_f32(),
            b[offset + 5].to_f32(),
            b[offset + 6].to_f32(),
            b[offset + 7].to_f32(),
        ]);
        let diff = va - vb;
        sum += diff.abs();
    }

    let mut result: f32 = sum.reduce_add();

    for i in (len - remainder)..len {
        result += (a[i].to_f32() - b[i].to_f32()).abs();
    }

    result
}

/// L1 distance for sparse vectors.
fn l1_distance_sparse<T: DatapointValue>(a: &DatapointPtr<'_, T>, b: &DatapointPtr<'_, T>) -> f32 {
    // For sparse-sparse, we need to merge the indices
    if a.is_sparse() && b.is_sparse() {
        let a_indices = a.indices().unwrap();
        let a_values = a.values();
        let b_indices = b.indices().unwrap();
        let b_values = b.values();

        let mut sum = 0.0f32;
        let mut i = 0;
        let mut j = 0;

        while i < a_indices.len() && j < b_indices.len() {
            if a_indices[i] == b_indices[j] {
                sum += (a_values[i].to_f32() - b_values[j].to_f32()).abs();
                i += 1;
                j += 1;
            } else if a_indices[i] < b_indices[j] {
                sum += a_values[i].to_f32().abs();
                i += 1;
            } else {
                sum += b_values[j].to_f32().abs();
                j += 1;
            }
        }

        while i < a_indices.len() {
            sum += a_values[i].to_f32().abs();
            i += 1;
        }

        while j < b_indices.len() {
            sum += b_values[j].to_f32().abs();
            j += 1;
        }

        sum
    } else {
        // One dense, one sparse - convert sparse to dense representation
        let dim = a.dimensionality().max(b.dimensionality()) as usize;
        let mut sum = 0.0f32;
        for d in 0..dim {
            sum += (a.get(d as u64) .to_f32() - b.get(d as u64).to_f32()).abs();
        }
        sum
    }
}

/// Compute L2 (Euclidean) distance between two datapoints.
#[inline]
pub fn l2_distance<T: DatapointValue>(a: &DatapointPtr<'_, T>, b: &DatapointPtr<'_, T>) -> f32 {
    squared_l2_distance(a, b).sqrt()
}

/// Compute squared L2 distance between two datapoints.
#[inline]
pub fn squared_l2_distance<T: DatapointValue>(
    a: &DatapointPtr<'_, T>,
    b: &DatapointPtr<'_, T>,
) -> f32 {
    if a.is_dense() && b.is_dense() {
        squared_l2_distance_dense(a.values(), b.values())
    } else {
        squared_l2_distance_sparse(a, b)
    }
}

/// Squared L2 distance for dense vectors.
#[inline]
fn squared_l2_distance_dense<T: DatapointValue>(a: &[T], b: &[T]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "simd")]
    {
        squared_l2_distance_dense_simd(a, b)
    }

    #[cfg(not(feature = "simd"))]
    {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x.to_f32() - y.to_f32();
                diff * diff
            })
            .sum()
    }
}

#[cfg(feature = "simd")]
fn squared_l2_distance_dense_simd<T: DatapointValue>(a: &[T], b: &[T]) -> f32 {
    // Fast path for f32 using direct memory loads
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // Safety: We've verified T is f32
        let a_f32 = unsafe { &*(a as *const [T] as *const [f32]) };
        let b_f32 = unsafe { &*(b as *const [T] as *const [f32]) };
        return squared_l2_distance_f32_simd(a_f32, b_f32);
    }

    // Generic path for other types
    squared_l2_distance_generic_simd(a, b)
}

#[cfg(feature = "simd")]
#[inline]
fn squared_l2_distance_f32_simd(a: &[f32], b: &[f32]) -> f32 {
    // Use the dispatch module which auto-selects AVX2 when available
    crate::simd::squared_l2_f32(a, b)
}

#[cfg(feature = "simd")]
fn squared_l2_distance_generic_simd<T: DatapointValue>(a: &[T], b: &[T]) -> f32 {
    use wide::f32x8;

    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = f32x8::ZERO;

    for i in 0..chunks {
        let offset = i * 8;
        let va = f32x8::new([
            a[offset].to_f32(),
            a[offset + 1].to_f32(),
            a[offset + 2].to_f32(),
            a[offset + 3].to_f32(),
            a[offset + 4].to_f32(),
            a[offset + 5].to_f32(),
            a[offset + 6].to_f32(),
            a[offset + 7].to_f32(),
        ]);
        let vb = f32x8::new([
            b[offset].to_f32(),
            b[offset + 1].to_f32(),
            b[offset + 2].to_f32(),
            b[offset + 3].to_f32(),
            b[offset + 4].to_f32(),
            b[offset + 5].to_f32(),
            b[offset + 6].to_f32(),
            b[offset + 7].to_f32(),
        ]);
        let diff = va - vb;
        sum += diff * diff;
    }

    let mut result: f32 = sum.reduce_add();

    for i in (len - remainder)..len {
        let diff = a[i].to_f32() - b[i].to_f32();
        result += diff * diff;
    }

    result
}

/// Squared L2 distance for sparse vectors.
fn squared_l2_distance_sparse<T: DatapointValue>(
    a: &DatapointPtr<'_, T>,
    b: &DatapointPtr<'_, T>,
) -> f32 {
    if a.is_sparse() && b.is_sparse() {
        let a_indices = a.indices().unwrap();
        let a_values = a.values();
        let b_indices = b.indices().unwrap();
        let b_values = b.values();

        let mut sum = 0.0f32;
        let mut i = 0;
        let mut j = 0;

        while i < a_indices.len() && j < b_indices.len() {
            if a_indices[i] == b_indices[j] {
                let diff = a_values[i].to_f32() - b_values[j].to_f32();
                sum += diff * diff;
                i += 1;
                j += 1;
            } else if a_indices[i] < b_indices[j] {
                let v = a_values[i].to_f32();
                sum += v * v;
                i += 1;
            } else {
                let v = b_values[j].to_f32();
                sum += v * v;
                j += 1;
            }
        }

        while i < a_indices.len() {
            let v = a_values[i].to_f32();
            sum += v * v;
            i += 1;
        }

        while j < b_indices.len() {
            let v = b_values[j].to_f32();
            sum += v * v;
            j += 1;
        }

        sum
    } else {
        let dim = a.dimensionality().max(b.dimensionality()) as usize;
        let mut sum = 0.0f32;
        for d in 0..dim {
            let diff = a.get(d as u64).to_f32() - b.get(d as u64).to_f32();
            sum += diff * diff;
        }
        sum
    }
}

/// Compute dot product between two datapoints.
#[inline]
pub fn dot_product<T: DatapointValue>(a: &DatapointPtr<'_, T>, b: &DatapointPtr<'_, T>) -> f32 {
    if a.is_dense() && b.is_dense() {
        dot_product_dense(a.values(), b.values())
    } else {
        dot_product_sparse(a, b)
    }
}

/// Dot product for dense vectors.
#[inline]
fn dot_product_dense<T: DatapointValue>(a: &[T], b: &[T]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "simd")]
    {
        dot_product_dense_simd(a, b)
    }

    #[cfg(not(feature = "simd"))]
    {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| x.to_f32() * y.to_f32())
            .sum()
    }
}

#[cfg(feature = "simd")]
fn dot_product_dense_simd<T: DatapointValue>(a: &[T], b: &[T]) -> f32 {
    // Fast path for f32 using direct memory loads
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // Safety: We've verified T is f32
        let a_f32 = unsafe { &*(a as *const [T] as *const [f32]) };
        let b_f32 = unsafe { &*(b as *const [T] as *const [f32]) };
        return dot_product_f32_simd(a_f32, b_f32);
    }

    // Generic path for other types
    dot_product_generic_simd(a, b)
}

#[cfg(feature = "simd")]
#[inline]
fn dot_product_f32_simd(a: &[f32], b: &[f32]) -> f32 {
    // Use the dispatch module which auto-selects AVX2 when available
    crate::simd::dot_product_f32(a, b)
}

#[cfg(feature = "simd")]
fn dot_product_generic_simd<T: DatapointValue>(a: &[T], b: &[T]) -> f32 {
    use wide::f32x8;

    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = f32x8::ZERO;

    for i in 0..chunks {
        let offset = i * 8;
        let va = f32x8::new([
            a[offset].to_f32(),
            a[offset + 1].to_f32(),
            a[offset + 2].to_f32(),
            a[offset + 3].to_f32(),
            a[offset + 4].to_f32(),
            a[offset + 5].to_f32(),
            a[offset + 6].to_f32(),
            a[offset + 7].to_f32(),
        ]);
        let vb = f32x8::new([
            b[offset].to_f32(),
            b[offset + 1].to_f32(),
            b[offset + 2].to_f32(),
            b[offset + 3].to_f32(),
            b[offset + 4].to_f32(),
            b[offset + 5].to_f32(),
            b[offset + 6].to_f32(),
            b[offset + 7].to_f32(),
        ]);
        sum += va * vb;
    }

    let mut result: f32 = sum.reduce_add();

    for i in (len - remainder)..len {
        result += a[i].to_f32() * b[i].to_f32();
    }

    result
}

/// Dot product for sparse vectors.
fn dot_product_sparse<T: DatapointValue>(a: &DatapointPtr<'_, T>, b: &DatapointPtr<'_, T>) -> f32 {
    if a.is_sparse() && b.is_sparse() {
        let a_indices = a.indices().unwrap();
        let a_values = a.values();
        let b_indices = b.indices().unwrap();
        let b_values = b.values();

        let mut sum = 0.0f32;
        let mut i = 0;
        let mut j = 0;

        while i < a_indices.len() && j < b_indices.len() {
            if a_indices[i] == b_indices[j] {
                sum += a_values[i].to_f32() * b_values[j].to_f32();
                i += 1;
                j += 1;
            } else if a_indices[i] < b_indices[j] {
                i += 1;
            } else {
                j += 1;
            }
        }

        sum
    } else if a.is_sparse() {
        // a sparse, b dense
        let indices = a.indices().unwrap();
        let values = a.values();
        let b_values = b.values();

        indices
            .iter()
            .zip(values.iter())
            .map(|(&idx, &val)| val.to_f32() * b_values[idx as usize].to_f32())
            .sum()
    } else {
        // a dense, b sparse
        let a_values = a.values();
        let indices = b.indices().unwrap();
        let values = b.values();

        indices
            .iter()
            .zip(values.iter())
            .map(|(&idx, &val)| a_values[idx as usize].to_f32() * val.to_f32())
            .sum()
    }
}

/// Compute negative dot product (for similarity search where lower is better).
#[inline]
pub fn negative_dot_product<T: DatapointValue>(
    a: &DatapointPtr<'_, T>,
    b: &DatapointPtr<'_, T>,
) -> f32 {
    -dot_product(a, b)
}

/// Compute cosine similarity between two datapoints.
#[inline]
pub fn cosine_similarity<T: DatapointValue>(
    a: &DatapointPtr<'_, T>,
    b: &DatapointPtr<'_, T>,
) -> f32 {
    if a.is_dense() && b.is_dense() {
        cosine_similarity_dense(a.values(), b.values())
    } else {
        // Fall back to three-pass computation for sparse vectors
        let dot = dot_product(a, b);
        let norm_a = a.squared_l2_norm().sqrt();
        let norm_b = b.squared_l2_norm().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

/// Single-pass cosine similarity for dense vectors.
#[inline]
fn cosine_similarity_dense<T: DatapointValue>(a: &[T], b: &[T]) -> f32 {
    #[cfg(feature = "simd")]
    {
        cosine_similarity_dense_simd(a, b)
    }

    #[cfg(not(feature = "simd"))]
    {
        let mut dot_ab = 0.0f32;
        let mut dot_aa = 0.0f32;
        let mut dot_bb = 0.0f32;

        for (&x, &y) in a.iter().zip(b.iter()) {
            let xf = x.to_f32();
            let yf = y.to_f32();
            dot_ab += xf * yf;
            dot_aa += xf * xf;
            dot_bb += yf * yf;
        }

        let norm_a = dot_aa.sqrt();
        let norm_b = dot_bb.sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_ab / (norm_a * norm_b)
        }
    }
}

#[cfg(feature = "simd")]
fn cosine_similarity_dense_simd<T: DatapointValue>(a: &[T], b: &[T]) -> f32 {
    // Fast path for f32 using direct memory loads
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // Safety: We've verified T is f32
        let a_f32 = unsafe { &*(a as *const [T] as *const [f32]) };
        let b_f32 = unsafe { &*(b as *const [T] as *const [f32]) };
        return cosine_similarity_f32_simd(a_f32, b_f32);
    }

    // Generic path - falls back to three-pass
    let mut dot_ab = 0.0f32;
    let mut dot_aa = 0.0f32;
    let mut dot_bb = 0.0f32;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x.to_f32();
        let yf = y.to_f32();
        dot_ab += xf * yf;
        dot_aa += xf * xf;
        dot_bb += yf * yf;
    }

    let norm_a = dot_aa.sqrt();
    let norm_b = dot_bb.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_ab / (norm_a * norm_b)
    }
}

/// Single-pass cosine similarity with SIMD for f32.
/// Computes dot(a,b), ||a||^2, ||b||^2 in one pass.
#[cfg(feature = "simd")]
#[inline]
fn cosine_similarity_f32_simd(a: &[f32], b: &[f32]) -> f32 {
    use crate::simd::portable::PortableF32x8;
    use crate::simd::traits::*;

    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut dot_ab = PortableF32x8::zero();
    let mut dot_aa = PortableF32x8::zero();
    let mut dot_bb = PortableF32x8::zero();

    for i in 0..chunks {
        let offset = i * 8;
        let va = PortableF32x8::load(&a[offset..]);
        let vb = PortableF32x8::load(&b[offset..]);
        dot_ab = dot_ab.add(va.mul(vb));
        dot_aa = dot_aa.add(va.mul(va));
        dot_bb = dot_bb.add(vb.mul(vb));
    }

    let mut sum_ab: f32 = dot_ab.horizontal_sum();
    let mut sum_aa: f32 = dot_aa.horizontal_sum();
    let mut sum_bb: f32 = dot_bb.horizontal_sum();

    // Handle remainder
    for i in (len - remainder)..len {
        sum_ab += a[i] * b[i];
        sum_aa += a[i] * a[i];
        sum_bb += b[i] * b[i];
    }

    let norm_a = sum_aa.sqrt();
    let norm_b = sum_bb.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        sum_ab / (norm_a * norm_b)
    }
}

/// Compute cosine distance between two datapoints.
#[inline]
pub fn cosine_distance<T: DatapointValue>(
    a: &DatapointPtr<'_, T>,
    b: &DatapointPtr<'_, T>,
) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Compute Hamming distance between two datapoints.
/// For integer types, counts differing positions.
/// For floating types, counts positions where values differ.
#[inline]
pub fn hamming_distance<T: DatapointValue>(
    a: &DatapointPtr<'_, T>,
    b: &DatapointPtr<'_, T>,
) -> f32 {
    debug_assert!(a.is_dense() && b.is_dense());

    let a_values = a.values();
    let b_values = b.values();

    a_values
        .iter()
        .zip(b_values.iter())
        .filter(|(&x, &y)| x.to_f32() != y.to_f32())
        .count() as f32
}

/// Compute limited inner product.
#[inline]
pub fn limited_inner_product<T: DatapointValue>(
    a: &DatapointPtr<'_, T>,
    b: &DatapointPtr<'_, T>,
) -> f32 {
    let norm_a_sq = a.squared_l2_norm();
    let norm_b_sq = b.squared_l2_norm();

    if norm_a_sq > 1.0 || norm_b_sq > 1.0 {
        f32::INFINITY
    } else {
        -dot_product(a, b)
    }
}

/// Compute general inner product (negative dot product).
#[inline]
pub fn general_inner_product<T: DatapointValue>(
    a: &DatapointPtr<'_, T>,
    b: &DatapointPtr<'_, T>,
) -> f32 {
    negative_dot_product(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_format::Datapoint;

    #[test]
    fn test_l1_distance() {
        let a = Datapoint::dense(vec![1.0f32, 2.0, 3.0]);
        let b = Datapoint::dense(vec![4.0f32, 5.0, 6.0]);
        let dist = l1_distance(&a.as_ptr(), &b.as_ptr());
        assert!((dist - 9.0).abs() < 1e-6); // |1-4| + |2-5| + |3-6| = 9
    }

    #[test]
    fn test_l2_distance() {
        let a = Datapoint::dense(vec![0.0f32, 0.0, 0.0]);
        let b = Datapoint::dense(vec![3.0f32, 4.0, 0.0]);
        let dist = l2_distance(&a.as_ptr(), &b.as_ptr());
        assert!((dist - 5.0).abs() < 1e-6); // sqrt(9 + 16) = 5
    }

    #[test]
    fn test_squared_l2_distance() {
        let a = Datapoint::dense(vec![1.0f32, 2.0, 3.0]);
        let b = Datapoint::dense(vec![4.0f32, 5.0, 6.0]);
        let dist = squared_l2_distance(&a.as_ptr(), &b.as_ptr());
        assert!((dist - 27.0).abs() < 1e-6); // 9 + 9 + 9 = 27
    }

    #[test]
    fn test_dot_product() {
        let a = Datapoint::dense(vec![1.0f32, 2.0, 3.0]);
        let b = Datapoint::dense(vec![4.0f32, 5.0, 6.0]);
        let dot = dot_product(&a.as_ptr(), &b.as_ptr());
        assert!((dot - 32.0).abs() < 1e-6); // 4 + 10 + 18 = 32
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Datapoint::dense(vec![1.0f32, 0.0]);
        let b = Datapoint::dense(vec![1.0f32, 0.0]);
        let sim = cosine_similarity(&a.as_ptr(), &b.as_ptr());
        assert!((sim - 1.0).abs() < 1e-6);

        let c = Datapoint::dense(vec![0.0f32, 1.0]);
        let sim2 = cosine_similarity(&a.as_ptr(), &c.as_ptr());
        assert!(sim2.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = Datapoint::dense(vec![1.0f32, 0.0]);
        let b = Datapoint::dense(vec![1.0f32, 0.0]);
        let dist = cosine_distance(&a.as_ptr(), &b.as_ptr());
        assert!(dist.abs() < 1e-6);

        let c = Datapoint::dense(vec![0.0f32, 1.0]);
        let dist2 = cosine_distance(&a.as_ptr(), &c.as_ptr());
        assert!((dist2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hamming_distance() {
        let a = Datapoint::dense(vec![1.0f32, 0.0, 1.0, 0.0]);
        let b = Datapoint::dense(vec![1.0f32, 1.0, 0.0, 0.0]);
        let dist = hamming_distance(&a.as_ptr(), &b.as_ptr());
        assert!((dist - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_dot_product() {
        let a = Datapoint::sparse(vec![1.0f32, 2.0], vec![0, 2], 4);
        let b = Datapoint::sparse(vec![3.0f32, 4.0], vec![0, 3], 4);
        let dot = dot_product(&a.as_ptr(), &b.as_ptr());
        assert!((dot - 3.0).abs() < 1e-6); // Only index 0 overlaps: 1*3 = 3
    }

    #[test]
    fn test_sparse_l2_distance() {
        let a = Datapoint::sparse(vec![1.0f32], vec![0], 3);
        let b = Datapoint::sparse(vec![1.0f32], vec![1], 3);
        let dist = squared_l2_distance(&a.as_ptr(), &b.as_ptr());
        assert!((dist - 2.0).abs() < 1e-6); // 1^2 + 1^2 = 2
    }
}
