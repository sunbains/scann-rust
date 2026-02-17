//! One-to-many distance computations.
//!
//! This module provides batch distance computations from a single query
//! to multiple database points, optimized for throughput.

#[cfg(feature = "simd")]
use std::any::TypeId;
use crate::data_format::DatapointPtr;
use crate::types::DatapointValue;
use rayon::prelude::*;

/// Compute squared L2 distances from one query to many database points.
///
/// This is optimized for batch operations and uses SIMD when available.
pub fn one_to_many_squared_l2<T: DatapointValue>(
    query: &DatapointPtr<'_, T>,
    database: &[&[T]],
    results: &mut [f32],
) {
    debug_assert_eq!(database.len(), results.len());

    if database.is_empty() {
        return;
    }

    let query_values = query.values();

    #[cfg(feature = "simd")]
    {
        one_to_many_squared_l2_simd(query_values, database, results);
    }

    #[cfg(not(feature = "simd"))]
    {
        for (i, db_point) in database.iter().enumerate() {
            results[i] = query_values
                .iter()
                .zip(db_point.iter())
                .map(|(&q, &d)| {
                    let diff = q.to_f32() - d.to_f32();
                    diff * diff
                })
                .sum();
        }
    }
}

#[cfg(feature = "simd")]
fn one_to_many_squared_l2_simd<T: DatapointValue>(
    query: &[T],
    database: &[&[T]],
    results: &mut [f32],
) {
    use wide::f32x8;

    let dim = query.len();
    let chunks = dim / 8;
    let remainder = dim % 8;

    // Preload query into SIMD registers
    let mut query_chunks: Vec<f32x8> = Vec::with_capacity(chunks);
    for i in 0..chunks {
        let offset = i * 8;
        query_chunks.push(f32x8::new([
            query[offset].to_f32(),
            query[offset + 1].to_f32(),
            query[offset + 2].to_f32(),
            query[offset + 3].to_f32(),
            query[offset + 4].to_f32(),
            query[offset + 5].to_f32(),
            query[offset + 6].to_f32(),
            query[offset + 7].to_f32(),
        ]));
    }

    for (idx, db_point) in database.iter().enumerate() {
        let mut sum = f32x8::ZERO;

        for (i, q_chunk) in query_chunks.iter().enumerate() {
            let offset = i * 8;
            let d_chunk = f32x8::new([
                db_point[offset].to_f32(),
                db_point[offset + 1].to_f32(),
                db_point[offset + 2].to_f32(),
                db_point[offset + 3].to_f32(),
                db_point[offset + 4].to_f32(),
                db_point[offset + 5].to_f32(),
                db_point[offset + 6].to_f32(),
                db_point[offset + 7].to_f32(),
            ]);
            let diff = *q_chunk - d_chunk;
            sum += diff * diff;
        }

        let mut result: f32 = sum.reduce_add();

        // Handle remainder
        for i in (dim - remainder)..dim {
            let diff = query[i].to_f32() - db_point[i].to_f32();
            result += diff * diff;
        }

        results[idx] = result;
    }
}

/// Compute dot products from one query to many database points.
pub fn one_to_many_dot_product<T: DatapointValue>(
    query: &DatapointPtr<'_, T>,
    database: &[&[T]],
    results: &mut [f32],
) {
    debug_assert_eq!(database.len(), results.len());

    if database.is_empty() {
        return;
    }

    let query_values = query.values();

    #[cfg(feature = "simd")]
    {
        one_to_many_dot_product_simd(query_values, database, results);
    }

    #[cfg(not(feature = "simd"))]
    {
        for (i, db_point) in database.iter().enumerate() {
            results[i] = -query_values
                .iter()
                .zip(db_point.iter())
                .map(|(&q, &d)| q.to_f32() * d.to_f32())
                .sum::<f32>();
        }
    }
}

#[cfg(feature = "simd")]
fn one_to_many_dot_product_simd<T: DatapointValue>(
    query: &[T],
    database: &[&[T]],
    results: &mut [f32],
) {
    use wide::f32x8;

    let dim = query.len();
    let chunks = dim / 8;
    let remainder = dim % 8;

    // Preload query into SIMD registers
    let mut query_chunks: Vec<f32x8> = Vec::with_capacity(chunks);
    for i in 0..chunks {
        let offset = i * 8;
        query_chunks.push(f32x8::new([
            query[offset].to_f32(),
            query[offset + 1].to_f32(),
            query[offset + 2].to_f32(),
            query[offset + 3].to_f32(),
            query[offset + 4].to_f32(),
            query[offset + 5].to_f32(),
            query[offset + 6].to_f32(),
            query[offset + 7].to_f32(),
        ]));
    }

    for (idx, db_point) in database.iter().enumerate() {
        let mut sum = f32x8::ZERO;

        for (i, q_chunk) in query_chunks.iter().enumerate() {
            let offset = i * 8;
            let d_chunk = f32x8::new([
                db_point[offset].to_f32(),
                db_point[offset + 1].to_f32(),
                db_point[offset + 2].to_f32(),
                db_point[offset + 3].to_f32(),
                db_point[offset + 4].to_f32(),
                db_point[offset + 5].to_f32(),
                db_point[offset + 6].to_f32(),
                db_point[offset + 7].to_f32(),
            ]);
            sum += *q_chunk * d_chunk;
        }

        let mut result: f32 = sum.reduce_add();

        // Handle remainder
        for i in (dim - remainder)..dim {
            result += query[i].to_f32() * db_point[i].to_f32();
        }

        // Negate for distance (lower is better)
        results[idx] = -result;
    }
}

/// Parallel one-to-many squared L2 distance computation.
pub fn one_to_many_squared_l2_parallel<T: DatapointValue + Sync>(
    query: &DatapointPtr<'_, T>,
    database: &[&[T]],
    results: &mut [f32],
) {
    if database.len() < 1000 {
        // Use sequential for small datasets
        one_to_many_squared_l2(query, database, results);
        return;
    }

    let query_values = query.values();

    results
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, result)| {
            let db_point = database[i];
            *result = query_values
                .iter()
                .zip(db_point.iter())
                .map(|(&q, &d)| {
                    let diff = q.to_f32() - d.to_f32();
                    diff * diff
                })
                .sum();
        });
}

/// Compute squared L2 distances using the dataset's raw data layout.
/// This is optimized for DenseDataset with contiguous storage.
pub fn one_to_many_squared_l2_strided<T: DatapointValue>(
    query: &[T],
    data: &[T],
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= num_points);

    #[cfg(feature = "simd")]
    {
        one_to_many_squared_l2_strided_simd(query, data, stride, num_points, results);
    }

    #[cfg(not(feature = "simd"))]
    {
        let dim = query.len();
        for i in 0..num_points {
            let offset = i * stride;
            let mut sum = 0.0f32;
            for j in 0..dim {
                let diff = query[j].to_f32() - data[offset + j].to_f32();
                sum += diff * diff;
            }
            results[i] = sum;
        }
    }
}

#[cfg(feature = "simd")]
fn one_to_many_squared_l2_strided_simd<T: DatapointValue>(
    query: &[T],
    data: &[T],
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    // Fast path for f32 using dispatch module (auto-selects AVX2)
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // Safety: We've verified T is f32
        let query_f32 = unsafe { &*(query as *const [T] as *const [f32]) };
        let data_f32 = unsafe { &*(data as *const [T] as *const [f32]) };
        crate::simd::one_to_many_squared_l2_f32(query_f32, data_f32, stride, num_points, results);
        return;
    }

    // Generic fallback
    one_to_many_squared_l2_strided_generic(query, data, stride, num_points, results);
}

#[cfg(feature = "simd")]
fn one_to_many_squared_l2_strided_generic<T: DatapointValue>(
    query: &[T],
    data: &[T],
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    use wide::f32x8;

    let dim = query.len();
    let chunks = dim / 8;
    let remainder = dim % 8;

    // Preload query into SIMD registers
    let mut query_chunks: Vec<f32x8> = Vec::with_capacity(chunks);
    for i in 0..chunks {
        let offset = i * 8;
        query_chunks.push(f32x8::new([
            query[offset].to_f32(),
            query[offset + 1].to_f32(),
            query[offset + 2].to_f32(),
            query[offset + 3].to_f32(),
            query[offset + 4].to_f32(),
            query[offset + 5].to_f32(),
            query[offset + 6].to_f32(),
            query[offset + 7].to_f32(),
        ]));
    }

    for i in 0..num_points {
        let base = i * stride;
        let mut sum = f32x8::ZERO;

        for (j, q_chunk) in query_chunks.iter().enumerate() {
            let offset = base + j * 8;
            let d_chunk = f32x8::new([
                data[offset].to_f32(),
                data[offset + 1].to_f32(),
                data[offset + 2].to_f32(),
                data[offset + 3].to_f32(),
                data[offset + 4].to_f32(),
                data[offset + 5].to_f32(),
                data[offset + 6].to_f32(),
                data[offset + 7].to_f32(),
            ]);
            let diff = *q_chunk - d_chunk;
            sum += diff * diff;
        }

        let mut result: f32 = sum.reduce_add();

        // Handle remainder
        for j in (dim - remainder)..dim {
            let diff = query[j].to_f32() - data[base + j].to_f32();
            result += diff * diff;
        }

        results[i] = result;
    }
}

/// Compute dot products using strided data layout.
pub fn one_to_many_dot_product_strided<T: DatapointValue>(
    query: &[T],
    data: &[T],
    stride: usize,
    num_points: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= num_points);

    #[cfg(feature = "simd")]
    {
        // Fast path for f32 using dispatch module (auto-selects AVX2)
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // Safety: We've verified T is f32
            let query_f32 = unsafe { &*(query as *const [T] as *const [f32]) };
            let data_f32 = unsafe { &*(data as *const [T] as *const [f32]) };
            crate::simd::one_to_many_dot_product_f32(query_f32, data_f32, stride, num_points, results);
            return;
        }
    }

    // Generic fallback
    let dim = query.len();
    for i in 0..num_points {
        let offset = i * stride;
        let mut sum = 0.0f32;
        for j in 0..dim {
            sum += query[j].to_f32() * data[offset + j].to_f32();
        }
        // Negate for distance
        results[i] = -sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_format::Datapoint;

    #[test]
    fn test_one_to_many_squared_l2() {
        let query = Datapoint::dense(vec![1.0f32, 2.0, 3.0]);
        let db: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0], // distance = 0
            vec![2.0, 3.0, 4.0], // distance = 3
            vec![0.0, 0.0, 0.0], // distance = 14
        ];
        let db_refs: Vec<&[f32]> = db.iter().map(|v| v.as_slice()).collect();
        let mut results = vec![0.0f32; 3];

        one_to_many_squared_l2(&query.as_ptr(), &db_refs, &mut results);

        assert!((results[0] - 0.0).abs() < 1e-6);
        assert!((results[1] - 3.0).abs() < 1e-6);
        assert!((results[2] - 14.0).abs() < 1e-6);
    }

    #[test]
    fn test_one_to_many_dot_product() {
        let query = Datapoint::dense(vec![1.0f32, 2.0, 3.0]);
        let db: Vec<Vec<f32>> = vec![
            vec![1.0, 1.0, 1.0], // dot = 6
            vec![2.0, 2.0, 2.0], // dot = 12
        ];
        let db_refs: Vec<&[f32]> = db.iter().map(|v| v.as_slice()).collect();
        let mut results = vec![0.0f32; 2];

        one_to_many_dot_product(&query.as_ptr(), &db_refs, &mut results);

        // Results are negative dot products
        assert!((results[0] - (-6.0)).abs() < 1e-6);
        assert!((results[1] - (-12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_one_to_many_strided() {
        // Simulate strided data layout
        let query = vec![1.0f32, 2.0];
        let stride = 4; // Padding to 4 elements
        let data = vec![
            1.0, 2.0, 0.0, 0.0, // Point 0: [1, 2]
            3.0, 4.0, 0.0, 0.0, // Point 1: [3, 4]
        ];
        let mut results = vec![0.0f32; 2];

        one_to_many_squared_l2_strided(&query, &data, stride, 2, &mut results);

        assert!((results[0] - 0.0).abs() < 1e-6);
        assert!((results[1] - 8.0).abs() < 1e-6); // (1-3)^2 + (2-4)^2 = 8
    }
}
