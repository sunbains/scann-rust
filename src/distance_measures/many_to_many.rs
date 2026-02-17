//! Many-to-many distance computations.
//!
//! Efficiently compute pairwise distances between two sets of vectors.

use crate::types::DatapointValue;
use rayon::prelude::*;

/// Helper to convert to f32 without ambiguity with NumCast::to_f32
#[inline(always)]
fn to_f32<T: DatapointValue>(v: T) -> f32 {
    DatapointValue::to_f32(v)
}

/// Compute pairwise squared L2 distances between two sets of vectors.
///
/// Returns a matrix where result[i][j] = squared_l2(a[i], b[j]).
pub fn pairwise_squared_l2<T: DatapointValue>(
    a: &[&[T]],
    b: &[&[T]],
) -> Vec<Vec<f32>> {
    // Parallel computation over rows of a
    a.par_iter()
        .map(|a_vec| {
            b.iter()
                .map(|b_vec| squared_l2_dense(a_vec, b_vec))
                .collect()
        })
        .collect()
}

/// Compute pairwise dot products between two sets of vectors.
pub fn pairwise_dot_product<T: DatapointValue>(
    a: &[&[T]],
    b: &[&[T]],
) -> Vec<Vec<f32>> {
    a.par_iter()
        .map(|a_vec| {
            b.iter()
                .map(|b_vec| dot_product_dense(a_vec, b_vec))
                .collect()
        })
        .collect()
}

/// Compute pairwise cosine distances between two sets of vectors.
pub fn pairwise_cosine<T: DatapointValue>(
    a: &[&[T]],
    b: &[&[T]],
) -> Vec<Vec<f32>> {
    // Precompute norms
    let a_norms: Vec<f32> = a.iter()
        .map(|v| v.iter().map(|&x| { let f = to_f32(x); f * f }).sum::<f32>().sqrt())
        .collect();

    let b_norms: Vec<f32> = b.iter()
        .map(|v| v.iter().map(|&x| { let f = to_f32(x); f * f }).sum::<f32>().sqrt())
        .collect();

    a.par_iter()
        .enumerate()
        .map(|(i, a_vec)| {
            b.iter()
                .enumerate()
                .map(|(j, b_vec)| {
                    let dot = dot_product_dense(a_vec, b_vec);
                    let norm_product = a_norms[i] * b_norms[j];
                    if norm_product > 1e-10 {
                        1.0 - dot / norm_product
                    } else {
                        1.0
                    }
                })
                .collect()
        })
        .collect()
}

/// Dense squared L2 distance.
#[inline]
fn squared_l2_dense<T: DatapointValue>(a: &[T], b: &[T]) -> f32 {
    #[cfg(feature = "simd")]
    {
        squared_l2_dense_simd(a, b)
    }

    #[cfg(not(feature = "simd"))]
    {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = to_f32(x) - to_f32(y);
                diff * diff
            })
            .sum()
    }
}

#[cfg(feature = "simd")]
fn squared_l2_dense_simd<T: DatapointValue>(a: &[T], b: &[T]) -> f32 {
    use wide::f32x8;

    let len = a.len().min(b.len());
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = f32x8::ZERO;

    for i in 0..chunks {
        let offset = i * 8;
        let va = f32x8::new([
            to_f32(a[offset]),
            to_f32(a[offset + 1]),
            to_f32(a[offset + 2]),
            to_f32(a[offset + 3]),
            to_f32(a[offset + 4]),
            to_f32(a[offset + 5]),
            to_f32(a[offset + 6]),
            to_f32(a[offset + 7]),
        ]);
        let vb = f32x8::new([
            to_f32(b[offset]),
            to_f32(b[offset + 1]),
            to_f32(b[offset + 2]),
            to_f32(b[offset + 3]),
            to_f32(b[offset + 4]),
            to_f32(b[offset + 5]),
            to_f32(b[offset + 6]),
            to_f32(b[offset + 7]),
        ]);
        let diff = va - vb;
        sum += diff * diff;
    }

    let mut result: f32 = sum.reduce_add();

    for i in (len - remainder)..len {
        let diff = to_f32(a[i]) - to_f32(b[i]);
        result += diff * diff;
    }

    result
}

/// Dense dot product.
#[inline]
fn dot_product_dense<T: DatapointValue>(a: &[T], b: &[T]) -> f32 {
    #[cfg(feature = "simd")]
    {
        dot_product_dense_simd(a, b)
    }

    #[cfg(not(feature = "simd"))]
    {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| to_f32(x) * to_f32(y))
            .sum()
    }
}

#[cfg(feature = "simd")]
fn dot_product_dense_simd<T: DatapointValue>(a: &[T], b: &[T]) -> f32 {
    use wide::f32x8;

    let len = a.len().min(b.len());
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = f32x8::ZERO;

    for i in 0..chunks {
        let offset = i * 8;
        let va = f32x8::new([
            to_f32(a[offset]),
            to_f32(a[offset + 1]),
            to_f32(a[offset + 2]),
            to_f32(a[offset + 3]),
            to_f32(a[offset + 4]),
            to_f32(a[offset + 5]),
            to_f32(a[offset + 6]),
            to_f32(a[offset + 7]),
        ]);
        let vb = f32x8::new([
            to_f32(b[offset]),
            to_f32(b[offset + 1]),
            to_f32(b[offset + 2]),
            to_f32(b[offset + 3]),
            to_f32(b[offset + 4]),
            to_f32(b[offset + 5]),
            to_f32(b[offset + 6]),
            to_f32(b[offset + 7]),
        ]);
        sum += va * vb;
    }

    let mut result: f32 = sum.reduce_add();

    for i in (len - remainder)..len {
        result += to_f32(a[i]) * to_f32(b[i]);
    }

    result
}

/// Batch distance computation result.
#[derive(Debug, Clone)]
pub struct BatchDistanceResult {
    /// Distance matrix (row-major).
    pub distances: Vec<f32>,
    /// Number of query vectors.
    pub num_queries: usize,
    /// Number of database vectors.
    pub num_database: usize,
}

impl BatchDistanceResult {
    /// Get distance between query i and database j.
    pub fn get(&self, query_idx: usize, db_idx: usize) -> f32 {
        self.distances[query_idx * self.num_database + db_idx]
    }

    /// Get all distances for a specific query.
    pub fn query_distances(&self, query_idx: usize) -> &[f32] {
        let start = query_idx * self.num_database;
        &self.distances[start..start + self.num_database]
    }

    /// Get top-k nearest neighbors for a query.
    pub fn top_k(&self, query_idx: usize, k: usize) -> Vec<(usize, f32)> {
        let dists = self.query_distances(query_idx);
        let mut indexed: Vec<(usize, f32)> = dists.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }
}

/// Compute batch distances between queries and database vectors.
pub fn batch_squared_l2<T: DatapointValue>(
    queries: &[&[T]],
    database: &[&[T]],
) -> BatchDistanceResult {
    let num_queries = queries.len();
    let num_database = database.len();

    let distances: Vec<f32> = queries
        .par_iter()
        .flat_map(|query| {
            database
                .iter()
                .map(|db_vec| squared_l2_dense(query, db_vec))
                .collect::<Vec<_>>()
        })
        .collect();

    BatchDistanceResult {
        distances,
        num_queries,
        num_database,
    }
}

/// Compute batch dot products.
pub fn batch_dot_product<T: DatapointValue>(
    queries: &[&[T]],
    database: &[&[T]],
) -> BatchDistanceResult {
    let num_queries = queries.len();
    let num_database = database.len();

    let distances: Vec<f32> = queries
        .par_iter()
        .flat_map(|query| {
            database
                .iter()
                .map(|db_vec| -dot_product_dense(query, db_vec)) // Negative for min-heap compatibility
                .collect::<Vec<_>>()
        })
        .collect();

    BatchDistanceResult {
        distances,
        num_queries,
        num_database,
    }
}

// ============================================================================
// SIMD-optimized batch distance computation with cache blocking
// ============================================================================

/// Compute batch squared L2 distances with SIMD acceleration and cache blocking.
///
/// This function is optimized for large matrices by processing in cache-friendly
/// blocks and using AVX2 intrinsics when available.
///
/// # Arguments
/// * `queries` - Query vectors (num_queries x dim)
/// * `database` - Database vectors (num_database x dim)
/// * `results` - Output buffer (num_queries x num_database), row-major
pub fn batch_squared_l2_simd(
    queries: &[f32],
    database: &[f32],
    query_stride: usize,
    db_stride: usize,
    num_queries: usize,
    num_database: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= num_queries * num_database);

    // Block sizes for cache efficiency
    const QUERY_BLOCK: usize = 64;
    const DB_BLOCK: usize = 256;

    // Process in cache-friendly blocks
    for q_start in (0..num_queries).step_by(QUERY_BLOCK) {
        let q_end = (q_start + QUERY_BLOCK).min(num_queries);

        for db_start in (0..num_database).step_by(DB_BLOCK) {
            let db_end = (db_start + DB_BLOCK).min(num_database);

            // Process this block
            for qi in q_start..q_end {
                let query = &queries[qi * query_stride..(qi + 1) * query_stride];

                for di in db_start..db_end {
                    let db_vec = &database[di * db_stride..(di + 1) * db_stride];

                    // Use SIMD dispatch
                    let dist = crate::simd::squared_l2_f32(query, db_vec);
                    results[qi * num_database + di] = dist;
                }
            }
        }
    }
}

/// Compute batch dot products with SIMD acceleration and cache blocking.
pub fn batch_dot_product_simd(
    queries: &[f32],
    database: &[f32],
    query_stride: usize,
    db_stride: usize,
    num_queries: usize,
    num_database: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= num_queries * num_database);

    const QUERY_BLOCK: usize = 64;
    const DB_BLOCK: usize = 256;

    for q_start in (0..num_queries).step_by(QUERY_BLOCK) {
        let q_end = (q_start + QUERY_BLOCK).min(num_queries);

        for db_start in (0..num_database).step_by(DB_BLOCK) {
            let db_end = (db_start + DB_BLOCK).min(num_database);

            for qi in q_start..q_end {
                let query = &queries[qi * query_stride..(qi + 1) * query_stride];

                for di in db_start..db_end {
                    let db_vec = &database[di * db_stride..(di + 1) * db_stride];

                    // Negative dot product for distance semantics
                    let dot = crate::simd::dot_product_f32(query, db_vec);
                    results[qi * num_database + di] = -dot;
                }
            }
        }
    }
}

/// SIMD-optimized batch distance result with strided storage.
pub struct BatchDistanceMatrix {
    /// Flattened distance matrix (row-major).
    data: Vec<f32>,
    /// Number of queries (rows).
    num_queries: usize,
    /// Number of database vectors (columns).
    num_database: usize,
}

impl BatchDistanceMatrix {
    /// Create from contiguous query and database data.
    pub fn from_squared_l2(
        queries: &[f32],
        database: &[f32],
        dim: usize,
        num_queries: usize,
        num_database: usize,
    ) -> Self {
        let mut data = vec![0.0f32; num_queries * num_database];
        batch_squared_l2_simd(
            queries,
            database,
            dim,
            dim,
            num_queries,
            num_database,
            &mut data,
        );
        Self {
            data,
            num_queries,
            num_database,
        }
    }

    /// Create from contiguous query and database data using dot product.
    pub fn from_dot_product(
        queries: &[f32],
        database: &[f32],
        dim: usize,
        num_queries: usize,
        num_database: usize,
    ) -> Self {
        let mut data = vec![0.0f32; num_queries * num_database];
        batch_dot_product_simd(
            queries,
            database,
            dim,
            dim,
            num_queries,
            num_database,
            &mut data,
        );
        Self {
            data,
            num_queries,
            num_database,
        }
    }

    /// Get distance between query i and database j.
    #[inline]
    pub fn get(&self, query_idx: usize, db_idx: usize) -> f32 {
        self.data[query_idx * self.num_database + db_idx]
    }

    /// Get all distances for a specific query.
    pub fn query_distances(&self, query_idx: usize) -> &[f32] {
        let start = query_idx * self.num_database;
        &self.data[start..start + self.num_database]
    }

    /// Get top-k nearest neighbors for a query.
    pub fn top_k(&self, query_idx: usize, k: usize) -> Vec<(usize, f32)> {
        let dists = self.query_distances(query_idx);
        let mut indexed: Vec<(usize, f32)> = dists.iter().enumerate()
            .map(|(i, &d)| (i, d))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// Get the raw data.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get dimensions.
    pub fn shape(&self) -> (usize, usize) {
        (self.num_queries, self.num_database)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pairwise_squared_l2() {
        let a: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let b: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let a_refs: Vec<&[f32]> = a.iter().map(|v| v.as_slice()).collect();
        let b_refs: Vec<&[f32]> = b.iter().map(|v| v.as_slice()).collect();

        let result = pairwise_squared_l2(&a_refs, &b_refs);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 3);

        assert!((result[0][0] - 0.0).abs() < 1e-6); // (1,0,0) to (1,0,0)
        assert!((result[0][1] - 2.0).abs() < 1e-6); // (1,0,0) to (0,1,0)
        assert!((result[0][2] - 2.0).abs() < 1e-6); // (1,0,0) to (0,0,1)
    }

    #[test]
    fn test_pairwise_dot_product() {
        let a: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0],
        ];
        let b: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];

        let a_refs: Vec<&[f32]> = a.iter().map(|v| v.as_slice()).collect();
        let b_refs: Vec<&[f32]> = b.iter().map(|v| v.as_slice()).collect();

        let result = pairwise_dot_product(&a_refs, &b_refs);

        assert!((result[0][0] - 1.0).abs() < 1e-6);
        assert!((result[0][1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_result_top_k() {
        let queries: Vec<Vec<f32>> = vec![vec![1.0, 0.0]];
        let database: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.5, 0.5],
        ];

        let q_refs: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();
        let d_refs: Vec<&[f32]> = database.iter().map(|v| v.as_slice()).collect();

        let result = batch_squared_l2(&q_refs, &d_refs);
        let top2 = result.top_k(0, 2);

        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 0); // First should be index 0 (exact match)
        assert!(top2[0].1 < 0.01);
    }

    #[test]
    fn test_batch_simd_squared_l2() {
        // 2 queries, 3 database vectors, 8 dimensions
        let queries = vec![
            1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // Query 0
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,     // Query 1
        ];
        let database = vec![
            1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // DB 0
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,     // DB 1
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,     // DB 2
        ];

        let mut results = vec![0.0f32; 6]; // 2 x 3
        batch_squared_l2_simd(&queries, &database, 8, 8, 2, 3, &mut results);

        // Query 0 to DB 0: 0.0 (exact match)
        assert!(results[0].abs() < 0.01, "Q0-D0: expected ~0, got {}", results[0]);
        // Query 0 to DB 1: 2.0
        assert!((results[1] - 2.0).abs() < 0.01, "Q0-D1: expected ~2, got {}", results[1]);
        // Query 0 to DB 2: 2.0
        assert!((results[2] - 2.0).abs() < 0.01, "Q0-D2: expected ~2, got {}", results[2]);
        // Query 1 to DB 0: 2.0
        assert!((results[3] - 2.0).abs() < 0.01, "Q1-D0: expected ~2, got {}", results[3]);
        // Query 1 to DB 1: 0.0 (exact match)
        assert!(results[4].abs() < 0.01, "Q1-D1: expected ~0, got {}", results[4]);
    }

    #[test]
    fn test_batch_distance_matrix() {
        let queries = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ];
        let database = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,  // Same as query
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,     // Different
        ];

        let matrix = BatchDistanceMatrix::from_squared_l2(&queries, &database, 8, 1, 2);

        assert_eq!(matrix.shape(), (1, 2));

        // Distance to self should be 0
        let dist0 = matrix.get(0, 0);
        assert!(dist0.abs() < 0.01, "Distance to self: {}", dist0);

        // Distance to different vector should be non-zero
        let dist1 = matrix.get(0, 1);
        assert!(dist1 > 0.0, "Distance to different: {}", dist1);

        // Top-k should return closest first
        let top = matrix.top_k(0, 2);
        assert_eq!(top[0].0, 0); // First DB vector is closest
    }
}
