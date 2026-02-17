//! Sparse vector distance computations.
//!
//! Includes Jaccard distance, non-zero intersection, and other
//! set-based distance measures.

use crate::types::{DatapointValue, DimensionIndex};

/// Helper to convert to f32 without ambiguity with NumCast::to_f32
#[inline(always)]
fn val_to_f32<T: DatapointValue>(v: T) -> f32 {
    DatapointValue::to_f32(v)
}

/// Compute Jaccard distance between two sparse binary vectors.
///
/// Jaccard distance = 1 - |A ∩ B| / |A ∪ B|
pub fn jaccard_distance_sparse(
    a_indices: &[DimensionIndex],
    b_indices: &[DimensionIndex],
) -> f32 {
    if a_indices.is_empty() && b_indices.is_empty() {
        return 0.0;
    }

    let (intersection, union) = count_intersection_union_sorted(a_indices, b_indices);

    if union == 0 {
        return 0.0;
    }

    1.0 - (intersection as f32 / union as f32)
}

/// Compute Jaccard similarity (1 - Jaccard distance).
pub fn jaccard_similarity_sparse(
    a_indices: &[DimensionIndex],
    b_indices: &[DimensionIndex],
) -> f32 {
    1.0 - jaccard_distance_sparse(a_indices, b_indices)
}

/// Count intersection and union size for sorted index arrays.
fn count_intersection_union_sorted(a: &[DimensionIndex], b: &[DimensionIndex]) -> (usize, usize) {
    let mut intersection = 0usize;
    let mut union = 0usize;
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            intersection += 1;
            union += 1;
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            union += 1;
            i += 1;
        } else {
            union += 1;
            j += 1;
        }
    }

    union += (a.len() - i) + (b.len() - j);

    (intersection, union)
}

/// Non-zero intersection distance.
///
/// Returns the negative count of dimensions where both vectors are non-zero.
/// (Negative because lower is more similar in ScaNN convention)
pub fn non_zero_intersect_distance<T: DatapointValue>(
    a_values: &[T],
    a_indices: &[DimensionIndex],
    b_values: &[T],
    b_indices: &[DimensionIndex],
) -> f32 {
    let mut count = 0i32;
    let mut i = 0;
    let mut j = 0;

    while i < a_indices.len() && j < b_indices.len() {
        if a_indices[i] == b_indices[j] {
            // Both non-zero at this index
            if val_to_f32(a_values[i]) != 0.0 && val_to_f32(b_values[j]) != 0.0 {
                count += 1;
            }
            i += 1;
            j += 1;
        } else if a_indices[i] < b_indices[j] {
            i += 1;
        } else {
            j += 1;
        }
    }

    -count as f32
}

/// Weighted Jaccard distance.
///
/// For weighted sets where values represent counts or weights.
pub fn weighted_jaccard_distance<T: DatapointValue>(
    a_values: &[T],
    a_indices: &[DimensionIndex],
    b_values: &[T],
    b_indices: &[DimensionIndex],
) -> f32 {
    let mut min_sum = 0.0f32;
    let mut max_sum = 0.0f32;

    let mut i = 0;
    let mut j = 0;

    while i < a_indices.len() && j < b_indices.len() {
        if a_indices[i] == b_indices[j] {
            let a_val = val_to_f32(a_values[i]).abs();
            let b_val = val_to_f32(b_values[j]).abs();
            min_sum += a_val.min(b_val);
            max_sum += a_val.max(b_val);
            i += 1;
            j += 1;
        } else if a_indices[i] < b_indices[j] {
            max_sum += val_to_f32(a_values[i]).abs();
            i += 1;
        } else {
            max_sum += val_to_f32(b_values[j]).abs();
            j += 1;
        }
    }

    while i < a_indices.len() {
        max_sum += val_to_f32(a_values[i]).abs();
        i += 1;
    }

    while j < b_indices.len() {
        max_sum += val_to_f32(b_values[j]).abs();
        j += 1;
    }

    if max_sum == 0.0 {
        0.0
    } else {
        1.0 - min_sum / max_sum
    }
}

/// Dice coefficient (Sørensen–Dice coefficient).
///
/// Dice = 2 * |A ∩ B| / (|A| + |B|)
pub fn dice_coefficient_sparse(
    a_indices: &[DimensionIndex],
    b_indices: &[DimensionIndex],
) -> f32 {
    if a_indices.is_empty() && b_indices.is_empty() {
        return 1.0;
    }

    let (intersection, _) = count_intersection_union_sorted(a_indices, b_indices);
    let total = a_indices.len() + b_indices.len();

    if total == 0 {
        return 1.0;
    }

    2.0 * intersection as f32 / total as f32
}

/// Dice distance (1 - Dice coefficient).
pub fn dice_distance_sparse(
    a_indices: &[DimensionIndex],
    b_indices: &[DimensionIndex],
) -> f32 {
    1.0 - dice_coefficient_sparse(a_indices, b_indices)
}

/// Overlap coefficient (Szymkiewicz–Simpson coefficient).
///
/// Overlap = |A ∩ B| / min(|A|, |B|)
pub fn overlap_coefficient_sparse(
    a_indices: &[DimensionIndex],
    b_indices: &[DimensionIndex],
) -> f32 {
    if a_indices.is_empty() || b_indices.is_empty() {
        return 0.0;
    }

    let (intersection, _) = count_intersection_union_sorted(a_indices, b_indices);
    let min_size = a_indices.len().min(b_indices.len());

    intersection as f32 / min_size as f32
}

/// Compute sparse L1 distance (Manhattan distance for sparse vectors).
pub fn sparse_l1_distance<T: DatapointValue>(
    a_values: &[T],
    a_indices: &[DimensionIndex],
    b_values: &[T],
    b_indices: &[DimensionIndex],
) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    let mut j = 0;

    while i < a_indices.len() && j < b_indices.len() {
        if a_indices[i] == b_indices[j] {
            sum += (val_to_f32(a_values[i]) - val_to_f32(b_values[j])).abs();
            i += 1;
            j += 1;
        } else if a_indices[i] < b_indices[j] {
            sum += val_to_f32(a_values[i]).abs();
            i += 1;
        } else {
            sum += val_to_f32(b_values[j]).abs();
            j += 1;
        }
    }

    while i < a_indices.len() {
        sum += val_to_f32(a_values[i]).abs();
        i += 1;
    }

    while j < b_indices.len() {
        sum += val_to_f32(b_values[j]).abs();
        j += 1;
    }

    sum
}

/// Compute sparse squared L2 distance.
pub fn sparse_squared_l2_distance<T: DatapointValue>(
    a_values: &[T],
    a_indices: &[DimensionIndex],
    b_values: &[T],
    b_indices: &[DimensionIndex],
) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    let mut j = 0;

    while i < a_indices.len() && j < b_indices.len() {
        if a_indices[i] == b_indices[j] {
            let diff = val_to_f32(a_values[i]) - val_to_f32(b_values[j]);
            sum += diff * diff;
            i += 1;
            j += 1;
        } else if a_indices[i] < b_indices[j] {
            let v = val_to_f32(a_values[i]);
            sum += v * v;
            i += 1;
        } else {
            let v = val_to_f32(b_values[j]);
            sum += v * v;
            j += 1;
        }
    }

    while i < a_indices.len() {
        let v = val_to_f32(a_values[i]);
        sum += v * v;
        i += 1;
    }

    while j < b_indices.len() {
        let v = val_to_f32(b_values[j]);
        sum += v * v;
        j += 1;
    }

    sum
}

/// Compute sparse dot product.
pub fn sparse_dot_product<T: DatapointValue>(
    a_values: &[T],
    a_indices: &[DimensionIndex],
    b_values: &[T],
    b_indices: &[DimensionIndex],
) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    let mut j = 0;

    while i < a_indices.len() && j < b_indices.len() {
        if a_indices[i] == b_indices[j] {
            sum += val_to_f32(a_values[i]) * val_to_f32(b_values[j]);
            i += 1;
            j += 1;
        } else if a_indices[i] < b_indices[j] {
            i += 1;
        } else {
            j += 1;
        }
    }

    sum
}

/// Compute sparse cosine distance.
pub fn sparse_cosine_distance<T: DatapointValue>(
    a_values: &[T],
    a_indices: &[DimensionIndex],
    b_values: &[T],
    b_indices: &[DimensionIndex],
) -> f32 {
    let dot = sparse_dot_product(a_values, a_indices, b_values, b_indices);

    let norm_a: f32 = a_values.iter().map(|&v| {
        let x = val_to_f32(v);
        x * x
    }).sum::<f32>().sqrt();

    let norm_b: f32 = b_values.iter().map(|&v| {
        let x = val_to_f32(v);
        x * x
    }).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    1.0 - dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard_distance() {
        // Sets: A = {0, 1, 2}, B = {1, 2, 3}
        // Intersection = {1, 2}, Union = {0, 1, 2, 3}
        // Jaccard = 1 - 2/4 = 0.5
        let a: Vec<u64> = vec![0, 1, 2];
        let b: Vec<u64> = vec![1, 2, 3];

        let dist = jaccard_distance_sparse(&a, &b);
        assert!((dist - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_identical() {
        let a: Vec<u64> = vec![0, 1, 2];
        let dist = jaccard_distance_sparse(&a, &a);
        assert!((dist - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a: Vec<u64> = vec![0, 1];
        let b: Vec<u64> = vec![2, 3];

        let dist = jaccard_distance_sparse(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_non_zero_intersect() {
        let a_vals = vec![1.0f32, 2.0];
        let a_idx = vec![0u64, 2];
        let b_vals = vec![3.0f32, 4.0];
        let b_idx = vec![2u64, 3];

        let dist = non_zero_intersect_distance(&a_vals, &a_idx, &b_vals, &b_idx);
        // Intersection at index 2
        assert_eq!(dist, -1.0);
    }

    #[test]
    fn test_weighted_jaccard() {
        let a_vals = vec![2.0f32, 3.0];
        let a_idx = vec![0u64, 1];
        let b_vals = vec![1.0f32, 4.0];
        let b_idx = vec![0u64, 1];

        // min(2,1) + min(3,4) = 1 + 3 = 4
        // max(2,1) + max(3,4) = 2 + 4 = 6
        // weighted jaccard = 1 - 4/6 = 1/3
        let dist = weighted_jaccard_distance(&a_vals, &a_idx, &b_vals, &b_idx);
        assert!((dist - 1.0/3.0).abs() < 1e-6);
    }

    #[test]
    fn test_dice_coefficient() {
        let a: Vec<u64> = vec![0, 1, 2];
        let b: Vec<u64> = vec![1, 2, 3];

        // |A ∩ B| = 2, |A| + |B| = 6
        // Dice = 2 * 2 / 6 = 2/3
        let coef = dice_coefficient_sparse(&a, &b);
        assert!((coef - 2.0/3.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_l1_distance() {
        let a_vals = vec![1.0f32, 2.0];
        let a_idx = vec![0u64, 1];
        let b_vals = vec![3.0f32, 1.0];
        let b_idx = vec![0u64, 2];

        // Index 0: |1-3| = 2
        // Index 1: |2-0| = 2
        // Index 2: |0-1| = 1
        // Total: 5
        let dist = sparse_l1_distance(&a_vals, &a_idx, &b_vals, &b_idx);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_dot_product() {
        let a_vals = vec![1.0f32, 2.0, 3.0];
        let a_idx = vec![0u64, 2, 4];
        let b_vals = vec![4.0f32, 5.0];
        let b_idx = vec![2u64, 3];

        // Only overlap at index 2: 2.0 * 4.0 = 8.0
        let dot = sparse_dot_product(&a_vals, &a_idx, &b_vals, &b_idx);
        assert!((dot - 8.0).abs() < 1e-6);
    }
}
