//! Linear algebra utilities for ScaNN.
//!
//! This module provides matrix operations, eigendecomposition, and other
//! linear algebra primitives needed for PCA and projections.

use nalgebra::{DMatrix, DVector, SVD};
use crate::error::{Result, ScannError};

/// Compute the mean of each column in a matrix.
pub fn column_mean(data: &DMatrix<f32>) -> DVector<f32> {
    let n = data.nrows() as f32;
    let mut mean = DVector::zeros(data.ncols());
    for row in data.row_iter() {
        for (i, val) in row.iter().enumerate() {
            mean[i] += val / n;
        }
    }
    mean
}

/// Center data by subtracting the column mean.
pub fn center_data(data: &DMatrix<f32>) -> (DMatrix<f32>, DVector<f32>) {
    let mean = column_mean(data);
    let mut centered = data.clone();
    for mut row in centered.row_iter_mut() {
        for (i, val) in row.iter_mut().enumerate() {
            *val -= mean[i];
        }
    }
    (centered, mean)
}

/// Compute the covariance matrix of centered data.
pub fn covariance_matrix(centered_data: &DMatrix<f32>) -> DMatrix<f32> {
    let n = centered_data.nrows() as f32;
    (centered_data.transpose() * centered_data) / (n - 1.0)
}

/// Principal Component Analysis result.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Principal components (columns are eigenvectors).
    pub components: DMatrix<f32>,
    /// Explained variance for each component.
    pub explained_variance: DVector<f32>,
    /// Mean of the original data.
    pub mean: DVector<f32>,
    /// Number of components.
    pub n_components: usize,
}

impl PcaResult {
    /// Transform data using PCA projection.
    pub fn transform(&self, data: &DMatrix<f32>) -> DMatrix<f32> {
        let mut centered = data.clone();
        for mut row in centered.row_iter_mut() {
            for (i, val) in row.iter_mut().enumerate() {
                *val -= self.mean[i];
            }
        }
        // Project onto principal components
        let components_subset = self.components.columns(0, self.n_components);
        &centered * components_subset
    }

    /// Inverse transform projected data back to original space.
    pub fn inverse_transform(&self, data: &DMatrix<f32>) -> DMatrix<f32> {
        let components_subset = self.components.columns(0, self.n_components);
        let mut result = data * components_subset.transpose();
        for mut row in result.row_iter_mut() {
            for (i, val) in row.iter_mut().enumerate() {
                *val += self.mean[i];
            }
        }
        result
    }

    /// Get the projection matrix (for a single vector transformation).
    pub fn projection_matrix(&self) -> DMatrix<f32> {
        self.components.columns(0, self.n_components).into_owned()
    }
}

/// Fit PCA on the given data.
///
/// # Arguments
/// * `data` - Matrix where rows are samples and columns are features
/// * `n_components` - Number of principal components to keep (None = all)
pub fn fit_pca(data: &DMatrix<f32>, n_components: Option<usize>) -> Result<PcaResult> {
    if data.nrows() < 2 {
        return Err(ScannError::invalid_argument(
            "PCA requires at least 2 samples",
        ));
    }

    let n_features = data.ncols();
    let n_components = n_components.unwrap_or(n_features).min(n_features);

    // Center the data
    let (centered, mean) = center_data(data);

    // Compute SVD
    let svd = SVD::new(centered.clone(), true, true);

    let _u = svd.u.ok_or_else(|| ScannError::internal("SVD failed to compute U matrix"))?;
    let singular_values = svd.singular_values;
    let v_t = svd.v_t.ok_or_else(|| ScannError::internal("SVD failed to compute V^T matrix"))?;

    // Principal components are the right singular vectors (V^T transposed = V)
    let components = v_t.transpose();

    // Explained variance is proportional to singular values squared
    let n = data.nrows() as f32;
    let explained_variance = singular_values.map(|s| s * s / (n - 1.0));

    Ok(PcaResult {
        components,
        explained_variance,
        mean,
        n_components,
    })
}

/// Compute the squared L2 norm of each row in a matrix.
pub fn row_norms_squared(data: &DMatrix<f32>) -> DVector<f32> {
    let mut norms = DVector::zeros(data.nrows());
    for (i, row) in data.row_iter().enumerate() {
        norms[i] = row.iter().map(|x| x * x).sum();
    }
    norms
}

/// Normalize rows to unit length.
pub fn normalize_rows(data: &mut DMatrix<f32>) {
    for mut row in data.row_iter_mut() {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for val in row.iter_mut() {
                *val /= norm;
            }
        }
    }
}

/// Generate a random orthogonal matrix using QR decomposition.
pub fn random_orthogonal_matrix(dim: usize, seed: u64) -> DMatrix<f32> {
    use rand::{SeedableRng, Rng};
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);

    // Generate random matrix
    let mut random_matrix = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            random_matrix[(i, j)] = rng.gen::<f32>() * 2.0 - 1.0;
        }
    }

    // QR decomposition to get orthogonal matrix
    let qr = random_matrix.qr();
    qr.q()
}

/// Matrix multiplication wrapper.
pub fn matmul(a: &DMatrix<f32>, b: &DMatrix<f32>) -> DMatrix<f32> {
    a * b
}

/// Compute pairwise squared L2 distances between rows of two matrices.
pub fn pairwise_squared_distances(a: &DMatrix<f32>, b: &DMatrix<f32>) -> DMatrix<f32> {
    let n = a.nrows();
    let m = b.nrows();

    // ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b
    let a_norms = row_norms_squared(a);
    let b_norms = row_norms_squared(b);
    let ab = a * b.transpose();

    let mut distances = DMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            distances[(i, j)] = a_norms[i] + b_norms[j] - 2.0 * ab[(i, j)];
            // Clamp to avoid negative values due to numerical errors
            if distances[(i, j)] < 0.0 {
                distances[(i, j)] = 0.0;
            }
        }
    }
    distances
}

/// Convert a slice of f32 vectors to a nalgebra DMatrix.
pub fn vecs_to_matrix(vecs: &[Vec<f32>]) -> DMatrix<f32> {
    if vecs.is_empty() {
        return DMatrix::zeros(0, 0);
    }
    let n_rows = vecs.len();
    let n_cols = vecs[0].len();
    let mut matrix = DMatrix::zeros(n_rows, n_cols);
    for (i, vec) in vecs.iter().enumerate() {
        for (j, &val) in vec.iter().enumerate() {
            matrix[(i, j)] = val;
        }
    }
    matrix
}

/// Convert a DMatrix to a vector of f32 vectors.
pub fn matrix_to_vecs(matrix: &DMatrix<f32>) -> Vec<Vec<f32>> {
    let mut vecs = Vec::with_capacity(matrix.nrows());
    for row in matrix.row_iter() {
        vecs.push(row.iter().cloned().collect());
    }
    vecs
}

/// Eigendecomposition result.
#[derive(Debug, Clone)]
pub struct EigenResult {
    /// Eigenvalues (sorted in descending order).
    pub eigenvalues: DVector<f32>,
    /// Eigenvectors (columns correspond to eigenvalues).
    pub eigenvectors: DMatrix<f32>,
}

/// Compute eigendecomposition of a symmetric matrix.
pub fn symmetric_eigen(matrix: &DMatrix<f32>) -> Result<EigenResult> {
    use nalgebra::SymmetricEigen;

    let eigen = SymmetricEigen::new(matrix.clone());

    // Sort by eigenvalue in descending order
    let mut indices: Vec<usize> = (0..eigen.eigenvalues.len()).collect();
    indices.sort_by(|&a, &b| {
        eigen.eigenvalues[b]
            .partial_cmp(&eigen.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sorted_values = DVector::zeros(eigen.eigenvalues.len());
    let mut sorted_vectors = DMatrix::zeros(eigen.eigenvectors.nrows(), eigen.eigenvectors.ncols());

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        sorted_values[new_idx] = eigen.eigenvalues[old_idx];
        for i in 0..sorted_vectors.nrows() {
            sorted_vectors[(i, new_idx)] = eigen.eigenvectors[(i, old_idx)];
        }
    }

    Ok(EigenResult {
        eigenvalues: sorted_values,
        eigenvectors: sorted_vectors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_mean() {
        let data = DMatrix::from_row_slice(3, 2, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]);
        let mean = column_mean(&data);
        assert!((mean[0] - 3.0).abs() < 1e-6);
        assert!((mean[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_center_data() {
        let data = DMatrix::from_row_slice(3, 2, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]);
        let (centered, _mean) = center_data(&data);

        // Check that centered data has zero mean
        let new_mean = column_mean(&centered);
        assert!(new_mean[0].abs() < 1e-6);
        assert!(new_mean[1].abs() < 1e-6);
    }

    #[test]
    fn test_pca() {
        // Create synthetic data with clear structure
        let data = DMatrix::from_row_slice(4, 3, &[
            1.0, 0.1, 0.1,
            2.0, 0.2, 0.2,
            3.0, 0.3, 0.3,
            4.0, 0.4, 0.4,
        ]);

        let pca = fit_pca(&data, Some(2)).unwrap();

        assert_eq!(pca.n_components, 2);
        assert!(pca.explained_variance[0] >= pca.explained_variance[1]);

        // Transform and inverse transform should approximately recover original
        let transformed = pca.transform(&data);
        let recovered = pca.inverse_transform(&transformed);

        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                assert!((data[(i, j)] - recovered[(i, j)]).abs() < 0.1);
            }
        }
    }

    #[test]
    fn test_random_orthogonal_matrix() {
        let q = random_orthogonal_matrix(4, 42);

        // Check orthogonality: Q^T * Q should be identity
        let qt_q = q.transpose() * &q;
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((qt_q[(i, j)] - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_pairwise_distances() {
        let a = DMatrix::from_row_slice(2, 3, &[
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
        ]);
        let b = DMatrix::from_row_slice(2, 3, &[
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ]);

        let distances = pairwise_squared_distances(&a, &b);

        assert!((distances[(0, 0)] - 0.0).abs() < 1e-6); // (0,0,0) to (0,0,0)
        assert!((distances[(0, 1)] - 1.0).abs() < 1e-6); // (0,0,0) to (0,1,0)
        assert!((distances[(1, 0)] - 1.0).abs() < 1e-6); // (1,0,0) to (0,0,0)
        assert!((distances[(1, 1)] - 2.0).abs() < 1e-6); // (1,0,0) to (0,1,0)
    }
}
