//! Optimized Product Quantization (OPQ) Projection.
//!
//! OPQ learns a rotation matrix that minimizes quantization error
//! for product quantization.

use crate::data_format::{Dataset, DenseDataset};
use crate::error::{Result, ScannError};
use crate::projection::Projection;
use crate::utils::linear_algebra::{vecs_to_matrix, symmetric_eigen, random_orthogonal_matrix};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Configuration for OPQ projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpqConfig {
    /// Input/output dimensionality.
    pub dim: usize,
    /// Number of subspaces for product quantization.
    pub num_subspaces: usize,
    /// Number of training iterations.
    pub num_iterations: usize,
    /// Random seed.
    pub seed: u64,
}

impl OpqConfig {
    /// Create a new OPQ configuration.
    pub fn new(dim: usize, num_subspaces: usize) -> Self {
        Self {
            dim,
            num_subspaces,
            num_iterations: 20,
            seed: 42,
        }
    }

    /// Set the number of iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.num_iterations = iterations;
        self
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// OPQ (Optimized Product Quantization) projection.
///
/// Learns a rotation matrix that minimizes quantization error when
/// used with product quantization.
#[derive(Clone)]
pub struct OpqProjection {
    config: OpqConfig,
    /// Rotation matrix (orthogonal).
    rotation_matrix: Option<DMatrix<f32>>,
    /// Whether trained.
    trained: bool,
}

impl OpqProjection {
    /// Create a new OPQ projection.
    pub fn new(config: OpqConfig) -> Self {
        Self {
            config,
            rotation_matrix: None,
            trained: false,
        }
    }

    /// Train the OPQ rotation matrix.
    pub fn train(&mut self, dataset: &DenseDataset<f32>) -> Result<()> {
        if dataset.is_empty() {
            return Err(ScannError::invalid_argument("Cannot train on empty dataset"));
        }

        // Collect data
        let mut vecs = Vec::with_capacity(dataset.size());
        for i in 0..dataset.size() {
            if let Some(dp) = dataset.get(i as u32) {
                vecs.push(dp.values().to_vec());
            }
        }

        self.train_on_vecs(&vecs)
    }

    /// Train on vectors.
    pub fn train_on_vecs(&mut self, data: &[Vec<f32>]) -> Result<()> {
        if data.is_empty() {
            return Err(ScannError::invalid_argument("Cannot train on empty data"));
        }

        let dim = data[0].len();
        if dim != self.config.dim {
            return Err(ScannError::invalid_argument(
                "Data dimension does not match config",
            ));
        }

        let subspace_dim = dim / self.config.num_subspaces;
        if dim % self.config.num_subspaces != 0 {
            return Err(ScannError::invalid_argument(
                "Dimension must be divisible by num_subspaces",
            ));
        }

        // Initialize with random orthogonal matrix
        let mut rotation = random_orthogonal_matrix(dim, self.config.seed);
        let data_matrix = vecs_to_matrix(data);

        // Iterative refinement
        for _iter in 0..self.config.num_iterations {
            // Rotate data
            let rotated = &data_matrix * &rotation;

            // For each subspace, compute optimal rotation
            // This is a simplified version - full OPQ would use k-means
            let mut new_rotation = DMatrix::zeros(dim, dim);

            for s in 0..self.config.num_subspaces {
                let start = s * subspace_dim;
                let _end = start + subspace_dim;

                // Extract subspace data
                let subspace_data = rotated.columns(start, subspace_dim).into_owned();

                // Compute covariance
                let cov = subspace_data.transpose() * &subspace_data;

                // Eigendecomposition
                if let Ok(eigen) = symmetric_eigen(&cov) {
                    // Use eigenvectors for this subspace block
                    for i in 0..subspace_dim {
                        for j in 0..subspace_dim {
                            new_rotation[(start + i, start + j)] = eigen.eigenvectors[(i, j)];
                        }
                    }
                } else {
                    // Fall back to identity for this block
                    for i in 0..subspace_dim {
                        new_rotation[(start + i, start + i)] = 1.0;
                    }
                }
            }

            // Update rotation: R = R * new_rotation
            rotation = &rotation * &new_rotation;
        }

        self.rotation_matrix = Some(rotation);
        self.trained = true;

        Ok(())
    }

    /// Check if trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get the rotation matrix.
    pub fn rotation_matrix(&self) -> Option<&DMatrix<f32>> {
        self.rotation_matrix.as_ref()
    }
}

impl Projection<f32> for OpqProjection {
    fn input_dim(&self) -> usize {
        self.config.dim
    }

    fn output_dim(&self) -> usize {
        self.config.dim
    }

    fn project(&self, input: &[f32]) -> Vec<f32> {
        if !self.trained {
            return input.to_vec();
        }

        let rotation = self.rotation_matrix.as_ref().unwrap();
        let mut output = vec![0.0f32; self.config.dim];

        for j in 0..self.config.dim {
            for i in 0..self.config.dim.min(input.len()) {
                output[j] += input[i] * rotation[(i, j)];
            }
        }

        output
    }

    fn inverse_project(&self, input: &[f32]) -> Option<Vec<f32>> {
        if !self.trained {
            return Some(input.to_vec());
        }

        let rotation = self.rotation_matrix.as_ref().unwrap();
        let mut output = vec![0.0f32; self.config.dim];

        // Inverse is transpose for orthogonal matrix
        for i in 0..self.config.dim {
            for j in 0..self.config.dim.min(input.len()) {
                output[i] += input[j] * rotation[(i, j)];
            }
        }

        Some(output)
    }

    fn is_trainable(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opq_projection() {
        // Create synthetic data
        let data: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..8).map(|j| ((i * j) as f32 / 100.0).sin()).collect()
            })
            .collect();

        let mut opq = OpqProjection::new(OpqConfig::new(8, 2).with_iterations(5));
        opq.train_on_vecs(&data).unwrap();

        assert!(opq.is_trained());

        // Project and inverse should approximately recover original
        let input: Vec<f32> = (0..8).map(|i| i as f32 / 10.0).collect();
        let projected = opq.project(&input);
        let recovered = opq.inverse_project(&projected).unwrap();

        assert_eq!(projected.len(), 8);
        assert_eq!(recovered.len(), 8);

        // Check approximate recovery
        for (orig, rec) in input.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.5);
        }
    }
}
