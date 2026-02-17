//! Random Orthogonal Projection implementation.

use crate::projection::Projection;
use crate::utils::linear_algebra::random_orthogonal_matrix;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Configuration for random orthogonal projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomProjectionConfig {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Output dimensionality.
    pub output_dim: usize,
    /// Random seed.
    pub seed: u64,
}

impl RandomProjectionConfig {
    /// Create a new random projection configuration.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            seed: 42,
        }
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Random orthogonal projection for dimensionality reduction.
///
/// Uses a random orthogonal matrix to project high-dimensional data
/// to a lower-dimensional space while approximately preserving distances.
#[derive(Clone)]
pub struct RandomOrthogonalProjection {
    config: RandomProjectionConfig,
    /// Projection matrix.
    projection_matrix: DMatrix<f32>,
}

impl RandomOrthogonalProjection {
    /// Create a new random orthogonal projection.
    pub fn new(config: RandomProjectionConfig) -> Self {
        let max_dim = config.input_dim.max(config.output_dim);

        // Generate a random orthogonal matrix
        let full_matrix = random_orthogonal_matrix(max_dim, config.seed);

        // Extract the submatrix we need (input_dim x output_dim)
        let projection_matrix = full_matrix.view(
            (0, 0),
            (config.input_dim, config.output_dim),
        ).into_owned();

        Self {
            config,
            projection_matrix,
        }
    }

    /// Get the projection matrix.
    pub fn projection_matrix(&self) -> &DMatrix<f32> {
        &self.projection_matrix
    }
}

impl Projection<f32> for RandomOrthogonalProjection {
    fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    fn output_dim(&self) -> usize {
        self.config.output_dim
    }

    fn project(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; self.config.output_dim];

        for j in 0..self.config.output_dim {
            for i in 0..self.config.input_dim.min(input.len()) {
                output[j] += input[i] * self.projection_matrix[(i, j)];
            }
        }

        output
    }

    fn inverse_project(&self, input: &[f32]) -> Option<Vec<f32>> {
        // Random orthogonal projection is approximately invertible
        // using the transpose (since Q^T * Q ≈ I for square Q)
        let mut output = vec![0.0f32; self.config.input_dim];

        for i in 0..self.config.input_dim {
            for j in 0..self.config.output_dim.min(input.len()) {
                output[i] += input[j] * self.projection_matrix[(i, j)];
            }
        }

        Some(output)
    }
}

/// Gaussian random projection (non-orthogonal).
///
/// Uses a random Gaussian matrix for projection, which can be more
/// efficient to generate for very high dimensions.
#[derive(Clone)]
pub struct GaussianRandomProjection {
    config: RandomProjectionConfig,
    projection_matrix: DMatrix<f32>,
}

impl GaussianRandomProjection {
    /// Create a new Gaussian random projection.
    pub fn new(config: RandomProjectionConfig) -> Self {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand_distr::{Normal, Distribution};

        let mut rng = StdRng::seed_from_u64(config.seed);
        let normal = Normal::new(0.0f32, 1.0 / (config.output_dim as f32).sqrt()).unwrap();

        let mut projection_matrix = DMatrix::zeros(config.input_dim, config.output_dim);
        for i in 0..config.input_dim {
            for j in 0..config.output_dim {
                projection_matrix[(i, j)] = normal.sample(&mut rng);
            }
        }

        Self {
            config,
            projection_matrix,
        }
    }

    /// Get the projection matrix.
    pub fn projection_matrix(&self) -> &DMatrix<f32> {
        &self.projection_matrix
    }
}

impl Projection<f32> for GaussianRandomProjection {
    fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    fn output_dim(&self) -> usize {
        self.config.output_dim
    }

    fn project(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; self.config.output_dim];

        for j in 0..self.config.output_dim {
            for i in 0..self.config.input_dim.min(input.len()) {
                output[j] += input[i] * self.projection_matrix[(i, j)];
            }
        }

        output
    }
}

/// Sparse random projection using the Achlioptas method.
///
/// Uses a sparse matrix with entries {-1, 0, +1} for efficient computation.
#[derive(Clone)]
pub struct SparseRandomProjection {
    config: RandomProjectionConfig,
    /// Non-zero entries: (row, col, value).
    entries: Vec<(usize, usize, f32)>,
    /// Density of the random matrix (default: 1/sqrt(input_dim)).
    density: f32,
}

impl SparseRandomProjection {
    /// Create a new sparse random projection.
    pub fn new(config: RandomProjectionConfig) -> Self {
        Self::with_density(config, None)
    }

    /// Create with specified density.
    pub fn with_density(config: RandomProjectionConfig, density: Option<f32>) -> Self {
        use rand::{SeedableRng, Rng};
        use rand::rngs::StdRng;

        let density = density.unwrap_or(1.0 / (config.input_dim as f32).sqrt());
        let scale = (1.0 / (density * config.output_dim as f32)).sqrt();

        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut entries = Vec::new();

        for i in 0..config.input_dim {
            for j in 0..config.output_dim {
                if rng.gen::<f32>() < density {
                    let value = if rng.gen::<bool>() { scale } else { -scale };
                    entries.push((i, j, value));
                }
            }
        }

        Self {
            config,
            entries,
            density,
        }
    }
}

impl Projection<f32> for SparseRandomProjection {
    fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    fn output_dim(&self) -> usize {
        self.config.output_dim
    }

    fn project(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; self.config.output_dim];

        for &(i, j, val) in &self.entries {
            if i < input.len() {
                output[j] += input[i] * val;
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_orthogonal_projection() {
        let config = RandomProjectionConfig::new(10, 5).with_seed(42);
        let proj = RandomOrthogonalProjection::new(config);

        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let output = proj.project(&input);

        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_gaussian_random_projection() {
        let config = RandomProjectionConfig::new(100, 20);
        let proj = GaussianRandomProjection::new(config);

        let input: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let output = proj.project(&input);

        assert_eq!(output.len(), 20);
    }

    #[test]
    fn test_sparse_random_projection() {
        let config = RandomProjectionConfig::new(100, 20);
        let proj = SparseRandomProjection::new(config);

        let input: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let output = proj.project(&input);

        assert_eq!(output.len(), 20);
    }

    #[test]
    fn test_orthogonal_preserves_norms_approximately() {
        let config = RandomProjectionConfig::new(100, 100).with_seed(123);
        let proj = RandomOrthogonalProjection::new(config);

        let input: Vec<f32> = (0..100).map(|i| (i as f32) / 10.0).collect();
        let output = proj.project(&input);

        let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let output_norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();

        // For square orthogonal matrix, norms should be exactly preserved
        assert!((input_norm - output_norm).abs() < 0.1);
    }
}
