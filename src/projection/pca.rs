//! PCA Projection implementation.

use crate::data_format::{Dataset, DenseDataset};
use crate::error::{Result, ScannError};
use crate::projection::Projection;
use crate::utils::linear_algebra::{fit_pca, vecs_to_matrix};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Configuration for PCA projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcaConfig {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Output dimensionality (number of principal components).
    pub output_dim: usize,
    /// Maximum number of training samples.
    pub max_training_samples: Option<usize>,
    /// Random seed for sampling.
    pub seed: Option<u64>,
}

impl PcaConfig {
    /// Create a new PCA configuration.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            max_training_samples: Some(100_000),
            seed: None,
        }
    }

    /// Set the maximum number of training samples.
    pub fn with_max_samples(mut self, max: usize) -> Self {
        self.max_training_samples = Some(max);
        self
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// PCA (Principal Component Analysis) projection.
#[derive(Clone)]
pub struct PcaProjection {
    config: PcaConfig,
    /// Projection matrix (input_dim x output_dim).
    projection_matrix: Option<DMatrix<f32>>,
    /// Mean vector for centering.
    mean: Option<Vec<f32>>,
    /// Whether the projection has been trained.
    trained: bool,
}

impl PcaProjection {
    /// Create a new PCA projection.
    pub fn new(config: PcaConfig) -> Self {
        Self {
            config,
            projection_matrix: None,
            mean: None,
            trained: false,
        }
    }

    /// Train the PCA projection on a dataset.
    pub fn train(&mut self, dataset: &DenseDataset<f32>) -> Result<()> {
        if dataset.is_empty() {
            return Err(ScannError::invalid_argument("Cannot train on empty dataset"));
        }

        let dim = dataset.dimensionality() as usize;
        if dim != self.config.input_dim {
            return Err(ScannError::invalid_argument(format!(
                "Dataset dimensionality {} does not match config {}",
                dim, self.config.input_dim
            )));
        }

        // Collect training data
        let max_samples = self.config.max_training_samples.unwrap_or(dataset.size());
        let num_samples = dataset.size().min(max_samples);

        let mut training_vecs = Vec::with_capacity(num_samples);

        if num_samples < dataset.size() {
            // Random sampling
            use rand::{SeedableRng, seq::SliceRandom};
            use rand::rngs::StdRng;

            let mut indices: Vec<usize> = (0..dataset.size()).collect();
            let mut rng = StdRng::seed_from_u64(self.config.seed.unwrap_or(42));
            indices.shuffle(&mut rng);

            for &idx in indices.iter().take(num_samples) {
                if let Some(dp) = dataset.get(idx as u32) {
                    training_vecs.push(dp.values().to_vec());
                }
            }
        } else {
            for i in 0..num_samples {
                if let Some(dp) = dataset.get(i as u32) {
                    training_vecs.push(dp.values().to_vec());
                }
            }
        }

        self.train_on_vecs(&training_vecs)
    }

    /// Train on a vector of vectors.
    pub fn train_on_vecs(&mut self, data: &[Vec<f32>]) -> Result<()> {
        if data.is_empty() {
            return Err(ScannError::invalid_argument("Cannot train on empty data"));
        }

        let matrix = vecs_to_matrix(data);
        let pca_result = fit_pca(&matrix, Some(self.config.output_dim))?;

        self.projection_matrix = Some(pca_result.projection_matrix());
        self.mean = Some(pca_result.mean.iter().cloned().collect());
        self.trained = true;

        Ok(())
    }

    /// Check if the projection has been trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get the projection matrix.
    pub fn projection_matrix(&self) -> Option<&DMatrix<f32>> {
        self.projection_matrix.as_ref()
    }

    /// Get the mean vector.
    pub fn mean(&self) -> Option<&[f32]> {
        self.mean.as_deref()
    }
}

impl Projection<f32> for PcaProjection {
    fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    fn output_dim(&self) -> usize {
        self.config.output_dim
    }

    fn project(&self, input: &[f32]) -> Vec<f32> {
        if !self.trained {
            // Return truncated input if not trained
            return input[..self.config.output_dim.min(input.len())].to_vec();
        }

        let proj_matrix = self.projection_matrix.as_ref().unwrap();
        let mean = self.mean.as_ref().unwrap();

        // Center the input
        let centered: Vec<f32> = input.iter()
            .zip(mean.iter())
            .map(|(&x, &m)| x - m)
            .collect();

        // Project: output = input * projection_matrix
        let mut output = vec![0.0f32; self.config.output_dim];
        for j in 0..self.config.output_dim {
            for i in 0..self.config.input_dim.min(centered.len()) {
                output[j] += centered[i] * proj_matrix[(i, j)];
            }
        }

        output
    }

    fn inverse_project(&self, input: &[f32]) -> Option<Vec<f32>> {
        if !self.trained {
            return None;
        }

        let proj_matrix = self.projection_matrix.as_ref().unwrap();
        let mean = self.mean.as_ref().unwrap();

        // Inverse project: original = projected * projection_matrix^T + mean
        let mut output = vec![0.0f32; self.config.input_dim];
        for i in 0..self.config.input_dim {
            for j in 0..self.config.output_dim.min(input.len()) {
                output[i] += input[j] * proj_matrix[(i, j)];
            }
            output[i] += mean[i];
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
    fn test_pca_projection_train() {
        // Create synthetic data with clear principal direction
        let data: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                let x = i as f32 / 10.0;
                vec![x, x * 0.5 + 0.1, x * 0.25 + 0.2, 0.1]
            })
            .collect();

        let mut pca = PcaProjection::new(PcaConfig::new(4, 2));
        pca.train_on_vecs(&data).unwrap();

        assert!(pca.is_trained());

        // Project a point
        let input = vec![1.0, 0.6, 0.35, 0.1];
        let projected = pca.project(&input);
        assert_eq!(projected.len(), 2);

        // Inverse project should approximately recover original
        let recovered = pca.inverse_project(&projected).unwrap();
        assert_eq!(recovered.len(), 4);
    }

    #[test]
    fn test_pca_dimensionality() {
        let pca = PcaProjection::new(PcaConfig::new(100, 10));
        assert_eq!(pca.input_dim(), 100);
        assert_eq!(pca.output_dim(), 10);
    }
}
