//! Codebook for product quantization.
//!
//! This module provides codebooks that map subvectors to discrete codes.

use crate::data_format::{Dataset, DenseDataset};
use crate::error::{Result, ScannError};
use crate::trees::kmeans::{KMeans, KMeansConfig};
use crate::types::DatapointIndex;
use serde::{Deserialize, Serialize};

/// Configuration for codebook training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebookConfig {
    /// Number of codes (clusters) per subspace.
    pub num_codes: usize,

    /// Number of subspaces (dimension blocks).
    pub num_subspaces: usize,

    /// Maximum training iterations for K-means.
    pub max_iterations: usize,

    /// Convergence threshold.
    pub convergence_threshold: f64,

    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for CodebookConfig {
    fn default() -> Self {
        Self {
            num_codes: 256,
            num_subspaces: 8,
            max_iterations: 100,
            convergence_threshold: 1e-5,
            seed: None,
        }
    }
}

impl CodebookConfig {
    /// Create a new configuration.
    pub fn new(num_codes: usize, num_subspaces: usize) -> Self {
        Self {
            num_codes,
            num_subspaces,
            ..Default::default()
        }
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Codebook for a single subspace.
#[derive(Debug, Clone)]
pub struct SubspaceCodebook {
    /// Centroids for this subspace.
    pub centroids: Vec<Vec<f32>>,

    /// Dimension of each centroid.
    pub dim: usize,
}

impl SubspaceCodebook {
    /// Create a new subspace codebook.
    pub fn new(centroids: Vec<Vec<f32>>) -> Self {
        let dim = centroids.first().map(|c| c.len()).unwrap_or(0);
        Self { centroids, dim }
    }

    /// Get the number of codes.
    pub fn num_codes(&self) -> usize {
        self.centroids.len()
    }

    /// Encode a subvector to the nearest centroid index.
    pub fn encode(&self, subvector: &[f32]) -> u8 {
        let mut min_dist = f32::INFINITY;
        let mut min_idx = 0u8;

        for (i, centroid) in self.centroids.iter().enumerate() {
            let dist = Self::squared_distance(subvector, centroid);
            if dist < min_dist {
                min_dist = dist;
                min_idx = i as u8;
            }
        }

        min_idx
    }

    /// Compute distances from query subvector to all centroids.
    pub fn compute_distances(&self, query_subvector: &[f32]) -> Vec<f32> {
        self.centroids
            .iter()
            .map(|c| Self::squared_distance(query_subvector, c))
            .collect()
    }

    /// Squared distance.
    #[inline]
    fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }
}

/// Product quantization codebook.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Codebooks for each subspace.
    pub subspaces: Vec<SubspaceCodebook>,

    /// Configuration.
    pub config: CodebookConfig,

    /// Total dimensionality.
    pub dimensionality: usize,

    /// Dimensions per subspace.
    pub dims_per_subspace: usize,
}

impl Codebook {
    /// Create a new codebook with the given configuration.
    pub fn new(config: CodebookConfig) -> Self {
        Self {
            subspaces: Vec::new(),
            config,
            dimensionality: 0,
            dims_per_subspace: 0,
        }
    }

    /// Train the codebook from a dataset.
    pub fn train(&mut self, dataset: &DenseDataset<f32>) -> Result<()> {
        if dataset.is_empty() {
            return Err(ScannError::invalid_argument("Cannot train on empty dataset"));
        }

        let n = dataset.size();
        self.dimensionality = dataset.dimensionality() as usize;

        if self.dimensionality % self.config.num_subspaces != 0 {
            return Err(ScannError::invalid_argument(format!(
                "Dimensionality {} must be divisible by num_subspaces {}",
                self.dimensionality, self.config.num_subspaces
            )));
        }

        self.dims_per_subspace = self.dimensionality / self.config.num_subspaces;

        // Extract data
        let data: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                dataset
                    .get(i as DatapointIndex)
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();

        // Train codebook for each subspace
        self.subspaces = Vec::with_capacity(self.config.num_subspaces);

        for s in 0..self.config.num_subspaces {
            let start = s * self.dims_per_subspace;
            let end = start + self.dims_per_subspace;

            // Extract subvectors for this subspace
            let subvectors: Vec<Vec<f32>> = data
                .iter()
                .map(|v| v[start..end].to_vec())
                .collect();

            let subvector_dataset = DenseDataset::from_vecs(subvectors);

            // Train K-means on subvectors
            let kmeans_config = KMeansConfig::new(self.config.num_codes)
                .with_max_iterations(self.config.max_iterations)
                .with_convergence_threshold(self.config.convergence_threshold)
                .with_seed(self.config.seed.unwrap_or(42) + s as u64);

            let kmeans = KMeans::new(kmeans_config);
            let result = kmeans.fit(&subvector_dataset)?;

            self.subspaces.push(SubspaceCodebook::new(result.centers));
        }

        Ok(())
    }

    /// Encode a datapoint to codes.
    pub fn encode(&self, datapoint: &[f32]) -> Vec<u8> {
        self.subspaces
            .iter()
            .enumerate()
            .map(|(s, codebook)| {
                let start = s * self.dims_per_subspace;
                let end = start + self.dims_per_subspace;
                codebook.encode(&datapoint[start..end])
            })
            .collect()
    }

    /// Decode codes back to a vector.
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        if codes.len() != self.subspaces.len() {
            return vec![0.0; self.dimensionality];
        }

        let mut result = vec![0.0f32; self.dimensionality];

        for (s, (&code, codebook)) in codes.iter().zip(self.subspaces.iter()).enumerate() {
            let start = s * self.dims_per_subspace;
            if let Some(centroid) = codebook.centroids.get(code as usize) {
                for (i, &v) in centroid.iter().enumerate() {
                    result[start + i] = v;
                }
            }
        }

        result
    }

    /// Encode a dataset.
    pub fn encode_dataset(&self, dataset: &DenseDataset<f32>) -> Vec<Vec<u8>> {
        (0..dataset.size())
            .map(|i| {
                let dp = dataset.get(i as DatapointIndex).unwrap();
                self.encode(dp.values())
            })
            .collect()
    }

    /// Get the number of subspaces.
    pub fn num_subspaces(&self) -> usize {
        self.subspaces.len()
    }

    /// Get the number of codes per subspace.
    pub fn num_codes(&self) -> usize {
        self.config.num_codes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dataset() -> DenseDataset<f32> {
        let mut data = Vec::new();
        for i in 0..100 {
            let mut vec = Vec::with_capacity(16);
            for j in 0..16 {
                vec.push((i * j) as f32 / 100.0);
            }
            data.push(vec);
        }
        DenseDataset::from_vecs(data)
    }

    #[test]
    fn test_codebook_train() {
        let dataset = create_test_dataset();
        let config = CodebookConfig::new(8, 4).with_seed(42);
        let mut codebook = Codebook::new(config);
        codebook.train(&dataset).unwrap();

        assert_eq!(codebook.num_subspaces(), 4);
        assert_eq!(codebook.dims_per_subspace, 4);
        for subspace in &codebook.subspaces {
            assert!(subspace.num_codes() <= 8);
        }
    }

    #[test]
    fn test_codebook_encode() {
        let dataset = create_test_dataset();
        let config = CodebookConfig::new(8, 4).with_seed(42);
        let mut codebook = Codebook::new(config);
        codebook.train(&dataset).unwrap();

        let datapoint = vec![0.5f32; 16];
        let codes = codebook.encode(&datapoint);

        assert_eq!(codes.len(), 4);
        for &code in &codes {
            assert!(code < 8);
        }
    }

    #[test]
    fn test_codebook_encode_dataset() {
        let dataset = create_test_dataset();
        let config = CodebookConfig::new(8, 4).with_seed(42);
        let mut codebook = Codebook::new(config);
        codebook.train(&dataset).unwrap();

        let codes = codebook.encode_dataset(&dataset);

        assert_eq!(codes.len(), 100);
        for code_vec in &codes {
            assert_eq!(code_vec.len(), 4);
        }
    }
}
