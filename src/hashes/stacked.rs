//! Stacked Quantizers for multi-level product quantization.
//!
//! Stacked quantizers apply multiple levels of quantization to
//! progressively reduce quantization error.

use crate::data_format::{Dataset, DenseDataset};
use crate::error::{Result, ScannError};
use crate::hashes::codebook::{Codebook, CodebookConfig};
use serde::{Deserialize, Serialize};

/// Configuration for stacked quantizers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackedQuantizerConfig {
    /// Number of quantization levels.
    pub num_levels: usize,
    /// Number of codes per subspace at each level.
    pub codes_per_level: Vec<usize>,
    /// Number of subspaces at each level.
    pub subspaces_per_level: Vec<usize>,
    /// Random seed.
    pub seed: u64,
}

impl Default for StackedQuantizerConfig {
    fn default() -> Self {
        Self {
            num_levels: 2,
            codes_per_level: vec![256, 256],
            subspaces_per_level: vec![8, 8],
            seed: 42,
        }
    }
}

impl StackedQuantizerConfig {
    /// Create a two-level stacked quantizer config.
    pub fn two_level(
        first_codes: usize,
        first_subspaces: usize,
        second_codes: usize,
        second_subspaces: usize,
    ) -> Self {
        Self {
            num_levels: 2,
            codes_per_level: vec![first_codes, second_codes],
            subspaces_per_level: vec![first_subspaces, second_subspaces],
            seed: 42,
        }
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Stacked quantizer with multiple codebook levels.
pub struct StackedQuantizer {
    config: StackedQuantizerConfig,
    /// Codebooks for each level.
    codebooks: Vec<Codebook>,
    /// Whether trained.
    trained: bool,
    /// Dimensionality.
    dimensionality: usize,
}

impl StackedQuantizer {
    /// Create a new stacked quantizer.
    pub fn new(config: StackedQuantizerConfig) -> Self {
        Self {
            config,
            codebooks: Vec::new(),
            trained: false,
            dimensionality: 0,
        }
    }

    /// Train the stacked quantizer.
    pub fn train(&mut self, dataset: &DenseDataset<f32>) -> Result<()> {
        if dataset.is_empty() {
            return Err(ScannError::invalid_argument("Cannot train on empty dataset"));
        }

        self.dimensionality = dataset.dimensionality() as usize;
        self.codebooks.clear();

        // Current residuals (start with original data)
        let mut residuals: Vec<Vec<f32>> = (0..dataset.size())
            .filter_map(|i| dataset.get(i as u32).map(|dp| dp.values().to_vec()))
            .collect();

        // Train each level
        for level in 0..self.config.num_levels {
            let num_codes = self.config.codes_per_level[level];
            let num_subspaces = self.config.subspaces_per_level[level];

            // Create codebook config for this level
            let codebook_config = CodebookConfig::new(num_codes, num_subspaces)
                .with_seed(self.config.seed + level as u64);

            let mut codebook = Codebook::new(codebook_config);

            // Train on residuals
            let residual_dataset = DenseDataset::from_vecs(residuals.clone());
            codebook.train(&residual_dataset)?;

            // Compute new residuals
            let new_residuals: Vec<Vec<f32>> = residuals
                .iter()
                .map(|vec| {
                    let codes = codebook.encode(vec);
                    let reconstruction = codebook.decode(&codes);
                    vec.iter()
                        .zip(reconstruction.iter())
                        .map(|(&v, &r)| v - r)
                        .collect()
                })
                .collect();

            self.codebooks.push(codebook);
            residuals = new_residuals;
        }

        self.trained = true;
        Ok(())
    }

    /// Encode a vector using all levels.
    pub fn encode(&self, vector: &[f32]) -> Vec<Vec<u8>> {
        if !self.trained {
            return Vec::new();
        }

        let mut codes = Vec::with_capacity(self.config.num_levels);
        let mut residual = vector.to_vec();

        for codebook in &self.codebooks {
            let level_codes = codebook.encode(&residual);

            // Compute residual for next level
            let reconstruction = codebook.decode(&level_codes);
            residual = residual
                .iter()
                .zip(reconstruction.iter())
                .map(|(&r, &c)| r - c)
                .collect();

            codes.push(level_codes);
        }

        codes
    }

    /// Decode codes back to a vector.
    pub fn decode(&self, codes: &[Vec<u8>]) -> Vec<f32> {
        if codes.is_empty() || !self.trained {
            return vec![0.0; self.dimensionality];
        }

        let mut result = vec![0.0f32; self.dimensionality];

        for (codebook, level_codes) in self.codebooks.iter().zip(codes.iter()) {
            let level_reconstruction = codebook.decode(level_codes);
            for (r, &v) in result.iter_mut().zip(level_reconstruction.iter()) {
                *r += v;
            }
        }

        result
    }

    /// Encode a dataset.
    pub fn encode_dataset(&self, dataset: &DenseDataset<f32>) -> Vec<Vec<Vec<u8>>> {
        (0..dataset.size())
            .filter_map(|i| dataset.get(i as u32).map(|dp| self.encode(dp.values())))
            .collect()
    }

    /// Get the number of levels.
    pub fn num_levels(&self) -> usize {
        self.config.num_levels
    }

    /// Check if trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get the codebooks.
    pub fn codebooks(&self) -> &[Codebook] {
        &self.codebooks
    }

    /// Compute quantization error for a vector.
    pub fn quantization_error(&self, vector: &[f32]) -> f32 {
        let codes = self.encode(vector);
        let reconstruction = self.decode(&codes);

        vector
            .iter()
            .zip(reconstruction.iter())
            .map(|(&v, &r)| {
                let diff = v - r;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }
}

/// Additive quantization variant where centroids are summed.
pub struct AdditiveQuantizer {
    config: StackedQuantizerConfig,
    /// Codebooks (one per "addend").
    codebooks: Vec<Codebook>,
    /// Whether trained.
    trained: bool,
    /// Dimensionality.
    dimensionality: usize,
}

impl AdditiveQuantizer {
    /// Create a new additive quantizer.
    pub fn new(config: StackedQuantizerConfig) -> Self {
        Self {
            config,
            codebooks: Vec::new(),
            trained: false,
            dimensionality: 0,
        }
    }

    /// Train using beam search optimization.
    pub fn train(&mut self, dataset: &DenseDataset<f32>) -> Result<()> {
        // Simplified training: use residual quantization approach
        // (Full additive quantization would use joint optimization)

        if dataset.is_empty() {
            return Err(ScannError::invalid_argument("Cannot train on empty dataset"));
        }

        self.dimensionality = dataset.dimensionality() as usize;
        self.codebooks.clear();

        let mut residuals: Vec<Vec<f32>> = (0..dataset.size())
            .filter_map(|i| dataset.get(i as u32).map(|dp| dp.values().to_vec()))
            .collect();

        for level in 0..self.config.num_levels {
            let num_codes = self.config.codes_per_level[level];
            // For additive quantization, use full dimension
            let num_subspaces = 1;

            let codebook_config = CodebookConfig::new(num_codes, num_subspaces)
                .with_seed(self.config.seed + level as u64);

            let mut codebook = Codebook::new(codebook_config);
            let residual_dataset = DenseDataset::from_vecs(residuals.clone());
            codebook.train(&residual_dataset)?;

            // Update residuals
            residuals = residuals
                .iter()
                .map(|vec| {
                    let codes = codebook.encode(vec);
                    let reconstruction = codebook.decode(&codes);
                    vec.iter()
                        .zip(reconstruction.iter())
                        .map(|(&v, &r)| v - r)
                        .collect()
                })
                .collect();

            self.codebooks.push(codebook);
        }

        self.trained = true;
        Ok(())
    }

    /// Encode a vector.
    pub fn encode(&self, vector: &[f32]) -> Vec<Vec<u8>> {
        if !self.trained {
            return Vec::new();
        }

        let mut codes = Vec::new();
        let mut residual = vector.to_vec();

        for codebook in &self.codebooks {
            let level_codes = codebook.encode(&residual);
            let reconstruction = codebook.decode(&level_codes);

            residual = residual
                .iter()
                .zip(reconstruction.iter())
                .map(|(&r, &c)| r - c)
                .collect();

            codes.push(level_codes);
        }

        codes
    }

    /// Decode codes to a vector.
    pub fn decode(&self, codes: &[Vec<u8>]) -> Vec<f32> {
        if !self.trained {
            return vec![0.0; self.dimensionality];
        }

        let mut result = vec![0.0f32; self.dimensionality];

        for (codebook, level_codes) in self.codebooks.iter().zip(codes.iter()) {
            let reconstruction = codebook.decode(level_codes);
            for (r, &v) in result.iter_mut().zip(reconstruction.iter()) {
                *r += v;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dataset() -> DenseDataset<f32> {
        let data: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..16)
                    .map(|j| ((i * j) as f32 / 100.0).sin())
                    .collect()
            })
            .collect();
        DenseDataset::from_vecs(data)
    }

    #[test]
    fn test_stacked_quantizer() {
        let dataset = create_test_dataset();
        let config = StackedQuantizerConfig::two_level(16, 4, 16, 4);
        let mut sq = StackedQuantizer::new(config);

        sq.train(&dataset).unwrap();
        assert!(sq.is_trained());
        assert_eq!(sq.num_levels(), 2);

        // Test encode/decode
        let vector: Vec<f32> = (0..16).map(|i| i as f32 / 10.0).collect();
        let codes = sq.encode(&vector);
        assert_eq!(codes.len(), 2);

        let reconstructed = sq.decode(&codes);
        assert_eq!(reconstructed.len(), 16);

        // Reconstruction should be reasonably close
        let error = sq.quantization_error(&vector);
        assert!(error < 2.0);
    }

    #[test]
    fn test_stacked_quantizer_dataset() {
        let dataset = create_test_dataset();
        let config = StackedQuantizerConfig::two_level(8, 4, 8, 4);
        let mut sq = StackedQuantizer::new(config);

        sq.train(&dataset).unwrap();

        let encoded = sq.encode_dataset(&dataset);
        assert_eq!(encoded.len(), 100);

        // Each encoded point should have 2 levels
        for codes in &encoded {
            assert_eq!(codes.len(), 2);
        }
    }
}
