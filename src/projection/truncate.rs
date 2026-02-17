//! Truncation Projection implementation.
//!
//! Simple dimensionality reduction by keeping only the first N dimensions.

use crate::projection::Projection;
use serde::{Deserialize, Serialize};

/// Configuration for truncation projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncateConfig {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Output dimensionality (must be <= input_dim).
    pub output_dim: usize,
    /// Starting dimension (for windowed truncation).
    pub start_dim: usize,
}

impl TruncateConfig {
    /// Create a new truncation configuration.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim: output_dim.min(input_dim),
            start_dim: 0,
        }
    }

    /// Set the starting dimension for windowed truncation.
    pub fn with_start(mut self, start: usize) -> Self {
        self.start_dim = start;
        self
    }
}

/// Truncation projection - keeps a subset of dimensions.
///
/// This is the simplest form of dimensionality reduction, useful when
/// the data has meaningful ordering (e.g., from PCA) or when the first
/// dimensions are known to be most important.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TruncateProjection {
    config: TruncateConfig,
}

impl TruncateProjection {
    /// Create a new truncation projection.
    pub fn new(config: TruncateConfig) -> Self {
        Self { config }
    }

    /// Create with simple parameters.
    pub fn simple(input_dim: usize, output_dim: usize) -> Self {
        Self::new(TruncateConfig::new(input_dim, output_dim))
    }

    /// Create a windowed truncation.
    pub fn windowed(input_dim: usize, start: usize, length: usize) -> Self {
        Self::new(TruncateConfig::new(input_dim, length).with_start(start))
    }

    /// Get input dimensionality (non-trait method for enum dispatch).
    pub fn get_input_dim(&self) -> usize {
        self.config.input_dim
    }

    /// Get output dimensionality (non-trait method for enum dispatch).
    pub fn get_output_dim(&self) -> usize {
        self.config.output_dim
    }
}

impl Projection<f32> for TruncateProjection {
    fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    fn output_dim(&self) -> usize {
        self.config.output_dim
    }

    fn project(&self, input: &[f32]) -> Vec<f32> {
        let start = self.config.start_dim;
        let end = (start + self.config.output_dim).min(input.len());

        if start >= input.len() {
            return vec![0.0; self.config.output_dim];
        }

        let mut output = input[start..end].to_vec();
        // Pad with zeros if needed
        while output.len() < self.config.output_dim {
            output.push(0.0);
        }
        output
    }

    fn project_into(&self, input: &[f32], output: &mut [f32]) {
        let start = self.config.start_dim;
        let len = self.config.output_dim.min(output.len());

        for i in 0..len {
            let src_idx = start + i;
            output[i] = if src_idx < input.len() {
                input[src_idx]
            } else {
                0.0
            };
        }
    }

    fn inverse_project(&self, input: &[f32]) -> Option<Vec<f32>> {
        // Pad with zeros in the truncated dimensions
        let mut output = vec![0.0f32; self.config.input_dim];
        let start = self.config.start_dim;

        for (i, &val) in input.iter().enumerate() {
            let dst_idx = start + i;
            if dst_idx < output.len() {
                output[dst_idx] = val;
            }
        }

        Some(output)
    }
}

// Also implement for other numeric types
impl Projection<f64> for TruncateProjection {
    fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    fn output_dim(&self) -> usize {
        self.config.output_dim
    }

    fn project(&self, input: &[f64]) -> Vec<f64> {
        let start = self.config.start_dim;
        let end = (start + self.config.output_dim).min(input.len());

        if start >= input.len() {
            return vec![0.0; self.config.output_dim];
        }

        let mut output = input[start..end].to_vec();
        while output.len() < self.config.output_dim {
            output.push(0.0);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_projection() {
        let proj = TruncateProjection::simple(10, 5);

        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let output = proj.project(&input);

        assert_eq!(output.len(), 5);
        assert_eq!(output, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_truncate_windowed() {
        let proj = TruncateProjection::windowed(10, 3, 4);

        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let output = proj.project(&input);

        assert_eq!(output.len(), 4);
        assert_eq!(output, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_truncate_inverse() {
        let proj = TruncateProjection::simple(10, 5);

        let input: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let output = proj.inverse_project(&input).unwrap();

        assert_eq!(output.len(), 10);
        assert_eq!(&output[0..5], &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&output[5..10], &[0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_truncate_short_input() {
        let proj = TruncateProjection::simple(10, 8);

        let input: Vec<f32> = vec![1.0, 2.0, 3.0]; // Only 3 elements
        let output = proj.project(&input);

        assert_eq!(output.len(), 8);
        assert_eq!(&output[0..3], &[1.0, 2.0, 3.0]);
        // Rest should be zeros
        assert!(output[3..].iter().all(|&x| x == 0.0));
    }
}
