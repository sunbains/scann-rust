//! Projection module for dimensionality reduction.
//!
//! This module provides various projection methods including:
//! - PCA (Principal Component Analysis)
//! - Random Orthogonal Projection
//! - OPQ (Optimized Product Quantization)
//! - Truncation Projection
//! - Chunking Projection

mod pca;
mod random;
mod opq;
mod truncate;
mod chunking;

pub use pca::{PcaProjection, PcaConfig};
pub use random::{RandomOrthogonalProjection, RandomProjectionConfig};
pub use opq::{OpqProjection, OpqConfig};
pub use truncate::{TruncateProjection, TruncateConfig};
pub use chunking::{ChunkingProjection, ChunkingConfig};

use crate::types::DatapointValue;
use serde::{Deserialize, Serialize};

/// Trait for projection methods.
pub trait Projection<T: DatapointValue>: Send + Sync {
    /// Get the input dimensionality.
    fn input_dim(&self) -> usize;

    /// Get the output dimensionality.
    fn output_dim(&self) -> usize;

    /// Project a single datapoint.
    fn project(&self, input: &[T]) -> Vec<T>;

    /// Project a single datapoint into a pre-allocated buffer.
    fn project_into(&self, input: &[T], output: &mut [T]) {
        let projected = self.project(input);
        output[..projected.len()].copy_from_slice(&projected);
    }

    /// Project a batch of datapoints.
    fn project_batch(&self, inputs: &[Vec<T>]) -> Vec<Vec<T>> {
        inputs.iter().map(|v| self.project(v)).collect()
    }

    /// Inverse project (if supported).
    fn inverse_project(&self, _input: &[T]) -> Option<Vec<T>> {
        None
    }

    /// Check if this projection is trainable.
    fn is_trainable(&self) -> bool {
        false
    }
}

/// Identity projection (no-op).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityProjection {
    dim: usize,
}

impl IdentityProjection {
    /// Create a new identity projection.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Get the dimensionality (non-trait method for enum dispatch).
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl<T: DatapointValue> Projection<T> for IdentityProjection {
    fn input_dim(&self) -> usize {
        self.dim
    }

    fn output_dim(&self) -> usize {
        self.dim
    }

    fn project(&self, input: &[T]) -> Vec<T> {
        input.to_vec()
    }

    fn project_into(&self, input: &[T], output: &mut [T]) {
        output[..input.len()].copy_from_slice(input);
    }

    fn inverse_project(&self, input: &[T]) -> Option<Vec<T>> {
        Some(input.to_vec())
    }
}

/// Enum wrapping all projection types for dynamic dispatch.
#[derive(Clone)]
pub enum ProjectionType {
    Identity(IdentityProjection),
    Pca(PcaProjection),
    Random(RandomOrthogonalProjection),
    Opq(OpqProjection),
    Truncate(TruncateProjection),
    Chunking(ChunkingProjection),
}

impl ProjectionType {
    /// Project a single datapoint.
    pub fn project(&self, input: &[f32]) -> Vec<f32> {
        match self {
            ProjectionType::Identity(p) => p.project(input),
            ProjectionType::Pca(p) => p.project(input),
            ProjectionType::Random(p) => p.project(input),
            ProjectionType::Opq(p) => p.project(input),
            ProjectionType::Truncate(p) => p.project(input),
            ProjectionType::Chunking(p) => p.project(input),
        }
    }

    /// Get output dimensionality.
    pub fn output_dim(&self) -> usize {
        match self {
            ProjectionType::Identity(p) => p.dim(),
            ProjectionType::Pca(p) => Projection::<f32>::output_dim(p),
            ProjectionType::Random(p) => Projection::<f32>::output_dim(p),
            ProjectionType::Opq(p) => Projection::<f32>::output_dim(p),
            ProjectionType::Truncate(p) => p.get_output_dim(),
            ProjectionType::Chunking(p) => Projection::<f32>::output_dim(p),
        }
    }

    /// Get input dimensionality.
    pub fn input_dim(&self) -> usize {
        match self {
            ProjectionType::Identity(p) => p.dim(),
            ProjectionType::Pca(p) => Projection::<f32>::input_dim(p),
            ProjectionType::Random(p) => Projection::<f32>::input_dim(p),
            ProjectionType::Opq(p) => Projection::<f32>::input_dim(p),
            ProjectionType::Truncate(p) => p.get_input_dim(),
            ProjectionType::Chunking(p) => Projection::<f32>::input_dim(p),
        }
    }
}

/// Factory for creating projections.
pub struct ProjectionFactory;

impl ProjectionFactory {
    /// Create an identity projection.
    pub fn identity(dim: usize) -> ProjectionType {
        ProjectionType::Identity(IdentityProjection::new(dim))
    }

    /// Create a PCA projection.
    pub fn pca(config: PcaConfig) -> ProjectionType {
        ProjectionType::Pca(PcaProjection::new(config))
    }

    /// Create a random orthogonal projection.
    pub fn random(config: RandomProjectionConfig) -> ProjectionType {
        ProjectionType::Random(RandomOrthogonalProjection::new(config))
    }

    /// Create an OPQ projection.
    pub fn opq(config: OpqConfig) -> ProjectionType {
        ProjectionType::Opq(OpqProjection::new(config))
    }

    /// Create a truncation projection.
    pub fn truncate(config: TruncateConfig) -> ProjectionType {
        ProjectionType::Truncate(TruncateProjection::new(config))
    }

    /// Create a chunking projection.
    pub fn chunking(config: ChunkingConfig) -> ProjectionType {
        ProjectionType::Chunking(ChunkingProjection::new(config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_projection() {
        let proj = IdentityProjection::new(4);
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let output = proj.project(&input);
        assert_eq!(output, input);
    }
}
