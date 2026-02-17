//! Chunking Projection implementation.
//!
//! Splits high-dimensional vectors into chunks and optionally applies
//! separate projections to each chunk.

use crate::projection::{Projection, ProjectionType};
use serde::{Deserialize, Serialize};

/// Configuration for chunking projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Number of chunks.
    pub num_chunks: usize,
    /// Whether to apply per-chunk projections.
    pub project_chunks: bool,
    /// Output dimension per chunk (if projecting).
    pub chunk_output_dim: Option<usize>,
}

impl ChunkingConfig {
    /// Create a new chunking configuration.
    pub fn new(input_dim: usize, num_chunks: usize) -> Self {
        assert!(input_dim % num_chunks == 0, "input_dim must be divisible by num_chunks");
        Self {
            input_dim,
            num_chunks,
            project_chunks: false,
            chunk_output_dim: None,
        }
    }

    /// Enable per-chunk projection.
    pub fn with_projection(mut self, output_dim_per_chunk: usize) -> Self {
        self.project_chunks = true;
        self.chunk_output_dim = Some(output_dim_per_chunk);
        self
    }
}

/// Chunking projection that splits vectors into chunks.
///
/// This is useful for product quantization where each subspace is
/// processed independently.
#[derive(Clone)]
pub struct ChunkingProjection {
    config: ChunkingConfig,
    /// Dimension of each chunk.
    chunk_dim: usize,
    /// Per-chunk projections (if enabled).
    chunk_projections: Vec<Option<ProjectionType>>,
}

impl ChunkingProjection {
    /// Create a new chunking projection.
    pub fn new(config: ChunkingConfig) -> Self {
        let chunk_dim = config.input_dim / config.num_chunks;
        let chunk_projections = vec![None; config.num_chunks];

        Self {
            config,
            chunk_dim,
            chunk_projections,
        }
    }

    /// Set a projection for a specific chunk.
    pub fn set_chunk_projection(&mut self, chunk_idx: usize, projection: ProjectionType) {
        if chunk_idx < self.chunk_projections.len() {
            self.chunk_projections[chunk_idx] = Some(projection);
        }
    }

    /// Get the number of chunks.
    pub fn num_chunks(&self) -> usize {
        self.config.num_chunks
    }

    /// Get the dimension of each input chunk.
    pub fn chunk_dim(&self) -> usize {
        self.chunk_dim
    }

    /// Get a single chunk from the input.
    pub fn get_chunk<'a>(&self, input: &'a [f32], chunk_idx: usize) -> &'a [f32] {
        let start = chunk_idx * self.chunk_dim;
        let end = start + self.chunk_dim;
        &input[start..end.min(input.len())]
    }

    /// Project a single chunk.
    pub fn project_chunk(&self, input: &[f32], chunk_idx: usize) -> Vec<f32> {
        let chunk = self.get_chunk(input, chunk_idx);

        if let Some(proj) = &self.chunk_projections[chunk_idx] {
            proj.project(chunk)
        } else {
            chunk.to_vec()
        }
    }

    /// Get output dimension per chunk.
    pub fn output_chunk_dim(&self) -> usize {
        self.config.chunk_output_dim.unwrap_or(self.chunk_dim)
    }
}

impl Projection<f32> for ChunkingProjection {
    fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    fn output_dim(&self) -> usize {
        self.output_chunk_dim() * self.config.num_chunks
    }

    fn project(&self, input: &[f32]) -> Vec<f32> {
        let out_chunk_dim = self.output_chunk_dim();
        let mut output = Vec::with_capacity(out_chunk_dim * self.config.num_chunks);

        for chunk_idx in 0..self.config.num_chunks {
            let projected_chunk = self.project_chunk(input, chunk_idx);
            output.extend(projected_chunk);
        }

        output
    }

    fn inverse_project(&self, input: &[f32]) -> Option<Vec<f32>> {
        let out_chunk_dim = self.output_chunk_dim();
        let mut output = Vec::with_capacity(self.config.input_dim);

        for chunk_idx in 0..self.config.num_chunks {
            let start = chunk_idx * out_chunk_dim;
            let end = (start + out_chunk_dim).min(input.len());
            let chunk = &input[start..end];

            if let Some(proj) = &self.chunk_projections[chunk_idx] {
                if let Some(inv) = inverse_project_type(proj, chunk) {
                    output.extend(inv);
                } else {
                    // Pad with zeros if inverse not available
                    output.extend(vec![0.0f32; self.chunk_dim]);
                }
            } else {
                output.extend(chunk);
            }
        }

        Some(output)
    }
}

// Helper to call inverse_project on ProjectionType
fn inverse_project_type(proj: &ProjectionType, input: &[f32]) -> Option<Vec<f32>> {
    match proj {
        ProjectionType::Identity(p) => p.inverse_project(input),
        ProjectionType::Pca(p) => p.inverse_project(input),
        ProjectionType::Random(p) => p.inverse_project(input),
        ProjectionType::Opq(p) => p.inverse_project(input),
        ProjectionType::Truncate(p) => p.inverse_project(input),
        ProjectionType::Chunking(p) => p.inverse_project(input),
    }
}

/// Iterator over chunks of a vector.
pub struct ChunkIterator<'a> {
    data: &'a [f32],
    chunk_size: usize,
    current: usize,
}

impl<'a> ChunkIterator<'a> {
    /// Create a new chunk iterator.
    pub fn new(data: &'a [f32], num_chunks: usize) -> Self {
        let chunk_size = data.len() / num_chunks;
        Self {
            data,
            chunk_size,
            current: 0,
        }
    }
}

impl<'a> Iterator for ChunkIterator<'a> {
    type Item = &'a [f32];

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.current * self.chunk_size;
        if start >= self.data.len() {
            return None;
        }

        let end = (start + self.chunk_size).min(self.data.len());
        self.current += 1;
        Some(&self.data[start..end])
    }
}

/// Split a vector into chunks.
pub fn split_into_chunks(data: &[f32], num_chunks: usize) -> Vec<&[f32]> {
    ChunkIterator::new(data, num_chunks).collect()
}

/// Interleave chunks back into a single vector.
pub fn interleave_chunks(chunks: &[Vec<f32>]) -> Vec<f32> {
    if chunks.is_empty() {
        return Vec::new();
    }

    let total_len: usize = chunks.iter().map(|c| c.len()).sum();
    let mut output = Vec::with_capacity(total_len);

    for chunk in chunks {
        output.extend(chunk);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunking_basic() {
        let config = ChunkingConfig::new(12, 3);
        let proj = ChunkingProjection::new(config);

        assert_eq!(proj.num_chunks(), 3);
        assert_eq!(proj.chunk_dim(), 4);

        let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let output = proj.project(&input);

        assert_eq!(output.len(), 12);
        assert_eq!(output, input);
    }

    #[test]
    fn test_get_chunk() {
        let config = ChunkingConfig::new(12, 3);
        let proj = ChunkingProjection::new(config);

        let input: Vec<f32> = (0..12).map(|i| i as f32).collect();

        assert_eq!(proj.get_chunk(&input, 0), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(proj.get_chunk(&input, 1), &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(proj.get_chunk(&input, 2), &[8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_chunk_iterator() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let chunks: Vec<_> = ChunkIterator::new(&data, 3).collect();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(chunks[1], &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(chunks[2], &[8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_split_and_interleave() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let chunks = split_into_chunks(&data, 3);
        let owned_chunks: Vec<Vec<f32>> = chunks.iter().map(|c| c.to_vec()).collect();
        let result = interleave_chunks(&owned_chunks);

        assert_eq!(result, data);
    }
}
