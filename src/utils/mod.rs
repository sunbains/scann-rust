//! Utility functions and types for ScaNN.

pub(crate) mod parallel;
pub(crate) mod random;
mod reordering;
pub mod linear_algebra;
pub mod bits;
pub mod gmm;

pub use parallel::{ParallelFor, ThreadPool, MIN_PARALLEL_SIZE};
pub use random::{RandomSampler, sample_indices};
pub use reordering::ReorderingHelper;
pub use linear_algebra::{PcaResult, fit_pca, vecs_to_matrix, matrix_to_vecs};
pub use bits::*;
pub use gmm::{GaussianMixture, GmmConfig, CovarianceType};
