//! # ScaNN - Scalable Nearest Neighbors
//!
//! A high-performance Rust implementation of Google's ScaNN library for efficient
//! approximate nearest neighbor search.
//!
//! ## Overview
//!
//! ScaNN provides highly optimized algorithms for vector similarity search:
//!
//! - **Brute-force search**: Exact nearest neighbor search with SIMD acceleration
//! - **Partitioning**: K-means trees for approximate search on large datasets
//! - **Asymmetric hashing**: LUT16 with PSHUFB for extremely fast scoring
//! - **SIMD optimizations**: AVX2-accelerated distance computations (6-10x speedup)
//! - **Projections**: PCA, Random Orthogonal, OPQ for dimensionality reduction
//! - **Quantization**: Int8, FP8, BFloat16 for 2-4x memory reduction
//! - **Tree-X-Hybrid**: Combined tree + hashing for large-scale search
//! - **Dynamic updates**: Add, update, delete points from indices
//!
//! ## Quick Start
//!
//! ```rust
//! use scann::prelude::*;
//!
//! // Create a dataset
//! let data = vec![
//!     vec![1.0, 2.0, 3.0],
//!     vec![4.0, 5.0, 6.0],
//!     vec![7.0, 8.0, 9.0],
//! ];
//!
//! // Build a brute-force searcher
//! let dataset = DenseDataset::from_vecs(data);
//! let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);
//!
//! // Search for nearest neighbors
//! let query = vec![1.0, 2.0, 3.0];
//! let results = searcher.search(&query, 2).unwrap();
//!
//! for (idx, distance) in results {
//!     println!("Index: {}, Distance: {:.4}", idx, distance);
//! }
//! ```
//!
//! ## Batched Search
//!
//! For high throughput, use batched search:
//!
//! ```rust
//! use scann::prelude::*;
//!
//! let data: Vec<Vec<f32>> = (0..1000)
//!     .map(|i| vec![i as f32; 64])
//!     .collect();
//! let dataset = DenseDataset::from_vecs(data);
//! let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);
//!
//! // Batch 100 queries for better cache utilization
//! let queries: Vec<Vec<f32>> = (0..100)
//!     .map(|i| vec![i as f32; 64])
//!     .collect();
//! let results = searcher.search_batched(&queries, 10).unwrap();
//! // Achieves ~47k QPS on 10k point dataset
//! ```
//!
//! ## Memory-Efficient Search
//!
//! Use scalar quantization for 4x memory reduction:
//!
//! ```rust
//! use scann::prelude::*;
//! use scann::brute_force::{ScalarQuantizedBruteForceSearcher, ScalarQuantizedConfig};
//!
//! let data: Vec<Vec<f32>> = (0..1000)
//!     .map(|i| (0..128).map(|j| (i * j) as f32 / 1000.0).collect())
//!     .collect();
//! let dataset = DenseDataset::from_vecs(data);
//!
//! // Int8 quantization: 4x memory reduction, ~99% recall
//! let config = ScalarQuantizedConfig::squared_l2();
//! let searcher = ScalarQuantizedBruteForceSearcher::new(&dataset, config).unwrap();
//!
//! let query: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
//! let results = searcher.search(&query, 10).unwrap();
//! ```
//!
//! ## Distance Measures
//!
//! Supported distance measures:
//!
//! | Measure | Description | Use Case |
//! |---------|-------------|----------|
//! | `SquaredL2` | Squared Euclidean | General purpose (fastest) |
//! | `L2` | Euclidean | When actual distances needed |
//! | `L1` | Manhattan | Sparse data, outlier robust |
//! | `Cosine` | Cosine distance | Text/image embeddings |
//! | `DotProduct` | Negative dot product | Maximum inner product search |
//! | `Hamming` | Hamming distance | Binary vectors |
//! | `Jaccard` | Jaccard distance | Set similarity |
//!
//! ## Performance
//!
//! Typical performance on modern CPUs (AVX2):
//!
//! | Operation | Dataset | Throughput |
//! |-----------|---------|------------|
//! | Brute Force | 10k x 64d | 2,941 QPS |
//! | Batched (100q) | 10k x 64d | 47,619 QPS |
//! | SIMD Dot Product | 128d | 83M ops/sec |
//! | LUT16 Batch | 1k points | 48M lookups/sec |
//!
//! ## Module Overview
//!
//! - [`brute_force`]: Exact nearest neighbor search
//! - [`distance_measures`]: Distance computation functions
//! - [`data_format`]: Dataset and datapoint types
//! - [`trees`]: K-means clustering and tree structures
//! - [`hashes`]: Asymmetric hashing and product quantization
//! - [`projection`]: Dimensionality reduction (PCA, Random, OPQ)
//! - [`quantization`]: Value quantization (Int8, FP8, BFloat16)
//! - [`restricts`]: Search filtering and crowding
//! - [`mutator`]: Dynamic index updates
//! - [`tree_x_hybrid`]: Combined tree + hashing search
//! - [`simd`]: Low-level SIMD operations

// Allow dead code for utility functions that may be used in the future
#![allow(dead_code)]
// Allow some clippy lints that are not critical
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::approx_constant)]
#![allow(clippy::module_inception)]
#![allow(clippy::bool_assert_comparison)]
#![allow(clippy::vec_init_then_push)]

pub mod data_format;
pub mod distance_measures;
pub mod brute_force;
pub mod partitioning;
pub mod trees;
pub mod hashes;
pub mod utils;
pub mod projection;
pub mod quantization;
pub mod restricts;
pub mod mutator;
pub mod tree_x_hybrid;
pub mod simd;

mod types;
mod error;
mod config;
mod searcher;
mod scann;

pub use types::*;
pub use error::{ScannError, Result};
pub use config::ScannConfig;
pub use searcher::{Searcher, SearchParameters, SearchResult, NNResult};
pub use scann::{Scann, ScannBuilder, SearchMode};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::data_format::{Datapoint, DatapointPtr, Dataset, DenseDataset, SparseDataset};
    pub use crate::distance_measures::DistanceMeasure;
    pub use crate::brute_force::BruteForceSearcher;
    pub use crate::partitioning::{Partitioner, TreePartitioner};
    pub use crate::trees::{KMeans, KMeansTree};
    pub use crate::hashes::{AsymmetricHasher, StackedQuantizer, Lut16LookupTables};
    pub use crate::searcher::{Searcher, SearchParameters, SearchResult, NNResult};
    pub use crate::config::ScannConfig;
    pub use crate::scann::{Scann, ScannBuilder, SearchMode};
    pub use crate::error::{ScannError, Result};
    pub use crate::types::*;

    // Projections
    pub use crate::projection::{
        Projection, IdentityProjection, PcaProjection, PcaConfig,
        RandomOrthogonalProjection, RandomProjectionConfig,
        OpqProjection, OpqConfig,
        TruncateProjection, TruncateConfig,
        ChunkingProjection, ChunkingConfig,
        ProjectionType, ProjectionFactory,
    };

    // Quantization
    pub use crate::quantization::{
        QuantizationType, Quantizer, QuantizationStats,
        ScalarQuantizer, ScalarQuantizerConfig, QuantizedDataset,
        Fp8Value, Fp8Quantizer, Fp8Config,
        BFloat16Dataset, bf16_to_f32, f32_to_bf16,
    };

    // Restricts
    pub use crate::restricts::{
        RestrictFilter, NoRestrict, RestrictAllowlist, RestrictDenylist,
        CrowdingConfig, CrowdingConstraint, CrowdingMultidimensional,
    };

    // Mutator
    pub use crate::mutator::{
        Mutation, MutationType, MutationBuffer, MutableDataset,
    };

    // Tree-X-Hybrid
    pub use crate::tree_x_hybrid::{
        TreeXHybridSearcher, TreeXHybridConfig,
    };

    // Utilities
    pub use crate::utils::{
        GaussianMixture, GmmConfig,
    };
}
