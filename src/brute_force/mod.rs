//! Brute-force nearest neighbor search.
//!
//! This module provides exact nearest neighbor search by exhaustively
//! computing distances to all datapoints.

mod searcher;
mod top_k;
mod scalar_quantized;

pub use searcher::BruteForceSearcher;
pub use top_k::{TopK, FastTopNeighbors, FixedTopK, MAX_FIXED_K};
pub use scalar_quantized::{ScalarQuantizedBruteForceSearcher, ScalarQuantizedConfig};
