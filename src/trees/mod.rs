//! Tree structures for partitioning.
//!
//! This module provides K-means trees for hierarchical partitioning of the dataset.

pub(crate) mod kmeans;
pub(crate) mod kmeans_tree;

pub use kmeans::{KMeans, KMeansConfig, KMeansResult};
pub use kmeans_tree::{KMeansTree, KMeansTreeNode, KMeansTreeConfig};
