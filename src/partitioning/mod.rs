//! Partitioning for approximate nearest neighbor search.
//!
//! This module provides partitioning strategies for reducing the search space.

mod partitioner;
mod tree_partitioner;

pub use partitioner::{Partitioner, PartitionerConfig, PartitionResult};
pub use tree_partitioner::TreePartitioner;
