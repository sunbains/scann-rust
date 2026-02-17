//! Data format types for ScaNN.
//!
//! This module provides the core data structures for representing datapoints
//! and datasets, equivalent to the C++ data_format module.

mod datapoint;
mod dataset;
mod docid;

pub use datapoint::{Datapoint, DatapointPtr};
pub use dataset::{Dataset, DenseDataset, SparseDataset};
pub use docid::{DocId, DocIdCollection};
