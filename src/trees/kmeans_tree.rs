//! K-means tree for hierarchical partitioning.
//!
//! This module provides hierarchical K-means trees for efficient approximate search.

use crate::data_format::{Dataset, DenseDataset, DatapointPtr};
use crate::distance_measures::DistanceMeasure;
use crate::error::{Result, ScannError};
use crate::trees::kmeans::{KMeans, KMeansConfig};
use crate::types::{DatapointIndex, DatapointValue};
use serde::{Deserialize, Serialize};

/// Helper to convert to f32 without ambiguity with NumCast::to_f32
#[inline(always)]
fn val_to_f32<T: DatapointValue>(v: T) -> f32 {
    DatapointValue::to_f32(v)
}

/// Configuration for K-means tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansTreeConfig {
    /// Number of children per node (branching factor).
    pub num_children: usize,

    /// Maximum depth of the tree (1 = flat).
    pub max_depth: usize,

    /// Minimum number of points per leaf.
    pub min_leaf_size: usize,

    /// K-means configuration for each level.
    pub kmeans_max_iterations: usize,

    /// Convergence threshold.
    pub convergence_threshold: f64,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,

    /// Distance measure.
    pub distance_measure: DistanceMeasure,
}

impl Default for KMeansTreeConfig {
    fn default() -> Self {
        Self {
            num_children: 100,
            max_depth: 1,
            min_leaf_size: 1,
            kmeans_max_iterations: 100,
            convergence_threshold: 1e-5,
            seed: None,
            distance_measure: DistanceMeasure::SquaredL2,
        }
    }
}

impl KMeansTreeConfig {
    /// Create a new configuration.
    pub fn new(num_children: usize) -> Self {
        Self {
            num_children,
            ..Default::default()
        }
    }

    /// Set the maximum depth.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set the minimum leaf size.
    pub fn with_min_leaf_size(mut self, size: usize) -> Self {
        self.min_leaf_size = size;
        self
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// A node in the K-means tree.
#[derive(Debug, Clone)]
pub struct KMeansTreeNode {
    /// Center of this node (cluster centroid).
    pub center: Vec<f32>,

    /// Indices of datapoints in this node (leaf nodes only).
    pub datapoint_indices: Vec<DatapointIndex>,

    /// Child nodes (internal nodes only).
    pub children: Vec<KMeansTreeNode>,

    /// Whether this is a leaf node.
    pub is_leaf: bool,

    /// Depth in the tree (0 = root).
    pub depth: usize,
}

impl KMeansTreeNode {
    /// Create a new leaf node.
    pub fn leaf(center: Vec<f32>, indices: Vec<DatapointIndex>, depth: usize) -> Self {
        Self {
            center,
            datapoint_indices: indices,
            children: Vec::new(),
            is_leaf: true,
            depth,
        }
    }

    /// Create a new internal node.
    pub fn internal(center: Vec<f32>, children: Vec<KMeansTreeNode>, depth: usize) -> Self {
        Self {
            center,
            datapoint_indices: Vec::new(),
            children,
            is_leaf: false,
            depth,
        }
    }

    /// Get the number of datapoints in this subtree.
    pub fn size(&self) -> usize {
        if self.is_leaf {
            self.datapoint_indices.len()
        } else {
            self.children.iter().map(|c| c.size()).sum()
        }
    }

    /// Get the number of leaf nodes in this subtree.
    pub fn num_leaves(&self) -> usize {
        if self.is_leaf {
            1
        } else {
            self.children.iter().map(|c| c.num_leaves()).sum()
        }
    }

    /// Collect all datapoint indices in this subtree.
    pub fn all_indices(&self) -> Vec<DatapointIndex> {
        if self.is_leaf {
            self.datapoint_indices.clone()
        } else {
            self.children.iter().flat_map(|c| c.all_indices()).collect()
        }
    }
}

/// K-means tree for hierarchical partitioning.
pub struct KMeansTree {
    /// Root node of the tree.
    root: Option<KMeansTreeNode>,

    /// Configuration.
    config: KMeansTreeConfig,

    /// Dimensionality of datapoints.
    dimensionality: usize,
}

impl KMeansTree {
    /// Create a new K-means tree with the given configuration.
    pub fn new(config: KMeansTreeConfig) -> Self {
        Self {
            root: None,
            config,
            dimensionality: 0,
        }
    }

    /// Create a tree with the given number of children.
    pub fn with_children(num_children: usize) -> Self {
        Self::new(KMeansTreeConfig::new(num_children))
    }

    /// Build the tree from a dataset.
    pub fn build<T: DatapointValue + Sync>(&mut self, dataset: &DenseDataset<T>) -> Result<()> {
        if dataset.is_empty() {
            return Err(ScannError::invalid_argument("Cannot build tree from empty dataset"));
        }

        self.dimensionality = dataset.dimensionality() as usize;

        // Convert dataset to f32
        let n = dataset.size();
        let data: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                dataset
                    .get(i as DatapointIndex)
                    .unwrap()
                    .values()
                    .iter()
                    .map(|&v| val_to_f32(v))
                    .collect()
            })
            .collect();

        // Build tree recursively
        let indices: Vec<DatapointIndex> = (0..n as DatapointIndex).collect();
        self.root = Some(self.build_node(&data, indices, 0)?);

        Ok(())
    }

    /// Recursively build a tree node.
    fn build_node(
        &self,
        data: &[Vec<f32>],
        indices: Vec<DatapointIndex>,
        depth: usize,
    ) -> Result<KMeansTreeNode> {
        let n = indices.len();

        // Compute center
        let center = self.compute_center(data, &indices);

        // Check if we should make this a leaf
        if depth >= self.config.max_depth || n <= self.config.min_leaf_size || n <= self.config.num_children {
            return Ok(KMeansTreeNode::leaf(center, indices, depth));
        }

        // Run K-means to partition
        let subset: Vec<Vec<f32>> = indices.iter().map(|&i| data[i as usize].clone()).collect();
        let subset_dataset = DenseDataset::from_vecs(subset);

        let kmeans_config = KMeansConfig::new(self.config.num_children.min(n))
            .with_max_iterations(self.config.kmeans_max_iterations)
            .with_convergence_threshold(self.config.convergence_threshold)
            .with_seed(self.config.seed.unwrap_or(42) + depth as u64);

        let kmeans = KMeans::new(kmeans_config);
        let result = kmeans.fit(&subset_dataset)?;

        // Group indices by cluster assignment
        let mut clusters: Vec<Vec<DatapointIndex>> = vec![Vec::new(); result.centers.len()];
        for (i, &assignment) in result.assignments.iter().enumerate() {
            clusters[assignment].push(indices[i]);
        }

        // Filter out empty clusters
        let non_empty_clusters: Vec<(Vec<f32>, Vec<DatapointIndex>)> = result
            .centers
            .into_iter()
            .zip(clusters)
            .filter(|(_, indices)| !indices.is_empty())
            .collect();

        if non_empty_clusters.len() == 1 {
            // All points in one cluster - make leaf
            return Ok(KMeansTreeNode::leaf(center, indices, depth));
        }

        // Build child nodes recursively
        let mut children = Vec::with_capacity(non_empty_clusters.len());
        for (_child_center, child_indices) in non_empty_clusters {
            let child = self.build_node(data, child_indices, depth + 1)?;
            children.push(child);
        }

        Ok(KMeansTreeNode::internal(center, children, depth))
    }

    /// Compute the center of a set of points.
    fn compute_center(&self, data: &[Vec<f32>], indices: &[DatapointIndex]) -> Vec<f32> {
        if indices.is_empty() {
            return vec![0.0; self.dimensionality];
        }

        let mut sum = vec![0.0f64; self.dimensionality];
        for &idx in indices {
            for (j, &v) in data[idx as usize].iter().enumerate() {
                sum[j] += v as f64;
            }
        }

        let n = indices.len() as f64;
        sum.into_iter().map(|s| (s / n) as f32).collect()
    }

    /// Get the root node.
    pub fn root(&self) -> Option<&KMeansTreeNode> {
        self.root.as_ref()
    }

    /// Get the total number of leaves.
    pub fn num_leaves(&self) -> usize {
        self.root.as_ref().map(|r| r.num_leaves()).unwrap_or(0)
    }

    /// Get the total size (number of datapoints).
    pub fn size(&self) -> usize {
        self.root.as_ref().map(|r| r.size()).unwrap_or(0)
    }

    /// Search for the k nearest leaf nodes to a query.
    pub fn search_leaves<T: DatapointValue>(
        &self,
        query: &DatapointPtr<'_, T>,
        k: usize,
    ) -> Vec<(usize, f32, &KMeansTreeNode)> {
        let query_vec: Vec<f32> = query.values().iter().map(|&v| val_to_f32(v)).collect();
        let mut results = Vec::new();

        if let Some(root) = &self.root {
            self.search_leaves_recursive(root, &query_vec, k, &mut results);
        }

        // Sort by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Recursive leaf search.
    fn search_leaves_recursive<'a>(
        &'a self,
        node: &'a KMeansTreeNode,
        query: &[f32],
        k: usize,
        results: &mut Vec<(usize, f32, &'a KMeansTreeNode)>,
    ) {
        let dist = Self::squared_distance(query, &node.center);

        if node.is_leaf {
            results.push((node.depth, dist, node));
            return;
        }

        // Get distances to all children
        let mut child_dists: Vec<(usize, f32)> = node
            .children
            .iter()
            .enumerate()
            .map(|(i, c)| (i, Self::squared_distance(query, &c.center)))
            .collect();

        // Sort by distance
        child_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Search children in order of distance
        for (idx, _) in child_dists {
            self.search_leaves_recursive(&node.children[idx], query, k, results);

            // Early termination if we have enough leaves
            if results.len() >= k * 2 {
                break;
            }
        }
    }

    /// Squared Euclidean distance.
    #[inline]
    fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    /// Get all leaf nodes.
    pub fn leaves(&self) -> Vec<&KMeansTreeNode> {
        let mut leaves = Vec::new();
        if let Some(root) = &self.root {
            self.collect_leaves(root, &mut leaves);
        }
        leaves
    }

    /// Recursively collect leaves.
    fn collect_leaves<'a>(&'a self, node: &'a KMeansTreeNode, leaves: &mut Vec<&'a KMeansTreeNode>) {
        if node.is_leaf {
            leaves.push(node);
        } else {
            for child in &node.children {
                self.collect_leaves(child, leaves);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_format::Datapoint;

    fn create_clustered_data() -> DenseDataset<f32> {
        let mut data = Vec::new();

        // Cluster around (0, 0)
        for i in 0..20 {
            data.push(vec![0.0 + (i as f32) * 0.1, 0.0 + (i as f32) * 0.05]);
        }

        // Cluster around (10, 10)
        for i in 0..20 {
            data.push(vec![10.0 + (i as f32) * 0.1, 10.0 + (i as f32) * 0.05]);
        }

        // Cluster around (0, 10)
        for i in 0..20 {
            data.push(vec![0.0 + (i as f32) * 0.1, 10.0 + (i as f32) * 0.05]);
        }

        DenseDataset::from_vecs(data)
    }

    #[test]
    fn test_kmeans_tree_build() {
        let dataset = create_clustered_data();
        let config = KMeansTreeConfig::new(3).with_seed(42);
        let mut tree = KMeansTree::new(config);
        tree.build(&dataset).unwrap();

        assert!(tree.root().is_some());
        assert_eq!(tree.size(), 60);
        assert!(tree.num_leaves() >= 1);
    }

    #[test]
    fn test_kmeans_tree_search() {
        let dataset = create_clustered_data();
        let config = KMeansTreeConfig::new(3).with_seed(42);
        let mut tree = KMeansTree::new(config);
        tree.build(&dataset).unwrap();

        let query = Datapoint::dense(vec![0.0f32, 0.0]);
        let results = tree.search_leaves(&query.as_ptr(), 2);

        assert!(!results.is_empty());
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_kmeans_tree_multi_level() {
        let dataset = create_clustered_data();
        let config = KMeansTreeConfig::new(2)
            .with_max_depth(3)
            .with_min_leaf_size(5)
            .with_seed(42);
        let mut tree = KMeansTree::new(config);
        tree.build(&dataset).unwrap();

        assert!(tree.root().is_some());
        let leaves = tree.leaves();
        assert!(leaves.len() > 1);
    }

    #[test]
    fn test_kmeans_tree_flat() {
        let dataset = create_clustered_data();
        let config = KMeansTreeConfig::new(10).with_max_depth(1).with_seed(42);
        let mut tree = KMeansTree::new(config);
        tree.build(&dataset).unwrap();

        // With max_depth=1, should have flat structure
        let leaves = tree.leaves();
        for leaf in leaves {
            assert_eq!(leaf.depth, 1);
        }
    }
}
