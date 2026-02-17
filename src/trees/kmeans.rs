//! K-means clustering implementation.
//!
//! This module provides efficient K-means clustering for building partitions.

use crate::data_format::{Dataset, DenseDataset};
use crate::distance_measures::DistanceMeasure;
use crate::error::{Result, ScannError};
use crate::simd::squared_l2_f32;
use crate::types::{DatapointIndex, DatapointValue};
use crate::utils::parallel::maybe_parallel_map_threshold;
use crate::utils::random::RandomSampler;

/// Helper to convert to f32 without ambiguity with NumCast::to_f32
#[inline(always)]
fn val_to_f32<T: DatapointValue>(v: T) -> f32 {
    DatapointValue::to_f32(v)
}

/// Configuration for K-means clustering.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters.
    pub num_clusters: usize,

    /// Maximum number of iterations.
    pub max_iterations: usize,

    /// Convergence threshold (relative change in total distance).
    pub convergence_threshold: f64,

    /// Initialization method.
    pub init_method: KMeansInit,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,

    /// Number of random restarts.
    pub num_restarts: usize,

    /// Distance measure for clustering.
    pub distance_measure: DistanceMeasure,

    /// Minimum dimension size to use SIMD for distance computation.
    /// Set to 0 to always use SIMD, or usize::MAX to always use scalar.
    /// Default is 64 (SIMD overhead only pays off for larger vectors).
    pub simd_threshold: usize,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            num_clusters: 10,
            max_iterations: 100,
            convergence_threshold: 1e-5,
            init_method: KMeansInit::KMeansPlusPlus,
            seed: None,
            num_restarts: 1,
            distance_measure: DistanceMeasure::SquaredL2,
            simd_threshold: 128,
        }
    }
}

impl KMeansConfig {
    /// Create a new configuration with the given number of clusters.
    pub fn new(num_clusters: usize) -> Self {
        Self {
            num_clusters,
            ..Default::default()
        }
    }

    /// Set the maximum number of iterations.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set the convergence threshold.
    pub fn with_convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Set the initialization method.
    pub fn with_init_method(mut self, method: KMeansInit) -> Self {
        self.init_method = method;
        self
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the distance measure.
    pub fn with_distance_measure(mut self, measure: DistanceMeasure) -> Self {
        self.distance_measure = measure;
        self
    }

    /// Set the SIMD threshold for distance computation.
    /// SIMD is used when dimension >= threshold.
    /// - Set to 0 to always use SIMD (best for high dimensions)
    /// - Set to usize::MAX to always use scalar (best for very small dimensions)
    /// - Default is 64
    pub fn with_simd_threshold(mut self, threshold: usize) -> Self {
        self.simd_threshold = threshold;
        self
    }
}

/// Initialization method for K-means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KMeansInit {
    /// Random initialization from data points.
    Random,

    /// K-means++ initialization (better spread).
    KMeansPlusPlus,

    /// Use provided initial centers.
    Provided,
}

/// K-means clustering result.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Cluster centers.
    pub centers: Vec<Vec<f32>>,

    /// Cluster assignments for each point.
    pub assignments: Vec<usize>,

    /// Number of points in each cluster.
    pub cluster_sizes: Vec<usize>,

    /// Total within-cluster sum of squares.
    pub inertia: f64,

    /// Number of iterations performed.
    pub num_iterations: usize,

    /// Whether the algorithm converged.
    pub converged: bool,
}

/// K-means clustering algorithm.
pub struct KMeans {
    config: KMeansConfig,
}

impl KMeans {
    /// Create a new K-means instance with the given configuration.
    pub fn new(config: KMeansConfig) -> Self {
        Self { config }
    }

    /// Create a K-means instance with the given number of clusters.
    pub fn with_clusters(num_clusters: usize) -> Self {
        Self::new(KMeansConfig::new(num_clusters))
    }

    /// Fit K-means to the given dataset.
    pub fn fit<T: DatapointValue + Sync>(&self, dataset: &DenseDataset<T>) -> Result<KMeansResult> {
        if dataset.is_empty() {
            return Err(ScannError::invalid_argument("Cannot cluster empty dataset"));
        }

        let n = dataset.size();
        let k = self.config.num_clusters.min(n);

        if k == 0 {
            return Err(ScannError::invalid_argument("Number of clusters must be > 0"));
        }

        let dim = dataset.dimensionality() as usize;

        // Convert dataset to f32 for clustering
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

        let mut best_result: Option<KMeansResult> = None;
        let mut best_inertia = f64::INFINITY;

        for restart in 0..self.config.num_restarts {
            let seed = self.config.seed.map(|s| s + restart as u64);
            let result = self.fit_single(&data, k, dim, seed)?;

            if result.inertia < best_inertia {
                best_inertia = result.inertia;
                best_result = Some(result);
            }
        }

        best_result.ok_or_else(|| ScannError::internal("K-means failed to produce result"))
    }

    /// Single run of K-means.
    fn fit_single(
        &self,
        data: &[Vec<f32>],
        k: usize,
        dim: usize,
        seed: Option<u64>,
    ) -> Result<KMeansResult> {
        let n = data.len();

        // Initialize centers
        let mut centers = self.initialize_centers(data, k, dim, seed)?;
        let mut assignments = vec![0usize; n];
        let mut prev_inertia = f64::INFINITY;
        let mut num_iterations = 0;
        let mut converged = false;

        for iter in 0..self.config.max_iterations {
            num_iterations = iter + 1;

            // Assignment step
            let (new_assignments, inertia) = self.assign_clusters(data, &centers);
            assignments = new_assignments;

            // Check convergence
            let relative_change = (prev_inertia - inertia).abs() / (prev_inertia + 1e-10);
            if relative_change < self.config.convergence_threshold {
                converged = true;
                break;
            }
            prev_inertia = inertia;

            // Update step
            let (new_centers, _new_sizes) = self.update_centers(data, &assignments, k, dim);
            centers = new_centers;
        }

        // Final assignment for accurate cluster sizes
        let (final_assignments, final_inertia) = self.assign_clusters(data, &centers);

        let mut final_sizes = vec![0usize; k];
        for &a in &final_assignments {
            final_sizes[a] += 1;
        }

        Ok(KMeansResult {
            centers,
            assignments: final_assignments,
            cluster_sizes: final_sizes,
            inertia: final_inertia,
            num_iterations,
            converged,
        })
    }

    /// Initialize cluster centers.
    fn initialize_centers(
        &self,
        data: &[Vec<f32>],
        k: usize,
        dim: usize,
        seed: Option<u64>,
    ) -> Result<Vec<Vec<f32>>> {
        let n = data.len();
        let mut sampler = match seed {
            Some(s) => RandomSampler::with_seed(s),
            None => RandomSampler::new(),
        };

        match self.config.init_method {
            KMeansInit::Random => {
                let indices = sampler.sample_indices(n, k);
                Ok(indices.iter().map(|&i| data[i].clone()).collect())
            }
            KMeansInit::KMeansPlusPlus => {
                self.kmeans_plusplus_init(data, k, dim, &mut sampler)
            }
            KMeansInit::Provided => {
                Err(ScannError::invalid_argument(
                    "Provided initialization requires initial centers",
                ))
            }
        }
    }

    /// K-means++ initialization.
    fn kmeans_plusplus_init(
        &self,
        data: &[Vec<f32>],
        k: usize,
        _dim: usize,
        sampler: &mut RandomSampler,
    ) -> Result<Vec<Vec<f32>>> {
        let n = data.len();
        let mut centers = Vec::with_capacity(k);

        // Choose first center randomly
        let first_idx = sampler.sample_indices(n, 1)[0];
        centers.push(data[first_idx].clone());

        let simd_thresh = self.config.simd_threshold;

        // Distance to nearest center for each point
        let mut min_distances: Vec<f32> = data
            .iter()
            .map(|p| Self::squared_distance_with_threshold(p, &centers[0], simd_thresh))
            .collect();

        for _ in 1..k {
            // Sample proportional to squared distance
            let total: f32 = min_distances.iter().sum();
            if total == 0.0 {
                // All remaining points are duplicates of centers
                let idx = sampler.sample_indices(n, 1)[0];
                centers.push(data[idx].clone());
            } else {
                let threshold = sampler.random_f32() * total;
                let mut cumulative = 0.0f32;
                let mut selected = 0;
                for (i, &d) in min_distances.iter().enumerate() {
                    cumulative += d;
                    if cumulative >= threshold {
                        selected = i;
                        break;
                    }
                }
                centers.push(data[selected].clone());
            }

            // Update minimum distances
            let new_center = centers.last().unwrap();
            for (i, d) in min_distances.iter_mut().enumerate() {
                let new_dist = Self::squared_distance_with_threshold(&data[i], new_center, simd_thresh);
                if new_dist < *d {
                    *d = new_dist;
                }
            }
        }

        Ok(centers)
    }

    /// Assign points to nearest clusters.
    fn assign_clusters(&self, data: &[Vec<f32>], centers: &[Vec<f32>]) -> (Vec<usize>, f64) {
        // Use parallel execution only for larger datasets
        const KMEANS_PARALLEL_THRESHOLD: usize = 512;

        let simd_thresh = self.config.simd_threshold;

        let assignments: Vec<(usize, f32)> = maybe_parallel_map_threshold(
            data,
            KMEANS_PARALLEL_THRESHOLD,
            |point| {
                let mut min_dist = f32::INFINITY;
                let mut min_idx = 0;
                for (i, center) in centers.iter().enumerate() {
                    let dist = Self::squared_distance_with_threshold(point, center, simd_thresh);
                    if dist < min_dist {
                        min_dist = dist;
                        min_idx = i;
                    }
                }
                (min_idx, min_dist)
            },
        );

        let inertia: f64 = assignments.iter().map(|(_, d)| *d as f64).sum();
        let assignments: Vec<usize> = assignments.into_iter().map(|(a, _)| a).collect();

        (assignments, inertia)
    }

    /// Update cluster centers.
    fn update_centers(
        &self,
        data: &[Vec<f32>],
        assignments: &[usize],
        k: usize,
        dim: usize,
    ) -> (Vec<Vec<f32>>, Vec<usize>) {
        let mut sums = vec![vec![0.0f64; dim]; k];
        let mut counts = vec![0usize; k];

        for (i, &cluster) in assignments.iter().enumerate() {
            counts[cluster] += 1;
            for (j, &v) in data[i].iter().enumerate() {
                sums[cluster][j] += v as f64;
            }
        }

        let centers: Vec<Vec<f32>> = sums
            .iter()
            .zip(counts.iter())
            .enumerate()
            .map(|(i, (sum, &count))| {
                if count > 0 {
                    sum.iter().map(|&s| (s / count as f64) as f32).collect()
                } else {
                    // Empty cluster - reinitialize randomly
                    data[i % data.len()].clone()
                }
            })
            .collect();

        (centers, counts)
    }

    /// Squared Euclidean distance with configurable SIMD threshold.
    /// Uses SIMD when dimension >= threshold, scalar otherwise.
    #[inline(always)]
    fn squared_distance_with_threshold(a: &[f32], b: &[f32], simd_threshold: usize) -> f32 {
        if a.len() >= simd_threshold {
            squared_l2_f32(a, b)
        } else {
            // Inline scalar computation for best performance
            let mut sum = 0.0f32;
            for i in 0..a.len() {
                let d = a[i] - b[i];
                sum += d * d;
            }
            sum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_clustered_data() -> DenseDataset<f32> {
        // Create 3 clear clusters
        let mut data = Vec::new();

        // Cluster around (0, 0)
        for i in 0..10 {
            data.push(vec![0.0 + (i as f32) * 0.1, 0.0 + (i as f32) * 0.05]);
        }

        // Cluster around (10, 10)
        for i in 0..10 {
            data.push(vec![10.0 + (i as f32) * 0.1, 10.0 + (i as f32) * 0.05]);
        }

        // Cluster around (0, 10)
        for i in 0..10 {
            data.push(vec![0.0 + (i as f32) * 0.1, 10.0 + (i as f32) * 0.05]);
        }

        DenseDataset::from_vecs(data)
    }

    #[test]
    fn test_kmeans_basic() {
        let dataset = create_clustered_data();
        let kmeans = KMeans::new(KMeansConfig::new(3).with_seed(42));
        let result = kmeans.fit(&dataset).unwrap();

        assert_eq!(result.centers.len(), 3);
        assert_eq!(result.assignments.len(), 30);
        assert!(result.converged || result.num_iterations > 0);
    }

    #[test]
    fn test_kmeans_cluster_sizes() {
        let dataset = create_clustered_data();
        let kmeans = KMeans::new(KMeansConfig::new(3).with_seed(42));
        let result = kmeans.fit(&dataset).unwrap();

        let total_size: usize = result.cluster_sizes.iter().sum();
        assert_eq!(total_size, 30);
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let dataset = DenseDataset::from_vecs(vec![
            vec![1.0f32, 2.0],
            vec![1.1, 2.1],
            vec![0.9, 1.9],
        ]);
        let kmeans = KMeans::new(KMeansConfig::new(1).with_seed(42));
        let result = kmeans.fit(&dataset).unwrap();

        assert_eq!(result.centers.len(), 1);
        assert!(result.assignments.iter().all(|&a| a == 0));
    }

    #[test]
    fn test_kmeans_random_init() {
        let dataset = create_clustered_data();
        let kmeans = KMeans::new(
            KMeansConfig::new(3)
                .with_init_method(KMeansInit::Random)
                .with_seed(42)
        );
        let result = kmeans.fit(&dataset).unwrap();

        assert_eq!(result.centers.len(), 3);
    }

    #[test]
    fn test_kmeans_more_clusters_than_points() {
        let dataset = DenseDataset::from_vecs(vec![
            vec![1.0f32, 2.0],
            vec![3.0, 4.0],
        ]);
        let kmeans = KMeans::new(KMeansConfig::new(5).with_seed(42));
        let result = kmeans.fit(&dataset).unwrap();

        // Should create at most n clusters
        assert!(result.centers.len() <= 2);
    }
}
