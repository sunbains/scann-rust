//! Gaussian Mixture Model utilities.
//!
//! Provides GMM training and inference for clustering and density estimation.

use crate::data_format::{Dataset, DenseDataset};
use crate::error::{Result, ScannError};
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::{Normal, Distribution};
use serde::{Deserialize, Serialize};

/// Configuration for GMM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmmConfig {
    /// Number of components.
    pub num_components: usize,
    /// Maximum number of EM iterations.
    pub max_iterations: usize,
    /// Convergence threshold.
    pub tolerance: f64,
    /// Regularization for covariance.
    pub reg_covar: f64,
    /// Random seed.
    pub seed: u64,
    /// Covariance type.
    pub covariance_type: CovarianceType,
}

/// Type of covariance matrix.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CovarianceType {
    /// Full covariance matrix.
    Full,
    /// Diagonal covariance matrix.
    Diagonal,
    /// Spherical (scalar variance).
    Spherical,
}

impl Default for GmmConfig {
    fn default() -> Self {
        Self {
            num_components: 8,
            max_iterations: 100,
            tolerance: 1e-3,
            reg_covar: 1e-6,
            seed: 42,
            covariance_type: CovarianceType::Diagonal,
        }
    }
}

impl GmmConfig {
    /// Create a new GMM config.
    pub fn new(num_components: usize) -> Self {
        Self {
            num_components,
            ..Default::default()
        }
    }

    /// Set the covariance type.
    pub fn with_covariance_type(mut self, cov_type: CovarianceType) -> Self {
        self.covariance_type = cov_type;
        self
    }

    /// Set max iterations.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// A Gaussian component.
#[derive(Clone)]
pub struct GaussianComponent {
    /// Mean vector.
    pub mean: DVector<f64>,
    /// Covariance (diagonal or full).
    pub covariance: GaussianCovariance,
    /// Mixing weight.
    pub weight: f64,
}

/// Covariance representation.
#[derive(Clone)]
pub enum GaussianCovariance {
    /// Full covariance matrix.
    Full(DMatrix<f64>),
    /// Diagonal covariance (stored as vector).
    Diagonal(DVector<f64>),
    /// Spherical (single variance value).
    Spherical(f64),
}

impl GaussianCovariance {
    /// Compute log determinant.
    pub fn log_det(&self) -> f64 {
        match self {
            GaussianCovariance::Full(cov) => {
                cov.clone().lu().determinant().abs().ln()
            }
            GaussianCovariance::Diagonal(diag) => {
                diag.iter().map(|&x| x.ln()).sum()
            }
            GaussianCovariance::Spherical(var) => {
                // For spherical, we need dimensionality
                // This is a simplified version
                var.ln()
            }
        }
    }

    /// Compute Mahalanobis distance squared.
    pub fn mahalanobis_sq(&self, x: &DVector<f64>, mean: &DVector<f64>) -> f64 {
        let diff = x - mean;

        match self {
            GaussianCovariance::Full(cov) => {
                // (x - mu)^T * cov^-1 * (x - mu)
                if let Some(cov_inv) = cov.clone().try_inverse() {
                    (&diff.transpose() * &cov_inv * &diff)[(0, 0)]
                } else {
                    f64::INFINITY
                }
            }
            GaussianCovariance::Diagonal(diag) => {
                diff.iter()
                    .zip(diag.iter())
                    .map(|(&d, &v)| d * d / v)
                    .sum()
            }
            GaussianCovariance::Spherical(var) => {
                diff.iter().map(|&d| d * d).sum::<f64>() / var
            }
        }
    }
}

/// Gaussian Mixture Model.
pub struct GaussianMixture {
    config: GmmConfig,
    /// Components.
    components: Vec<GaussianComponent>,
    /// Dimensionality.
    dimensionality: usize,
    /// Whether fitted.
    fitted: bool,
    /// Log-likelihood of the training data.
    log_likelihood: f64,
}

impl GaussianMixture {
    /// Create a new GMM.
    pub fn new(config: GmmConfig) -> Self {
        Self {
            config,
            components: Vec::new(),
            dimensionality: 0,
            fitted: false,
            log_likelihood: f64::NEG_INFINITY,
        }
    }

    /// Fit the GMM to data.
    pub fn fit(&mut self, data: &DenseDataset<f32>) -> Result<()> {
        if data.is_empty() {
            return Err(ScannError::invalid_argument("Cannot fit on empty data"));
        }

        self.dimensionality = data.dimensionality() as usize;
        let n = data.size();
        let _k = self.config.num_components;

        // Convert to f64 matrix
        let mut x = DMatrix::zeros(n, self.dimensionality);
        for i in 0..n {
            if let Some(dp) = data.get(i as u32) {
                for (j, &val) in dp.values().iter().enumerate() {
                    x[(i, j)] = val as f64;
                }
            }
        }

        // Initialize with k-means++
        self.initialize_kmeans_plusplus(&x)?;

        // EM iterations
        let mut prev_ll = f64::NEG_INFINITY;

        for _iter in 0..self.config.max_iterations {
            // E-step
            let responsibilities = self.e_step(&x);

            // M-step
            self.m_step(&x, &responsibilities);

            // Check convergence
            let ll = self.compute_log_likelihood(&x);

            if (ll - prev_ll).abs() < self.config.tolerance {
                break;
            }
            prev_ll = ll;
            self.log_likelihood = ll;
        }

        self.fitted = true;
        Ok(())
    }

    /// Initialize using k-means++ style.
    fn initialize_kmeans_plusplus(&mut self, x: &DMatrix<f64>) -> Result<()> {
        let n = x.nrows();
        let d = x.ncols();
        let k = self.config.num_components;

        let mut rng = StdRng::seed_from_u64(self.config.seed);

        // Choose first center randomly
        let mut centers = Vec::with_capacity(k);
        let first_idx = rng.gen_range(0..n);
        centers.push(x.row(first_idx).transpose());

        // Choose remaining centers
        for _ in 1..k {
            // Compute distances to nearest center
            let mut distances: Vec<f64> = (0..n)
                .map(|i| {
                    let point = x.row(i).transpose();
                    centers
                        .iter()
                        .map(|c| (&point - c).norm_squared())
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();

            // Normalize to probabilities
            let total: f64 = distances.iter().sum();
            if total > 0.0 {
                for d in &mut distances {
                    *d /= total;
                }
            }

            // Sample next center
            let r: f64 = rng.gen();
            let mut cumsum = 0.0;
            let mut next_idx = 0;
            for (i, &p) in distances.iter().enumerate() {
                cumsum += p;
                if cumsum >= r {
                    next_idx = i;
                    break;
                }
            }

            centers.push(x.row(next_idx).transpose());
        }

        // Initialize components
        self.components.clear();
        let init_weight = 1.0 / k as f64;

        for center in centers {
            let covariance = match self.config.covariance_type {
                CovarianceType::Full => {
                    GaussianCovariance::Full(DMatrix::identity(d, d))
                }
                CovarianceType::Diagonal => {
                    GaussianCovariance::Diagonal(DVector::from_element(d, 1.0))
                }
                CovarianceType::Spherical => {
                    GaussianCovariance::Spherical(1.0)
                }
            };

            self.components.push(GaussianComponent {
                mean: center,
                covariance,
                weight: init_weight,
            });
        }

        Ok(())
    }

    /// E-step: compute responsibilities.
    fn e_step(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let n = x.nrows();
        let k = self.components.len();

        let mut resp = DMatrix::zeros(n, k);

        for i in 0..n {
            let point = x.row(i).transpose();

            let mut log_probs = Vec::with_capacity(k);
            for comp in &self.components {
                let log_prob = self.log_component_probability(&point, comp);
                log_probs.push(log_prob);
            }

            // Log-sum-exp for numerical stability
            let max_log = log_probs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_log).exp()).sum();
            let log_normalizer = max_log + sum_exp.ln();

            for (j, &lp) in log_probs.iter().enumerate() {
                resp[(i, j)] = (lp - log_normalizer).exp();
            }
        }

        resp
    }

    /// M-step: update parameters.
    fn m_step(&mut self, x: &DMatrix<f64>, resp: &DMatrix<f64>) {
        let n = x.nrows() as f64;
        let d = x.ncols();

        for (k, comp) in self.components.iter_mut().enumerate() {
            let resp_k: f64 = resp.column(k).iter().sum();

            if resp_k < 1e-10 {
                continue;
            }

            // Update weight
            comp.weight = resp_k / n;

            // Update mean
            let mut new_mean = DVector::zeros(d);
            for i in 0..x.nrows() {
                let point = x.row(i).transpose();
                new_mean += resp[(i, k)] * &point;
            }
            comp.mean = new_mean / resp_k;

            // Update covariance
            match &mut comp.covariance {
                GaussianCovariance::Diagonal(diag) => {
                    let mut new_diag = DVector::zeros(d);
                    for i in 0..x.nrows() {
                        let diff = x.row(i).transpose() - &comp.mean;
                        for j in 0..d {
                            new_diag[j] += resp[(i, k)] * diff[j] * diff[j];
                        }
                    }
                    *diag = new_diag / resp_k + DVector::from_element(d, self.config.reg_covar);
                }
                GaussianCovariance::Spherical(var) => {
                    let mut total_var = 0.0;
                    for i in 0..x.nrows() {
                        let diff = x.row(i).transpose() - &comp.mean;
                        total_var += resp[(i, k)] * diff.norm_squared();
                    }
                    *var = total_var / (resp_k * d as f64) + self.config.reg_covar;
                }
                GaussianCovariance::Full(cov) => {
                    let mut new_cov = DMatrix::zeros(d, d);
                    for i in 0..x.nrows() {
                        let diff = x.row(i).transpose() - &comp.mean;
                        new_cov += resp[(i, k)] * (&diff * diff.transpose());
                    }
                    *cov = new_cov / resp_k + DMatrix::identity(d, d) * self.config.reg_covar;
                }
            }
        }
    }

    /// Compute log probability for a component.
    fn log_component_probability(&self, x: &DVector<f64>, comp: &GaussianComponent) -> f64 {
        let d = x.len() as f64;
        let mahal_sq = comp.covariance.mahalanobis_sq(x, &comp.mean);
        let log_det = match &comp.covariance {
            GaussianCovariance::Diagonal(diag) => diag.iter().map(|v| v.ln()).sum(),
            GaussianCovariance::Spherical(var) => d * var.ln(),
            GaussianCovariance::Full(cov) => cov.clone().lu().determinant().abs().ln(),
        };

        comp.weight.ln() - 0.5 * (d * (2.0 * std::f64::consts::PI).ln() + log_det + mahal_sq)
    }

    /// Compute log-likelihood.
    fn compute_log_likelihood(&self, x: &DMatrix<f64>) -> f64 {
        let mut ll = 0.0;

        for i in 0..x.nrows() {
            let point = x.row(i).transpose();

            let log_probs: Vec<f64> = self.components
                .iter()
                .map(|comp| self.log_component_probability(&point, comp))
                .collect();

            let max_log = log_probs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_log).exp()).sum();

            ll += max_log + sum_exp.ln();
        }

        ll
    }

    /// Predict component probabilities for a point.
    pub fn predict_proba(&self, point: &[f32]) -> Vec<f64> {
        if !self.fitted {
            return vec![0.0; self.config.num_components];
        }

        let x = DVector::from_iterator(point.len(), point.iter().map(|&v| v as f64));

        let log_probs: Vec<f64> = self.components
            .iter()
            .map(|comp| self.log_component_probability(&x, comp))
            .collect();

        let max_log = log_probs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_log).exp()).sum();

        log_probs
            .iter()
            .map(|&lp| (lp - max_log).exp() / sum_exp)
            .collect()
    }

    /// Predict the most likely component.
    pub fn predict(&self, point: &[f32]) -> usize {
        let probs = self.predict_proba(point);
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Sample from the mixture.
    pub fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<Vec<f64>> {
        let mut samples = Vec::with_capacity(n);

        for _ in 0..n {
            // Choose component
            let r: f64 = rng.gen();
            let mut cumsum = 0.0;
            let mut comp_idx = 0;

            for (i, comp) in self.components.iter().enumerate() {
                cumsum += comp.weight;
                if cumsum >= r {
                    comp_idx = i;
                    break;
                }
            }

            // Sample from component
            let comp = &self.components[comp_idx];
            samples.push(self.sample_from_component(comp, rng));
        }

        samples
    }

    /// Sample from a single component.
    fn sample_from_component(&self, comp: &GaussianComponent, rng: &mut impl Rng) -> Vec<f64> {
        let d = comp.mean.len();
        let normal = Normal::new(0.0, 1.0).unwrap();

        match &comp.covariance {
            GaussianCovariance::Diagonal(diag) => {
                (0..d)
                    .map(|i| comp.mean[i] + diag[i].sqrt() * normal.sample(rng))
                    .collect()
            }
            GaussianCovariance::Spherical(var) => {
                let std = var.sqrt();
                (0..d)
                    .map(|i| comp.mean[i] + std * normal.sample(rng))
                    .collect()
            }
            GaussianCovariance::Full(_cov) => {
                // Simplified: use diagonal approximation
                (0..d)
                    .map(|i| comp.mean[i] + normal.sample(rng))
                    .collect()
            }
        }
    }

    /// Get the number of components.
    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    /// Get the means.
    pub fn means(&self) -> Vec<Vec<f64>> {
        self.components.iter().map(|c| c.mean.iter().cloned().collect()).collect()
    }

    /// Get the weights.
    pub fn weights(&self) -> Vec<f64> {
        self.components.iter().map(|c| c.weight).collect()
    }

    /// Check if fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Get log-likelihood.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> DenseDataset<f32> {
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0f32, 1.0).unwrap();

        let mut data = Vec::new();

        // Cluster 1 centered at (0, 0)
        for _ in 0..50 {
            data.push(vec![
                normal.sample(&mut rng),
                normal.sample(&mut rng),
            ]);
        }

        // Cluster 2 centered at (5, 5)
        for _ in 0..50 {
            data.push(vec![
                5.0 + normal.sample(&mut rng),
                5.0 + normal.sample(&mut rng),
            ]);
        }

        DenseDataset::from_vecs(data)
    }

    #[test]
    fn test_gmm_fit() {
        let data = create_test_data();
        let config = GmmConfig::new(2)
            .with_covariance_type(CovarianceType::Diagonal)
            .with_seed(42);

        let mut gmm = GaussianMixture::new(config);
        gmm.fit(&data).unwrap();

        assert!(gmm.is_fitted());
        assert_eq!(gmm.num_components(), 2);
    }

    #[test]
    fn test_gmm_predict() {
        let data = create_test_data();
        let config = GmmConfig::new(2)
            .with_covariance_type(CovarianceType::Diagonal)
            .with_seed(42);

        let mut gmm = GaussianMixture::new(config);
        gmm.fit(&data).unwrap();

        // Point near cluster 1
        let pred1 = gmm.predict(&[0.0, 0.0]);

        // Point near cluster 2
        let pred2 = gmm.predict(&[5.0, 5.0]);

        // Should be different clusters
        assert_ne!(pred1, pred2);
    }

    #[test]
    fn test_gmm_sample() {
        let data = create_test_data();
        let config = GmmConfig::new(2);

        let mut gmm = GaussianMixture::new(config);
        gmm.fit(&data).unwrap();

        let mut rng = StdRng::seed_from_u64(123);
        let samples = gmm.sample(100, &mut rng);

        assert_eq!(samples.len(), 100);
        assert_eq!(samples[0].len(), 2);
    }
}
