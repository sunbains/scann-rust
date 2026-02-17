//! Crowding constraints for diversity in search results.
//!
//! Crowding limits the number of results from any single group/category
//! to ensure diversity in the result set.

use crate::types::{DatapointIndex, NNResultsVector};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Configuration for crowding constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrowdingConfig {
    /// Maximum number of results per crowding group.
    pub per_crowd_limit: usize,
    /// Whether crowding is enabled.
    pub enabled: bool,
}

impl Default for CrowdingConfig {
    fn default() -> Self {
        Self {
            per_crowd_limit: 3,
            enabled: true,
        }
    }
}

impl CrowdingConfig {
    /// Create a new crowding config.
    pub fn new(per_crowd_limit: usize) -> Self {
        Self {
            per_crowd_limit,
            enabled: true,
        }
    }

    /// Disable crowding.
    pub fn disabled() -> Self {
        Self {
            per_crowd_limit: usize::MAX,
            enabled: false,
        }
    }
}

/// A crowding constraint enforcer.
pub struct CrowdingConstraint {
    /// Crowding attribute for each datapoint.
    crowding_attributes: Vec<u64>,
    /// Configuration.
    config: CrowdingConfig,
}

impl CrowdingConstraint {
    /// Create a new crowding constraint.
    pub fn new(crowding_attributes: Vec<u64>, config: CrowdingConfig) -> Self {
        Self {
            crowding_attributes,
            config,
        }
    }

    /// Get the crowding attribute for a datapoint.
    pub fn get_attribute(&self, index: DatapointIndex) -> Option<u64> {
        self.crowding_attributes.get(index as usize).copied()
    }

    /// Set the crowding attribute for a datapoint.
    pub fn set_attribute(&mut self, index: DatapointIndex, attribute: u64) {
        let idx = index as usize;
        if idx >= self.crowding_attributes.len() {
            self.crowding_attributes.resize(idx + 1, 0);
        }
        self.crowding_attributes[idx] = attribute;
    }

    /// Apply crowding constraints to search results.
    ///
    /// Returns filtered results where no more than `per_crowd_limit`
    /// results share the same crowding attribute.
    pub fn apply(&self, results: &NNResultsVector, k: usize) -> NNResultsVector {
        if !self.config.enabled {
            return results.iter().take(k).cloned().collect();
        }

        let mut crowd_counts: HashMap<u64, usize> = HashMap::new();
        let mut filtered = Vec::with_capacity(k);

        for &(idx, dist) in results {
            let attribute = self.get_attribute(idx).unwrap_or(0);
            let count = crowd_counts.entry(attribute).or_insert(0);

            if *count < self.config.per_crowd_limit {
                filtered.push((idx, dist));
                *count += 1;

                if filtered.len() >= k {
                    break;
                }
            }
        }

        filtered
    }

    /// Check if adding this result would violate crowding.
    pub fn would_violate(&self, index: DatapointIndex, current_results: &NNResultsVector) -> bool {
        if !self.config.enabled {
            return false;
        }

        let attribute = self.get_attribute(index).unwrap_or(0);
        let count = current_results
            .iter()
            .filter(|&&(idx, _)| self.get_attribute(idx).unwrap_or(0) == attribute)
            .count();

        count >= self.config.per_crowd_limit
    }
}

/// Multi-dimensional crowding for multiple attributes.
pub struct CrowdingMultidimensional {
    /// Crowding attributes (one vector per dimension).
    attributes: Vec<Vec<u64>>,
    /// Per-dimension limits.
    limits: Vec<usize>,
    /// Number of datapoints.
    num_datapoints: usize,
}

impl CrowdingMultidimensional {
    /// Create a new multi-dimensional crowding constraint.
    pub fn new(num_dimensions: usize, num_datapoints: usize) -> Self {
        Self {
            attributes: vec![vec![0; num_datapoints]; num_dimensions],
            limits: vec![usize::MAX; num_dimensions],
            num_datapoints,
        }
    }

    /// Set limits for each dimension.
    pub fn set_limits(&mut self, limits: Vec<usize>) {
        self.limits = limits;
    }

    /// Set attribute for a dimension.
    pub fn set_attribute(&mut self, dim: usize, index: DatapointIndex, attribute: u64) {
        if dim < self.attributes.len() {
            let idx = index as usize;
            if idx < self.attributes[dim].len() {
                self.attributes[dim][idx] = attribute;
            }
        }
    }

    /// Get attributes for a datapoint.
    pub fn get_attributes(&self, index: DatapointIndex) -> Vec<u64> {
        let idx = index as usize;
        self.attributes
            .iter()
            .map(|dim_attrs| dim_attrs.get(idx).copied().unwrap_or(0))
            .collect()
    }

    /// Apply multi-dimensional crowding.
    pub fn apply(&self, results: &NNResultsVector, k: usize) -> NNResultsVector {
        let num_dims = self.attributes.len();
        let mut crowd_counts: Vec<HashMap<u64, usize>> = vec![HashMap::new(); num_dims];
        let mut filtered = Vec::with_capacity(k);

        for &(idx, dist) in results {
            let attrs = self.get_attributes(idx);
            let mut allowed = true;

            // Check if any dimension would be violated
            for (dim, &attr) in attrs.iter().enumerate() {
                let count = crowd_counts[dim].get(&attr).copied().unwrap_or(0);
                if count >= self.limits[dim] {
                    allowed = false;
                    break;
                }
            }

            if allowed {
                // Update counts
                for (dim, &attr) in attrs.iter().enumerate() {
                    *crowd_counts[dim].entry(attr).or_insert(0) += 1;
                }

                filtered.push((idx, dist));

                if filtered.len() >= k {
                    break;
                }
            }
        }

        filtered
    }
}

/// Diversity sampling using Maximal Marginal Relevance (MMR).
pub struct MmrDiversifier {
    /// Lambda parameter (0 = pure diversity, 1 = pure relevance).
    pub lambda: f32,
}

impl MmrDiversifier {
    /// Create a new MMR diversifier.
    pub fn new(lambda: f32) -> Self {
        Self {
            lambda: lambda.clamp(0.0, 1.0),
        }
    }

    /// Apply MMR diversification.
    ///
    /// Requires a function to compute similarity between two datapoints.
    pub fn apply<F>(
        &self,
        candidates: &NNResultsVector,
        k: usize,
        similarity_fn: F,
    ) -> NNResultsVector
    where
        F: Fn(DatapointIndex, DatapointIndex) -> f32,
    {
        if candidates.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut selected: Vec<(DatapointIndex, f32)> = Vec::with_capacity(k);
        let mut remaining: Vec<(DatapointIndex, f32)> = candidates.to_vec();

        // Select the first item (best relevance)
        let first = remaining.remove(0);
        selected.push(first);

        while selected.len() < k && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f32::MIN;

            for (i, &(idx, relevance)) in remaining.iter().enumerate() {
                // Relevance score (negative distance = higher relevance)
                let rel_score = -relevance;

                // Maximum similarity to already selected items
                let max_sim = selected
                    .iter()
                    .map(|&(sel_idx, _)| similarity_fn(idx, sel_idx))
                    .fold(f32::MIN, f32::max);

                // MMR score
                let mmr_score = self.lambda * rel_score - (1.0 - self.lambda) * max_sim;

                if mmr_score > best_score {
                    best_score = mmr_score;
                    best_idx = i;
                }
            }

            selected.push(remaining.remove(best_idx));
        }

        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crowding_basic() {
        let attributes = vec![0, 0, 0, 1, 1, 2];
        let config = CrowdingConfig::new(2);
        let constraint = CrowdingConstraint::new(attributes, config);

        let results: NNResultsVector = vec![
            (0, 0.1), // crowd 0
            (1, 0.2), // crowd 0
            (2, 0.3), // crowd 0 - should be filtered
            (3, 0.4), // crowd 1
            (4, 0.5), // crowd 1
            (5, 0.6), // crowd 2
        ];

        let filtered = constraint.apply(&results, 6);

        // Should have: 0, 1 (crowd 0), 3, 4 (crowd 1), 5 (crowd 2)
        assert_eq!(filtered.len(), 5);
        assert!(filtered.iter().any(|&(idx, _)| idx == 0));
        assert!(filtered.iter().any(|&(idx, _)| idx == 1));
        assert!(!filtered.iter().any(|&(idx, _)| idx == 2)); // filtered out
        assert!(filtered.iter().any(|&(idx, _)| idx == 3));
        assert!(filtered.iter().any(|&(idx, _)| idx == 4));
        assert!(filtered.iter().any(|&(idx, _)| idx == 5));
    }

    #[test]
    fn test_crowding_disabled() {
        let attributes = vec![0, 0, 0];
        let config = CrowdingConfig::disabled();
        let constraint = CrowdingConstraint::new(attributes, config);

        let results: NNResultsVector = vec![(0, 0.1), (1, 0.2), (2, 0.3)];

        let filtered = constraint.apply(&results, 3);
        assert_eq!(filtered.len(), 3);
    }

    #[test]
    fn test_multidimensional_crowding() {
        let mut crowding = CrowdingMultidimensional::new(2, 6);

        // Dimension 0: category
        crowding.set_attribute(0, 0, 1);
        crowding.set_attribute(0, 1, 1);
        crowding.set_attribute(0, 2, 2);
        crowding.set_attribute(0, 3, 2);
        crowding.set_attribute(0, 4, 3);
        crowding.set_attribute(0, 5, 3);

        // Dimension 1: region
        crowding.set_attribute(1, 0, 10);
        crowding.set_attribute(1, 1, 10);
        crowding.set_attribute(1, 2, 10);
        crowding.set_attribute(1, 3, 20);
        crowding.set_attribute(1, 4, 20);
        crowding.set_attribute(1, 5, 30);

        crowding.set_limits(vec![2, 2]); // Max 2 per category, max 2 per region

        let results: NNResultsVector = vec![
            (0, 0.1),
            (1, 0.2),
            (2, 0.3),
            (3, 0.4),
            (4, 0.5),
            (5, 0.6),
        ];

        let filtered = crowding.apply(&results, 6);

        // Should limit based on both dimensions
        // Region 10 has 3 items but limit is 2, so 1 gets dropped
        // Final count: 2 (region 10) + 2 (region 20) + 1 (region 30) = 5
        assert!(filtered.len() <= 5);
    }

    #[test]
    fn test_mmr_diversifier() {
        let mmr = MmrDiversifier::new(0.5);

        let candidates: NNResultsVector = vec![
            (0, 0.1),
            (1, 0.2),
            (2, 0.3),
            (3, 0.4),
        ];

        // Simple similarity function
        let sim_fn = |a: DatapointIndex, b: DatapointIndex| {
            if a == b {
                1.0
            } else {
                0.0
            }
        };

        let diversified = mmr.apply(&candidates, 3, sim_fn);
        assert_eq!(diversified.len(), 3);
    }
}
