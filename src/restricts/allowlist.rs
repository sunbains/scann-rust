//! Allowlist and denylist filtering.

use crate::restricts::RestrictFilter;
use crate::types::DatapointIndex;
use bitvec::prelude::*;
use std::collections::{HashMap, HashSet};

/// A filter that allows only specified indices.
#[derive(Clone)]
pub struct RestrictAllowlist {
    /// Bitmap for efficient membership testing.
    bitmap: BitVec,
    /// Number of allowed indices.
    count: usize,
}

impl RestrictAllowlist {
    /// Create an empty allowlist with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            bitmap: bitvec![0; capacity],
            count: 0,
        }
    }

    /// Create an allowlist from a set of indices.
    pub fn from_indices(indices: &[DatapointIndex], capacity: usize) -> Self {
        let mut bitmap = bitvec![0; capacity];
        let mut count = 0;

        for &idx in indices {
            let i = idx as usize;
            if i < capacity && !bitmap[i] {
                bitmap.set(i, true);
                count += 1;
            }
        }

        Self { bitmap, count }
    }

    /// Create an allowlist from a HashSet.
    pub fn from_set(set: &HashSet<DatapointIndex>, capacity: usize) -> Self {
        let mut bitmap = bitvec![0; capacity];
        let mut count = 0;

        for &idx in set {
            let i = idx as usize;
            if i < capacity {
                bitmap.set(i, true);
                count += 1;
            }
        }

        Self { bitmap, count }
    }

    /// Add an index to the allowlist.
    pub fn add(&mut self, index: DatapointIndex) {
        let i = index as usize;
        if i < self.bitmap.len() && !self.bitmap[i] {
            self.bitmap.set(i, true);
            self.count += 1;
        }
    }

    /// Remove an index from the allowlist.
    pub fn remove(&mut self, index: DatapointIndex) {
        let i = index as usize;
        if i < self.bitmap.len() && self.bitmap[i] {
            self.bitmap.set(i, false);
            self.count -= 1;
        }
    }

    /// Get all allowed indices.
    pub fn indices(&self) -> Vec<DatapointIndex> {
        self.bitmap
            .iter_ones()
            .map(|i| i as DatapointIndex)
            .collect()
    }

    /// Clear the allowlist.
    pub fn clear(&mut self) {
        self.bitmap.fill(false);
        self.count = 0;
    }

    /// Get the capacity.
    pub fn capacity(&self) -> usize {
        self.bitmap.len()
    }
}

impl RestrictFilter for RestrictAllowlist {
    fn is_allowed(&self, index: DatapointIndex) -> bool {
        let i = index as usize;
        i < self.bitmap.len() && self.bitmap[i]
    }

    fn num_allowed(&self) -> Option<usize> {
        Some(self.count)
    }
}

/// A filter that denies specified indices.
#[derive(Clone)]
pub struct RestrictDenylist {
    /// Bitmap for denied indices.
    bitmap: BitVec,
    /// Number of denied indices.
    denied_count: usize,
    /// Total capacity.
    capacity: usize,
}

impl RestrictDenylist {
    /// Create an empty denylist.
    pub fn new(capacity: usize) -> Self {
        Self {
            bitmap: bitvec![0; capacity],
            denied_count: 0,
            capacity,
        }
    }

    /// Create a denylist from indices.
    pub fn from_indices(indices: &[DatapointIndex], capacity: usize) -> Self {
        let mut bitmap = bitvec![0; capacity];
        let mut denied_count = 0;

        for &idx in indices {
            let i = idx as usize;
            if i < capacity && !bitmap[i] {
                bitmap.set(i, true);
                denied_count += 1;
            }
        }

        Self {
            bitmap,
            denied_count,
            capacity,
        }
    }

    /// Add an index to the denylist.
    pub fn deny(&mut self, index: DatapointIndex) {
        let i = index as usize;
        if i < self.bitmap.len() && !self.bitmap[i] {
            self.bitmap.set(i, true);
            self.denied_count += 1;
        }
    }

    /// Remove an index from the denylist (allow it again).
    pub fn allow(&mut self, index: DatapointIndex) {
        let i = index as usize;
        if i < self.bitmap.len() && self.bitmap[i] {
            self.bitmap.set(i, false);
            self.denied_count -= 1;
        }
    }

    /// Clear the denylist.
    pub fn clear(&mut self) {
        self.bitmap.fill(false);
        self.denied_count = 0;
    }
}

impl RestrictFilter for RestrictDenylist {
    fn is_allowed(&self, index: DatapointIndex) -> bool {
        let i = index as usize;
        i < self.bitmap.len() && !self.bitmap[i]
    }

    fn num_allowed(&self) -> Option<usize> {
        Some(self.capacity - self.denied_count)
    }
}

/// Token-based filtering for multi-valued attributes.
///
/// Each datapoint can have multiple tokens, and the filter allows
/// datapoints that have at least one of the query tokens.
#[derive(Clone)]
pub struct RestrictTokenMap {
    /// Map from token to list of datapoint indices.
    token_to_indices: HashMap<u64, Vec<DatapointIndex>>,
    /// Total number of datapoints.
    num_datapoints: usize,
}

impl RestrictTokenMap {
    /// Create a new token map.
    pub fn new(num_datapoints: usize) -> Self {
        Self {
            token_to_indices: HashMap::new(),
            num_datapoints,
        }
    }

    /// Add a token for a datapoint.
    pub fn add_token(&mut self, index: DatapointIndex, token: u64) {
        self.token_to_indices
            .entry(token)
            .or_default()
            .push(index);
    }

    /// Set tokens for a datapoint.
    pub fn set_tokens(&mut self, index: DatapointIndex, tokens: &[u64]) {
        for &token in tokens {
            self.add_token(index, token);
        }
    }

    /// Get indices for a token.
    pub fn get_indices(&self, token: u64) -> Option<&[DatapointIndex]> {
        self.token_to_indices.get(&token).map(|v| v.as_slice())
    }

    /// Create an allowlist for the given tokens.
    pub fn create_allowlist(&self, tokens: &[u64]) -> RestrictAllowlist {
        let mut allowlist = RestrictAllowlist::new(self.num_datapoints);

        for &token in tokens {
            if let Some(indices) = self.get_indices(token) {
                for &idx in indices {
                    allowlist.add(idx);
                }
            }
        }

        allowlist
    }

    /// Get the number of unique tokens.
    pub fn num_tokens(&self) -> usize {
        self.token_to_indices.len()
    }
}

/// Sparse allowlist using a HashSet (for when only a few indices are allowed).
#[derive(Clone)]
pub struct SparseAllowlist {
    allowed: HashSet<DatapointIndex>,
}

impl SparseAllowlist {
    /// Create an empty sparse allowlist.
    pub fn new() -> Self {
        Self {
            allowed: HashSet::new(),
        }
    }

    /// Create from indices.
    pub fn from_indices(indices: impl IntoIterator<Item = DatapointIndex>) -> Self {
        Self {
            allowed: indices.into_iter().collect(),
        }
    }

    /// Add an index.
    pub fn add(&mut self, index: DatapointIndex) {
        self.allowed.insert(index);
    }

    /// Remove an index.
    pub fn remove(&mut self, index: DatapointIndex) {
        self.allowed.remove(&index);
    }

    /// Get all indices.
    pub fn indices(&self) -> impl Iterator<Item = DatapointIndex> + '_ {
        self.allowed.iter().copied()
    }
}

impl Default for SparseAllowlist {
    fn default() -> Self {
        Self::new()
    }
}

impl RestrictFilter for SparseAllowlist {
    fn is_allowed(&self, index: DatapointIndex) -> bool {
        self.allowed.contains(&index)
    }

    fn num_allowed(&self) -> Option<usize> {
        Some(self.allowed.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allowlist() {
        let allowlist = RestrictAllowlist::from_indices(&[1, 3, 5, 7], 10);

        assert!(!allowlist.is_allowed(0));
        assert!(allowlist.is_allowed(1));
        assert!(!allowlist.is_allowed(2));
        assert!(allowlist.is_allowed(3));
        assert_eq!(allowlist.num_allowed(), Some(4));
    }

    #[test]
    fn test_denylist() {
        let denylist = RestrictDenylist::from_indices(&[0, 2, 4], 10);

        assert!(!denylist.is_allowed(0));
        assert!(denylist.is_allowed(1));
        assert!(!denylist.is_allowed(2));
        assert!(denylist.is_allowed(3));
        assert_eq!(denylist.num_allowed(), Some(7));
    }

    #[test]
    fn test_allowlist_modify() {
        let mut allowlist = RestrictAllowlist::new(10);

        allowlist.add(5);
        assert!(allowlist.is_allowed(5));
        assert!(!allowlist.is_allowed(6));

        allowlist.add(6);
        assert!(allowlist.is_allowed(6));

        allowlist.remove(5);
        assert!(!allowlist.is_allowed(5));
    }

    #[test]
    fn test_token_map() {
        let mut token_map = RestrictTokenMap::new(10);

        token_map.set_tokens(0, &[100, 200]);
        token_map.set_tokens(1, &[100, 300]);
        token_map.set_tokens(2, &[200, 300]);

        let allowlist = token_map.create_allowlist(&[100]);
        assert!(allowlist.is_allowed(0));
        assert!(allowlist.is_allowed(1));
        assert!(!allowlist.is_allowed(2));

        let allowlist2 = token_map.create_allowlist(&[200, 300]);
        assert!(allowlist2.is_allowed(0));
        assert!(allowlist2.is_allowed(1));
        assert!(allowlist2.is_allowed(2));
    }

    #[test]
    fn test_sparse_allowlist() {
        let mut allowlist = SparseAllowlist::new();
        allowlist.add(1000);
        allowlist.add(5000);
        allowlist.add(9999);

        assert!(!allowlist.is_allowed(0));
        assert!(allowlist.is_allowed(1000));
        assert!(allowlist.is_allowed(5000));
        assert!(allowlist.is_allowed(9999));
        assert_eq!(allowlist.num_allowed(), Some(3));
    }
}
