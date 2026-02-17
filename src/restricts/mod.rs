//! Restricts and filtering module for ScaNN.
//!
//! This module provides mechanisms to filter search results based on:
//! - Allowlists/Denylists
//! - Crowding constraints (diversity)
//! - Custom predicates

mod allowlist;
mod crowding;

pub use allowlist::{RestrictAllowlist, RestrictDenylist, RestrictTokenMap};
pub use crowding::{CrowdingConfig, CrowdingConstraint, CrowdingMultidimensional};

use crate::types::DatapointIndex;

/// Trait for filtering datapoint indices.
pub trait RestrictFilter: Send + Sync {
    /// Check if a datapoint index is allowed.
    fn is_allowed(&self, index: DatapointIndex) -> bool;

    /// Get the number of allowed indices (if known).
    fn num_allowed(&self) -> Option<usize> {
        None
    }

    /// Check if all indices are allowed (no filtering).
    fn allows_all(&self) -> bool {
        false
    }
}

/// No restriction filter - allows all indices.
#[derive(Debug, Clone, Default)]
pub struct NoRestrict;

impl RestrictFilter for NoRestrict {
    fn is_allowed(&self, _index: DatapointIndex) -> bool {
        true
    }

    fn allows_all(&self) -> bool {
        true
    }
}

/// Predicate-based filter.
pub struct PredicateFilter<F>
where
    F: Fn(DatapointIndex) -> bool + Send + Sync,
{
    predicate: F,
}

impl<F> PredicateFilter<F>
where
    F: Fn(DatapointIndex) -> bool + Send + Sync,
{
    /// Create a new predicate filter.
    pub fn new(predicate: F) -> Self {
        Self { predicate }
    }
}

impl<F> RestrictFilter for PredicateFilter<F>
where
    F: Fn(DatapointIndex) -> bool + Send + Sync,
{
    fn is_allowed(&self, index: DatapointIndex) -> bool {
        (self.predicate)(index)
    }
}

/// Range-based filter - allows indices in a range.
#[derive(Debug, Clone)]
pub struct RangeFilter {
    start: DatapointIndex,
    end: DatapointIndex,
}

impl RangeFilter {
    /// Create a new range filter [start, end).
    pub fn new(start: DatapointIndex, end: DatapointIndex) -> Self {
        Self { start, end }
    }
}

impl RestrictFilter for RangeFilter {
    fn is_allowed(&self, index: DatapointIndex) -> bool {
        index >= self.start && index < self.end
    }

    fn num_allowed(&self) -> Option<usize> {
        Some((self.end - self.start) as usize)
    }
}

/// Combined filter that ANDs multiple filters.
pub struct AndFilter {
    filters: Vec<Box<dyn RestrictFilter>>,
}

impl AndFilter {
    /// Create a new AND filter.
    pub fn new(filters: Vec<Box<dyn RestrictFilter>>) -> Self {
        Self { filters }
    }

    /// Add a filter.
    pub fn add(&mut self, filter: Box<dyn RestrictFilter>) {
        self.filters.push(filter);
    }
}

impl RestrictFilter for AndFilter {
    fn is_allowed(&self, index: DatapointIndex) -> bool {
        self.filters.iter().all(|f| f.is_allowed(index))
    }

    fn allows_all(&self) -> bool {
        self.filters.iter().all(|f| f.allows_all())
    }
}

/// Combined filter that ORs multiple filters.
pub struct OrFilter {
    filters: Vec<Box<dyn RestrictFilter>>,
}

impl OrFilter {
    /// Create a new OR filter.
    pub fn new(filters: Vec<Box<dyn RestrictFilter>>) -> Self {
        Self { filters }
    }

    /// Add a filter.
    pub fn add(&mut self, filter: Box<dyn RestrictFilter>) {
        self.filters.push(filter);
    }
}

impl RestrictFilter for OrFilter {
    fn is_allowed(&self, index: DatapointIndex) -> bool {
        self.filters.iter().any(|f| f.is_allowed(index))
    }

    fn allows_all(&self) -> bool {
        self.filters.iter().any(|f| f.allows_all())
    }
}

/// NOT filter - inverts another filter.
pub struct NotFilter {
    inner: Box<dyn RestrictFilter>,
}

impl NotFilter {
    /// Create a new NOT filter.
    pub fn new(inner: Box<dyn RestrictFilter>) -> Self {
        Self { inner }
    }
}

impl RestrictFilter for NotFilter {
    fn is_allowed(&self, index: DatapointIndex) -> bool {
        !self.inner.is_allowed(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_restrict() {
        let filter = NoRestrict;
        assert!(filter.is_allowed(0));
        assert!(filter.is_allowed(1000));
        assert!(filter.allows_all());
    }

    #[test]
    fn test_range_filter() {
        let filter = RangeFilter::new(10, 20);
        assert!(!filter.is_allowed(5));
        assert!(filter.is_allowed(10));
        assert!(filter.is_allowed(15));
        assert!(!filter.is_allowed(20));
        assert_eq!(filter.num_allowed(), Some(10));
    }

    #[test]
    fn test_predicate_filter() {
        let filter = PredicateFilter::new(|idx| idx % 2 == 0);
        assert!(filter.is_allowed(0));
        assert!(!filter.is_allowed(1));
        assert!(filter.is_allowed(100));
    }

    #[test]
    fn test_and_filter() {
        let f1 = Box::new(RangeFilter::new(0, 100));
        let f2 = Box::new(PredicateFilter::new(|idx| idx % 2 == 0));
        let filter = AndFilter::new(vec![f1, f2]);

        assert!(filter.is_allowed(0));
        assert!(!filter.is_allowed(1));
        assert!(!filter.is_allowed(101));
    }

    #[test]
    fn test_or_filter() {
        let f1 = Box::new(RangeFilter::new(0, 10));
        let f2 = Box::new(RangeFilter::new(90, 100));
        let filter = OrFilter::new(vec![f1, f2]);

        assert!(filter.is_allowed(5));
        assert!(!filter.is_allowed(50));
        assert!(filter.is_allowed(95));
    }
}
