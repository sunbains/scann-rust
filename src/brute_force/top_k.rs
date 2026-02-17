//! Top-K selection utilities.
//!
//! This module provides efficient data structures for tracking the top-k
//! nearest neighbors during search.

use crate::types::{DatapointIndex, NNResultPair};
use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;

/// Maximum k for using the fixed-size array implementation.
/// Above this threshold, the heap-based implementation is used.
pub const MAX_FIXED_K: usize = 32;

/// A max-heap based top-k tracker.
///
/// Maintains the k smallest distances seen so far using a max-heap.
/// This allows efficient pruning: if the current distance is >= the
/// largest distance in the heap, we can skip that point.
#[derive(Debug)]
pub struct TopK {
    /// Max-heap of (distance, index) pairs.
    /// We use negative distances for max-heap to get min-k behavior.
    heap: BinaryHeap<(OrderedFloat<f32>, DatapointIndex)>,

    /// Maximum capacity.
    k: usize,
}

impl TopK {
    /// Create a new top-k tracker.
    pub fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k + 1),
            k,
        }
    }

    /// Get the current size.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Get the capacity (k).
    pub fn capacity(&self) -> usize {
        self.k
    }

    /// Get the current threshold distance.
    /// If we have k elements, this is the largest distance in the heap.
    /// Otherwise, returns infinity.
    pub fn threshold(&self) -> f32 {
        if self.heap.len() >= self.k {
            self.heap.peek().map(|(d, _)| d.0).unwrap_or(f32::INFINITY)
        } else {
            f32::INFINITY
        }
    }

    /// Try to push a new element.
    /// Returns true if the element was added (distance < threshold).
    pub fn push(&mut self, index: DatapointIndex, distance: f32) -> bool {
        if self.heap.len() < self.k {
            self.heap.push((OrderedFloat(distance), index));
            true
        } else if let Some(&(max_dist, _)) = self.heap.peek() {
            if distance < max_dist.0 {
                self.heap.pop();
                self.heap.push((OrderedFloat(distance), index));
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Check if a distance would be accepted.
    #[inline]
    pub fn would_accept(&self, distance: f32) -> bool {
        self.heap.len() < self.k || distance < self.threshold()
    }

    /// Clear the tracker.
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// Get results sorted by distance (ascending).
    pub fn results(&self) -> Vec<NNResultPair> {
        let mut results: Vec<_> = self.heap
            .iter()
            .map(|(d, idx)| (*idx, d.0))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Drain results sorted by distance (ascending).
    pub fn drain_sorted(&mut self) -> Vec<NNResultPair> {
        let mut results: Vec<_> = self.heap
            .drain()
            .map(|(d, idx)| (idx, d.0))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

/// Fixed-size top-k tracker optimized for small k.
///
/// Uses a simple array-based max-heap without OrderedFloat wrapper.
/// More efficient than BinaryHeap for k <= MAX_FIXED_K.
#[derive(Debug)]
pub struct FixedTopK<const K: usize> {
    /// Distances (max-heap property: largest at index 0).
    distances: [f32; K],
    /// Corresponding indices.
    indices: [DatapointIndex; K],
    /// Current number of elements.
    size: usize,
}

impl<const K: usize> FixedTopK<K> {
    /// Create a new fixed-size top-k tracker.
    #[inline]
    pub fn new() -> Self {
        Self {
            distances: [f32::INFINITY; K],
            indices: [0; K],
            size: 0,
        }
    }

    /// Get the current size.
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get the capacity (k).
    #[inline]
    pub fn capacity(&self) -> usize {
        K
    }

    /// Get the current threshold distance.
    #[inline]
    pub fn threshold(&self) -> f32 {
        if self.size >= K {
            self.distances[0] // Max element at root
        } else {
            f32::INFINITY
        }
    }

    /// Push a new element, maintaining max-heap property.
    #[inline]
    pub fn push(&mut self, index: DatapointIndex, distance: f32) -> bool {
        if self.size < K {
            // Not full yet, just add and sift up
            let pos = self.size;
            self.distances[pos] = distance;
            self.indices[pos] = index;
            self.size += 1;
            self.sift_up(pos);
            true
        } else if distance < self.distances[0] {
            // Replace the max element
            self.distances[0] = distance;
            self.indices[0] = index;
            self.sift_down(0);
            true
        } else {
            false
        }
    }

    /// Sift up to maintain max-heap property.
    #[inline]
    fn sift_up(&mut self, mut pos: usize) {
        while pos > 0 {
            let parent = (pos - 1) / 2;
            if self.distances[pos] > self.distances[parent] {
                self.distances.swap(pos, parent);
                self.indices.swap(pos, parent);
                pos = parent;
            } else {
                break;
            }
        }
    }

    /// Sift down to maintain max-heap property.
    #[inline]
    fn sift_down(&mut self, mut pos: usize) {
        loop {
            let left = 2 * pos + 1;
            let right = 2 * pos + 2;
            let mut largest = pos;

            if left < self.size && self.distances[left] > self.distances[largest] {
                largest = left;
            }
            if right < self.size && self.distances[right] > self.distances[largest] {
                largest = right;
            }

            if largest != pos {
                self.distances.swap(pos, largest);
                self.indices.swap(pos, largest);
                pos = largest;
            } else {
                break;
            }
        }
    }

    /// Check if a distance would be accepted.
    #[inline]
    pub fn would_accept(&self, distance: f32) -> bool {
        self.size < K || distance < self.distances[0]
    }

    /// Clear the tracker.
    #[inline]
    pub fn clear(&mut self) {
        self.size = 0;
        self.distances = [f32::INFINITY; K];
    }

    /// Get results sorted by distance (ascending).
    pub fn results(&self) -> Vec<NNResultPair> {
        let mut results: Vec<_> = (0..self.size)
            .map(|i| (self.indices[i], self.distances[i]))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

impl<const K: usize> Default for FixedTopK<K> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fast top-k neighbors tracker with epsilon bounds.
///
/// This is a more optimized version that supports epsilon-approximate
/// search and batch operations.
#[derive(Debug)]
pub struct FastTopNeighbors {
    /// Indices of top neighbors.
    indices: Vec<DatapointIndex>,

    /// Distances to top neighbors.
    distances: Vec<f32>,

    /// Current size.
    size: usize,

    /// Maximum capacity.
    capacity: usize,

    /// Epsilon for approximate search (0 = exact).
    epsilon: f32,
}

impl FastTopNeighbors {
    /// Create a new tracker with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            indices: vec![0; capacity],
            distances: vec![f32::INFINITY; capacity],
            size: 0,
            capacity,
            epsilon: 0.0,
        }
    }

    /// Create a new tracker with epsilon-approximate bounds.
    pub fn with_epsilon(capacity: usize, epsilon: f32) -> Self {
        Self {
            indices: vec![0; capacity],
            distances: vec![f32::INFINITY; capacity],
            size: 0,
            capacity,
            epsilon,
        }
    }

    /// Get the current size.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get the capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the current threshold distance.
    pub fn threshold(&self) -> f32 {
        if self.size >= self.capacity {
            let max_dist = self.distances[..self.size]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            max_dist * (1.0 + self.epsilon)
        } else {
            f32::INFINITY
        }
    }

    /// Push a new candidate.
    pub fn push(&mut self, index: DatapointIndex, distance: f32) {
        if self.size < self.capacity {
            // Not yet full, just append
            self.indices[self.size] = index;
            self.distances[self.size] = distance;
            self.size += 1;
        } else {
            // Find the max distance and replace if better
            let mut max_idx = 0;
            let mut max_dist = self.distances[0];
            for i in 1..self.size {
                if self.distances[i] > max_dist {
                    max_dist = self.distances[i];
                    max_idx = i;
                }
            }

            if distance < max_dist {
                self.indices[max_idx] = index;
                self.distances[max_idx] = distance;
            }
        }
    }

    /// Push a batch of candidates.
    pub fn push_batch(&mut self, indices: &[DatapointIndex], distances: &[f32]) {
        debug_assert_eq!(indices.len(), distances.len());
        for (&idx, &dist) in indices.iter().zip(distances.iter()) {
            if dist < self.threshold() {
                self.push(idx, dist);
            }
        }
    }

    /// Clear the tracker.
    pub fn clear(&mut self) {
        self.size = 0;
        self.distances.fill(f32::INFINITY);
    }

    /// Get results sorted by distance.
    pub fn results(&self) -> Vec<NNResultPair> {
        let mut results: Vec<_> = self.indices[..self.size]
            .iter()
            .zip(self.distances[..self.size].iter())
            .map(|(&idx, &dist)| (idx, dist))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get the indices slice.
    pub fn indices(&self) -> &[DatapointIndex] {
        &self.indices[..self.size]
    }

    /// Get the distances slice.
    pub fn distances(&self) -> &[f32] {
        &self.distances[..self.size]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_basic() {
        let mut top_k = TopK::new(3);

        assert!(top_k.push(0, 5.0));
        assert!(top_k.push(1, 3.0));
        assert!(top_k.push(2, 7.0));

        assert_eq!(top_k.len(), 3);
        assert!((top_k.threshold() - 7.0).abs() < 1e-6);

        // This should replace 7.0
        assert!(top_k.push(3, 4.0));
        assert!((top_k.threshold() - 5.0).abs() < 1e-6);

        // This should not be added
        assert!(!top_k.push(4, 6.0));

        let results = top_k.results();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 1); // distance 3.0
        assert_eq!(results[1].0, 3); // distance 4.0
        assert_eq!(results[2].0, 0); // distance 5.0
    }

    #[test]
    fn test_top_k_empty() {
        let top_k = TopK::new(5);
        assert!(top_k.is_empty());
        assert_eq!(top_k.threshold(), f32::INFINITY);
    }

    #[test]
    fn test_fast_top_neighbors() {
        let mut ftn = FastTopNeighbors::new(3);

        ftn.push(0, 5.0);
        ftn.push(1, 3.0);
        ftn.push(2, 7.0);

        assert_eq!(ftn.len(), 3);

        // Push a better candidate
        ftn.push(3, 2.0);
        assert_eq!(ftn.len(), 3);

        let results = ftn.results();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 3); // distance 2.0
        assert_eq!(results[1].0, 1); // distance 3.0
    }

    #[test]
    fn test_fast_top_neighbors_batch() {
        let mut ftn = FastTopNeighbors::new(3);

        let indices = vec![0, 1, 2, 3, 4];
        let distances = vec![5.0, 3.0, 7.0, 1.0, 4.0];

        ftn.push_batch(&indices, &distances);

        let results = ftn.results();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 3); // distance 1.0
        assert_eq!(results[1].0, 1); // distance 3.0
        assert_eq!(results[2].0, 4); // distance 4.0
    }

    #[test]
    fn test_fixed_top_k_basic() {
        let mut top_k: FixedTopK<3> = FixedTopK::new();

        assert!(top_k.push(0, 5.0));
        assert!(top_k.push(1, 3.0));
        assert!(top_k.push(2, 7.0));

        assert_eq!(top_k.len(), 3);
        assert!((top_k.threshold() - 7.0).abs() < 1e-6);

        // This should replace 7.0
        assert!(top_k.push(3, 4.0));
        assert!((top_k.threshold() - 5.0).abs() < 1e-6);

        // This should not be added
        assert!(!top_k.push(4, 6.0));

        let results = top_k.results();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 1); // distance 3.0
        assert_eq!(results[1].0, 3); // distance 4.0
        assert_eq!(results[2].0, 0); // distance 5.0
    }

    #[test]
    fn test_fixed_top_k_stress() {
        let mut top_k: FixedTopK<10> = FixedTopK::new();

        // Push 100 random-ish values
        for i in 0..100 {
            let dist = ((i * 7) % 100) as f32;
            top_k.push(i, dist);
        }

        assert_eq!(top_k.len(), 10);

        let results = top_k.results();
        assert_eq!(results.len(), 10);

        // Verify sorted order
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }

        // Verify we have the 10 smallest
        let max_dist = results.last().unwrap().1;
        assert!(max_dist < 10.0); // Smallest 10 from 0..100 step 7 should be < 10
    }
}
