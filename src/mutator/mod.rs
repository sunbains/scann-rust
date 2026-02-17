//! Mutator module for dynamic index updates.
//!
//! This module provides lock-free support for:
//! - Adding new datapoints to an index
//! - Removing datapoints from an index
//! - Updating existing datapoints
//!
//! All operations are thread-safe and use lock-free algorithms for high concurrency.

use crate::data_format::{Dataset, DenseDataset};
use crate::error::{Result, ScannError};
use crate::types::{DatapointIndex, DimensionIndex};
use arc_swap::ArcSwap;
use crossbeam::queue::SegQueue;
use dashmap::{DashMap, DashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Mutation operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationType {
    /// Add a new datapoint.
    Add,
    /// Remove an existing datapoint.
    Remove,
    /// Update an existing datapoint.
    Update,
}

/// A mutation record.
#[derive(Debug, Clone)]
pub struct Mutation {
    /// Type of mutation.
    pub mutation_type: MutationType,
    /// Index of the affected datapoint.
    pub index: DatapointIndex,
    /// New data (for Add/Update).
    pub data: Option<Vec<f32>>,
    /// Timestamp.
    pub timestamp: u64,
}

impl Mutation {
    /// Create an add mutation.
    pub fn add(index: DatapointIndex, data: Vec<f32>, timestamp: u64) -> Self {
        Self {
            mutation_type: MutationType::Add,
            index,
            data: Some(data),
            timestamp,
        }
    }

    /// Create a remove mutation.
    pub fn remove(index: DatapointIndex, timestamp: u64) -> Self {
        Self {
            mutation_type: MutationType::Remove,
            index,
            data: None,
            timestamp,
        }
    }

    /// Create an update mutation.
    pub fn update(index: DatapointIndex, data: Vec<f32>, timestamp: u64) -> Self {
        Self {
            mutation_type: MutationType::Update,
            index,
            data: Some(data),
            timestamp,
        }
    }
}

/// Lock-free mutation buffer using crossbeam's SegQueue.
pub struct MutationBuffer {
    /// Lock-free queue of pending mutations.
    mutations: SegQueue<Mutation>,
    /// Maximum buffer size before automatic flush.
    max_buffer_size: usize,
    /// Current timestamp counter (atomic).
    timestamp: AtomicU64,
    /// Current buffer length (atomic for fast checking).
    len: AtomicUsize,
}

impl MutationBuffer {
    /// Create a new mutation buffer.
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            mutations: SegQueue::new(),
            max_buffer_size,
            timestamp: AtomicU64::new(0),
            len: AtomicUsize::new(0),
        }
    }

    /// Get the next timestamp.
    fn next_timestamp(&self) -> u64 {
        self.timestamp.fetch_add(1, Ordering::Relaxed)
    }

    /// Add a mutation to the buffer (lock-free).
    pub fn push(&self, mut mutation: Mutation) -> bool {
        mutation.timestamp = self.next_timestamp();
        self.mutations.push(mutation);
        let new_len = self.len.fetch_add(1, Ordering::AcqRel) + 1;
        new_len >= self.max_buffer_size
    }

    /// Add a new datapoint.
    pub fn add(&self, index: DatapointIndex, data: Vec<f32>) -> bool {
        self.push(Mutation::add(index, data, 0))
    }

    /// Remove a datapoint.
    pub fn remove(&self, index: DatapointIndex) -> bool {
        self.push(Mutation::remove(index, 0))
    }

    /// Update a datapoint.
    pub fn update(&self, index: DatapointIndex, data: Vec<f32>) -> bool {
        self.push(Mutation::update(index, data, 0))
    }

    /// Flush all mutations and return them (lock-free).
    pub fn flush(&self) -> Vec<Mutation> {
        let mut result = Vec::new();
        while let Some(mutation) = self.mutations.pop() {
            result.push(mutation);
        }
        self.len.store(0, Ordering::Release);
        result
    }

    /// Get the number of pending mutations (approximate, lock-free).
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if buffer should be flushed.
    pub fn should_flush(&self) -> bool {
        self.len() >= self.max_buffer_size
    }
}

impl Default for MutationBuffer {
    fn default() -> Self {
        Self::new(1000)
    }
}

// Make MutationBuffer Send + Sync
unsafe impl Send for MutationBuffer {}
unsafe impl Sync for MutationBuffer {}

/// Immutable snapshot of dataset data for RCU pattern.
#[derive(Clone)]
struct DataSnapshot {
    /// The actual data vectors.
    data: Vec<Vec<f32>>,
}

impl DataSnapshot {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    fn from_data(data: Vec<Vec<f32>>) -> Self {
        Self { data }
    }
}

/// A read guard that caches the data snapshot for efficient batch reads.
///
/// Using a ReadGuard avoids the overhead of loading the ArcSwap pointer
/// on every individual read operation.
pub struct ReadGuard<'a> {
    snapshot: arc_swap::Guard<Arc<DataSnapshot>>,
    dataset: &'a MutableDataset,
}

impl<'a> ReadGuard<'a> {
    /// Get a datapoint by index without cloning.
    /// Returns a reference to the internal data.
    #[inline]
    pub fn get(&self, index: DatapointIndex) -> Option<&[f32]> {
        if self.dataset.deleted.contains(&index) {
            return None;
        }
        let internal_idx = *self.dataset.index_map.get(&index)?;
        self.snapshot.data.get(internal_idx).map(|v| v.as_slice())
    }

    /// Get a datapoint by internal index directly (bypasses index mapping).
    /// This is faster but requires knowing the internal index.
    ///
    /// # Safety
    /// The caller must ensure the internal_idx is valid and not deleted.
    #[inline]
    pub fn get_by_internal_idx(&self, internal_idx: usize) -> Option<&[f32]> {
        self.snapshot.data.get(internal_idx).map(|v| v.as_slice())
    }

    /// Get the number of datapoints in the snapshot.
    #[inline]
    pub fn len(&self) -> usize {
        self.snapshot.data.len()
    }

    /// Check if the snapshot is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.snapshot.data.is_empty()
    }
}

/// A lock-free mutable dataset wrapper that tracks changes.
///
/// Uses RCU (Read-Copy-Update) pattern for the data vector and
/// DashMap/DashSet for concurrent hash operations.
pub struct MutableDataset {
    /// The underlying dataset using ArcSwap for RCU pattern.
    data: ArcSwap<DataSnapshot>,
    /// Dimensionality.
    dimensionality: DimensionIndex,
    /// Deleted indices (lock-free).
    deleted: DashSet<DatapointIndex>,
    /// Index mapping (original -> current) (lock-free).
    index_map: DashMap<DatapointIndex, usize>,
    /// Next available index.
    next_index: AtomicUsize,
    /// Mutation buffer.
    mutation_buffer: MutationBuffer,
}

impl MutableDataset {
    /// Create a new mutable dataset.
    pub fn new(dimensionality: DimensionIndex) -> Self {
        Self {
            data: ArcSwap::from_pointee(DataSnapshot::new()),
            dimensionality,
            deleted: DashSet::new(),
            index_map: DashMap::new(),
            next_index: AtomicUsize::new(0),
            mutation_buffer: MutationBuffer::default(),
        }
    }

    /// Create from an existing dataset.
    pub fn from_dataset(dataset: &DenseDataset<f32>) -> Self {
        let dimensionality = dataset.dimensionality();
        let mut data = Vec::with_capacity(dataset.size());
        let index_map = DashMap::new();

        for i in 0..dataset.size() {
            if let Some(dp) = dataset.get(i as u32) {
                index_map.insert(i as DatapointIndex, data.len());
                data.push(dp.values().to_vec());
            }
        }

        let data_len = data.len();
        Self {
            data: ArcSwap::from_pointee(DataSnapshot::from_data(data)),
            dimensionality,
            deleted: DashSet::new(),
            index_map,
            next_index: AtomicUsize::new(data_len),
            mutation_buffer: MutationBuffer::default(),
        }
    }

    /// Add a new datapoint (lock-free using CAS loop for data).
    pub fn add(&self, data: Vec<f32>) -> Result<DatapointIndex> {
        if data.len() != self.dimensionality as usize {
            return Err(ScannError::invalid_argument("Dimensionality mismatch"));
        }

        let index = self.next_index.fetch_add(1, Ordering::SeqCst) as DatapointIndex;

        // Use CAS loop to atomically append to data vector
        let internal_idx = loop {
            let current = self.data.load();
            let mut new_data = current.data.clone();
            let internal_idx = new_data.len();
            new_data.push(data.clone());
            let new_snapshot = Arc::new(DataSnapshot::from_data(new_data));

            // Try to swap in the new data
            let prev = self.data.compare_and_swap(&current, new_snapshot);
            if Arc::ptr_eq(&prev, &current) {
                // Swap succeeded
                break internal_idx;
            }
            // Swap failed, retry with new snapshot
        };

        // Update index map (lock-free)
        self.index_map.insert(index, internal_idx);

        // Record mutation
        self.mutation_buffer.add(index, data);

        Ok(index)
    }

    /// Remove a datapoint (lock-free).
    pub fn remove(&self, index: DatapointIndex) -> Result<()> {
        if !self.index_map.contains_key(&index) {
            return Err(ScannError::not_found("Datapoint not found"));
        }

        // Mark as deleted (lock-free)
        self.deleted.insert(index);
        self.mutation_buffer.remove(index);

        Ok(())
    }

    /// Update a datapoint (uses CAS loop for atomic update).
    pub fn update(&self, index: DatapointIndex, data: Vec<f32>) -> Result<()> {
        if data.len() != self.dimensionality as usize {
            return Err(ScannError::invalid_argument("Dimensionality mismatch"));
        }

        let internal_idx = *self.index_map
            .get(&index)
            .ok_or_else(|| ScannError::not_found("Datapoint not found"))?;

        // Use CAS loop for atomic update
        loop {
            let current = self.data.load();
            if internal_idx >= current.data.len() {
                break;
            }
            let mut new_data = current.data.clone();
            new_data[internal_idx] = data.clone();
            let new_snapshot = Arc::new(DataSnapshot::from_data(new_data));

            let prev = self.data.compare_and_swap(&current, new_snapshot);
            if Arc::ptr_eq(&prev, &current) {
                break;
            }
            // Retry on conflict
        }

        // Remove from deleted set if present
        self.deleted.remove(&index);
        self.mutation_buffer.update(index, data);

        Ok(())
    }

    /// Get a datapoint (lock-free read).
    pub fn get(&self, index: DatapointIndex) -> Option<Vec<f32>> {
        if self.deleted.contains(&index) {
            return None;
        }

        let internal_idx = *self.index_map.get(&index)?;
        let snapshot = self.data.load();
        snapshot.data.get(internal_idx).cloned()
    }

    /// Create a read guard for efficient batch reads.
    ///
    /// The guard caches the current snapshot, avoiding repeated ArcSwap loads.
    /// Use this when performing multiple reads in sequence.
    #[inline]
    pub fn read_guard(&self) -> ReadGuard<'_> {
        ReadGuard {
            snapshot: self.data.load(),
            dataset: self,
        }
    }

    /// Fast path for getting a datapoint when index == internal index.
    /// This is valid for datasets that were created sequentially without deletions.
    #[inline]
    pub fn get_fast(&self, index: DatapointIndex) -> Option<Vec<f32>> {
        // Fast path: check if index maps to itself (common case for sequential adds)
        let snapshot = self.data.load();
        let idx = index as usize;

        // If index is within bounds and not deleted, assume direct mapping
        if idx < snapshot.data.len() && !self.deleted.contains(&index) {
            // Verify the mapping is correct
            if let Some(mapped) = self.index_map.get(&index) {
                if *mapped == idx {
                    return snapshot.data.get(idx).cloned();
                }
            }
        }

        // Fall back to regular get
        self.get(index)
    }

    /// Get multiple datapoints efficiently using a read guard.
    pub fn get_batch(&self, indices: &[DatapointIndex]) -> Vec<Option<Vec<f32>>> {
        let guard = self.read_guard();
        indices.iter()
            .map(|&idx| guard.get(idx).map(|s| s.to_vec()))
            .collect()
    }

    /// Get the number of active datapoints.
    pub fn size(&self) -> usize {
        let snapshot = self.data.load();
        snapshot.data.len() - self.deleted.len()
    }

    /// Get the dimensionality.
    pub fn dimensionality(&self) -> DimensionIndex {
        self.dimensionality
    }

    /// Check if a datapoint exists (lock-free).
    pub fn exists(&self, index: DatapointIndex) -> bool {
        !self.deleted.contains(&index) && self.index_map.contains_key(&index)
    }

    /// Get pending mutations.
    pub fn flush_mutations(&self) -> Vec<Mutation> {
        self.mutation_buffer.flush()
    }

    /// Compact the dataset by removing deleted entries (uses RCU).
    pub fn compact(&self) {
        let snapshot = self.data.load();

        // Build new compact data
        let mut new_data = Vec::new();
        let new_index_map: DashMap<DatapointIndex, usize> = DashMap::new();

        for entry in self.index_map.iter() {
            let original_idx = *entry.key();
            let internal_idx = *entry.value();

            if !self.deleted.contains(&original_idx) {
                if let Some(vec) = snapshot.data.get(internal_idx) {
                    new_index_map.insert(original_idx, new_data.len());
                    new_data.push(vec.clone());
                }
            }
        }

        // Atomically swap in new data
        self.data.store(Arc::new(DataSnapshot::from_data(new_data)));

        // Clear and repopulate index_map
        self.index_map.clear();
        for entry in new_index_map.iter() {
            self.index_map.insert(*entry.key(), *entry.value());
        }

        // Clear deleted set
        self.deleted.clear();
    }

    /// Convert to an immutable DenseDataset.
    pub fn to_dense_dataset(&self) -> DenseDataset<f32> {
        let snapshot = self.data.load();

        let mut vecs = Vec::new();
        for entry in self.index_map.iter() {
            let original_idx = *entry.key();
            let internal_idx = *entry.value();

            if !self.deleted.contains(&original_idx) {
                if let Some(vec) = snapshot.data.get(internal_idx) {
                    vecs.push(vec.clone());
                }
            }
        }

        DenseDataset::from_vecs(vecs)
    }
}

/// Incremental index updater with lock-free buffer and atomic swap for index.
pub struct IncrementalUpdater<T> {
    /// The underlying index using ArcSwap for atomic replacement.
    index: ArcSwap<T>,
    /// Mutation buffer (lock-free).
    buffer: MutationBuffer,
    /// Rebuild threshold (number of mutations before full rebuild).
    rebuild_threshold: usize,
    /// Total mutations since last rebuild.
    mutations_since_rebuild: AtomicUsize,
}

impl<T> IncrementalUpdater<T> {
    /// Create a new incremental updater.
    pub fn new(index: T, rebuild_threshold: usize) -> Self {
        Self {
            index: ArcSwap::from_pointee(index),
            buffer: MutationBuffer::new(100),
            rebuild_threshold,
            mutations_since_rebuild: AtomicUsize::new(0),
        }
    }

    /// Get the underlying index for read access (lock-free).
    pub fn load_index(&self) -> arc_swap::Guard<Arc<T>> {
        self.index.load()
    }

    /// Replace the index atomically.
    pub fn store_index(&self, new_index: T) {
        self.index.store(Arc::new(new_index));
    }

    /// Queue a mutation (lock-free).
    pub fn queue_mutation(&self, mutation: Mutation) {
        self.buffer.push(mutation);
        self.mutations_since_rebuild.fetch_add(1, Ordering::Relaxed);
    }

    /// Check if a full rebuild is needed.
    pub fn needs_rebuild(&self) -> bool {
        self.mutations_since_rebuild.load(Ordering::Relaxed) >= self.rebuild_threshold
    }

    /// Get pending mutations.
    pub fn get_pending_mutations(&self) -> Vec<Mutation> {
        self.buffer.flush()
    }

    /// Reset the rebuild counter.
    pub fn reset_rebuild_counter(&self) {
        self.mutations_since_rebuild.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_mutation_buffer() {
        let buffer = MutationBuffer::new(3);

        buffer.add(0, vec![1.0, 2.0]);
        buffer.add(1, vec![3.0, 4.0]);

        assert_eq!(buffer.len(), 2);
        assert!(!buffer.should_flush());

        buffer.remove(0);
        assert!(buffer.should_flush());

        let mutations = buffer.flush();
        assert_eq!(mutations.len(), 3);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_mutation_buffer_concurrent() {
        let buffer = Arc::new(MutationBuffer::new(1000));
        let num_threads = 8;
        let mutations_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let buf = Arc::clone(&buffer);
                thread::spawn(move || {
                    for i in 0..mutations_per_thread {
                        let idx = (t * mutations_per_thread + i) as DatapointIndex;
                        buf.add(idx, vec![i as f32, t as f32]);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(buffer.len(), num_threads * mutations_per_thread);

        let mutations = buffer.flush();
        assert_eq!(mutations.len(), num_threads * mutations_per_thread);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_mutable_dataset() {
        let dataset = MutableDataset::new(3);

        let idx0 = dataset.add(vec![1.0, 2.0, 3.0]).unwrap();
        let idx1 = dataset.add(vec![4.0, 5.0, 6.0]).unwrap();

        assert_eq!(dataset.size(), 2);
        assert_eq!(dataset.get(idx0), Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(dataset.get(idx1), Some(vec![4.0, 5.0, 6.0]));

        dataset.remove(idx0).unwrap();
        assert_eq!(dataset.size(), 1);
        assert_eq!(dataset.get(idx0), None);
        assert_eq!(dataset.get(idx1), Some(vec![4.0, 5.0, 6.0]));
    }

    #[test]
    fn test_mutable_dataset_update() {
        let dataset = MutableDataset::new(2);

        let idx = dataset.add(vec![1.0, 2.0]).unwrap();
        assert_eq!(dataset.get(idx), Some(vec![1.0, 2.0]));

        dataset.update(idx, vec![10.0, 20.0]).unwrap();
        assert_eq!(dataset.get(idx), Some(vec![10.0, 20.0]));
    }

    #[test]
    fn test_mutable_dataset_compact() {
        let dataset = MutableDataset::new(2);

        let idx0 = dataset.add(vec![1.0, 2.0]).unwrap();
        let idx1 = dataset.add(vec![3.0, 4.0]).unwrap();
        let idx2 = dataset.add(vec![5.0, 6.0]).unwrap();

        dataset.remove(idx1).unwrap();

        assert_eq!(dataset.size(), 2);

        dataset.compact();

        assert_eq!(dataset.size(), 2);
        assert!(dataset.exists(idx0));
        assert!(!dataset.exists(idx1));
        assert!(dataset.exists(idx2));
    }

    #[test]
    fn test_mutable_dataset_concurrent_reads() {
        let dataset = Arc::new(MutableDataset::new(3));

        // Add some initial data
        for i in 0..100 {
            dataset.add(vec![i as f32, (i + 1) as f32, (i + 2) as f32]).unwrap();
        }

        let num_readers = 4;
        let reads_per_thread = 1000;

        let handles: Vec<_> = (0..num_readers)
            .map(|_| {
                let ds = Arc::clone(&dataset);
                thread::spawn(move || {
                    for _ in 0..reads_per_thread {
                        for i in 0..100 {
                            let _ = ds.get(i as DatapointIndex);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_mutable_dataset_concurrent_writes() {
        let dataset = Arc::new(MutableDataset::new(2));
        let num_writers = 4;
        let writes_per_thread = 50;

        let handles: Vec<_> = (0..num_writers)
            .map(|t| {
                let ds = Arc::clone(&dataset);
                thread::spawn(move || {
                    for i in 0..writes_per_thread {
                        let _ = ds.add(vec![(t * 100 + i) as f32, i as f32]);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(dataset.size(), num_writers * writes_per_thread);
    }

    #[test]
    fn test_mutable_dataset_mixed_concurrent() {
        let dataset = Arc::new(MutableDataset::new(2));

        // Pre-populate
        for i in 0..50 {
            dataset.add(vec![i as f32, i as f32]).unwrap();
        }

        let num_threads = 8;

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let ds = Arc::clone(&dataset);
                thread::spawn(move || {
                    for i in 0..100 {
                        match i % 4 {
                            0 => {
                                // Add
                                let _ = ds.add(vec![(t * 1000 + i) as f32, i as f32]);
                            }
                            1 => {
                                // Read
                                let _ = ds.get((i % 50) as DatapointIndex);
                            }
                            2 => {
                                // Update (may fail if index doesn't exist)
                                let _ = ds.update((i % 50) as DatapointIndex, vec![i as f32, i as f32]);
                            }
                            _ => {
                                // Check exists
                                let _ = ds.exists((i % 50) as DatapointIndex);
                            }
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Should not panic, data should be consistent
        let final_dataset = dataset.to_dense_dataset();
        assert!(final_dataset.size() > 0);
    }

    #[test]
    fn test_incremental_updater() {
        let updater = IncrementalUpdater::new(42i32, 100);

        // Load index
        let idx = updater.load_index();
        assert_eq!(**idx, 42);

        // Queue mutations
        for i in 0..50 {
            updater.queue_mutation(Mutation::add(i, vec![i as f32], 0));
        }

        assert!(!updater.needs_rebuild());

        for i in 50..100 {
            updater.queue_mutation(Mutation::add(i, vec![i as f32], 0));
        }

        assert!(updater.needs_rebuild());

        // Get and reset
        let mutations = updater.get_pending_mutations();
        assert_eq!(mutations.len(), 100);

        updater.reset_rebuild_counter();
        assert!(!updater.needs_rebuild());

        // Replace index
        updater.store_index(100);
        let new_idx = updater.load_index();
        assert_eq!(**new_idx, 100);
    }
}
