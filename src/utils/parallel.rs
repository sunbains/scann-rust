//! Parallel execution utilities.

use rayon::prelude::*;

/// Minimum number of items before parallelization is beneficial.
/// Below this threshold, sequential execution is faster due to reduced overhead.
pub const MIN_PARALLEL_SIZE: usize = 1024;

/// Minimum work per item (in approximate cycles) to justify parallelization.
/// For cheap operations, the threshold should be higher.
pub const MIN_PARALLEL_WORK: usize = 256;

/// Trait for parallel execution.
pub trait ParallelFor {
    /// Execute a function in parallel over a range.
    fn parallel_for<F>(&self, start: usize, end: usize, f: F)
    where
        F: Fn(usize) + Sync + Send;

    /// Execute a function in parallel over a range with batch size.
    fn parallel_for_batched<F>(&self, start: usize, end: usize, batch_size: usize, f: F)
    where
        F: Fn(usize, usize) + Sync + Send;
}

/// Thread pool wrapper using rayon.
pub struct ThreadPool {
    num_threads: usize,
}

impl ThreadPool {
    /// Create a new thread pool with the default number of threads.
    pub fn new() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
        }
    }

    /// Create a thread pool with a specific number of threads.
    pub fn with_threads(num_threads: usize) -> Self {
        Self { num_threads }
    }

    /// Get the number of threads.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

impl Default for ThreadPool {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelFor for ThreadPool {
    fn parallel_for<F>(&self, start: usize, end: usize, f: F)
    where
        F: Fn(usize) + Sync + Send,
    {
        (start..end).into_par_iter().for_each(f);
    }

    fn parallel_for_batched<F>(&self, start: usize, end: usize, batch_size: usize, f: F)
    where
        F: Fn(usize, usize) + Sync + Send,
    {
        let batches: Vec<_> = (start..end)
            .step_by(batch_size)
            .map(|s| (s, (s + batch_size).min(end)))
            .collect();

        batches.into_par_iter().for_each(|(s, e)| f(s, e));
    }
}

/// Execute a map operation in parallel.
pub fn parallel_map<T, U, F>(items: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    items.par_iter().map(f).collect()
}

/// Execute a filter-map operation in parallel.
pub fn parallel_filter_map<T, U, F>(items: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> Option<U> + Sync + Send,
{
    items.par_iter().filter_map(f).collect()
}

/// Execute a map operation, choosing parallel or sequential based on size.
/// Uses parallel execution only when the number of items exceeds MIN_PARALLEL_SIZE.
#[inline]
pub fn maybe_parallel_map<T, U, F>(items: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    if items.len() >= MIN_PARALLEL_SIZE {
        items.par_iter().map(&f).collect()
    } else {
        items.iter().map(f).collect()
    }
}

/// Execute a map operation with a custom threshold.
#[inline]
pub fn maybe_parallel_map_threshold<T, U, F>(items: &[T], threshold: usize, f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    if items.len() >= threshold {
        items.par_iter().map(&f).collect()
    } else {
        items.iter().map(f).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_parallel_for() {
        let pool = ThreadPool::new();
        let counter = AtomicUsize::new(0);

        pool.parallel_for(0, 100, |_| {
            counter.fetch_add(1, Ordering::SeqCst);
        });

        assert_eq!(counter.load(Ordering::SeqCst), 100);
    }

    #[test]
    fn test_parallel_for_batched() {
        let pool = ThreadPool::new();
        let counter = AtomicUsize::new(0);

        pool.parallel_for_batched(0, 100, 10, |start, end| {
            counter.fetch_add(end - start, Ordering::SeqCst);
        });

        assert_eq!(counter.load(Ordering::SeqCst), 100);
    }

    #[test]
    fn test_parallel_map() {
        let items: Vec<i32> = (0..100).collect();
        let results = parallel_map(&items, |x| x * 2);

        assert_eq!(results.len(), 100);
        for (i, &r) in results.iter().enumerate() {
            assert_eq!(r, (i as i32) * 2);
        }
    }
}
