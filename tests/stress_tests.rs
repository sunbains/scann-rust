//! Stress tests for ScaNN Rust library.
//!
//! These tests verify correctness and performance under heavy load.

use scann::prelude::*;
use rand::prelude::*;
use std::time::Instant;

/// Generate random dataset.
fn generate_random_dataset(n: usize, dim: usize, seed: u64) -> DenseDataset<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let data: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();
    DenseDataset::from_vecs(data)
}

/// Generate random queries.
fn generate_random_queries(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

/// Verify that brute-force results are correct.
fn verify_results(results: &[(u32, f32)]) {
    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(
            results[i].1 >= results[i - 1].1,
            "Results not sorted: {} >= {} failed",
            results[i].1,
            results[i - 1].1
        );
    }
}

#[test]
fn stress_test_brute_force_small() {
    const N: usize = 1000;
    const DIM: usize = 64;
    const K: usize = 10;
    const NUM_QUERIES: usize = 100;

    let dataset = generate_random_dataset(N, DIM, 42);
    let queries = generate_random_queries(NUM_QUERIES, DIM, 123);

    let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

    for query in &queries {
        let results = searcher.search(query, K).unwrap();
        assert_eq!(results.len(), K);
        verify_results(&results);
    }
}

#[test]
fn stress_test_brute_force_medium() {
    const N: usize = 10_000;
    const DIM: usize = 128;
    const K: usize = 100;
    const NUM_QUERIES: usize = 50;

    let dataset = generate_random_dataset(N, DIM, 42);
    let queries = generate_random_queries(NUM_QUERIES, DIM, 123);

    let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

    let start = Instant::now();
    for query in &queries {
        let results = searcher.search(query, K).unwrap();
        assert_eq!(results.len(), K);
        verify_results(&results);
    }
    let elapsed = start.elapsed();

    println!(
        "Brute-force medium: {} queries in {:?} ({:.2} QPS)",
        NUM_QUERIES,
        elapsed,
        NUM_QUERIES as f64 / elapsed.as_secs_f64()
    );
}

#[test]
fn stress_test_brute_force_batched() {
    const N: usize = 5000;
    const DIM: usize = 64;
    const K: usize = 10;
    const NUM_QUERIES: usize = 100;

    let dataset = generate_random_dataset(N, DIM, 42);
    let queries = generate_random_queries(NUM_QUERIES, DIM, 123);

    let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

    let start = Instant::now();
    let results = searcher.search_batched(&queries, K).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(results.len(), NUM_QUERIES);
    for r in &results {
        assert_eq!(r.len(), K);
        verify_results(r);
    }

    println!(
        "Batched brute-force: {} queries in {:?} ({:.2} QPS)",
        NUM_QUERIES,
        elapsed,
        NUM_QUERIES as f64 / elapsed.as_secs_f64()
    );
}

#[test]
fn stress_test_kmeans_clustering() {
    const N: usize = 5000;
    const DIM: usize = 32;
    const K: usize = 50;

    let dataset = generate_random_dataset(N, DIM, 42);

    let config = scann::trees::KMeansConfig::new(K)
        .with_max_iterations(50)
        .with_seed(42);

    let kmeans = KMeans::new(config);

    let start = Instant::now();
    let result = kmeans.fit(&dataset).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(result.centers.len(), K);
    assert_eq!(result.assignments.len(), N);

    let total_points: usize = result.cluster_sizes.iter().sum();
    assert_eq!(total_points, N);

    println!(
        "K-means clustering: {} points into {} clusters in {:?}",
        N, K, elapsed
    );
}

#[test]
fn stress_test_partitioning() {
    const N: usize = 10_000;
    const DIM: usize = 64;
    const NUM_PARTITIONS: usize = 100;
    const NUM_QUERIES: usize = 100;
    const PARTITIONS_TO_SEARCH: usize = 10;

    let dataset = generate_random_dataset(N, DIM, 42);
    let queries = generate_random_queries(NUM_QUERIES, DIM, 123);

    let config = scann::partitioning::PartitionerConfig::new(NUM_PARTITIONS)
        .with_partitions_to_search(PARTITIONS_TO_SEARCH);

    let mut partitioner = TreePartitioner::new(config);

    let start = Instant::now();
    partitioner.build(&dataset).unwrap();
    let build_time = start.elapsed();

    println!("Partitioner build: {:?}", build_time);

    assert_eq!(partitioner.num_partitions(), NUM_PARTITIONS);
    assert_eq!(partitioner.num_datapoints(), N);

    let start = Instant::now();
    for query in &queries {
        let query_dp = Datapoint::dense(query.clone());
        let result = partitioner.partition(&query_dp.as_ptr(), PARTITIONS_TO_SEARCH).unwrap();
        assert_eq!(result.len(), PARTITIONS_TO_SEARCH);
    }
    let query_time = start.elapsed();

    println!(
        "Partitioning: {} queries in {:?} ({:.2} QPS)",
        NUM_QUERIES,
        query_time,
        NUM_QUERIES as f64 / query_time.as_secs_f64()
    );
}

#[test]
fn stress_test_asymmetric_hashing() {
    const N: usize = 5000;
    const DIM: usize = 64;
    const NUM_CODES: usize = 16;
    const NUM_SUBSPACES: usize = 8;
    const K: usize = 10;
    const NUM_QUERIES: usize = 100;

    let dataset = generate_random_dataset(N, DIM, 42);
    let queries = generate_random_queries(NUM_QUERIES, DIM, 123);

    let config = scann::hashes::AsymmetricHasherConfig::new(NUM_CODES, NUM_SUBSPACES)
        .with_seed(42);

    let mut hasher = scann::hashes::AsymmetricHasher::new(config);

    let start = Instant::now();
    hasher.build(dataset).unwrap();
    let build_time = start.elapsed();

    println!("Asymmetric hasher build: {:?}", build_time);

    assert_eq!(hasher.num_datapoints(), N);

    let start = Instant::now();
    for query in &queries {
        let results = hasher.search(query, K).unwrap();
        assert_eq!(results.len(), K);
    }
    let query_time = start.elapsed();

    println!(
        "AH search: {} queries in {:?} ({:.2} QPS)",
        NUM_QUERIES,
        query_time,
        NUM_QUERIES as f64 / query_time.as_secs_f64()
    );
}

#[test]
fn stress_test_scann_brute_force() {
    const N: usize = 10_000;
    const DIM: usize = 64;
    const K: usize = 10;
    const NUM_QUERIES: usize = 100;

    let dataset = generate_random_dataset(N, DIM, 42);
    let queries = generate_random_queries(NUM_QUERIES, DIM, 123);

    let scann = Scann::brute_force(dataset).unwrap();

    let start = Instant::now();
    let results = scann.search_batched(&queries, K).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(results.len(), NUM_QUERIES);
    for r in &results {
        assert_eq!(r.len(), K);
    }

    println!(
        "ScaNN brute-force: {} queries in {:?} ({:.2} QPS)",
        NUM_QUERIES,
        elapsed,
        NUM_QUERIES as f64 / elapsed.as_secs_f64()
    );
}

#[test]
fn stress_test_concurrent_queries() {
    use std::thread;
    use std::sync::Arc;

    const N: usize = 5000;
    const DIM: usize = 32;
    const K: usize = 10;
    const NUM_THREADS: usize = 4;
    const QUERIES_PER_THREAD: usize = 50;

    let dataset = generate_random_dataset(N, DIM, 42);
    let scann = Arc::new(Scann::brute_force(dataset).unwrap());

    let start = Instant::now();
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|t| {
            let scann = Arc::clone(&scann);
            thread::spawn(move || {
                let queries = generate_random_queries(QUERIES_PER_THREAD, DIM, 100 + t as u64);
                for query in &queries {
                    let results = scann.search(query, K).unwrap();
                    assert_eq!(results.len(), K);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
    let elapsed = start.elapsed();

    let total_queries = NUM_THREADS * QUERIES_PER_THREAD;
    println!(
        "Concurrent queries: {} queries across {} threads in {:?} ({:.2} QPS)",
        total_queries,
        NUM_THREADS,
        elapsed,
        total_queries as f64 / elapsed.as_secs_f64()
    );
}

#[test]
fn stress_test_high_dimensional() {
    const N: usize = 1000;
    const DIM: usize = 512;
    const K: usize = 10;
    const NUM_QUERIES: usize = 20;

    let dataset = generate_random_dataset(N, DIM, 42);
    let queries = generate_random_queries(NUM_QUERIES, DIM, 123);

    let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

    let start = Instant::now();
    for query in &queries {
        let results = searcher.search(query, K).unwrap();
        assert_eq!(results.len(), K);
        verify_results(&results);
    }
    let elapsed = start.elapsed();

    println!(
        "High-dimensional ({} dims): {} queries in {:?}",
        DIM, NUM_QUERIES, elapsed
    );
}

#[test]
fn stress_test_recall_verification() {
    // Verify that brute-force returns the actual nearest neighbors
    const N: usize = 1000;
    const DIM: usize = 32;
    const K: usize = 10;

    let dataset = generate_random_dataset(N, DIM, 42);
    let query = generate_random_queries(1, DIM, 123).pop().unwrap();

    let searcher = BruteForceSearcher::new(dataset.clone(), DistanceMeasure::SquaredL2);
    let results = searcher.search(&query, K).unwrap();

    // Manually compute distances and verify
    let query_dp = Datapoint::dense(query);
    let mut all_distances: Vec<(u32, f32)> = (0..N as u32)
        .map(|i| {
            let dp = dataset.get(i).unwrap();
            let dist = DistanceMeasure::SquaredL2.distance(&query_dp.as_ptr(), &dp);
            (i, dist)
        })
        .collect();

    all_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Verify top-K matches
    for i in 0..K {
        assert_eq!(
            results[i].0, all_distances[i].0,
            "Index mismatch at position {}: got {}, expected {}",
            i, results[i].0, all_distances[i].0
        );
        assert!(
            (results[i].1 - all_distances[i].1).abs() < 1e-5,
            "Distance mismatch at position {}: got {}, expected {}",
            i, results[i].1, all_distances[i].1
        );
    }
}

#[test]
fn stress_test_dataset_operations() {
    const N: usize = 10_000;
    const DIM: usize = 64;

    let mut dataset = DenseDataset::<f32>::with_capacity(N, DIM as u64);

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let start = Instant::now();
    for _ in 0..N {
        let values: Vec<f32> = (0..DIM).map(|_| rng.gen::<f32>()).collect();
        let dp = Datapoint::dense(values);
        dataset.append(&dp.as_ptr()).unwrap();
    }
    let elapsed = start.elapsed();

    assert_eq!(dataset.size(), N);

    println!(
        "Dataset append: {} points in {:?} ({:.2} points/sec)",
        N,
        elapsed,
        N as f64 / elapsed.as_secs_f64()
    );
}

#[test]
fn stress_test_multiple_distance_measures() {
    const N: usize = 1000;
    const DIM: usize = 64;
    const K: usize = 10;

    let dataset = generate_random_dataset(N, DIM, 42);
    let query = generate_random_queries(1, DIM, 123).pop().unwrap();

    let measures = [
        DistanceMeasure::SquaredL2,
        DistanceMeasure::L2,
        DistanceMeasure::L1,
        DistanceMeasure::Cosine,
        DistanceMeasure::DotProduct,
    ];

    for measure in &measures {
        let searcher = BruteForceSearcher::new(dataset.clone(), *measure);
        let results = searcher.search(&query, K).unwrap();

        assert_eq!(results.len(), K);
        verify_results(&results);

        println!("{:?}: top result distance = {:.4}", measure, results[0].1);
    }
}
