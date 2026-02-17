//! Partitioned search example.
//!
//! This example demonstrates how to use tree-based partitioning
//! and asymmetric hashing for approximate nearest neighbor search.

use scann::prelude::*;
use scann::hashes::{AsymmetricHasher, AsymmetricHasherConfig};
use scann::partitioning::PartitionerConfig;
use std::time::Instant;

fn main() {
    println!("ScaNN Rust - Partitioned Search Example\n");

    // Generate a dataset
    let num_points = 20_000;
    let dim = 64;

    println!("Generating dataset: {} points, {} dimensions", num_points, dim);

    let data: Vec<Vec<f32>> = (0..num_points)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let t = (i as f32 / num_points as f32) * std::f32::consts::PI * 2.0;
                    let phase = (j as f32 / dim as f32) * std::f32::consts::PI;
                    (t + phase).sin() + ((i as f32 * j as f32) % 7.0) / 10.0
                })
                .collect()
        })
        .collect();

    let dataset = DenseDataset::from_vecs(data.clone());

    // 1. Brute-force search (baseline)
    println!("\n=== Brute-Force Search (Baseline) ===");

    let bf_searcher = BruteForceSearcher::new(dataset.clone(), DistanceMeasure::SquaredL2);

    let query: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.1).sin()).collect();
    let k = 10;

    let start = Instant::now();
    let bf_results = bf_searcher.search(&query, k).unwrap();
    let bf_time = start.elapsed();

    println!("Brute-force time: {:?}", bf_time);
    println!("Top 5 results:");
    for (rank, (idx, dist)) in bf_results.iter().take(5).enumerate() {
        println!("  {}: idx={}, dist={:.4}", rank + 1, idx, dist);
    }

    // 2. Tree-based partitioning
    println!("\n=== Tree-Based Partitioning ===");

    let num_partitions = 100;
    let partitions_to_search = 10;

    let part_config = PartitionerConfig::new(num_partitions)
        .with_partitions_to_search(partitions_to_search);

    let mut partitioner = TreePartitioner::new(part_config);

    let start = Instant::now();
    partitioner.build(&dataset).unwrap();
    let build_time = start.elapsed();

    println!("Partitioner build time: {:?}", build_time);
    println!("Number of partitions: {}", partitioner.num_partitions());

    // Query the partitioner
    let query_dp = Datapoint::dense(query.clone());
    let start = Instant::now();
    let partition_result = partitioner.partition(&query_dp.as_ptr(), partitions_to_search).unwrap();
    let partition_time = start.elapsed();

    println!("Partition query time: {:?}", partition_time);
    println!(
        "Searching {} partitions out of {}",
        partition_result.len(),
        num_partitions
    );

    // Search within selected partitions
    let mut candidates: Vec<u32> = Vec::new();
    for token in &partition_result.tokens {
        if let Some(indices) = partitioner.partition_indices(*token) {
            candidates.extend(indices);
        }
    }

    println!("Total candidates: {} / {} ({:.1}% of data)",
        candidates.len(),
        num_points,
        100.0 * candidates.len() as f64 / num_points as f64
    );

    // 3. Asymmetric Hashing
    println!("\n=== Asymmetric Hashing ===");

    let num_codes = 16;
    let num_subspaces = 8;

    let ah_config = AsymmetricHasherConfig::new(num_codes, num_subspaces).with_seed(42);
    let mut hasher = AsymmetricHasher::new(ah_config);

    let start = Instant::now();
    hasher.build(dataset.clone()).unwrap();
    let ah_build_time = start.elapsed();

    println!("AH build time: {:?}", ah_build_time);
    println!("Codes per point: {} ({}x{} subspaces)",
        num_subspaces,
        num_subspaces,
        num_codes
    );

    let start = Instant::now();
    let ah_results = hasher.search(&query, k).unwrap();
    let ah_time = start.elapsed();

    println!("AH search time: {:?}", ah_time);
    println!("Top 5 AH results:");
    for (rank, (idx, dist)) in ah_results.iter().take(5).enumerate() {
        println!("  {}: idx={}, dist={:.4}", rank + 1, idx, dist);
    }

    // Compute recall
    let bf_indices: std::collections::HashSet<_> = bf_results.iter().map(|(i, _)| *i).collect();
    let ah_indices: std::collections::HashSet<_> = ah_results.iter().map(|(i, _)| *i).collect();
    let recall = bf_indices.intersection(&ah_indices).count() as f64 / k as f64;
    println!("Recall@{}: {:.1}%", k, recall * 100.0);

    // 4. AH with reordering
    println!("\n=== AH with Exact Reordering ===");

    let pre_reorder_k = 100;
    let start = Instant::now();
    let ah_reorder_results = hasher.search_with_reordering(&query, k, pre_reorder_k).unwrap();
    let ah_reorder_time = start.elapsed();

    println!("AH+reorder time: {:?}", ah_reorder_time);

    let ah_reorder_indices: std::collections::HashSet<_> =
        ah_reorder_results.iter().map(|(i, _)| *i).collect();
    let reorder_recall = bf_indices.intersection(&ah_reorder_indices).count() as f64 / k as f64;
    println!("Recall@{} with reordering: {:.1}%", k, reorder_recall * 100.0);

    // 5. Using ScaNN builder
    println!("\n=== Using ScaNN Builder ===");

    let scann = ScannBuilder::new()
        .num_neighbors(k as u32)
        .distance_measure(DistanceMeasure::SquaredL2)
        .brute_force()
        .build(dataset)
        .unwrap();

    let start = Instant::now();
    let _scann_results = scann.search(&query, k).unwrap();
    let scann_time = start.elapsed();

    println!("ScaNN search time: {:?}", scann_time);

    // Summary
    println!("\n=== Performance Summary ===");
    println!("{:<25} {:>15} {:>15}", "Method", "Time", "Speedup");
    println!("{:-<55}", "");
    println!("{:<25} {:>15?} {:>15}", "Brute-force", bf_time, "1.0x");
    println!(
        "{:<25} {:>15?} {:>14.1}x",
        "Asymmetric Hashing",
        ah_time,
        bf_time.as_secs_f64() / ah_time.as_secs_f64()
    );
    println!(
        "{:<25} {:>15?} {:>14.1}x",
        "AH + Reordering",
        ah_reorder_time,
        bf_time.as_secs_f64() / ah_reorder_time.as_secs_f64()
    );

    println!("\nDone!");
}
