//! Batched search example.
//!
//! This example demonstrates how to perform efficient batched searches
//! for multiple queries simultaneously.

use scann::prelude::*;
use std::time::Instant;

fn main() {
    println!("ScaNN Rust - Batched Search Example\n");

    // Generate a larger dataset
    let num_points = 10_000;
    let dim = 64;

    println!("Generating dataset: {} points, {} dimensions", num_points, dim);

    let mut data = Vec::with_capacity(num_points);
    for i in 0..num_points {
        let point: Vec<f32> = (0..dim)
            .map(|j| (i as f32 * 0.01 + j as f32 * 0.1).sin())
            .collect();
        data.push(point);
    }

    let dataset = DenseDataset::from_vecs(data);

    // Create the searcher
    let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

    // Generate batch of queries
    let num_queries = 100;
    println!("Generating {} queries", num_queries);

    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|i| {
            (0..dim)
                .map(|j| (i as f32 * 0.05 + j as f32 * 0.2).cos())
                .collect()
        })
        .collect();

    let k = 10;

    // Sequential search
    println!("\n--- Sequential Search ---");
    let start = Instant::now();
    let mut sequential_results = Vec::with_capacity(num_queries);
    for query in &queries {
        sequential_results.push(searcher.search(query, k).unwrap());
    }
    let sequential_time = start.elapsed();
    println!(
        "Sequential: {} queries in {:?} ({:.2} QPS)",
        num_queries,
        sequential_time,
        num_queries as f64 / sequential_time.as_secs_f64()
    );

    // Batched search
    println!("\n--- Batched Search ---");
    let start = Instant::now();
    let batched_results = searcher.search_batched(&queries, k).unwrap();
    let batched_time = start.elapsed();
    println!(
        "Batched: {} queries in {:?} ({:.2} QPS)",
        num_queries,
        batched_time,
        num_queries as f64 / batched_time.as_secs_f64()
    );

    // Verify results match
    println!("\n--- Verification ---");
    let mut matches = 0;
    for (seq, batch) in sequential_results.iter().zip(batched_results.iter()) {
        if seq.len() == batch.len() && seq.iter().zip(batch.iter()).all(|(a, b)| a.0 == b.0) {
            matches += 1;
        }
    }
    println!("Results match: {}/{}", matches, num_queries);

    // Show sample results
    println!("\n--- Sample Results (Query 0) ---");
    println!("{:>5} {:>10} {:>15}", "Rank", "Index", "Distance");
    println!("{:-<32}", "");
    for (rank, (index, distance)) in batched_results[0].iter().take(5).enumerate() {
        println!("{:>5} {:>10} {:>15.6}", rank + 1, index, distance);
    }

    // Using ScaNN interface
    println!("\n--- Using ScaNN Interface ---");

    let data: Vec<Vec<f32>> = (0..num_points)
        .map(|i| {
            (0..dim)
                .map(|j| (i as f32 * 0.01 + j as f32 * 0.1).sin())
                .collect()
        })
        .collect();
    let dataset = DenseDataset::from_vecs(data);

    let scann = Scann::brute_force(dataset).unwrap();

    let start = Instant::now();
    let _results = scann.search_batched(&queries, k).unwrap();
    let scann_time = start.elapsed();

    println!(
        "ScaNN batched: {} queries in {:?} ({:.2} QPS)",
        num_queries,
        scann_time,
        num_queries as f64 / scann_time.as_secs_f64()
    );

    // Summary
    println!("\n--- Summary ---");
    println!("Dataset size: {} points x {} dimensions", num_points, dim);
    println!("Number of queries: {}", num_queries);
    println!("Top-k: {}", k);
    println!(
        "Speedup (batched vs sequential): {:.2}x",
        sequential_time.as_secs_f64() / batched_time.as_secs_f64()
    );

    println!("\nDone!");
}
