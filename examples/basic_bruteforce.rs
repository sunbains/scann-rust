//! Basic brute-force search example.
//!
//! This example demonstrates how to use ScaNN for exact nearest neighbor search
//! using brute-force scanning.

use scann::prelude::*;

fn main() {
    println!("ScaNN Rust - Basic Brute-Force Example\n");

    // Create a simple dataset of 3D points
    let data = vec![
        vec![0.0f32, 0.0, 0.0],  // Point 0: origin
        vec![1.0, 0.0, 0.0],     // Point 1: unit x
        vec![0.0, 1.0, 0.0],     // Point 2: unit y
        vec![0.0, 0.0, 1.0],     // Point 3: unit z
        vec![1.0, 1.0, 1.0],     // Point 4: diagonal
        vec![0.5, 0.5, 0.5],     // Point 5: center
        vec![2.0, 0.0, 0.0],     // Point 6: far x
        vec![0.0, 2.0, 0.0],     // Point 7: far y
        vec![0.0, 0.0, 2.0],     // Point 8: far z
        vec![-1.0, -1.0, -1.0],  // Point 9: negative diagonal
    ];

    println!("Dataset: {} points, {} dimensions", data.len(), data[0].len());

    // Create a dense dataset
    let dataset = DenseDataset::from_vecs(data);

    // Create a brute-force searcher with squared L2 distance
    let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

    // Define a query point
    let query = vec![0.4f32, 0.4, 0.4];
    println!("\nQuery: {:?}", query);

    // Search for the 5 nearest neighbors
    let k = 5;
    let results = searcher.search(&query, k).unwrap();

    println!("\nTop {} nearest neighbors:", k);
    println!("{:>5} {:>10} {:>15}", "Rank", "Index", "Distance");
    println!("{:-<32}", "");

    for (rank, (index, distance)) in results.iter().enumerate() {
        println!("{:>5} {:>10} {:>15.6}", rank + 1, index, distance);
    }

    // Demonstrate different distance measures
    println!("\n--- Different Distance Measures ---\n");

    let measures = [
        (DistanceMeasure::SquaredL2, "Squared L2"),
        (DistanceMeasure::L2, "L2 (Euclidean)"),
        (DistanceMeasure::L1, "L1 (Manhattan)"),
        (DistanceMeasure::Cosine, "Cosine"),
        (DistanceMeasure::DotProduct, "Dot Product"),
    ];

    // Recreate dataset for each measure
    let data = vec![
        vec![0.0f32, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
        vec![0.5, 0.5, 0.5],
    ];

    for (measure, name) in &measures {
        let dataset = DenseDataset::from_vecs(data.clone());
        let searcher = BruteForceSearcher::new(dataset, *measure);
        let results = searcher.search(&query, 3).unwrap();

        print!("{}: ", name);
        for (index, distance) in &results {
            print!("(idx={}, dist={:.4}) ", index, distance);
        }
        println!();
    }

    // Demonstrate radius search
    println!("\n--- Radius Search ---\n");

    let dataset = DenseDataset::from_vecs(vec![
        vec![0.0f32, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
        vec![0.5, 0.5, 0.5],
    ]);

    let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);
    let query = vec![0.0f32, 0.0, 0.0];
    let radius = 1.5; // Squared distance threshold

    let results = searcher.search_radius(&query, radius).unwrap();

    println!(
        "Points within squared distance {} of origin: {} points",
        radius,
        results.len()
    );
    for (index, distance) in &results {
        println!("  Index {}: distance = {:.4}", index, distance);
    }

    println!("\nDone!");
}
