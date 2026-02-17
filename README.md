# ScaNN Rust Library

A high-performance Rust implementation of Google's ScaNN (Scalable Nearest Neighbors) library for efficient approximate nearest neighbor search.

## Features

- **Brute-force search** with SIMD-optimized distance computations
- **Distance measures**: L1, L2, SquaredL2, Cosine, DotProduct, Hamming, Jaccard
- **K-means clustering** and tree-based partitioning
- **Asymmetric hashing** with LUT16 SIMD acceleration
- **Quantization**: Scalar (Int8), FP8, BFloat16 for memory efficiency
- **Projections**: PCA, Random Orthogonal, OPQ, Chunking, Truncate
- **Tree-X-Hybrid** search for large-scale datasets
- **Dynamic index mutations** (add/update/delete)
- **Parallel search** with configurable threading

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
scann = { path = "path/to/scann/rust" }
```

## Quick Start

```rust
use scann::prelude::*;

fn main() -> Result<()> {
    // Create a dataset
    let data = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
        vec![1.1, 2.1, 3.1],
    ];
    let dataset = DenseDataset::from_vecs(data);

    // Build a brute-force searcher
    let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

    // Search for k nearest neighbors
    let query = vec![1.0, 2.0, 3.0];
    let results = searcher.search(&query, 2)?;

    for (idx, distance) in results {
        println!("Index: {}, Distance: {:.4}", idx, distance);
    }

    Ok(())
}
```

## API Reference

### Core Types

#### `DenseDataset<T>`

A dense dataset storing vectors in contiguous memory.

```rust
// Create from vectors
let dataset = DenseDataset::from_vecs(vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
]);

// Create from flat array
let flat_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let dataset = DenseDataset::from_flat(flat_data, 3); // 3 = dimensionality

// Access properties
println!("Size: {}", dataset.size());
println!("Dimensions: {}", dataset.dimensionality());

// Get a datapoint
let point = dataset.get(0).unwrap();
```

#### `DistanceMeasure`

Enum representing available distance measures:

```rust
pub enum DistanceMeasure {
    L1,              // Manhattan distance
    L2,              // Euclidean distance
    SquaredL2,       // Squared Euclidean (faster, no sqrt)
    Cosine,          // Cosine distance (1 - similarity)
    DotProduct,      // Negative dot product (for max inner product)
    Hamming,         // Hamming distance for binary vectors
    Jaccard,         // Jaccard distance for sets
}
```

### Searchers

#### `BruteForceSearcher`

Exact nearest neighbor search by exhaustive distance computation.

```rust
use scann::prelude::*;

// Create searcher
let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

// Single query search
let results = searcher.search(&query, k)?;

// Batched search (more efficient for multiple queries)
let queries: Vec<Vec<f32>> = vec![query1, query2, query3];
let batch_results = searcher.search_batched(&queries, k)?;

// Radius search (find all points within distance)
let results = searcher.search_radius(&query, max_distance, max_results)?;

// Enable parallel search
let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2)
    .with_parallel(true);
```

#### `ScalarQuantizedBruteForceSearcher`

Memory-efficient brute-force search using Int8 quantization (4x memory reduction).

```rust
use scann::brute_force::{ScalarQuantizedBruteForceSearcher, ScalarQuantizedConfig};

// Create with squared L2 distance
let config = ScalarQuantizedConfig::squared_l2();
let searcher = ScalarQuantizedBruteForceSearcher::new(&dataset, config)?;

// Or with dot product
let config = ScalarQuantizedConfig::dot_product();
let searcher = ScalarQuantizedBruteForceSearcher::new(&dataset, config)?;

// Search (same API as BruteForceSearcher)
let results = searcher.search(&query, k)?;

// Check compression ratio
println!("Memory usage: {} bytes", searcher.memory_usage());
```

#### `Scann` (High-Level Interface)

The main ScaNN interface supporting multiple search modes.

```rust
use scann::prelude::*;

// Build with ScannBuilder
let scann = ScannBuilder::new()
    .with_dataset(dataset)
    .with_distance_measure(DistanceMeasure::SquaredL2)
    .with_num_neighbors(10)
    .brute_force()  // or .partitioned() or .hashed()
    .build()?;

// Search
let results = scann.search(&query, 10)?;

// Batched search
let results = scann.search_batched(&queries, 10)?;
```

### Partitioning

#### `KMeans`

K-means clustering for dataset partitioning.

```rust
use scann::trees::{KMeans, KMeansConfig};

// Configure k-means
let config = KMeansConfig::new(100)  // 100 clusters
    .with_max_iterations(25)
    .with_seed(42);

// Fit to dataset
let kmeans = KMeans::new(config);
let result = kmeans.fit(&dataset)?;

// Access centroids
let centroids = result.centroids();
let assignments = result.assignments();
```

#### `KMeansTree`

Hierarchical k-means tree for efficient partitioning.

```rust
use scann::trees::{KMeansTree, KMeansTreeConfig};

let config = KMeansTreeConfig::new()
    .with_num_children(100)
    .with_max_leaf_size(1000)
    .with_max_depth(3);

let tree = KMeansTree::build(&dataset, config)?;

// Find nearest leaf
let leaf_id = tree.find_nearest_leaf(&query);

// Get points in leaf
let points = tree.get_leaf_points(leaf_id);
```

#### `TreePartitioner`

Tree-based partitioner for search.

```rust
use scann::partitioning::TreePartitioner;

let partitioner = TreePartitioner::new(&dataset, num_partitions)?;

// Get partition for a query
let partition_id = partitioner.partition(&query);

// Get points in partition
let points = partitioner.get_partition_points(partition_id);
```

### Asymmetric Hashing

#### `AsymmetricHasher`

Product quantization with asymmetric distance computation.

```rust
use scann::hashes::{AsymmetricHasher, AsymmetricHasherConfig};

let config = AsymmetricHasherConfig::new()
    .with_num_clusters_per_block(16)
    .with_num_blocks(8);

let hasher = AsymmetricHasher::new(config);
let trained = hasher.train(&dataset)?;

// Encode database
let codes = trained.encode_database(&dataset);

// Create lookup table for query
let lut = trained.create_lookup_table(&query);

// Compute distances
let distances = trained.compute_distances(&lut, &codes);
```

#### `Lut16LookupTables` (LUT16 SIMD)

High-performance 4-bit lookup tables with SIMD acceleration.

```rust
use scann::hashes::{Lut16LookupTables, PackedCodes4Bit, Lut16SimdTables};

// Create lookup tables (16 entries per subspace)
let tables: Vec<[f32; 16]> = compute_distance_tables(&query, &codebook);
let lut = Lut16LookupTables::from_tables(&tables);

// Pack codes (4-bit per entry)
let packed = PackedCodes4Bit::from_codes(&codes);

// Compute distances with SIMD
let simd_tables = lut.to_simd_tables();
let mut distances = vec![0.0f32; num_points];
simd_tables.compute_distances_batch(packed.data(), num_points, &mut distances);

// See "Performance" below for measured throughput on this machine.
```

### Quantization

#### `ScalarQuantizer`

Int8 scalar quantization for memory reduction.

```rust
use scann::quantization::{ScalarQuantizer, ScalarQuantizerConfig, QuantizedDataset};

// Create quantizer
let config = ScalarQuantizerConfig::new();
let quantizer = ScalarQuantizer::from_dataset(&dataset, config)?;

// Quantize dataset
let quantized: QuantizedDataset = QuantizedDataset::from_dataset(&dataset, &quantizer)?;

// Access quantized data
let scale = quantizer.scale();
let offset = quantizer.offset();

// Dequantize a value
let original = quantizer.dequantize_value(quantized_value);
```

#### `BFloat16Dataset`

BFloat16 storage for 2x memory reduction with minimal precision loss.

```rust
use scann::quantization::{BFloat16Dataset, f32_to_bf16, bf16_to_f32};

// Convert dataset
let bf16_dataset = BFloat16Dataset::from_f32_dataset(&dataset);

// Convert individual values
let bf16_val = f32_to_bf16(1.5f32);
let f32_val = bf16_to_f32(bf16_val);
```

#### `Fp8Value`

FP8 (E4M3) quantization for 4x memory reduction.

```rust
use scann::quantization::{Fp8Value, Fp8Quantizer, Fp8Config};

let config = Fp8Config::e4m3();  // or e5m2()
let quantizer = Fp8Quantizer::new(config);

// Quantize
let fp8 = Fp8Value::from_f32(1.5);
let back = fp8.to_f32_e4m3();
```

### Projections

#### `PcaProjection`

PCA dimensionality reduction.

```rust
use scann::projection::{PcaProjection, PcaConfig};

let config = PcaConfig::new(64);  // reduce to 64 dimensions
let pca = PcaProjection::fit(&dataset, config)?;

// Project a vector
let projected = pca.project(&original_vector);

// Get explained variance
let variance = pca.explained_variance_ratio();
```

#### `RandomOrthogonalProjection`

Random orthogonal projection for dimensionality reduction.

```rust
use scann::projection::{RandomOrthogonalProjection, RandomProjectionConfig};

let config = RandomProjectionConfig::new(128, 64)  // from 128 to 64 dims
    .with_seed(42);
let proj = RandomOrthogonalProjection::new(config);

let projected = proj.project(&vector);
```

#### `OpqProjection`

Optimized Product Quantization projection.

```rust
use scann::projection::{OpqProjection, OpqConfig};

let config = OpqConfig::new()
    .with_num_subspaces(8)
    .with_iterations(10);

let opq = OpqProjection::fit(&dataset, config)?;
let rotated = opq.project(&vector);
```

### Restricts and Filtering

#### `RestrictAllowlist` / `RestrictDenylist`

Filter search results by allowed/denied indices.

```rust
use scann::restricts::{RestrictAllowlist, RestrictDenylist};

// Only search within specific indices
let allowlist = RestrictAllowlist::from_indices(&[0, 1, 5, 10, 20]);
let results = searcher.search_with_restrict(&query, k, &allowlist)?;

// Exclude specific indices
let denylist = RestrictDenylist::from_indices(&[3, 7, 15]);
let results = searcher.search_with_restrict(&query, k, &denylist)?;
```

#### `CrowdingConstraint`

Diversify results by limiting items per category.

```rust
use scann::restricts::{CrowdingConstraint, CrowdingConfig};

// Limit to 2 results per category
let config = CrowdingConfig::new(2);
let crowding = CrowdingConstraint::new(config, &category_labels);

let diverse_results = crowding.apply(&results);
```

### Tree-X-Hybrid Search

Combines tree partitioning with asymmetric hashing for large-scale search.

```rust
use scann::tree_x_hybrid::{TreeXHybridSearcher, TreeXHybridConfig};

let config = TreeXHybridConfig::new()
    .with_num_partitions(100)
    .with_num_leaves_to_search(10)
    .with_reordering_num_neighbors(100);

let searcher = TreeXHybridSearcher::new(&dataset, config)?;

// Search returns approximate results
let results = searcher.search(&query, k)?;
```

### Dynamic Mutations

Add, update, or delete points from an index.

```rust
use scann::mutator::{MutableDataset, Mutation, MutationType};

let mut mutable = MutableDataset::new(dataset);

// Add a point
let new_id = mutable.add(&new_point)?;

// Update a point
mutable.update(existing_id, &updated_point)?;

// Delete a point
mutable.delete(existing_id)?;

// Apply batch mutations
let mutations = vec![
    Mutation::new(MutationType::Add, new_point1),
    Mutation::new(MutationType::Delete, point_to_delete),
];
mutable.apply_batch(mutations)?;
```

### SIMD Operations

Low-level SIMD-optimized distance computations.

```rust
use scann::simd::{dot_product_f32, squared_l2_f32, simd_support_level, SimdSupportLevel};

// Check SIMD support
match simd_support_level() {
    SimdSupportLevel::Avx2 => println!("AVX2 supported"),
    SimdSupportLevel::Sse41 => println!("SSE4.1 supported"),
    SimdSupportLevel::None => println!("No SIMD"),
}

// SIMD dot product (auto-dispatches to best implementation)
let dot = dot_product_f32(&vec_a, &vec_b);

// SIMD squared L2
let dist = squared_l2_f32(&vec_a, &vec_b);
```

### One-to-Many Asymmetric Distances

Efficient float query vs quantized database distances.

```rust
use scann::distance_measures::{
    one_to_many_int8_float_squared_l2,
    one_to_many_int8_float_dot_product,
    one_to_many_bf16_float_squared_l2,
};

// Float query vs Int8 database
let mut results = vec![0.0f32; num_points];
one_to_many_int8_float_squared_l2(
    &query,           // f32 query
    &int8_database,   // i8 database (contiguous)
    inv_multiplier,   // 1.0 / quantization_scale
    dimensionality,
    num_points,
    &mut results,
);

// Float query vs BFloat16 database
one_to_many_bf16_float_squared_l2(
    &query,
    &bf16_database,
    dimensionality,
    num_points,
    &mut results,
);
```

### Many-to-Many Batch Distances

Compute distance matrices efficiently.

```rust
use scann::distance_measures::many_to_many::{batch_squared_l2_simd, BatchDistanceMatrix};

// Compute all pairwise distances
let matrix = BatchDistanceMatrix::from_squared_l2(
    &queries_flat,    // contiguous query data
    &database_flat,   // contiguous database data
    dimensionality,
    num_queries,
    num_database,
);

// Access distances
let dist = matrix.get(query_idx, db_idx);

// Get k-nearest for each query
let results = matrix.top_k(k);
```

## Examples

### Example 1: Image Similarity Search

```rust
use scann::prelude::*;

fn image_similarity_search() -> Result<()> {
    // Load image embeddings (e.g., from a neural network)
    let embeddings: Vec<Vec<f32>> = load_image_embeddings()?;
    let dataset = DenseDataset::from_vecs(embeddings);

    // Use cosine distance for normalized embeddings
    let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::Cosine)
        .with_parallel(true);

    // Find similar images
    let query_embedding = get_query_embedding(&query_image)?;
    let similar = searcher.search(&query_embedding, 10)?;

    for (idx, distance) in similar {
        println!("Image {}: similarity = {:.4}", idx, 1.0 - distance);
    }

    Ok(())
}
```

### Example 2: Large-Scale Search with Partitioning

```rust
use scann::prelude::*;
use scann::tree_x_hybrid::{TreeXHybridSearcher, TreeXHybridConfig};

fn large_scale_search() -> Result<()> {
    // 1 million 128-dimensional vectors
    let dataset = load_million_vectors()?;

    // Configure tree-x-hybrid for speed/accuracy tradeoff
    let config = TreeXHybridConfig::new()
        .with_num_partitions(1000)      // 1000 partitions
        .with_num_leaves_to_search(50)  // search 5% of partitions
        .with_reordering_num_neighbors(200);

    let searcher = TreeXHybridSearcher::new(&dataset, config)?;

    // Search is approximate and typically faster than brute force.
    let results = searcher.search(&query, 10)?;

    Ok(())
}
```

### Example 3: Memory-Efficient Search with Quantization

```rust
use scann::prelude::*;
use scann::brute_force::{ScalarQuantizedBruteForceSearcher, ScalarQuantizedConfig};

fn memory_efficient_search() -> Result<()> {
    let dataset = load_dataset()?;

    // Original memory: 10M points * 128 dims * 4 bytes = 5.12 GB
    // Quantized memory: 10M points * 128 dims * 1 byte = 1.28 GB (4x reduction)

    let config = ScalarQuantizedConfig::squared_l2();
    let searcher = ScalarQuantizedBruteForceSearcher::new(&dataset, config)?;

    println!("Memory usage: {} MB", searcher.memory_usage() / 1_000_000);

    // Search with quantized vectors (accuracy depends on your data/config).
    let results = searcher.search(&query, 10)?;

    Ok(())
}
```

### Example 4: Batched Search for High Throughput

```rust
use scann::prelude::*;

fn high_throughput_search() -> Result<()> {
    let dataset = DenseDataset::from_vecs(load_vectors()?);
    let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2)
        .with_parallel(true);

    // Process queries in batches for better cache utilization
    let queries: Vec<Vec<f32>> = load_queries()?;
    let batch_size = 100;

    for batch in queries.chunks(batch_size) {
        let results = searcher.search_batched(batch, 10)?;
        process_results(&results)?;
    }

    // Batched search is usually much higher throughput than sequential queries.
    Ok(())
}
```

### Example 5: Streaming Updates

```rust
use scann::prelude::*;
use scann::mutator::MutableDataset;

fn streaming_index() -> Result<()> {
    let initial_data = load_initial_data()?;
    let mut mutable = MutableDataset::new(DenseDataset::from_vecs(initial_data));

    // Stream new data
    for new_point in stream_new_data() {
        let id = mutable.add(&new_point)?;
        println!("Added point with id: {}", id);

        // Periodically rebuild for better performance
        if mutable.pending_mutations() > 10000 {
            mutable.rebuild()?;
        }
    }

    Ok(())
}
```

## Performance

Machine used for these numbers:
- CPU: Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz (2 sockets, 48 cores / 96 threads)
- SIMD: AVX2 available
- RAM: 123 GiB
- OS: Linux 6.14.0-37-generic (x86_64)

Commands:

```bash
cargo test -q
cargo bench --bench scann_benchmark
```

Criterion medians from this machine:

| Operation | Benchmark | Time (median) | Throughput |
|-----------|-----------|---------------|------------|
| Brute Force (k=10, 10 queries) | `brute_force/search_k10/10000` | 1.3753 ms | 7,271 queries/s |
| Batched Search (100 queries) | `batched_search/batched` | 885.45 us | 112,937 queries/s |
| Scalar Quantized (k=10, 10 queries) | `scalar_quantized/int8_quantized/10000` | 2.2031 ms | 4,539 queries/s |
| SIMD Dot Product | `simd/dot_product/128` | 11.739 ns | 85.2M ops/s |
| LUT16 Batch (1k points, 16 subspaces) | `lut16/batch_distances/16` | 20.595 us | 48.6M lookups/s |
| Int8 Asymmetric (10k x 128d) | `one_to_many_asymmetric/int8_squared_l2` | 180.11 us | 55.5M points/s |

### Observed Speedups

| Comparison | Speedup |
|------------|---------|
| Batched vs sequential (`batched_search`) | 16.4x faster |

## Feature Flags

```toml
[features]
default = ["simd"]
simd = []           # Enable SIMD optimizations (recommended)
parallel = ["rayon"] # Enable parallel search
serialize = ["serde"] # Enable serialization
```

## Thread Safety

- `DenseDataset<T>` is `Send + Sync`
- `BruteForceSearcher` is `Send + Sync` (safe for concurrent searches)
- `MutableDataset` requires `&mut` for mutations

## License

Apache License 2.0

## Acknowledgments

This is a Rust port of [Google's ScaNN library](https://github.com/google-research/google-research/tree/master/scann).
