//! Benchmarks for ScaNN Rust library.

#![allow(clippy::manual_div_ceil)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use scann::prelude::*;
use rand::prelude::*;

fn generate_dataset(n: usize, dim: usize, seed: u64) -> DenseDataset<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let data: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();
    DenseDataset::from_vecs(data)
}

fn generate_queries(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

fn benchmark_brute_force(c: &mut Criterion) {
    let mut group = c.benchmark_group("brute_force");

    for &n in &[1000, 5000, 10000] {
        let dataset = generate_dataset(n, 64, 42);
        let queries = generate_queries(10, 64, 123);
        let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

        group.bench_with_input(
            BenchmarkId::new("search_k10", n),
            &n,
            |b, _| {
                b.iter(|| {
                    for query in &queries {
                        let _ = black_box(searcher.search(query, 10).unwrap());
                    }
                })
            },
        );
    }

    group.finish();
}

fn benchmark_distance_measures(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_measures");

    let dim = 128;
    let a = Datapoint::dense((0..dim).map(|i| i as f32 / dim as f32).collect::<Vec<_>>());
    let b = Datapoint::dense((0..dim).map(|i| (dim - i) as f32 / dim as f32).collect::<Vec<_>>());

    let measures = [
        (DistanceMeasure::SquaredL2, "squared_l2"),
        (DistanceMeasure::L2, "l2"),
        (DistanceMeasure::L1, "l1"),
        (DistanceMeasure::Cosine, "cosine"),
        (DistanceMeasure::DotProduct, "dot_product"),
    ];

    for (measure, name) in &measures {
        group.bench_function(*name, |bench| {
            bench.iter(|| {
                black_box(measure.distance(&a.as_ptr(), &b.as_ptr()))
            })
        });
    }

    group.finish();
}

fn benchmark_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans");

    let dataset = generate_dataset(5000, 32, 42);

    for &k in &[10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("cluster", k),
            &k,
            |b, &k| {
                let config = scann::trees::KMeansConfig::new(k)
                    .with_max_iterations(20)
                    .with_seed(42);
                let kmeans = KMeans::new(config);
                b.iter(|| {
                    black_box(kmeans.fit(&dataset).unwrap())
                })
            },
        );
    }

    group.finish();
}

fn benchmark_batched_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("batched_search");

    let n = 10000;
    let dim = 64;
    let dataset = generate_dataset(n, dim, 42);
    let queries = generate_queries(100, dim, 123);
    let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

    group.bench_function("sequential", |b| {
        b.iter(|| {
            for query in &queries {
                let _ = black_box(searcher.search(query, 10).unwrap());
            }
        })
    });

    group.bench_function("batched", |b| {
        b.iter(|| {
            black_box(searcher.search_batched(&queries, 10).unwrap())
        })
    });

    group.finish();
}

fn benchmark_dataset_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset");

    let dim = 64;

    for &n in &[1000, 5000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("from_vecs", n),
            &n,
            |b, &n| {
                let data: Vec<Vec<f32>> = (0..n)
                    .map(|i| (0..dim).map(|j| (i * j) as f32 / 1000.0).collect())
                    .collect();
                b.iter(|| {
                    black_box(DenseDataset::from_vecs(data.clone()))
                })
            },
        );
    }

    group.finish();
}

fn benchmark_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd");

    // Test SIMD operations with different vector sizes
    for &dim in &[64, 128, 256, 512] {
        let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..dim).map(|i| (dim - i) as f32 * 0.01).collect();

        group.bench_with_input(
            BenchmarkId::new("dot_product", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    black_box(scann::simd::dot_product_f32(&a, &b))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("squared_l2", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    black_box(scann::simd::squared_l2_f32(&a, &b))
                })
            },
        );
    }

    group.finish();
}

fn benchmark_lut16(c: &mut Criterion) {
    use scann::hashes::Lut16SimdTables;

    let mut group = c.benchmark_group("lut16");

    // Create lookup tables
    for &num_subspaces in &[8, 16, 32] {
        let tables: Vec<[f32; 16]> = (0..num_subspaces)
            .map(|_| std::array::from_fn(|i| i as f32))
            .collect();
        let table_refs: Vec<&[f32; 16]> = tables.iter().collect();
        let simd_tables = Lut16SimdTables::from_float_tables(&table_refs);

        // Create packed codes for 1000 datapoints
        let num_points = 1000;
        let bytes_per_point = (num_subspaces + 1) / 2;
        let packed_codes: Vec<u8> = (0..num_points * bytes_per_point)
            .map(|i| (i % 256) as u8)
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch_distances", num_subspaces),
            &num_subspaces,
            |bench, _| {
                let mut results = vec![0.0f32; num_points];
                bench.iter(|| {
                    simd_tables.compute_distances_batch(&packed_codes, num_points, &mut results);
                    black_box(results[0])
                })
            },
        );
    }

    group.finish();
}

fn benchmark_one_to_many_asymmetric(c: &mut Criterion) {
    use scann::distance_measures::{one_to_many_int8_float_squared_l2, one_to_many_int8_float_dot_product};

    let mut group = c.benchmark_group("one_to_many_asymmetric");

    let dim = 128;
    let num_points = 10000;
    let inv_multiplier = 1.0 / 127.0;

    // Generate test data
    let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
    let database: Vec<i8> = (0..num_points * dim)
        .map(|i| (i % 256) as u8 as i8)
        .collect();

    group.bench_function("int8_squared_l2", |bench| {
        let mut results = vec![0.0f32; num_points];
        bench.iter(|| {
            one_to_many_int8_float_squared_l2(
                &query,
                &database,
                inv_multiplier,
                dim,
                num_points,
                &mut results,
            );
            black_box(results[0])
        })
    });

    group.bench_function("int8_dot_product", |bench| {
        let mut results = vec![0.0f32; num_points];
        bench.iter(|| {
            one_to_many_int8_float_dot_product(
                &query,
                &database,
                inv_multiplier,
                dim,
                num_points,
                &mut results,
            );
            black_box(results[0])
        })
    });

    group.finish();
}

fn benchmark_scalar_quantized_search(c: &mut Criterion) {
    use scann::brute_force::{ScalarQuantizedBruteForceSearcher, ScalarQuantizedConfig};

    let mut group = c.benchmark_group("scalar_quantized");

    let dim = 128;
    for &n in &[1000, 5000, 10000] {
        let dataset = generate_dataset(n, dim, 42);
        let queries = generate_queries(10, dim, 123);

        // Float searcher for comparison
        let float_searcher = BruteForceSearcher::new(dataset.clone(), DistanceMeasure::SquaredL2);

        // Quantized searcher
        let config = ScalarQuantizedConfig::squared_l2();
        let quant_searcher = ScalarQuantizedBruteForceSearcher::new(&dataset, config).unwrap();

        group.bench_with_input(
            BenchmarkId::new("float32", n),
            &n,
            |b, _| {
                b.iter(|| {
                    for query in &queries {
                        let _ = black_box(float_searcher.search(query, 10).unwrap());
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("int8_quantized", n),
            &n,
            |b, _| {
                b.iter(|| {
                    for query in &queries {
                        let _ = black_box(quant_searcher.search(query, 10).unwrap());
                    }
                })
            },
        );
    }

    group.finish();
}

fn benchmark_many_to_many(c: &mut Criterion) {
    use scann::distance_measures::many_to_many::{batch_squared_l2_simd, BatchDistanceMatrix};

    let mut group = c.benchmark_group("many_to_many");

    let dim = 64;

    for &num_queries in &[10, 50, 100] {
        let num_db = 1000;

        // Generate contiguous data
        let queries: Vec<f32> = (0..num_queries * dim)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let database: Vec<f32> = (0..num_db * dim)
            .map(|i| (i as f32) * 0.001)
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch_squared_l2", num_queries),
            &num_queries,
            |b, &num_q| {
                let mut results = vec![0.0f32; num_q * num_db];
                b.iter(|| {
                    batch_squared_l2_simd(
                        &queries,
                        &database,
                        dim,
                        dim,
                        num_q,
                        num_db,
                        &mut results,
                    );
                    black_box(results[0])
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("batch_matrix", num_queries),
            &num_queries,
            |b, &num_q| {
                b.iter(|| {
                    black_box(BatchDistanceMatrix::from_squared_l2(
                        &queries,
                        &database,
                        dim,
                        num_q,
                        num_db,
                    ))
                })
            },
        );
    }

    group.finish();
}

fn benchmark_lock_free_mutator(c: &mut Criterion) {
    use scann::mutator::{MutationBuffer, MutableDataset};
    use std::sync::Arc;
    use std::thread;

    let mut group = c.benchmark_group("lock_free_mutator");

    // Benchmark MutationBuffer throughput
    group.bench_function("mutation_buffer_single_thread", |b| {
        let buffer = MutationBuffer::new(100000);
        let mut idx = 0u32;
        b.iter(|| {
            buffer.add(idx, vec![1.0, 2.0, 3.0]);
            idx = idx.wrapping_add(1);
            black_box(())
        })
    });

    // Benchmark MutableDataset add (single-threaded)
    group.bench_function("mutable_dataset_add", |b| {
        let dataset = MutableDataset::new(64);
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        b.iter(|| {
            let _ = black_box(dataset.add(data.clone()));
        })
    });

    // Benchmark MutableDataset get (single-threaded)
    group.bench_function("mutable_dataset_get", |b| {
        let dataset = MutableDataset::new(64);
        for i in 0..1000 {
            let data: Vec<f32> = (0..64).map(|j| (i * j) as f32).collect();
            dataset.add(data).unwrap();
        }
        let mut idx = 0u32;
        b.iter(|| {
            let result = dataset.get(idx % 1000);
            idx = idx.wrapping_add(1);
            black_box(result)
        })
    });

    // Benchmark MutableDataset get with read_guard (single-threaded)
    group.bench_function("mutable_dataset_get_guard", |b| {
        let dataset = MutableDataset::new(64);
        for i in 0..1000 {
            let data: Vec<f32> = (0..64).map(|j| (i * j) as f32).collect();
            dataset.add(data).unwrap();
        }
        b.iter(|| {
            let guard = dataset.read_guard();
            let mut sum = 0.0f32;
            for idx in 0..100 {
                if let Some(data) = guard.get(idx) {
                    sum += data[0];
                }
            }
            black_box(sum)
        })
    });

    // Benchmark concurrent writes throughput
    for &num_threads in &[2, 4, 8] {
        group.bench_function(format!("concurrent_adds_{}_threads", num_threads), |b| {
            b.iter_custom(|iters| {
                let dataset = Arc::new(MutableDataset::new(16));
                let ops_per_thread = (iters as usize) / num_threads;

                let start = std::time::Instant::now();

                let handles: Vec<_> = (0..num_threads)
                    .map(|_| {
                        let ds = Arc::clone(&dataset);
                        thread::spawn(move || {
                            for _ in 0..ops_per_thread {
                                let _ = ds.add(vec![1.0; 16]);
                            }
                        })
                    })
                    .collect();

                for handle in handles {
                    handle.join().unwrap();
                }

                start.elapsed()
            })
        });
    }

    // Benchmark concurrent reads throughput
    for &num_threads in &[2, 4, 8] {
        group.bench_function(format!("concurrent_reads_{}_threads", num_threads), |b| {
            let dataset = Arc::new(MutableDataset::new(16));
            for i in 0..1000 {
                dataset.add(vec![i as f32; 16]).unwrap();
            }

            b.iter_custom(|iters| {
                let ops_per_thread = (iters as usize) / num_threads;

                let start = std::time::Instant::now();

                let handles: Vec<_> = (0..num_threads)
                    .map(|t| {
                        let ds = Arc::clone(&dataset);
                        thread::spawn(move || {
                            for i in 0..ops_per_thread {
                                let _ = ds.get(((t * 100 + i) % 1000) as u32);
                            }
                        })
                    })
                    .collect();

                for handle in handles {
                    handle.join().unwrap();
                }

                start.elapsed()
            })
        });
    }

    group.finish();
}

fn benchmark_top_k(c: &mut Criterion) {
    use scann::brute_force::{TopK, FixedTopK};

    let mut group = c.benchmark_group("top_k");

    // Generate test data: 10000 random distances
    let distances: Vec<f32> = (0..10000)
        .map(|i| ((i * 17) % 10000) as f32 / 100.0)
        .collect();

    group.bench_function("heap_k10", |b| {
        b.iter(|| {
            let mut top_k = TopK::new(10);
            for (i, &d) in distances.iter().enumerate() {
                top_k.push(i as u32, d);
            }
            black_box(top_k.results())
        })
    });

    group.bench_function("fixed_k10", |b| {
        b.iter(|| {
            let mut top_k: FixedTopK<10> = FixedTopK::new();
            for (i, &d) in distances.iter().enumerate() {
                top_k.push(i as u32, d);
            }
            black_box(top_k.results())
        })
    });

    group.bench_function("heap_k32", |b| {
        b.iter(|| {
            let mut top_k = TopK::new(32);
            for (i, &d) in distances.iter().enumerate() {
                top_k.push(i as u32, d);
            }
            black_box(top_k.results())
        })
    });

    group.bench_function("fixed_k32", |b| {
        b.iter(|| {
            let mut top_k: FixedTopK<32> = FixedTopK::new();
            for (i, &d) in distances.iter().enumerate() {
                top_k.push(i as u32, d);
            }
            black_box(top_k.results())
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_brute_force,
    benchmark_distance_measures,
    benchmark_kmeans,
    benchmark_batched_search,
    benchmark_dataset_operations,
    benchmark_simd_operations,
    benchmark_lut16,
    benchmark_one_to_many_asymmetric,
    benchmark_scalar_quantized_search,
    benchmark_many_to_many,
    benchmark_lock_free_mutator,
    benchmark_top_k,
);
criterion_main!(benches);
