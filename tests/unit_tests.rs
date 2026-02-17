//! Comprehensive unit tests for ScaNN Rust library.

use scann::prelude::*;

mod datapoint_tests {
    use super::*;

    #[test]
    fn test_dense_datapoint_creation() {
        let dp = Datapoint::dense(vec![1.0f32, 2.0, 3.0, 4.0]);
        assert!(dp.is_dense());
        assert!(!dp.is_sparse());
        assert_eq!(dp.dimensionality(), 4);
        assert_eq!(dp.nonzero_entries(), 4);
    }

    #[test]
    fn test_sparse_datapoint_creation() {
        let dp = Datapoint::sparse(vec![1.0f32, 2.0], vec![0, 3], 5);
        assert!(dp.is_sparse());
        assert!(!dp.is_dense());
        assert_eq!(dp.dimensionality(), 5);
        assert_eq!(dp.nonzero_entries(), 2);
    }

    #[test]
    fn test_datapoint_get() {
        let dp = Datapoint::dense(vec![1.0f32, 2.0, 3.0]);
        assert_eq!(dp.get(0), 1.0);
        assert_eq!(dp.get(1), 2.0);
        assert_eq!(dp.get(2), 3.0);
    }

    #[test]
    fn test_sparse_datapoint_get() {
        let dp = Datapoint::sparse(vec![5.0f32, 10.0], vec![1, 3], 5);
        assert_eq!(dp.get(0), 0.0);
        assert_eq!(dp.get(1), 5.0);
        assert_eq!(dp.get(2), 0.0);
        assert_eq!(dp.get(3), 10.0);
        assert_eq!(dp.get(4), 0.0);
    }

    #[test]
    fn test_datapoint_l2_norm() {
        let dp = Datapoint::dense(vec![3.0f32, 4.0]);
        assert!((dp.l2_norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_datapoint_normalize() {
        let mut dp = Datapoint::dense(vec![3.0f32, 4.0]);
        dp.normalize();
        assert!((dp.l2_norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_datapoint_to_dense() {
        let sparse = Datapoint::sparse(vec![1.0f32, 2.0], vec![0, 2], 3);
        let dense = sparse.to_dense();
        assert!(dense.is_dense());
        assert_eq!(dense.values(), &[1.0, 0.0, 2.0]);
    }
}

mod dataset_tests {
    use super::*;

    #[test]
    fn test_dense_dataset_from_vecs() {
        let data = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let dataset = DenseDataset::from_vecs(data);

        assert_eq!(dataset.size(), 3);
        assert_eq!(dataset.dimensionality(), 3);
        assert!(dataset.is_dense());
    }

    #[test]
    fn test_dense_dataset_from_flat() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dataset = DenseDataset::from_flat(data, 3).unwrap();

        assert_eq!(dataset.size(), 2);
        assert_eq!(dataset.dimensionality(), 3);
    }

    #[test]
    fn test_dense_dataset_get() {
        let dataset = DenseDataset::from_vecs(vec![
            vec![1.0f32, 2.0],
            vec![3.0, 4.0],
        ]);

        let dp0 = dataset.get(0).unwrap();
        assert_eq!(dp0.values(), &[1.0, 2.0]);

        let dp1 = dataset.get(1).unwrap();
        assert_eq!(dp1.values(), &[3.0, 4.0]);
    }

    #[test]
    fn test_dense_dataset_append() {
        let mut dataset = DenseDataset::<f32>::new();

        let dp1 = Datapoint::dense(vec![1.0, 2.0, 3.0]);
        dataset.append(&dp1.as_ptr()).unwrap();

        let dp2 = Datapoint::dense(vec![4.0, 5.0, 6.0]);
        dataset.append(&dp2.as_ptr()).unwrap();

        assert_eq!(dataset.size(), 2);
    }

    #[test]
    fn test_sparse_dataset() {
        let mut dataset = SparseDataset::<f32>::new(5);
        dataset.append(vec![1.0, 2.0], vec![0, 3]).unwrap();
        dataset.append(vec![3.0], vec![2]).unwrap();

        assert_eq!(dataset.size(), 2);
        assert!(!dataset.is_dense());
    }
}

mod distance_tests {
    use super::*;

    fn make_dp(values: Vec<f32>) -> Datapoint<f32> {
        Datapoint::dense(values)
    }

    #[test]
    fn test_l1_distance() {
        let a = make_dp(vec![1.0, 2.0, 3.0]);
        let b = make_dp(vec![4.0, 5.0, 6.0]);
        let dist = DistanceMeasure::L1.distance(&a.as_ptr(), &b.as_ptr());
        assert!((dist - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance() {
        let a = make_dp(vec![0.0, 0.0]);
        let b = make_dp(vec![3.0, 4.0]);
        let dist = DistanceMeasure::L2.distance(&a.as_ptr(), &b.as_ptr());
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_squared_l2_distance() {
        let a = make_dp(vec![0.0, 0.0]);
        let b = make_dp(vec![3.0, 4.0]);
        let dist = DistanceMeasure::SquaredL2.distance(&a.as_ptr(), &b.as_ptr());
        assert!((dist - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = make_dp(vec![1.0, 0.0]);
        let b = make_dp(vec![1.0, 0.0]);
        let dist = DistanceMeasure::Cosine.distance(&a.as_ptr(), &b.as_ptr());
        assert!(dist.abs() < 1e-6); // Same direction = 0 distance

        let c = make_dp(vec![0.0, 1.0]);
        let dist2 = DistanceMeasure::Cosine.distance(&a.as_ptr(), &c.as_ptr());
        assert!((dist2 - 1.0).abs() < 1e-6); // Orthogonal = 1 distance
    }

    #[test]
    fn test_dot_product_distance() {
        let a = make_dp(vec![1.0, 2.0, 3.0]);
        let b = make_dp(vec![4.0, 5.0, 6.0]);
        let dist = DistanceMeasure::DotProduct.distance(&a.as_ptr(), &b.as_ptr());
        // Dot product is 4 + 10 + 18 = 32, negated for distance
        assert!((dist - (-32.0)).abs() < 1e-6);
    }

    #[test]
    fn test_hamming_distance() {
        let a = make_dp(vec![1.0, 0.0, 1.0, 0.0]);
        let b = make_dp(vec![1.0, 1.0, 0.0, 0.0]);
        let dist = DistanceMeasure::Hamming.distance(&a.as_ptr(), &b.as_ptr());
        assert!((dist - 2.0).abs() < 1e-6);
    }
}

mod brute_force_tests {
    use super::*;

    fn create_dataset() -> DenseDataset<f32> {
        DenseDataset::from_vecs(vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ])
    }

    #[test]
    fn test_brute_force_exact_match() {
        let dataset = create_dataset();
        let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

        let query = vec![0.0, 0.0, 0.0];
        let results = searcher.search(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1.abs() < 1e-6);
    }

    #[test]
    fn test_brute_force_top_k() {
        let dataset = create_dataset();
        let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

        let query = vec![0.0, 0.0, 0.0];
        let results = searcher.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_brute_force_batched() {
        let dataset = create_dataset();
        let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

        let queries = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ];
        let results = searcher.search_batched(&queries, 2).unwrap();

        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(r.len(), 2);
        }
    }

    #[test]
    fn test_brute_force_radius() {
        let dataset = create_dataset();
        let searcher = BruteForceSearcher::new(dataset, DistanceMeasure::SquaredL2);

        let query = vec![0.0, 0.0, 0.0];
        let results = searcher.search_radius(&query, 1.5).unwrap();

        // Should include origin (0) and unit vectors (1,2,3)
        assert_eq!(results.len(), 4);
    }
}

mod kmeans_tests {
    use super::*;

    fn create_clustered_data() -> DenseDataset<f32> {
        let mut data = Vec::new();

        // Cluster 1
        for i in 0..10 {
            data.push(vec![0.0 + (i as f32) * 0.1, 0.0]);
        }

        // Cluster 2
        for i in 0..10 {
            data.push(vec![10.0 + (i as f32) * 0.1, 10.0]);
        }

        DenseDataset::from_vecs(data)
    }

    #[test]
    fn test_kmeans_clustering() {
        let dataset = create_clustered_data();
        let kmeans = KMeans::with_clusters(2);
        let result = kmeans.fit(&dataset).unwrap();

        assert_eq!(result.centers.len(), 2);
        assert_eq!(result.assignments.len(), 20);
    }

    #[test]
    fn test_kmeans_tree() {
        let dataset = create_clustered_data();
        let config = scann::trees::KMeansTreeConfig::new(2).with_seed(42);
        let mut tree = KMeansTree::new(config);
        tree.build(&dataset).unwrap();

        assert!(tree.root().is_some());
        assert_eq!(tree.size(), 20);
    }
}

mod partitioning_tests {
    use super::*;

    fn create_dataset() -> DenseDataset<f32> {
        let mut data = Vec::new();
        for i in 0..50 {
            data.push(vec![(i as f32) / 10.0, (i as f32) / 5.0]);
        }
        DenseDataset::from_vecs(data)
    }

    #[test]
    fn test_tree_partitioner() {
        let dataset = create_dataset();
        let config = scann::partitioning::PartitionerConfig::new(5);
        let mut partitioner = TreePartitioner::new(config);
        partitioner.build(&dataset).unwrap();

        assert_eq!(partitioner.num_partitions(), 5);
        assert_eq!(partitioner.num_datapoints(), 50);
    }

    #[test]
    fn test_partitioner_query() {
        let dataset = create_dataset();
        let config = scann::partitioning::PartitionerConfig::new(5);
        let mut partitioner = TreePartitioner::new(config);
        partitioner.build(&dataset).unwrap();

        let query = Datapoint::dense(vec![0.5f32, 0.5]);
        let result = partitioner.partition(&query.as_ptr(), 2).unwrap();

        assert_eq!(result.len(), 2);
    }
}

mod hash_tests {
    use super::*;
    use scann::hashes::{AsymmetricHasher, AsymmetricHasherConfig};

    fn create_dataset() -> DenseDataset<f32> {
        let mut data = Vec::new();
        for i in 0..100 {
            let mut vec = Vec::with_capacity(16);
            for j in 0..16 {
                vec.push((i * j) as f32 / 100.0);
            }
            data.push(vec);
        }
        DenseDataset::from_vecs(data)
    }

    #[test]
    fn test_asymmetric_hasher() {
        let dataset = create_dataset();
        let config = AsymmetricHasherConfig::new(8, 4).with_seed(42);
        let mut hasher = AsymmetricHasher::new(config);
        hasher.build(dataset).unwrap();

        assert_eq!(hasher.num_datapoints(), 100);
        assert_eq!(hasher.dimensionality(), 16);
    }

    #[test]
    fn test_asymmetric_hasher_search() {
        let dataset = create_dataset();
        let config = AsymmetricHasherConfig::new(8, 4).with_seed(42);
        let mut hasher = AsymmetricHasher::new(config);
        hasher.build(dataset).unwrap();

        let query = vec![0.5f32; 16];
        let results = hasher.search(&query, 10).unwrap();

        assert_eq!(results.len(), 10);
    }
}

mod scann_tests {
    use super::*;

    fn create_dataset() -> DenseDataset<f32> {
        let mut data = Vec::new();
        for i in 0..100 {
            let mut vec = Vec::with_capacity(8);
            for j in 0..8 {
                vec.push((i as f32 + j as f32 * 0.1).sin());
            }
            data.push(vec);
        }
        DenseDataset::from_vecs(data)
    }

    #[test]
    fn test_scann_brute_force() {
        let dataset = create_dataset();
        let scann = Scann::brute_force(dataset).unwrap();

        assert_eq!(scann.search_mode(), SearchMode::BruteForce);
        assert_eq!(scann.size(), 100);

        let query = vec![0.5f32; 8];
        let results = scann.search(&query, 10).unwrap();

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_scann_builder() {
        let dataset = create_dataset();
        let scann = ScannBuilder::new()
            .num_neighbors(5)
            .distance_measure(DistanceMeasure::SquaredL2)
            .brute_force()
            .build(dataset)
            .unwrap();

        let query = vec![0.5f32; 8];
        let results = scann.search(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_scann_searcher_trait() {
        let dataset = create_dataset();
        let scann = Scann::brute_force(dataset).unwrap();

        let query = vec![0.5f32; 8];
        let result = scann.search(&query, 5).unwrap();

        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_scann_batched() {
        let dataset = create_dataset();
        let scann = Scann::brute_force(dataset).unwrap();

        let queries = vec![
            vec![0.1f32; 8],
            vec![0.5f32; 8],
            vec![0.9f32; 8],
        ];
        let results = scann.search_batched(&queries, 5).unwrap();

        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.len(), 5);
        }
    }
}
