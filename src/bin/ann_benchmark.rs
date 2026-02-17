use rand::{rngs::StdRng, Rng, SeedableRng};
use scann::data_format::DenseDataset;
use scann::distance_measures::DistanceMeasure;
use scann::{Scann, ScannBuilder};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Algorithm {
    BruteForce,
    Partitioned,
    Hashed,
    TreeAh,
}

impl FromStr for Algorithm {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "brute-force" | "brute_force" | "bruteforce" => Ok(Self::BruteForce),
            "partitioned" => Ok(Self::Partitioned),
            "hashed" => Ok(Self::Hashed),
            "tree-ah" | "tree_ah" | "treeah" => Ok(Self::TreeAh),
            _ => Err(format!("unsupported algorithm: {s}")),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum DistanceArg {
    SquaredL2,
    L2,
    L1,
    Cosine,
    DotProduct,
}

impl DistanceArg {
    fn to_measure(self) -> DistanceMeasure {
        match self {
            Self::SquaredL2 => DistanceMeasure::SquaredL2,
            Self::L2 => DistanceMeasure::L2,
            Self::L1 => DistanceMeasure::L1,
            Self::Cosine => DistanceMeasure::Cosine,
            Self::DotProduct => DistanceMeasure::DotProduct,
        }
    }
}

impl FromStr for DistanceArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "squared-l2" | "squared_l2" => Ok(Self::SquaredL2),
            "l2" => Ok(Self::L2),
            "l1" => Ok(Self::L1),
            "cosine" => Ok(Self::Cosine),
            "dot-product" | "dot_product" => Ok(Self::DotProduct),
            _ => Err(format!("unsupported distance: {s}")),
        }
    }
}

#[derive(Debug)]
struct Args {
    data_json: Option<PathBuf>,
    algorithm: Algorithm,
    distance: DistanceArg,
    k: usize,
    num_partitions: u32,
    partitions_to_search: u32,
    num_blocks: u32,
    limit_train: Option<usize>,
    limit_test: Option<usize>,
    synthetic_train: usize,
    synthetic_test: usize,
    dim: usize,
    seed: u64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            data_json: None,
            algorithm: Algorithm::BruteForce,
            distance: DistanceArg::SquaredL2,
            k: 10,
            num_partitions: 100,
            partitions_to_search: 10,
            num_blocks: 8,
            limit_train: None,
            limit_test: None,
            synthetic_train: 10_000,
            synthetic_test: 200,
            dim: 64,
            seed: 42,
        }
    }
}

#[derive(Debug)]
struct BenchmarkData {
    train: Vec<Vec<f32>>,
    test: Vec<Vec<f32>>,
    gt: Vec<Vec<u32>>,
    source: String,
    dimension: usize,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    dataset: String,
    algorithm: Algorithm,
    distance: DistanceArg,
    k: usize,
    train_size: usize,
    test_size: usize,
    dimension: usize,
    build_seconds: f64,
    search_seconds: f64,
    qps: f64,
    recall_at_k: f64,
    index_rss_delta_bytes: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct JsonDataset {
    train: Vec<Vec<f32>>,
    test: Vec<Vec<f32>>,
    neighbors: Vec<Vec<u32>>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let distance = args.distance.to_measure();

    let data = if let Some(path) = &args.data_json {
        load_json_dataset(path, args.k, args.limit_train, args.limit_test)?
    } else {
        generate_synthetic_dataset(
            args.synthetic_train,
            args.synthetic_test,
            args.dim,
            args.k,
            args.seed,
        )
    };

    let before_rss = current_rss_bytes();
    let build_start = Instant::now();
    let index = build_index(
        &data.train,
        args.algorithm,
        distance,
        args.k,
        args.num_partitions,
        args.partitions_to_search,
        args.num_blocks,
    )?;
    let build_seconds = build_start.elapsed().as_secs_f64();
    let after_rss = current_rss_bytes();

    let search_start = Instant::now();
    let mut retrieved: Vec<Vec<u32>> = Vec::with_capacity(data.test.len());
    for query in &data.test {
        let result = index.search(query, args.k)?;
        retrieved.push(result.iter().map(|(idx, _)| *idx).collect());
    }
    let search_seconds = search_start.elapsed().as_secs_f64();

    let recall_at_k = average_recall_at_k(&retrieved, &data.gt, args.k);
    let qps = if search_seconds > 0.0 {
        data.test.len() as f64 / search_seconds
    } else {
        0.0
    };
    let index_rss_delta_bytes = match (before_rss, after_rss) {
        (Some(before), Some(after)) if after >= before => Some(after - before),
        _ => None,
    };

    let report = BenchmarkReport {
        dataset: data.source,
        algorithm: args.algorithm,
        distance: args.distance,
        k: args.k,
        train_size: data.train.len(),
        test_size: data.test.len(),
        dimension: data.dimension,
        build_seconds,
        search_seconds,
        qps,
        recall_at_k,
        index_rss_delta_bytes,
    };

    println!("=== ANN-Benchmarks style report ===");
    println!("dataset: {}", report.dataset);
    println!("algorithm: {:?}", report.algorithm);
    println!("distance: {:?}", report.distance);
    println!("k: {}", report.k);
    println!(
        "train/test/dim: {}/{}/{}",
        report.train_size, report.test_size, report.dimension
    );
    println!("build_seconds: {:.6}", report.build_seconds);
    println!("search_seconds: {:.6}", report.search_seconds);
    println!("qps: {:.2}", report.qps);
    println!("recall@{}: {:.6}", report.k, report.recall_at_k);
    if let Some(bytes) = report.index_rss_delta_bytes {
        println!("index_rss_delta_bytes: {}", bytes);
    } else {
        println!("index_rss_delta_bytes: unavailable");
    }
    println!("json: {}", serde_json::to_string(&report)?);

    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);

    while let Some(flag) = it.next() {
        let value = it.next();
        match flag.as_str() {
            "--data-json" => {
                let v = required_value(&flag, value)?;
                args.data_json = Some(PathBuf::from(v));
            }
            "--algorithm" => {
                let v = required_value(&flag, value)?;
                args.algorithm = Algorithm::from_str(&v)?;
            }
            "--distance" => {
                let v = required_value(&flag, value)?;
                args.distance = DistanceArg::from_str(&v)?;
            }
            "--k" => {
                let v = required_value(&flag, value)?;
                args.k = v.parse()?;
            }
            "--num-partitions" => {
                let v = required_value(&flag, value)?;
                args.num_partitions = v.parse()?;
            }
            "--partitions-to-search" => {
                let v = required_value(&flag, value)?;
                args.partitions_to_search = v.parse()?;
            }
            "--num-blocks" => {
                let v = required_value(&flag, value)?;
                args.num_blocks = v.parse()?;
            }
            "--limit-train" => {
                let v = required_value(&flag, value)?;
                args.limit_train = Some(v.parse()?);
            }
            "--limit-test" => {
                let v = required_value(&flag, value)?;
                args.limit_test = Some(v.parse()?);
            }
            "--synthetic-train" => {
                let v = required_value(&flag, value)?;
                args.synthetic_train = v.parse()?;
            }
            "--synthetic-test" => {
                let v = required_value(&flag, value)?;
                args.synthetic_test = v.parse()?;
            }
            "--dim" => {
                let v = required_value(&flag, value)?;
                args.dim = v.parse()?;
            }
            "--seed" => {
                let v = required_value(&flag, value)?;
                args.seed = v.parse()?;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                return Err(format!("unknown argument: {flag}").into());
            }
        }
    }

    Ok(args)
}

fn required_value(flag: &str, value: Option<String>) -> Result<String, Box<dyn Error>> {
    value.ok_or_else(|| format!("missing value for {flag}").into())
}

fn print_help() {
    println!(
        "ann_benchmark\n\
        ANN-Benchmarks-style runner for scann-rust.\n\n\
        Usage:\n\
          cargo run --release --bin ann_benchmark -- [options]\n\n\
        Options:\n\
          --data-json <path>          JSON file with train/test/neighbors arrays\n\
          --algorithm <name>          brute-force|partitioned|hashed|tree-ah\n\
          --distance <name>           squared-l2|l2|l1|cosine|dot-product\n\
          --k <int>                   neighbors to retrieve (default: 10)\n\
          --num-partitions <int>      partition count (default: 100)\n\
          --partitions-to-search <i>  partitions searched per query (default: 10)\n\
          --num-blocks <int>          hash blocks (default: 8)\n\
          --limit-train <int>         cap train rows from input JSON\n\
          --limit-test <int>          cap test rows from input JSON\n\
          --synthetic-train <int>     train vectors for synthetic mode\n\
          --synthetic-test <int>      test vectors for synthetic mode\n\
          --dim <int>                 dimensionality for synthetic mode\n\
          --seed <int>                RNG seed for synthetic mode\n\
          --help                      print this help\n"
    );
}

fn build_index(
    train: &[Vec<f32>],
    algorithm: Algorithm,
    distance: DistanceMeasure,
    k: usize,
    num_partitions: u32,
    partitions_to_search: u32,
    num_blocks: u32,
) -> Result<Scann, Box<dyn Error>> {
    let dataset = DenseDataset::from_vecs(train.to_vec());
    let builder = ScannBuilder::new()
        .num_neighbors(k as u32)
        .distance_measure(distance);

    let index = match algorithm {
        Algorithm::BruteForce => builder.brute_force().build(dataset)?,
        Algorithm::Partitioned => builder
            .tree(num_partitions, partitions_to_search)
            .build(dataset)?,
        Algorithm::Hashed => builder.hash(num_blocks).build(dataset)?,
        Algorithm::TreeAh => builder
            .tree(num_partitions, partitions_to_search)
            .hash(num_blocks)
            .build(dataset)?,
    };
    Ok(index)
}

fn load_json_dataset(
    path: &PathBuf,
    k: usize,
    limit_train: Option<usize>,
    limit_test: Option<usize>,
) -> Result<BenchmarkData, Box<dyn Error>> {
    let raw = fs::read_to_string(path)?;
    let mut json: JsonDataset = serde_json::from_str(&raw)?;

    if let Some(n) = limit_train {
        json.train.truncate(n.min(json.train.len()));
    }
    if let Some(n) = limit_test {
        json.test.truncate(n.min(json.test.len()));
        json.neighbors.truncate(n.min(json.neighbors.len()));
    }

    if json.train.is_empty() || json.test.is_empty() || json.neighbors.is_empty() {
        return Err("dataset JSON must include non-empty train/test/neighbors".into());
    }

    for row in &json.neighbors {
        if row.len() < k {
            return Err(format!("neighbors rows must have at least {k} entries").into());
        }
    }

    let gt: Vec<Vec<u32>> = json
        .neighbors
        .into_iter()
        .take(json.test.len())
        .map(|row| row.into_iter().take(k).collect())
        .collect();

    let dim = json.train[0].len();

    Ok(BenchmarkData {
        train: json.train,
        test: json.test,
        gt,
        source: path.display().to_string(),
        dimension: dim,
    })
}

fn generate_synthetic_dataset(
    train_size: usize,
    test_size: usize,
    dim: usize,
    k: usize,
    seed: u64,
) -> BenchmarkData {
    let mut rng = StdRng::seed_from_u64(seed);
    let train: Vec<Vec<f32>> = (0..train_size)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();
    let test: Vec<Vec<f32>> = (0..test_size)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();
    let gt = exact_ground_truth(&train, &test, k);

    BenchmarkData {
        train,
        test,
        gt,
        source: format!("synthetic_n{}_q{}_d{}", train_size, test_size, dim),
        dimension: dim,
    }
}

fn exact_ground_truth(train: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|query| {
            let mut distances: Vec<(u32, f32)> = train
                .iter()
                .enumerate()
                .map(|(idx, point)| (idx as u32, squared_l2(query, point)))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            distances.into_iter().take(k).map(|(idx, _)| idx).collect()
        })
        .collect()
}

fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn average_recall_at_k(retrieved: &[Vec<u32>], ground_truth: &[Vec<u32>], k: usize) -> f64 {
    if retrieved.is_empty() || ground_truth.is_empty() || k == 0 {
        return 0.0;
    }

    let n = retrieved.len().min(ground_truth.len());
    let mut total = 0.0f64;

    for i in 0..n {
        let mut hits = 0usize;
        for id in retrieved[i].iter().take(k) {
            if ground_truth[i].iter().take(k).any(|gt_id| gt_id == id) {
                hits += 1;
            }
        }
        total += hits as f64 / k as f64;
    }

    total / n as f64
}

fn current_rss_bytes() -> Option<u64> {
    let statm = fs::read_to_string("/proc/self/statm").ok()?;
    let mut fields = statm.split_whitespace();
    let _size_pages = fields.next()?;
    let rss_pages: u64 = fields.next()?.parse().ok()?;
    Some(rss_pages * 4096)
}

#[cfg(test)]
mod tests {
    use super::average_recall_at_k;

    #[test]
    fn recall_at_k_basic() {
        let retrieved = vec![vec![1, 2, 3], vec![5, 7, 9]];
        let gt = vec![vec![1, 4, 3], vec![5, 6, 7]];
        let recall = average_recall_at_k(&retrieved, &gt, 3);
        assert!((recall - (2.0 / 3.0)).abs() < 1e-6);
    }
}
