//! Quantization module for memory-efficient vector storage.
//!
//! This module provides various quantization methods including:
//! - Scalar quantization (Int8, Int4)
//! - FP8 (8-bit floating point)
//! - BFloat16 (Brain floating point)

mod scalar;
mod fp8;
mod bfloat16;

pub use scalar::{ScalarQuantizer, ScalarQuantizerConfig, QuantizedDataset};
pub use fp8::{Fp8Value, Fp8Quantizer, Fp8Config};
pub use bfloat16::{BFloat16Dataset, bf16_to_f32, f32_to_bf16};

use crate::data_format::{Dataset, DenseDataset};
use serde::{Deserialize, Serialize};

/// Quantization type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Default)]
pub enum QuantizationType {
    /// No quantization (full precision).
    #[default]
    None,
    /// 8-bit signed integer quantization.
    Int8,
    /// 4-bit signed integer quantization.
    Int4,
    /// 8-bit floating point.
    Fp8,
    /// 16-bit brain floating point.
    BFloat16,
}


/// Trait for quantizers.
pub trait Quantizer: Send + Sync {
    /// The quantized value type.
    type QuantizedType: Copy + Send + Sync;

    /// Quantize a single float value.
    fn quantize_value(&self, value: f32) -> Self::QuantizedType;

    /// Dequantize a single value.
    fn dequantize_value(&self, quantized: Self::QuantizedType) -> f32;

    /// Quantize a vector.
    fn quantize(&self, values: &[f32]) -> Vec<Self::QuantizedType> {
        values.iter().map(|&v| self.quantize_value(v)).collect()
    }

    /// Dequantize a vector.
    fn dequantize(&self, quantized: &[Self::QuantizedType]) -> Vec<f32> {
        quantized.iter().map(|&q| self.dequantize_value(q)).collect()
    }

    /// Get the number of bits used for quantization.
    fn bits(&self) -> usize;
}

/// Statistics for calibrating quantization.
#[derive(Debug, Clone, Default)]
pub struct QuantizationStats {
    /// Minimum value observed.
    pub min_value: f32,
    /// Maximum value observed.
    pub max_value: f32,
    /// Mean value.
    pub mean: f32,
    /// Standard deviation.
    pub std_dev: f32,
}

impl QuantizationStats {
    /// Compute statistics from a dataset.
    pub fn from_dataset(dataset: &DenseDataset<f32>) -> Self {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut count = 0u64;

        for i in 0..dataset.size() {
            if let Some(dp) = dataset.get(i as u32) {
                for &val in dp.values() {
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                    sum += val as f64;
                    sum_sq += (val as f64) * (val as f64);
                    count += 1;
                }
            }
        }

        let mean = if count > 0 { (sum / count as f64) as f32 } else { 0.0 };
        let variance = if count > 1 {
            ((sum_sq - sum * sum / count as f64) / (count - 1) as f64) as f32
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        Self {
            min_value: min_val,
            max_value: max_val,
            mean,
            std_dev,
        }
    }

    /// Compute statistics from vectors.
    pub fn from_vecs(data: &[Vec<f32>]) -> Self {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut count = 0u64;

        for vec in data {
            for &val in vec {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
                sum += val as f64;
                sum_sq += (val as f64) * (val as f64);
                count += 1;
            }
        }

        let mean = if count > 0 { (sum / count as f64) as f32 } else { 0.0 };
        let variance = if count > 1 {
            ((sum_sq - sum * sum / count as f64) / (count - 1) as f64) as f32
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        Self {
            min_value: min_val,
            max_value: max_val,
            mean,
            std_dev,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_stats() {
        let data = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];

        let stats = QuantizationStats::from_vecs(&data);

        assert_eq!(stats.min_value, 1.0);
        assert_eq!(stats.max_value, 6.0);
        assert!((stats.mean - 3.5).abs() < 0.01);
    }
}
