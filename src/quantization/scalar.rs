//! Scalar quantization implementation.
//!
//! Provides Int8 and Int4 quantization for memory-efficient storage.

use crate::data_format::{Dataset, DenseDataset};
use crate::error::{Result, ScannError};
use crate::quantization::{Quantizer, QuantizationStats};
use crate::types::DatapointIndex;
use serde::{Deserialize, Serialize};

/// Configuration for scalar quantizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizerConfig {
    /// Number of bits for quantization (4 or 8).
    pub bits: usize,
    /// Minimum value for quantization range.
    pub min_value: Option<f32>,
    /// Maximum value for quantization range.
    pub max_value: Option<f32>,
    /// Use symmetric quantization around zero.
    pub symmetric: bool,
    /// Number of standard deviations for range estimation.
    pub num_std_devs: f32,
}

impl Default for ScalarQuantizerConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            min_value: None,
            max_value: None,
            symmetric: false,
            num_std_devs: 3.0,
        }
    }
}

impl ScalarQuantizerConfig {
    /// Create a new Int8 quantizer config.
    pub fn int8() -> Self {
        Self {
            bits: 8,
            ..Default::default()
        }
    }

    /// Create a new Int4 quantizer config.
    pub fn int4() -> Self {
        Self {
            bits: 4,
            ..Default::default()
        }
    }

    /// Set explicit min/max range.
    pub fn with_range(mut self, min: f32, max: f32) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }

    /// Use symmetric quantization.
    pub fn symmetric(mut self) -> Self {
        self.symmetric = true;
        self
    }
}

/// Scalar quantizer for converting f32 to Int8/Int4.
#[derive(Clone, Debug)]
pub struct ScalarQuantizer {
    config: ScalarQuantizerConfig,
    /// Minimum value of the quantization range.
    min_value: f32,
    /// Maximum value of the quantization range.
    max_value: f32,
    /// Scale factor for quantization.
    scale: f32,
    /// Inverse scale for dequantization.
    inv_scale: f32,
    /// Zero point offset.
    zero_point: i32,
    /// Number of quantization levels.
    num_levels: i32,
}

impl ScalarQuantizer {
    /// Create a new scalar quantizer with the given config.
    pub fn new(config: ScalarQuantizerConfig) -> Self {
        let num_levels = (1 << config.bits) - 1;
        Self {
            config,
            min_value: 0.0,
            max_value: 1.0,
            scale: 1.0,
            inv_scale: 1.0,
            zero_point: 0,
            num_levels,
        }
    }

    /// Calibrate the quantizer from data statistics.
    pub fn calibrate(&mut self, stats: &QuantizationStats) {
        if let (Some(min), Some(max)) = (self.config.min_value, self.config.max_value) {
            self.min_value = min;
            self.max_value = max;
        } else if self.config.symmetric {
            // Symmetric quantization around zero
            let abs_max = stats.min_value.abs().max(stats.max_value.abs());
            self.min_value = -abs_max;
            self.max_value = abs_max;
        } else {
            // Use statistics with std dev clipping
            let range = self.config.num_std_devs * stats.std_dev;
            self.min_value = (stats.mean - range).max(stats.min_value);
            self.max_value = (stats.mean + range).min(stats.max_value);
        }

        // Compute scale and zero point
        let range = self.max_value - self.min_value;
        if range > 1e-10 {
            self.scale = range / self.num_levels as f32;
            self.inv_scale = self.num_levels as f32 / range;
            self.zero_point = (-self.min_value * self.inv_scale).round() as i32;
        } else {
            self.scale = 1.0;
            self.inv_scale = 1.0;
            self.zero_point = 0;
        }
    }

    /// Calibrate from a dataset.
    pub fn calibrate_from_dataset(&mut self, dataset: &DenseDataset<f32>) {
        let stats = QuantizationStats::from_dataset(dataset);
        self.calibrate(&stats);
    }

    /// Get the scale factor.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get the zero point.
    pub fn zero_point(&self) -> i32 {
        self.zero_point
    }

    /// Get min value.
    pub fn min_value(&self) -> f32 {
        self.min_value
    }

    /// Get max value.
    pub fn max_value(&self) -> f32 {
        self.max_value
    }
}

impl Quantizer for ScalarQuantizer {
    type QuantizedType = i8;

    fn quantize_value(&self, value: f32) -> i8 {
        let clamped = value.clamp(self.min_value, self.max_value);
        let quantized = ((clamped - self.min_value) * self.inv_scale).round() as i32;
        quantized.clamp(0, self.num_levels) as i8
    }

    fn dequantize_value(&self, quantized: i8) -> f32 {
        // Treat i8 as unsigned byte (values 0-255 stored as -128 to 127)
        let unsigned_val = quantized as u8;
        unsigned_val as f32 * self.scale + self.min_value
    }

    fn bits(&self) -> usize {
        self.config.bits
    }
}

/// A dataset of quantized vectors.
pub struct QuantizedDataset {
    /// Quantized data storage.
    data: Vec<i8>,
    /// Number of datapoints.
    num_points: usize,
    /// Dimensionality.
    dimensionality: usize,
    /// Stride between datapoints.
    stride: usize,
    /// The quantizer used.
    quantizer: ScalarQuantizer,
}

impl QuantizedDataset {
    /// Create from a dense dataset.
    pub fn from_dataset(
        dataset: &DenseDataset<f32>,
        mut quantizer: ScalarQuantizer,
    ) -> Result<Self> {
        if dataset.is_empty() {
            return Err(ScannError::invalid_argument("Cannot quantize empty dataset"));
        }

        // Calibrate quantizer
        quantizer.calibrate_from_dataset(dataset);

        let num_points = dataset.size();
        let dimensionality = dataset.dimensionality() as usize;
        let stride = dimensionality;

        let mut data = Vec::with_capacity(num_points * stride);

        for i in 0..num_points {
            if let Some(dp) = dataset.get(i as u32) {
                let quantized = quantizer.quantize(dp.values());
                data.extend(quantized);
            }
        }

        Ok(Self {
            data,
            num_points,
            dimensionality,
            stride,
            quantizer,
        })
    }

    /// Get a quantized datapoint.
    pub fn get_quantized(&self, index: DatapointIndex) -> Option<&[i8]> {
        let idx = index as usize;
        if idx >= self.num_points {
            return None;
        }

        let offset = idx * self.stride;
        Some(&self.data[offset..offset + self.dimensionality])
    }

    /// Get a dequantized datapoint.
    pub fn get_dequantized(&self, index: DatapointIndex) -> Option<Vec<f32>> {
        self.get_quantized(index)
            .map(|q| self.quantizer.dequantize(q))
    }

    /// Get the number of datapoints.
    pub fn size(&self) -> usize {
        self.num_points
    }

    /// Get the dimensionality.
    pub fn dimensionality(&self) -> usize {
        self.dimensionality
    }

    /// Get the quantizer.
    pub fn quantizer(&self) -> &ScalarQuantizer {
        &self.quantizer
    }

    /// Compute squared L2 distance between a query and a quantized point.
    pub fn squared_l2_distance(&self, query: &[f32], index: DatapointIndex) -> Option<f32> {
        let quantized = self.get_quantized(index)?;

        let mut sum = 0.0f32;
        for (i, &q) in quantized.iter().enumerate() {
            let dequant = self.quantizer.dequantize_value(q);
            let diff = query.get(i).copied().unwrap_or(0.0) - dequant;
            sum += diff * diff;
        }

        Some(sum)
    }

    /// Compute dot product between a query and a quantized point.
    pub fn dot_product(&self, query: &[f32], index: DatapointIndex) -> Option<f32> {
        let quantized = self.get_quantized(index)?;

        let mut sum = 0.0f32;
        for (i, &q) in quantized.iter().enumerate() {
            let dequant = self.quantizer.dequantize_value(q);
            sum += query.get(i).copied().unwrap_or(0.0) * dequant;
        }

        Some(sum)
    }

    /// Precompute query-related values for faster distance computation.
    pub fn precompute_query(&self, query: &[f32]) -> PrecomputedQuery {
        PrecomputedQuery::new(query, &self.quantizer)
    }

    /// Get raw data.
    pub fn raw_data(&self) -> &[i8] {
        &self.data
    }
}

/// Precomputed values for a query to speed up distance computations.
pub struct PrecomputedQuery {
    /// Query vector.
    query: Vec<f32>,
    /// Squared norm of query.
    query_sq_norm: f32,
    /// Lookup table for dequantization (optional optimization).
    dequant_lut: Vec<f32>,
}

impl PrecomputedQuery {
    /// Create precomputed query values.
    pub fn new(query: &[f32], quantizer: &ScalarQuantizer) -> Self {
        let query_sq_norm: f32 = query.iter().map(|x| x * x).sum();

        // Build dequantization lookup table
        let num_levels = 256;
        let dequant_lut: Vec<f32> = (0..num_levels)
            .map(|i| quantizer.dequantize_value(i as i8))
            .collect();

        Self {
            query: query.to_vec(),
            query_sq_norm,
            dequant_lut,
        }
    }

    /// Compute squared L2 distance using precomputed values.
    pub fn squared_l2_distance(&self, quantized: &[i8]) -> f32 {
        let mut dot = 0.0f32;
        let mut db_sq_norm = 0.0f32;

        for (i, &q) in quantized.iter().enumerate() {
            let idx = (q as u8) as usize;
            let dequant = self.dequant_lut.get(idx).copied().unwrap_or(0.0);

            if i < self.query.len() {
                dot += self.query[i] * dequant;
            }
            db_sq_norm += dequant * dequant;
        }

        // ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        self.query_sq_norm + db_sq_norm - 2.0 * dot
    }
}

/// Int4 packed storage (2 values per byte).
pub struct Int4PackedData {
    data: Vec<u8>,
    len: usize,
}

impl Int4PackedData {
    /// Create from i8 values (assuming values are in range -8 to 7).
    pub fn from_int8(values: &[i8]) -> Self {
        let mut data = Vec::with_capacity((values.len() + 1) / 2);

        for chunk in values.chunks(2) {
            let lo = (chunk[0] + 8) as u8 & 0x0F;
            let hi = if chunk.len() > 1 {
                ((chunk[1] + 8) as u8 & 0x0F) << 4
            } else {
                0
            };
            data.push(lo | hi);
        }

        Self {
            data,
            len: values.len(),
        }
    }

    /// Get a value at index.
    pub fn get(&self, index: usize) -> Option<i8> {
        if index >= self.len {
            return None;
        }

        let byte_idx = index / 2;
        let byte = self.data.get(byte_idx)?;

        let nibble = if index % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };

        Some(nibble as i8 - 8)
    }

    /// Get length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Unpack to i8 values.
    pub fn unpack(&self) -> Vec<i8> {
        (0..self.len).filter_map(|i| self.get(i)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_quantizer_basic() {
        let mut quantizer = ScalarQuantizer::new(ScalarQuantizerConfig::int8());
        let stats = QuantizationStats {
            min_value: -1.0,
            max_value: 1.0,
            mean: 0.0,
            std_dev: 0.5,
        };
        quantizer.calibrate(&stats);

        // Test round-trip (note: default quantizer has narrow range,
        // use higher tolerance as this is just checking basic functionality)
        let original = 0.5f32;
        let quantized = quantizer.quantize_value(original);
        let dequantized = quantizer.dequantize_value(quantized);

        // Int8 quantization has limited precision (255 levels over the range)
        // With range [-1, 1] and 255 levels, step size is ~0.008
        assert!((original - dequantized).abs() < 0.02);
    }

    #[test]
    fn test_quantized_dataset() {
        let data = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![-1.0, 0.0, 1.0],
        ];
        let dataset = DenseDataset::from_vecs(data);

        let quantizer = ScalarQuantizer::new(ScalarQuantizerConfig::int8());
        let qdata = QuantizedDataset::from_dataset(&dataset, quantizer).unwrap();

        assert_eq!(qdata.size(), 3);
        assert_eq!(qdata.dimensionality(), 3);

        // Check dequantization approximately preserves values
        // The range is [-1, 6] so step size is 7/255 ≈ 0.027
        let dequant = qdata.get_dequantized(1).unwrap();
        assert!((dequant[0] - 4.0).abs() < 1.0, "got {} expected ~4.0", dequant[0]);
        assert!((dequant[1] - 5.0).abs() < 1.0, "got {} expected ~5.0", dequant[1]);
        assert!((dequant[2] - 6.0).abs() < 1.0, "got {} expected ~6.0", dequant[2]);
    }

    #[test]
    fn test_int4_packing() {
        let values: Vec<i8> = vec![-7, 3, 0, 7, -8, 5];
        let packed = Int4PackedData::from_int8(&values);

        assert_eq!(packed.len(), 6);

        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(packed.get(i), Some(expected));
        }

        let unpacked = packed.unpack();
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_precomputed_query() {
        let mut quantizer = ScalarQuantizer::new(ScalarQuantizerConfig::int8());
        let stats = QuantizationStats {
            min_value: 0.0,
            max_value: 10.0,
            mean: 5.0,
            std_dev: 2.0,
        };
        quantizer.calibrate(&stats);

        let query = vec![1.0f32, 2.0, 3.0];
        let precomputed = PrecomputedQuery::new(&query, &quantizer);

        let quantized = quantizer.quantize(&[1.0, 2.0, 3.0]);
        let dist = precomputed.squared_l2_distance(&quantized);

        // Distance should be small (approximately 0)
        assert!(dist < 0.5);
    }
}
