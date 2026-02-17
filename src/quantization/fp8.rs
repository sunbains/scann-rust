//! FP8 (8-bit floating point) quantization.
//!
//! Provides E4M3 and E5M2 floating point formats.

use crate::quantization::Quantizer;
use serde::{Deserialize, Serialize};

/// FP8 format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum Fp8Format {
    /// 4-bit exponent, 3-bit mantissa (E4M3).
    /// Better for inference (larger range).
    #[default]
    E4M3,
    /// 5-bit exponent, 2-bit mantissa (E5M2).
    /// Better for training (more precision in small values).
    E5M2,
}


/// Configuration for FP8 quantizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fp8Config {
    /// FP8 format to use.
    pub format: Fp8Format,
    /// Scale factor for the values.
    pub scale: f32,
}

impl Default for Fp8Config {
    fn default() -> Self {
        Self {
            format: Fp8Format::E4M3,
            scale: 1.0,
        }
    }
}

impl Fp8Config {
    /// Create E4M3 config.
    pub fn e4m3() -> Self {
        Self {
            format: Fp8Format::E4M3,
            scale: 1.0,
        }
    }

    /// Create E5M2 config.
    pub fn e5m2() -> Self {
        Self {
            format: Fp8Format::E5M2,
            scale: 1.0,
        }
    }

    /// Set the scale factor.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }
}

/// 8-bit floating point value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fp8Value(pub u8);

impl Fp8Value {
    /// Create from raw bits.
    pub fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Get raw bits.
    pub fn to_bits(self) -> u8 {
        self.0
    }

    /// Convert from f32 using E4M3 format.
    pub fn from_f32_e4m3(value: f32) -> Self {
        // E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits
        // Exponent bias: 7
        // Max value: 448, Min positive: 2^-9

        if value == 0.0 {
            return Self(0);
        }

        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x7FFFFF;

        // Handle special cases
        if exp == 0xFF {
            // Infinity or NaN -> max value
            return Self(((sign as u8) << 7) | 0x7E);
        }

        // Compute E4M3 exponent (bias 7)
        let fp8_exp = exp - 127 + 7;

        if fp8_exp <= 0 {
            // Underflow to zero or subnormal
            return Self((sign as u8) << 7);
        }

        if fp8_exp >= 15 {
            // Overflow to max
            return Self(((sign as u8) << 7) | 0x7E);
        }

        // Round mantissa to 3 bits
        let fp8_mantissa = ((mantissa >> 20) + ((mantissa >> 19) & 1)) & 0x7;

        Self(((sign as u8) << 7) | ((fp8_exp as u8) << 3) | (fp8_mantissa as u8))
    }

    /// Convert to f32 from E4M3 format.
    pub fn to_f32_e4m3(self) -> f32 {
        let bits = self.0;

        let sign = (bits >> 7) & 1;
        let exp = ((bits >> 3) & 0xF) as i32;
        let mantissa = (bits & 0x7) as u32;

        if exp == 0 && mantissa == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }

        // Convert to f32
        let fp32_exp = if exp == 0 {
            // Subnormal
            126 - 7
        } else {
            exp - 7 + 127
        };

        let fp32_mantissa = mantissa << 20;
        let fp32_bits = ((sign as u32) << 31) | ((fp32_exp as u32) << 23) | fp32_mantissa;

        f32::from_bits(fp32_bits)
    }

    /// Convert from f32 using E5M2 format.
    pub fn from_f32_e5m2(value: f32) -> Self {
        // E5M2: 1 sign bit, 5 exponent bits, 2 mantissa bits
        // Exponent bias: 15

        if value == 0.0 {
            return Self(0);
        }

        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x7FFFFF;

        // Handle special cases
        if exp == 0xFF {
            // Infinity or NaN
            return Self(((sign as u8) << 7) | 0x7C);
        }

        // Compute E5M2 exponent (bias 15)
        let fp8_exp = exp - 127 + 15;

        if fp8_exp <= 0 {
            // Underflow
            return Self((sign as u8) << 7);
        }

        if fp8_exp >= 31 {
            // Overflow
            return Self(((sign as u8) << 7) | 0x7C);
        }

        // Round mantissa to 2 bits
        let fp8_mantissa = ((mantissa >> 21) + ((mantissa >> 20) & 1)) & 0x3;

        Self(((sign as u8) << 7) | ((fp8_exp as u8) << 2) | (fp8_mantissa as u8))
    }

    /// Convert to f32 from E5M2 format.
    pub fn to_f32_e5m2(self) -> f32 {
        let bits = self.0;

        let sign = (bits >> 7) & 1;
        let exp = ((bits >> 2) & 0x1F) as i32;
        let mantissa = (bits & 0x3) as u32;

        if exp == 0 && mantissa == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }

        // Convert to f32
        let fp32_exp = if exp == 0 {
            126 - 15
        } else {
            exp - 15 + 127
        };

        let fp32_mantissa = mantissa << 21;
        let fp32_bits = ((sign as u32) << 31) | ((fp32_exp as u32) << 23) | fp32_mantissa;

        f32::from_bits(fp32_bits)
    }
}

/// FP8 quantizer.
#[derive(Clone, Debug)]
pub struct Fp8Quantizer {
    config: Fp8Config,
}

impl Fp8Quantizer {
    /// Create a new FP8 quantizer.
    pub fn new(config: Fp8Config) -> Self {
        Self { config }
    }

    /// Create E4M3 quantizer.
    pub fn e4m3() -> Self {
        Self::new(Fp8Config::e4m3())
    }

    /// Create E5M2 quantizer.
    pub fn e5m2() -> Self {
        Self::new(Fp8Config::e5m2())
    }

    /// Get the format.
    pub fn format(&self) -> Fp8Format {
        self.config.format
    }

    /// Get the scale.
    pub fn scale(&self) -> f32 {
        self.config.scale
    }

    /// Set the scale based on data range.
    pub fn calibrate_scale(&mut self, max_abs_value: f32) {
        let fp8_max = match self.config.format {
            Fp8Format::E4M3 => 448.0,
            Fp8Format::E5M2 => 57344.0,
        };
        self.config.scale = fp8_max / max_abs_value.max(1e-10);
    }
}

impl Quantizer for Fp8Quantizer {
    type QuantizedType = Fp8Value;

    fn quantize_value(&self, value: f32) -> Fp8Value {
        let scaled = value * self.config.scale;
        match self.config.format {
            Fp8Format::E4M3 => Fp8Value::from_f32_e4m3(scaled),
            Fp8Format::E5M2 => Fp8Value::from_f32_e5m2(scaled),
        }
    }

    fn dequantize_value(&self, quantized: Fp8Value) -> f32 {
        let value = match self.config.format {
            Fp8Format::E4M3 => quantized.to_f32_e4m3(),
            Fp8Format::E5M2 => quantized.to_f32_e5m2(),
        };
        value / self.config.scale
    }

    fn bits(&self) -> usize {
        8
    }
}

/// Vectorized FP8 operations.
pub mod simd {
    use super::*;

    /// Convert a slice of f32 to FP8 E4M3.
    pub fn f32_to_fp8_e4m3(input: &[f32], output: &mut [Fp8Value]) {
        for (i, &val) in input.iter().enumerate() {
            if i < output.len() {
                output[i] = Fp8Value::from_f32_e4m3(val);
            }
        }
    }

    /// Convert a slice of FP8 E4M3 to f32.
    pub fn fp8_e4m3_to_f32(input: &[Fp8Value], output: &mut [f32]) {
        for (i, &val) in input.iter().enumerate() {
            if i < output.len() {
                output[i] = val.to_f32_e4m3();
            }
        }
    }

    /// Compute dot product between f32 query and FP8 database vector.
    pub fn dot_product_f32_fp8_e4m3(query: &[f32], database: &[Fp8Value]) -> f32 {
        let mut sum = 0.0f32;
        for (i, &q) in query.iter().enumerate() {
            if i < database.len() {
                sum += q * database[i].to_f32_e4m3();
            }
        }
        sum
    }

    /// Compute squared L2 distance between f32 query and FP8 database vector.
    pub fn squared_l2_f32_fp8_e4m3(query: &[f32], database: &[Fp8Value]) -> f32 {
        let mut sum = 0.0f32;
        for (i, &q) in query.iter().enumerate() {
            if i < database.len() {
                let diff = q - database[i].to_f32_e4m3();
                sum += diff * diff;
            }
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp8_e4m3_roundtrip() {
        let values = [0.0, 1.0, -1.0, 0.5, 2.0, 100.0, -0.1];

        for &val in &values {
            let fp8 = Fp8Value::from_f32_e4m3(val);
            let recovered = fp8.to_f32_e4m3();

            // Should be approximately equal (some precision loss expected)
            if val != 0.0 {
                let rel_error = ((val - recovered) / val).abs();
                assert!(rel_error < 0.2, "val={}, recovered={}", val, recovered);
            }
        }
    }

    #[test]
    fn test_fp8_e5m2_roundtrip() {
        // E5M2 has 5 exponent bits and 2 mantissa bits
        // Max representable is ~57344, but precision is very low
        // Only test values within reasonable range and precision
        let values = [0.0, 1.0, -1.0, 0.5, 2.0, 4.0, -0.125];

        for &val in &values {
            let fp8 = Fp8Value::from_f32_e5m2(val);
            let recovered = fp8.to_f32_e5m2();

            if val.abs() > 1e-6 {
                let rel_error = ((val - recovered) / val).abs();
                // E5M2 has only 2 mantissa bits so 25% relative error is expected
                assert!(rel_error < 0.5, "val={}, recovered={}", val, recovered);
            }
        }
    }

    #[test]
    fn test_fp8_quantizer() {
        let quantizer = Fp8Quantizer::e4m3();

        let values = [1.0f32, 2.0, 3.0, 4.0];
        let quantized: Vec<_> = values.iter().map(|&v| quantizer.quantize_value(v)).collect();
        let dequantized: Vec<_> = quantized.iter().map(|&q| quantizer.dequantize_value(q)).collect();

        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.5);
        }
    }

    #[test]
    fn test_simd_operations() {
        let query = vec![1.0f32, 2.0, 3.0, 4.0];
        let database: Vec<Fp8Value> = query.iter().map(|&v| Fp8Value::from_f32_e4m3(v)).collect();

        let dot = simd::dot_product_f32_fp8_e4m3(&query, &database);
        let expected: f32 = query.iter().map(|x| x * x).sum();

        assert!((dot - expected).abs() < 1.0);

        let dist = simd::squared_l2_f32_fp8_e4m3(&query, &database);
        // Distance to self should be small
        assert!(dist < 1.0);
    }
}
