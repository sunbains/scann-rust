//! Core type definitions for ScaNN.
//!
//! This module contains the fundamental type aliases and traits used throughout the library.

use std::ops::{Add, Mul, Sub, Div};
use num_traits::{Float, NumCast, Zero, One};

/// Index type for datapoints in a dataset.
/// Can represent up to 4 billion datapoints with u32.
pub type DatapointIndex = u32;

/// Index type for dimensions within a datapoint.
/// Uses u64 to support very high-dimensional data.
pub type DimensionIndex = u64;

/// A nearest neighbor result: (index, distance).
pub type NNResultPair = (DatapointIndex, f32);

/// Vector of nearest neighbor results.
pub type NNResultsVector = Vec<NNResultPair>;

/// Trait for numeric types that can be used as datapoint values.
pub trait DatapointValue:
    Copy
    + Clone
    + Default
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Zero
    + One
    + NumCast
    + Send
    + Sync
    + 'static
{
    /// Convert to f32 for distance computations.
    fn to_f32(self) -> f32;

    /// Create from f32.
    fn from_f32(v: f32) -> Self;

    /// Check if this is a floating-point type.
    fn is_floating() -> bool;
}

impl DatapointValue for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }

    #[inline]
    fn is_floating() -> bool {
        true
    }
}

impl DatapointValue for f64 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v as f64
    }

    #[inline]
    fn is_floating() -> bool {
        true
    }
}

impl DatapointValue for i8 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v as i8
    }

    #[inline]
    fn is_floating() -> bool {
        false
    }
}

impl DatapointValue for u8 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v as u8
    }

    #[inline]
    fn is_floating() -> bool {
        false
    }
}

impl DatapointValue for i16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v as i16
    }

    #[inline]
    fn is_floating() -> bool {
        false
    }
}

impl DatapointValue for u16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v as u16
    }

    #[inline]
    fn is_floating() -> bool {
        false
    }
}

impl DatapointValue for i32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v as i32
    }

    #[inline]
    fn is_floating() -> bool {
        false
    }
}

impl DatapointValue for u32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v as u32
    }

    #[inline]
    fn is_floating() -> bool {
        false
    }
}

impl DatapointValue for i64 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v as i64
    }

    #[inline]
    fn is_floating() -> bool {
        false
    }
}

impl DatapointValue for u64 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v as u64
    }

    #[inline]
    fn is_floating() -> bool {
        false
    }
}

/// Trait for floating-point types used in distance computations.
pub trait FloatValue: DatapointValue + Float {
    /// Square root.
    fn sqrt_val(self) -> Self;

    /// Absolute value.
    fn abs_val(self) -> Self;

    /// Maximum of two values.
    fn max_val(self, other: Self) -> Self;

    /// Minimum of two values.
    fn min_val(self, other: Self) -> Self;
}

impl FloatValue for f32 {
    #[inline]
    fn sqrt_val(self) -> Self {
        self.sqrt()
    }

    #[inline]
    fn abs_val(self) -> Self {
        self.abs()
    }

    #[inline]
    fn max_val(self, other: Self) -> Self {
        self.max(other)
    }

    #[inline]
    fn min_val(self, other: Self) -> Self {
        self.min(other)
    }
}

impl FloatValue for f64 {
    #[inline]
    fn sqrt_val(self) -> Self {
        self.sqrt()
    }

    #[inline]
    fn abs_val(self) -> Self {
        self.abs()
    }

    #[inline]
    fn max_val(self, other: Self) -> Self {
        self.max(other)
    }

    #[inline]
    fn min_val(self, other: Self) -> Self {
        self.min(other)
    }
}

/// Span type for immutable slices with lifetime.
pub type ConstSpan<'a, T> = &'a [T];

/// Span type for mutable slices with lifetime.
pub type MutableSpan<'a, T> = &'a mut [T];

/// Memory alignment for SIMD operations.
pub const SIMD_ALIGNMENT: usize = 64;

/// Check if a pointer is aligned for SIMD operations.
#[inline]
pub fn is_simd_aligned<T>(ptr: *const T) -> bool {
    (ptr as usize) % SIMD_ALIGNMENT == 0
}

/// Round up to the nearest multiple of alignment.
#[inline]
pub const fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datapoint_value_f32() {
        let v: f32 = 3.14;
        assert_eq!(v.to_f32(), 3.14);
        assert_eq!(f32::from_f32(2.71), 2.71);
        assert!(f32::is_floating());
    }

    #[test]
    fn test_datapoint_value_i32() {
        let v: i32 = 42;
        assert_eq!(v.to_f32(), 42.0);
        assert_eq!(i32::from_f32(42.9), 42);
        assert!(!i32::is_floating());
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
    }
}
