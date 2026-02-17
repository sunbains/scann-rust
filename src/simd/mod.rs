//! SIMD abstraction layer for ScaNN.
//!
//! This module provides a unified interface for SIMD operations across different
//! architectures. It supports:
//! - Portable SIMD via the `wide` crate (default)
//! - AVX2 optimizations on x86_64 via `std::arch`
//!
//! # Architecture
//!
//! The module is organized into layers:
//! - `traits`: Core SIMD traits that define the interface
//! - `portable`: Portable implementations using the `wide` crate
//! - `x86`: x86_64-specific implementations using AVX2 intrinsics
//! - `dispatch`: Runtime CPU feature detection and dispatch
//!
//! # Usage
//!
//! ```rust,ignore
//! use scann::simd::dispatch;
//!
//! // The dispatch module automatically selects the best implementation
//! let result = dispatch::dot_product_f32(&a, &b);
//! ```

pub mod traits;
pub mod portable;
#[cfg(target_arch = "x86_64")]
pub mod x86;
pub mod dispatch;

#[cfg(test)]
mod tests;

// Re-export commonly used types
pub use traits::{SimdVector, SimdF32, SimdI8, SimdI16, SimdI32, SimdU8, SimdAdd, SimdSub, SimdMul, SimdHorizontal};
pub use dispatch::{
    dot_product_f32, squared_l2_f32, horizontal_sum_f32, l1_distance_f32,
    simd_support_level, SimdSupportLevel,
    lut16_distances_batch, one_to_many_dot_product_f32, one_to_many_squared_l2_f32,
};
pub use portable::{PortableF32x8, PortableI32x8, PortableU8x16, PortableI8x16};
