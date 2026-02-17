//! Asymmetric hashing for fast approximate scoring.
//!
//! This module implements product quantization and asymmetric hashing
//! for efficient approximate distance computation.

mod codebook;
mod hasher;
mod lut;
mod lut16;
pub mod lut16_simd;
mod stacked;

pub use codebook::{Codebook, CodebookConfig};
pub use hasher::{AsymmetricHasher, AsymmetricHasherConfig};
pub use lut::{LookupTable, LutFormat};
pub use lut16::{Lut16Config, Lut16Table, Lut16LookupTables, PackedCodes4Bit};
pub use lut16_simd::Lut16SimdTables;
pub use stacked::{StackedQuantizer, StackedQuantizerConfig, AdditiveQuantizer};
