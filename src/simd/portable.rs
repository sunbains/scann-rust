//! Portable SIMD implementations using the `wide` crate.
//!
//! These implementations work on any platform and serve as the fallback
//! when architecture-specific optimizations are not available.

use wide::{f32x8, i32x8, i16x8, u8x16};
use crate::simd::traits::*;

// ============================================================================
// F32x8 - 8-lane f32 SIMD
// ============================================================================

/// 8-lane f32 SIMD vector using the `wide` crate.
#[derive(Clone, Copy)]
pub struct PortableF32x8(pub f32x8);

impl SimdVector for PortableF32x8 {
    type Element = f32;
    const LANES: usize = 8;

    #[inline]
    fn zero() -> Self {
        Self(f32x8::ZERO)
    }

    #[inline]
    fn splat(value: f32) -> Self {
        Self(f32x8::splat(value))
    }

    #[inline]
    fn load(slice: &[f32]) -> Self {
        debug_assert!(slice.len() >= 8);
        Self(f32x8::new([
            slice[0], slice[1], slice[2], slice[3],
            slice[4], slice[5], slice[6], slice[7],
        ]))
    }

    #[inline]
    fn store(self, slice: &mut [f32]) {
        debug_assert!(slice.len() >= 8);
        let arr = self.0.to_array();
        slice[..8].copy_from_slice(&arr);
    }
}

impl SimdAdd for PortableF32x8 {
    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl SimdSub for PortableF32x8 {
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl SimdMul for PortableF32x8 {
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }
}

impl SimdHorizontal for PortableF32x8 {
    #[inline]
    fn horizontal_sum(self) -> f32 {
        self.0.reduce_add()
    }

    #[inline]
    fn horizontal_min(self) -> f32 {
        let arr = self.0.to_array();
        arr.iter().copied().fold(f32::INFINITY, f32::min)
    }

    #[inline]
    fn horizontal_max(self) -> f32 {
        let arr = self.0.to_array();
        arr.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }
}

impl SimdF32 for PortableF32x8 {
    #[inline]
    fn fused_multiply_add(self, b: Self, c: Self) -> Self {
        // wide doesn't have FMA, so we emulate it
        Self(self.0 * b.0 + c.0)
    }

    #[inline]
    fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    #[inline]
    fn abs(self) -> Self {
        Self(self.0.abs())
    }
}

// ============================================================================
// I32x8 - 8-lane i32 SIMD
// ============================================================================

/// 8-lane i32 SIMD vector.
#[derive(Clone, Copy)]
pub struct PortableI32x8(pub i32x8);

impl SimdVector for PortableI32x8 {
    type Element = i32;
    const LANES: usize = 8;

    #[inline]
    fn zero() -> Self {
        Self(i32x8::ZERO)
    }

    #[inline]
    fn splat(value: i32) -> Self {
        Self(i32x8::splat(value))
    }

    #[inline]
    fn load(slice: &[i32]) -> Self {
        debug_assert!(slice.len() >= 8);
        Self(i32x8::new([
            slice[0], slice[1], slice[2], slice[3],
            slice[4], slice[5], slice[6], slice[7],
        ]))
    }

    #[inline]
    fn store(self, slice: &mut [i32]) {
        debug_assert!(slice.len() >= 8);
        let arr = self.0.to_array();
        slice[..8].copy_from_slice(&arr);
    }
}

impl SimdAdd for PortableI32x8 {
    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl SimdSub for PortableI32x8 {
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl SimdMul for PortableI32x8 {
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }
}

impl SimdHorizontal for PortableI32x8 {
    #[inline]
    fn horizontal_sum(self) -> i32 {
        self.0.reduce_add()
    }

    #[inline]
    fn horizontal_min(self) -> i32 {
        let arr = self.0.to_array();
        arr.iter().copied().min().unwrap_or(0)
    }

    #[inline]
    fn horizontal_max(self) -> i32 {
        let arr = self.0.to_array();
        arr.iter().copied().max().unwrap_or(0)
    }
}

impl SimdI32 for PortableI32x8 {
    type F32Vec = PortableF32x8;

    #[inline]
    fn to_f32(self) -> PortableF32x8 {
        let arr = self.0.to_array();
        PortableF32x8(f32x8::new([
            arr[0] as f32, arr[1] as f32, arr[2] as f32, arr[3] as f32,
            arr[4] as f32, arr[5] as f32, arr[6] as f32, arr[7] as f32,
        ]))
    }
}

// ============================================================================
// I16x8 - 8-lane i16 SIMD
// ============================================================================

/// 8-lane i16 SIMD vector.
#[derive(Clone, Copy)]
pub struct PortableI16x8(pub i16x8);

impl SimdVector for PortableI16x8 {
    type Element = i16;
    const LANES: usize = 8;

    #[inline]
    fn zero() -> Self {
        Self(i16x8::ZERO)
    }

    #[inline]
    fn splat(value: i16) -> Self {
        Self(i16x8::splat(value))
    }

    #[inline]
    fn load(slice: &[i16]) -> Self {
        debug_assert!(slice.len() >= 8);
        Self(i16x8::new([
            slice[0], slice[1], slice[2], slice[3],
            slice[4], slice[5], slice[6], slice[7],
        ]))
    }

    #[inline]
    fn store(self, slice: &mut [i16]) {
        debug_assert!(slice.len() >= 8);
        let arr = self.0.to_array();
        slice[..8].copy_from_slice(&arr);
    }
}

impl SimdAdd for PortableI16x8 {
    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl SimdSub for PortableI16x8 {
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl SimdMul for PortableI16x8 {
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }
}

impl SimdI16 for PortableI16x8 {
    type I32Vec = PortableI32x8;

    #[inline]
    fn expand_lo_to_i32(self) -> PortableI32x8 {
        let arr = self.0.to_array();
        PortableI32x8(i32x8::new([
            arr[0] as i32, arr[1] as i32, arr[2] as i32, arr[3] as i32,
            0, 0, 0, 0,
        ]))
    }

    #[inline]
    fn expand_hi_to_i32(self) -> PortableI32x8 {
        let arr = self.0.to_array();
        PortableI32x8(i32x8::new([
            arr[4] as i32, arr[5] as i32, arr[6] as i32, arr[7] as i32,
            0, 0, 0, 0,
        ]))
    }

    #[inline]
    fn madd(self, other: Self) -> PortableI32x8 {
        let a = self.0.to_array();
        let b = other.0.to_array();
        PortableI32x8(i32x8::new([
            a[0] as i32 * b[0] as i32 + a[1] as i32 * b[1] as i32,
            a[2] as i32 * b[2] as i32 + a[3] as i32 * b[3] as i32,
            a[4] as i32 * b[4] as i32 + a[5] as i32 * b[5] as i32,
            a[6] as i32 * b[6] as i32 + a[7] as i32 * b[7] as i32,
            0, 0, 0, 0,
        ]))
    }
}

// ============================================================================
// U8x16 - 16-lane u8 SIMD (critical for LUT16)
// ============================================================================

/// 16-lane u8 SIMD vector.
///
/// This is essential for LUT16 operations using shuffle_bytes (PSHUFB).
#[derive(Clone, Copy)]
pub struct PortableU8x16(pub u8x16);

impl SimdVector for PortableU8x16 {
    type Element = u8;
    const LANES: usize = 16;

    #[inline]
    fn zero() -> Self {
        Self(u8x16::ZERO)
    }

    #[inline]
    fn splat(value: u8) -> Self {
        Self(u8x16::splat(value))
    }

    #[inline]
    fn load(slice: &[u8]) -> Self {
        debug_assert!(slice.len() >= 16);
        Self(u8x16::new([
            slice[0], slice[1], slice[2], slice[3],
            slice[4], slice[5], slice[6], slice[7],
            slice[8], slice[9], slice[10], slice[11],
            slice[12], slice[13], slice[14], slice[15],
        ]))
    }

    #[inline]
    fn store(self, slice: &mut [u8]) {
        debug_assert!(slice.len() >= 16);
        let arr = self.0.to_array();
        slice[..16].copy_from_slice(&arr);
    }
}

impl SimdU8 for PortableU8x16 {
    type U16Vec = PortableU16x8;

    #[inline]
    fn shuffle_bytes(self, indices: Self) -> Self {
        // Portable implementation of PSHUFB
        let lut = self.0.to_array();
        let idx = indices.0.to_array();
        let mut result = [0u8; 16];

        for i in 0..16 {
            let index = idx[i];
            if index < 128 {
                result[i] = lut[(index & 0x0F) as usize];
            } else {
                result[i] = 0;
            }
        }

        Self(u8x16::new(result))
    }

    #[inline]
    fn bitand(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[inline]
    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    fn bitxor(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }

    #[inline]
    fn shr4(self) -> Self {
        let arr = self.0.to_array();
        let mut result = [0u8; 16];
        for i in 0..16 {
            result[i] = arr[i] >> 4;
        }
        Self(u8x16::new(result))
    }

    #[inline]
    fn expand_lo_to_u16(self) -> PortableU16x8 {
        let arr = self.0.to_array();
        PortableU16x8([
            arr[0] as u16, arr[1] as u16, arr[2] as u16, arr[3] as u16,
            arr[4] as u16, arr[5] as u16, arr[6] as u16, arr[7] as u16,
        ])
    }

    #[inline]
    fn expand_hi_to_u16(self) -> PortableU16x8 {
        let arr = self.0.to_array();
        PortableU16x8([
            arr[8] as u16, arr[9] as u16, arr[10] as u16, arr[11] as u16,
            arr[12] as u16, arr[13] as u16, arr[14] as u16, arr[15] as u16,
        ])
    }
}

// ============================================================================
// U16x8 - 8-lane u16 SIMD
// ============================================================================

/// 8-lane u16 SIMD vector.
#[derive(Clone, Copy)]
pub struct PortableU16x8(pub [u16; 8]);

impl SimdVector for PortableU16x8 {
    type Element = u16;
    const LANES: usize = 8;

    #[inline]
    fn zero() -> Self {
        Self([0; 8])
    }

    #[inline]
    fn splat(value: u16) -> Self {
        Self([value; 8])
    }

    #[inline]
    fn load(slice: &[u16]) -> Self {
        debug_assert!(slice.len() >= 8);
        let mut arr = [0u16; 8];
        arr.copy_from_slice(&slice[..8]);
        Self(arr)
    }

    #[inline]
    fn store(self, slice: &mut [u16]) {
        debug_assert!(slice.len() >= 8);
        slice[..8].copy_from_slice(&self.0);
    }
}

impl SimdAdd for PortableU16x8 {
    #[inline]
    fn add(self, other: Self) -> Self {
        let mut result = [0u16; 8];
        for i in 0..8 {
            result[i] = self.0[i].wrapping_add(other.0[i]);
        }
        Self(result)
    }
}

impl SimdU16 for PortableU16x8 {
    type U32Vec = PortableU32x8;

    #[inline]
    fn expand_lo_to_u32(self) -> PortableU32x8 {
        PortableU32x8([
            self.0[0] as u32, self.0[1] as u32, self.0[2] as u32, self.0[3] as u32,
            0, 0, 0, 0,
        ])
    }

    #[inline]
    fn expand_hi_to_u32(self) -> PortableU32x8 {
        PortableU32x8([
            self.0[4] as u32, self.0[5] as u32, self.0[6] as u32, self.0[7] as u32,
            0, 0, 0, 0,
        ])
    }
}

// ============================================================================
// U32x8 - 8-lane u32 SIMD
// ============================================================================

/// 8-lane u32 SIMD vector.
#[derive(Clone, Copy)]
pub struct PortableU32x8(pub [u32; 8]);

impl SimdVector for PortableU32x8 {
    type Element = u32;
    const LANES: usize = 8;

    #[inline]
    fn zero() -> Self {
        Self([0; 8])
    }

    #[inline]
    fn splat(value: u32) -> Self {
        Self([value; 8])
    }

    #[inline]
    fn load(slice: &[u32]) -> Self {
        debug_assert!(slice.len() >= 8);
        let mut arr = [0u32; 8];
        arr.copy_from_slice(&slice[..8]);
        Self(arr)
    }

    #[inline]
    fn store(self, slice: &mut [u32]) {
        debug_assert!(slice.len() >= 8);
        slice[..8].copy_from_slice(&self.0);
    }
}

impl SimdAdd for PortableU32x8 {
    #[inline]
    fn add(self, other: Self) -> Self {
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[i] = self.0[i].wrapping_add(other.0[i]);
        }
        Self(result)
    }
}

impl SimdHorizontal for PortableU32x8 {
    #[inline]
    fn horizontal_sum(self) -> u32 {
        self.0.iter().sum()
    }

    #[inline]
    fn horizontal_min(self) -> u32 {
        self.0.iter().copied().min().unwrap_or(0)
    }

    #[inline]
    fn horizontal_max(self) -> u32 {
        self.0.iter().copied().max().unwrap_or(0)
    }
}

impl SimdU32 for PortableU32x8 {
    type F32Vec = PortableF32x8;

    #[inline]
    fn to_f32(self) -> PortableF32x8 {
        PortableF32x8(f32x8::new([
            self.0[0] as f32, self.0[1] as f32, self.0[2] as f32, self.0[3] as f32,
            self.0[4] as f32, self.0[5] as f32, self.0[6] as f32, self.0[7] as f32,
        ]))
    }
}

// ============================================================================
// I8x16 - 16-lane i8 SIMD
// ============================================================================

/// 16-lane i8 SIMD vector.
#[derive(Clone, Copy)]
pub struct PortableI8x16(pub [i8; 16]);

impl SimdVector for PortableI8x16 {
    type Element = i8;
    const LANES: usize = 16;

    #[inline]
    fn zero() -> Self {
        Self([0; 16])
    }

    #[inline]
    fn splat(value: i8) -> Self {
        Self([value; 16])
    }

    #[inline]
    fn load(slice: &[i8]) -> Self {
        debug_assert!(slice.len() >= 16);
        let mut arr = [0i8; 16];
        arr.copy_from_slice(&slice[..16]);
        Self(arr)
    }

    #[inline]
    fn store(self, slice: &mut [i8]) {
        debug_assert!(slice.len() >= 16);
        slice[..16].copy_from_slice(&self.0);
    }
}

impl SimdAdd for PortableI8x16 {
    #[inline]
    fn add(self, other: Self) -> Self {
        let mut result = [0i8; 16];
        for i in 0..16 {
            result[i] = self.0[i].wrapping_add(other.0[i]);
        }
        Self(result)
    }
}

impl SimdSub for PortableI8x16 {
    #[inline]
    fn sub(self, other: Self) -> Self {
        let mut result = [0i8; 16];
        for i in 0..16 {
            result[i] = self.0[i].wrapping_sub(other.0[i]);
        }
        Self(result)
    }
}

impl SimdI8 for PortableI8x16 {
    type I16Vec = PortableI16x8;

    #[inline]
    fn expand_lo_to_i16(self) -> PortableI16x8 {
        PortableI16x8(i16x8::new([
            self.0[0] as i16, self.0[1] as i16, self.0[2] as i16, self.0[3] as i16,
            self.0[4] as i16, self.0[5] as i16, self.0[6] as i16, self.0[7] as i16,
        ]))
    }

    #[inline]
    fn expand_hi_to_i16(self) -> PortableI16x8 {
        PortableI16x8(i16x8::new([
            self.0[8] as i16, self.0[9] as i16, self.0[10] as i16, self.0[11] as i16,
            self.0[12] as i16, self.0[13] as i16, self.0[14] as i16, self.0[15] as i16,
        ]))
    }
}

// ============================================================================
// Convenience type aliases
// ============================================================================

/// Default f32 SIMD type for the current platform.
pub type F32Simd = PortableF32x8;

/// Default i32 SIMD type for the current platform.
pub type I32Simd = PortableI32x8;

/// Default u8 SIMD type for the current platform.
pub type U8Simd = PortableU8x16;

/// Default i8 SIMD type for the current platform.
pub type I8Simd = PortableI8x16;
