//! Core SIMD traits defining the interface for vectorized operations.
//!
//! These traits provide a unified abstraction over different SIMD implementations,
//! allowing code to be written once and automatically dispatch to the best
//! available implementation.

/// Base trait for all SIMD vector types.
///
/// Provides the fundamental operations that all SIMD vectors must support.
pub trait SimdVector: Sized + Copy + Clone + Send + Sync {
    /// The scalar element type.
    type Element: Copy;

    /// Number of lanes in the vector.
    const LANES: usize;

    /// Create a vector with all lanes set to zero.
    fn zero() -> Self;

    /// Create a vector with all lanes set to the same value.
    fn splat(value: Self::Element) -> Self;

    /// Load a vector from a slice (must have at least LANES elements).
    fn load(slice: &[Self::Element]) -> Self;

    /// Load a vector from a slice, using zero for out-of-bounds elements.
    fn load_or_zero(slice: &[Self::Element]) -> Self {
        if slice.len() >= Self::LANES {
            Self::load(slice)
        } else {
            let mut arr = vec![unsafe { std::mem::zeroed() }; Self::LANES];
            arr[..slice.len()].copy_from_slice(slice);
            Self::load(&arr)
        }
    }

    /// Store the vector to a mutable slice.
    fn store(self, slice: &mut [Self::Element]);

    /// Convert the vector to a Vec.
    fn to_array(self) -> Vec<Self::Element> {
        let mut arr = vec![unsafe { std::mem::zeroed() }; Self::LANES];
        self.store(&mut arr);
        arr
    }
}

/// Trait for SIMD vectors that support addition.
pub trait SimdAdd: SimdVector {
    /// Add two vectors element-wise.
    fn add(self, other: Self) -> Self;

    /// Add a scalar to all lanes.
    fn add_scalar(self, scalar: Self::Element) -> Self {
        self.add(Self::splat(scalar))
    }
}

/// Trait for SIMD vectors that support subtraction.
pub trait SimdSub: SimdVector {
    /// Subtract two vectors element-wise.
    fn sub(self, other: Self) -> Self;
}

/// Trait for SIMD vectors that support multiplication.
pub trait SimdMul: SimdVector {
    /// Multiply two vectors element-wise.
    fn mul(self, other: Self) -> Self;

    /// Multiply by a scalar.
    fn mul_scalar(self, scalar: Self::Element) -> Self {
        self.mul(Self::splat(scalar))
    }
}

/// Trait for SIMD vectors that support horizontal operations.
pub trait SimdHorizontal: SimdVector {
    /// Sum all lanes and return a scalar.
    fn horizontal_sum(self) -> Self::Element;

    /// Find the minimum value across all lanes.
    fn horizontal_min(self) -> Self::Element;

    /// Find the maximum value across all lanes.
    fn horizontal_max(self) -> Self::Element;
}

/// SIMD operations specific to 32-bit floats.
pub trait SimdF32: SimdVector<Element = f32> + SimdAdd + SimdSub + SimdMul + SimdHorizontal {
    /// Fused multiply-add: a * b + c
    fn fused_multiply_add(self, b: Self, c: Self) -> Self;

    /// Element-wise square root.
    fn sqrt(self) -> Self;

    /// Element-wise minimum.
    fn min(self, other: Self) -> Self;

    /// Element-wise maximum.
    fn max(self, other: Self) -> Self;

    /// Element-wise absolute value.
    fn abs(self) -> Self;

    /// Compute squared L2 distance between two vectors.
    /// Returns the sum of (a[i] - b[i])^2 for all lanes.
    fn squared_l2_distance(self, other: Self) -> f32 {
        let diff = self.sub(other);
        diff.mul(diff).horizontal_sum()
    }

    /// Compute dot product of two vectors.
    fn dot_product(self, other: Self) -> f32 {
        self.mul(other).horizontal_sum()
    }
}

/// SIMD operations for signed 8-bit integers.
pub trait SimdI8: SimdVector<Element = i8> + SimdAdd + SimdSub {
    /// The i16 vector type for expanded results.
    type I16Vec: SimdI16;

    /// Expand the lower half to 16-bit integers.
    fn expand_lo_to_i16(self) -> Self::I16Vec;

    /// Expand the upper half to 16-bit integers.
    fn expand_hi_to_i16(self) -> Self::I16Vec;
}

/// SIMD operations for signed 16-bit integers.
pub trait SimdI16: SimdVector<Element = i16> + SimdAdd + SimdSub + SimdMul {
    /// The i32 vector type for expanded results.
    type I32Vec: SimdI32;

    /// Expand the lower half to 32-bit integers.
    fn expand_lo_to_i32(self) -> Self::I32Vec;

    /// Expand the upper half to 32-bit integers.
    fn expand_hi_to_i32(self) -> Self::I32Vec;

    /// Multiply and add adjacent pairs, returning i32 results.
    /// Result[i] = a[2i] * b[2i] + a[2i+1] * b[2i+1]
    fn madd(self, other: Self) -> Self::I32Vec;
}

/// SIMD operations for signed 32-bit integers.
pub trait SimdI32: SimdVector<Element = i32> + SimdAdd + SimdSub + SimdMul + SimdHorizontal {
    /// The f32 vector type for conversion.
    type F32Vec: SimdF32;

    /// Convert to 32-bit floats.
    fn to_f32(self) -> Self::F32Vec;
}

/// SIMD operations for unsigned 8-bit integers.
///
/// This is critical for LUT16 operations where we use `shuffle_bytes` (PSHUFB).
pub trait SimdU8: SimdVector<Element = u8> {
    /// The u16 vector type for expanded results.
    type U16Vec: SimdU16;

    /// Shuffle bytes using indices from another vector.
    ///
    /// For each lane i in `indices`, if indices[i] < 128:
    ///   result[i] = self[indices[i] & 0x0F]
    /// else:
    ///   result[i] = 0
    ///
    /// This maps to PSHUFB on x86.
    fn shuffle_bytes(self, indices: Self) -> Self;

    /// Bitwise AND.
    fn bitand(self, other: Self) -> Self;

    /// Bitwise OR.
    fn bitor(self, other: Self) -> Self;

    /// Bitwise XOR.
    fn bitxor(self, other: Self) -> Self;

    /// Logical right shift by 4 bits (for extracting high nibble).
    fn shr4(self) -> Self;

    /// Expand unsigned bytes to 16-bit integers (lower half).
    fn expand_lo_to_u16(self) -> Self::U16Vec;

    /// Expand unsigned bytes to 16-bit integers (upper half).
    fn expand_hi_to_u16(self) -> Self::U16Vec;
}

/// SIMD operations for unsigned 16-bit integers.
pub trait SimdU16: SimdVector<Element = u16> + SimdAdd {
    /// The u32 vector type for expanded results.
    type U32Vec: SimdU32;

    /// Expand to 32-bit integers (lower half).
    fn expand_lo_to_u32(self) -> Self::U32Vec;

    /// Expand to 32-bit integers (upper half).
    fn expand_hi_to_u32(self) -> Self::U32Vec;
}

/// SIMD operations for unsigned 32-bit integers.
pub trait SimdU32: SimdVector<Element = u32> + SimdAdd + SimdHorizontal {
    /// The f32 vector type for conversion.
    type F32Vec: SimdF32;

    /// Convert to 32-bit floats.
    fn to_f32(self) -> Self::F32Vec;
}

/// Marker trait for types that can be used with SIMD lookup operations.
pub trait SimdLookup: SimdU8 {
    /// Perform a 16-entry lookup table operation on 4-bit indices.
    ///
    /// The low 4 bits of each byte in `codes` are used as indices into
    /// the 16-entry lookup table `lut`.
    ///
    /// This is the fundamental operation for LUT16 distance computation.
    fn lut16_lookup(lut: Self, codes: Self) -> Self {
        // Mask to low 4 bits
        let mask = Self::splat(0x0F);
        let indices = codes.bitand(mask);
        lut.shuffle_bytes(indices)
    }
}

// Blanket implementation: any SimdU8 can do lookups
impl<T: SimdU8> SimdLookup for T {}
