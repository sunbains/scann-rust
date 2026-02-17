//! Tests for SIMD implementations.

#[cfg(test)]
mod tests {
    use crate::simd::dispatch::*;
    use crate::simd::portable::*;
    use crate::simd::traits::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    // ========================================================================
    // Portable F32x8 tests
    // ========================================================================

    #[test]
    fn test_f32x8_zero() {
        let v = PortableF32x8::zero();
        let arr = v.to_array();
        assert!(arr.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_f32x8_splat() {
        let v = PortableF32x8::splat(3.14);
        let arr = v.to_array();
        assert!(arr.iter().all(|&x| approx_eq(x, 3.14)));
    }

    #[test]
    fn test_f32x8_load_store() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = PortableF32x8::load(&input);
        let mut output = [0.0f32; 8];
        v.store(&mut output);
        assert_eq!(input, output);
    }

    #[test]
    fn test_f32x8_add() {
        let a = PortableF32x8::load(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = PortableF32x8::load(&[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let c = a.add(b);
        let arr = c.to_array();
        assert!(arr.iter().all(|&x| approx_eq(x, 9.0)));
    }

    #[test]
    fn test_f32x8_mul() {
        let a = PortableF32x8::load(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = PortableF32x8::load(&[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
        let c = a.mul(b);
        let arr = c.to_array();
        let expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        for i in 0..8 {
            assert!(approx_eq(arr[i], expected[i]));
        }
    }

    #[test]
    fn test_f32x8_horizontal_sum() {
        let v = PortableF32x8::load(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let sum = v.horizontal_sum();
        assert!(approx_eq(sum, 36.0)); // 1+2+3+4+5+6+7+8 = 36
    }

    #[test]
    fn test_f32x8_fma() {
        let a = PortableF32x8::splat(2.0);
        let b = PortableF32x8::splat(3.0);
        let c = PortableF32x8::splat(4.0);
        let result = a.fused_multiply_add(b, c);
        let arr = result.to_array();
        // 2 * 3 + 4 = 10
        assert!(arr.iter().all(|&x| approx_eq(x, 10.0)));
    }

    // ========================================================================
    // Portable U8x16 tests
    // ========================================================================

    #[test]
    fn test_u8x16_shuffle() {
        // LUT with values 0-15
        let lut = PortableU8x16::load(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        // Indices
        let indices = PortableU8x16::load(&[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]);

        let result = lut.shuffle_bytes(indices);
        let arr = result.to_array();

        // Each result should equal lut[indices[i] & 0x0F]
        let idx_arr = indices.to_array();
        for i in 0..16 {
            assert_eq!(arr[i], idx_arr[i] & 0x0F);
        }
    }

    #[test]
    fn test_u8x16_lut16_lookup() {
        // Distance values for codes 0-15
        let lut = PortableU8x16::load(&[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]);

        // Codes to look up (4-bit values)
        let codes = PortableU8x16::load(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        let result = SimdLookup::lut16_lookup(lut, codes);
        let arr = result.to_array();

        // Results should match LUT values
        let expected = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
        assert_eq!(arr, expected);
    }

    // ========================================================================
    // Dispatch function tests
    // ========================================================================

    #[test]
    fn test_dot_product_dispatch() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let result = dot_product_f32(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!(approx_eq(result, expected));
    }

    #[test]
    fn test_dot_product_large() {
        let size = 128;
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.1).collect();

        let result = dot_product_f32(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!((result - expected).abs() < 0.1);
    }

    #[test]
    fn test_squared_l2_dispatch() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = squared_l2_f32(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();

        assert!(approx_eq(result, expected));
        assert!(approx_eq(result, 8.0)); // 8 * 1^2 = 8
    }

    #[test]
    fn test_squared_l2_large() {
        let size = 256;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| i as f32 + 1.0).collect();

        let result = squared_l2_f32(&a, &b);
        let expected = size as f32; // each diff is 1.0, so sum of squares is size

        assert!((result - expected).abs() < 0.01);
    }

    #[test]
    fn test_horizontal_sum_dispatch() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = horizontal_sum_f32(&values);
        let expected: f32 = values.iter().sum();

        assert!(approx_eq(result, expected));
    }

    #[test]
    fn test_simd_support_level() {
        let level = simd_support_level();
        // Should at least be Portable
        assert!(level >= SimdSupportLevel::Portable);
    }

    // ========================================================================
    // One-to-many tests
    // ========================================================================

    #[test]
    fn test_one_to_many_dot_product() {
        let query = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let database = vec![
            // Point 0
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            // Point 1
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            // Point 2
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        ];
        let mut results = vec![0.0f32; 3];

        one_to_many_dot_product_f32(&query, &database, 8, 3, &mut results);

        // Dot products: 36, 72, 18 (negated for distance)
        assert!(approx_eq(results[0], -36.0));
        assert!(approx_eq(results[1], -72.0));
        assert!(approx_eq(results[2], -18.0));
    }

    #[test]
    fn test_one_to_many_squared_l2() {
        let query = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let database = vec![
            // Point 0: same as query
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            // Point 1: all zeros
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // Point 2: query + 1
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        ];
        let mut results = vec![0.0f32; 3];

        one_to_many_squared_l2_f32(&query, &database, 8, 3, &mut results);

        // Distance to self: 0
        assert!(approx_eq(results[0], 0.0));
        // Distance to zeros: sum of squares = 1+4+9+16+25+36+49+64 = 204
        assert!(approx_eq(results[1], 204.0));
        // Distance to query+1: 8 * 1^2 = 8
        assert!(approx_eq(results[2], 8.0));
    }

    // ========================================================================
    // LUT16 tests
    // ========================================================================

    #[test]
    fn test_lut16_batch_portable() {
        // Simple LUT: code 0 -> 0, code 1 -> 1, ..., code 15 -> 15
        let mut lut = vec![0u8; 16 * 2]; // 2 subspaces
        for i in 0..16 {
            lut[i] = i as u8;
            lut[16 + i] = (15 - i) as u8;
        }

        // 4 datapoints, 2 subspaces, packed as 1 byte per datapoint
        // Point 0: codes [0, 0] -> packed 0x00
        // Point 1: codes [1, 1] -> packed 0x11
        // Point 2: codes [15, 0] -> packed 0x0F
        // Point 3: codes [0, 15] -> packed 0xF0
        let packed_codes = vec![0x00, 0x11, 0x0F, 0xF0];

        let mut results = vec![0.0f32; 4];
        lut16_distances_batch(&packed_codes, &lut, 2, 4, &mut results);

        // Point 0: lut[0][0] + lut[1][0] = 0 + 15 = 15
        assert!(approx_eq(results[0], 15.0));
        // Point 1: lut[0][1] + lut[1][1] = 1 + 14 = 15
        assert!(approx_eq(results[1], 15.0));
        // Point 2: lut[0][15] + lut[1][0] = 15 + 15 = 30
        assert!(approx_eq(results[2], 30.0));
        // Point 3: lut[0][0] + lut[1][15] = 0 + 0 = 0
        assert!(approx_eq(results[3], 0.0));
    }
}
