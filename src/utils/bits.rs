//! Bit manipulation utilities.

/// Count the number of set bits (popcount) in a byte slice.
#[inline]
pub fn popcount_bytes(data: &[u8]) -> u32 {
    data.iter().map(|&b| b.count_ones()).sum()
}

/// Count the number of set bits in a u64.
#[inline]
pub fn popcount_u64(x: u64) -> u32 {
    x.count_ones()
}

/// Compute Hamming distance between two byte slices.
pub fn hamming_distance_bytes(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

/// Compute Hamming distance between two u64 values.
#[inline]
pub fn hamming_distance_u64(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Bit iterator over bytes (LSB first).
pub struct BitIterator<'a> {
    data: &'a [u8],
    byte_idx: usize,
    bit_idx: u8,
}

impl<'a> BitIterator<'a> {
    /// Create a new bit iterator.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_idx: 0,
            bit_idx: 0,
        }
    }
}

impl<'a> Iterator for BitIterator<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.byte_idx >= self.data.len() {
            return None;
        }

        let bit = (self.data[self.byte_idx] >> self.bit_idx) & 1 == 1;

        self.bit_idx += 1;
        if self.bit_idx >= 8 {
            self.bit_idx = 0;
            self.byte_idx += 1;
        }

        Some(bit)
    }
}

/// Pack boolean values into bytes.
pub fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let num_bytes = (bits.len() + 7) / 8;
    let mut result = vec![0u8; num_bytes];

    for (i, &bit) in bits.iter().enumerate() {
        if bit {
            result[i / 8] |= 1 << (i % 8);
        }
    }

    result
}

/// Unpack bytes into boolean values.
pub fn unpack_bits(bytes: &[u8], num_bits: usize) -> Vec<bool> {
    BitIterator::new(bytes).take(num_bits).collect()
}

/// Find the index of the lowest set bit.
#[inline]
pub fn lowest_set_bit(x: u64) -> Option<u32> {
    if x == 0 {
        None
    } else {
        Some(x.trailing_zeros())
    }
}

/// Find the index of the highest set bit.
#[inline]
pub fn highest_set_bit(x: u64) -> Option<u32> {
    if x == 0 {
        None
    } else {
        Some(63 - x.leading_zeros())
    }
}

/// Clear the lowest set bit.
#[inline]
pub fn clear_lowest_bit(x: u64) -> u64 {
    x & (x - 1)
}

/// Isolate the lowest set bit.
#[inline]
pub fn isolate_lowest_bit(x: u64) -> u64 {
    x & x.wrapping_neg()
}

/// Round up to the next power of 2.
#[inline]
pub fn next_power_of_2(x: u64) -> u64 {
    if x == 0 {
        return 1;
    }
    1u64 << (64 - (x - 1).leading_zeros())
}

/// Check if a value is a power of 2.
#[inline]
pub fn is_power_of_2(x: u64) -> bool {
    x != 0 && (x & (x - 1)) == 0
}

/// Interleave bits of two 32-bit values (Morton code).
pub fn interleave_bits(x: u32, y: u32) -> u64 {
    let mut result = 0u64;
    for i in 0..32 {
        result |= ((x as u64 >> i) & 1) << (2 * i);
        result |= ((y as u64 >> i) & 1) << (2 * i + 1);
    }
    result
}

/// De-interleave bits (reverse Morton code).
pub fn deinterleave_bits(z: u64) -> (u32, u32) {
    let mut x = 0u32;
    let mut y = 0u32;
    for i in 0..32 {
        x |= (((z >> (2 * i)) & 1) as u32) << i;
        y |= (((z >> (2 * i + 1)) & 1) as u32) << i;
    }
    (x, y)
}

/// Bit-parallel prefix sum (for vectorized operations).
pub fn prefix_popcount(data: &[u64]) -> Vec<u32> {
    let mut result = Vec::with_capacity(data.len() + 1);
    result.push(0);

    let mut sum = 0u32;
    for &x in data {
        sum += x.count_ones();
        result.push(sum);
    }

    result
}

/// Select the k-th set bit (1-indexed).
pub fn select_bit(x: u64, k: u32) -> Option<u32> {
    if k == 0 || k > x.count_ones() {
        return None;
    }

    let mut remaining = k;
    let mut value = x;

    while remaining > 0 {
        let pos = value.trailing_zeros();
        remaining -= 1;
        if remaining == 0 {
            return Some(pos);
        }
        value = clear_lowest_bit(value);
    }

    None
}

/// Compact sparse indices using bit manipulation.
pub fn compact_sparse_indices(bitmap: &[u64]) -> Vec<u32> {
    let mut indices = Vec::new();

    for (word_idx, &word) in bitmap.iter().enumerate() {
        let mut w = word;
        let base = (word_idx * 64) as u32;

        while w != 0 {
            let bit_pos = w.trailing_zeros();
            indices.push(base + bit_pos);
            w = clear_lowest_bit(w);
        }
    }

    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_popcount() {
        assert_eq!(popcount_bytes(&[0xFF]), 8);
        assert_eq!(popcount_bytes(&[0x00]), 0);
        assert_eq!(popcount_bytes(&[0xAA, 0x55]), 8);
        assert_eq!(popcount_u64(0xFFFF_FFFF_FFFF_FFFF), 64);
    }

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance_bytes(&[0xFF], &[0x00]), 8);
        assert_eq!(hamming_distance_bytes(&[0xFF], &[0xFF]), 0);
        assert_eq!(hamming_distance_bytes(&[0xAA], &[0x55]), 8);
    }

    #[test]
    fn test_bit_iterator() {
        let data = [0b10101010u8, 0b01010101u8];
        let bits: Vec<bool> = BitIterator::new(&data).collect();

        assert_eq!(bits.len(), 16);
        assert_eq!(bits[0], false); // LSB of first byte
        assert_eq!(bits[1], true);
        assert_eq!(bits[8], true); // LSB of second byte
    }

    #[test]
    fn test_pack_unpack_bits() {
        let original = vec![true, false, true, false, true, false, true, false, true];
        let packed = pack_bits(&original);
        let unpacked = unpack_bits(&packed, original.len());

        assert_eq!(original, unpacked);
    }

    #[test]
    fn test_lowest_highest_bit() {
        assert_eq!(lowest_set_bit(0b1010), Some(1));
        assert_eq!(lowest_set_bit(0b1000), Some(3));
        assert_eq!(lowest_set_bit(0), None);

        assert_eq!(highest_set_bit(0b1010), Some(3));
        assert_eq!(highest_set_bit(0b0001), Some(0));
        assert_eq!(highest_set_bit(0), None);
    }

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(0), 1);
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(5), 8);
        assert_eq!(next_power_of_2(8), 8);
        assert_eq!(next_power_of_2(9), 16);
    }

    #[test]
    fn test_interleave_deinterleave() {
        let x = 0b1010u32;
        let y = 0b0101u32;
        let z = interleave_bits(x, y);
        let (x2, y2) = deinterleave_bits(z);

        assert_eq!(x, x2);
        assert_eq!(y, y2);
    }

    #[test]
    fn test_select_bit() {
        assert_eq!(select_bit(0b10100, 1), Some(2));
        assert_eq!(select_bit(0b10100, 2), Some(4));
        assert_eq!(select_bit(0b10100, 3), None);
    }
}
