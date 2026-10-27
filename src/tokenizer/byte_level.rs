//! GPT-2's reversible byte <-> unicode mapping.
//!
//! Byte-level BPE operates on raw UTF-8 bytes, but `tokenizer.json` stores the
//! vocabulary as JSON strings. GPT-2 therefore maps every byte to a *printable*
//! code point: the 188 bytes that are already visible Latin-1 characters map to
//! themselves, and the remaining 68 (control bytes, space, 0x7F-0xA0, 0xAD) are
//! shifted up to U+0100.. in order. That is why a leading space shows up as
//! `Ġ` (U+0120 = 256 + 32) in the vocab.
//!
//! Kiln only uses this mapping at load time, to turn vocab strings back into
//! byte sequences. Encoding and decoding then work directly on bytes.

/// Code point that represents `byte` in the vocab file.
fn byte_to_char(byte: u8) -> char {
    let printable = matches!(byte, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    if printable {
        char::from(byte)
    } else {
        // Rank of `byte` among the non-printable bytes, offset into U+0100..
        let rank = (0..byte)
            .filter(|&b| !matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF))
            .count() as u32;
        char::from_u32(256 + rank).expect("rank < 68, always a valid code point")
    }
}

/// Inverse of [`byte_to_char`], built once as a lookup table.
pub(super) struct ByteDecoder {
    // Every mapped code point is < 256 + 68, so a flat table indexed by code point works.
    table: [Option<u8>; 324],
}

impl ByteDecoder {
    pub(super) fn new() -> Self {
        let mut table = [None; 324];
        for byte in 0..=255u8 {
            table[byte_to_char(byte) as usize] = Some(byte);
        }
        Self { table }
    }

    /// Decode a vocab string (e.g. `"Ġhello"`) into its raw bytes (`b" hello"`).
    pub(super) fn decode(&self, token: &str) -> Option<Vec<u8>> {
        token
            .chars()
            .map(|c| self.table.get(c as usize).copied().flatten())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_is_a_bijection_onto_known_points() {
        let dec = ByteDecoder::new();
        for byte in 0..=255u8 {
            let c = byte_to_char(byte);
            assert_eq!(dec.decode(&c.to_string()), Some(vec![byte]));
        }
        assert_eq!(byte_to_char(b' '), 'Ġ');
        assert_eq!(byte_to_char(b'\n'), 'Ċ');
        assert_eq!(byte_to_char(b'a'), 'a');
    }
}
