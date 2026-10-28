//! Weight quantization formats.
//!
//! Decode streams every weight once per token, so its speed is set by bytes
//! per weight, not FLOPs. Both formats are symmetric (no zero point) and keep
//! activations in f32; weights are dequantized inside the dot-product kernel.
//!
//! | format | layout                                            | bits/weight |
//! |--------|---------------------------------------------------|-------------|
//! | `Q8`   | i8 per weight, one f32 scale per output row       | 8.06        |
//! | `Q4`   | 4-bit per weight, one f16 scale per `group` weights | 4.5 (g=32) |
//!
//! Q8 can afford a single scale per row (per output channel): 255 levels are
//! fine-grained enough that one outlier only costs a little resolution. With
//! only 16 levels, one outlier would flatten a whole 576-wide row, so Q4
//! localizes the damage with a scale per small group.

use std::fmt;
use std::str::FromStr;

use anyhow::{Result, bail, ensure};
use half::f16;

/// Weights per Q4 block: the unit the SIMD kernel unpacks at once.
pub const Q4_BLOCK: usize = 32;

/// Storage format chosen at load time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    F32,
    Q8,
    /// 4-bit with one scale per `group` weights (a multiple of 32).
    Q4 {
        group: usize,
    },
}

impl Quant {
    pub const Q4_DEFAULT_GROUP: usize = 32;
}

impl FromStr for Quant {
    type Err = anyhow::Error;

    /// `f32`, `q8`, `q4` (group 32) or `q4g64`-style explicit group sizes.
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "f32" => Self::F32,
            "q8" => Self::Q8,
            "q4" => Self::Q4 {
                group: Self::Q4_DEFAULT_GROUP,
            },
            _ => match s.strip_prefix("q4g").map(str::parse::<usize>) {
                Some(Ok(group)) if group > 0 && group.is_multiple_of(Q4_BLOCK) => {
                    Self::Q4 { group }
                }
                _ => bail!("unknown quantization {s:?} (expected f32, q8, q4 or q4g<N>)"),
            },
        })
    }
}

impl fmt::Display for Quant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => f.write_str("f32"),
            Self::Q8 => f.write_str("q8"),
            Self::Q4 { group } if *group == Self::Q4_DEFAULT_GROUP => f.write_str("q4"),
            Self::Q4 { group } => write!(f, "q4g{group}"),
        }
    }
}

/// Int8, per-output-channel: `w[r][c] ≈ q[r][c] · scale[r]`, `q ∈ [-127, 127]`.
pub struct Q8 {
    pub(super) q: Vec<i8>,
    pub(super) scales: Vec<f32>,
}

impl Q8 {
    pub fn quantize(data: &[f32], cols: usize) -> Self {
        let mut q = Vec::with_capacity(data.len());
        let mut scales = Vec::with_capacity(data.len() / cols);
        for row in data.chunks_exact(cols) {
            // Map the largest magnitude to ±127 (not -128, to stay symmetric).
            let amax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let scale = amax / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            q.extend(row.iter().map(|&v| (v * inv).round() as i8));
            scales.push(scale);
        }
        Self { q, scales }
    }

    pub(super) fn dequantize_row(&self, r: usize, cols: usize, out: &mut [f32]) {
        let s = self.scales[r];
        for (o, &q) in out.iter_mut().zip(&self.q[r * cols..(r + 1) * cols]) {
            *o = f32::from(q) * s;
        }
    }

    pub(super) fn bytes(&self) -> usize {
        self.q.len() + 4 * self.scales.len()
    }
}

/// 4-bit groups: `w ≈ (q - 8) · scale[group]`, `q ∈ [0, 15]`.
///
/// Packing per 32-weight block (16 bytes): byte `j` holds weight `j` in its low
/// nibble and weight `j + 16` in its high nibble. The kernel can then unpack a
/// whole block with one AND and one shift, and the two halves land on
/// contiguous runs of the activation vector (`x[0..16]`, `x[16..32]`).
pub struct Q4 {
    pub(super) packed: Vec<u8>,
    pub(super) scales: Vec<f16>,
    pub(super) group: usize,
}

impl Q4 {
    pub fn quantize(data: &[f32], cols: usize, group: usize) -> Result<Self> {
        ensure!(
            group.is_multiple_of(Q4_BLOCK) && cols.is_multiple_of(group),
            "q4 needs the row length ({cols}) to be a multiple of the group size ({group}), \
             itself a multiple of {Q4_BLOCK}"
        );
        let mut packed = Vec::with_capacity(data.len() / 2);
        let mut scales = Vec::with_capacity(data.len() / group);
        for g in data.chunks_exact(group) {
            // As in llama.cpp's Q4_0: the scale is *signed*, chosen so the
            // largest-magnitude weight maps exactly to level 0 (= -8). This uses
            // all 16 levels for it instead of wasting one on symmetry.
            let extreme = g
                .iter()
                .fold(0.0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
            let scale = f16::from_f32(extreme / -8.0);
            // Quantize with the *rounded* f16 scale, the one the kernel will use.
            let s = scale.to_f32();
            let inv = if s != 0.0 { 1.0 / s } else { 0.0 };
            let level = |v: f32| ((v * inv).round() + 8.0).clamp(0.0, 15.0) as u8;
            for block in g.as_chunks::<Q4_BLOCK>().0.iter() {
                let (lo, hi) = block.split_at(Q4_BLOCK / 2);
                packed.extend(lo.iter().zip(hi).map(|(&a, &b)| level(a) | (level(b) << 4)));
            }
            scales.push(scale);
        }
        Ok(Self {
            packed,
            scales,
            group,
        })
    }

    pub(super) fn dequantize_row(&self, r: usize, cols: usize, out: &mut [f32]) {
        let packed = &self.packed[r * cols / 2..(r + 1) * cols / 2];
        let scales = &self.scales[r * cols / self.group..(r + 1) * cols / self.group];
        for (b, (block, out)) in packed
            .as_chunks::<{ Q4_BLOCK / 2 }>()
            .0
            .iter()
            .zip(out.as_chunks_mut::<Q4_BLOCK>().0.iter_mut())
            .enumerate()
        {
            let s = scales[b * Q4_BLOCK / self.group].to_f32();
            let (lo, hi) = out.split_at_mut(Q4_BLOCK / 2);
            for ((&byte, l), h) in block.iter().zip(lo).zip(hi) {
                *l = (f32::from(byte & 0x0F) - 8.0) * s;
                *h = (f32::from(byte >> 4) - 8.0) * s;
            }
        }
    }

    pub(super) fn bytes(&self) -> usize {
        self.packed.len() + 2 * self.scales.len()
    }
}

/// An activation vector quantized to int8 in blocks of 32 (llama.cpp's Q8_0),
/// so that a Q4 row can be dotted with it in integer arithmetic.
///
/// Done once per mat-vec and shared by every weight row. Per block `b`:
/// `x ≈ q · d[b]`, so a block's dot product with Q4 weights is
/// `s · d[b] · Σ (n - 8)·q`, an integer sum.
pub struct Q8Acts {
    pub(super) q: Vec<i8>,
    pub(super) d: Vec<f32>,
}

impl Q8Acts {
    pub fn quantize(x: &[f32]) -> Self {
        let blocks = x.len() / Q4_BLOCK;
        let mut q = Vec::with_capacity(x.len());
        let mut d = Vec::with_capacity(blocks);
        for block in x.as_chunks::<Q4_BLOCK>().0 {
            let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let scale = amax / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            // Round half away from zero without `f32::round`, which is a
            // libm call on baseline x86-64 and showed up in decode profiles.
            q.extend(block.iter().map(|&v| {
                let t = v * inv;
                (t + 0.5f32.copysign(t)) as i8
            }));
            d.push(scale);
        }
        Self { q, d }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::kernels::tests::data;

    fn max_err(q: &dyn Fn(usize, &mut [f32]), w: &[f32], cols: usize) -> f32 {
        let mut row = vec![0.0; cols];
        let mut worst = 0.0f32;
        for (r, wr) in w.chunks_exact(cols).enumerate() {
            q(r, &mut row);
            for (a, b) in row.iter().zip(wr) {
                worst = worst.max((a - b).abs());
            }
        }
        worst
    }

    #[test]
    fn q8_error_is_at_most_half_a_step() {
        let cols = 96;
        let w = data(4 * cols, 3);
        let q = Q8::quantize(&w, cols);
        let step = q.scales.iter().fold(0.0f32, |m, &s| m.max(s));
        let err = max_err(&|r, out| q.dequantize_row(r, cols, out), &w, cols);
        assert!(err <= 0.5 * step + 1e-7, "{err} > {step}/2");
    }

    #[test]
    fn q4_error_is_bounded_by_its_group_scale() {
        for group in [32, 64] {
            let cols = 128;
            let w = data(3 * cols, 5);
            let q = Q4::quantize(&w, cols, group).unwrap();
            let step = q.scales.iter().fold(0.0f32, |m, s| m.max(s.to_f32().abs()));
            let err = max_err(&|r, out| q.dequantize_row(r, cols, out), &w, cols);
            // Half a step from rounding, plus up to one step where +8 clamps to +7.
            assert!(err <= 1.5 * step, "group {group}: {err} vs step {step}");
        }
    }

    #[test]
    fn q4_extreme_value_is_exact() {
        let mut w = vec![0.01f32; 32];
        w[5] = -0.8;
        let q = Q4::quantize(&w, 32, 32).unwrap();
        let mut out = vec![0.0; 32];
        q.dequantize_row(0, 32, &mut out);
        assert!((out[5] + 0.8).abs() < 1e-3);
    }

    #[test]
    fn parses_cli_names() {
        assert_eq!("q4".parse::<Quant>().unwrap(), Quant::Q4 { group: 32 });
        assert_eq!("q4g64".parse::<Quant>().unwrap(), Quant::Q4 { group: 64 });
        assert!("q4g48".parse::<Quant>().is_err());
        assert_eq!(Quant::Q4 { group: 64 }.to_string(), "q4g64");
    }
}
