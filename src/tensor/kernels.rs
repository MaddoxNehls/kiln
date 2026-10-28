//! Inner-loop kernels: dot products and register tiles.
//!
//! Every kernel has a portable scalar version and an AVX2+FMA version written
//! with `std::arch` intrinsics. The SIMD path is chosen at runtime (CPUID), so
//! one binary runs everywhere; the scalar path is kept as the fallback and as
//! the reference the SIMD kernels are tested against.

use std::sync::atomic::{AtomicU8, Ordering};

use half::f16;

use super::quant::{Q4_BLOCK, Q8Acts};

/// Weight rows per register tile.
pub const TILE_ROWS: usize = 4;
/// Tokens per register tile.
pub const TILE_TOKENS: usize = 3;

pub type Tile = [[f32; TILE_TOKENS]; TILE_ROWS];

const UNKNOWN: u8 = 0;
const SIMD: u8 = 1;
const SCALAR: u8 = 2;
static MODE: AtomicU8 = AtomicU8::new(UNKNOWN);

/// Whether the AVX2+FMA(+F16C) kernels are in use. Detected once; `KILN_NO_SIMD=1`
/// forces the scalar path.
pub fn simd_enabled() -> bool {
    match MODE.load(Ordering::Relaxed) {
        SIMD => true,
        SCALAR => false,
        _ => {
            let on = detect_simd() && std::env::var_os("KILN_NO_SIMD").is_none();
            MODE.store(if on { SIMD } else { SCALAR }, Ordering::Relaxed);
            on
        }
    }
}

/// Force the scalar kernels on or off (benchmarks and tests). Enabling is
/// ignored on CPUs without AVX2+FMA.
pub fn set_simd_enabled(on: bool) {
    let on = on && detect_simd();
    MODE.store(if on { SIMD } else { SCALAR }, Ordering::Relaxed);
}

fn detect_simd() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `a · b`.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if simd_enabled() {
        // SAFETY: AVX2 and FMA were detected at runtime.
        return unsafe { avx2::dot(a, b) };
    }
    scalar::dot(a, b)
}

/// All `TILE_ROWS × TILE_TOKENS` dot products between weight rows `w` and
/// activation rows `x` (all of equal length).
#[inline]
pub fn tile(w: [&[f32]; TILE_ROWS], x: [&[f32]; TILE_TOKENS]) -> Tile {
    let k = w[0].len();
    assert!(w.iter().all(|r| r.len() == k) && x.iter().all(|r| r.len() == k));
    #[cfg(target_arch = "x86_64")]
    if simd_enabled() {
        // SAFETY: AVX2 and FMA were detected at runtime; lengths checked above.
        return unsafe { avx2::tile(w, x) };
    }
    scalar::tile(w, x)
}

/// `Σ q[i]·x[i]` for an int8 weight row (the caller applies the row scale).
#[inline]
pub fn dot_q8(q: &[i8], x: &[f32]) -> f32 {
    assert_eq!(q.len(), x.len());
    #[cfg(target_arch = "x86_64")]
    if simd_enabled() {
        // SAFETY: AVX2 and FMA were detected at runtime; lengths checked above.
        return unsafe { avx2::dot_q8(q, x) };
    }
    scalar::dot_q8(q, x)
}

/// Dot product of a packed 4-bit row with int8-quantized activations, in
/// integer arithmetic (the decode hot path for Q4 weights).
#[inline]
pub fn dot_q4_q8(packed: &[u8], scales: &[f16], group: usize, x: &Q8Acts) -> f32 {
    assert!(group.is_multiple_of(Q4_BLOCK) && x.q.len().is_multiple_of(group));
    assert!(packed.len() * 2 == x.q.len() && scales.len() * group == x.q.len());
    #[cfg(target_arch = "x86_64")]
    if simd_enabled() {
        // SAFETY: AVX2, FMA and F16C were detected at runtime; shapes checked above.
        return unsafe { avx2::dot_q4_q8(packed, scales, group, x) };
    }
    scalar::dot_q4_q8(packed, scales, group, x)
}

pub(crate) mod scalar {
    use half::f16;

    use super::{Q4_BLOCK, Q8Acts, TILE_ROWS, TILE_TOKENS, Tile};

    /// Dot product with 8 independent accumulators.
    ///
    /// Float addition is not associative, so the compiler must keep a naive
    /// `sum += a*b` loop strictly serial: one add every ~4 cycles of latency.
    /// Splitting the sum into 8 lanes breaks that dependency chain and lets
    /// LLVM map the lanes onto a vector register.
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        let (a8, a_tail) = a.as_chunks::<8>();
        let (b8, b_tail) = b.as_chunks::<8>();
        let mut acc = [0.0f32; 8];
        for (x, y) in a8.iter().zip(b8) {
            for ((s, &xi), &yi) in acc.iter_mut().zip(x).zip(y) {
                *s += xi * yi;
            }
        }
        let tail: f32 = a_tail.iter().zip(b_tail).map(|(x, y)| x * y).sum();
        acc.iter().sum::<f32>() + tail
    }

    /// Portable tile: independent dot products. A register-tiled version in
    /// plain Rust was measured 1.6-4× *slower* (LLVM scalarizes the 4×3×8
    /// accumulator block instead of keeping it in vector registers, see
    /// docs/perf-log.md), so tiling is left to the explicit SIMD kernel.
    pub fn tile(w: [&[f32]; TILE_ROWS], x: [&[f32]; TILE_TOKENS]) -> Tile {
        std::array::from_fn(|r| std::array::from_fn(|t| dot(w[r], x[t])))
    }

    pub fn dot_q8(q: &[i8], x: &[f32]) -> f32 {
        let (q8, q_tail) = q.as_chunks::<8>();
        let (x8, x_tail) = x.as_chunks::<8>();
        let mut acc = [0.0f32; 8];
        for (qc, xc) in q8.iter().zip(x8) {
            for ((s, &qi), &xi) in acc.iter_mut().zip(qc).zip(xc) {
                *s += f32::from(qi) * xi;
            }
        }
        let tail: f32 = q_tail
            .iter()
            .zip(x_tail)
            .map(|(&q, x)| f32::from(q) * x)
            .sum();
        acc.iter().sum::<f32>() + tail
    }

    pub fn dot_q4_q8(packed: &[u8], scales: &[f16], group: usize, x: &Q8Acts) -> f32 {
        let per_group = group / Q4_BLOCK;
        let mut sum = 0.0;
        for (b, (bytes, xq)) in packed
            .as_chunks::<{ Q4_BLOCK / 2 }>()
            .0
            .iter()
            .zip(x.q.as_chunks::<Q4_BLOCK>().0)
            .enumerate()
        {
            let (x_lo, x_hi) = xq.split_at(Q4_BLOCK / 2);
            let dot: i32 = bytes
                .iter()
                .zip(x_lo)
                .zip(x_hi)
                .map(|((&n, &l), &h)| {
                    (i32::from(n & 0x0F) - 8) * i32::from(l)
                        + (i32::from(n >> 4) - 8) * i32::from(h)
                })
                .sum();
            let s = scales[b / per_group].to_f32();
            sum += s * x.d[b] * dot as f32;
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use std::arch::x86_64::*;

    use half::f16;

    use super::{Q4_BLOCK, Q8Acts, TILE_ROWS, TILE_TOKENS, Tile};

    /// Horizontal sum of the 8 lanes.
    #[target_feature(enable = "avx2,fma")]
    pub(super) fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
    }

    /// Load 8 floats from `s[i..i + 8]`.
    ///
    /// # Safety
    /// `i + 8 <= s.len()` and the CPU supports AVX.
    #[inline(always)]
    pub(super) unsafe fn load(s: &[f32], i: usize) -> __m256 {
        debug_assert!(i + 8 <= s.len());
        // SAFETY: in bounds per the caller contract; `loadu` has no alignment requirement.
        unsafe { _mm256_loadu_ps(s.as_ptr().add(i)) }
    }

    /// 4 accumulators × 8 lanes. FMA has 4-cycle latency on two ports, so a
    /// single accumulator would run at 1/8 of peak; 4 independent chains are
    /// enough here because 2 loads per FMA cap the loop first.
    ///
    /// # Safety
    /// CPU supports AVX2+FMA; `a.len() == b.len()`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let k = a.len();
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        // SAFETY (all loads): `i + 32 <= k` / `i + 8 <= k` guard every access.
        while i + 32 <= k {
            for (j, s) in acc.iter_mut().enumerate() {
                let o = i + 8 * j;
                *s = unsafe { _mm256_fmadd_ps(load(a, o), load(b, o), *s) };
            }
            i += 32;
        }
        while i + 8 <= k {
            acc[0] = unsafe { _mm256_fmadd_ps(load(a, i), load(b, i), acc[0]) };
            i += 8;
        }
        let v = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        let tail: f32 = a[i..].iter().zip(&b[i..]).map(|(x, y)| x * y).sum();
        hsum(v) + tail
    }

    /// 4 rows × 3 tokens register tile: 12 accumulators + 3 activation
    /// vectors + 1 weight vector = all 16 ymm registers, no spills. Per 8-wide
    /// step: 7 loads feed 12 FMAs (vs 2 loads per FMA in `dot`).
    ///
    /// # Safety
    /// CPU supports AVX2+FMA; all slices have the same length.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn tile(w: [&[f32]; TILE_ROWS], x: [&[f32]; TILE_TOKENS]) -> Tile {
        let k = w[0].len();
        let mut acc = [[_mm256_setzero_ps(); TILE_TOKENS]; TILE_ROWS];
        let mut i = 0;
        while i + 8 <= k {
            // SAFETY: `i + 8 <= k` and every slice has length `k`.
            let xv: [__m256; TILE_TOKENS] = std::array::from_fn(|t| unsafe { load(x[t], i) });
            for (acc_r, wr) in acc.iter_mut().zip(&w) {
                let wv = unsafe { load(wr, i) };
                for (s, &xt) in acc_r.iter_mut().zip(&xv) {
                    *s = _mm256_fmadd_ps(wv, xt, *s);
                }
            }
            i += 8;
        }
        std::array::from_fn(|r| {
            std::array::from_fn(|t| {
                let tail: f32 = w[r][i..].iter().zip(&x[t][i..]).map(|(a, b)| a * b).sum();
                hsum(acc[r][t]) + tail
            })
        })
    }

    /// 8 int8 weights at `q[i..i + 8]`, sign-extended and converted to f32.
    ///
    /// # Safety
    /// `i + 8 <= q.len()` and the CPU supports AVX2.
    #[inline(always)]
    unsafe fn load_q8(q: &[i8], i: usize) -> __m256 {
        debug_assert!(i + 8 <= q.len());
        // SAFETY: 8 bytes in bounds and AVX2 available per the caller contract.
        unsafe {
            let bytes = _mm_loadl_epi64(q.as_ptr().add(i).cast());
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes))
        }
    }

    /// Int8 weights × f32 activations. Weights are widened in registers, so
    /// memory traffic is 1 byte per weight instead of 4.
    ///
    /// # Safety
    /// CPU supports AVX2+FMA; `q.len() == x.len()`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q8(q: &[i8], x: &[f32]) -> f32 {
        let k = q.len();
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        // SAFETY (all loads): `i + 32 <= k` / `i + 8 <= k` guard every access.
        while i + 32 <= k {
            for (j, s) in acc.iter_mut().enumerate() {
                let o = i + 8 * j;
                *s = unsafe { _mm256_fmadd_ps(load_q8(q, o), load(x, o), *s) };
            }
            i += 32;
        }
        while i + 8 <= k {
            acc[0] = unsafe { _mm256_fmadd_ps(load_q8(q, i), load(x, i), acc[0]) };
            i += 8;
        }
        let v = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        let tail: f32 = q[i..]
            .iter()
            .zip(&x[i..])
            .map(|(&q, x)| f32::from(q) * x)
            .sum();
        hsum(v) + tail
    }

    /// 4-bit weights × int8 activations, 32 products per block in three
    /// integer instructions:
    ///
    /// ```text
    /// 16 bytes ─► [lo nibbles | hi nibbles] as 32 × u8 (0..15) ─┐
    /// 32 × i8 activations ───────────────────────────────────────┴► maddubs: 16 × i16 pair sums
    ///   ─► madd with 1s: 8 × i32 ─► f32 ─► FMA by (weight scale · activation scale)
    /// ```
    ///
    /// `maddubs` multiplies unsigned by signed bytes, which is why the nibbles
    /// stay unsigned (0..15); the `- 8` is a second `maddubs` of 8s against the
    /// same activations, subtracted in i16. An earlier version applied it as a
    /// precomputed scalar per block: that serial scalar chain cost ~25% of the
    /// kernel's time.
    ///
    /// # Safety
    /// CPU supports AVX2+FMA+F16C; shapes as checked by the safe wrapper.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q4_q8(packed: &[u8], scales: &[f16], group: usize, x: &Q8Acts) -> f32 {
        let low_nibble = _mm256_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        let ones = _mm256_set1_epi16(1);
        let per_group = group / Q4_BLOCK;
        // One group: Σ over its blocks of (n·q dot products × activation
        // scale), as 8 f32 lanes, and the group's weight scale.
        let group_sum = |g: usize| {
            // `set1` (a full-register write) rather than `cvtsi32`: LLVM
            // lowered the latter to a `vpinsrw` merging into whatever register
            // held `offset`, chaining every group to the previous one's
            // scalar add (~12 cycles per block instead of ~4).
            let bits = scales[g].to_bits() as i16;
            let s = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(bits)));
            let mut sum = _mm256_setzero_ps();
            for b in g * per_group..(g + 1) * per_group {
                // SAFETY: block `b` is in bounds of `packed` (16 bytes) and of
                // `x.q` (32 bytes) per the shape contract.
                let (bytes, xq) = unsafe {
                    (
                        _mm_loadu_si128(packed.as_ptr().add(b * 16).cast()),
                        _mm256_loadu_si256(x.q.as_ptr().add(b * Q4_BLOCK).cast()),
                    )
                };
                let nibbles = _mm256_and_si256(
                    _mm256_set_m128i(_mm_srli_epi16::<4>(bytes), bytes),
                    low_nibble,
                );
                // (n - 8)·q = n·q - 8·q, both halves as unsigned × signed
                // pair sums; |pair| ≤ 2·15·127 so i16 cannot overflow.
                let pairs = _mm256_sub_epi16(
                    _mm256_maddubs_epi16(nibbles, xq),
                    _mm256_maddubs_epi16(eight, xq),
                );
                let dots = _mm256_cvtepi32_ps(_mm256_madd_epi16(pairs, ones));
                sum = _mm256_fmadd_ps(dots, _mm256_set1_ps(x.d[b]), sum);
            }
            (sum, _mm256_set1_ps(s))
        };
        // Two named accumulators, two groups per iteration, so consecutive
        // FMAs do not serialize and both accumulators stay in registers.
        let (mut acc0, mut acc1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let mut g = 0;
        while g + 2 <= scales.len() {
            let (sum0, s0) = group_sum(g);
            acc0 = _mm256_fmadd_ps(sum0, s0, acc0);
            let (sum1, s1) = group_sum(g + 1);
            acc1 = _mm256_fmadd_ps(sum1, s1, acc1);
            g += 2;
        }
        if g < scales.len() {
            let (sum0, s0) = group_sum(g);
            acc0 = _mm256_fmadd_ps(sum0, s0, acc0);
        }
        hsum(_mm256_add_ps(acc0, acc1))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn data(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761) | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    pub(crate) fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-4 * (1.0 + b.abs())
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_kernels_match_scalar() {
        if !detect_simd() {
            eprintln!("no AVX2+FMA on this CPU; nothing to compare");
            return;
        }
        // Lengths around every unroll boundary (8, 32) plus the model's sizes.
        for k in [0, 1, 7, 8, 9, 31, 32, 33, 63, 100, 576, 1536] {
            let w: Vec<Vec<f32>> = (0..TILE_ROWS).map(|r| data(k, r as u32 + 1)).collect();
            let x: Vec<Vec<f32>> = (0..TILE_TOKENS).map(|t| data(k, t as u32 + 9)).collect();
            let want = scalar::dot(&w[0], &x[0]);
            // SAFETY: feature detected above.
            let got = unsafe { avx2::dot(&w[0], &x[0]) };
            assert!(close(got, want), "dot k={k}: {got} vs {want}");

            let wr = std::array::from_fn(|r| w[r].as_slice());
            let xr = std::array::from_fn(|t| x[t].as_slice());
            let want = scalar::tile(wr, xr);
            // SAFETY: feature detected above; equal lengths.
            let got = unsafe { avx2::tile(wr, xr) };
            for (gr, wr) in got.iter().zip(&want) {
                for (g, w) in gr.iter().zip(wr) {
                    assert!(close(*g, *w), "tile k={k}: {g} vs {w}");
                }
            }

            let q: Vec<i8> = w[1].iter().map(|v| (v * 250.0) as i8).collect();
            let want = scalar::dot_q8(&q, &x[1]);
            // SAFETY: feature detected above; equal lengths.
            let got = unsafe { avx2::dot_q8(&q, &x[1]) };
            assert!(close(got, want), "dot_q8 k={k}: {got} vs {want}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_q4_kernel_matches_scalar() {
        if !detect_simd() {
            return;
        }
        for (cols, group) in [(32, 32), (96, 32), (576, 32), (576, 64), (1536, 64)] {
            let w = data(cols, 11);
            let x = data(cols, 12);
            let q = crate::tensor::quant::Q4::quantize(&w, cols, group).unwrap();
            let mut deq = vec![0.0; cols];
            q.dequantize_row(0, cols, &mut deq);
            let want = scalar::dot(&deq, &x);

            // SIMD and scalar do the same integer sums, so they agree to float
            // round-off; quantizing the activations costs < 1% vs the f32 dot.
            let acts = Q8Acts::quantize(&x);
            let int_scalar = scalar::dot_q4_q8(&q.packed, &q.scales, group, &acts);
            // SAFETY: feature detected above; shapes come from `quantize`.
            let int_simd = unsafe { avx2::dot_q4_q8(&q.packed, &q.scales, group, &acts) };
            assert!(close(int_simd, int_scalar), "{int_simd} vs {int_scalar}");
            let scale = deq.iter().zip(&x).map(|(w, x)| (w * x).abs()).sum::<f32>();
            assert!(
                (int_scalar - want).abs() < 0.01 * scale,
                "{int_scalar} vs {want}"
            );
        }
    }
}
