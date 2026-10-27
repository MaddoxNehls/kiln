//! Element-wise and per-row kernels used by the transformer blocks.
//! All are O(n) next to the O(n·d) matmuls, so they favor clarity.

/// RMSNorm, row by row: `y = x / sqrt(mean(x²) + eps) * weight`.
///
/// Unlike LayerNorm there is no mean subtraction and no bias: only the scale
/// of the activation vector is normalized. Statistics accumulate in f64 so
/// that 576-wide sums do not drift from PyTorch's fp32 reduction tree.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let d = weight.len();
    for (xr, yr) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mean_sq = xr.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / d as f64;
        let inv_rms = (1.0 / (mean_sq + f64::from(eps)).sqrt()) as f32;
        for ((y, &v), &w) in yr.iter_mut().zip(xr).zip(weight) {
            *y = v * inv_rms * w;
        }
    }
}

/// Numerically stable in-place softmax: subtracting the max makes every
/// exponent <= 0, so `exp` cannot overflow, and the result is unchanged
/// because softmax is invariant to adding a constant.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    x.iter_mut().for_each(|v| *v *= inv);
}

/// SwiGLU gate, in place: `gate = silu(gate) * up`, with `silu(g) = g·σ(g)`.
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up) {
        *g = *g / (1.0 + (-*g).exp()) * u;
    }
}

/// `acc += x` (residual connection).
pub fn add_assign(acc: &mut [f32], x: &[f32]) {
    acc.iter_mut().zip(x).for_each(|(a, &b)| *a += b);
}

/// `acc += alpha * x`.
pub fn axpy(acc: &mut [f32], alpha: f32, x: &[f32]) {
    acc.iter_mut().zip(x).for_each(|(a, &b)| *a += alpha * b);
}

/// Rotary position embedding with precomputed `cos`/`sin` tables.
///
/// RoPE encodes position by rotating pairs of q/k dimensions by an angle
/// `pos · θ_i`, with `θ_i = base^(-2i/d)`. Since a rotation by `a` dotted with a
/// rotation by `b` depends only on `a - b`, the attention score `q_m · k_n`
/// depends only on the *relative* offset `m - n`.
///
/// HF Llama pairs dimension `i` with `i + d/2` ("rotate_half"), not adjacent
/// dims `(2i, 2i+1)` as in the original paper. Weights are trained for one
/// convention, so getting this wrong produces fluent-looking garbage.
pub struct Rope {
    half: usize,
    /// `[max_pos, half]`, row `p` holds `cos(p · θ_i)` for each `i`.
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    pub fn new(head_dim: usize, base: f32, position_scale: f32, max_pos: usize) -> Self {
        let half = head_dim / 2;
        // Same arithmetic order as HF (fp32 inv_freq, fp32 angle) so that the
        // angles match bit-for-bit as far as possible.
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / base.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let mut cos = Vec::with_capacity(max_pos * half);
        let mut sin = Vec::with_capacity(max_pos * half);
        for pos in 0..max_pos {
            let p = pos as f32 / position_scale;
            for &f in &inv_freq {
                let angle = f64::from(p * f);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Self { half, cos, sin }
    }

    pub fn max_positions(&self) -> usize {
        self.cos.len() / self.half
    }

    /// Rotate every head in `x` (`[n_heads, head_dim]`) to position `pos`.
    pub fn apply(&self, x: &mut [f32], pos: usize) {
        let cos = &self.cos[pos * self.half..(pos + 1) * self.half];
        let sin = &self.sin[pos * self.half..(pos + 1) * self.half];
        for head in x.chunks_exact_mut(2 * self.half) {
            let (lo, hi) = head.split_at_mut(self.half);
            for (((a, b), &c), &s) in lo.iter_mut().zip(hi).zip(cos).zip(sin) {
                let (x0, x1) = (*a, *b);
                *a = x0 * c - x1 * s;
                *b = x1 * c + x0 * s;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_stable_and_normalized() {
        let mut x = [1000.0, 1001.0, 1002.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn rms_norm_unit_weight_gives_unit_rms() {
        let x = [3.0, -4.0, 0.0, 5.0];
        let mut y = [0.0; 4];
        rms_norm(&x, &[1.0; 4], 0.0, &mut y);
        let rms = (y.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rope_scores_depend_only_on_relative_position() {
        let rope = Rope::new(8, 10_000.0, 1.0, 64);
        let q0: Vec<f32> = (0..8).map(|i| i as f32 * 0.3 - 1.0).collect();
        let k0: Vec<f32> = (0..8).map(|i| 1.0 - i as f32 * 0.2).collect();
        let score = |m: usize, n: usize| {
            let (mut q, mut k) = (q0.clone(), k0.clone());
            rope.apply(&mut q, m);
            rope.apply(&mut k, n);
            crate::tensor::dot(&q, &k)
        };
        assert!((score(5, 2) - score(40, 37)).abs() < 1e-4);
        assert!((score(0, 0) - crate::tensor::dot(&q0, &k0)).abs() < 1e-6);
    }
}
