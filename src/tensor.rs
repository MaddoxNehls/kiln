//! Weight matrices and the matrix-vector kernel: the one hot spot.
//!
//! Over 95% of inference FLOPs are `x · Wᵀ` against a weight matrix, so everything
//! that makes inference fast later (blocking, SIMD, int8/int4 weights) lives
//! behind [`Matrix::matmul`]. The model code never touches weight storage.
//!
//! Layout: weights stay in PyTorch's `nn.Linear` layout, `[out_features,
//! in_features]` row-major. Then each output element is a dot product of one
//! *contiguous* weight row with the input vector: both operands stream through
//! the cache sequentially and no transpose is ever materialized.

use rayon::prelude::*;

/// A dense row-major f32 matrix of shape `[rows, cols]`.
pub struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

/// Output rows handed to one rayon task. Large enough to amortize scheduling,
/// small enough to balance across cores even for the 576-row projections.
const ROWS_PER_TASK: usize = 16;

impl Matrix {
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "matrix data does not match shape");
        Self { data, rows, cols }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// `out[n, rows] = x[n, cols] · selfᵀ` for a batch of `n` input rows.
    ///
    /// Work is split over *weight rows*, not tokens: each weight row is loaded
    /// once and reused for every token in the batch. In decode (`n == 1`) this
    /// is a pure memory-bandwidth-bound mat-vec; in prefill the reuse is what
    /// makes processing a prompt much faster per token than generating.
    pub fn matmul(&self, x: &[f32], out: &mut [f32]) {
        let n = x.len() / self.cols;
        assert_eq!(
            x.len(),
            n * self.cols,
            "input is not a whole number of rows"
        );
        assert_eq!(out.len(), n * self.rows, "output has the wrong shape");

        if n == 1 {
            // A [1, rows] output is the same memory as its transpose.
            self.matmul_transposed(x, n, out);
            return;
        }
        // Each task owns a contiguous block of output *rows of W*, so compute
        // the transposed result [rows, n] and transpose once at the end.
        let mut out_t = vec![0.0; n * self.rows];
        self.matmul_transposed(x, n, &mut out_t);
        for (r, row) in out_t.chunks_exact(n).enumerate() {
            for (t, &v) in row.iter().enumerate() {
                out[t * self.rows + r] = v;
            }
        }
    }

    /// `out_t[rows, n]`: row `r` holds `W[r] · x[t]` for every token `t`.
    fn matmul_transposed(&self, x: &[f32], n: usize, out_t: &mut [f32]) {
        out_t
            .par_chunks_mut(n * ROWS_PER_TASK)
            .enumerate()
            .for_each(|(block, out_block)| {
                let first_row = block * ROWS_PER_TASK;
                for (i, outs) in out_block.chunks_exact_mut(n).enumerate() {
                    let w = self.row(first_row + i);
                    for (o, xt) in outs.iter_mut().zip(x.chunks_exact(self.cols)) {
                        *o = dot(w, xt);
                    }
                }
            });
    }
}

/// Dot product with 8 independent accumulators.
///
/// Float addition is not associative, so the compiler must keep a naive
/// `sum += a*b` loop strictly serial: one add every ~4 cycles of latency.
/// Splitting the sum into 8 lanes breaks that dependency chain and lets LLVM
/// map the lanes onto a vector register. (Explicit SIMD is milestone 4.)
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(w: &Matrix, x: &[f32]) -> Vec<f32> {
        let n = x.len() / w.cols;
        let mut out = vec![0.0; n * w.rows];
        for t in 0..n {
            for r in 0..w.rows {
                out[t * w.rows + r] = (0..w.cols)
                    .map(|c| w.row(r)[c] as f64 * x[t * w.cols + c] as f64)
                    .sum::<f64>() as f32;
            }
        }
        out
    }

    #[test]
    fn matmul_matches_naive_for_odd_shapes() {
        // Odd sizes exercise the dot tail and a partial final row block.
        let (rows, cols) = (37, 29);
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7) % 13) as f32 - 6.0)
            .collect();
        let w = Matrix::new(data, rows, cols);
        for n in [1, 3] {
            let x: Vec<f32> = (0..n * cols).map(|i| (i % 5) as f32 * 0.5 - 1.0).collect();
            let mut out = vec![0.0; n * rows];
            w.matmul(&x, &mut out);
            for (a, b) in out.iter().zip(naive(&w, &x)) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
        }
    }
}
