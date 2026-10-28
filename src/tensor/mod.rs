//! Weight matrices and the matrix-vector kernel: the one hot spot.
//!
//! Over 95% of inference FLOPs are `x · Wᵀ` against a weight matrix, so everything
//! that makes inference fast (blocking, SIMD, int8/int4 weights) lives behind
//! [`Matrix::matmul`]. The model code never touches weight storage.
//!
//! Layout: weights stay in PyTorch's `nn.Linear` layout, `[out_features,
//! in_features]` row-major. Then each output element is a dot product of one
//! *contiguous* weight row with the input vector: both operands stream through
//! the cache sequentially and no transpose is ever materialized. Quantized
//! formats keep the same row-major order, so a row is still one contiguous run.

pub mod kernels;
pub mod quant;

use anyhow::Result;
use rayon::prelude::*;

pub use kernels::dot;
use kernels::{TILE_ROWS, TILE_TOKENS};
pub use quant::Quant;
use quant::{Q4, Q8, Q8Acts};

/// A row-major weight matrix of shape `[rows, cols]`, in one of the storage
/// formats of [`Quant`].
pub struct Matrix {
    rows: usize,
    cols: usize,
    storage: Storage,
}

enum Storage {
    F32(Vec<f32>),
    Q8(Q8),
    Q4(Q4),
}

/// Minimum multiply-adds per parallel task, about 256 KiB of f32 weights in a
/// decode step. Below this, handing work to another core (a cache-line
/// ping-pong plus possibly waking a parked thread) costs more than the work.
const MIN_TASK_MACS: usize = 1 << 16;

/// Tasks per thread: a few, so a core that finishes early can steal work.
const TASKS_PER_THREAD: usize = 4;

/// Tokens per cache block in prefill. A block of activations (64 × 1536 f32 =
/// 384 KiB at worst) stays resident in L2 while every weight tile of the task
/// sweeps over it; each 4-row weight tile (≤ 24 KiB) stays in L1 while it is
/// multiplied against all 64 tokens.
const TOKEN_BLOCK: usize = 64;

impl Matrix {
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "matrix data does not match shape");
        Self {
            rows,
            cols,
            storage: Storage::F32(data),
        }
    }

    /// Build a matrix from f32 data, quantizing it to `quant`.
    pub fn quantized(data: Vec<f32>, rows: usize, cols: usize, quant: Quant) -> Result<Self> {
        assert_eq!(data.len(), rows * cols, "matrix data does not match shape");
        let storage = match quant {
            Quant::F32 => Storage::F32(data),
            Quant::Q8 => Storage::Q8(Q8::quantize(&data, cols)),
            Quant::Q4 { group } => Storage::Q4(Q4::quantize(&data, cols, group)?),
        };
        Ok(Self {
            rows,
            cols,
            storage,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn quant(&self) -> Quant {
        match &self.storage {
            Storage::F32(_) => Quant::F32,
            Storage::Q8(_) => Quant::Q8,
            Storage::Q4(q) => Quant::Q4 { group: q.group },
        }
    }

    /// Bytes of weight storage, including scales.
    pub fn bytes(&self) -> usize {
        match &self.storage {
            Storage::F32(d) => 4 * d.len(),
            Storage::Q8(q) => q.bytes(),
            Storage::Q4(q) => q.bytes(),
        }
    }

    /// Row `r` as f32 (dequantized if needed), e.g. an embedding lookup.
    pub fn row_into(&self, r: usize, out: &mut [f32]) {
        assert_eq!(out.len(), self.cols);
        match &self.storage {
            Storage::F32(d) => out.copy_from_slice(&d[r * self.cols..(r + 1) * self.cols]),
            Storage::Q8(q) => q.dequantize_row(r, self.cols, out),
            Storage::Q4(q) => q.dequantize_row(r, self.cols, out),
        }
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
        let rows_per_task = self.rows_per_task(n);
        // Q4 decode dots against int8 activations; quantize them once here
        // for all rows.
        let acts = (n == 1 && matches!(self.storage, Storage::Q4(_))).then(|| Q8Acts::quantize(x));
        out_t
            .par_chunks_mut(n * rows_per_task)
            .enumerate()
            .for_each(|(block, out_block)| {
                let first_row = block * rows_per_task;
                if n == 1 {
                    for (r, o) in out_block.iter_mut().enumerate() {
                        *o = self.row_dot(first_row + r, x, acts.as_ref());
                    }
                } else {
                    self.tiled_block(x, n, first_row, out_block);
                }
            });
    }

    /// Rows per parallel task: enough tasks to balance the pool, but never so
    /// small that scheduling dominates. A multiple of the tile height.
    fn rows_per_task(&self, n: usize) -> usize {
        let by_work = MIN_TASK_MACS.div_ceil(n * self.cols);
        let by_balance = self
            .rows
            .div_ceil(rayon::current_num_threads() * TASKS_PER_THREAD);
        by_work.max(by_balance).next_multiple_of(TILE_ROWS)
    }

    /// Decode path: one weight row against one activation vector, reading
    /// the weights in their stored format. `acts` is `x` quantized to int8,
    /// required for Q4 storage.
    fn row_dot(&self, r: usize, x: &[f32], acts: Option<&Q8Acts>) -> f32 {
        let c = self.cols;
        match &self.storage {
            Storage::F32(d) => dot(&d[r * c..(r + 1) * c], x),
            Storage::Q8(q) => q.scales[r] * kernels::dot_q8(&q.q[r * c..(r + 1) * c], x),
            Storage::Q4(q) => {
                let groups = c / q.group;
                kernels::dot_q4_q8(
                    &q.packed[r * c / 2..(r + 1) * c / 2],
                    &q.scales[r * groups..(r + 1) * groups],
                    q.group,
                    acts.expect("Q4 rows need quantized activations"),
                )
            }
        }
    }

    /// Rows `first..first + count` as contiguous f32: borrowed for f32
    /// storage, dequantized into `scratch` otherwise.
    fn rows_f32<'a>(&'a self, first: usize, count: usize, scratch: &'a mut Vec<f32>) -> &'a [f32] {
        let c = self.cols;
        if let Storage::F32(d) = &self.storage {
            return &d[first * c..(first + count) * c];
        }
        scratch.resize(count * c, 0.0);
        for (i, out) in scratch.chunks_exact_mut(c).enumerate() {
            self.row_into(first + i, out);
        }
        scratch
    }

    /// Prefill path for one task's rows: cache-blocked over tokens, register-
    /// tiled `TILE_ROWS × TILE_TOKENS` inside a block. Quantized weights are
    /// dequantized one tile at a time into an L1-sized scratch buffer and
    /// reused for the whole token block, so prefill runs the f32 kernel.
    fn tiled_block(&self, x: &[f32], n: usize, first_row: usize, out_block: &mut [f32]) {
        let rows = out_block.len() / n;
        let c = self.cols;
        let token = |t: usize| &x[t * c..(t + 1) * c];
        let mut scratch = Vec::new();
        for t_block in (0..n).step_by(TOKEN_BLOCK) {
            let t_end = (t_block + TOKEN_BLOCK).min(n);
            for r in (0..rows).step_by(TILE_ROWS) {
                let tile_rows = TILE_ROWS.min(rows - r);
                let w_rows = self.rows_f32(first_row + r, tile_rows, &mut scratch);
                let mut t = t_block;
                if tile_rows == TILE_ROWS {
                    let w = std::array::from_fn(|i| &w_rows[i * c..(i + 1) * c]);
                    while t + TILE_TOKENS <= t_end {
                        let tile = kernels::tile(w, std::array::from_fn(|j| token(t + j)));
                        for (i, tile_row) in tile.iter().enumerate() {
                            let o = (r + i) * n + t;
                            out_block[o..o + TILE_TOKENS].copy_from_slice(tile_row);
                        }
                        t += TILE_TOKENS;
                    }
                }
                // Leftovers that do not fill a tile.
                for t in t..t_end {
                    for (i, wr) in w_rows.chunks_exact(c).enumerate() {
                        out_block[(r + i) * n + t] = dot(wr, token(t));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::kernels::tests::data;

    fn naive(w: &Matrix, x: &[f32]) -> Vec<f32> {
        let n = x.len() / w.cols;
        let mut row = vec![0.0; w.cols];
        let mut out = vec![0.0; n * w.rows];
        for r in 0..w.rows {
            w.row_into(r, &mut row);
            for t in 0..n {
                out[t * w.rows + r] = (0..w.cols)
                    .map(|c| row[c] as f64 * x[t * w.cols + c] as f64)
                    .sum::<f64>() as f32;
            }
        }
        out
    }

    fn check(w: &Matrix, label: &str) {
        // n covers decode, partial tiles, a partial token block and more.
        for n in [1, 2, 3, 7, 70] {
            let x = data(n * w.cols, n as u32);
            let mut out = vec![0.0; n * w.rows];
            w.matmul(&x, &mut out);
            // Q4 decode also quantizes the activations to int8, which costs
            // up to ~1% of the dot product's magnitude; everything else is
            // exact up to f32 round-off.
            let tol = if n == 1 && matches!(w.quant(), Quant::Q4 { .. }) {
                2e-2
            } else {
                1e-3
            };
            for (a, b) in out.iter().zip(naive(w, &x)) {
                assert!(
                    (a - b).abs() < tol * (1.0 + b.abs()),
                    "{label} n={n}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn matmul_matches_naive_for_odd_shapes() {
        // Odd sizes exercise the dot tail and partial tiles in both dimensions.
        let (rows, cols) = (37, 29);
        check(&Matrix::new(data(rows * cols, 1), rows, cols), "f32");
    }

    #[test]
    fn quantized_matmul_matches_its_own_dequantization() {
        let (rows, cols) = (37, 128);
        for quant in [Quant::Q8, Quant::Q4 { group: 32 }, Quant::Q4 { group: 64 }] {
            let w = Matrix::quantized(data(rows * cols, 7), rows, cols, quant).unwrap();
            check(&w, &quant.to_string());
        }
    }

    #[test]
    fn scalar_and_simd_paths_agree_on_matmul() {
        // Both paths through the full matmul (tiling, partitioning, dequant).
        let (rows, cols) = (64, 576);
        let x = data(5 * cols, 3);
        for quant in [Quant::F32, Quant::Q8, Quant::Q4 { group: 32 }] {
            let w = Matrix::quantized(data(rows * cols, 9), rows, cols, quant).unwrap();
            let run = |simd| {
                kernels::set_simd_enabled(simd);
                let mut out = vec![0.0; 5 * rows];
                w.matmul(&x, &mut out);
                let mut one = vec![0.0; rows];
                w.matmul(&x[..cols], &mut one);
                kernels::set_simd_enabled(true);
                (out, one)
            };
            let (simd, scalar) = (run(true), run(false));
            for (a, b) in simd
                .0
                .iter()
                .chain(&simd.1)
                .zip(scalar.0.iter().chain(&scalar.1))
            {
                assert!(
                    (a - b).abs() < 1e-3 * (1.0 + b.abs()),
                    "{quant}: {a} vs {b}"
                );
            }
        }
    }
}
