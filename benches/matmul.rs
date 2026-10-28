//! Matmul throughput on SmolLM2-135M's actual weight shapes.
//!
//! `n = 1` is a decode step (mat-vec, bandwidth bound); `n = 128` is a prefill
//! batch (compute bound once the kernel reuses each weight row).
//!
//! ```text
//! cargo bench --bench matmul
//! RAYON_NUM_THREADS=1 cargo bench --bench matmul
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use kiln::tensor::{Matrix, Quant};

const SHAPES: [(&str, usize, usize); 4] = [
    ("attn_576x576", 576, 576),
    ("ffn_up_1536x576", 1536, 576),
    ("ffn_down_576x1536", 576, 1536),
    ("lm_head_49152x576", 49152, 576),
];

fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).max(1);
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    for quant in [Quant::F32, Quant::Q8, Quant::Q4 { group: 32 }] {
        for n in [1usize, 128] {
            let mut group = c.benchmark_group(format!("matmul_{quant}_n{n}"));
            group
                .warm_up_time(Duration::from_millis(500))
                .measurement_time(Duration::from_secs(2))
                .sample_size(20);
            for (name, rows, cols) in SHAPES {
                let w =
                    Matrix::quantized(pseudo_random(rows * cols, 1), rows, cols, quant).unwrap();
                let x = pseudo_random(n * cols, 2);
                let mut out = vec![0.0; n * rows];
                // Elements = multiply-accumulates, so the reported rate is MAC/s.
                group.throughput(Throughput::Elements((rows * cols * n) as u64));
                group.bench_function(BenchmarkId::from_parameter(name), |b| {
                    b.iter(|| w.matmul(&x, &mut out));
                });
            }
            group.finish();
        }
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
