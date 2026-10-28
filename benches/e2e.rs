//! End-to-end forward-pass throughput on the real model.
//!
//! - `prefill_128`: a 128-token prompt into an empty cache.
//! - `decode_at_128`: one token on top of a 128-token cache (the steady-state
//!   cost of generating a token).
//!
//! ```text
//! cargo bench --bench e2e
//! RAYON_NUM_THREADS=4 KILN_QUANT=q4 cargo bench --bench e2e
//! ```

use std::path::PathBuf;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use kiln::model::{Logits, Model};
use kiln::tensor::Quant;

const PROMPT_LEN: usize = 128;

fn bench(c: &mut Criterion) {
    let dir = std::env::var_os("KILN_MODEL_DIR").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/smollm2-135m-instruct"),
        PathBuf::from,
    );
    let quant: Quant = std::env::var("KILN_QUANT")
        .map_or(Ok(Quant::F32), |q| q.parse())
        .expect("KILN_QUANT");
    let model = Model::load_quantized(&dir, quant).expect("model (run scripts/download-model.sh)");
    let prompt: Vec<u32> = (0..PROMPT_LEN as u32).map(|i| 100 + i * 37).collect();

    let mut group = c.benchmark_group(format!("e2e_{quant}"));
    group
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(10);

    let mut cache = model.new_cache(PROMPT_LEN + 1);
    group.throughput(Throughput::Elements(PROMPT_LEN as u64));
    group.bench_function("prefill_128", |b| {
        b.iter(|| {
            cache.truncate(0);
            model.forward(&prompt, &mut cache, Logits::Last).unwrap()
        });
    });

    cache.truncate(0);
    model.forward(&prompt, &mut cache, Logits::Last).unwrap();
    group.throughput(Throughput::Elements(1));
    group.bench_function("decode_at_128", |b| {
        b.iter(|| {
            cache.truncate(PROMPT_LEN);
            model.forward(&[42], &mut cache, Logits::Last).unwrap()
        });
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
