//! The Llama decoder-only transformer (SmolLM2, Llama 2/3, Qwen2-style without biases).
//!
//! ```text
//! tokens ─► embed ─► [ x += Attn(RMSNorm(x)) ; x += MLP(RMSNorm(x)) ] × L ─► RMSNorm ─► lm_head ─► logits
//! ```
//!
//! Activations for a batch of `n` tokens are row-major `[n, hidden]` buffers.
//! A whole prompt goes through in one call (prefill), so every weight matrix is
//! streamed from memory once per prompt rather than once per prompt token.

use std::path::Path;

use anyhow::{Result, ensure};
use rayon::prelude::*;

use crate::config::Config;
use crate::kv_cache::{KvCache, LayerView};
use crate::ops::{Rope, add_assign, axpy, rms_norm, softmax, swiglu};
use crate::tensor::{Matrix, Quant, dot};
use crate::tokenizer::TokenId;
use crate::weights::WeightFile;

pub struct Model {
    config: Config,
    /// `[vocab, hidden]`. With tied embeddings this doubles as the output
    /// projection: logits are dot products of the final hidden state with
    /// every token's embedding.
    embed: Matrix,
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
    /// `None` when `tie_word_embeddings` (saves 28M params on SmolLM2-135M).
    lm_head: Option<Matrix>,
    rope: Rope,
}

/// Projections that read the same input are fused into one matrix at load
/// time (rows stacked): one matmul, one parallel region and one pass over the
/// input instead of three (Q, K, V) or two (gate, up).
struct Layer {
    attn_norm: Vec<f32>,
    wqkv: Matrix, // [(heads + 2·kv_heads)·head_dim, hidden]: Q rows, then K, then V
    wo: Matrix,   // [hidden, heads·head_dim]
    mlp_norm: Vec<f32>,
    w_gate_up: Matrix, // [2·ffn, hidden]: gate rows, then up
    w_down: Matrix,    // [hidden, ffn]
}

/// Minimum multiply-adds per parallel attention task.
const ATTN_MIN_TASK: usize = 1 << 15;

/// Which positions to compute logits for. Generation only needs the last one,
/// which skips a `[vocab, hidden]` matmul per prompt token during prefill.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Logits {
    Last,
    All,
}

impl Model {
    /// Load `config.json` and `model.safetensors` from a model directory, in f32.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_quantized(dir, Quant::F32)
    }

    /// Load and store every weight matrix (embeddings included) in `quant`
    /// format, quantizing from the bf16 checkpoint tensor by tensor. Norm
    /// weights stay f32: they are tiny and multiply every activation.
    pub fn load_quantized(dir: impl AsRef<Path>, quant: Quant) -> Result<Self> {
        let dir = dir.as_ref();
        let config = Config::from_file(dir.join("config.json"))?;
        let file = WeightFile::open(dir.join("model.safetensors"))?;
        let t = file.tensors()?;

        let (d, ffn, vocab) = (
            config.hidden_size,
            config.intermediate_size,
            config.vocab_size,
        );
        let q_dim = config.num_attention_heads * config.head_dim();
        let kv_dim = config.num_kv_heads() * config.head_dim();

        // Layers are independent, so widen/quantize them in parallel.
        let layers = (0..config.num_hidden_layers)
            .into_par_iter()
            .map(|i| -> Result<Layer> {
                let p = |name: &str| format!("model.layers.{i}.{name}.weight");
                let (q, k, v) = (
                    p("self_attn.q_proj"),
                    p("self_attn.k_proj"),
                    p("self_attn.v_proj"),
                );
                let (gate, up) = (p("mlp.gate_proj"), p("mlp.up_proj"));
                Ok(Layer {
                    attn_norm: t.load(&p("input_layernorm"), &[d])?,
                    wqkv: t.load_matrix(&[(&q, q_dim), (&k, kv_dim), (&v, kv_dim)], d, quant)?,
                    wo: t.load_matrix(&[(&p("self_attn.o_proj"), d)], q_dim, quant)?,
                    mlp_norm: t.load(&p("post_attention_layernorm"), &[d])?,
                    w_gate_up: t.load_matrix(&[(&gate, ffn), (&up, ffn)], d, quant)?,
                    w_down: t.load_matrix(&[(&p("mlp.down_proj"), d)], ffn, quant)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // The embedding table doubles as the output projection, where 4-bit
        // error lands directly on the logits: under Q4 it is kept at Q8 (as
        // llama.cpp keeps its output tensor at higher precision). Measured on
        // WikiText-2: perplexity 31.8 with a Q4 embedding vs 28.6 with Q8.
        let embed_quant = match quant {
            Quant::Q4 { .. } => Quant::Q8,
            other => other,
        };
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(t.load_matrix(&[("lm_head.weight", vocab)], d, embed_quant)?)
        };

        let rope = Rope::new(
            config.head_dim(),
            config.rope_theta,
            config.rope_position_scale()?,
            config.max_position_embeddings,
        );
        Ok(Self {
            embed: t.load_matrix(&[("model.embed_tokens.weight", vocab)], d, embed_quant)?,
            final_norm: t.load("model.norm.weight", &[d])?,
            layers,
            lm_head,
            rope,
            config,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Allocate a KV cache for up to `capacity` positions (clamped to the
    /// model's trained context length).
    pub fn new_cache(&self, capacity: usize) -> KvCache {
        KvCache::new(&self.config, capacity.min(self.rope.max_positions()))
    }

    /// Weight storage format of the transformer layers.
    pub fn quant(&self) -> Quant {
        self.layers[0].wqkv.quant()
    }

    /// Bytes of weights, i.e. what one decode step has to stream from memory.
    pub fn weight_bytes(&self) -> usize {
        let layers: usize = self
            .layers
            .iter()
            .map(|l| {
                l.wqkv.bytes()
                    + l.wo.bytes()
                    + l.w_gate_up.bytes()
                    + l.w_down.bytes()
                    + 4 * (l.attn_norm.len() + l.mlp_norm.len())
            })
            .sum();
        layers
            + self.embed.bytes()
            + self.lm_head.as_ref().map_or(0, Matrix::bytes)
            + 4 * self.final_norm.len()
    }

    fn lm_head(&self) -> &Matrix {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }

    /// Run `tokens` through the model, continuing from the `cache.len()`
    /// positions already cached, and append their K/V to the cache.
    ///
    /// Returns row-major logits: `[1, vocab]` for [`Logits::Last`] or
    /// `[tokens.len(), vocab]` for [`Logits::All`].
    pub fn forward(
        &self,
        tokens: &[TokenId],
        cache: &mut KvCache,
        logits: Logits,
    ) -> Result<Vec<f32>> {
        // Hop onto a pool thread once per forward pass. Called from outside
        // the pool, each of the ~200 parallel loops in a decode step would
        // inject a job and park the caller on an OS event; from inside, they
        // are cheap fork-joins through the work-stealing deques.
        rayon::scope(|_| self.forward_in_pool(tokens, cache, logits))
    }

    fn forward_in_pool(
        &self,
        tokens: &[TokenId],
        cache: &mut KvCache,
        logits: Logits,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let (n, d) = (tokens.len(), cfg.hidden_size);
        let start = cache.len();
        ensure!(n > 0, "forward called with no tokens");
        ensure!(
            start + n <= cache.capacity(),
            "context full: {start} cached + {n} new > capacity {}",
            cache.capacity()
        );

        let mut x = vec![0.0; n * d];
        for (&tok, row) in tokens.iter().zip(x.chunks_exact_mut(d)) {
            ensure!(
                (tok as usize) < cfg.vocab_size,
                "token id {tok} out of range"
            );
            self.embed.row_into(tok as usize, row);
        }

        let mut buf = Buffers::new(cfg, n);
        for (l, layer) in self.layers.iter().enumerate() {
            // --- self-attention block (pre-norm, residual) ---
            rms_norm(&x, &layer.attn_norm, cfg.rms_norm_eps, &mut buf.normed);
            layer.wqkv.matmul(&buf.normed, &mut buf.qkv);
            let (q_dim, kv_dim) = (buf.q.len() / n, buf.k.len() / n);
            for (t, row) in buf.qkv.chunks_exact_mut(q_dim + 2 * kv_dim).enumerate() {
                let (q, kv) = row.split_at_mut(q_dim);
                let (k, v) = kv.split_at_mut(kv_dim);
                self.rope.apply(q, start + t);
                self.rope.apply(k, start + t);
                buf.q[t * q_dim..(t + 1) * q_dim].copy_from_slice(q);
                buf.k[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(k);
                buf.v[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(v);
            }
            cache.write(l, start, &buf.k, &buf.v);
            self.attention(&buf.q, cache.layer(l), start, &mut buf.attn);
            layer.wo.matmul(&buf.attn, &mut buf.proj);
            add_assign(&mut x, &buf.proj);

            // --- SwiGLU MLP block: down(silu(gate(x)) * up(x)) ---
            rms_norm(&x, &layer.mlp_norm, cfg.rms_norm_eps, &mut buf.normed);
            layer.w_gate_up.matmul(&buf.normed, &mut buf.gate_up);
            let ffn = cfg.intermediate_size;
            for (row, out) in buf
                .gate_up
                .chunks_exact_mut(2 * ffn)
                .zip(buf.gate.chunks_exact_mut(ffn))
            {
                let (gate, up) = row.split_at_mut(ffn);
                swiglu(gate, up);
                out.copy_from_slice(gate);
            }
            layer.w_down.matmul(&buf.gate, &mut buf.proj);
            add_assign(&mut x, &buf.proj);
        }
        cache.advance(n);

        let hidden = match logits {
            Logits::Last => &x[(n - 1) * d..],
            Logits::All => &x[..],
        };
        let rows = hidden.len() / d;
        let mut normed = vec![0.0; rows * d];
        rms_norm(hidden, &self.final_norm, cfg.rms_norm_eps, &mut normed);
        let mut out = vec![0.0; rows * cfg.vocab_size];
        self.lm_head().matmul(&normed, &mut out);
        Ok(out)
    }

    /// Causal grouped-query attention for the `n` query rows in `q`
    /// (`[n, heads, head_dim]`), whose own K/V are already in `cache` at
    /// positions `start..start + n`.
    ///
    /// GQA: query head `h` reads K/V head `h / (heads / kv_heads)`. SmolLM2 has
    /// 9 query heads sharing 3 K/V heads, which cuts the KV cache 3×.
    ///
    /// The causal mask is implicit: the query at position `p` only ever reads
    /// cached positions `0..=p`, so there is nothing to mask out.
    fn attention(&self, q: &[f32], cache: LayerView<'_>, start: usize, out: &mut [f32]) {
        let cfg = &self.config;
        let (heads, hd) = (cfg.num_attention_heads, cfg.head_dim());
        let group = heads / cfg.num_kv_heads();
        let scale = 1.0 / (hd as f32).sqrt();

        // One item per (token, head) pair; `i = t * heads + h` matches the
        // [n, heads, head_dim] layout of both `q` and `out`. Items are batched
        // so each task does at least ATTN_MIN_TASK multiply-adds: in decode at
        // short context, all 9 heads together are only a few microseconds.
        let per_item = 2 * (start + q.len() / (heads * hd)) * hd;
        let min_items = ATTN_MIN_TASK.div_ceil(per_item);
        out.par_chunks_mut(hd)
            .enumerate()
            .with_min_len(min_items)
            .for_each(|(i, o)| {
                let (t, h) = (i / heads, i % heads);
                let visible = start + t + 1;
                let qh = &q[i * hd..(i + 1) * hd];
                let keys = cache.keys(h / group, visible);
                let values = cache.values(h / group, visible);

                // Scaled dot-product scores; the 1/sqrt(d) keeps their variance
                // ~1 so softmax does not saturate as head_dim grows.
                let mut scores: Vec<f32> =
                    keys.chunks_exact(hd).map(|k| dot(qh, k) * scale).collect();
                softmax(&mut scores);
                o.fill(0.0);
                for (&w, v) in scores.iter().zip(values.chunks_exact(hd)) {
                    axpy(o, w, v);
                }
            });
    }
}

/// Per-forward scratch activations, allocated once and reused by every layer.
struct Buffers {
    normed: Vec<f32>,
    qkv: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    gate_up: Vec<f32>,
    gate: Vec<f32>,
}

impl Buffers {
    fn new(cfg: &Config, n: usize) -> Self {
        let q_dim = cfg.num_attention_heads * cfg.head_dim();
        let kv_dim = cfg.num_kv_heads() * cfg.head_dim();
        let zeros = |len: usize| vec![0.0; n * len];
        Self {
            normed: zeros(cfg.hidden_size),
            qkv: zeros(q_dim + 2 * kv_dim),
            q: zeros(q_dim),
            k: zeros(kv_dim),
            v: zeros(kv_dim),
            attn: zeros(q_dim),
            proj: zeros(cfg.hidden_size),
            gate_up: zeros(2 * cfg.intermediate_size),
            gate: zeros(cfg.intermediate_size),
        }
    }
}
