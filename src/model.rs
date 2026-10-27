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
use crate::tensor::{Matrix, dot};
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

struct Layer {
    attn_norm: Vec<f32>,
    wq: Matrix, // [heads·head_dim, hidden]
    wk: Matrix, // [kv_heads·head_dim, hidden]
    wv: Matrix, // [kv_heads·head_dim, hidden]
    wo: Matrix, // [hidden, heads·head_dim]
    mlp_norm: Vec<f32>,
    w_gate: Matrix, // [ffn, hidden]
    w_up: Matrix,   // [ffn, hidden]
    w_down: Matrix, // [hidden, ffn]
}

/// Which positions to compute logits for. Generation only needs the last one,
/// which skips a `[vocab, hidden]` matmul per prompt token during prefill.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Logits {
    Last,
    All,
}

impl Model {
    /// Load `config.json` and `model.safetensors` from a model directory.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
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

        // Layers are independent, so widen bf16 -> f32 for them in parallel.
        let layers = (0..config.num_hidden_layers)
            .into_par_iter()
            .map(|i| -> Result<Layer> {
                let p = format!("model.layers.{i}");
                Ok(Layer {
                    attn_norm: t.load(&format!("{p}.input_layernorm.weight"), &[d])?,
                    wq: t.load_matrix(&format!("{p}.self_attn.q_proj.weight"), q_dim, d)?,
                    wk: t.load_matrix(&format!("{p}.self_attn.k_proj.weight"), kv_dim, d)?,
                    wv: t.load_matrix(&format!("{p}.self_attn.v_proj.weight"), kv_dim, d)?,
                    wo: t.load_matrix(&format!("{p}.self_attn.o_proj.weight"), d, q_dim)?,
                    mlp_norm: t.load(&format!("{p}.post_attention_layernorm.weight"), &[d])?,
                    w_gate: t.load_matrix(&format!("{p}.mlp.gate_proj.weight"), ffn, d)?,
                    w_up: t.load_matrix(&format!("{p}.mlp.up_proj.weight"), ffn, d)?,
                    w_down: t.load_matrix(&format!("{p}.mlp.down_proj.weight"), d, ffn)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(t.load_matrix("lm_head.weight", vocab, d)?)
        };

        let rope = Rope::new(
            config.head_dim(),
            config.rope_theta,
            config.rope_position_scale()?,
            config.max_position_embeddings,
        );
        Ok(Self {
            embed: t.load_matrix("model.embed_tokens.weight", vocab, d)?,
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
        let cfg = &self.config;
        let (n, d) = (tokens.len(), cfg.hidden_size);
        let start = cache.len();
        ensure!(n > 0, "forward called with no tokens");
        ensure!(
            start + n <= cache.capacity(),
            "context full: {start} cached + {n} new > capacity {}",
            cache.capacity()
        );

        let mut x = Vec::with_capacity(n * d);
        for &tok in tokens {
            ensure!(
                (tok as usize) < cfg.vocab_size,
                "token id {tok} out of range"
            );
            x.extend_from_slice(self.embed.row(tok as usize));
        }

        let mut buf = Buffers::new(cfg, n);
        for (l, layer) in self.layers.iter().enumerate() {
            // --- self-attention block (pre-norm, residual) ---
            rms_norm(&x, &layer.attn_norm, cfg.rms_norm_eps, &mut buf.normed);
            layer.wq.matmul(&buf.normed, &mut buf.q);
            layer.wk.matmul(&buf.normed, &mut buf.k);
            layer.wv.matmul(&buf.normed, &mut buf.v);
            let q_dim = buf.q.len() / n;
            let kv_dim = buf.k.len() / n;
            for t in 0..n {
                self.rope
                    .apply(&mut buf.q[t * q_dim..(t + 1) * q_dim], start + t);
                self.rope
                    .apply(&mut buf.k[t * kv_dim..(t + 1) * kv_dim], start + t);
            }
            cache.write(l, start, &buf.k, &buf.v);
            self.attention(&buf.q, cache.layer(l), start, &mut buf.attn);
            layer.wo.matmul(&buf.attn, &mut buf.proj);
            add_assign(&mut x, &buf.proj);

            // --- SwiGLU MLP block: down(silu(gate(x)) * up(x)) ---
            rms_norm(&x, &layer.mlp_norm, cfg.rms_norm_eps, &mut buf.normed);
            layer.w_gate.matmul(&buf.normed, &mut buf.gate);
            layer.w_up.matmul(&buf.normed, &mut buf.up);
            swiglu(&mut buf.gate, &buf.up);
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

        // One task per (token, head) pair; `i = t * heads + h` matches the
        // [n, heads, head_dim] layout of both `q` and `out`.
        out.par_chunks_mut(hd).enumerate().for_each(|(i, o)| {
            let (t, h) = (i / heads, i % heads);
            let visible = start + t + 1;
            let qh = &q[i * hd..(i + 1) * hd];
            let keys = cache.keys(h / group, visible);
            let values = cache.values(h / group, visible);

            // Scaled dot-product scores; the 1/sqrt(d) keeps their variance
            // ~1 so softmax does not saturate as head_dim grows.
            let mut scores: Vec<f32> = keys.chunks_exact(hd).map(|k| dot(qh, k) * scale).collect();
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
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
}

impl Buffers {
    fn new(cfg: &Config, n: usize) -> Self {
        let q_dim = cfg.num_attention_heads * cfg.head_dim();
        let kv_dim = cfg.num_kv_heads() * cfg.head_dim();
        let zeros = |len: usize| vec![0.0; n * len];
        Self {
            normed: zeros(cfg.hidden_size),
            q: zeros(q_dim),
            k: zeros(kv_dim),
            v: zeros(kv_dim),
            attn: zeros(q_dim),
            proj: zeros(cfg.hidden_size),
            gate: zeros(cfg.intermediate_size),
            up: zeros(cfg.intermediate_size),
        }
    }
}
