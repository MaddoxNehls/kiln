//! Kiln: a from-scratch LLM inference engine for Llama-architecture models.
//!
//! Module map, bottom-up:
//!
//! - [`tensor`]: weight matrices (f32/int8/int4) and the SIMD matmul kernels
//! - [`ops`]: RMSNorm, softmax, SwiGLU, RoPE
//! - [`weights`]: memory-mapped safetensors -> f32
//! - [`model`]: the transformer forward pass
//! - [`kv_cache`]: per-layer key/value storage for incremental decoding
//! - [`tokenizer`]: byte-level BPE from `tokenizer.json`
//! - [`sampler`], [`chat`], [`generate`]: everything between logits and text
//! - [`perplexity`]: language-modeling quality, used to price quantization

pub mod chat;
pub mod config;
pub mod generate;
pub mod kv_cache;
pub mod model;
pub mod ops;
pub mod perplexity;
pub mod sampler;
pub mod tensor;
pub mod tokenizer;
pub mod weights;
