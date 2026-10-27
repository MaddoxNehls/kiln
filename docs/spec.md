# Kiln: Spec (v1)

> A from-scratch LLM inference engine in Rust. Load real open-weight models and generate text on a CPU, with no ML frameworks.

## Goal
Show deep understanding of how LLMs actually run: tokenization, transformer math, memory layout, quantization, and performance engineering. Correctness is proven against a reference implementation.

## Scope
- **Models:** Llama-architecture small models (e.g. SmolLM2-135M/360M, Qwen2.5-0.5B), loaded from **safetensors** (and GGUF in a stretch goal)
- **Tokenizer:** byte-level BPE implemented from scratch, loading `tokenizer.json`; validated against HF `tokenizers` on a large text sample (exact token match)
- **Forward pass:** embeddings, RMSNorm, RoPE, grouped-query attention, SwiGLU MLP; **KV cache**
- **Sampling:** greedy, temperature, top-k, top-p, repetition penalty; streaming tokens to stdout
- **Quantization:** int8 per-channel and 4-bit group quantization, with a measured perplexity cost
- **Performance:** cache-friendly matmul, SIMD (AVX2/NEON via `std::arch` or portable SIMD), multithreading with rayon, memory-mapped weights
- **Server mode:** an OpenAI-compatible `/v1/chat/completions` endpoint with streaming (SSE)

## Correctness
- A logits-match test against PyTorch/HF transformers on fixed prompts (max abs diff within tolerance)
- Perplexity on a WikiText sample: fp32 vs int8 vs int4

## Benchmarks (README headline)
- tokens/sec (prefill and decode) by model size, thread count, and quantization; comparison with llama.cpp on the same machine as a reference point (not trying to beat it, just to show the gap and explain it)
- A flame-graph-driven optimization log: "naive → blocked matmul → SIMD → threads: 0.8 → 3.1 → 11 → 38 tok/s"

## Stack
Rust stable, `safetensors`, `memmap2`, `rayon`, `axum` for the server, `criterion` for benchmarks. A Python script (dev only) generates the reference logits.

## Milestones
1. Tokenizer + exact-match tests
2. fp32 forward pass + logits match
3. KV cache + sampling + CLI chat
4. Performance work + benchmark log
5. Quantization + perplexity table
6. OpenAI-compatible server, README, write-up ("What I learned building an LLM inference engine")
