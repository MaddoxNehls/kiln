# Kiln

A from-scratch LLM inference engine in Rust. Kiln loads a real open-weight model
(SmolLM2-135M-Instruct, Llama architecture) from safetensors and generates text
on the CPU, with no ML or tensor framework: the tokenizer, the transformer math,
the KV cache and the sampler are all implemented here and checked against
Hugging Face `tokenizers`/`transformers`.

> Status: milestones 1-3 of [`docs/spec.md`](docs/spec.md) are done (tokenizer,
> fp32 forward pass, KV cache + sampling + streaming CLI). Performance work,
> quantization and the HTTP server are next.

```console
$ kiln generate --prompt "Explain in two sentences why the sky is blue." --max-tokens 100 --seed 42
loaded models/smollm2-135m-instruct (30 layers, d=576, 9q/3kv heads) in 0.13s on 16 threads
The sky appears blue because the Earth's atmosphere scatters sunlight in all directions,
including blue light. As the sun moves across the sky during sunrise or sunset, different
wavelengths of light are scattered at varying rates by the atmosphere. ...
[prefill 40 tok in 101 ms = 397.1 tok/s | decode 99 tok in 1751 ms = 56.5 tok/s | max tokens]
```

## Quick start

```bash
scripts/download-model.sh                 # ~270 MB into models/smollm2-135m-instruct/
cargo build --release
./target/release/kiln generate --prompt "Write a haiku about Rust." --max-tokens 64
./target/release/kiln chat                # multi-turn; /reset clears history
cargo test                                # needs the downloaded model
```

Sampling flags (shared by `generate` and `chat`): `--temperature` (0 = greedy),
`--top-k`, `--top-p`, `--repeat-penalty`, `--repeat-last-n`, `--seed`. Also
`--system`, `--ctx` (KV cache size), `--threads`, and `--raw` on `generate` to
skip the chat template.

## Architecture

```text
                      tokenizer.json        tokenizer_config.json      config.json + model.safetensors
                            │                        │                              │
                            ▼                        ▼                              ▼
 user text ──► chat::ChatTemplate ──► tokenizer::Tokenizer ──► token ids ──► model::Model::forward ◄── weights (mmap → f32)
   ▲            (ChatML render)       (added tokens, pre-          │            │        ▲
   │                                   tokenize, BPE)              │            │        │ K/V append + read
   │                                                               │            ▼        │
   │                                                  generate::Session ──► kv_cache::KvCache
   │                                                   (prefix reuse,           │
   │                                                    prefill/decode loop)    ▼ logits
   │                                                               │     sampler::Sampler
   │                                                               │     (rep. penalty, T, top-k, top-p)
   │                                                               ▼
   └──────────── StreamDecoder (UTF-8 safe) ◄──────────────── next token id
```

| Module | Role |
|---|---|
| `tensor` | `Matrix` and `matmul`, the one compute hot spot and the boundary where SIMD and quantized weights will plug in |
| `ops` | RMSNorm, softmax, SwiGLU, RoPE tables |
| `weights` | memory-mapped safetensors, bf16/f16 → f32 widening, shape checks |
| `model` | the Llama forward pass, batched over a whole prompt |
| `kv_cache` | per-layer K/V storage, head-major |
| `tokenizer` | byte-level BPE, GPT-2 pre-tokenizer, streaming decoder |
| `sampler` | greedy / temperature / top-k / top-p / repetition penalty, seedable SplitMix64 |
| `chat` | ChatML template, checked against `apply_chat_template` |
| `generate` | `Session`: prefill, decode loop, stop conditions, timing |

**One crate, library + binary.** Every module is a stage of a single pipeline with
one consumer, so a workspace would add manifests without adding any boundary that
pays off yet. `src/lib.rs` is the engine and `src/main.rs` is a thin CLI, so the
planned server (milestone 6) becomes a second binary or a feature on the same library.

## How the forward pass maps to code

`Model::forward(tokens, cache, logits)` in [`src/model.rs`](src/model.rs) runs `n`
tokens as one `[n, hidden]` batch (the whole prompt at prefill, one token per decode step):

| Math | Code |
|---|---|
| `x = E[tokens]` | rows of `embed` copied into `x` |
| `h = RMSNorm(x) = x / sqrt(mean(x²) + ε) · g` | `ops::rms_norm` |
| `q, k, v = h·Wqᵀ, h·Wkᵀ, h·Wvᵀ` | `Matrix::matmul` (weights kept in PyTorch `[out, in]` layout, so each output is a dot of two contiguous rows) |
| rotate `q, k` by `pos · base^(-2i/d)` | `ops::Rope::apply`: HF "rotate_half" pairing `(i, i + d/2)`, θ = `rope_theta` = 100000 from config, cos/sin precomputed |
| append `k, v` | `KvCache::write`, scattering into a `[kv_heads, capacity, head_dim]` layout |
| `softmax(q·kᵀ / sqrt(d)) · v`, causal, GQA | `Model::attention`: query head `h` reads kv head `h / 3`; causality is implicit because position `p` only reads cache rows `0..=p` |
| `x += o·Woᵀ` | `matmul` + `ops::add_assign` |
| `x += W_down(silu(W_gate·h) ⊙ W_up·h)` | `matmul` ×3 + `ops::swiglu` |
| final RMSNorm, `logits = h·Eᵀ` | tied embeddings (`tie_word_embeddings: true`), so `embed` doubles as the LM head; at generation time only the last row is computed |

Layout decisions worth calling out:

- **Weights stay `[out, in]`.** `y = W·x` then streams each weight row and `x`
  sequentially, with no transposes. Parallelism is over weight rows, so each row is
  loaded once and reused for every token in a prefill batch.
- **The KV cache is head-major.** Attention for one head scans every cached position
  of that head, so that scan is one contiguous stream. Appending a token becomes a
  scatter of 3 small rows per layer, which is cheap by comparison. Memory is
  preallocated, so decode never reallocates.
- **bf16 → f32 at load.** bf16 is the top 16 bits of an f32, so widening is exact.
  The file is mmapped and converted in parallel across layers (~0.13 s for 135M params).

## Correctness

All checked by `cargo test` against fixtures made by the dev-only scripts in
[`scripts/reference/`](scripts/reference) (a uv project: `tokenizers`, CPU `torch`, `transformers`).

| Test | Reference | Result |
|---|---|---|
| Tokenizer encode, 35 cases (code, CJK, Hindi, Arabic, emoji ZWJ sequences, whitespace runs, digits in 4 scripts, special tokens, unmappable bytes) | HF `tokenizers` | exact id match |
| Logits, 3 prompts (60 positions, top-10 each) | `transformers` fp32, eager attention | **max abs diff 3.05e-5**, identical top-1 at every position |
| Full 49,152-wide final logits row | same | max abs diff 2.81e-5, mean 5.8e-6 |
| Chunked prefill + token-by-token decode vs one full pass | Kiln itself | bit-identical (diff 0.0) |
| Greedy generation, 24 tokens × 3 prompts | `model.generate(do_sample=False)` | identical token ids |
| Chat template rendering | `apply_chat_template` | byte-identical |

The logits test tolerance is 1e-3: about 30× the measured error, which leaves room
for the summation-order changes SIMD and threading will bring, while a real bug
(wrong RoPE pairing, a missing norm) moves logits by more than 0.1.

Regenerating fixtures:

```bash
uv run --project scripts/reference python scripts/reference/gen_tokenizer_fixture.py
uv run --project scripts/reference python scripts/reference/gen_logits_fixture.py
python scripts/gen_unicode_tables.py   # \p{L} table for the pre-tokenizer
```

### Tokenizer details that the exact-match test caught or pinned down

- SmolLM2 chains a `Digits(individual_digits)` pre-tokenizer before GPT-2's regex.
  It splits on every Unicode number (`٣`, `１`, `²`, `Ⅻ`), not only ASCII digits.
- `\p{L}` is a generated general-category table, not `char::is_alphabetic`.
  Alphabetic also includes combining marks, which would mis-split Hindi and accented text.
- The GPT-2 regex's `\s+(?!\S)` look-ahead is implemented in a hand-written scanner,
  because Rust's `regex` crate has no look-around.
- 21 byte values have no vocab entry. HF silently drops them (no `unk_token`, no
  `byte_fallback`), and Kiln matches that behavior.

## Performance (milestone 3 baseline)

AMD Ryzen 7 7800X3D (8 cores / 16 threads), Windows 11, `cargo build --release`,
fp32 weights, greedy decoding:

| Threads | Prefill, 38-token prompt | Prefill, 420-token prompt | Decode |
|---:|---:|---:|---:|
| 1  | 102 tok/s | 95 tok/s  | 57.5 tok/s |
| 4  | 332 tok/s | –         | 76.4 tok/s |
| 8  | 451 tok/s | 461 tok/s | 70.0 tok/s |
| 16 | 436 tok/s | 557 tok/s | 63.3 tok/s |

**Decode is memory-bandwidth bound already.** Every decoded token reads all 134.5M
fp32 parameters, about 538 MB. At 57 tok/s on one thread that is ~31 GB/s; at
4 threads it is ~41 GB/s, close to what dual-channel DDR5 sustains in practice. More
threads don't help, and past 4 threads the rayon fan-out over small 576-row
matrices costs more than it gains. The tied 49k × 576 LM head alone is 21% of that
traffic. Shrinking the bytes per weight (int8/int4) is therefore the main decode
lever, and it's why quantization is a milestone and not an afterthought.

Prefill reuses each weight row across the whole batch, so it scales with cores.
It is still far from compute peak, because the inner kernel is a scalar 8-accumulator
dot product that LLVM auto-vectorizes.

## Next steps

1. **Performance (milestone 4).** Explicit AVX2/FMA (and NEON) dot kernels behind
   `Matrix::matmul`; a register-blocked GEMM for prefill (tile tokens × rows so
   activations stay in registers); fused QKV and gate/up projections; skip rayon
   below a work threshold; a flame-graph-driven optimization log and llama.cpp
   comparison with `criterion` benchmarks.
2. **Quantization (milestone 5).** int8 per-channel and 4-bit group-quantized
   `Matrix` variants (dequantize in the dot kernel), plus a perplexity table on
   WikiText for fp32 / int8 / int4.
3. **Server (milestone 6).** An OpenAI-compatible `/v1/chat/completions` with SSE
   streaming on `axum`, reusing `Session` prefix caching across requests.

Known limitations: the chat template supports the ChatML family (SmolLM2, Qwen2)
rather than arbitrary Jinja; RoPE scaling supports `null` and `linear`, and anything
else (llama3, yarn) is rejected rather than approximated; attention and the sampler
allocate small scratch buffers per call.
