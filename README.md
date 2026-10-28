# Kiln

A from-scratch LLM inference engine in Rust. Kiln loads a real open-weight model
(SmolLM2-135M-Instruct, Llama architecture) from safetensors and generates text
on the CPU, with no ML or tensor framework: the tokenizer, the transformer math,
the KV cache and the sampler are all implemented here and checked against
Hugging Face `tokenizers`/`transformers`.

> Status: milestones 1-5 of [`docs/spec.md`](docs/spec.md) are done: the
> tokenizer, the fp32 forward pass, the KV cache with sampling and a streaming
> CLI, AVX2 kernels with a performance log, and int8/int4 quantization with a
> perplexity table. The HTTP server (milestone 6) is next.

```console
$ kiln generate --quant q8 --threads 8 --prompt "Explain in two sentences why the sky is blue." --max-tokens 100 --seed 42
loaded models/smollm2-135m-instruct (30 layers, d=576, 9q/3kv heads) as q8: 135.4 MB of weights, 0.28s, 8 threads
The sky appears blue because the Earth's atmosphere scatters sunlight in all directions,
including blue light. This scattering occurs because the molecules of gases like oxygen and
nitrogen absorb shorter wavelengths of light (blue) more efficiently than longer wavelengths
(red). As a result, blue light is scattered out to one side, giving the sky its characteristic blue color.
[prefill 40 tok in 60 ms = 664.2 tok/s | decode 69 tok in 204 ms = 339.0 tok/s | stop token]
```

## Quick start

```bash
scripts/download-model.sh                 # ~270 MB into models/smollm2-135m-instruct/
cargo build --release
./target/release/kiln generate --prompt "Write a haiku about Rust." --max-tokens 64
./target/release/kiln chat --quant q8     # multi-turn; /reset clears history
scripts/download-wikitext.sh              # WikiText-2 test split into data/
./target/release/kiln perplexity --quant q4
cargo test                                # needs the downloaded model
cargo bench --bench e2e                   # KILN_QUANT=q8|q4, RAYON_NUM_THREADS=N
```

Every command takes `--quant f32|q8|q4` (weights are quantized at load time
from the bf16 checkpoint; `q4g64` selects 64-weight groups) and `--threads`.
Sampling flags (shared by `generate` and `chat`): `--temperature` (0 = greedy),
`--top-k`, `--top-p`, `--repeat-penalty`, `--repeat-last-n`, `--seed`. Also
`--system`, `--ctx` (KV cache size), and `--raw` on `generate` to skip the chat
template. On this 8-core CPU, `--threads 8` beats the default of 16 logical
threads for decode.

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
| `tensor` | `Matrix` (f32 / q8 / q4 storage) and `matmul`: work partitioning, cache-blocked prefill tiles, decode mat-vecs |
| `tensor::kernels` | scalar and AVX2+FMA kernels (`dot`, 4×3 `tile`, `dot_q8`, `dot_q4_q8`), chosen at runtime |
| `tensor::quant` | int8 per-row and 4-bit group formats, activation quantization |
| `ops` | RMSNorm, softmax, SwiGLU, RoPE tables |
| `weights` | memory-mapped safetensors, bf16/f16 → f32 widening, shape checks |
| `model` | the Llama forward pass, batched over a whole prompt |
| `kv_cache` | per-layer K/V storage, head-major |
| `tokenizer` | byte-level BPE, GPT-2 pre-tokenizer, streaming decoder |
| `sampler` | greedy / temperature / top-k / top-p / repetition penalty, seedable SplitMix64 |
| `chat` | ChatML template, checked against `apply_chat_template` |
| `generate` | `Session`: prefill, decode loop, stop conditions, timing |
| `perplexity` | fixed-protocol WikiText-2 perplexity, matched against transformers |

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
| `q, k, v = h·Wqᵀ, h·Wkᵀ, h·Wvᵀ` | one fused `Matrix::matmul` over the stacked `[Wq; Wk; Wv]` (weights kept in PyTorch `[out, in]` layout, so each output is a dot of two contiguous rows) |
| rotate `q, k` by `pos · base^(-2i/d)` | `ops::Rope::apply`: HF "rotate_half" pairing `(i, i + d/2)`, θ = `rope_theta` = 100000 from config, cos/sin precomputed |
| append `k, v` | `KvCache::write`, scattering into a `[kv_heads, capacity, head_dim]` layout |
| `softmax(q·kᵀ / sqrt(d)) · v`, causal, GQA | `Model::attention`: query head `h` reads kv head `h / 3`; causality is implicit because position `p` only reads cache rows `0..=p` |
| `x += o·Woᵀ` | `matmul` + `ops::add_assign` |
| `x += W_down(silu(W_gate·h) ⊙ W_up·h)` | fused `[W_gate; W_up]` matmul, `ops::swiglu`, `W_down` matmul |
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

Quantized models are checked too (`quantized_models_stay_close_to_reference`).
Against fp32 transformers, top-1 agreement over the 60 fixture positions is
56/60 for q8 and 45/60 for q4.

### Perplexity (WikiText-2 test, quality cost of quantization)

The text is tokenized once (303,415 tokens) and cut into non-overlapping
512-token windows. The first 40 windows are scored, 20,440 predictions in
total. `scripts/reference/perplexity_ref.py` runs the same protocol in
transformers.

| weights | bits/weight | weight MB | perplexity | vs fp32 |
|---|---:|---:|---:|---:|
| transformers fp32 (reference) | 32 | 538 | 22.4815 | |
| Kiln f32 | 32 | 538 | **22.4815** | identical |
| Kiln q8 (per-row int8) | 8.06 | 135 | 22.5308 | +0.2% |
| Kiln q4, groups of 32 (+ q8 embedding) | ~5.3 avg | 88 | 28.5758 | +27% |
| Kiln q4, groups of 64 (+ q8 embedding) | ~5.0 avg | 81 | 29.9041 | +33% |
| Kiln q4, groups of 32, q4 embedding too | 4.5 | 76 | 31.7716 | +41% |

Notes on the table:

- **Kiln f32 vs transformers:** the perplexities agree to four decimals.
- **q8:** essentially free.
- **q4:** costly on a 135M-parameter model. Small models have little
  redundancy to absorb 16-level weights, and the tied embedding / LM head is
  the most sensitive tensor. That is why q4 keeps the embedding at q8, the
  same choice llama.cpp makes for its output tensor.
- **Group size:** 32 is the default. 64 saves 0.25 bits/weight but costs 1.3
  perplexity points.

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

## Performance

AMD Ryzen 7 7800X3D (8 cores / 16 threads, 96 MB L3), Windows 11, release build,
`cargo bench --bench e2e`. Prefill is a 128-token prompt; decode is one token on
a 128-token cache. All rows are in tok/s and were measured back to back in one
session.

| weights | threads | prefill | decode | weight MB | RSS during generation |
|---|---:|---:|---:|---:|---:|
| M3 baseline, f32 | 1 / 8 | 102 / 496 | 56 / 68 | 538 | |
| f32 | 1 / 8 | 306 / 729 | 77 / 84 | 538 | 534 MB |
| q8 | 1 / 8 | 295 / 715 | 143 / **346** | 135 | 150 MB |
| q4 | 1 / 8 | 288 / 716 | 119 / 291 | 88 | 107 MB |
| llama.cpp b11932, F16 | 8 | 1427 | 173 | 257 | |
| llama.cpp b11932, Q8_0 | 8 | 1866 | 350 | 136 | |
| llama.cpp b11932, Q4_0 (pure) | 8 | 3300 | 714 | 72 | |

The full table, every optimization step with its numbers, the profile, and the
dead ends are in [`docs/perf-log.md`](docs/perf-log.md). The short version:

1. **Prefill (compute bound): 3× faster on 1 thread.** The changes were AVX2+FMA
   kernels and a 4×3 register tile (12 accumulators plus 4 operand vectors fill
   all 16 ymm registers) inside cache blocking: 64-token activation blocks stay
   in L2 while each 4-row weight tile stays in L1. I first wrote the tile in
   portable Rust. LLVM spilled the accumulator block and that version was 4×
   *slower*, so the tile is written with `std::arch` intrinsics, and a runtime
   CPU check falls back to tested scalar code.
2. **Threading.** Decode used to get slower past 4 threads. Two changes fixed
   that: running each forward pass on a rayon pool thread (so ~200 parallel
   loops per token are cheap joins, not cross-thread injections), and sizing
   tasks by work (≥ 64K multiply-adds). Projections that share an input
   (QKV, gate+up) are fused at load time.
3. **Decode (memory bound): quantization.** A decode step streams every weight
   once. f32 decode tops out at ~48 GB/s of weights (90 tok/s at best).
   q8 reads 4× fewer bytes and runs **4×** faster at the same ~47 GB/s, which
   confirms the bandwidth model. q4 reads 1.5× fewer bytes again but is *not*
   faster than q8, because at 88 MB the whole q4 model fits in this CPU's
   96 MB L3. Decode stops being memory bound, and nibble unpacking plus
   fork/join overhead set the speed. The q4 decode kernel is an integer dot
   product against activations quantized to int8 per 32-block (`vpmaddubsw`,
   as in llama.cpp). Making it faster meant removing a serial scalar chain and
   an accumulator spill, both found by isolating the kernel.

Against llama.cpp on the same machine: Kiln matches it on q8 decode (346 vs 350
tok/s), because both hit the same memory wall. It trails by 2–2.5× on f16/q4
decode and by 2.6–4.6× on prefill. llama.cpp uses f16 weights in place of f32,
a persistent spin-barrier thread pool instead of fork/join, and integer GEMMs
for quantized prefill. `docs/perf-log.md` explains each gap.

## Next steps

1. **Server (milestone 6).** An OpenAI-compatible `/v1/chat/completions` with SSE
   streaming on `axum`, reusing `Session` prefix caching across requests.
2. **Persistent workers with spin barriers** instead of per-matmul fork/join.
   This is the main remaining decode cost once the weights fit in cache.
3. **Integer GEMM for quantized prefill**, and f16/bf16 weights consumed
   directly by the kernels, which halves f32 decode traffic with no accuracy
   loss.

Known limitations:

- **Chat templates:** Kiln supports the ChatML family (SmolLM2, Qwen2) rather
  than arbitrary Jinja.
- **RoPE scaling:** `null` and `linear` are supported. Anything else (llama3,
  yarn) is rejected rather than approximated.
- **SIMD:** x86-64 AVX2 only. Other CPUs, ARM/NEON included, run the scalar
  fallback.
- **Q4 numerics differ by path:** decode quantizes activations to int8, while
  prefill dequantizes weights to f32. The drift between the two paths is
  bounded in the tests and costs no top-1 accuracy on the fixtures.
