# Kiln performance log (milestones 4–5)

All numbers come from this machine: AMD Ryzen 7 7800X3D (8 cores / 16 threads,
96 MB L3), dual-channel DDR5, Windows 11, `cargo build --release`. The model is
SmolLM2-135M-Instruct (134.5M params).

How it was measured:

- `cargo bench --bench e2e`: `prefill_128` runs a 128-token prompt into an empty
  KV cache. `decode_at_128` runs one token on top of a 128-token cache, which is
  the steady-state cost per generated token. Thread count is set with
  `RAYON_NUM_THREADS`, and the format with `KILN_QUANT`.
- `cargo bench --bench matmul`: one matmul on the model's real weight shapes,
  reported in multiply-accumulates per second.
- No sampling profiler was available without admin rights (samply and the ETW
  tools need it). Instead I temporarily instrumented the forward pass with
  per-stage timers (QKV, attention, o_proj, MLP, lm_head), measured, and then
  removed the timers. The stage breakdowns below come from that.
- Every step was re-checked against the logits fixture (tolerance 1e-3). The
  max abs diff stayed at 1.9e-5 to 3.2e-5 throughout.

**Noise warning.** This is a desktop that was in use while I measured, and at
times another job held the CPU near 100%. Within one session, runs agree to
about ±10%. Between sessions they can differ by 25%. For that reason, the
before/after table at the end was measured back to back in one quiet session.
The baseline was rebuilt from commit `c8fbca9` in a separate git worktree for it.
Numbers inside individual steps below were taken during development and only
compare well with their neighbours.

## Starting point: profile of the M3 baseline

At 8 threads, from the stage timers:

| stage | prefill 128 | decode |
|---|---:|---:|
| QKV matmuls | 16% | 19% |
| attention | 4% | 6% |
| o_proj | 8% | 9% |
| MLP matmuls | 65% | 53% |
| lm_head | 1% | 12% |

Matmuls take more than 90% of the time. Decode reads all 538 MB of f32 weights
for every token. At 57–73 tok/s that is 31–39 GB/s, which is close to what this
memory system delivers, so decode was already memory-bandwidth bound. The small
576×576 matmuls ran at about 17 GB/s compared with 47 GB/s for lm_head, which
pointed to per-call threading overhead.

## Step 1: register tiling in portable Rust (negative result, reverted)

The idea was to compute a tile of 4 weight rows × 3 tokens in one pass over
`k`. Each loaded value is then reused 3–4 times: 7 loads per 12 FMAs instead of
2 loads per FMA. I paired this with cache blocking: 64-token activation blocks
stay in L2 while 4-row weight tiles stay in L1.

Matmul at n=128, 1 thread, 576×576:

| kernel | Gelem/s |
|---|---:|
| baseline (one 8-lane `dot` per output) | 11.8 |
| 4×3 tile, portable Rust | 2.7 |
| 2×2 tile, portable Rust | 7.5 |
| 2×2 tile, `-C target-cpu=native` | 12.0 |
| 4×3 tile, `-C target-cpu=native` | 3.6 |

LLVM did not keep the 4×3×8 accumulator block in vector registers. It
scalarized the block and spilled it, even when AVX2 was enabled. The tiling
idea was sound, but the compiler could not express it, so the scalar fallback
keeps independent dot products. The tiling and cache-blocking structure
(`Matrix::tiled_block`) stayed, and step 2 fills it in with intrinsics.

## Step 2: AVX2 + FMA with `std::arch`

- `dot` uses 4 accumulators × 8 lanes with FMA.
- `tile` is 4 rows × 3 tokens: 12 accumulators + 3 activation vectors + 1
  weight vector, which is exactly the 16 ymm registers, so nothing spills.
- The CPU is checked once at runtime (`is_x86_feature_detected!`). The scalar
  path remains as the fallback (`KILN_NO_SIMD=1` forces it), and unit tests
  compare it against the SIMD kernels, including the full tiled matmul.

| matmul n=128 (Gelem/s) | baseline 1T | AVX2 1T | baseline 16T | AVX2 16T |
|---|---:|---:|---:|---:|
| 576×576 | 11.8 | 49.1 | 63 | 170 |
| 576×1536 (ffn_down) | 11.5 | 50.5 | 75 | 178 |

End to end in the same session: prefill went from 81 to 343 tok/s on 1 thread
(4.2×) and from 531 to 921 on 16 threads. Decode only went from 44 to 62 tok/s
on 1 thread, and 16 threads still lost to 4 threads (59 vs 68 tok/s). Faster
arithmetic does not help a loop that is waiting on memory.

## Step 3: threading (fixing the decode regression past 4 threads)

**3a. One pool entry per forward pass.** Every `par_iter` called from the main
thread is injected into rayon's pool, and the caller then sleeps on an OS event
until it finishes. A decode step has about 200 of these. Wrapping the whole
forward pass in `rayon::scope` moves it onto a pool thread, where nested
parallel loops become cheap work-stealing joins.

| decode tok/s | 1T | 4T | 8T | 16T |
|---|---:|---:|---:|---:|
| before | 62 | 68 | 55 | 59 |
| after | 68 | 76 | 74 | 59 |

**3b. Sizing tasks by work.** Each parallel task now gets at least 64K MACs
(`MIN_TASK_MACS`) and there are about 4 tasks per thread. Before this, row
blocks had a fixed 16 rows, so a 576×576 mat-vec became 36 tasks of about 1 µs
each. Attention batches its (token, head) items the same way (`with_min_len`).
At short context, all 9 heads of a decode step take only a few microseconds.

| decode tok/s | 1T | 4T | 8T | 16T |
|---|---:|---:|---:|---:|
| after 3b | 75 | 71 | 73 | 72 |

After this change, decode at 16 threads matches 1 thread instead of losing 20%.
Throughput stays flat at about 73 tok/s, or 39 GB/s. That flat line is the
memory-bandwidth limit for f32 weights, not a threading problem.

**3c. Fused projections.** Q, K and V are stacked into one `[768, 576]` matrix,
and gate and up into one `[3072, 576]` matrix, at load time. That gives one
parallel region and one pass over the input instead of three, or two. The
machine was loaded when I measured this step, so I have no clean isolated
number for it. It is included in the final table.

## Step 4: quantization (milestone 5)

Formats, quantized from the bf16 checkpoint at load time:

- **q8:** int8 with one f32 scale per output row (8.06 bits/weight).
- **q4:** 4-bit with an f16 scale per group of 32 weights (4.5 bits/weight).
  The layout is llama.cpp's Q4_0: the scale is signed, the largest-magnitude
  weight maps to −8, and the high and low nibbles of a byte cover `x[0..16]`
  and `x[16..32]`.
- **Group size:** I measured 32 and 64. Perplexity was 28.58 for g32 and 29.90
  for g64. g64 saves 0.25 bits/weight, which is not worth 1.3 perplexity points
  on a model this small, so the default is 32.
- **Embedding under q4:** the tied embedding / LM head is kept at q8. With a q4
  embedding, perplexity is 31.8 instead of 28.6.

Kernels:

- **Decode, q8:** `dot_q8` sign-extends 8 int8 weights, converts them to f32 and
  does an FMA. Only the weights shrink in memory.
- **Prefill, both formats:** each 4-row tile is dequantized once into an
  L1-sized f32 scratch buffer and reused across the 64-token block. Prefill
  therefore runs the f32 AVX2 tile kernel and runs at f32 speed.

**Decode, q4: the kernel iterations.** The first q4 kernel converted nibbles to
f32 and did an FMA against f32 activations. It made q4 decode *slower* than q8
(61 vs 118 tok/s on 1 thread, noisy session). The model fits in L3, so the
kernel was compute bound, and each 32-weight block needed 4 widenings, 4
conversions and 4 FMAs.

1. **Integer dot against int8 activations, as in llama.cpp.** The activation
   vector is quantized once per mat-vec to int8 in blocks of 32. Each block is
   then `maddubs(nibbles, x_i8)` → `madd(·, 1)` → one f32 conversion → one FMA
   with the combined scale.
2. **Breaking the dependency chain.** The `−8` offset was first applied as a
   scalar per block. That serial scalar chain cost about 25% of the kernel
   (isolated mat-vec timing: 70 µs with the scalar offset, 55 µs with the
   offset code removed). Folding it in as a second `maddubs` of 8s, subtracted
   in i16, fixed that. The accumulators were also turned into named registers
   instead of an array indexed by `b & 1`, which spilled to the stack.
3. **Replacing `f32::round`.** It is a libm call on baseline x86-64, and
   activation quantization now rounds by hand.

Cost of int8 activations (test `quantized_models_stay_close_to_reference`):
on the fixture prompts, decode-path logits move by up to 1.8 relative to the
prefill path. Top-1 agreement with fp32 transformers is 47/60 positions through
the decode path and 45/60 through the prefill path, so int8 activations cost no
measurable accuracy on top of 4-bit weights.

## Final table (one quiet session, back to back)

Measured with `cargo bench --bench e2e`, in tok/s:

| build | threads | prefill 128 | decode @128 |
|---|---:|---:|---:|
| M3 baseline (`c8fbca9`), f32 | 1 | 102 | 56 |
| | 4 | 316 | 74 |
| | 8 | 496 | 68 |
| | 16 | 575 | 59 |
| final, f32, scalar fallback (`KILN_NO_SIMD=1`) | 1 | 93 | 66 |
| | 8 | 400 | 80 |
| **final, f32, AVX2** | 1 | 306 | 77 |
| | 4 | 634 | 90 |
| | 8 | 729 | 84 |
| | 16 | 767 | 75 |
| **final, q8** | 1 | 295 | 143 |
| | 4 | 629 | 305 |
| | 8 | 715 | **346** |
| | 16 | 757 | 267 |
| **final, q4** | 1 | 288 | 119 |
| | 4 | 623 | 239 |
| | 8 | 716 | 291 |
| | 16 | 743 | 243 |

Overall gains against the baseline:

- **Prefill:** 3.0× on 1 thread and 1.3× on 16 threads.
- **Decode, f32:** 1.2–1.4×. Bandwidth caps it.
- **Decode, q8:** 5.1× at 8 threads.
- **Decode, q4:** 4.3× at 8 threads.

Two patterns hold across every format:

- 16 threads is slower than 8 for decode. The extra 8 threads are SMT siblings
  sharing the same cores and memory path, and only add scheduling overhead.
- Prefill barely depends on the format, because quantized weights are
  dequantized into the f32 tile kernel.

## Reading the decode numbers

The useful question is how many bytes each decode step reads, and how fast
those bytes arrive:

| format | weight bytes | best decode | effective weight bandwidth |
|---|---:|---:|---:|
| f32 | 538 MB | 90 tok/s | 48 GB/s |
| q8 | 135 MB | 346 tok/s | 47 GB/s |
| q4 (+q8 embedding) | 88 MB | 291 tok/s | 26 GB/s |

- **f32 → q8:** 4× fewer bytes gave a 3.8× speedup at the same ~47 GB/s.
  Decode is bandwidth bound, and quantization buys speed in direct proportion
  to the bytes it removes.
- **q8 → q4:** another 1.5× fewer bytes, but no speedup. The 88 MB q4 model
  fits in the 7800X3D's 96 MB L3, so the bandwidth limit is gone and the limits
  become the kernel's arithmetic (unpacking nibbles costs more per weight than
  widening int8) and the fork/join overhead of about 150 parallel regions per
  token.

That is the main thing I learned here: "quantize because decode is memory bound"
is true only until the model fits in cache. On a CPU with a smaller L3, or with
a larger model, q4 would pull ahead of q8.

## llama.cpp reference (same machine, same model)

llama.cpp `b11932`, the prebuilt `win-cpu-x64` zip (SHA-256 verified),
extracted outside the repo. The `f16` GGUF was quantized with
`llama-quantize --pure` so that every tensor, the embedding included, uses the
named format. Command: `llama-bench -p 128 -n 64 -t 1,4,8 -r 3`.

| | threads | llama.cpp pp128 | Kiln prefill 128 | llama.cpp tg64 | Kiln decode |
|---|---:|---:|---:|---:|---:|
| F16 / f32 | 8 | 1427 | 729 | 173 | 84 |
| Q8_0 / q8 | 1 | 495 | 295 | 167 | 143 |
| Q8_0 / q8 | 8 | 1866 | 715 | 350 | 346 |
| Q4_0 / q4 | 1 | 729 | 288 | 264 | 119 |
| Q4_0 / q4 | 8 | 3300 | 716 | 714 | 291 |

Why the gaps look the way they do:

- **Q8 decode is at parity.** Both engines hit the same bandwidth wall.
- **F16 decode runs at 2× Kiln's f32.** llama.cpp keeps weights at 2 bytes and
  reads half as much memory.
- **Q4 decode runs at 2.5× Kiln's.** llama.cpp has more tightly scheduled
  kernels and a single persistent thread pool with spin barriers, where Kiln
  pays a rayon fork/join per matmul.
- **Prefill runs at 2.6–4.6× Kiln's.** llama.cpp runs integer GEMMs against
  quantized activations, while Kiln dequantizes into an f32 GEMM. Kiln's f32
  tile is also a simple 4×3 micro-kernel, not a packed GEMM.

## Next steps, by expected payoff

1. **Persistent workers with spin barriers.** Replace the per-matmul rayon
   fork/join with one worker set per forward pass. This targets the
   q4/q8-in-cache regime.
2. **Integer GEMM for quantized prefill.** Quantize activations per token block
   and use `maddubs`/VNNI tiles.
3. **Run f16/bf16 weights directly** with `vcvtph2ps` in the kernel. That halves
   f32 decode traffic for anyone who doesn't want to quantize.
4. **Packed-panel GEMM for f32 prefill:** larger tiles, and B-panels laid out
   in kernel order.
