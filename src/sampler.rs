//! Turning logits into the next token.
//!
//! Pipeline, applied in this order (same as HF `generate` and llama.cpp):
//!
//! ```text
//! logits ─► repetition penalty ─► (temperature == 0 ? argmax)
//!        ─► top-k ─► softmax(logits / T) ─► top-p ─► draw
//! ```

use crate::tokenizer::TokenId;

#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// 0 means greedy (argmax). Higher flattens the distribution.
    pub temperature: f32,
    /// Keep only the `k` most likely tokens. 0 disables.
    pub top_k: usize,
    /// Keep the smallest set of tokens whose probability mass reaches `p`
    /// ("nucleus"). 1.0 disables.
    pub top_p: f32,
    /// > 1 discourages tokens seen in the last `repeat_last_n` tokens. 1.0 disables.
    pub repetition_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            repeat_last_n: 64,
            seed: 0,
        }
    }
}

pub struct Sampler {
    params: SamplingParams,
    rng: SplitMix64,
    /// Reused `(token, logit)` buffer: avoids a 49k-element allocation per token.
    candidates: Vec<(TokenId, f32)>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        Self {
            rng: SplitMix64(params.seed),
            params,
            candidates: Vec::new(),
        }
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Choose the next token from `logits`; `history` is the context so far,
    /// used only by the repetition penalty.
    pub fn sample(&mut self, logits: &[f32], history: &[TokenId]) -> TokenId {
        let p = &self.params;
        self.candidates.clear();
        self.candidates
            .extend(logits.iter().enumerate().map(|(i, &l)| (i as TokenId, l)));

        // Repetition penalty (CTRL, Keskar et al. 2019): shrink the logit of
        // every recently seen token toward "less likely". Dividing a negative
        // logit would *raise* it, hence the sign split. Each token is penalized
        // once however often it appeared, hence the dedup.
        if p.repetition_penalty != 1.0 {
            let recent = &history[history.len().saturating_sub(p.repeat_last_n)..];
            let mut seen = recent.to_vec();
            seen.sort_unstable();
            seen.dedup();
            for &id in &seen {
                if let Some((_, l)) = self.candidates.get_mut(id as usize) {
                    *l = if *l > 0.0 {
                        *l / p.repetition_penalty
                    } else {
                        *l * p.repetition_penalty
                    };
                }
            }
        }

        if p.temperature <= 0.0 {
            return argmax(&self.candidates);
        }

        // Top-k by partial selection (O(V)), then sort only the survivors.
        let by_logit_desc = |a: &(TokenId, f32), b: &(TokenId, f32)| b.1.total_cmp(&a.1);
        if p.top_k > 0 && p.top_k < self.candidates.len() {
            self.candidates
                .select_nth_unstable_by(p.top_k - 1, by_logit_desc);
            self.candidates.truncate(p.top_k);
        }
        self.candidates.sort_unstable_by(by_logit_desc);

        // Softmax with temperature, reusing the logit slot for the weight.
        // Sorted descending, so the first entry is the max.
        let max = self.candidates[0].1;
        let mut total = 0.0;
        for (_, l) in self.candidates.iter_mut() {
            *l = ((*l - max) / p.temperature).exp();
            total += *l;
        }

        // Top-p: cut the sorted tail once the kept mass reaches `top_p`.
        if p.top_p < 1.0 {
            let target = p.top_p * total;
            let mut mass = 0.0;
            let keep = self
                .candidates
                .iter()
                .position(|&(_, w)| {
                    mass += w;
                    mass >= target
                })
                .map_or(self.candidates.len(), |i| i + 1);
            self.candidates.truncate(keep);
            // `position` stops at the cut, so `mass` is exactly the kept weight.
            total = mass;
        }

        // Inverse-CDF draw over the unnormalized weights.
        let mut r = self.rng.next_f32() * total;
        for &(id, w) in &self.candidates {
            r -= w;
            if r <= 0.0 {
                return id;
            }
        }
        // Float round-off can leave r a hair above 0: fall back to the last kept token.
        self.candidates.last().expect("non-empty vocabulary").0
    }
}

/// Index of the largest logit; ties go to the lowest id, like `torch.argmax`.
fn argmax(candidates: &[(TokenId, f32)]) -> TokenId {
    candidates
        .iter()
        .fold((0, f32::NEG_INFINITY), |best, &(id, l)| {
            if l > best.1 { (id, l) } else { best }
        })
        .0
}

/// SplitMix64 (Steele et al. 2014): a tiny, fast, seedable PRNG with good
/// statistical quality. More than enough for sampling, and keeps runs
/// reproducible without pulling in `rand`.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1): the top 24 bits fill an f32 mantissa exactly.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler(f: impl FnOnce(&mut SamplingParams)) -> Sampler {
        let mut p = SamplingParams {
            seed: 7,
            ..Default::default()
        };
        f(&mut p);
        Sampler::new(p)
    }

    const LOGITS: [f32; 5] = [1.0, 3.0, 2.5, -1.0, 0.0];

    #[test]
    fn greedy_picks_argmax() {
        let mut s = sampler(|_| {});
        assert_eq!(s.sample(&LOGITS, &[]), 1);
    }

    #[test]
    fn top_k_one_is_greedy_at_any_temperature() {
        let mut s = sampler(|p| {
            p.temperature = 5.0;
            p.top_k = 1;
        });
        assert!((0..100).all(|_| s.sample(&LOGITS, &[]) == 1));
    }

    #[test]
    fn tiny_top_p_is_greedy() {
        let mut s = sampler(|p| {
            p.temperature = 1.0;
            p.top_p = 1e-6;
        });
        assert!((0..100).all(|_| s.sample(&LOGITS, &[]) == 1));
    }

    #[test]
    fn top_p_keeps_only_the_nucleus() {
        // softmax(LOGITS) ≈ [.09, .50, .30, .01, .03]: top_p = 0.7 keeps {1, 2}.
        let mut s = sampler(|p| {
            p.temperature = 1.0;
            p.top_p = 0.7;
        });
        for _ in 0..500 {
            assert!(matches!(s.sample(&LOGITS, &[]), 1 | 2));
        }
    }

    #[test]
    fn same_seed_same_tokens() {
        let draw = || {
            let mut s = sampler(|p| p.temperature = 1.0);
            (0..50).map(|_| s.sample(&LOGITS, &[])).collect::<Vec<_>>()
        };
        assert_eq!(draw(), draw());
    }

    #[test]
    fn sampling_follows_the_softmax_distribution() {
        let mut s = sampler(|p| p.temperature = 1.0);
        let n = 20_000;
        let mut counts = [0usize; 5];
        for _ in 0..n {
            counts[s.sample(&LOGITS, &[]) as usize] += 1;
        }
        let mut probs = LOGITS;
        crate::ops::softmax(&mut probs);
        for (c, p) in counts.iter().zip(probs) {
            assert!(
                (*c as f32 / n as f32 - p).abs() < 0.015,
                "{counts:?} vs {probs:?}"
            );
        }
    }

    #[test]
    fn repetition_penalty_demotes_seen_tokens() {
        let mut s = sampler(|p| p.repetition_penalty = 1.5);
        // 3.0 / 1.5 = 2.0 < 2.5, so token 2 now wins.
        assert_eq!(s.sample(&LOGITS, &[1, 1, 1]), 2);
        // Negative logits are multiplied, i.e. pushed further down.
        let mut s = sampler(|p| p.repetition_penalty = 100.0);
        let logits = [-0.1, -0.2];
        assert_eq!(s.sample(&logits, &[0]), 1);
    }
}
