//! Perplexity: how surprised the model is by real text, `exp(mean NLL)`.
//!
//! The protocol is fixed so numbers are comparable across formats and with the
//! HF reference (`scripts/reference/perplexity_ref.py`): the text is
//! tokenized once and cut into non-overlapping windows of `ctx` tokens
//! (stride = ctx). Each window starts from an empty KV cache and scores its
//! `ctx - 1` next-token predictions; the first token of a window has no
//! context and is not scored.

use anyhow::{Result, ensure};

use crate::model::{Logits, Model};
use crate::tokenizer::TokenId;

#[derive(Clone, Copy, Debug)]
pub struct Perplexity {
    pub windows: usize,
    pub scored_tokens: usize,
    /// Mean negative log-likelihood, nats per token.
    pub mean_nll: f64,
}

impl Perplexity {
    pub fn ppl(&self) -> f64 {
        self.mean_nll.exp()
    }
}

/// Evaluate up to `max_windows` windows of `ctx` tokens. `on_window` gets the
/// running result after each window (progress reporting).
pub fn evaluate(
    model: &Model,
    tokens: &[TokenId],
    ctx: usize,
    max_windows: usize,
    mut on_window: impl FnMut(&Perplexity),
) -> Result<Perplexity> {
    ensure!(ctx >= 2, "context must be at least 2 tokens");
    let windows = (tokens.len() / ctx).min(max_windows);
    ensure!(windows > 0, "text has fewer than {ctx} tokens");
    let vocab = model.config().vocab_size;
    let mut cache = model.new_cache(ctx);
    let mut nll_sum = 0.0f64;
    let mut result = Perplexity {
        windows: 0,
        scored_tokens: 0,
        mean_nll: 0.0,
    };
    for window in tokens.chunks_exact(ctx).take(windows) {
        cache.truncate(0);
        let logits = model.forward(window, &mut cache, Logits::All)?;
        for (row, &target) in logits.chunks_exact(vocab).zip(&window[1..]) {
            nll_sum += log_sum_exp(row) - f64::from(row[target as usize]);
        }
        result.windows += 1;
        result.scored_tokens += ctx - 1;
        result.mean_nll = nll_sum / result.scored_tokens as f64;
        on_window(&result);
    }
    Ok(result)
}

/// `log Σ exp(x_i)`, shifted by the max so nothing overflows; accumulated in
/// f64 because the sum runs over 49k terms.
fn log_sum_exp(row: &[f32]) -> f64 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row.iter().map(|&v| f64::from(v - max).exp()).sum();
    f64::from(max) + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_sum_exp_is_stable() {
        let v = log_sum_exp(&[1000.0, 1000.0]);
        assert!((v - (1000.0 + 2f64.ln())).abs() < 1e-9);
    }
}
