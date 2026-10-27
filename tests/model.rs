//! Forward-pass correctness against HF transformers (fp32), plus KV-cache and
//! generation consistency. Fixture: scripts/reference/gen_logits_fixture.py.

mod common;

use kiln::chat::{ChatTemplate, Message, Role};
use kiln::generate::{Session, StopReason};
use kiln::model::Logits;
use kiln::sampler::{Sampler, SamplingParams};
use serde::Deserialize;

/// Max |kiln - torch| over every compared logit (logits span roughly [-20, 35]).
/// Measured: ~3e-5, i.e. fp32 round-off from a different summation order.
/// The bound leaves headroom for future SIMD/threading reorderings while still
/// failing on any real bug: a wrong RoPE pairing or norm moves logits by >0.1.
const LOGIT_TOLERANCE: f32 = 1e-3;

#[derive(Deserialize)]
struct Fixture {
    top_k: usize,
    cases: Vec<Case>,
    chat_template: Templates,
}

#[derive(Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
    topk_ids: Vec<Vec<u32>>,
    topk_values: Vec<Vec<f32>>,
    greedy: Vec<u32>,
}

#[derive(Deserialize)]
struct Templates {
    default_system: String,
    with_system: String,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("fixtures/logits_ref.json")).expect("valid fixture")
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn full_forward(ids: &[u32]) -> Vec<f32> {
    let model = common::model();
    let mut cache = model.new_cache(ids.len());
    model.forward(ids, &mut cache, Logits::All).unwrap()
}

#[test]
fn tokenizer_reproduces_fixture_prompts() {
    let tok = common::tokenizer();
    for case in fixture().cases {
        assert_eq!(tok.encode(&case.text), case.ids, "{:?}", case.text);
    }
}

#[test]
fn logits_match_transformers() {
    let fx = fixture();
    let vocab = common::model().config().vocab_size;
    let mut max_diff = 0.0f32;
    let mut positions = 0;
    for case in &fx.cases {
        let logits = full_forward(&case.ids);
        for (pos, row) in logits.chunks_exact(vocab).enumerate() {
            for (&id, &want) in case.topk_ids[pos].iter().zip(&case.topk_values[pos]) {
                max_diff = max_diff.max((row[id as usize] - want).abs());
            }
            assert_eq!(
                argmax(row),
                case.topk_ids[pos][0],
                "top-1 differs at position {pos} of {:?}",
                case.text
            );
            positions += 1;
        }
    }
    eprintln!(
        "top-{} logits at {positions} positions: max abs diff {max_diff:.2e}, top-1 identical",
        fx.top_k
    );
    assert!(max_diff < LOGIT_TOLERANCE, "max abs diff {max_diff}");
}

#[test]
fn full_logits_row_matches_transformers() {
    let reference: Vec<f32> = include_bytes!("fixtures/logits_row0.f32")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let ids = &fixture().cases[0].ids;
    let logits = full_forward(ids);
    let last = &logits[logits.len() - reference.len()..];
    let max_diff = last
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let mean_diff = last
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / last.len() as f32;
    eprintln!(
        "full {}-wide row: max abs diff {max_diff:.2e}, mean {mean_diff:.2e}",
        reference.len()
    );
    assert!(max_diff < LOGIT_TOLERANCE, "max abs diff {max_diff}");
    assert_eq!(argmax(last), argmax(&reference));
}

/// Prefill-in-chunks then token-by-token decode must equal one full pass:
/// the KV cache is an optimization, never a change in math.
#[test]
fn incremental_decode_equals_full_recompute() {
    let model = common::model();
    let vocab = model.config().vocab_size;
    let ids = &fixture().cases[1].ids;
    let full = full_forward(ids);

    let mut cache = model.new_cache(ids.len());
    let split = ids.len() / 2;
    let first = model
        .forward(&ids[..split], &mut cache, Logits::Last)
        .unwrap();
    let mut rows = vec![first];
    for &id in &ids[split..] {
        rows.push(model.forward(&[id], &mut cache, Logits::Last).unwrap());
    }
    assert_eq!(cache.len(), ids.len());

    let mut max_diff = 0.0f32;
    for (step, row) in rows.iter().enumerate() {
        let pos = split - 1 + step;
        let want = &full[pos * vocab..(pos + 1) * vocab];
        for (a, b) in row.iter().zip(want) {
            max_diff = max_diff.max((a - b).abs());
        }
    }
    eprintln!("incremental vs full: max abs diff {max_diff:.2e}");
    assert!(max_diff < 1e-4, "max abs diff {max_diff}");
}

#[test]
fn greedy_generation_matches_transformers() {
    let model = common::model();
    for case in fixture().cases {
        let mut session = Session::new(model, case.ids.len() + case.greedy.len());
        let mut sampler = Sampler::new(SamplingParams::default());
        let out = session
            .generate(
                model,
                &case.ids,
                case.greedy.len(),
                &[],
                &mut sampler,
                |_| {},
            )
            .unwrap();
        assert_eq!(out.stop, StopReason::MaxTokens);
        assert_eq!(out.tokens, case.greedy, "{:?}", case.text);
    }
}

#[test]
fn session_reuses_cached_prefix() {
    let model = common::model();
    let ids = &fixture().cases[1].ids;
    let mut session = Session::new(model, 128);
    let mut sampler = Sampler::new(SamplingParams::default());
    let first = session
        .generate(model, ids, 4, &[], &mut sampler, |_| {})
        .unwrap();
    // Extend the conversation with what was generated plus a bit more.
    let mut longer = ids.clone();
    longer.extend(&first.tokens);
    longer.extend([198, 504]);
    let second = session
        .generate(model, &longer, 1, &[], &mut sampler, |_| {})
        .unwrap();
    // Everything except the last sampled token (never fed back) was cached.
    assert_eq!(second.stats.prefill_tokens, 1 + 2);
}

#[test]
fn chat_template_matches_apply_chat_template() {
    let fx = fixture();
    let dir = common::model_dir();
    let template = ChatTemplate::from_tokenizer_config(dir.join("tokenizer_config.json")).unwrap();
    let user = |s: &str| Message::new(Role::User, s);
    assert_eq!(
        template.render(&[user("What is the capital of France?")], true),
        fx.chat_template.default_system
    );
    let convo = [
        Message::new(Role::System, "You are terse."),
        user("Hi!"),
        Message::new(Role::Assistant, "Hello."),
        user("Name a color."),
    ];
    assert_eq!(template.render(&convo, true), fx.chat_template.with_system);
}
