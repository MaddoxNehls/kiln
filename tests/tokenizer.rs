//! Exact-match tests against HF `tokenizers`.
//! Fixture: scripts/reference/gen_tokenizer_fixture.py.

mod common;

use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
    /// False when the text contains bytes the vocab cannot represent.
    lossless: bool,
}

fn cases() -> Vec<Case> {
    let json = include_str!("fixtures/tokenizer_cases.json");
    serde_json::from_str(json).expect("valid fixture")
}

#[test]
fn encode_matches_hf_tokenizers_exactly() {
    let tok = common::tokenizer();
    let cases = cases();
    let mut failures = Vec::new();
    for case in &cases {
        let ids = tok.encode(&case.text);
        if ids != case.ids {
            failures.push(format!(
                "text {:?}\n  want {:?}\n  got  {:?}",
                case.text, case.ids, ids
            ));
        }
    }
    let total_tokens: usize = cases.iter().map(|c| c.ids.len()).sum();
    assert!(
        failures.is_empty(),
        "{}/{} cases differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
    eprintln!("{} cases, {total_tokens} tokens: exact match", cases.len());
}

#[test]
fn decode_round_trips() {
    let tok = common::tokenizer();
    for case in cases().into_iter().filter(|c| c.lossless) {
        assert_eq!(tok.decode(&case.ids, false), case.text);
    }
}

#[test]
fn stream_decoder_never_splits_characters() {
    let tok = common::tokenizer();
    for case in cases() {
        let expected = tok.decode(&case.ids, false);
        let mut dec = tok.stream_decoder(false);
        let mut out = String::new();
        for &id in &case.ids {
            out.push_str(&dec.push(id));
        }
        out.push_str(&dec.finish());
        assert_eq!(out, expected);
    }
}

#[test]
fn special_tokens_are_recognized_and_skippable() {
    let tok = common::tokenizer();
    let im_start = tok.token_id("<|im_start|>").unwrap();
    let ids = tok.encode("<|im_start|>user");
    assert_eq!(ids[0], im_start);
    assert!(tok.is_special(im_start));
    assert_eq!(tok.decode(&ids, true), "user");
}
