//! Pre-tokenization: split text into "words" that BPE merges never cross.
//!
//! SmolLM2 chains two HF pre-tokenizers:
//!
//! 1. `Digits { individual_digits: true }` -- every Unicode number character
//!    (`\p{N}`, so also `٣`, `²`, `Ⅻ`) becomes its own piece.
//! 2. `ByteLevel { use_regex: true }` -- GPT-2's split regex, applied to each piece:
//!
//!    ```text
//!    's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//!    ```
//!
//! The regex is implemented as a hand-written scanner rather than pulled in from
//! a regex engine: Rust's `regex` crate has no look-ahead, and the scanner makes
//! the one subtle rule (`\s+(?!\S)`) explicit.

use super::unicode_tables::LETTER;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Letter,
    Number,
    Space,
    Other,
}

fn is_letter(c: char) -> bool {
    let cp = c as u32;
    LETTER
        .binary_search_by(|&(lo, hi)| {
            if hi < cp {
                std::cmp::Ordering::Less
            } else if lo > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

fn class(c: char) -> Class {
    // `char::is_numeric` is exactly general category N (Nd | Nl | No), and
    // `char::is_whitespace` is the White_Space property, which is what `\s` means
    // in the reference (Oniguruma, Unicode mode).
    if c.is_whitespace() {
        Class::Space
    } else if c.is_numeric() {
        Class::Number
    } else if is_letter(c) {
        Class::Letter
    } else {
        Class::Other
    }
}

/// Split `text` into pre-tokens, appending borrowed slices to `out`.
pub(super) fn pre_tokenize<'a>(text: &'a str, split_digits: bool, out: &mut Vec<&'a str>) {
    if !split_digits {
        split_gpt2(text, out);
        return;
    }
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if c.is_numeric() {
            split_gpt2(&text[start..i], out);
            let end = i + c.len_utf8();
            out.push(&text[i..end]);
            start = end;
        }
    }
    split_gpt2(&text[start..], out);
}

fn split_gpt2<'a>(mut s: &'a str, out: &mut Vec<&'a str>) {
    while !s.is_empty() {
        let len = match_len(s);
        out.push(&s[..len]);
        s = &s[len..];
    }
}

/// Byte length of the regex match anchored at the start of non-empty `s`.
/// Alternatives are tried in the regex's order, so earlier ones win.
fn match_len(s: &str) -> usize {
    // 's|'t|'re|'ve|'m|'ll|'d  (lowercase only, as in GPT-2)
    if let Some(after) = s.strip_prefix('\'') {
        for suffix in ["s", "t", "re", "ve", "m", "ll", "d"] {
            if after.starts_with(suffix) {
                return 1 + suffix.len();
            }
        }
    }

    //  ?\p{L}+ |  ?\p{N}+ |  ?[^\s\p{L}\p{N}]+
    // An optional ASCII space, then a run of one non-space class.
    let mut chars = s.chars();
    let first = chars.next().expect("match_len called on empty input");
    let (prefix, head) = match chars.next() {
        Some(next) if first == ' ' && class(next) != Class::Space => (1, next),
        _ => (0, first),
    };
    let head_class = class(head);
    if head_class != Class::Space {
        let run: usize = s[prefix..]
            .chars()
            .take_while(|&c| class(c) == head_class)
            .map(char::len_utf8)
            .sum();
        return prefix + run;
    }

    // \s+(?!\S) | \s+
    // Take the whole whitespace run if it ends the input. Otherwise give the
    // last whitespace char back, so it can prefix the next word (" world")
    // -- unless the run is a single char, in which case `\s+` takes it alone.
    let mut run_len = 0;
    let mut last_start = 0;
    for c in s.chars().take_while(|c| c.is_whitespace()) {
        last_start = run_len;
        run_len += c.len_utf8();
    }
    if run_len == s.len() || last_start == 0 {
        run_len
    } else {
        last_start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(text: &str) -> Vec<&str> {
        let mut out = Vec::new();
        pre_tokenize(text, true, &mut out);
        out
    }

    #[test]
    fn words_take_their_leading_space() {
        assert_eq!(split("Hello world!"), ["Hello", " world", "!"]);
    }

    #[test]
    fn whitespace_run_gives_last_char_to_next_word() {
        assert_eq!(split("a   b"), ["a", "  ", " b"]);
        assert_eq!(split("a \n\tb"), ["a", " \n", "\t", "b"]);
        assert_eq!(split("end   "), ["end", "   "]);
    }

    #[test]
    fn contractions_and_digits() {
        assert_eq!(split("it's 42"), ["it", "'s", " ", "4", "2"]);
        assert_eq!(split("IT'S"), ["IT", "'", "S"]);
    }

    #[test]
    fn letter_table_follows_general_category() {
        assert!(is_letter('é') && is_letter('ж') && is_letter('日'));
        // Combining marks are Alphabetic but not \p{L}.
        assert!(!is_letter('\u{0301}') && !is_letter('\u{093E}'));
    }
}
