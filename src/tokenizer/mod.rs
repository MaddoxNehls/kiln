//! Byte-level BPE tokenizer, loaded from a Hugging Face `tokenizer.json`.
//!
//! Encoding pipeline (mirrors HF `tokenizers` for GPT-2-style models):
//!
//! ```text
//! text ─► split out added/special tokens ("<|im_start|>") ─► pre-tokenize each
//!      ─► UTF-8 bytes of each word ─► BPE merges ─► token ids
//! ```
//!
//! BPE runs on token *ids*, not strings: a word starts as one id per byte, and
//! merging the adjacent pair `(a, b)` with the lowest merge rank replaces it
//! with the merged id. The `(a, b) -> (rank, merged)` table is built once at load.

mod byte_level;
mod pretokenize;
mod unicode_tables;

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

pub type TokenId = u32;

pub struct Tokenizer {
    /// Raw bytes of every token, indexed by id. Decoding is concatenation.
    id_to_bytes: Vec<Vec<u8>>,
    /// Token id of each single byte. SmolLM2's vocab lacks 21 of them (rare
    /// control bytes and bytes that never occur in valid UTF-8).
    byte_to_id: [Option<TokenId>; 256],
    merges: HashMap<(TokenId, TokenId), Merge>,
    /// Added tokens (`<|im_start|>`, ...) matched verbatim before pre-tokenization.
    added: Vec<(String, TokenId)>,
    special: Vec<bool>,
    split_digits: bool,
}

#[derive(Clone, Copy)]
struct Merge {
    rank: u32,
    id: TokenId,
}

impl Tokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let json =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let file: TokenizerFile =
            serde_json::from_str(&json).with_context(|| format!("parsing {}", path.display()))?;
        Self::from_parsed(file)
    }

    fn from_parsed(file: TokenizerFile) -> Result<Self> {
        let model = file.model;
        ensure!(
            model.kind == "BPE",
            "unsupported tokenizer model {:?}",
            model.kind
        );
        ensure!(
            file.normalizer.is_none(),
            "tokenizer normalizers are not supported"
        );
        let split_digits = parse_pre_tokenizer(&file.pre_tokenizer)?;

        let byte_decoder = byte_level::ByteDecoder::new();
        let vocab_size = model.vocab.len().max(
            file.added_tokens
                .iter()
                .map(|t| t.id as usize + 1)
                .max()
                .unwrap_or(0),
        );
        let mut id_to_bytes = vec![Vec::new(); vocab_size];
        let mut bytes_to_id = HashMap::with_capacity(model.vocab.len());
        for (token, &id) in &model.vocab {
            let bytes = byte_decoder
                .decode(token)
                .with_context(|| format!("vocab entry {token:?} is not byte-level encoded"))?;
            id_to_bytes[id as usize] = bytes.clone();
            bytes_to_id.insert(bytes, id);
        }

        let mut byte_to_id = [None; 256];
        for (byte, slot) in byte_to_id.iter_mut().enumerate() {
            *slot = bytes_to_id.get(&vec![byte as u8]).copied();
        }

        let mut merges = HashMap::with_capacity(model.merges.len());
        for (rank, merge) in model.merges.iter().enumerate() {
            let (left, right) = merge.pair()?;
            let lookup = |s: &str| -> Result<(TokenId, Vec<u8>)> {
                let bytes = byte_decoder
                    .decode(s)
                    .with_context(|| format!("merge part {s:?} is not byte-level encoded"))?;
                let id = *bytes_to_id
                    .get(&bytes)
                    .with_context(|| format!("merge part {s:?} not in vocab"))?;
                Ok((id, bytes))
            };
            let (l, mut merged) = lookup(left)?;
            let (r, r_bytes) = lookup(right)?;
            merged.extend_from_slice(&r_bytes);
            let id = *bytes_to_id
                .get(&merged)
                .with_context(|| format!("merge result {left}{right} not in vocab"))?;
            // Earlier merges win if a pair is listed twice, as in HF.
            merges.entry((l, r)).or_insert(Merge {
                rank: rank as u32,
                id,
            });
        }

        let mut special = vec![false; vocab_size];
        let mut added = Vec::with_capacity(file.added_tokens.len());
        for t in file.added_tokens {
            ensure!(
                !(t.lstrip || t.rstrip || t.single_word || t.normalized),
                "added token {:?} uses unsupported matching options",
                t.content
            );
            id_to_bytes[t.id as usize] = t.content.as_bytes().to_vec();
            special[t.id as usize] = t.special;
            added.push((t.content, t.id));
        }

        Ok(Self {
            id_to_bytes,
            byte_to_id,
            merges,
            added,
            special,
            split_digits,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    /// Id of an added token by its exact text, e.g. `"<|im_end|>"`.
    pub fn token_id(&self, content: &str) -> Option<TokenId> {
        self.added
            .iter()
            .find(|(c, _)| c == content)
            .map(|&(_, id)| id)
    }

    pub fn is_special(&self, id: TokenId) -> bool {
        self.special.get(id as usize).copied().unwrap_or(false)
    }

    /// Encode text to token ids. Added tokens appearing in `text` are recognized
    /// (as HF does by default); no BOS/EOS is inserted.
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        let mut ids = Vec::with_capacity(text.len() / 3);
        let mut words = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let (plain, added) = self.split_at_added_token(rest);
            words.clear();
            pretokenize::pre_tokenize(plain, self.split_digits, &mut words);
            for word in &words {
                self.bpe(word.as_bytes(), &mut ids);
            }
            match added {
                Some((id, len)) => {
                    ids.push(id);
                    rest = &rest[plain.len() + len..];
                }
                None => break,
            }
        }
        ids
    }

    /// Find the leftmost added token in `text` (longest wins on ties). Returns
    /// the plain text before it, and the token's id and byte length.
    fn split_at_added_token<'a>(&self, text: &'a str) -> (&'a str, Option<(TokenId, usize)>) {
        for (pos, _) in text.char_indices() {
            let tail = &text[pos..];
            let best = self
                .added
                .iter()
                .filter(|(content, _)| tail.starts_with(content.as_str()))
                .max_by_key(|(content, _)| content.len());
            if let Some((content, id)) = best {
                return (&text[..pos], Some((*id, content.len())));
            }
        }
        (text, None)
    }

    /// Byte-pair-encode one pre-token, appending its ids to `out`.
    ///
    /// Each step merges the adjacent pair with the lowest rank. Words are short
    /// (a few bytes to a few dozen), so the O(n^2) scan beats a heap here.
    ///
    /// A byte with no vocab entry is dropped, which is what HF does for a BPE
    /// model with neither `unk_token` nor `byte_fallback`.
    fn bpe(&self, word: &[u8], out: &mut Vec<TokenId>) {
        let mut parts: Vec<TokenId> = word
            .iter()
            .filter_map(|&b| self.byte_to_id[b as usize])
            .collect();
        loop {
            let best = parts
                .windows(2)
                .enumerate()
                .filter_map(|(i, pair)| self.merges.get(&(pair[0], pair[1])).map(|m| (i, *m)))
                .min_by_key(|(_, m)| m.rank);
            let Some((i, merge)) = best else { break };
            parts[i] = merge.id;
            parts.remove(i + 1);
        }
        out.extend_from_slice(&parts);
    }

    /// Raw bytes of a token. Byte-level tokens may hold partial UTF-8 sequences.
    pub fn token_bytes(&self, id: TokenId) -> &[u8] {
        self.id_to_bytes.get(id as usize).map_or(&[], Vec::as_slice)
    }

    /// Decode ids to text. Invalid UTF-8 (e.g. a sequence cut mid-character) is
    /// replaced with U+FFFD, matching HF.
    pub fn decode(&self, ids: &[TokenId], skip_special: bool) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter(|&&id| !(skip_special && self.is_special(id)))
            .flat_map(|&id| self.token_bytes(id).iter().copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Incremental decoder for streaming output, one token at a time.
    pub fn stream_decoder(&self, skip_special: bool) -> StreamDecoder<'_> {
        StreamDecoder {
            tokenizer: self,
            pending: Vec::new(),
            skip_special,
        }
    }
}

/// Streams text out of a token sequence without ever splitting a character.
///
/// A single emoji is often 2-4 byte-level tokens, so printing each token's
/// bytes as they arrive would emit garbage. Bytes are buffered until they form
/// complete UTF-8, and only that prefix is released.
pub struct StreamDecoder<'t> {
    tokenizer: &'t Tokenizer,
    pending: Vec<u8>,
    skip_special: bool,
}

impl StreamDecoder<'_> {
    /// Feed one token; returns any text that is now complete.
    pub fn push(&mut self, id: TokenId) -> String {
        if !(self.skip_special && self.tokenizer.is_special(id)) {
            self.pending
                .extend_from_slice(self.tokenizer.token_bytes(id));
        }
        let mut text = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    text.push_str(s);
                    self.pending.clear();
                    return text;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    // `valid_up_to` marks a prefix already verified as UTF-8.
                    text.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap());
                    match e.error_len() {
                        // Incomplete sequence at the end: wait for more bytes.
                        None => {
                            self.pending.drain(..valid);
                            return text;
                        }
                        // Genuinely invalid bytes: replace and keep going.
                        Some(bad) => {
                            text.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..valid + bad);
                        }
                    }
                }
            }
        }
    }

    /// Flush whatever is left at end of stream (lossily).
    pub fn finish(&mut self) -> String {
        let text = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        text
    }
}

/// Accepts the pre-tokenizer configurations Kiln implements and reports whether
/// digits are split individually.
fn parse_pre_tokenizer(cfg: &serde_json::Value) -> Result<bool> {
    let kind = |v: &serde_json::Value| v["type"].as_str().unwrap_or_default().to_owned();
    let byte_level_ok = |v: &serde_json::Value| {
        kind(v) == "ByteLevel"
            && v["add_prefix_space"] == false
            && v["use_regex"].as_bool().unwrap_or(true)
    };
    match kind(cfg).as_str() {
        "ByteLevel" if byte_level_ok(cfg) => Ok(false),
        "Sequence" => match cfg["pretokenizers"].as_array().map(Vec::as_slice) {
            Some([digits, byte_level])
                if kind(digits) == "Digits"
                    && digits["individual_digits"] == true
                    && byte_level_ok(byte_level) =>
            {
                Ok(true)
            }
            _ => bail!("unsupported pre_tokenizer sequence: {cfg}"),
        },
        _ => bail!("unsupported pre_tokenizer: {cfg}"),
    }
}

#[derive(Deserialize)]
struct TokenizerFile {
    added_tokens: Vec<AddedToken>,
    normalizer: Option<serde_json::Value>,
    pre_tokenizer: serde_json::Value,
    model: BpeModel,
}

#[derive(Deserialize)]
struct AddedToken {
    id: TokenId,
    content: String,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    normalized: bool,
    #[serde(default)]
    special: bool,
}

#[derive(Deserialize)]
struct BpeModel {
    #[serde(rename = "type")]
    kind: String,
    vocab: HashMap<String, TokenId>,
    merges: Vec<MergeEntry>,
}

/// `tokenizer.json` stores merges either as `"a b"` or (newer files) `["a", "b"]`.
#[derive(Deserialize)]
#[serde(untagged)]
enum MergeEntry {
    Joined(String),
    Pair(String, String),
}

impl MergeEntry {
    fn pair(&self) -> Result<(&str, &str)> {
        match self {
            Self::Joined(s) => s
                .split_once(' ')
                .with_context(|| format!("malformed merge {s:?}")),
            Self::Pair(a, b) => Ok((a, b)),
        }
    }
}
