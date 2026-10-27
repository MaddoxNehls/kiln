//! Chat prompt formatting.
//!
//! Instruct models are fine-tuned on one exact prompt format, and drifting from
//! it (a missing newline, a missing default system prompt) measurably degrades
//! answers. HF ships the format as a Jinja template in `tokenizer_config.json`.
//! Kiln does not embed a Jinja engine: it recognizes the ChatML family that
//! SmolLM2 (and Qwen2) use and extracts the template's default system prompt.
//! The tests check the result byte-for-byte against `apply_chat_template`.
//!
//! ```text
//! <|im_start|>system\n{system}<|im_end|>\n
//! <|im_start|>user\n{content}<|im_end|>\n
//! <|im_start|>assistant\n            <- generation prompt; the model writes the rest
//! ```

use std::fmt;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

const START: &str = "<|im_start|>";
const END: &str = "<|im_end|>";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        })
    }
}

#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChatTemplate {
    /// Inserted when a conversation does not start with a system message.
    default_system: Option<String>,
}

#[derive(Deserialize)]
struct TokenizerConfig {
    chat_template: Option<String>,
}

impl ChatTemplate {
    pub fn from_tokenizer_config(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let cfg: TokenizerConfig =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        Self::from_jinja(cfg.chat_template.as_deref().unwrap_or_default())
    }

    fn from_jinja(template: &str) -> Result<Self> {
        ensure!(
            template.contains(START) && template.contains(END),
            "only ChatML (<|im_start|>/<|im_end|>) chat templates are supported"
        );
        // The default system prompt appears as a string literal:
        //   '<|im_start|>system\nYou are a helpful AI assistant ...<|im_end|>
        let marker = format!("{START}system\n");
        let default_system = template.find(&marker).and_then(|i| {
            let rest = &template[i + marker.len()..];
            let body = &rest[..rest.find(END)?];
            // A Jinja expression here means "use the provided message", not a default.
            (!body.contains("{{") && !body.contains("{%")).then(|| body.to_owned())
        });
        Ok(Self { default_system })
    }

    /// The token that ends an assistant turn; generation stops on it.
    pub fn end_of_turn(&self) -> &'static str {
        END
    }

    pub fn render(&self, messages: &[Message], add_generation_prompt: bool) -> String {
        let mut out = String::new();
        let mut turn = |role: Role, content: &str| {
            out.push_str(&format!("{START}{role}\n{content}{END}\n"));
        };
        if let (Some(system), Some(first)) = (&self.default_system, messages.first())
            && first.role != Role::System
        {
            turn(Role::System, system);
        }
        for m in messages {
            turn(m.role, &m.content);
        }
        if add_generation_prompt {
            out.push_str(&format!("{START}{}\n", Role::Assistant));
        }
        out
    }
}
