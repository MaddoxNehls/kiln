//! Model hyper-parameters, read from the Hugging Face `config.json`.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// GQA: fewer K/V heads than query heads. Absent means plain multi-head attention.
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    pub eos_token_id: Option<OneOrMany>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany {
    One(u32),
    Many(Vec<u32>),
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_act() -> String {
    "silu".into()
}

impl Config {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let cfg: Self =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reject configurations this engine would silently compute wrong.
    fn validate(&self) -> Result<()> {
        ensure!(
            self.hidden_act == "silu",
            "unsupported activation {:?}",
            self.hidden_act
        );
        ensure!(
            !self.attention_bias && !self.mlp_bias,
            "linear biases are not supported"
        );
        ensure!(
            self.num_attention_heads.is_multiple_of(self.num_kv_heads()),
            "query heads must be a multiple of kv heads"
        );
        ensure!(
            self.head_dim().is_multiple_of(2),
            "RoPE needs an even head_dim"
        );
        self.rope_position_scale()?;
        Ok(())
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Factor that positions are divided by before computing RoPE angles.
    /// `rope_scaling: null` gives 1.0; "linear" scaling (position
    /// interpolation) gives `factor`. Other schemes (llama3, yarn, dynamic) are
    /// rejected rather than approximated.
    pub fn rope_position_scale(&self) -> Result<f32> {
        let Some(scaling) = &self.rope_scaling else {
            return Ok(1.0);
        };
        let kind = scaling["rope_type"]
            .as_str()
            .or_else(|| scaling["type"].as_str())
            .unwrap_or("default");
        match kind {
            "default" => Ok(1.0),
            "linear" => Ok(scaling["factor"]
                .as_f64()
                .context("linear rope_scaling needs a factor")? as f32),
            other => bail!("unsupported rope_scaling type {other:?}"),
        }
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
            None => Vec::new(),
            Some(OneOrMany::One(id)) => vec![*id],
            Some(OneOrMany::Many(ids)) => ids.clone(),
        }
    }
}
