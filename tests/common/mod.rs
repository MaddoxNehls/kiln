#![allow(dead_code)] // each test binary uses a different subset

use std::path::PathBuf;
use std::sync::OnceLock;

use kiln::model::Model;
use kiln::tokenizer::Tokenizer;

/// Model directory (override with `KILN_MODEL_DIR`). Panics with instructions if
/// the weights are missing: a correctness suite that silently skips is worse
/// than one that fails.
pub fn model_dir() -> PathBuf {
    let dir = std::env::var_os("KILN_MODEL_DIR").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/smollm2-135m-instruct"),
        PathBuf::from,
    );
    assert!(
        dir.join("model.safetensors").exists(),
        "no model at {}; run scripts/download-model.sh first",
        dir.display()
    );
    dir
}

pub fn tokenizer() -> &'static Tokenizer {
    static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();
    TOKENIZER.get_or_init(|| {
        Tokenizer::from_file(model_dir().join("tokenizer.json")).expect("load tokenizer")
    })
}

/// The fp32 model is ~540 MB; load it once per test binary.
pub fn model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| Model::load(model_dir()).expect("load model"))
}
