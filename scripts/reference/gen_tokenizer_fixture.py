"""Generate exact-match fixtures for Kiln's BPE tokenizer.

Encodes a varied text sample with the reference HF `tokenizers` implementation
and writes {text, ids} pairs to tests/fixtures/tokenizer_cases.json.

    uv run --project scripts/reference python scripts/reference/gen_tokenizer_fixture.py
"""

import json
import pathlib
import sys

from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "smollm2-135m-instruct"
OUT = ROOT / "tests" / "fixtures" / "tokenizer_cases.json"

CASES = [
    "",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces and trailing spaces   ",
    "tabs\tand\nnewlines\r\n\n\nand    runs     of      spaces",
    "It's, I'm, you're, we've, they'll, he'd, IT'S, don't",
    "Numbers: 3.14159, 1,000,000 and 2024-10-27 plus 007 and x86_64",
    "Arabic-Indic digits ٠١٢٣ and fullwidth １２３ and superscript ²³ and roman Ⅻ",
    "def fib(n: int) -> int:\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
    "fn main() {\n\tlet v: Vec<u8> = (0..=255).collect();\n\tprintln!(\"{:?}\", &v[..4]);\n}\n",
    "<html><body class=\"x\">a &amp; b</body></html>",
    "SELECT id, name FROM users WHERE age >= 21 ORDER BY name DESC;",
    "naïve café résumé façade Ångström Zürich",
    "Привет, мир! Как дела?",
    "日本語のテキストと中文文本以及한국어 텍스트",
    "नमस्ते दुनिया, यह हिंदी है",
    "مرحبا بالعالم",
    "Emoji: 😀😃😄 👍🏽 👨‍👩‍👧‍👦 🇺🇸 ❤️ 🔥🔥🔥",
    "Math: ∀x ∈ ℝ, ∑ᵢ xᵢ² ≥ 0 and α+β=γ",
    "Mixed\u00a0nbsp\u2009thin\u3000ideographic spaces",
    "Zero\u200bwidth\u200djoiner and combining e\u0301 a\u0308",
    "<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n",
    "Special-looking but not special: <|im_start  |> <im_end> <|endoftext|>!",
    "URL https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct?x=1&y=2#frag",
    r"email: someone@example.com, path C:\Users\name\file.txt",
    "!!!??? ... --- *** ### @@@ $$$ %%% ^^^",
    "CamelCaseIdentifier snake_case_identifier SCREAMING_CASE kebab-case",
    "    \n    \n",
    " ",
    "\n",
    "a",
    "'s 's 'S",
    "1 12 123 1234 12345",
    # Bytes 0x04/0x06 have no vocab entry (no unk, no byte_fallback): HF drops them.
    "control\x04char\x06s and a bell\x07",
]


def main() -> None:
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    long_text = "\n\n".join(CASES[1:]) * 3
    cases = []
    for text in CASES + [long_text]:
        ids = tok.encode(text).ids
        # Round-trip check on the reference side (except where bytes are dropped).
        lossless = tok.decode(ids, skip_special_tokens=False) == text
        assert lossless or "\x04" in text, repr(text)
        cases.append({"text": text, "ids": ids, "lossless": lossless})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases -> {OUT.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
