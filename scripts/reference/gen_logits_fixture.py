"""Generate reference logits for Kiln's forward-pass tests.

Runs SmolLM2-135M-Instruct in fp32 through HF transformers on a few fixed
prompts and writes:

  tests/fixtures/logits_ref.json      per prompt: input ids, top-k ids/values at
                                      every position, and a greedy continuation
  tests/fixtures/logits_row0.f32      the full final-position logits row of
                                      prompt 0, raw little-endian f32

The fixture is deliberately small: top-k everywhere, one full vocab row.

    uv run --project scripts/reference python scripts/reference/gen_logits_fixture.py
"""

import json
import pathlib
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "smollm2-135m-instruct"
OUT_JSON = ROOT / "tests" / "fixtures" / "logits_ref.json"
OUT_ROW = ROOT / "tests" / "fixtures" / "logits_row0.f32"

TOP_K = 10
GREEDY_TOKENS = 24


def main() -> None:
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    # Upcast bf16 weights to fp32 so the reference does the same math Kiln does.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    chat = [{"role": "user", "content": "What is the capital of France?"}]
    chat_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prompts = [
        "The capital of France is",
        chat_text,
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    ]

    cases = []
    for i, text in enumerate(prompts):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            logits = model(torch.tensor([ids])).logits[0]  # [seq, vocab]
            top = torch.topk(logits, TOP_K, dim=-1)
            out = model.generate(
                torch.tensor([ids]),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                max_new_tokens=GREEDY_TOKENS,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=tok.pad_token_id,
            )
        cases.append(
            {
                "text": text,
                "ids": ids,
                "topk_ids": top.indices.tolist(),
                "topk_values": top.values.tolist(),
                "greedy": out[0, len(ids) :].tolist(),
            }
        )
        if i == 0:
            logits[-1].numpy().astype("<f4").tofile(OUT_ROW)

    rendered = tok.apply_chat_template(
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Name a color."},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    fixture = {
        "top_k": TOP_K,
        "cases": cases,
        "chat_template": {"default_system": chat_text, "with_system": rendered},
    }
    OUT_JSON.write_text(json.dumps(fixture, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"wrote {len(cases)} prompts -> {OUT_JSON.relative_to(ROOT)}, "
        f"{OUT_ROW.relative_to(ROOT)} ({np.fromfile(OUT_ROW, '<f4').size} floats)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
