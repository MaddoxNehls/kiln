"""Reference perplexity on WikiText-2 with HF transformers (fp32), using exactly
Kiln's protocol (`kiln perplexity`): tokenize the whole test file, cut it into
non-overlapping windows of CTX tokens, score the CTX-1 next-token predictions
of each window, report exp(mean NLL) over the first WINDOWS windows.

    uv run --project scripts/reference python scripts/reference/perplexity_ref.py
"""

import math
import pathlib
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "smollm2-135m-instruct"
TEXT = ROOT / "data" / "wikitext-2-raw" / "wiki.test.raw"
CTX = 512
WINDOWS = 40


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    ids = tok(TEXT.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
    print(f"{len(ids)} tokens", file=sys.stderr)
    nll, scored = 0.0, 0
    with torch.no_grad():
        for w in range(min(WINDOWS, len(ids) // CTX)):
            window = torch.tensor([ids[w * CTX : (w + 1) * CTX]])
            logits = model(window).logits[0, :-1].double()
            targets = window[0, 1:]
            nll += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            scored += CTX - 1
    print(f"transformers fp32: perplexity {math.exp(nll / scored):.4f} over {scored} tokens")


if __name__ == "__main__":
    main()
