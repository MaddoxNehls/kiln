#!/usr/bin/env bash
# Download SmolLM2-135M-Instruct weights + tokenizer from Hugging Face.
# Usage: scripts/download-model.sh [repo] [dest]
set -euo pipefail

REPO="${1:-HuggingFaceTB/SmolLM2-135M-Instruct}"
DEST="${2:-models/smollm2-135m-instruct}"
BASE="https://huggingface.co/${REPO}/resolve/main"

mkdir -p "$DEST"
for f in config.json tokenizer.json tokenizer_config.json generation_config.json model.safetensors; do
  if [[ -s "$DEST/$f" ]]; then
    echo "skip   $f (exists)"
    continue
  fi
  echo "fetch  $f"
  curl -fL --retry 3 -o "$DEST/$f.part" "$BASE/$f"
  mv "$DEST/$f.part" "$DEST/$f"
done
echo "done -> $DEST"
