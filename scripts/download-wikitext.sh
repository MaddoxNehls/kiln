#!/usr/bin/env bash
# Download the WikiText-2 (raw) test split used for perplexity evaluation.
# Usage: scripts/download-wikitext.sh [dest]
set -euo pipefail

DEST="${1:-data/wikitext-2-raw}"
URL="https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
SHA256="ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"

if [[ -s "$DEST/wiki.test.raw" ]]; then
  echo "skip   $DEST/wiki.test.raw (exists)"
  exit 0
fi
mkdir -p "$DEST"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl -fL --retry 3 -o "$tmp/wikitext.zip" "$URL"
echo "$SHA256  $tmp/wikitext.zip" | sha256sum -c -
unzip -q -j "$tmp/wikitext.zip" "wikitext-2-raw/wiki.test.raw" -d "$DEST"
echo "done -> $DEST/wiki.test.raw ($(wc -c < "$DEST/wiki.test.raw") bytes)"
