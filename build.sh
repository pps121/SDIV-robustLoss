#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# build.sh  —  Convert Research_Understanding_Writeup.md → .html
# Usage:  bash build.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="$DIR/Research_Understanding_Writeup.md"
OUTPUT="$DIR/Research_Understanding_Writeup.html"
CSS="$DIR/style.css"

if ! command -v pandoc &>/dev/null; then
  echo "Error: pandoc not found. Install via: brew install pandoc"
  exit 1
fi

pandoc "$INPUT" \
  --standalone \
  --css="style.css" \
  --mathjax \
  --metadata title="Robust Neural Learning via Divergences — Research Understanding" \
  --highlight-style=tango \
  --from markdown+tex_math_dollars+smart \
  --to html5 \
  --output "$OUTPUT"

echo "✓ Generated: $OUTPUT"
echo "  Open in browser → File > Print → Save as PDF"
