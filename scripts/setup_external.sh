#!/usr/bin/env bash
# Install the third-party tools for the CVV and POP arms into vendor/ (gitignored), pinned.
# Nothing is installed into ~/.claude or system-wide: each tool gets its own node_modules / venv here,
# and the arms attach them per run (--plugin-dir / --mcp-config). Needs git, node>=24, npm, uv, ffmpeg.
# Pins must match PINS in src/vwatch/arms/external.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
V="$ROOT/vendor"
CVV_SHA=4a4f99016f249a24018cd19b66531ebbcd58024b
POP_SHA=e6a2b2887c235f5157bb47fdbd3ca8ae9f112cc1
WCPP_SHA=5eaee0cc9024eb9e9a99469199920c8b7f5cb9ec
SMALL_SHA=1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b
mkdir -p "$V"

checkout() {  # url dir sha
  [ -d "$V/$2/.git" ] || git clone -q "$1" "$V/$2"
  git -C "$V/$2" fetch -q origin "$3" 2>/dev/null || true
  git -C "$V/$2" checkout -q "$3"
}

# claude-video-vision: build its MCP server from the pinned source (the plugin's own .mcp.json would
# run `npx claude-video-vision@latest`, i.e. unpinned code from npm). --ignore-scripts: no install hooks.
checkout https://github.com/jordanrendric/claude-video-vision.git claude-video-vision "$CVV_SHA"
(cd "$V/claude-video-vision/mcp-server" && npm ci --ignore-scripts --no-audit --no-fund && npm run build)

# Its "local" audio backend shells out to whisper-cli (whisper.cpp). Built here instead of brew.
checkout https://github.com/ggml-org/whisper.cpp.git whisper.cpp "$WCPP_SHA"
[ -x "$V/build-tools/bin/cmake" ] || { uv venv -q "$V/build-tools" --python 3.12 && uv pip install -q --python "$V/build-tools/bin/python" cmake; }
(cd "$V/whisper.cpp" && PATH="$V/build-tools/bin:$PATH" cmake -B build -DCMAKE_BUILD_TYPE=Release \
   -DWHISPER_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF >/dev/null \
 && PATH="$V/build-tools/bin:$PATH" cmake --build build -j 8 --config Release --target whisper-cli >/dev/null)

# Models the plugin would otherwise curl on first use (same URLs, checksum checked like the plugin does).
mkdir -p "$V/cvv-models"
if [ ! -f "$V/cvv-models/ggml-small.bin" ]; then
  curl -fL -o "$V/cvv-models/ggml-small.bin.part" https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
  echo "$SMALL_SHA  $V/cvv-models/ggml-small.bin.part" | shasum -a 256 -c -
  mv "$V/cvv-models/ggml-small.bin.part" "$V/cvv-models/ggml-small.bin"
fi
[ -f "$V/cvv-models/ggml-silero-v5.1.2.bin" ] || curl -fL -o "$V/cvv-models/ggml-silero-v5.1.2.bin" \
  https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin

# popcorn: node MCP server + a private Python venv for its transcription backends (mlx-whisper is the
# README's pick on Apple Silicon and first in popcorn's auto order; faster-whisper elsewhere).
checkout https://github.com/haithamelmengad/popcorn.git popcorn "$POP_SHA"
(cd "$V/popcorn" && npm ci --ignore-scripts --no-audit --no-fund && npm run build)
[ -x "$V/popcorn-venv/bin/python3" ] || uv venv -q "$V/popcorn-venv" --python 3.12
uv pip install -q --python "$V/popcorn-venv/bin/python" faster-whisper
if [ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ]; then
  uv pip install -q --python "$V/popcorn-venv/bin/python" mlx-whisper
  HF_HOME="$V/hf-cache" "$V/popcorn-venv/bin/python" -c "
from huggingface_hub import snapshot_download
for r in ('mlx-community/whisper-base-mlx', 'mlx-community/whisper-small-mlx'):
    snapshot_download(r)"
fi
echo "external tools ready in $V"
