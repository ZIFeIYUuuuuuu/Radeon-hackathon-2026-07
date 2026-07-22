#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-https://github.com/ZIFeIYUuuuuuu/Radeon-hackathon-2026-07.git}"
BRANCH="${BRANCH:-claimcourt}"
DEST="${DEST:-/workspace/claimcourt}"

if [ -e "$DEST/.git" ]; then
  git -C "$DEST" fetch origin "$BRANCH"
  git -C "$DEST" reset --hard "origin/$BRANCH"
else
  git clone --depth 1 --branch "$BRANCH" "$REPO" "$DEST"
fi

cd "$DEST"
/opt/venv/bin/python -m pip install -r requirements.txt
mkdir -p data

echo "ClaimCourt source restored at $DEST"
echo "Start Ollama with qwen3:32b-q8_0, then run:"
echo "  /opt/venv/bin/streamlit run app.py --server.address 127.0.0.1 --server.port 8502"
