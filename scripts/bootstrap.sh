#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ "$(uname -m)" != arm64 ]; then echo "Use native arm64 Python on macOS" >&2; exit 1; fi
uv sync --locked --python 3.12
.venv/bin/python -m loopqwen doctor
