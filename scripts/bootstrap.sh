#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ "$(uname -m)" != arm64 ]; then echo "Use native arm64 Python on macOS" >&2; exit 1; fi
if command -v uv >/dev/null 2>&1; then
    uv sync --locked --python 3.12
else
    python3.12 -m venv .venv
    constraints=$(mktemp)
    trap 'rm -f "$constraints"' EXIT
    .venv/bin/python -c 'import tomllib; print("\n".join(p["name"]+"=="+p["version"] for p in tomllib.load(open("uv.lock","rb"))["package"]))' > "$constraints"
    .venv/bin/python -m pip install -c "$constraints" -e . pytest ruff
fi
.venv/bin/python -m loopqwen doctor
