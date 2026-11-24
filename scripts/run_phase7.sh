#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
exec .venv/bin/python -c '
import sys
from pathlib import Path
from scripts.phase2_supervise import supervise
rc = supervise([sys.executable, "-m", "loopqwen.phase7", *sys.argv[1:]], Path("outputs/phase7/study.lock"))
sys.exit(128-rc if rc<0 else rc)
' "$@"
