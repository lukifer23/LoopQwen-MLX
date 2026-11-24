#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
case "${1:-pilot}" in
  pilot) module=loopqwen.training.rsi ;;
  traces) module=loopqwen.training.verified_trace ;;
  *) echo "usage: scripts/run_rsi.sh [pilot|traces]" >&2; exit 2 ;;
esac
exec .venv/bin/python -c 'from pathlib import Path
import sys
from scripts.phase2_supervise import supervise
sys.exit(supervise([sys.executable, "-m", sys.argv[1]], Path("outputs/rsi1/study.lock")))' "$module"
