#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
exec .venv/bin/python -c 'from pathlib import Path
import sys
from scripts.phase2_supervise import supervise
sys.exit(supervise([sys.executable, "-m", "loopqwen.training.rsi"], Path("outputs/rsi1/study.lock")))'
