#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python=.venv/bin/python
if [ "${1:-}" != "--supervised" ]; then
  exec "$python" scripts/phase2_supervise.py
fi
if ! "$python" -c 'import matplotlib' >/dev/null 2>&1; then
  echo "Install the locked environment with uv sync --locked --group figures" >&2
  exit 1
fi
"$python" -m loopqwen data --preset v2
"$python" scripts/phase2_configs.py
"$python" scripts/phase2_register.py
"$python" -m loopqwen.benchmark.long_cache
"$python" -m loopqwen.benchmark.padded_cache
"$python" -c 'from loopqwen.benchmark.phase2_training import derivatives; from loopqwen.config import config; from loopqwen.io import save; save("outputs/phase2/derivatives.json", derivatives(config("configs/phase2/p2_a_trace_loop.yaml")))'
for run in p2_a_trace_dense p2_a_trace_loop p2_a_answer_dense p2_a_answer_loop; do
  "$python" -m loopqwen train --config "configs/phase2/$run.yaml"
done
"$python" -m loopqwen evaluate --experiment phase2_direct
"$python" scripts/phase2_study.py all
scripts/phase2_finish.sh
