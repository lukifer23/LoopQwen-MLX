#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python=.venv/bin/python
"$python" -m loopqwen data --preset v2
"$python" scripts/phase2_configs.py
"$python" scripts/phase2_register.py
for run in p2_a_trace_dense p2_a_trace_loop p2_a_answer_dense p2_a_answer_loop; do
  "$python" -m loopqwen train --config "configs/phase2/$run.yaml"
done
"$python" -m loopqwen evaluate --experiment phase2_direct
"$python" scripts/phase2_study.py all
