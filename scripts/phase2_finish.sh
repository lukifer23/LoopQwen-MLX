#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python=.venv/bin/python
"$python" -m loopqwen.evaluation.statistics
"$python" -m loopqwen.evaluation.inference_budget
"$python" -m loopqwen.evaluation.phase2_controls dev
"$python" -m loopqwen.evaluation.phase2_controls transfer
"$python" -m loopqwen.benchmark.phase2_training
"$python" -m loopqwen.evaluation.merge_audit p2_confirm_43_loop
"$python" scripts/phase2_repeat.py
"$python" -m loopqwen.benchmark.phase2_inference
"$python" scripts/phase2_anchor.py
"$python" scripts/phase2_integrity.py
"$python" scripts/phase2_compute.py
"$python" scripts/phase2_archive.py
"$python" scripts/phase2_report.py
"$python" -m loopqwen generate --checkpoint adapters/p2_confirm_43_loop --prompt 'Evaluate ((8 + 4) * 3 - 5).
Show your steps. End with Final answer: <number>.
Solution:' --max-tokens 96 > outputs/phase2/example.json
