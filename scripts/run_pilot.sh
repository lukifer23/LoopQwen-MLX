#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
py=.venv/bin/python
"$py" -m loopqwen download
"$py" -m loopqwen data --preset pilot
for name in smoke_dense smoke pilot_dense pilot_loop; do
    "$py" -m loopqwen train --config "configs/$name.yaml"
done
"$py" -m loopqwen evaluate --experiment pilot
"$py" -m loopqwen compare --experiment pilot
