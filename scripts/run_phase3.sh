#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python=.venv/bin/python
if [ "${1:-}" != "--supervised" ]; then
  exec "$python" scripts/phase2_supervise.py --phase3 "$@"
fi
"$python" scripts/phase2_supervise.py --check outputs/phase3/study.lock
shift
stage=${1:-screening}
study() { "$python" scripts/phase3_study.py "$@" --checkpointing off; }
case "$stage" in
smoke|screening|main)
  for arm in learn1 learn2; do
    if [ ! -f "outputs/phase3/$arm-lr.json" ]; then
      study "$arm" --updates 8
      "$python" -m loopqwen.evaluation.gate_screen "$arm"
    fi
  done
  [ "$stage" != smoke ] || exit 0
  learned=$("$python" -c 'from loopqwen.io import read; print(min(("learn1","learn2"),key=lambda a:read(f"outputs/phase3/{a}-lr.json")["loss"]["nll"]))')
  for arm in dense direct fixed "$learned"; do study "$arm" --targets 25000; done
  "$python" -m loopqwen.evaluation.phase3 screen "$learned"
  [ "$stage" = main ] || exit 0
  selected=$("$python" -c 'from loopqwen.io import read; print(read("outputs/phase3/selection.json")["selected"])')
  for arm in dense direct "$selected"; do study "$arm" --targets 100000; done
  "$python" -m loopqwen.evaluation.phase3 main "$selected"
  ;;
*) echo "Use smoke, screening or main" >&2; exit 2;;
esac
