# Next experiment

**Compare a trained K=1 and a trained K=3 decoder on terminal-only vocabulary answers over arbitrary successor maps.** Both receive the same explicit three-hop question, initial adapted weights, training data and response targets. A terminal-only protocol measures whether internal computation adds value when the model cannot write intermediate states into its output. The previous complete-trace control scored 96/96 at K=1. The six-hop categorical checkpoint produced only 6/192 complete trajectories on unrestricted successor maps. Those observations justify changing the evaluation task before increasing model capacity.

The registration fixes 256 new training maps, 128 development maps and 32 updates per arm. The K=3 advantage must reach five points and meet the categorical retention safeguard. The command has been implemented and syntax-checked. Native execution is pending a new authorized session. [Registration](../research/experiments/functional-terminal-01.json)

```sh
.venv/bin/python scripts/research_session.py
.venv/bin/python -m loopqwen.research.terminal
```

The expensive diagnostic curriculum, verified-success filtering on three-digit traces, and frozen-prefix caching have failed their respective cost or numerical tests. The frozen-block suffix trace fork preserved symbolic accuracy and underperformed on language output. Keep the gold-supervised combined checkpoint for retention-sensitive work. The single-pass trace fork is the preferred specialized room-trace deployment on the measured cohort.
