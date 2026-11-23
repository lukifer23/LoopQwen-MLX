# Phase 3 development comparison

Seed 73. Same V2 training order. Greedy traced generation on 60 development items.

| Arm | Execution | Correct | Verified | Truncated | Final-token NLL | Trace NLL | Prose NLL | Generation seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dense | trained | 36/60 | 34 | 0 | 0.0000 | 0.0603 | 3.6378 | 15.3 |
| direct | trained | 30/60 | 28 | 0 | 0.0000 | 0.0631 | 3.7682 | 19.1 |
| direct | bypass | 26/60 | 22 | 0 | 0.0000 | 0.0835 | 3.6388 | 16.1 |
| learn1 | trained | 36/60 | 35 | 0 | 0.0000 | 0.0605 | 3.6369 | 18.8 |
| learn1 | bypass | 35/60 | 34 | 0 | 0.0000 | 0.0606 | 3.6352 | 15.4 |
| learn1 | alpha0 | 35/60 | 34 | 0 | 0.0000 | 0.0606 | 3.6352 | 18.8 |
| learn1 | alpha0.05 | 36/60 | 34 | 0 | 0.0000 | 0.0604 | 3.6358 | 18.7 |
| learn1 | alpha0.10 | 38/60 | 36 | 0 | 0.0000 | 0.0606 | 3.6363 | 19.1 |

Paired changes count fixes and breaks in the model named first relative to its comparator.

| Pair | Fixes | Breaks | Difference (pp) | Paired item bootstrap 95% (pp) | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| direct-dense | 1 | 7 | -10.0 | [-18.3, -1.7] | 0.070 |
| direct-bypass | 7 | 3 | 6.7 | [-3.3, 16.7] | 0.344 |
| learn1-dense | 2 | 2 | 0.0 | [-6.7, 6.7] | 1.000 |
| learn1-bypass | 2 | 1 | 1.7 | [-3.3, 8.3] | 1.000 |

At 100,068 targets, the learned coefficient moved from 0.10000 to 0.09803. Dense and gated K=2 tied on final-answer correctness. Gated K=2 fixed two bypass errors and broke one correct bypass answer. The observed one-item net advantage triggered the registered 250K extension. The inference-time alpha sweep uses this same development cohort. Its highest observed accuracy came at alpha 0.10, which yielded 38/60. That sensitivity measurement has no confirmatory status.

These three primary checkpoints each saw 2,516 unique procedural instances and 236,835 useful input tokens. [Unmerged adapters](../adapters/phase3) · [Exact exposures](measurements/phase3/exposure.json) · [Resource and timing measurements](PHASE3_OPTIMIZATIONS.md)
