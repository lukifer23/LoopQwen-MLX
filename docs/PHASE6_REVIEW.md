# Independent review handoff

Review the published Phase 6 commit on `main` and provide one bounded next research prompt. Start with the [results](PHASE6_RESULTS.md), [pretraining registration](../experiments/phase6.json), [optimization report](PHASE6_OPTIMIZATIONS.md) and [rapid iteration policy](TRAINING.md#default-to-a-two-model-iteration). The original Phase 5 confirmation cohort stays sealed.

Verify the matched schedule, separate display-order streams and release checksums. Recompute the canonical floor from all 384 maps and compare the eight shuffled orders as correlated observations of each map. Inspect both the gain on shuffled displays and the loss on canonical inputs. Confirm the random, reverse and rotated conditions are paired by map and starting state. Check the three-head dense arm's label count and its extra 16,384 weights. The CPU archive verifier scores stored model probabilities. The native reload command recomputes predictions from released weights.

```sh
python3 scripts/verify_phase4_archive.py
python3 scripts/verify_phase5_archive.py
python3 scripts/verify_phase6_archive.py
.venv/bin/python scripts/phase6_predict.py --arm randomized --verify
```

Review the separately registered [two-model deployment comparison](../experiments/phase6_reasoning_pair.json). Both arms receive byte-identical thinking-enabled question prompts. Inspect visible-answer scoring, truncation, per-item wins and losses, actual generated tokens and synchronized latency. Keep categorical and ordinary vocabulary-head results distinct.

**Proposed next hypothesis.** One presentation per training map and 1,152 explicit third-hop targets may be insufficient to consolidate the third transition. Register a single additional randomized epoch with fresh deterministic displays, the same maps, original optimizer state, rank, depth mix and shared block. Preserve the original 480-update artifact. Use the existing one-epoch checkpoint as the exposure reference on the frozen map diagnostics. For the deployment benchmark compare the updated candidate directly against ordinary untouched Qwen with thinking enabled for both. Treat the deployment comparison as an end-to-end comparison of deployed systems.

Specify one primary metric, a numerical stop rule, time and exposure caps before beginning. Charge all training and inference time. Inspect an intermediate milestone before spending the full budget. If the paired generation benchmark produces too few valid answers, first require a scorable baseline protocol in the next registration. A protocol failure needs to be resolved before claiming a language-reasoning gain. Leave architecture expansion, large training grids and sealed evaluation gated.

Compare the merits of regenerated presentations, genuinely fresh verified maps, and a small open benchmark for external validity. Keep one changed training variable per rapid iteration. The data guide records concrete sources and the split, license and contamination checks needed before including them. [Generated and open data](PHASE6_DATA.md#generated-and-open-data-for-the-next-review)
