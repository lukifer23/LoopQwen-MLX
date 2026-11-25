# LoopQwen research objective

Improve verified ordinary language reasoning using shared recurrent computation on the pinned Qwen3.5-0.8B-Base and the 18 GiB M3 Pro. Promotion requires measured accuracy, completion, retention and actual cost against appropriate controls. The deployment target includes at least 131,072 input tokens plus decoding. Measure full-history capacity and scored retrieval separately. Keep separate categorical, verbal and general-language recommendations.

Each active session reads the [research state](research/state.json), registers a bounded comparison before measurement, archives paired predictions and updates the [capability scoreboard](docs/CAPABILITY_SCOREBOARD.md). Limit each session to 120 wall minutes, 75 native minutes and eight primary iterations. Reassess after three consecutive unsuccessful iterations. Run one leased Metal worker, stop on system memory warnings and preserve immutable training lineage. New remote publication requires explicit authorization.

The [current research decision](docs/NEXT_EXPERIMENTS.md) records study decisions and the reasons for advancing or closing each branch. The [chronological ledger](docs/RESEARCH_LOG.md) records completed and interrupted comparisons.
