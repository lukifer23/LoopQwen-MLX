# LoopQwen research objective

Improve verified ordinary language reasoning using shared recurrent computation on the pinned Qwen3.5-0.8B-Base and the 18 GiB M3 Pro. Promotion requires measured accuracy, completion, retention and actual cost against appropriate controls. Keep separate categorical, verbal and general-language recommendations.

Each active session reads the [research state](research/state.json), registers a bounded comparison before measurement, archives paired predictions and updates the [capability scoreboard](docs/CAPABILITY_SCOREBOARD.md). Limit each session to 120 wall minutes, 75 native minutes and eight primary iterations. Reassess after three consecutive unsuccessful iterations. Run one leased Metal worker, stop on system memory warnings and preserve immutable training lineage. New remote publication requires explicit authorization.

The [current research decision](docs/NEXT_EXPERIMENTS.md) records the closed execution window, the preserved optimizer cursor and the conditions for a future authorized run. The [chronological ledger](docs/RESEARCH_LOG.md) records completed and interrupted comparisons.
