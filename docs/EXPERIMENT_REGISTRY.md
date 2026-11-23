# Experiment registry

| Study | Cohort | Protocol and outcome |
|---|---|---|
| Original pilot and extended | Matched rank-8 adapters on the original procedural set | Dense scored 61/110 after 200 updates and 65/110 after 1,000. Direct recurrence scored 50/110 and 59/110. [Phase 1 results](RESULTS.md) |
| Phase 2 | V2, 84 updates per data-matched confirmatory arm, seeds 43, 53 and 67 | Traced-generation ID accuracy averaged 45.9% dense and 40.0% loop. [Phase 2 results](PHASE2_RESULTS.md) |
| Phase 3 gate LR | Seed 73, 8 updates, 1,360 identical targets | Both groups moved. Development trace NLL selected LR 0.001. |
| Phase 3 screen | 25,026 targets, 148 updates, 592 examples | Dense 28/60, direct 23/60, fixed 27/60, learned 30/60. The learned checkpoint at K=1 scored 29/60. |
| Phase 3 main | 100,068 targets, 629 updates, 2,516 examples per arm | Dense, direct and selected learned interpolation with true bypass evaluation. [Main report](PHASE3_RESULTS.md) |

The Phase 3 registration is in `experiments/phase3_dev.json`. The training schedule digest covers problem IDs, response protocol, prefix length, complete token IDs and batch boundaries. Source files, model revision, dependency versions, AdamW layout and sampler schema accompany strict checkpoints. The latest recovery generation and each crossed milestone are named by separate atomic pointers. A completed milestone can be evaluated while the 250K maximum remains available for extension.

From the repository root with the locked environment installed and the pinned checkpoint cached, use the following commands. Each study stage acquires the checkout study lock. Each model worker acquires a per-user kernel GPU lease. Re-running a completed stage uses its saved progress.

```sh
source .venv/bin/activate
python -m loopqwen data --preset v2
python -m loopqwen data --preset v3
python -m loopqwen.data.retention3
scripts/run_phase3.sh smoke
scripts/run_phase3.sh screening
scripts/run_phase3.sh main
scripts/run_phase3.sh promote
python scripts/phase3_exposure.py
python scripts/phase3_post.py
python -m loopqwen compare --experiment phase3_dev
python scripts/phase3_archive.py
uv sync --locked --group figures
python scripts/phase3_report.py
```

Full update states live under `checkpoints/p3_<arm>/`. The training history stores measured timings and model telemetry at every update. The shared planned sample schedule and resolved run YAMLs live under `outputs/phase3/`. Selection uses the first 60 balanced V2 validation problems with a 192-token cap. The five-family V3 development and development-OOD splits provide additional generalization evidence. The sealed 300-item ID and OOD splits are reserved for a frozen confirmation.

To resume a milestone manually with unchanged optimization settings, use the same resolved YAML and an absolute update number. For an intentional fork use `python -m loopqwen fork --checkpoint <generation> --config <new-config> --mode extend` or `--mode weights`. The parent is immutable. Extension preserves the schedule position, optimizer moments and data cursor. A weights-only fork begins a distinct adaptation run.

The archived run register, source hashes, artifact checksums and compressed per-example evaluations are in `docs/measurements/phase3/`. Local reruns after publication use a separate reproduction archive under `outputs/phase3/reproduction/<main-result-digest-prefix>/`. To render that reproduction, pass its directory to `phase3_report.py --archive` and an independent directory to `--destination`. The committed tables remain the original publication. Model cards under `adapters/phase3/` describe the unmerged inference artifacts.
