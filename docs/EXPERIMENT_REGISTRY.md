# Experiment registry

| Study | Cohort | Protocol and outcome |
|---|---|---|
| Original pilot and extended | Matched rank-8 adapters on the original procedural set | Dense scored 61/110 after 200 updates and 65/110 after 1,000. Direct recurrence scored 50/110 and 59/110. [Phase 1 results](RESULTS.md) |
| Phase 2 | V2, 84 updates per data-matched confirmatory arm, seeds 43, 53 and 67 | Traced-generation ID accuracy averaged 45.9% dense and 40.0% loop. [Phase 2 results](PHASE2_RESULTS.md) |
| Phase 3 gate LR | Seed 73, 8 updates, 1,360 identical targets | Both groups moved. Development trace NLL selected LR 0.001. |
| Phase 3 screen | 25,026 targets, 148 updates, 592 examples | Dense 28/60, direct 23/60, fixed 27/60, learned 30/60. The learned checkpoint at K=1 scored 29/60. |
| Phase 3 main | 100,068 targets, 629 updates, 2,516 examples per arm | Dense, direct and selected learned interpolation with true bypass evaluation. [Main report](PHASE3_RESULTS.md) |
| Phase 3 extension | 250,031 targets, 1,605 updates, 6,420 distinct examples per arm | Dense 58/150 ID and 41/150 OOD. Learned K=2 and its true K=1 bypass each scored 61/150 ID and 47/150 OOD. Confirmation criterion missed. [Promotion record](measurements/phase3/promotion.json) |

## Phase 4 experiments

The seed-109 pass closed after the primary 100K and 250K trained comparisons. Nine model/depth combinations were evaluated on 150 ID items, 150 OOD items and a 120-passage retention cohort. Recovery checkpoints include the 25K milestone. The publication includes four compact adapters and the scored records. The remaining pointer, boundary, weighting and export experiments are available as separate commands. [Findings](PHASE4_RESULTS.md) · [Archived measurements](measurements/phase4)

The [Phase 4 registration](PHASE4_PLAN.md) fixes the seed, source revision, data, logical batch size, milestones and promotion thresholds. The [study manifest](../experiments/phase4.json) distinguishes the 250K equal-deployment-cost comparison, the 25K/100K dual-exit comparison and the independent categorical pointer experiment. Seed 109 is independent of Phase 3 seed 73. Each training arm has its own immutable execution-policy YAML. The data-matched dense and time-matched dense models are two points on a single training trajectory.

From the project root, after the locked environment has been installed:

```sh
source .venv/bin/activate
python -m loopqwen download
python -m loopqwen data --preset v2
python -m loopqwen data --preset v3
python -m loopqwen.data.retention3
scripts/run_phase4.sh efficiency audit
scripts/run_phase4.sh replicate dual-exit evaluate --seed 109
python -m loopqwen data --preset v4
python scripts/phase4_data_audit.py
python scripts/phase4_exposure.py
python scripts/phase4_report.py --closeout
```

The `evaluate` command also scores the 25K checkpoints and untouched base when allowed to finish. The closeout archive selects only the completed primary comparisons reported here. The pointer diagnostic is a separate `scripts/run_phase4.sh latent --seed 109` stage. Export and expanded regression checks use `scripts/run_phase4.sh verify --seed 109`.

After the primary comparison, the registered capacity and weighting screens use `scripts/run_phase4.sh boundary v4 --seed 109`. The V4 stage creates the separate text dataset. The latent stage creates its disjoint permutations independently. Audit target counts and regenerate the research archive with `python scripts/phase4_exposure.py` followed by `python scripts/phase4_report.py`. Rendering the figure requires `uv sync --locked --group figures`.

The native trainer marks recovery generations under `checkpoints/p4_109_<arm>/`. The full categorical checkpoints are under `checkpoints/p4_state_109_<arm>/`. Phase 4 writes development scores under identities that include the evaluated weights and ordered token streams. A published archive has a checksum index, and subsequent publications resolve to a separate local reproduction directory. The computations only read development items from the earlier phases.

## Phase 3 reproduction

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

The published Phase 4 equal-data dense adapter is the evaluated update-1,610 snapshot of the time-matched dense trajectory. Its original metadata records the enclosing run as active at capture. That trajectory subsequently completed at update 2,160. The evaluated 1,610-update weights have their own artifact digest. Both milestones are listed in the [archived training record](measurements/phase4/training.json).

## Phase 5 pointer installation

Seed 109, pinned Qwen Base, original rank-eight span. The registered arms are the frozen probe, dense middle, dense full and direct recurrent. Each has a common 480-update schedule. Model selection uses development maps. The 384-map confirmation cohort has a separate immutable identity. [Registration](../experiments/phase5.json) · [Cohorts and priors](PHASE5_DATA.md) · [Promotion rules](PHASE5_PLAN.md)

Run `scripts/run_phase5.sh audit data profile smoke install --seed 109` for the primary installation. Run `scripts/run_phase5.sh evaluate diagnostics followup latency --seed 109` for the registered follow-up and timings. The installation and persistence stages are complete. The archive contains both compact checkpoints and the per-map predictions. Confirmation stayed sealed after the shuffled-edge result missed the promotion threshold. [Archive](measurements/phase5) · [Decision](measurements/phase5/decision.json) [Current results](PHASE5_RESULTS.md)

The installation threshold triggered the dense multi-state control, recurrent terminal-only control, forced depths four through six and 80 terminal-only continuation updates. The display-order test retained a separate promotion gate. The donor-diversity sensitivity and bounded zero-shot verbal checks have dated, separate registrations. [Supplemental donors](../experiments/phase5_donors.json) · [Transfer protocol](../experiments/phase5_transfer.json)

To reproduce the supplemental checks, run `scripts/run_phase5.sh donors transfer --seed 109` after the qualified follow-up. Archive a reproduction into a new directory with `.venv/bin/python scripts/phase5_archive.py --destination outputs/phase5/reproduction`. Render its plot with `.venv/bin/python scripts/phase5_figure.py --archive outputs/phase5/reproduction --output outputs/phase5/reproduction.png`.

## Phase 6 presentation-order generalization

The matched seed-109 study changes edge presentation while preserving the map schedule, pretrained weights and recurrent graph. The independent three-head control receives all intermediate labels. The primary measurements use 384 development maps, 12 presentations per map and map-clustered comparisons. [Preregistration](PHASE6_PLAN.md) · [Cohorts](PHASE6_DATA.md) · [Architecture](PHASE6_ARCHITECTURE.md)

The five matched installations and their twelve-order evaluations are complete. The randomized arm gained 47.04 points over canonical training on shuffled third-hop accuracy. Canonical third hops missed the 95% floor, closing all dependent training and confirmation stages. A separate, inference-only, thinking-enabled comparison uses the saved model and the untouched base. [Results](PHASE6_RESULTS.md) · [Pair registration](../experiments/phase6_reasoning_pair.json) · [Reviewer handoff](PHASE6_REVIEW.md)

## Phase 7 additional exposure

Seed 109, exact R480 optimizer continuation, shared rank-eight block, fresh randomized displays of original maps. System pressure closed training at R624 after 144 updates. The full frozen order cohort qualified this endpoint for causal diagnosis. Base-only language calibration has its own registration and independent gate. [Registration](../experiments/phase7.json) · [Results](PHASE7_RESULTS.md)
