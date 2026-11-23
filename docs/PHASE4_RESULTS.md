# Phase 4 development results

## The completed independent training replication

Both exits share the original 752,393,024 pretrained text parameters. A rank-eight adapter on layers 10–15 added 1,336,320 weights. The gated training arm added one scalar. Seed 109 was independent of the earlier seed 73, and both studies used V3 development for scoring. The Phase 3 sealed cohorts stayed reserved.

| Training and deployment | Targets | ID traced / 150 | OOD traced / 150 | Verified ID / OOD | Direct ID / OOD | Language NLL |
|---|---:|---:|---:|---:|---:|---:|
| Dense, K=1 | 250,211 | 63 | 45 | 57 / 40 | 15 / 5 | 3.0170 |
| Gated training, K=1 | 250,211 | 66 | 48 | 59 / 43 | 11 / 7 | 3.0193 |
| Same gated checkpoint, K=2 | 250,211 | 64 | 48 | 58 / 42 | 15 / 5 | 3.0238 |
| Time-matched dense, K=1 | 334,012 | 63 | 45 | 59 / 42 | 9 / 10 | 3.0195 |

Recurrently trained weights improved traced accuracy at ordinary deployment depth by two percentage points over both dense controls. Relative to equal-data dense, the gains came from six fixes and three breaks on ID, and four fixes and one break on OOD (exact paired p = 0.508 and 0.375). The time-matched dense control tied the K=1 gated model on verified ID solutions. This seed missed the registered three-point threshold and the criterion for verified gains on both splits. Direct-answer accuracy remained low. The equal-data comparison gave gated K=1 18 direct answers versus 20 for dense.

At inference, the second block application fixed one answer and broke three on ID. It fixed one and broke one on OOD. Neither fix had a verified derivation. Two of the four broken answers had been verified at K=1. Batched trace generation on the 150 ID items took 37.9 seconds at K=1 and 45.9 at K=2. On OOD it took 48.9 and 57.6 seconds. Use K=1 for this model. The sign of the equal-data training effect agrees with Phase 3; two development seeds and reuse of V3 items limit the claim.

The equal-data arms each processed 596,257 input tokens from 6,440 distinct examples in 1,610 updates. Dense used 757.5 seconds of update work and gated used 1,027.8. The time-matched control used 1,028.0 seconds for 2,160 updates and 8,640 examples. Peak MLX allocations were 4.42 GiB dense and 4.75 GiB gated. The M3 Pro has 18 GiB of unified memory.

## Joint supervision

Joint training completed 619 updates, 100,229 distinct supervised positions and 200,458 exit-target evaluations. It processed 235,064 input tokens from 2,476 examples at 177.5 supervised targets per update-second. The learned alpha moved from 0.1000 to 0.0961, and peak MLX allocation was 4.67 GiB. The first and second exit share their prefix and first pass. The objective weights were 0.25 and 0.75. On the same 300 problems, joint K=1 solved **98** with **90** verified solutions; K=2 solved **95** with **86** verified. Dense at the same budget solved 96 with 85 verified. Gated K=2-only training also solved 95 at K=2, with 82 verified. Joint supervision improved certificates relative to that K=2-only control but the extra pass still reduced both answer and certificate totals within the joint checkpoint. Its two fixed answers had zero verified derivations; it broke five previously correct answers, three of them verified.

## Decision and next experiment

We closed this pass after nine trained checkpoint/depth comparisons. Both registered promotion rules failed. Use the recurrently trained 250K weights at K=1 as the exploratory traced-task candidate. Direct answers, overall accuracy and seed uncertainty limit wider use. Keep the equal-data and equal-time dense adapters as references. Separate Phase 3 sealed files retain their original hashes.

**Next experiment:** run the bounded pointer-transition installation study on the independently verified eight-state maps. Supervise the correct intermediate state at each pass without putting gold states into the prompt. Compare the final state at the trained depth with the same model's first pass, an equally adapted dense terminal-state model and a frozen-backbone readout. Only continue to outcome-only training and unseen depths if the registered mechanism threshold is met. The code and data are available. No pointer model was trained in this pass.

We deferred wider adapters, weighted V4 training, additional seeds, the 25K generated-answer screen, fresh untouched-base generation, and stock/merged export parity checks. All three language training trajectories completed. Their 25K checkpoints and interim validation losses are preserved. The scored unmerged checkpoints reloaded for native generation. Historical source and result artifacts remain intact.

<!-- measured-tables -->

## Complete recorded outcomes

Exploratory seed 109 on M3 Pro, starting at `0a0bb9c`. Development evaluation used the complete 150+150 V3 sets at 100K and 250K, with paired item-level scoring. Trace generation uses 192 tokens and direct generation uses 24. Every trained comparison has a true single-pass exit. Phase 3 sealed cohorts were excluded. [Registration](PHASE4_PLAN.md)

| Arm · targets · depth | ID traced | OOD traced | ID verified | OOD verified | ID direct | OOD direct | Language NLL | Interactive s/request | Batched traced requests/s |
|---|---|---|---|---|---|---|---|---|---|
| dense-250000-k1 | 63/150 | 45/150 | 57 | 40 | 15 | 5 | 3.0170 | 1.236 | 3.21 |
| gated-250000-k1 | 66/150 | 48/150 | 59 | 43 | 11 | 7 | 3.0193 | 1.186 | 3.45 |
| gated-250000-k2 | 64/150 | 48/150 | 58 | 42 | 15 | 5 | 3.0238 | 1.420 | 2.90 |
| time-250000-k1 | 63/150 | 45/150 | 59 | 42 | 9 | 10 | 3.0195 | 1.191 | 3.23 |
| dense-100000-k1 | 52/150 | 44/150 | 47 | 38 | 9 | 4 | 3.0107 | 1.198 | 3.33 |
| gated-100000-k1 | 57/150 | 43/150 | 48 | 38 | 10 | 9 | 3.0097 | 1.194 | 3.35 |
| gated-100000-k2 | 51/150 | 44/150 | 45 | 37 | 10 | 7 | 3.0135 | 1.463 | 2.90 |
| joint-100000-k1 | 53/150 | 45/150 | 50 | 40 | 8 | 5 | 3.0114 | 1.161 | 3.54 |
| joint-100000-k2 | 52/150 | 43/150 | 48 | 38 | 9 | 6 | 3.0143 | 1.460 | 2.95 |

## Paired primary outcomes

| Reference / challenger | Split | Fixes | Breaks | Verified Δ | Δ pp | Paired item 95% | Exact p |
|---|---|---|---|---|---|---|---|
| dense-250000-k1/gated-250000-k1 | development/trace | 6 | 3 | 2 | +2.0 | [-2.0, 6.0] | 0.508 |
| dense-250000-k1/gated-250000-k1 | development_ood/trace | 4 | 1 | 3 | +2.0 | [-0.7, 5.3] | 0.375 |
| time-250000-k1/gated-250000-k1 | development/trace | 6 | 3 | 0 | +2.0 | [-2.0, 6.0] | 0.508 |
| time-250000-k1/gated-250000-k1 | development_ood/trace | 4 | 1 | 1 | +2.0 | [-0.7, 5.3] | 0.375 |
| gated-250000-k1/gated-250000-k2 | development/trace | 1 | 3 | -1 | -1.3 | [-4.0, 1.3] | 0.625 |
| gated-250000-k1/gated-250000-k2 | development_ood/trace | 1 | 1 | -1 | +0.0 | [-2.0, 2.0] | 1.000 |
| dense-100000-k1/joint-100000-k2 | development/trace | 6 | 6 | 1 | +0.0 | [-4.7, 4.7] | 1.000 |
| dense-100000-k1/joint-100000-k2 | development_ood/trace | 4 | 5 | 0 | -0.7 | [-4.7, 3.3] | 1.000 |
| gated-100000-k2/joint-100000-k2 | development/trace | 6 | 5 | 3 | +0.7 | [-3.3, 5.3] | 1.000 |
| gated-100000-k2/joint-100000-k2 | development_ood/trace | 3 | 4 | 1 | -0.7 | [-4.0, 2.7] | 1.000 |
| joint-100000-k1/joint-100000-k2 | development/trace | 2 | 3 | -2 | -0.7 | [-3.3, 2.0] | 1.000 |
| joint-100000-k1/joint-100000-k2 | development_ood/trace | 0 | 2 | -2 | -1.3 | [-3.3, 0.0] | 0.500 |
| gated-100000-k1/gated-100000-k2 | development/trace | 0 | 6 | -3 | -4.0 | [-7.3, -1.3] | 0.031 |
| gated-100000-k1/gated-100000-k2 | development_ood/trace | 2 | 1 | -1 | +0.7 | [-1.3, 3.3] | 1.000 |


## Registered decisions

Track A eligibility: False. Track B eligibility: False. Confidence intervals above resample items for this one seed. [Training cost and memory](PHASE4_OPTIMIZATIONS.md) · [Data and verification](PHASE4_DATA.md)
