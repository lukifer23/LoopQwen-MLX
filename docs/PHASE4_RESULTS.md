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

Joint training completed 619 updates, 100,229 distinct supervised positions and 200,458 exit-target evaluations. It processed 235,064 input tokens from 2,476 examples at 177.5 supervised targets per update-second. The learned alpha moved from 0.1000 to 0.0961, and peak MLX allocation was 4.67 GiB. The first and second exit share their prefix and first pass. The objective weights were 0.25 and 0.75. Development generation and the independent categorical mechanism study are recorded separately as their evaluations finish.
