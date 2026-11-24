# Phase 7 exposure and completed reasoning answers

The additional exposure improved the existing seed-109 checkpoint. R624 correctly predicted **384/384** canonical third hops and **3,067/3,072** third hops on eight frozen shuffles, compared with 360 and 2,885 at R480. The shuffled difference is **+5.92 percentage points**. All first and second hops were correct under all twelve presentations. The five shuffled third-hop errors were also the only incomplete shuffled trajectories. The worst third-hop condition was rotation by four (380/384). These are repeated displays of 384 development maps and a single training seed.

| Measurement | R480 | R624 |
|---|---:|---:|
| Absolute optimizer updates | 480 | 624 |
| Cumulative training map exposures | 3,840 | 4,992 |
| Distinct training maps | 3,840 | 3,840 |
| Canonical complete paths | 359/384 | 384/384 |
| Complete shuffled paths | 2,873/3,072 | 3,067/3,072 |
| Random third-hop accuracy | 93.91% | 99.84% |

The child restored the complete AdamW generation at update 480. Its next update used epoch one, batch zero. It received 1,152 first-hop, 768 second-hop and 384 third-hop labels in 144 updates (2,304 labels total). Updates took 69.18 synchronized seconds, or 16.65 maps and 33.31 labels per second. Timed training-loop wall time totaled 69.52 seconds across three worker attempts, excluding model loading, checkpoint restore and between-attempt waits. MLX peak was 3.91 GiB and the lowest recorded available system memory after an update was 3.46 GiB. macOS pressure warnings stopped workers at 504, 552 and 624. Each saved a valid checkpoint. After the third warning the run closed at 624. The 720 and 960 milestones were never reached. No execution settings were relaxed. The 300-second update cap was not the limiting factor.

The four numerical installation criteria passed at the safety endpoint. This supports additional exposure on the existing maps. Fresh map diversity and the full two-epoch effect remain unmeasured. The historical full dense comparator saw only one epoch. Phase 6 retains its original failed decision. [Prospective protocol](PHASE7_PLAN.md)
