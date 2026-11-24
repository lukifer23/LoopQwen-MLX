# Phase 6 categorical execution

All matched arms use native BF16 Qwen and the validated Gated DeltaNet backward, FP32 adapters, a logical batch of eight, a padded-position cap of 384, no activation checkpointing and one clipped AdamW update per logical batch. The schedule, target count and executed layer positions are recorded separately. Input maps have a fixed eight-class categorical vocabulary at the readout.

The trainer builds the shuffled map index once per epoch. On the exact seed-109 schedule a five-repeat CPU check reduced median scheduling time for all 480 batches from 0.231 seconds to 0.00062 seconds. All 480 sets of IDs and depths agreed. This difference accounts for a small fraction of a training run.

For a single prompt the recurrent evaluator computes the prefix once and keeps one carried representation through all requested exits. Collecting six readouts requires ten prefix layers and six shared-block applications. The independent-depth reference evaluates the prefix six times and the block 21 times. Dense terminal prompts include a hop count, so each requested depth receives its own forward. Independent depth-request latency is timed separately at batches one and eight.

The prefix reuse benchmark measures two consumers of identical token streams. It includes prefix construction, four timed eight-update sequences in counterbalanced order, cached tensor bytes, disk write/read costs and a gradient comparison on one real batch. The matched primary arms retain their registered execution policy. Training throughput uses synchronized update time, excluding initialization, checkpoint writes and evaluation.

Reproduce the CPU-only schedule comparison with `.venv/bin/python scripts/phase6_schedule_profile.py --output outputs/phase6/schedule-repeat.json`. Run `scripts/run_phase6.sh benchmark --arms randomized` for the measured native paths.

## Native measurements

The six-exit benchmark used eight real shuffled 42-token prompts at the installed randomized checkpoint. After one warmup per path, three synchronized counterbalanced timings per implementation gave a median of **994.1 ms** for six independent depths (993.1–1,005.1 ms) and **254.7 ms** for one shared unroll (251.2–262.7 ms), a **3.90×** speedup for the full trajectory. All 48 argmax states agreed and the maximum probability difference was zero. Peak allocations were 2.06 and 2.03 GiB. This speedup applies to retrieving all six exits. A single requested hop still incurs its actual measured forward cost.

The frozen-prefix benchmark used 64 real randomized training prompts in eight batch-eight updates at K=3. Materializing 5.25 MiB of BF16 prefix outputs took 0.384 s. Writing and loading the small cache took 1.49 and 0.71 ms. Eight updates took 3.69–3.73 s with live prefixes and 3.27 s with the cached prefixes, with 3.95 GiB peak in this shared-residency harness. The inspected cached and recomputed prefix tensors agreed exactly, but materializing them changed the measured loss (0.081502 versus 0.081555) and adapter/head gradients (7.87% relative norm difference, versus 0.89% for an uncached repeat). Both exceeded the registered numerical tolerance on gradients. The cause of the extra discrepancy is unresolved. Prefix caching stays disabled in production and all reported arms.

| Arm | Updates | Synchronized update s | Maps/s | Labels/s | Peak MLX GiB | Lowest sampled available GiB |
|---|---:|---:|---:|---:|---:|---:|
| Canonical recurrent | 480 | 175.1 | 21.9 | 42.8 | 3.91 | 4.12 |
| Randomized recurrent | 480 | 163.0 | 23.6 | 45.9 | 3.91 | 4.90 |
| Randomized dense middle | 480 | 102.6 | 37.4 | 37.4 | 3.40 | 5.69 |
| Randomized dense full | 480 | 189.9 | 20.2 | 20.2 | 4.71 | 5.27 |
| Randomized dense three-head | 480 | 92.8 | 41.4 | 80.7 | 3.18 | 5.41 |

The primary study processed 19,200 map exposures and 30,144 categorical labels across the five arms. Recurrent, full dense, dense middle and dense three-head executed 3,499,776, 4,423,680, 2,949,120 and 2,580,480 decoder-layer positions per arm, respectively. All runs stayed within the 6 GiB MLX peak limit. Synchronized update times exclude checkpointing and evaluation. For example, the full dense final twelve-order evaluation took 223.9 seconds, while its training updates took 189.9 seconds. This evaluation cost is part of the rationale for the two-model rapid-study policy.

## Reproduction details

The starting checkout was `3e0323cc87f1112ba36cf809fdbfe57d02e530d0`. Training ran serially while evaluation and documentation commits were added on main. The checkpoint metadata records the Git revision at the start of each arm and the hashes of training-relevant files, the pinned backbone/tokenizer assets, the resolved config and optimizer identity. The five arm revisions were `fdd2e16` (canonical), `675450f` (randomized), `ee3ab5f` (middle), `45ce779` (full) and `58dbfc2` (three-head). The archiver checks all five logical schedules against each other and the published seed-109 Phase 5 schedule, plus all randomized order digests across arms.
