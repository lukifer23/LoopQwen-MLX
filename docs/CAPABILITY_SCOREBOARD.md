# Capability scoreboard

| Capability / cohort | Operating point | Accuracy and cost |
|---|---|---|
| Three-hop categorical, 384 unseen maps × eight shuffles | R624, K=3 | 3,067/3,072. Batch-one K=3 median 48.8 ms. Cross-order patches 61–62/64 versus zero unpatched. |
| Six-hop categorical, seed-719 development | RSI static time control, K=6 | 255/256 complete trajectories after 496 additional presentations and 65.24 s selection/update work. |
| Verbal next digit, held-out state-machine wording | R624 plus 60 transfer updates, K=3 | 187/192, versus 39/192 before transfer. 480 new maps, 480 vocabulary and 480 state targets, 64.36 update s. |
| Verified verbal trajectories | Same transfer checkpoint, sampled room protocol | 0/32 verified, 17 correct terminal numbers, 100 generated tokens. |
| General language, 120 passages | Untouched Base K=1 | NLL 2.9827 over 9,328 targets. Transfer K=1 3.0329 and K=3 3.4881. |
| Thinking chat, demonstrated calibration | Untouched Base K=1 | Five valid and correct answers of eight. Gate missed. |
| Outer curriculum, seed 719 | Adaptive versus static | +4.69 points sixth-hop accuracy at equal updates; −2.73 at matched time. Seed 859 stopped on retention. |

All checkpoints use 752,393,024 unique backbone parameters. Categorical adaptations add 1,336,320 LoRA and 8,192 readout weights. The six-hop checkpoint has no measured verbal or prose score. [Exact lineage, paired scores and costs](PHASE7_RESULTS.md) · [Curriculum registry and intervals](RSI_RESULTS.md)

### Complete responses on fresh maps

| Checkpoint, K=3 | Correct final | Verified path | Terminated | Generated tokens | Median latency | Categorical hop 3 | Prose NLL K1 / K3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Released verbal parent | 45/64 | 0/64 | 64/64 | 198 | 130 ms | 95/96 | 3.033 / 3.488 |
| Block trace child | 64/64 | 64/64 | 64/64 | 576 | 273 ms | 81/96 | 3.039 / 3.487 |

After 32 updates, 256 fresh maps and 2,304 supervised vocabulary targets, the child passed the language criteria and missed categorical retention. Update work was 60.17 s with a 2.68 GiB peak. Both models received identical explicit three-hop questions. The all-correct development result has a 95% Wilson interval of 94.3–100% across maps and one training lineage. [Paired evidence and lineage](../research/results/trace-bootstrap-01)

| Additional measurement | Result |
|---|---|
| Frozen recurrent block with 1,803,776 fresh suffix LoRA parameters | 1/64 verified, 8/64 terminal, 52/64 completed. Categorical third hop 95/96. K=1 prose NLL 3.234. 36.90 update seconds for the same 2,304 targets. |
| Untouched Base thinking calibration, greedy first-turn stop | 3/8 valid and correct, 4,821 tokens, 72.43 seconds. Six completions required. |
| GSM8K raw numeric completion, 32 reserved publisher training items | Base 2/32, verbal parent K=3 1/32. Valid answers 26 versus 30. |
| ARC-Easy validation, 32 four-choice items | Both 18/32 with length-normalized choice likelihood, both 21/32 with summed likelihood. |

### Same requested trace, fresh 96-map cohort

| Checkpoint | Verified trajectories | Terminal correct | Median request | Generated tokens |
|---|---:|---:|---:|---:|
| Untouched Base K=1 | 0/96 | 5/96 | 159 ms | 884 |
| Trained at K=1, deployed K=1 | 96/96 | 96/96 | 198 ms | 864 |
| Trained at K=3, deployed K=1 | 32/96 | 36/96 | 209 ms | 864 |
| Trained at K=3, deployed K=3 | 95/96 | 95/96 | 447 ms | 864 |

Both trained models started from the identical Phase 7 verbal parent and used the same 256 examples, 2,304 targets and 32 updates. Dense training took 36.41 synchronized seconds and recurrent training took 60.17 seconds. All requests terminated. The same-weight depth gain was 63 wins and zero losses. Against trained dense, K=3 had zero wins and one loss, a paired 95% interval of −3.13 to 0 percentage points. One training lineage, one wording family and near-ceiling dense performance limit attribution. [Paired outputs and dense weights](../research/results/trace-depth-01)

### Preserving state prediction while writing complete paths

The joint K=3 fork generated **64/64** complete three-hop responses. On the categorical retention sample its exit counts were **96, 96, 95 out of 96**, matching the verbal parent. Full V3 language NLL was **3.0369 at K=1 and 3.4868 at K=3**. The joint fork adds three symbolic state labels per map to the same 256 training maps and 2,304 vocabulary targets. It used 768 state labels, 32 updates, 114.27 update seconds, 181.77 stage seconds and 2.68 GiB peak allocation. Each term gets half the objective after normalization over its own targets. This is the combined room-trace and categorical operating point. The trained dense model remains the fastest high-accuracy room-trace release measured on the fresh attribution cohort. [Joint result, paired outputs and adapter](../research/results/joint-trace-01)

### Zero-shot structural transfer

| Six-hop checkpoint, randomized symbolic prompt | Hop 1 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Entire six-hop path |
|---|---:|---:|---:|---:|---:|---:|---:|
| 192 unseen restricted permutations | 192 | 192 | 192 | 191 | 191 | 190 | 190 |
| 192 unseen unrestricted successor maps | 192 | 84 | 30 | 23 | 10 | 12 | 6 |

At the sixth hop of the unrestricted cohort, 189/192 gold states had already been visited. Only 12 of those 189 predictions were correct. Just four of 70 gold stationary states were predicted correctly. The model retrieved the first successor on every map in both cohorts, then lost accuracy on graphs with repeated destinations or shorter cycles. The sample changes both properties together. Its results leave their individual contributions unresolved. [Recorded probabilities, strata and verification](../research/results/functional-zero-01)
