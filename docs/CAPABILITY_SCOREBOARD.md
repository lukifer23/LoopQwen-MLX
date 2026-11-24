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
