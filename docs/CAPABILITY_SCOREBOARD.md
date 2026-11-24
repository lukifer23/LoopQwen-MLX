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

### Verified self-training on document references

| Generation, K=3 | Verified paths / 64 | Correct terminal / 64 | Categorical hop 3 / 96 | Generation-specific update s | Candidate rollout s |
|---|---:|---:|---:|---:|---:|
| Joint parent | 47 | 48 | 95 | — | — |
| Static gold 1 | 61 | 61 | 95 | 12.05 | 0 |
| Model-selected 1 | 50 | 51 | 95 | 11.77 | 25.63 |
| Static gold 2 | 60 | 60 | 93 | 11.83 | 0 |
| Model-selected 2 | 57 | 58 | 94 | 11.74 | 22.09 |

Each branch used eight updates in total, 32 distinct selected maps, 64 map exposures, 576 response targets and 192 auxiliary state labels. The learner generated 100 verified candidate responses across 128 attempts, on 25/32 and 27/32 distinct candidate maps. Generation two used its actual predecessor. Across both rounds, every selected response exactly matched the gold response tokens for its selected map. The two arms used different map selections. Static training had four unique wins and self-training one on the final cohort. The paired self-minus-gold difference was −4.69 points (95% map bootstrap −10.94 to +1.56). The gold branch lost two categorical items from 95/96, a 2.08-point drop against the registered two-point safeguard. Both rounds were measured. Promotion failed. [Generations, sampled responses and training lineages](../research/results/verified-generations-01)

The external screen's aggregate tie on ARC-Easy includes four unique wins for each system. On GSM8K, the candidate fixed one Base error and lost two Base successes. None of the fresh evaluation items were used to train a successor.

### Final document after three links, with just the answer generated

| Training and deployment depth | Correct completed answers | Median request | Update time | Retained symbolic third hops | V3 NLL at K=1 / K=3 |
|---|---:|---:|---:|---:|---:|
| K=1 | 61/128 | 107 ms | 40.31 s | 93/96 | 3.050 / 3.565 |
| K=3 | 44/128 | 150 ms | 67.80 s | 92/96 | 3.037 / 3.475 |

Both arms started from the joint verbal parent, which scored 95/96 on the retention cohort. Both trained on the same 256 arbitrary successor maps and 768 response tokens, then completed all 128 fresh requests (384 generated tokens each). The recurrent arm fixed four dense errors and lost 21 correct dense answers. Its paired accuracy difference was **−13.28 points** (95% map bootstrap **−20.31 to −6.25**). It missed both the five-point advantage and the two-item retention safeguard. Allowing a single final number leaves the dense advantage intact on this distribution. [Per-map outputs and both weights](../research/results/functional-terminal-01)

### Learning from training-time errors on arbitrary successor maps

| One six-hop training round | Sixth hop / 128 | Complete six-hop paths / 128 | Prior permutation third hops / 96 | Synchronized update s |
|---|---:|---:|---:|---:|
| Released six-hop parent | 4 | 1 | 96 | — |
| Static replay | 53 | 18 | 12 | 21.03 |
| Online replay | 33 | 5 | 11 | 21.89 |

Both trained branches saw 96 new maps and 32 repeats, with 768 state labels in 16 updates. The online selector used predictions already computed during training; selection took 0.37 ms, versus 0.35 ms for static. The two policies selected 17 of the same 32 replay maps. Cheaper feedback removed the diagnostic inference cost, but static learned more and both lost most former third-hop predictions. The registered retention stop closed round two. The 84-position physical cap peaked at 2.39 GiB over 68.69 stage seconds. A separately preserved 384-position attempt incurred 17.38 extra seconds before its memory stop. [Independent rescore and replay lineage](../research/results/functional-replay-cap84)

### Thinking-enabled GSM8K generation after equation supervision

| Same 32 fresh publisher training questions, cap 160 | Scorable final | Correct / all | Generated tokens | Generation time |
|---|---:|---:|---:|---:|
| Untouched Base K=1 | 0/32 | 0/32 | 5,120 | 79.93 s |
| Equation-trained K=3 at update 16 | 26/32 | 1/32 | 2,063 | 54.32 s |

The candidate trained on 128 of 256 scheduled questions, with 5,145 response tokens and 53.90 synchronized update seconds. First-turn boundaries and strict numerical scoring apply to both systems. The sole net gain gives a paired 95% question-level bootstrap interval of 0 to +9.38 percentage points. The registered criterion required two net gains. Memory pressure stopped optimization at update 16 and interrupted subsequent retention assessment after both generation cohorts were complete. Post-training categorical retention and prose NLL are unavailable.

The registered single-equality format counted eight arithmetically consistent candidate responses, all with wrong final answers. One additional response correctly computed the school enrollment as `168/0.2=168*5=840`. The chained equality falls outside the registered secondary format. A supplementary exact-rational audit accepted this chain, found false equalities in 13 other completed responses, and classified four as unsupported. Arithmetic consistency alone cannot verify that the equations represent the word problem. [All prompts, tokens and checkpoint](../research/results/gsm-equations-01) · [Supplementary audit](../research/sessions/2026-11-24-resumed/gsm-arithmetic-audit.json)
