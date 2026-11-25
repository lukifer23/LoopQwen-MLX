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

**Did the sixth pass produce a better answer than keeping the first prediction?** All three checkpoints predicted the first state on all 128 arbitrary maps. Repeating that prediction for every exit would score 37/128 final states and 30/128 complete paths because some maps reach a fixed point immediately. Static replay reached **53/128 final states** but **18/128 complete paths**. Compared map by map, it gained five complete paths and lost 17. Online replay reached **33/128 final states** and **5/128 complete paths**. On the 78 maps where the sixth step required movement, static replay predicted 26 terminal states correctly and online replay predicted 16. Neither produced a complete path on any of those 78. This exploratory comparison uses each checkpoint’s archived predictions. It introduces no additional model execution or latency measurement. [Reproduce the paired rescore](../scripts/audit_replay_paths.py)

### Thinking-enabled GSM8K generation after equation supervision

| Same 32 fresh publisher training questions, cap 160 | Scorable final | Correct / all | Generated tokens | Generation time |
|---|---:|---:|---:|---:|
| Untouched Base K=1 | 0/32 | 0/32 | 5,120 | 79.93 s |
| Equation-trained K=3 at update 16 | 26/32 | 1/32 | 2,063 | 54.32 s |

The candidate trained on 128 of 256 scheduled questions, with 5,145 response tokens and 53.90 synchronized update seconds. First-turn boundaries and strict numerical scoring apply to both systems. The sole net gain gives a paired 95% question-level bootstrap interval of 0 to +9.38 percentage points. The registered criterion required two net gains. Memory pressure stopped optimization at update 16 and interrupted subsequent retention assessment after both generation cohorts were complete. Post-training categorical retention and prose NLL are unavailable.

The registered single-equality format counted eight arithmetically consistent candidate responses, all with wrong final answers. One additional response correctly computed the school enrollment as `168/0.2=168*5=840`. The chained equality falls outside the registered secondary format. A supplementary exact-rational audit accepted this chain, found false equalities in 13 other completed responses, and classified four as unsupported. Arithmetic consistency alone cannot verify that the equations represent the word problem. [All prompts, tokens and checkpoint](../research/results/gsm-equations-01) · [Supplementary audit](../research/sessions/2026-11-24-resumed/gsm-arithmetic-audit.json)

An exploratory rescore divided the 128 terminal questions by their verified three-hop path. On paths with four distinct states, the trained K=1 and K=3 models scored **9/52 and 4/52**. When the last step stayed at the same state they scored **37/42 and 25/42**. On the other paths with revisits they each scored **15/34**. Most of the aggregate 17-answer gap came from stationary final steps. These groups describe where the predictions differed. Identifying the internal cause would require a separate experiment. [Paired topology counts](../research/sessions/2026-11-24-resumed/terminal-topology.json)

Two simple rules put the terminal-answer scores in context. Following just the first reference gives **51/128** correct third-hop answers. Choosing the most frequently cited document and breaking ties uniformly gives **48.08/128 in expectation**, even without reading the starting document. The trained K=1 model gained 23 answers and lost 13 relative to the one-edge rule, reaching 61/128. K=3 gained 14 and lost 21, reaching 44/128. K=1 chose a most-cited document on 103/128 requests and K=3 on 97/128. On the 57 maps where the true answer was outside that set, they scored **9/57** and **6/57**. This post-hoc comparison measures the opportunity for a shortcut. Testing whether either model relies on it would require varying irrelevant citations while holding the path fixed. Reproduce with `python3 scripts/audit_terminal_paths.py --shortcuts`. [Scoring code](../scripts/audit_terminal_paths.py)

### Mixed-task retention after completing both arms

| Training condition | Functional hop six / 128 | Complete paths / 128 | Complete moving paths / 78 | Permutation hop three / 96 | Update seconds |
|---|---:|---:|---:|---:|---:|
| Frozen six-hop parent | 4 | 1 | — | 96 | — |
| Functional-only control | 21 | 9 | 0 | 39 | 17.26 |
| Two old permutations per eight-map batch | 21 | 9 | 0 | 57 | 16.56 |

The mixed arm gained 17 net endpoints versus the parent (20 wins, three losses). Against the control it traded 13 wins for 13 losses. Both arms saw the same 96 fresh maps. Replaying old examples recovered 18 old-task answers over the control and still lost 39 against the parent. All nine successful paths for each arm ended at a stationary state. The promotion floor required at least 94 retained old-task answers. [Scored models and paired records](../research/results/functional-interleave-01)

### Natural-language GSM8K adaptation

| 32 matched development questions, 256-token cap | Correct | Scorable | Tokens generated | Generation seconds | Retention NLL |
|---|---:|---:|---:|---:|---:|
| Untouched Base K1 | 1 | 1 | 8,133 | 118.07 | 2.98266 |
| Trained stabilized K2 | 13 | 32 | 3,133 | 64.64 | 2.98727 |

The candidate gained nine net answers on the first half and three on the second half. Each assessment used the same five-domain retention corpus with 9,328 targets. NLL increased in four domains and decreased in reference text. The training run consumed 10,083 response targets and 44.72 synchronized update seconds. On the boxing question, both systems calculated that 20% of 95 is 19. The candidate delivered the final answer within the budget while Base continued checking. The candidate also lost a question Base solved: after two years of 20% evaporation, Base returned the correct 64%, while the candidate subtracted 20 percentage points twice and returned 60%. [Both token streams, retention by domain and weights](../research/sessions/2026-11-25/gsm-rationale-01)

### Fresh word problems at two output budgets

| Cohort | Base at 256 | Trained K2 at 256 | Base at 1,024 | Trained K2 at 1,024 |
|---|---:|---:|---:|---:|
| Public GSM1K, 24 items | 1 | 7 | 13 | 7 |
| SVAMP, 24 items | 8 | 16 | 18 | 16 |
| Correct / 48 | 9 | 23 | 31 | 23 |
| Scorable / 48 | 9 | 46 | 35 | 46 |
| Total generated tokens | 11,771 | 3,955 | 26,138 | 5,491 |
| Measured generation seconds | 172.89 | 82.57 | 383.11 | 114.11 |

The 256-token prefix gave K2 15 paired wins and one loss (two-sided exact McNemar p = 0.00052). At the registered primary limit of 1,024, K2 gained five and lost 13 (p = 0.096). Base delivered 22 additional correct answers after token 256. The trained model delivered its last correct answer at token 167. Its higher completion rate and lower elapsed time hold at both budgets. The long-budget accuracy requirement failed in both sources. The same greedy stream supplies both budgets and the shorter timings were captured during decoding. [Generation records, costs and thresholds](../research/results/rationale-transfer-01)

### Earlier checkpoint and matched dense control

| Same 32 development items, 256-token cap | Correct | Generation seconds | Retention NLL |
|---|---:|---:|---:|
| Untouched Base K1 | 1 | 118.07 | 2.98266 |
| K2 at update 22 | 6 | 62.36 | 2.98980 |
| K2 at update 32 | 13 | 64.64 | 2.98727 |
| Matched dense K1 at update 32 | 10 | 53.56 | 2.98723 |

The last ten K2 updates fixed nine previously incorrect responses and lost two. The earlier checkpoint missed the development threshold and had zero net gains on the second half. The 32-update K2 beat the matched dense control by three answers, with 21% more generation time and 40% more optimizer-update time. Both trained models received 10,083 identical supervised tokens. The earlier endpoint reused the Base outputs and times from the full comparison. The dense comparison reused the K2 outputs and times. Both verified the exact prompt token IDs. [Update 22 records](../research/sessions/2026-11-25/gsm-rationale-endpoint) · [Dense control](../research/results/rationale-dense-01)

### Follow-up evidence on language, retrieval and prefill

| Study | Measured comparison | Decision |
|---|---|---|
| Correctness checks in publisher solutions | On 32 new questions at 256 tokens, K2 scored 12 in 77.5 s and Base scored 3 in 116.7 s. At 1,024 tokens K2 scored 12 in 108.6 s and Base scored 18 in 242.5 s. | The longer-budget accuracy threshold failed. [Archive](../research/results/checked-rationale-01) |
| Short-ledger training plus arithmetic replay | K2 retrieved all four 2K values and both 16K values at the 10% position. It missed both independently sampled 16K values at the 75% position. Base completed zero correct turns in eight. Total time was 51.9 s for K2 and 45.3 s for Base. | Both query placements scored 3/4. Neither qualified for 128K. The original short arithmetic score fell from 13/32 to 6/32. [Archive](../research/results/ledger-curriculum-01) |
| Native resident prefill at 512 versus 128 tokens per block | Median first-token time improved from 4.57 to 4.08 s at 8K, and from 10.64 to 9.53 s at 16K. The 16K greedy streams diverged. Peak allocation rose by 0.62 GiB at 8K and 0.51 GiB at 16K. | Numerical and memory gates failed. Deployed prefill stays at 128 tokens per block. [Crossover measurements](../research/results/prefill-block-01) |

For the two wrong 16K retrievals, the model returned a six-digit value that belonged to another entry in the same ledger. The keys, positions and distractor sets varied across these items, so the two misses cannot isolate a position effect. [Frozen ledgers](../research/data/ledger-curriculum-01.json)

### Frozen arithmetic decoder with a learned passage reader

| Held-out register inputs | Reader K2 exact answers | Untouched Base K1 exact answers | Reader / Base total seconds |
|---|---:|---:|---:|
| 2K, two positions in table and prose | 4/4 | 0/4 | 6.4 / 8.4 |
| 16K, two positions in table and prose | 4/4 | 0/4 | 45.8 / 36.7 |
| All eight requests | 8/8 | 0/8 | 52.2 / 45.1 |

The reader selected the gold span on all eight inputs. Each correct answer was generated through Qwen’s vocabulary head and scored after the thinking span and first turn boundary. Base reached the 96-token cap on all eight. The learned module adds 32,929 parameters. The arithmetic adapter and the original Qwen weights stayed frozen. All 32 short arithmetic streams matched the parent exactly, and the V3 loss was 2.98727 against Base’s 2.98266. [Per-item outputs and checkpoint](../research/results/attention-memory-01)

### Preference training on sampled solutions

| Fresh 32-question cohort | Candidate correct | Base correct | Candidate / Base seconds | Paired wins / losses |
|---|---:|---:|---:|---:|
| 256-token prefix | 16 | 2 | 55.9 / 116.4 | 14 / 0 |
| 1,024-token allowance | 16 | 18 | 71.8 / 206.6 | 6 / 8 |

At 256 tokens the candidate gained seven answers on each source. At 1,024 it led 9–8 on SVAMP and trailed 7–10 on the publisher GSM8K sample. The 40 training pairs contained 3,820 chosen tokens and 3,713 rejected tokens. Sampling took 681.6 s and 30 preference updates took 86.6 s. The V3 NLL was 2.98823 against 2.98266 for Base. This trained variant missed its long-budget gate. [Sampled solutions, reference log probabilities and paired predictions](../research/results/paired-solutions-01)
