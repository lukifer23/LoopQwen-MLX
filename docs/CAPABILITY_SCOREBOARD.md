# Capability scoreboard

## Executable tools and human reading

| Frozen comparison | Recurrent | Comparator | Actual seconds (recurrent / comparator) |
| --- | ---: | ---: | ---: |
| Public numerical questions, correct call and grounded answer | 17/22 | 0/22 untouched Base | 15.15 / 82.82 |
| Catalog requests including chains, comparisons, direct answers and clarifications | 20/30 | 5/30 untouched Base | 13.14 / 51.56 |
| Same catalog with equal training at ordinary depth | 20/30 | 19/30 trained one-pass | 12.93 / 10.88 |
| Four-layer grounded model on new catalog requests | 38/50 | 7/50 untouched Base | 35.43 / 119.05 |
| Human question edits, original and edit both correct | 10/32 | 12/32 untouched Base | 8.29 / 133.93 |

These rows include distinct checkpoints. A two-pass catalog specialist transferred poorly to an independently authored 24-question schema test (0/24 versus 8/24). The four-layer model missed the minimum on both chaining and comparison. The external execution result uses the official instruction-tuned sibling as its starting point and a two-pass final attention layer. The human reading comparison also favored Base on shared-prefix decisions, 48/64 versus 41/64. [Technical report and charts](../paper/loopqwen.pdf) · [Public execution archive](../research/results/tool-dialogue-typed-01) · [Matched catalog control](../research/results/tool-dialogue-depth-01) · [Human contrast archive](../research/results/contrast-boolq-recurrent-01)


## Three-fact induction with a repeated native Qwen layer

Each prompt contains nine shuffled animal-class and color statements and asks for one individual's color. A learned projection compresses the outputs of Qwen layers 0–14 for each sentence and for the question. The same pretrained layer 15, with shared Q/K/V/O adapters, updates the question representation three times against the fixed premise tokens. Its final state supplies a learned adjustment to the full-prompt Base decoder's vocabulary logits. Training uses the publisher's three supporting-fact indices and answer strings. Generation receives only the prompt. Both trained arms have 552,810 trainable parameters and received the same 1,536 training stories, seed, 3,000 Adam updates and minibatches of 32. [Implementation](../loopqwen/native_evidence.py) · [Frozen registration and material](../research/experiments/native-induction-anchored-01.json)

| Two-model comparison | Completed correct answers | Correct common-prefix probes | End-to-end seconds |
|---|---:|---:|---:|
| Primary, three passes vs untouched Base at 192 tokens | 62 vs 0 of 64 | 62 vs 29 | 16.10 vs 181.00 |
| Primary, three passes vs matched one pass | 62 vs 33 of 64 | 62 vs 37 | 15.88 vs 16.04 |
| Disjoint confirmation, same trained three-pass and one-pass weights | 62 vs 25 of 64 | 62 vs 26 | 16.07 vs 16.04 |
| Confirmation, independently reinitialized and trained three-pass vs one-pass readers | 53 vs 25 of 64 | 53 vs 23 | 16.03 vs 16.07 |
| Confirmation, three-pass checkpoint run at three passes vs one | 62 vs 19 of 64 | 62 vs 23 | 15.89 vs 39.71 |
| Confirmation, three passes vs untouched Base at 768 tokens | 62 vs 2 of 64 | 62 vs 28 | 16.02 vs 654.19 |

Across 128 held-out stories, both trained models completed all responses in 13 tokens each. Three passes gained 67 answers and lost one relative to one pass. The two-sided exact paired tests gave p = 2.98 × 10⁻⁸ and p = 1.46 × 10⁻¹¹ on the respective panels. Elapsed time includes sentence encoding, native layer applications, full-prompt prefill and greedy generation. This control matches examples, supervision, parameters and optimizer updates. The 3,000 optimization updates took 114.63 seconds for three passes and 42.28 seconds for one. At inference on confirmation the batch-one median was 250 ms for each trained arm. The nearest-rank 95th percentiles were 260 ms and 259 ms. [Untouched Base comparison](../research/results/native-induction-anchored-01) · [Matched control](../research/results/native-induction-anchored-matched-01) · [Disjoint confirmation](../research/results/native-induction-anchored-replication-01)

With a new seed for both parameter initialization and minibatch sampling, the trained three-pass model gained 31 answers and lost three on the same confirmation panel relative to the reinitialized one-pass control. Its 53/64 was lower than the first seed's 62/64. Both seeds exceeded the registered eight-answer margin over their matched control. [Independent training-seed comparison](../research/results/native-induction-anchored-seed2-01)

A stronger one-pass control trained directly on answer quality, with its support-location loss removed, for 15,000 updates. It consumed 163.29 seconds of optimization versus 114.63 seconds for the three-pass reader. It answered 28/64 on confirmation with 29 correct common-prefix probes. The frozen three-pass reader gave 62/64 on both metrics. End-to-end inference took 16.05 seconds for the one-pass control and 16.06 seconds for three passes. [Longer answer-only control and native timings](../research/results/native-induction-long-control-01)

At 768 generated tokens per question, Base finished 14 turns and the trained model finished all 64. Nine completed Base answers named the animal category and five gave a color. Two of the five colors matched the source. Fifty Base answers exhausted the allowance. The task assumes an individual's color can be inferred from a classmate, even though the original questions omit an explicit statement of that convention. [Longer-budget paired responses](../research/results/native-induction-anchored-longbase-01)

A separate comparison stated the induction rule explicitly in a system message for both models. It used 32 new official test stories, each with a unique supporting classmate, and allowed 768 tokens. The unchanged three-pass weights answered 32/32 in 8.09 seconds, versus 5/32 for untouched Base in 335.64 seconds. Common-prefix scores were 32/32 versus 17/32. Base completed seven turns. [Rule, input tokens, paired responses and times](../research/results/native-induction-instructed-base-01)

In a post-hoc inspection, four stories in each panel had classmates of the queried individual with conflicting stated colors. All eight remain in the registered scores. Across the 120 stories with a unique implied color, the three-pass model answered 120 correctly and the one-pass model answered 55. On confirmation, the first, second and third supervised attention-head pairs gave the highest weight to the annotated supporting sentence on 64, 60 and 60 of 64 questions. At one pass those counts were 64, 14 and 14. Each probability vector is saved with the corresponding response. [Per-story scores, attention distributions and input IDs](../research/results/native-induction-anchored-replication-01)

On 32 fresh, unambiguous stories, a counterfactual exchanged the animal-class words of two people. The question and every other token stayed fixed. Each side had exactly one classmate supporting the new answer. The three-pass reader answered both members of 32/32 pairs correctly, versus 1/32 pairs for the independently trained one-pass control. Treating each original story as a single independent unit gave an exact paired p = 9.31 × 10⁻¹⁰. The individual answer totals were 64/64 versus 23/64 in 15.86 versus 15.68 seconds. [Interventions, outputs and pair-level audit](../research/results/native-induction-class-swap-01)

## Human-paraphrased family relationships

A separate CLUTRR experiment trained the same native layer method on 960 three-sentence family stories, with fourteen English kinship targets and a shared 192-token first-turn allowance. Training and evaluation were disjoint by story ID, full narrative and delexicalized full-story template. All models saw the original sentences and the two named people. The source graph supplied training-only support positions. On the 64-question primary panel the three-pass model answered 34/64 in 16.00 seconds, versus 1/64 for untouched Base in 174.53 seconds. Common-prefix scores were 34 and 6. Base completed 12 turns. Reviewing each visible answer for the two requested people confirmed one correct completion. [Dataset and benchmark paper](https://aclanthology.org/D19-1458/) · [Frozen comparison](../research/results/native-kinship-01) · [Completed-answer review](measurements/answer-audits/native-kinship-01.json)

Under the same training seed, stories, losses, capacity and update count, the one-pass model answered **44/64 in 15.91 seconds**. It scored 37/64 on the common-prefix probe. Three passes gained four answers and lost fourteen against one pass. The matched gate closed the reserved official-test evaluation and deployment for this branch. The first training attempt stopped before any optimizer update because the MLX four-token attention kernel lacked the gradient path. The completed runs right-padded only during differentiation and used the real final question position under causal attention. Inference used three sentence states and a question state. [Matched comparison](../research/results/native-kinship-matched-01) · [Charged interruption and recovery](../research/sessions/native-kinship-short-query.json)

## Standalone routing and 128K qualification

The first native-layer export routed a species question to the color reader. The successor adds a 4,098-parameter linear classifier trained on the submitted question alone. The existing story-level gate also has to accept the input. The intent fit uses five distinct training color questions and 66 alternative training questions. All previously trained reader and decoder tensors keep their original identities. On 32 stories with six alternative questions each, all 192 off-task requests selected their prior route. The nine prespecified paired generations reproduced the complete previous output. [Observed routing regression](../research/results/native-scope-retention-01) · [Scope training and decision](../research/experiments/native-language-release-02.json)

| Replayed capability in the standalone export | Observed result |
|---|---|
| Three-pass induction, 64 confirmation prompts | 64 identical routes and token streams in 16.05 s |
| Two-read deduction, 56 confirmation prompts | 56 identical routes and token streams in 14.86 s |
| Previous mixed benchmark, 80 requests | 80 identical routes and token streams in 418.17 s |
| Raw prose, 120 requests | Ordinary route throughout, identical NLL 2.982658115 over 9,328 scored tokens |
| Resident full-history lookup, two 131,083-token prompts | 2/2 correct, 30 emitted tokens each, 262.80 and 263.04 s, 4.33 GiB peak |
| Untouched Base, identical long inputs and allowances | 0/2 correct in 393.20 s |

The two key values swap their positions near tokens 13,098 and 98,294. The reader selected a span covering the referenced entry and produced the new answer on both requests. The native induction and register experiments measure different tasks in the same deployed model. [Full release replay](../research/results/native-language-release-02) · [Original Base and register pair](../research/results/conditional-language-128k)

## Learned two-fact deduction

Two disjoint publisher test panels used one bAbI English deduction question per unique story. Each trained arm received the same 240 training stories, supporting-fact labels, answer targets, 1,200 optimization updates and trainable parameterization. Answers were generated through the full Qwen vocabulary. The common-prefix probe scored the four possible nouns with identical input tokens.

| 192-token comparison | Correct answers | Correct probes | End-to-end seconds |
|---|---:|---:|---:|
| First panel, learned recurrent vs. untouched Base | 56 vs. 0 of 56 | 56 vs. 4 | 15.15 vs. 157.28 |
| First panel, learned recurrent vs. matched single read | 56 vs. 39 of 56 | 56 vs. 25 | 15.21 vs. 14.74 |
| Confirmation, learned recurrent vs. matched single read | 56 vs. 29 of 56 | 56 vs. 23 | 15.00 vs. 14.77 |
| Confirmation, same trained weights with the second pass enabled vs. disabled | 56 vs. 0 of 56 | 56 vs. 10 | 15.30 vs. 154.93 |

The matched models both generated exactly 14 tokens per answer. Across 112 questions the additional pass gained 44 correct answers with zero losses. Its computation includes learned evidence retrieval and a second execution of pretrained layers 10–15 on sentence states. Both reads recovered their annotated facts on 112/112 questions. With both learned reads retained, bypassing only the second Qwen span produced the same 56/56 answers and probes on confirmation. At a 768-token allowance, untouched Base completed 34/56 and answered 32 correctly in 471.48 seconds. The recurrent reader answered all 56 in 15.31 seconds. The dataset uses generated stories with a small vocabulary. [Paired primary](../research/results/deduction-matched-01) · [Disjoint confirmation](../research/results/deduction-replication-01) · [Same-weights intervention](../research/results/deduction-bypass-01) · [Span-only bypass](../research/results/deduction-span-bypass-01) · [Longer Base allowance](../research/results/deduction-long-base-01)


An independent initialization of both models gave 56/56 versus 29/56 on the same confirmation stories (15.24 versus 14.73 seconds). On 32 additional stories, exchanging exactly two rule-consequence tokens changed the correct answer while preserving the question and token multiset. The recurrent model answered both versions correctly on 32/32 pairs and the one-read model did so on 2/32. A separately trained model retaining both evidence reads and using one Qwen span execution tied the two-span model at 56/56. [New training seed](../research/results/deduction-second-seed-01) · [Counterfactual pairs](../research/results/deduction-consequence-swap-01) · [Matched two-read control](../research/results/deduction-two-read-control-01)

## Supplied-evidence science questions

On 40 human-authored QASC questions with six supplied statements and eight choices, the trained reader gave 23 correct completed answers in 23.62 seconds. Base gave four in 147.16 seconds at the common 256-token cap. With an identical answer prefix, the reader ranked the correct choice first on 32/40 and Base on 37/40. This missed the preregistered content threshold. [Protocol, outputs and scores](../research/results/science-composition-01)


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

### 131,078-token counterfactual retrieval

| Queried value position | Reader answer | Base answer within 96 tokens | Reader / Base elapsed |
|---|---:|---|---:|
| 13,124 | 245799 | Unfinished turn | 325.7 / 249.8 s |
| 98,297 | 307755 | Unfinished turn | 318.0 / 249.8 s |

Each input contains all 131,078 tokens, both values and the same final question. Only two equal-length names exchanged positions. All numeric rows retained exactly the same values at the same offsets. The reader selected both annotated spans and generated both correct first turns. The total time ratio was 1.29 against a registered ceiling of 1.75. One counterfactual document pair cannot estimate accuracy across arbitrary documents. [Both complete requests and independent audit](../research/results/attention-memory-128k) · [Frozen inputs](../research/data/attention-memory-128k.json)

### Natural prose transfer

| Twelve SQuAD-derived reading requests | Exact match | Mean token F1 | Scorable visible answers | Total seconds |
|---|---:|---:|---:|---:|
| Untouched Base K1 | 4/12 | 0.511 | 12/12 | 32.6 |
| Learned reader K2 | 2/12 | 0.408 | 8/12 | 27.5 |

The reader's selected span overlapped the human-annotated source paragraph on only three requests. Its two exact answers were also correct for Base. The reader gained zero and lost two, with early/middle/late exact counts of 1/1/0 versus 2/1/1 for Base. The preregistered 8/12 exact threshold and quality improvement both failed. This closes expansion of the frozen reader on this task. [Free-text outputs and costs](../research/results/prose-reading-01) · [Source, transformation and evaluation policy](../research/experiments/prose-reading-01.json)

### Shared 1,024-token reasoning budget

| Fresh 32 questions | Greedy exact / seconds | Final vote exact / seconds | All generated tokens | Attempts |
|---|---:|---:|---:|---:|
| Untouched Base K1 | 23 / 194.1 | 25 / 468.6 | 32,168 | 90 |
| Original arithmetic K2 | 20 / 69.5 | 20 / 283.2 | 13,575 | 135 |

Both systems first attempted each question greedily with the entire allowance available, then sampled within the remaining budget. Voting fixed two Base errors and lost none. It fixed one K2 error and lost one. The final candidate gained two paired questions and lost seven against Base (exact two-sided McNemar p = 0.180). The model scored 9/16 versus 10/16 on reserved GSM8K and 11/16 versus 15/16 on SVAMP. The registered longer-budget advantage failed. [All 225 attempts, stop boundaries and costs](../research/results/shared-budget-vote-01)

### Full-budget all-layer adapter

| 48 paired questions | Recurrent correct | Base correct | Recurrent / Base seconds | Paired wins / losses |
|---|---:|---:|---:|---:|
| Through 256 tokens | 21 | 7 | 85.1 / 172.9 | 16 / 2 |
| Through 1,024 tokens | 21 | 33 | 85.1 / 342.8 | 3 / 15 |
| Reserved GSM8K at 1,024 | 12/24 | 17/24 | 48.8 / 191.0 | 2 / 7 |
| SVAMP at 1,024 | 9/24 | 16/24 | 36.4 / 151.8 | 1 / 8 |

All 384 updates completed on MLX in 919.1 seconds of update time. V3 language NLL was 3.00277 for the recurrent adapter and 2.98266 for Base. Full-budget accuracy failed on both sources. [Individual generations and weights](../research/results/all-layer-reasoning-01)

### Completed answers under a content-based reading

| Independent panel, 1,024-token allowance | Recurrent | Untouched Base | Paired wins / losses |
|---|---:|---:|---:|
| Original arithmetic transfer | 23/48 | 35/48 | 2 / 14 |
| Inverse-checked rationale | 12/32 | 21/32 | 4 / 13 |
| Preference-trained solutions | 16/32 | 23/32 | 2 / 9 |
| All-layer arithmetic and reading | 21/48 | 38/48 | 0 / 17 |

This secondary analysis keeps each frozen generation and its original completion boundary. For every completed visible response rejected by the numeric scorer, a reviewer recorded the numeric answer actually expressed, including clear prose and equivalent fractions. Unfinished turns still score zero. The annotation view displays the question and visible answer. The scoring step joins each annotation to the model and publisher gold. The four primary decisions continue to use their registered strict rule. [Reviewed labels](measurements/answer-audits) · [Reproducible scoring](../scripts/research_answer_audit.py)

### Recurrent completion of Base deliberation

| 48 fresh questions at 1,024 tokens | Trained adaptive depth | Untouched Base |
|---|---:|---:|
| Strict numeric answers | 27/48 | 25/48 |
| Completed answers with unambiguous mathematical content | 31/48 | 30/48 |
| Reserved GSM8K, strict | 11/24 | 11/24 |
| SVAMP, strict | 16/24 | 14/24 |
| Generated tokens | 22,730 | 26,338 |
| Total seconds, including causal prefix replay | 353.05 | 380.68 |

The trained second pass activated on 18 requests. All 48 ordinary-depth prefixes matched Base exactly. The strict comparison gained three items and lost one (exact paired p = 0.625). The content review gained two and lost one. With a two-answer strict margin and a tie on GSM8K, this endpoint missed the four-answer, both-source promotion rule. [Fixed comparison and checkpoint](../research/results/recurrent-deliberation-01) · [Content review labels](measurements/answer-audits/recurrent-deliberation-01.json)

### Post-training transfer and privileged answer states

The frozen public post-trained sibling completed none of 48 training turns at 640 tokens, closing the sampled-teacher study. The independent hidden-state experiment used 12,418 teacher-distribution positions from 576 training documents. In 96 of those documents, the teacher saw the full publisher rationale and the student saw its first quarter. The 96 native updates took 316.7 seconds and supervised 34,145 hard targets. [Teacher attempts](../research/results/projected-distillation-01) · [Hidden targets and released student](../research/results/privileged-reasoning-01)

| 32 fresh questions | Standalone K2 | Untouched Base | K2 / Base seconds |
|---|---:|---:|---:|
| At 256 tokens | 9 | 5 | 81.3 / 113.5 |
| At 1,024 tokens | 9 | 16 | 197.1 / 271.9 |

At the full budget K2 gained four answers and lost eleven. It solved 4/16 reserved GSM8K and 5/16 SVAMP questions versus 7/16 and 9/16 for Base. Its 9,328-target language NLL was 2.99066 versus 2.98266. The retained post-training update and hidden-state supervision did not yield a deployable long-budget advantage. [Paired outputs](../research/results/privileged-reasoning-01) · [Completed-prose review](measurements/answer-audits/privileged-reasoning-01.json)

### Native reading on 24 previously unused natural articles

| Frozen two-model pair, at least 2,048 input tokens | Exact | Mean token F1 | Total seconds |
|---|---:|---:|---:|
| Untouched Base | 10/24 | 0.517 | 67.54 |
| Context-conditioned recurrent reader | 9/24 | 0.392 | 65.58 |

The learned 64-token passage window overlapped the annotated source paragraph on 20/24 requests. A separate recurrent adapter received 48 updates on annotated answers and their support sentences. The full-history scorer trained on 869 natural examples with supervised answer positions. Passage localization improved relative to the earlier reader's 3/12 on a different cohort, but the new decoder missed the direct answer-quality gate. Its independent native export reproduced all three decoder banks' logits in a fresh process. [Paired answers, training and native model](../research/results/natural-context-01)

### Late recurrent exit

The 896-token switch completed 14 of 16 candidate turns and all 16 Base turns before the 315-second worker limit. On the 14 matched items the candidate scored 9/14 in 123.75 seconds and Base scored 10/14 in 121.77 seconds. All four switched turns ended without a complete visible answer. Base scored 11/16 on the full panel. Even if the candidate had solved both remaining items, it could only have tied Base. The registered two-answer gain was already unreachable. Full-panel candidate accuracy and latency remain unmeasured. Its trained weights, Base prefix identities and completed turns were archived. [Partial paired generations](../research/results/late-recurrent-exit-01)

### Learned exit on authentic extended reasoning

| Frozen 1,024-token comparison | Learned exit | Untouched Base |
|---|---:|---:|
| Reserved GSM8K | 12/16 | 11/16 |
| SVAMP | 12/16 | 11/16 |
| Total under the registered scorer | 24/32 | 22/32 |
| Correct visible mathematical content | 25/32 | 23/32 |
| Generation seconds including replay | 215.89 | 231.63 |
| Generated tokens | 14,426 | 15,861 |
| Independent raw V3 NLL | 2.99538 | 2.98266 |

Seven requests reached the learned recurrent exit. The paired result was two wins and zero losses (two-sided exact p = 0.50). All 32 ordinary prefixes matched the control. The endpoint met its initial qualification and the blind content check. Independent confirmation uses 24 additional reserved GSM8K items and 24 additional operation-balanced SVAMP questions. The project's historical inventory already contains every question in the official GSM8K test split. [Measured endpoint](../research/results/learned-exit-01) · [Blind labels](measurements/answer-audits/learned-exit-01.json) · [Frozen confirmation](../research/experiments/learned-exit-confirm-01.json)

### Full-rank weight transfer

Mixing 60% of the post-trained sibling's internal weight difference into the Base backbone and training 128 recurrent updates improved V3 language NLL from 2.98266 to 2.91114. Strict arithmetic accuracy at 1,024 tokens fell to 12/32 versus 19/32, with 120.17 versus 277.14 seconds of generation. Reviewing the two noncanonical completed answers gave 12/32 versus 20/32 on mathematical content, with two candidate wins and ten losses. The prospective full-budget accuracy criterion closed this branch. [Paired generations](../research/results/fullrank-reasoning-01) · [Blind annotations](measurements/answer-audits/fullrank-reasoning-01.json)

### Independent full-budget arithmetic replication

| Frozen 1,024-token pair | Strict K2 policy / Base | Blinded completed content | Generation seconds K2 policy / Base |
|---|---:|---:|---:|
| First 32 | 24 / 22 | 25 / 23 | 215.89 / 231.63 |
| Independent 48 | 28 / 24 | 33 / 29 | 326.54 / 360.61 |
| All 80 | 52 / 46 | 58 / 52 | 542.44 / 592.24 |

Across both panels the policy gained four GSM8K and two SVAMP answers, without losing a correct answer. The six gains all came from unfinished Base turns. Every shared prefix already contained the correct calculation before token 704. The recurrent finisher selected and delivered those answers. It completed 79/80 first turns versus 65/80. [Six individual gain traces](measurements/answer-audits/learned-exit-gains.json) Raw V3 NLL was 2.99538 versus 2.98266. Training and interrogation of earlier alternative policies make the pooled two-sided McNemar p = 0.03125 a nominal, design-specific result. [Paired generations](../research/results/learned-exit-confirm-01) · [Blind audit and pooled counts](measurements/answer-audits/learned-exit-pooled.json)

### Human-written three-way language inference

On 48 questions in 16 counterfactual premise groups, the trained K2 model scored 35/48 against 27/48 for untouched Base. It solved all three related claims on 7 premises versus 2, and its likelihood probe selected the right relation 37/48 versus 30/48. Native unrestricted greedy generation took 14.23 versus 391.76 seconds at a shared 1,024-token cap. Raw V3 NLL improved from 2.98266 to 2.93196. Blind review of 28 noncanonical responses gave the same totals. The two-sided premise-cluster randomization p was 0.0762. Caption pairs gained six answers and out-of-training-genre MultiNLI gained two. [Training exposure and paired outputs](../research/results/entailment-triplets-01) · [Blind review](measurements/answer-audits/entailment-triplets-01.json) · [Independent confirmation protocol](../research/experiments/entailment-triplets-confirm-01.json)

### One deployed checkpoint on five workloads

The frozen shared-weight model answered 34/56 requests correctly versus 25/56 for Base, in 289.86 versus 322.33 seconds. It gained three answers at the full 1,024-token arithmetic budget, two at 256 tokens and four on the 2K/8K register lookups. The strict score had nine wins and zero losses. Blindly crediting noncanonical answers gave 37/56 versus 29/56. One short arithmetic answer was a loss under that review, leaving nine wins and one loss. [Blind annotations](measurements/answer-audits/conditional-depth-01.json)

All 20 ordinary reading and entailment responses exactly reproduced their Base tokens. Each of 120 raw language passages selected the ordinary weights and reproduced Base's NLL of 2.9826581 over 9,328 targets. The input selector was trained once on 384 separate prompts. Every request was scored through the public generation path, including routing and replay time. [Integrated adapter bundle, raw routes and recorded responses](../research/results/conditional-depth-01)

### Independent language-inference replication

The unchanged natural inference checkpoint answered 50/72 versus 27/72 under the strict scorer. Blind review credited Base for 12 additional answers expressed with words such as “supported”, producing 50/72 versus 39/72. Correct answers increased from 16 to 23 on caption pairs and from 23 to 27 on out-of-training-genre MultiNLI. The three-way likelihood probe scored 51/72 versus 24/72. Eight full triplets were correct versus one for Base. Generation took 24.85 versus 574.91 seconds. Over both panels, blind scores were 85/120 versus 66/120, with 33 wins and 14 losses. Swapping labels within each paired premise as a unit gives a pooled two-sided p of 0.0070. [Independent responses](../research/results/entailment-triplets-confirm-01) · [Pooled blinded results](measurements/answer-audits/entailment-triplets-pooled.json)

### First integrated full-history capacity

Both models processed two 131,081-token prompts on the M3 Pro. The conditional model returned both requested values in 640.81 seconds; Base returned neither in 493.38 seconds. Archive verification rejected the fixed-position interpretation: two swapped record names occupied four and three tokens, shifting intervening values by one position. The unchanged native measurements are preserved alongside the failed protocol decision. The generator now proves equality of the individual token widths and the complete token-level swap before a new prompt is frozen. [Invalidated intervention and native outputs](../research/results/conditional-depth-128k)

### Combined arithmetic, inference and register model

| Fresh family | Reviewed correct, combined model | Reviewed correct, Base |
|---|---:|---:|
| Arithmetic, 1,024 generated tokens | 17/24 | 14/24 |
| Arithmetic, 256 generated tokens | 7/8 | 1/8 |
| Natural inference, 1,024 generated tokens | 25/36 | 24/36 |
| Reading, 256 generated tokens | 2/8 | 2/8 |
| Register lookup at 2K or 8K input | 4/4 | 0/4 |
| **All requests** | **55/80** | **41/80** |

The complete model used **419.12 seconds** against **536.26 seconds** for untouched Base, including routing, full-prefix replay and full-history prefill. The strict first-turn scorer gave 49/80 versus 33/80. Blind review credited 14 extra correct answers with zero losses. The conservative selector sent 10/36 inference requests to the inference specialist; the other 26 used ordinary Base. Across those 26 and eight reading requests, all 34 ordinary outputs matched Base token for token. Each of the 120 raw retention passages selected the ordinary path and exactly reproduced Base's 2.9826581 NLL. [Paired requests and deployable checkpoint](../research/results/conditional-language-02) · [Blind annotations](measurements/answer-audits/conditional-language-02.json)

### What the extra pass contributed under equal training

| Same questions and training exposure | K2 | Trained K1 |
|---|---:|---:|
| Arithmetic answers across 80 questions | 52 | 52 |
| Arithmetic generation seconds | 528.14 | 524.69 |
| Natural inference judgments across 120 claims | 85 | 85 |
| Complete natural inference triplets across 40 premises | 15 | 15 |
| Three-way vocabulary probe across 120 claims | 88 | 91 |
| Natural inference generation seconds | 39.13 | 31.92 |

Both arithmetic finishers agreed on every item's correctness. The two inference models differed on six judged items, with three correct only at each depth. Training achieved a reproducible gain over untouched Base. These paired depth controls found equal final-answer accuracy and lower elapsed time for trained K1. [Arithmetic attribution](../research/results/learned-exit-dense-01) · [Natural inference attribution](../research/results/entailment-dense-01)

### Final resident full-history deployment

The combined checkpoint from the 80-request comparison processed two 131,083-token documents and retrieved both counterfactual values. Base retrieved neither. Candidate generation took 262.37 and 262.13 seconds; Base took 196.78 and 196.42 seconds. The first token arrived at about 261 seconds for the candidate. MLX peak allocation was 4.32 GiB for the run. Both three-token keys and the complete token permutation were verified before inference. On the 2,054-token preflight the resident and disk-backed candidates selected the same passage and emitted the same 29 tokens. The 128K evidence covers two versions of one synthetic lookup document. [Exact outputs, TTFT and archive verification](../research/results/conditional-language-128k)

## Frozen public tool model and independent reading tests

| Workload | Recurrent result | Comparator | Measured seconds, recurrent / comparator |
| --- | ---: | ---: | ---: |
| Human numerical requests with executed calls, 22 | 17 | 0 untouched Base | 15.15 / 82.82 |
| Same numerical requests, unchanged instruction-tuned sibling | 17 | 0 sibling | 15.18 / 123.78 |
| Same numerical requests, trained one-pass sibling | 17 | 18 matched K1 | 14.98 / 14.37 |
| Multiple competing function schemas, 24 | 18 | 8 untouched Base | 10.23 / 82.29 |
| Historical single-function transfer panel, 24 | 16 | 8 untouched Base | 7.18 / 95.46 |
| Two controlled catalogs, 80 | 53 | 12 untouched Base | 40.77 / 170.62 |
| Both original and edited BoolQ questions correct, first 32 passages | 10 | 12 untouched Base | 8.29 / 133.93 |
| Both original and edited BoolQ questions correct, other 32 passages | 15 | 19 untouched Base | 6.84 / 142.84 |

The 22 numerical executions measured 17 correct final quantities against eight for Base even when scoring answers independently of tool use. Median initial first-token latency was 92.81 ms against 91.46 ms. The same packaged tool model reproduced all 22 public transcripts and four of five live installed-version queries. The ordinary branch reproduced all 200 earlier language generations, raw NLL 2.982658 on 9,328 tokens and both full-history 131,083-token swaps. Two packaged backbones occupy 2.207 GiB. Public schema selection and live numerical execution measure different behaviors. [Paired transfer data](../paper/typed-transfer.csv) · [Latency profiles](../paper/tool-latency.csv) · [Offline export archive](../research/results/tool-language-release-01)
