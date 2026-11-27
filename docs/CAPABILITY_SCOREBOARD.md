# Capability scoreboard

## Learning from actual tool mistakes

| Same 84 requests | New trained loop | Untouched Base |
| --- | ---: | ---: |
| Grounded, all requests | 12 | 3 |
| Grounded new linked questions, 48 presentations | 1 | 0 |
| Required article observed in order | 9 | 0 |
| Both presentations correct, 24 questions | 0 | 0 |
| Grounded software decisions, 8 requests | 5 | 0 |
| Complete software pairs, 4 packages | 2 | 0 |
| Completed visible responses | 26 | 30 |
| Total wait / summed first-token wait, seconds | 199.53 / 22.87 | 671.88 / 22.23 |
| Peak native allocation, GiB | 1.175 | 1.736 |
| Uncached prompt / generated tokens per second | 2,645 / 105.55 | 2,824 / 69.91 |

The registered test on new source questions gave a net gain of one and paired p = 1. The loop lost one previously successful Base request. Against the previous trained policy on 36 identical requests, software improved from zero to five, scheduling fell from five to two and lookup fell from four to zero. Release 05 retains its separate long-context measurements. [Native study, weights and decision](../research/results/linked-rollin-02) · [Full paper](../paper/loopqwen-grounding-study.pdf) · [Matched historical requests](../paper/linked-rollin-retention.csv)


| Measured capability | Trained loop | Untouched Base | Weights |
| --- | ---: | ---: | --- |
| Human ordinary-domain table judgments | 25/72 | 13/72 | Valid-return preferences, K=2 |
| Grounded conference room / weekday decisions | 8/8 / 1/8 | 0/8 / 0/8 | Valid-return preferences, K=2 |
| Correct ordered date and calendar returns | 8/8 | 0/8 | Valid-return preferences, K=2 |
| Grounded Python compatibility | 0/16 | 0/16 | Valid-return preferences, K=2 |
| Complete time, same 112 requests | 197.98 s | 918.79 s | Valid-return preferences, K=2 |
| Summed first-token wait | 17.69 s | 17.55 s | Valid-return preferences, K=2 |
| Human cross-domain table judgments | 25/72 | 4/72 | Field selection, K=2 |
| Complete six-judgment tables | 0/12 | 0/12 | Field selection, K=2 |
| Grounded Python compatibility | 0/16 | 0/16 | Field selection, K=2 |
| Total time, same 96 requests | 99.21 s | 736.82 s | Field selection, K=2 |
| Summed first-token wait | 14.17 s | 14.09 s | Field selection, K=2 |
| Run peak MLX allocation | 1.109 GiB | 1.715 GiB | Field selection, K=2 |
| Human logical-story judgments | 22/72 | 2/72 | Proof curriculum, K=2 |
| Physical 131,084-token requests | 2/2 | 0/2 | Installed release 05 |

The three reasoning panels used separately trained policies. The field-selection policy missed its registered quality threshold. Its two-visit architecture has no same-policy 128K qualification. [Direct field comparison](../research/results/field-selection-01) · [Proof comparison](../research/results/proof-curriculum-01) · [Installed long-context traces](measurements/file-evidence-long-replay.json)

| Cross-view reading on one resident checkpoint | Trained loop | Untouched Base |
| --- | ---: | ---: |
| Both questions in both formats, complete articles | 0/20 | 0/20 |
| Correct file retrievals | 40/40 | 16/40 |
| Correct grounded file answers | 9/40 | 5/40 |
| Correct inline answers | 12/40 | 17/40 |
| Grounded table retention | 3/8 | 0/8 |
| Completed requests | 85/88 | 67/88 |
| Cold stream / first-token sum | 81.37 / 14.84 s | 477.58 / 14.61 s |
| Opening / decode tokens per second | 2,482.5 / 105.79 | 2,521.9 / 69.99 |
| Peak MLX allocation | 1.12 GiB | 1.74 GiB |

The arm-blind reading credited seven versus six grounded file replies and 16 versus 23 inline replies. That yielded zero versus one complete article. A fixed-weight intervention on the compiled student changed next-token logits after a real file return (RMS 0.7085) and restored all 248,320 logits exactly. The direct trial missed its primary gate. [Predictions and weights](../research/results/cross-view-reading-01) · [Paper and figures](../paper/cross-view.pdf) · [Blind reading](measurements/cross-view-answer-reading.json)

The new Base-derived two-layer decoder grounded 12 of 32 individual file judgments versus 4 for untouched Base. Across 16 independently changed files the loop grounded both correct answers zero times, Base once. The loop read the right source on both versions of 12 articles. On independent inline passages it solved 7/12 complete pairs versus 8/12. It completed 50/56 requests versus 34/56 and used 1.58 versus 1.71 GiB peak. Users waited 7.26 versus 6.93 seconds in total for first tokens. [All paired requests](../paper/file-evidence-requests.csv) · [Verified paths from file to model](measurements/file-evidence-return-path.json)


On two snapshots each of 24 actual Wikipedia articles, the compact Base-derived K2 reader gave **33/48 correct conclusions versus 40/48** and **10/24 correct article pairs versus 16/24**. A blinded semantic review gave 33 versus 39 and 10 versus 15. Stream time was **9.03 versus 228.95 s**, summed first-token time 3.97 versus 4.27 s, and peak allocation 1.04 versus 1.63 GiB. The loop decoded 95.19 versus 70.50 tokens/s. Its input processing was slower at 1,991 versus 2,223 tokens/s. [Registration and archive](../research/results/revision-evidence-01) · [Article-level measurements](../paper/revision-evidence-requests.csv)

The role-binding fit grounded **7/32 original WikiSQL answers versus 2/32** for Base, with **7 exact cell values versus 0**. Blind reading gives 7 versus 1 if a self-contradictory numerical answer is excluded. Both solved one of 24 complete WinoGrande pairs, despite 23 versus 11 correct individual sentences. The package score was two grounded messages versus one and zero complete conversations for both. Total wall time fell from 824.89 to 134.93 s. First waits rose from 11.69 to 11.95 s. [Direct pair](../research/results/role-binding-01) · [Request-level accuracy and cost](../paper/role-binding-requests.csv) · [Independent SQL verification](measurements/role-binding-source-check.json)

In the original Winograd panel the interpolated loop answered 17/40 versus 18/40 for untouched Base. For complete opposite-meaning pairs the scores fell to 0/20 versus 1/20. The loop and Base each grounded one of 16 file requests and solved zero complete inventories. Wall time favored the loop, 114.63 versus 370.32 s. Its summed first wait was longer, 6.93 versus 6.62 s. [Paired responses](../paper/instruction-interpolation-requests.csv) · [Native activation intervention](measurements/instruction-interpolation-activation.json)

On 24 held-out financial reports, a new two-pass model called the calculator on **24/24 arithmetic questions** and **1/24 lookups** about the same tables. Four calculations versus three for Base gave the correct answer and passed source-operand checks. Overall, final report quantities favored Base **18/48 versus 6/48**. On eight real package inventories the loop read two file sizes correctly versus zero, but neither computed a requested difference. Across 98 dialogues the loop scored **35 versus 42** in **92.09 versus 488.92 seconds**. The fit stays experimental. [Weights and paired outputs](../research/results/report-action-field-01) · [Every request and callback](../paper/report-action-requests.csv) · [Physical checks on all sixteen original wheels](measurements/report-distribution-bytes.json)

All 120 held-out induction questions with a single observed peer color were answered correctly by the trained three-pass model, against 55 for the equally trained one-pass control. In the eight stories with competing colors both returned an observed color. The original publisher scores across all 128 remain 124 and 58. [Every premise and native answer](../paper/induction-peer-requests.csv) · [Per-panel sensitivity](../paper/induction-peer-sensitivity.csv)

Both systems retained the full 158,693-token opening of Hawthorne’s novel across follow-ups. Neither completed either human question. The loop grounded both Hawthorne file answers and neither Piper file answer. Base grounded one on Hawthorne and both on Piper. Six complete requests took 353.20 versus 319.32 seconds. The released tool model missed its book-quality, wall-time, first-token and cold-prefill gates. [Sealed book conversations](../research/results/book-conversation-field-01) · [Calls, returned quantities and latency](../paper/book-conversation-requests.csv)

With an actual equality-checking calculator, the qualified loop produced nine correct grounded answers on 24 new human questions versus zero for Base, with six grounded successes involving checked equalities. Final quantities were 9/24 versus 10/24. Whole requests took 30.56 versus 93.89 s and summed first-token time was 3.14 versus 2.74 s. [All trajectories](../research/results/checked-equations-field-01) · [First-hand pencil calculation](measurements/checked-calculator-installed-cli.json)

The question-contrast adapter completed 118/118 user requests versus 91/118 for Base. Its correct quantities were 2/48 versus 13/48 on DROP and 12/24 versus 15/24 on ASDiv. Grounded ASDiv calculations were 12 versus 9. On the 12 two-request services its registered 9/12 versus 2/12 becomes 11/12 versus 9/12 after accepting ordinary quantity formatting with exactly one correct call. Allowing redundant correct lookups gives 11/12 versus 10/12. The registered reasoning gate failed. [Frozen study](../research/results/question-contrast-01) · [Scoring sensitivity](../paper/service-meaning-summary.csv)

Private conversation prefixes reproduced all six source-reading and file-tool exchanges. Across the four follow-ups, time to first token fell from 13.63 to 0.47 seconds. Whole-stream time fell from 36.21 to 13.04 seconds, peak allocation rose from 1.40 to 1.47 GiB, and both policies solved 2/6 requests. This is a verified runtime improvement at fixed model behavior. [Counterbalanced measurements](../research/results/conversation-document-prefill-01)

On the same requested file at two actual Git revisions, the new reader produced both changed byte counts exactly from 173,129 and 191,099 tokens. On an 82,862-token lockfile it named the right value in a sentence. The three cases scored 2/3 versus 0/3 under the bare-number rule and 3/3 versus 0/3 for numerical content. Cold first-token sums were 291.44 versus 793.25 seconds, total wall was 295.22 versus 810.54 seconds, and peak was 2.95 versus 4.82 GiB. The qualified release retains its former default. [Complete documents, evidence spans and replay](../research/results/document-frequency-reader-01)

A Base-initialized loop correctly distinguished both receipts on 16/24 new human questions versus 2/24 for Base (p=.000122). In 82 free dialogues it scored 29 versus 30. Arithmetic finals were 11/48 versus 19/48, with zero grounded repairs of an initially wrong numerical return for either arm. See the distinct conditional and end-to-end measures. [Free and matched comparisons](../research/results/observation-review-01)

On the official Qwen function template, the trained loop completed 8/12 real file and package conversations versus 2/12 for Base. On the same 22 public numerical functions, native-format scores were 21/22 and 20/22. Earlier compact-format results were 17/22 and 0/22. On 24 fresh human arithmetic questions final quantities favored Base, 15 versus 7. Across all 58 dialogues the whole-dialogue advantage was inconclusive (35 vs 30, p=.359). [Complete native message study](../research/results/native-function-dialogue-01) · [Actual calls and returned values](../paper/native-function-requests.csv) The quoted-filename and package-absence audit gives 10/12 versus 9/12 with exactly one correct call, or 10/12 each allowing redundant correct lookups. [Recount of all 24 exchanges](../paper/service-meaning-requests.csv)

After training on 119 actual calculator exceptions, the recurrent system achieved 44/120 strict outcomes versus 12/120 for Base in 139.63 versus 506.10 seconds. On 48 fresh human problems, it obtained numerical returns on 37 of 38 recovery requests. It produced five grounded successes after an error. Final numerical answers favored Base, 26 versus 10. The repair stays opt-in. [Full paired evidence](../research/results/callback-recovery-01) · [Arithmetic results](../paper/calculator-recovery.svg)

The new source-balanced recurrent return took 27.55 seconds to score 49,279 next tokens against 31.36 seconds for Base. On the fresh 24 works, 18 had worse loss and six improved (paired work sign test p=.0227). The mean gate on the registered probe fell from 0.25 to 0.00000276. The ordinary language champion remains pinned Base. [Native measurements](../research/results/selective-raw-return-01) · [Individual works](../paper/source-balanced-works.csv) · [Unscored user continuations](measurements/source-return-installed-cli.json)

On a fresh **131,086-token** register and its key swap, the looped model answered **2/2 versus 0/2** for untouched Base. Eight-bit feed-forward and output weights raised its post-first generation rate to **29.27 versus 27.42 tokens/s**. Including both weight conversions, the two requests finished in **167.39 versus 390.86 s** at **3.32 versus 3.78 GiB** peak MLX allocation. Both individual inputs favored the recurrent system on prompt and output rate. The comparison covers one document with two placements of the same queried name. [Sealed predictions and timings](../research/results/register-mixed-weight-full-01) · [Per-request conversion and physical history](../paper/mixed-weight-requests.csv)

The full repository trees at two immutable revisions contained 169,890 and 173,151 prompt tokens. Both systems retained the full history and scored **0/2**. The recurrent reader selected positions 10–35 on both inputs. The answers began at 66,221 and 66,792. The loop used 233.62 s and 2.71 GiB versus 644.01 s and 4.51 GiB for Base. [Full native traces](../research/results/repository-tree-field-01) · [Per-revision results](../paper/repository-tree-requests.csv)

## Reasoning and tools on one intact instruction backbone

The two-view recurrent assistant scored **67/120 versus 50/120** for untouched Base, with 36 paired gains and 19 losses (p = 0.0300). It took **90.44 versus 472.52 s** with 1.67 versus 1.68 GiB peak MLX allocation. Summed first-token time was 15.87 versus 13.64 s. The 48 fresh questions favored Base 38/48 to 29/48. Field word problems favored the candidate 10/10 to 3/10, while ten-document reading favored Base 8/10 to 2/10. Seven previously correct field tasks and one public execution regressed. [All 120 paired requests and trained weights](../research/results/shared-session-01) · [User outcomes and time by family](../paper/shared-session.svg) · [Installed arithmetic and NumPy comparison](measurements/shared-session-installed-cli.json)

On the 72 fresh human questions, the unified model delivered **45 correct answers within two seconds versus 18** for Base. By four seconds Base had delivered 46, and by eight seconds it had delivered 50. Both had the same generation allowance. The denominator includes every failed or unfinished request. [Per-deadline paired outcomes](../paper/correct-by-deadline.csv) · [Correctness over real elapsed time](../paper/correct-by-deadline.svg)

## Single resident instruction backbone

The folded model and untouched Base each supplied 39 correct final answers across 92 requests. Requiring the actual requested tool action gave 33 versus 30 successes (21 gains, 18 losses, paired exact p = 0.749). On the 40 fresh human questions there were zero gains and 18 losses. The folded model lost seven previously successful numerical or database actions. Wall time was 177.99 versus 364.74 seconds, summed first-token time was 7.73 versus 7.30 seconds and peak MLX allocation was 1.06 versus 1.66 GiB. Its 799,740,004 shared-and-tool parameter bytes and improved hidden reconstruction did not satisfy the user-level gate. [Measured decision](../research/results/folded-system-01) · [All requests](../paper/folded-requests.csv)


## One deployed assistant

The release 05 portable file restores all 66 files and the actual package output in 7.51 seconds at 1,544,038,920 bytes. [Portable release 05 receipt](../research/results/transport-lossless-04)

Release 05 matched Base on all 5,587 localized names in 19 languages, fixing 2,150 input mismatches and retaining every registered release 04 output. A Hindi package follow-up used the current tool but repeated the previous version string. [Installed release qualification](../research/results/packed-release-retention-04) · [Live conversation](measurements/unicode-package-installed-cli.json)

Release 04 qualified the packed long-context runtime. On the historical 131,084-token pair it restored two correct outputs against zero for shipped release 03, with lower peak MLX allocation on both requests. It also retained the original two 24K outputs. [Isolated release comparison](../research/results/packed-release-retention-03). Its parent reproduced 22 public tool sessions and all 200 shorter language outputs. In the mixed field comparison, candidate/Base correct counts were 8/8 versus 1/8 on files, 3/12 versus 0/12 on SQLite, 0/10 versus 0/10 on strict one-call package versions, 2/10 versus 3/10 on human arithmetic and 8/10 versus 8/10 on ten-document reading. The candidate supplied nine correct version strings. The complete stream took 83.97 versus 234.20 s, with summed first-token times of 20.33 versus 9.15 s and peak allocations of 1,810,571,734 versus 1,805,158,874 bytes. [Single-installation results](../research/results/release-mixed-field-01) · [All five families](../paper/mixed-deployment.csv)

The packed release reproduced 512 next-token decisions and 512 losses after consuming two entire books, all 12 shorter requests and two actual installed-version trajectories. Its two further 24,593-token register generations completed in 32.44 versus 37.52 s for the previous release with 13.74% lower peak. The independent 131,080-token comparison of that register policy against Base scored 2/2 versus 0/2 in 171.88 versus 391.67 s with 25.99% lower peak and slower post-first-token generation. [Offline crossover](../research/results/packed-release-retention-01) · [Physical 131K comparison](../research/results/register-packed-direct-full-01)

An outside-checkout replay uncovered a difference between shipped and research inference on the 131,084-token pair. Release 03 and the first compact bundle selected the correct spans but returned 876667 twice. Their bundled evidence mask omitted the packed cache class. Both book streams, 12 short outputs and two live tool calls matched. The compact candidate failed its release gate. [Qualification and transcripts](../research/results/packed-release-retention-02) · [Completed repair](../research/results/packed-release-retention-03)

On the corrected installed loader, release 04 reused one tokenizer per process. All 200 measured trajectories matched. Cumulative time to first token across mode switches fell from 29.58 to 16.31 seconds. End-to-end time across the two 50-request streams fell from 164.66 to 151.60 seconds. The installed assistant enables tokenizer reuse for the measured release. [Release 04 crossover](../research/results/session-tokenizer-02) · [Measured streams](../paper/corrected-tokenizer-streams.csv)

## Required arguments in actual conversations

The state-supervised two-pass model solved 32/54 requests versus 0/54 for Base in 54.86 versus 348.35 s. On the 22 public calls it retained all 17 earlier successes and added three. The missing-input test exposed a separate failure. It asked for the missing argument on 5/16 requests and called prematurely on 9/16. All four complete-and-missing pairs it solved involved files. It solved no complete pair in the time-zone domain. Peak allocation was 1.09 versus 1.66 GiB and accumulated first-token time was 7.25 versus 6.48 s. [Grounded decisions and trained weights](../research/results/dialogue-state-fields-01) · [Each call, answer and real follow-up](../paper/dialogue-state-conversations.csv)

## Learned tool selection and language transfer

| Measured comparison | Recurrent candidate | Untouched Base |
| --- | ---: | ---: |
| Grounded answers using full article and paper files, 32 questions | 1 correct, 32 requests with a call | 0 correct, no calls |
| Grounded execution, 78 public, human math and live resource requests | 45 correct in 54.56 s | 0 correct in 333.15 s |
| Both requested properties of the same live resource | 6/16 | 0/16 |
| Final numeric answer on the 24 human word problems | 6/24 | 14/24 |
| Human single-passage exact answers and mean token F1 | 13/64 and 0.419 | 16/64 and 0.495 |
| Raw next-token decisions on 300 passages, 36,126 targets | 15,076 in 20.81 s | 15,006 in 24.21 s |

The tool fit retained all 17 prior public successes and reached 19/22. The reading fit used less peak memory and completed every question in 17.73 s versus 145.60 s. The raw fit improved all three suites in aggregate, with 17 of 18 additional fresh correct tokens coming from four Python manuals collectively. The CSV, ZIP, XML and time-zone manuals contributed +11, +3, +4 and −1 respectively. The three protocols retain their original domain gates. [Tool trajectories](../research/results/tool-dialogue-intent-pairs-01) · [Human reading](../paper/passage-scores.csv) · [Raw text](../paper/selective-domains.csv)

On 24 additional works the two-horizon model improved loss on only 2/24, and made 32 fewer correct predictions on 14,351 fresh positions. Gains of 72 on WikiText and 33 on earlier retained text gave a pooled gain of 73. Base remains the general-language reference. [All 24 fresh works](../paper/future-token-sources.csv) · [Native model and decision](../research/results/future-token-prose-01) · [Free continuation from the installed CLI](measurements/future-token-installed-cli.json)

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

## Joint tool training and independent field checks

The two-pass adapter trained with paired changes to observations scored **112/132** against **15/132** for ordinary Base. It ran in **69.55 s** against **323.35 s** and peaked at **1.09 GiB** against **1.66 GiB**. Scores by panel were 26/30 versus 5/30, 42/50 versus 7/50, 18/22 versus 0/22, and 26/30 versus 3/30 across four previously unseen utility domains. The older typed checkpoint had supplied 70 correct answers on the 102 shared requests. The new fit supplied 86, with 21 gains and five losses. Each loss involved a comparison. The matched-depth and structural retention endpoints address separate questions about these weights. [Native paired records](../research/results/tool-dialogue-counterfactual-01) · [Paired prior-model changes](../paper/joint-tools.csv)

The original typed weights passed their own 22-request prefix crossover. The full-prefill sessions totaled 30.14 s and the reuse sessions 27.85 s. Across all 88 executions the outputs and observed callbacks matched. The qualification applies to those exact weights. [Balanced crossover](../research/results/tool-prefix-typed-01)

On human word problems, a model trained to write calculator programs delivered 17/48 correct computed answers against 0/48 for Base, but fewer correct final quantities overall, 17/48 against 20/48. The HotpotQA reader gave 11/40 answers containing the reference phrase against 20/40 for Base. [Field scores and elapsed time](../paper/field-performance.csv)

The equally trained one-pass continuation scored 108/132 versus 112/132 for two passes. Two passes gained eight and lost four, took 69.33 versus 60.21 seconds, and missed the registered resource threshold. The descriptive 95% world/function-cluster bootstrap interval for the accuracy difference spans −2.27 to +9.09 percentage points. Against untouched Base the corresponding interval spans +65.15 to +81.06 points. [Depth comparison](../research/results/tool-dialogue-counterfactual-depth-01) · [Clustered differences by panel](../paper/matched-tool-effects.csv)

On a fresh 48-question panel, the structured operand planner returned 20 correct final numbers against 28 for Base and made 19 correct grounded executions against zero. It completed all 48 sessions in 11.00 s against 30 completions in 182.43 s for Base. Each source lost four correct final answers. [Paired programs and outputs](../research/results/operand-planner-01)

The venue dialogue model solved both changed inventories on 7/28 human conversations against 0/28 and 20/22 public numerical requests against 0/22. Across their 78 sessions it scored 34 versus zero in 78.84/299.61 s. The fully held-out attraction service scored 0/12. Hotels scored 2/8 and restaurants 5/8. Its public numerical score retained all 17 earlier typed successes and added three. [Archived answer paths and interventions](../research/results/tool-dialogue-venue-state-02)

The page-prefill crossover retained every physical key/value entry and all generated token IDs on four register requests. Aggregate time to first token fell from 76.09 to 58.48 seconds, total generation time from 78.96 to 61.34 seconds, and maximum native bytes from 2,545,878,064 to 2,535,916,474. This gate qualified a fresh 131,073-token pair against untouched Base. [Archived source comparison](../research/results/register-page-prefill-01) · [Work and time by request](../paper/page-prefill.csv)

A new 131,073-token equal-width key swap returned both correct values in 176.55 seconds versus zero in 391.72 seconds for Base. First-token time summed to 173.99 versus 384.79 seconds. Eight recurrent sites each retained all 131,102 consumed positions. Peak allocation was 4.32 versus 3.78 GiB, a 14.38% increase. [Per-arm scores and resources](../research/results/register-page-full-01)

On 48 unseen human-authored table requests, the native structured parser achieved 14/48 grounded results against 0/48 for untouched Base in 24.55 versus 261.24 seconds. Peak was 1.21 versus 1.66 GiB. In publisher order its operation scores were 1, 2, 2, 3, 4 and 2 out of eight. The per-operation floor closed promotion. [Paired plans, outputs and fit](../research/results/table-parser-01)

The independently fitted two- and one-pass tool models completed 12/16 and 6/16 real source-file conversations in 13.24 and 36.45 seconds after loading. Every earlier two-pass trajectory replayed exactly. One pass hit the token limit on all eight initial listing exchanges despite sometimes making the right call. Nine gains and three losses gave paired p=0.146. The depth claim on this cohort remains unconfirmed. [Matched protocol and transcripts](../research/results/conversation-followup-depth-01)

On 52 held-out WikiText articles, the globally distilled recurrent decoder improved log likelihood on 50. Across WikiText and the existing retention corpus it predicted 9,515 versus 9,451 tokens correctly, reduced NLL from 3.09002 to 3.03556 and took 13.03 versus 14.73 seconds. On the retained 9,328 targets alone it lost 27 correct decisions, closing global replacement of the ordinary decoder. [Paired next-token predictions and independent recount](../research/results/raw-distillation-01)

On 48 new HotpotQA bridge questions the linked two-pass reader found both supporting documents on 23 and produced 17 exact matches versus 15. By the registered reference-content endpoint it scored 18 versus 19, with lower F1 and a slower first token. Full answers took 26.69 versus 132.05 seconds. [All ten-document selections and generated answers](../research/results/linked-reading-01)

A distinct 131,080-token register comparison with native direct 8-bit KV products met every registered quality and resource condition. The loop scored 2/2 versus 0/2, summed first-token time was 168.35 versus 384.65 s, summed complete time was 171.88 versus 391.67 s, and peak MLX usage was 2.80 versus 3.78 GiB. This is one full-length document with two fact placements. [Physical receipts from all attention sites and both decoded responses](../research/results/register-packed-direct-full-01)

Both versions of the full novel _McTeague_ retained 162,714 or 162,711 input tokens through six native global sites. The shipped router selected ordinary depth and reproduced every Base output token. Neither system identified the character who notified City Hall, before or after 529 corresponding name substitutions. Aggregate elapsed times were 596.38 and 596.42 seconds. [Human question, full-history receipts and exact generations](../research/results/narrative-full-book-01)

A two-pass semantic alignment model answered 44/48 Wikipedia paraphrase comparisons versus 5/48 for Base. With a fixed answer prefix it scored 44 versus 36, completing both judgments for 20 versus 13 anchors. On 32 real AskUbuntu comparisons it answered “no” every time. Neither model distinguished both alternatives for any support question, and the prespecified transfer requirement closed promotion. [Decisions, weights and generated answers](../research/results/meaning-alignment-01) · [Paired endpoints](../paper/meaning-pairs.svg)

On an independent 131,084-token document and its key swap, the fused recurrent decoder retrieved both changed values versus zero for Base, using 169.85 versus 390.94 seconds and 2.80 versus 3.78 GiB. Prompt throughput rose to 1,564 versus 683 tokens/s. Base emitted subsequent tokens faster, at 27.81 versus 26.25 tokens/s. This missed the preregistered release condition. [Measured comparison](../research/results/register-fused-base-full-01) · [Every request and physical history](../paper/fused-base-requests.csv)

The two-pass arithmetic decoder read up to 96 actual ordinary reasoning tokens and completed 28/48 grounded requests versus zero in 77.13 versus 166.59 seconds. Base answered 29 correctly when given credit for freely generated correct numbers. ASDiv tied 17/24 and SVAMP was 11/24 versus 12/24. [Weights and full trajectories](../research/results/draft-program-01) · [Per-question expressions, observations and timing](../paper/draft-program-requests.csv)


On 48 issuer-disjoint financial excerpts and independent school tables the stepwise recurrent calculator produced five correct final answers versus six for Base. Four answers were backed by a successful calculation, one by a chain reusing an earlier result. It completed 45 requests versus six, taking 97.99 versus 266.23 seconds with a 1.13 versus 1.70 GiB peak. The final-value and linked-computation gates failed. [Case-level answers and operations](../paper/report-calculation-requests.csv) · [Decision](../research/results/report-calculation-01)

The trained search adapter called the document tool on all 40 requests and received an annotated answer span on 22. It returned six grounded exact answers versus zero for Base, including four of eight questions about the project's actual guides. Including index construction it took 43.86 versus 156.09 s. First-token time increased from 8.43 to 9.10 s. Its registered floor was 14 grounded answers. [Decision and weights](../research/results/document-evidence-01) · [Every question, search and answer](../paper/document-evidence-responses.csv)

The joint two-pass model served 143/144 requests directly and delegated one short evidence task to its qualified parent. It scored 74 versus 62 overall (35 gains, 23 losses; paired $p=0.148$) and 45 versus 50 across new human questions. It retained every public execution previously correct on the typed adapter and added two, while losing twelve previously correct field tasks. Its 17.25 s summed first-token wait also exceeded Base's 15.59 s. [Full paired result](../research/results/shared-session-tokenizer-replay-01) · [Prompt-level results and real observations](../paper/unified-session-requests.csv)
