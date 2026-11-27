#set document(title:"Learning to bind the right argument",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:0.82in,y:0.73in),footer:context [#set text(size:8pt,fill:rgb("#48625e"))
LoopQwen-MLX · Current research report #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:11pt)
#set par(leading:0.57em,spacing:0.64em)
#show heading.where(level:1): h => block(above:9pt,below:5pt)[#set text(size:16pt,fill:rgb("#125b51"))
#h]
#let tbl(cells,cols) = table(columns:cols,stroke:none,inset:4pt,fill:(x,y)=>if y==0 {rgb("#dfede9")} else if calc.odd(y) {rgb("#f4f6f5")},..cells.flatten())
#text(size:26pt,weight:"bold")[Learning to bind the right argument]
#linebreak()
#text(size:14pt)[Paired human reasoning, executable table questions and independent file evidence in looped Qwen]
#v(4pt)
November 2026 · LoopQwen-MLX project · Complete technical report

= Abstract
#include "role-binding-abstract.typ"

#figure(image("role-binding.svg",width:100%),caption:[A complete meaning or table pair requires two correct answers to different questions. Each package inventory contributes one initial lookup and one dependent calculation.])

= Reading this report
This opening study tests a newly trained recurrent decoder. The second study tests interpolation of a previously trained tool adapter on a separate original Winograd panel. The main technical report that follows includes the matched induction controls, raw-language studies, native kernels, complete 128K-plus inputs, offline release and earlier failed deployments. The original 64 pages are preserved in full. The PDF bookmarks go directly to each study.

#pagebreak()
= Hypothesis and training
A name can occur in both variants of a sentence yet be the right answer to only one. Similarly, one table column can be the lookup key in one question and the returned field in another. Most of the supervised response tokens in a function call describe the common protocol. We tested whether increasing the training signal exactly where the required response diverges would help a shared layer resolve the two roles.

The candidate starts from the frozen Qwen3.5-0.8B instruction sibling in qualified offline release 05. The final attention layer runs twice with shared rank-32 adapters on four attention and three feed-forward projections. The two attention sites have independent full-history caches. There are 802,816 trainable parameters. We train a fresh adapter over the original donor, using a residual contribution of 0.35 on its second application. Serving fuses its full low-rank update in FP32 with one rounding to BF16. The same recurrent path processes every prompt and output token, with or without tools.

We selected 384 pairs of WinoGrande training sentences, 192 pairs of original WikiSQL training questions on distinct tables, and 160 existing native-tool rehearsal turns. Each table question supplied an executed call and a final answer conditioned on the actual return. Pair identities and complete tables were disjoint across training and evaluation. We excluded exact normalized question repeats and training pairs that shared eight consecutive original words with a test question. Demonstration targets used the official Qwen message template. The frozen first 23 layers produced token-level training features on Metal. Only training features were cached.

At each of 1,000 AdamW updates the model saw two randomly chosen pairs and two ordinary training turns. Standard response-token cross entropy trained the full vocabulary. At the first differing target token in each pair, each context also had to favor its own label over the sibling's label. If $d$ is the correct-minus-other logit, the additional loss for that context was $log(1+exp(1-d))$. We averaged it over both directions and applied a weight of 0.7. The learning rate was 0.00011, seed 219701 and weight decay 0.01. The observed fit consumed 163,358 supervised token positions, 168.40 seconds to prepare frozen features and 329.19 seconds for optimization. This supervision supplies alternatives only during fitting. On deployment the model freely generates the next token from its vocabulary.

= Human sources and field transfer
WinoGrande development pairs use the authors' original pair identifiers, the same two choices and opposing gold labels. We kept the human sentence intact, asked which phrase filled its blank, and fixed option order for both members of a pair. We selected complete pairs by hash. This is a subset of the labeled development split. It is distinct from both the unlabeled leaderboard test and the preceding study's original Winograd corpus. #link("https://github.com/allenai/winogrande")[Sakaguchi, Le Bras, Bhagavatula and Choi, WinoGrande.]

WikiSQL pairs use two original, independently phrased questions over one Wikipedia table, requiring different key/return-column roles and distinct answers. Eligibility requires one equality filter, an unambiguous row and headers, the actual key string in the human question, and a representable CSV field. We used sixteen new test tables. All 32 gold answers were independently reconstructed by executing the authors' queries on the original SQLite database. This addresses the selected lookup task. General SQL composition, joins and aggregation have separate evidence in the main report. #link("https://github.com/salesforce/WikiSQL")[Zhong, Xiong and Socher, WikiSQL.]

#pagebreak()
= Field protocol and decision rule
We exposed actual CSV files and a generic read-only keyed lookup to both models. Each independently chose the document, key column, literal key and value column. The study also enrolled the eight physical package inventories from the preceding supplement. Their sixteen questions form a separately reported transfer and retention panel. Across the eight pairs we downloaded and hashed all sixteen original, platform-specific wheels and measured *115,883,064 bytes* in total. The second question in each conversation required using the earlier observed size, looking up another file, and executing the difference on the same real calculator.

The ordinary BF16 Base and the trained K2 system received identical ordered opening tokens, native Qwen reasoning-enabled templates, a 768-token greedy allowance per turn and up to four model turns per user message. Subsequent histories contain that model's own words and tool observations. Initial model construction and weight fusion are charged to the first request. Decode rate excludes the first token of each model turn. Each dataset family reports completions, correct finals and operational successes separately. For free table answers the primary requires the annotated value as a bounded phrase and the gold lookup with its observed result. Exact grounded final text is a secondary safeguard against echoes or conflicting explanatory text.

We registered a single direct pair on 24 meaning pairs and 16 test-table pairs. Passing required more complete pairs in each domain, five or more net new complete pairs in aggregate and a two-sided exact paired test at 0.05. It required no decline in exact grounded table answers. Package transfer had to retain both the control's complete conversations and grounded answers, with a floor of two grounded answers. Every family had a completion and final-answer floor. Performance gates required strictly better total wall time, first-token time, MLX peak, opening rate and decode rate. The budget was 2,640 seconds, 7.8 GiB MLX peak and a host memory guard. No early model selection or test-time parameter sweep was used.

#tbl(csv("role-binding-costs.csv"),(2.5fr,1fr,1fr))
#tbl(csv("role-binding-argument-roles.csv"),(1.9fr,0.7fr,0.7fr,0.7fr))
The second table counts correct arguments in each question's first executed keyed lookup. Each component has its own numerator. Failed callbacks, incorrect literals and missing calls remain visible in the transcripts.

#pagebreak()
= Results, verification and limits
#include "role-binding-findings.typ"

The installed replay package contains the adapters, exact training and held-out material, all raw model and tool trajectories, original source archives, publication inputs and runtime wheel. Source checks independently execute original SQL and compare downloaded distribution files to their recorded lengths and SHA-256 hashes. The reader can examine the two models' answers and actual errors item by item in the supplied CSV. Paired times are from one native M3 Pro stream per arm. The loop ran first, so thermal history and run order limit generalization of small cost differences.

Both language corpora are public and could have appeared in either model's pretraining. A better score in an observed family establishes a system result on those items. The effects of instruction pretraining, adapters, precision, training objective and recurrent depth require other controls. The matched induction experiments and the separate 131K and longer field studies in the main report establish their own, narrower claims. Broad deployment requires independent new material, retention of previous capabilities, a matched trained one-pass model for architectural attribution, and actual full-length verification on the same candidate.

#pagebreak()
= Accuracy and time on the same request
#figure(image("role-binding-times.svg",width:92%),caption:[Each point is a paired user request. Below the dashed line the recurrent answer arrived sooner. Color and shape give the observed correctness outcome. Both axes use logarithmic seconds, including unsuccessful requests.])

The legend counts messages, so it can show which system was useful within a particular wall-time range. The decision test instead counts whole meaning and table pairs. The eight package inventories are an explicit held-over transfer panel. Some initial prompts exceed the lengths of ordinary one-line questions because they include native tool declarations. The first cold request also includes model construction.

All requests are available in the accompanying CSV. The compressed archive provides the next level of detail for reproduction, including every actual prompt after a callback, each emitted token, every validated call and tool return, and the native retained length for every physical full-attention site. The included reading sensitivity checks the visible replies against the original table answers. Further programmatic analysis can be performed without GPU access.
