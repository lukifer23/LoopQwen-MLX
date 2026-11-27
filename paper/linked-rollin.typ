#set document(title:"Learning on the learner’s own tool histories",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.85in,y:.72in),footer:context [#set text(size:8pt,fill:rgb("#52635f")) LoopQwen-MLX · Grounded generation #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:10.6pt)
#set par(leading:.48em,spacing:.45em)
#show heading.where(level:1): it => block(above:8pt,below:3pt)[#set text(size:16pt,fill:rgb("#165f56"))#it]
#let m=json("linked-rollin-paper.json")
#let grid(headers,rows,widths)=table(columns:widths,align:(x,y)=>if x==0 {left} else {right},inset:(x:4pt,y:3.1pt),stroke:none,fill:(x,y)=>if y==0 {rgb("#deece8")} else if calc.odd(y) {rgb("#f2f5f4")},..headers.map(v=>text(weight:"bold",v)),..rows.flatten().map(v=>[#v]))
#let cap(body)=text(size:9pt,fill:rgb("#43554f"),body)

#text(size:25pt,weight:"bold")[Learning on the learner’s own tool histories]
#linebreak()
#text(size:13.5pt)[Two visits to a shared decoder layer on human questions, file evidence and real tool calls]
#linebreak()
LoopQwen-MLX project · November 2026

= Abstract
A tool-using model can recover the wrong field, guess a valid page, or correctly retrieve an article and still miss its answer. We trained a two-visit decoder on the histories its own generations reached. At each first mistaken visible token, the objective rewards the source-backed token over the one the student actually produced. We froze 24 human HybridQA questions in two presentations and 36 prior reasoning and tool requests. #m.abstract

#image("linked-rollin-outcomes.svg",width:100%)
#cap[Figure 1. Bar heights show the fraction of requests meeting all requirements. Counts above bars show successful requests. The first two groups each contain 24 requests. The remaining groups have six, six, eight, eight and eight. All answers and actual callbacks appear in the request CSV.]

= What improvement would look like
For a user asking about Valley Fields, the table connects the venue to Milwaukee. The assistant must find the city's linked article and identify the Catholic missionaries described in it. Supplying a city, repeating a cell, or issuing a plausible lookup leaves part of the task incomplete. A useful assistant should also keep answering earlier practical questions and should incur a tolerable cost. We compare both outcomes and actual wait on the same requests.

#m.lifecycle

#pagebreak()
= The project’s evidence across model versions
The aim is one deployable recurrent checkpoint that improves human reasoning, executable tools and resource use while retaining a usable context of at least 131,072 tokens. The table below traces the transition from logical judgments, through ambiguous table evidence, to open questions requiring an actual article. The same ordinary Base checkpoint appears throughout. Each row identifies a different recurrent fit and its corresponding paired study.

#grid(("Human source","Loop","Base","Paired p","Full stream, s"),m.systems,(1.52fr,.65fr,.65fr,.75fr,1.15fr))
#cap[Accuracy uses the human prompts in each row. Wall time covers the entire corresponding mixed stream, including practical cases, in loop / Base order. The two presentation modes of a human source stay together in the significance test. The held articles, tasks and training recipes change across studies.]

= How much did another visit contribute?
The historical depth controls fit both models with matched data, initialization and parameter budget. In the three-fact task, the recurrent arm visited its shared block three times. In the tool controls its final layer ran twice. The ordinary-depth controls in this table receive the same training as their corresponding loop. The ordinary Base in the table above receives no training.

#grid(("Matched depth study","Loop","One visit","Loop / one visit, s"),m.history,(1.85fr,.7fr,.85fr,1.35fr))
#cap[The two independent induction tests favor recurrence at nearly identical elapsed time. The smaller typed-tool panel favors one visit by one request. The four-panel tool difference is four requests (paired p = 0.388), with slower execution for the loop. Each row measures a different population.]

Release 05 separately answered the changed 131,084-token inputs and reproduced all 512 registered book next-token decisions in its installation checks. On two complete published books, the loop and Base each missed both natural-language questions. Across those book requests and four real follow-ups the loop achieved 2/6 versus 3/6 in 353.20 versus 319.32 seconds. This result specifies where successful long-context register retrieval has yet to transfer.

#pagebreak()
= The paired experiment
We used 128 original training questions and 24 original validation questions from Chen et al.’s HybridQA [1]. Every question uses a complete original table and its linked article collection. In the inline condition the user sees every table cell and link. The assistant can retrieve articles by their page IDs. In the file condition it receives two named CSV files and can interrogate both. The tool returns data from real files. The held file condition requires a table lookup that returns the correct hyperlink, followed by a separate call returning the actual article. An alternative unique key that retrieves the correct link qualifies. A completed, normalized exact answer supplies the final requirement.

Within each view the two models receive identical initial token IDs, generation policy, sequence of requests, tool declarations and files. Free greedy decoding uses up to 512 tokens per turn and three turns, with thinking enabled. The code records all intermediate generated IDs, calls, observed data, final text, elapsed time, time to first token and per-site cache offsets. The recurrent arm runs before Base. Only the first request loads the model in each arm. Local hardware and this fixed order affect the timing comparison.

The 36 retained requests include two old human tables in two formats, four pairs about package requirements, four pairs about scheduled events, and eight earlier lookup questions. The new human questions test transfer. Repeated earlier requests test whether the fit lost useful behavior. The precommitted register requires at least 12/48 successful new requests, eight net additions, whole-source paired significance, improvement in each view, at least three complete questions, improved token F1, retention of each earlier Base success, and practical gains on complete pairs. The resource gate requires improvements in first-token and total wait, peak allocation, prompt rate and decode rate. The full register specifies all conditions.

= Source identity and leakage
The original source selector used distinct publisher page identifiers. A subsequent bidirectional check discovered that two identifiers denoted the same complete Milwaukee article. The first 64 training conversations were archived and the run stopped before optimization and held evaluation. Following the same hash order, we replaced the entire offending training source with the next admissible original question and table. Every held file, question, token sequence and threshold stayed fixed. Training restarted from release 05.

The repaired split excludes overlap between each target article and all opposite-split pages by page ID, normalized article text, complete answer-bearing sentence and 80% coverage of distinct five-word sequences. The largest share of a held target’s five-word sequences in a single training page was 3.60%. These checks apply to the supervised study corpora. The contents of the original foundation pretraining corpus are unknown. An independent source reconstruction checked 152 original bundles, 304 rendered prompts and 456 expected public CSV callbacks before training.

#pagebreak()
= Recurrent training at an observed mistake
The candidate starts from release 05’s quantized instruction sibling with an eight-bit prefix and a shared rank-32 adapter on seven projections in layer 23. Layer 23 executes twice with separate full attention histories and a 0.35 residual coefficient on the second visit. The control uses the six ordinary full-attention histories in the certified, pinned, untouched BF16 Qwen3.5-0.8B-Base. Instruction initialization, quantization and fitting also differ between the candidate and Base. Historical equal-training depth comparisons are reported in the earlier paper.

Before optimization and again after update 400, the student freely generates up to three 192-token turns on 64 nonoverlapping training sources. We replay the emitted actions against their tools. When the teacher and student disagree, we identify the next source-backed lookup or answer at the observed history. A completed student reasoning sequence stays in the conditioning context. The loss supervises the first differing visible token. If the student did not finish its thinking, the source teacher supplies the full turn. This use of states visited by the learner follows the motivation of dataset aggregation [2].

Let $d_1$ and $d_2$ be the logit difference between the target and observed competitor after the first and second physical visits. The correction objective combines positive sequence loss, first-difference cross-entropy, and two margins. The first margin is $log(1+exp(1-d_2))$. The second is $log(1+exp(0.35+d_1-d_2))$, with the first-visit difference held constant for differentiation. Corrections, canonical calls, source-derived answers and protected older turns receive 40%, 20%, 25% and 15% of each update. Each of the 800 AdamW steps uses four actual training views. #m.fit #m.diversity

#image("linked-rollin-learning.svg",width:100%)
#cap[Figure 2. Left, terminal outcomes on the two separate 64-source training cohorts. “Turns” and “Tokens” mean that the respective limit ended generation. Right, sampled training losses from different batches. Changes across student cohorts are descriptive.]
#grid(("Step","Sources","Answered","Grounded","Corrections","Token contrasts"),m.phases,(.55fr,.7fr,.75fr,.78fr,.95fr,1.12fr))

#pagebreak()
= Where the assistant found its evidence
#image("linked-rollin-evidence.svg",width:100%)
#cap[Figure 3. The first correct action, all required observations in chronological order, exact final answer, and the joint primary score are measured separately. A correct answer can occur without the required observations. A correct first action can be followed by an unfinished response.]

#grid(("Source subset","Measure","Loop","Base","Paired p"),m.scores,(1.45fr,1.1fr,.65fr,.65fr,.8fr))
#cap[The primary analysis keeps all 24 sources and all official answers. Before training, a single reviewer marked 20 directly supported original questions, two with qualifications, and two whose phrasing exceeded the available evidence. The other rows test sensitivity to source support and published lexical variants. P-values use a two-sided sign flip of the whole question, keeping its two presentations together.]

Mean open-answer token F1 was #m.f1 for loop and Base respectively. Normalization removes punctuation and articles. Blank or incomplete answers score zero. The exact decision uses the original published answer. The variant analysis adds independently listed alternatives such as “Society of Jesus” for “Jesuit”. The source did not support an unqualified answer to an education question that inferred one soldier’s graduation record from a description of other soldiers in the same division.

= Practical retention
#grid(("Family","Loop","Base","Requests"),m.practical,(1.5fr,.65fr,.65fr,.85fr))
#grid(("Whole source or paired task","Loop","Base"),m.whole,(2.1fr,.65fr,.65fr))
Among 36 prior requests, the fit lost #m.lost individual Base successes. Whole-pair success requires both members of the same pair. The software tasks ask about interpreters on opposite sides of a declared requirement. The schedule tasks require a room check and a weekday derived from the retrieved date.

#pagebreak()
= What users actually received
Before observing model outputs we chose one presentation of each original question, balanced 12/12 by presentation. We stripped model identifiers and deduplicated identical responses to the same evidence. A single project-agent reviewer rated factual content with the complete source available, and sealed the ratings before inspecting aggregate held results. This procedure controls labeling order and model identity. It does not supply independent human annotation or inter-rater reliability.

#grid(("Blinded response category","Loop","Base"),m.reviews,(2.35fr,.65fr,.65fr))
#cap[The first five rows are mutually exclusive and sum to 24 appearances per arm. The last two are additional annotations or intersections with the observed source. A factually acceptable answer may still miss a required tool call.]

#for e in m.examples [
  #block(above:8pt)[#strong(e.at(0)) #linebreak()
  Published answer · #e.at(1) #linebreak()
  Trained loop · #e.at(2) #linebreak()
  Untouched Base · #e.at(3)]
]
#cap[Examples are the first source in each presentation from the frozen review sheet. Final text longer than 220 characters is abbreviated here. The CSV preserves it in full.]

= Timing and memory
#grid(("Metric for 84 requests","Loop","Base"),m.costs,(2.5fr,1fr,1fr))
A source-stratified paired bootstrap with 20,000 draws estimates uncertainty within this panel. The 95% percentile intervals for the ratio of summed waits were #m.intervals. Request grouping reflects the repeated presentations, opposing claims and shared source tables. The measurements come from one 18-GiB Apple M3 Pro.

#pagebreak()
= Time to a usable answer
Across the entire 84-request workload, including time spent on failures, the loop and Base delivered #m.delivery grounded answers per minute, respectively. #m.common The following fixed deadlines count only requests with a fully correct grounded answer. Each request starts its own clock. No time from a failed request is credited as a successful completion.
#grid(("Wait budget","Loop, solved / 84","Base, solved / 84"),m.deadlines,(1.15fr,1.2fr,1.2fr))
#image("linked-rollin-cost.svg",width:100%)
#cap[Figure 4. Empirical distributions across 84 matched requests. First and full response time include the cost of the text the model actually generated. Throughput and peak memory use the original native trace.]

#pagebreak()
= What changed in the installed runtime
An independent same-weight crossover on the qualified release 05 changed how intermediate prefill blocks are materialized. Evaluating only the physical cache state advances every recurrent site and preserves its history. A separate numerical probe compared every first-token logit and each retained convolution, delta and full-attention state. Sixty-four fixed-prompt runs preserved generated tokens and histories. In a second test, 32 separate installed commands exercised four real two-message conversations in counterbalanced blocks. Both settings generated 5,824 tokens and reused 28,672 prompt tokens. All tool responses and histories matched. #m.runtime

#image("cache-state-conversations.svg",width:87%)
#cap[Figure 5. Each crossover pair compares the two prefill implementations on the same release weights. Four positive values in each series indicate a shorter summed native turn-opening time for every matched block. The two series use distinct workloads and sampling units.]

For an installed-package follow-up, both settings read the correct MLX version and then repeated the NumPy version from the previous answer. The identical behavior and lower latency answer different engineering questions. This incident motivates direct training on a changed subject and independent protection of earlier successful tasks.

#pagebreak()
= Scope, depth attribution and reproducibility
The registered gates missed were #m.unmatched. Each measurement applies to the particular adapter, runtime and available tools. The promoted general-language reference and historical release remain identified by their own weight digests in the archive. These repeated public-source studies provide development evidence. A new source population, multiple fit seeds and a matched ordinary-depth fit are required to estimate broader effects and isolate recurrent depth. The historical induction experiments isolate a depth advantage on two panels. Matched tool-control experiments in the following chapters limit the architectural conclusions on natural tool use.

Earlier release 05 verified full physical histories on two 131,084-token changed-input requests and returned both correct values. A different trained reader recovered changed-file sizes from 173,129 and 191,099 tokens, while a separate repository-tree task failed in both arms. A new direct comparison fixes two adjacent lockfile revisions in which the critical declarations occur after token 131,072. Its registration is separate from the observed-history fit. Long context capacity, the ability to retrieve a changed fact, and the quality of the final action each require their own measurement.

*Reproduction.* Use the pinned dependencies in uv.lock. The source repair, full expected callbacks, masked review and optimizer exposures are under docs/measurements. The complete native tokens, model outputs, final optimizer moments and archive hashes are under research/results/linked-rollin-02. The full 84-row request table is paper/linked-rollin-requests.csv. The native comparison and decoder require Apple Silicon MLX. The publication, analyses, and reconstruction of source examples can be checked separately. Exact adapter SHA-256 #linebreak() #text(font:"DejaVu Sans Mono",size:7.5pt)[#m.sha] #linebreak() Held panel SHA-256 #linebreak() #text(font:"DejaVu Sans Mono",size:7.5pt)[#m.panel]

= References and original material
[1] Wenhu Chen et al. HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data. Findings of EMNLP, 2020. #link("https://aclanthology.org/2020.findings-emnlp.91/")[Paper.] #link("https://github.com/wenhuchen/HybridQA")[Original questions and tables.]

[2] Stéphane Ross, Geoffrey Gordon and Drew Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS, 2011. #link("https://proceedings.mlr.press/v15/ross11a.html")[Paper.]

[3] Shunyu Yao et al. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. ICLR, 2025. Tool execution and the completed user task use separate outcome measures in our studies. #link("https://proceedings.iclr.cc/paper_files/paper/2025/file/1b126cc38b8638e07bef37e7b2bb72bf-Paper-Conference.pdf")[Proceedings.]

[4] The Qwen team. Qwen3.5-0.8B and Qwen3.5-0.8B-Base. The training and control revisions are separately pinned in the registration. #link("https://huggingface.co/Qwen/Qwen3.5-0.8B-Base")[Base model card.]

[5] The MLX team. MLX, a framework for machine learning research on Apple Silicon. #link("https://github.com/ml-explore/mlx")[Implementation.] The accompanying historical chapters detail the original training, baselines, matched depth, long-context verification, and the preference, inference and tool experiments.
