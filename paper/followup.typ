#set document(title:"Updating a recurrent answer when the evidence changes",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.85in,y:.72in),footer:context [#text(size:8pt,fill:rgb("#51645f"))[LoopQwen-MLX · Conversation and tool evidence] #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:10.7pt)
#set par(leading:.48em,spacing:.45em)
#show heading.where(level:1): it=>block(above:8pt,below:3pt)[#text(size:16pt,fill:rgb("#165f56"))[#it]]
#let d=json("followup-paper.json")
#let grid(h,rows,widths)=table(columns:widths,inset:(x:4pt,y:3pt),stroke:none,fill:(x,y)=>if y==0 {rgb("#deece8")} else if calc.odd(y) {rgb("#f2f5f4")},..h.map(x=>strong(x)),..rows.flatten().map(x=>[#x]))
#let cap(body)=text(size:9pt,fill:rgb("#40574f"),body)

#text(size:25pt,weight:"bold")[Updating a recurrent answer when the evidence changes]
#linebreak()
#text(size:14pt)[Human follow-up questions, real Python wheel metadata and protected optimizer steps]
#linebreak()
LoopQwen-MLX project · November 2026

= Abstract
An installed assistant correctly read the MLX version after a user switched packages, then repeated its previous NumPy answer. A later training run gained eight older answers and lost ten. We use two complementary forms of supervision to address these behaviors. A two-visit decoder learns to distinguish the current evidence from a plausible earlier answer, while each optimizer step is checked against sampled older tasks. #d.abstract

#image("followup-results.svg",width:100%)
#cap[Figure 1. Left, success under the original family-specific scoring rule. Each point on the right-hand curve is the observed full request latency for one model. The figure includes failed and unfinished generations.]

= Accuracy and completion
#grid(("Source and task","N","Loop correct","Base correct","Loop done","Base done"),d.overview,(1.9fr,.38fr,.8fr,.8fr,.75fr,.75fr))
Both models received identical initial token IDs, genuine thinking-enabled turns and up to three unrestricted 384-token generations. Trained instruction initialization and prefix quantization differ from untouched Base in addition to repeated depth. The report's accuracy comparison measures the complete candidate.

#pagebreak()
= What was asked and how it was scored
The QuAC collection contains real conversations between a questioner and an annotator who can see the text. We selected two consecutive questions from each of 128 original training titles and 24 different validation titles. We preserved the full section and original background. The second prompt includes the original first question and its human answer, fixing the same history for both models. Five of the 48 held questions have majority abstention. The authors' scoring procedure averages maximum token overlap after leaving out each reference. We reproduce that calculation across every selected question, including low-agreement items, and compare our scorer with the original program. This test measures responses to an authentic conversation history. It does not observe how errors accumulate when each model provides its own first answer. #link("https://aclanthology.org/D18-1241/")[Choi et al., EMNLP 2018.]

The practical questions come from original releases pinned in two independently maintained projects. Training uses 48 wheels from Pydantic's dependency lock. Evaluation uses 12 different project wheels from FastAPI's lock. The prior package's requirement is true and differs from the requested package's requirement. On each question two complete original wheels are available through the same public function. We ask for the exact current Requires-Python range and a yes/no answer on one actual boundary interpreter version. A grounded success requires reading the correct header of the correct wheel before completing both parts. This establishes declared version compatibility. Installation may also depend on platform tags and other dependencies. #link("https://packaging.python.org/en/latest/specifications/core-metadata/")[Core metadata specification.]

#grid(("Whole source or pair","Loop","Base"),d.whole,(2.4fr,.9fr,.9fr))
An exact title-level sign-flip test treats the first and second human questions as one unit. Passing requires eight net extra human successes and two-sided p ≤ 0.05, a ten-point increase in follow-up F1 and a gain in first-question F1. Four complete human dialogues and four complete actual archive pairs must each exceed Base. The 36 prior requests are retention tests. The full gate requires preserving every prior Base success, gains on both earlier practical pair types, no lower completion by family, and favorable latency, allocated memory, prompt and decode rates. The maximum was 4,050 s and 7.25 GiB. The complete registration is archived as followup-protected-01.

= What the masked reviewer read
We selected one question per held title, alternating between first and second questions in a frozen title order. A project-agent reviewer saw the original full text, all human annotations and the visible model response. Identical answers to identical questions received one rating. Model identity and aggregate scores were concealed until the judgments were sealed. This supplementary review has one rater. The source scorer and decision thresholds were fixed separately.

#pagebreak()
= Protecting earlier responses during adaptation
The final native attention layer and its seven rank-32 projection updates execute twice. Each visit has its own attention history. The second visit uses a 0.35 residual coefficient. Each optimization step uses one human answer, one action selecting a wheel and one answer interpreting its metadata. Two other randomly sampled training turns supply the protected reference loss. The active objective rewards the demonstrated sequence and increases the correct token's margin over the earlier alternative at the first visible difference. A stop-gradient first-visit margin supplies the reference for the second visit.

The active and rehearsal gradients enter AdamW with weights 0.85 and 0.15. Let $g$ be the rehearsal gradient and $v$ the proposed parameter displacement after adaptive scaling and weight decay. When $v dot g > 0$, subtract $g (v dot g)/(g dot g)$ from the proposal. Then recompute the actual rehearsal loss on the same two examples, trying full, half, quarter, eighth and sixteenth steps. If none meets the tolerance of 0.00005, restore the original parameters. Optimizer moments advance once. At least 50 proposals in every block of 100 must be accepted. This extends the reference-gradient idea in A-GEM with a check on the proposed optimizer displacement. The check applies to two training responses at a time. Held behavior requires direct measurement. #link("https://arxiv.org/abs/1812.00420")[Chaudhry et al., ICLR 2019.]

#d.fit

#image("followup-learning.svg",width:100%)
#cap[Figure 2. On the left, a positive value means the second visit increased the target’s logit margin over the specific previous answer or tool. The plotted values come from pre-update training computations. On the right, update categories can overlap.]

#grid(("Training decision","Draws","Favors gold only after","Favors foil after crossing","Mean margin change"),d.residuals,(1.15fr,.55fr,1.12fr,1.12fr,1fr))
A margin crossing measures two candidate tokens under the demonstrated prefix. Free generation still competes against the entire vocabulary. Every row in followup-training-margins.csv carries the original view identity and two observed margins.

#pagebreak()
= Grounding, earlier answers and cost
#grid(("Held observation","Trained loop","Untouched Base"),d.masked,(2.4fr,.9fr,1fr))
The first six rows cover the 24 masked evaluations per arm. A tool read counts any actual observation from the corresponding archive. Both files may be read on one request. Final constraint and verdict require an actual completed response. Success additionally requires grounding in a prior tool turn.

#grid(("Metric on 108 requests","Trained loop","Untouched Base"),d.costs,(2.2fr,.9fr,1.1fr))
The model lost #d.lost individual successes previously achieved by Base on the 36 retained requests. Registered criteria that missed were #d.failed. Execution order was fixed. Both arms ran on one Apple Silicon M3 Pro. Generation speed is conditional on the text each model actually produced. Peak allocation is the maximum MLX reading. Repeated tool schemas and prefix reuse are accounted for in the uncached prompt rate.

= Verification and scope
The repository preserves both source cohorts, every held prompt, 216 complete trajectories, actual tool observations, optimizer moments and the independent model-masked ratings. Rebuild the numerical table and figures with scripts.paper_followup. Verify every source-indexed score and native history with scripts.research_archive. The optimizer uses Apple Silicon MLX for every differentiated forward and update. Equal-training ordinary-depth comparison and same-weights long-context qualification each require their own passed prerequisite. Source licensing, the original wheel hashes and public corpus versions are in research/sources. Adapter SHA-256 #linebreak() #text(font:"DejaVu Sans Mono",size:7.5pt)[#d.sha]
