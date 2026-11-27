#set document(title:"Can a four-layer recurrent language model answer from a changed table?", author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.73in,y:.7in),footer:context [#text(size:8pt,fill:rgb("#51635c"))[LoopQwen-MLX · Direct four-layer comparison · November 2026] #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:10.7pt)
#set par(leading:.42em,spacing:.47em)
#show heading.where(level:1): it=>block(above:7pt,below:4pt)[#text(size:16pt,fill:rgb("#155a4c"))[#it]]
#let d=json("table-policy-paper.json")
#let tab(head,rows,cols)=table(columns:cols,inset:(x:4pt,y:2.7pt),stroke:none,fill:(x,y)=>if y==0 {rgb("#dceae5")} else if calc.odd(y){rgb("#f0f4f2")},..head.map(strong),..rows.flatten().map(x=>[#x]))
#let note(body)=text(size:9pt,fill:rgb("#43574e"),body)
#let showreply(x)=if x=="" [No visible reply.] else [#x]
#text(size:23pt,weight:"bold")[Can a four-layer recurrent language model answer from a changed table?]
#linebreak()
#text(size:14pt)[A matched comparison of tool use, human reasoning and measured cost]
#linebreak()
LoopQwen-MLX project · Native Apple Silicon MLX · November 2026

= Abstract
A Qwen3.5-0.8B-Base decoder reuses its final four native layers with separate causal histories. Training first fitted supervised turns, then optimized returns from the model’s own sampled database actions and visible answers. We compared its frozen policy with the original untouched BF16 Base on 268 requests using identical opening token IDs, unconstrained greedy generation and equal output limits. The loop delivered *#d.totals.at(0) of 256 grounded objective answers*. Base delivered *#d.totals.at(1)*. Pairing individual requests identifies #d.wins gains and #d.losses losses. A changed-table test required the same question to be answered correctly after the table contents changed. The original exact scorer credited #d.whole.at(0) of 24 whole questions to the loop and #d.whole.at(1) to Base. Source-visible masked reading credited #d.semantic_whole.at(0) and #d.semantic_whole.at(1), respectively. The combined release requirement #if d.passed [passed.] else [was missed.] This report includes each missed threshold and preserves the full native trajectories.

#image("table-policy-performance.svg",width:100%)
#note[Figure 1. The left panel counts objective successes among requests whose individual elapsed time is at most the horizontal value. It describes the latency distribution of correct answers. The right panel shows all 268 individual first-token times, including cold model construction on each arm’s first request.]

#tab(("Objective cohort","Requests","Loop correct","Base correct","Loop time (s)","Base time (s)"),d.toolplain,(1.25fr,.68fr,.83fr,.83fr,.9fr,.9fr))
#note[The additional twelve open questions enter timing and masked preference assessment. Their answers need original-request judgments without a fixed exact-match target.]

#pagebreak()
= Study design and comparison
The released Base text checkpoint has 24 physical layers. The candidate quantizes its embedding and prefix through layer 19 to four bits. It preserves the BF16 tensors of layers 20 through 23. It executes those three native Gated DeltaNet layers and one full-attention layer twice. Each second application has its own physical history. Four trainable rank-128 residuals and an initially zero channel gate determine how the repeated result contributes to the ordinary representation. No instruction-donor weights or external teacher entered this four-layer fit. The ordinary-depth comparator uses the pinned, untouched BF16 checkpoint. Quantization, supervised learning, execution learning and extra depth all differ between these two deployed systems.

The supervised stage made 1,024 optimizer updates on 3,101 complete training turns. All 32 predetermined training probes improved in full-vocabulary likelihood. The subsequent 192 native updates produced 1,280 sampled trajectories. Forty-two questions yielded nonzero policy gradients from their own sampled outcomes. The loop made 253 valid paired table reads in 256 query attempts and returned zero correct SQL scalars in training. After a known full-table read it delivered 25 original and 24 altered correct answers across 512 samples per version. An independent replay regenerated every one of 388 sampled token IDs. The two appended methods papers document feature parity, loss, reward, sampling and supervision exposure.

Each model saw the same ordered 268 opening requests and tool declarations. Every opening token was frozen before training. Both could reason freely, call the same real handlers and generate up to three turns with 384 tokens per turn. Subsequent inputs depend on the model’s own actual calls and tool results. Every emitted token, stop, error, observation, native allocation and elapsed time is retained. Measured request time includes model initialization on the first request, tokenization, callbacks and generation. The recurrent arm ran first on one Apple M3 Pro with 18 GiB of unified memory. CPU-side data preparation continued during this sequential run, with one native model worker. Small timing differences require replication under controlled load.

= Exact whole-question and individual results
#tab(("Changed source","Loop correct","Base correct"),d.versions,(1.6fr,1fr,1fr))
#tab(("Outcome across 48 presentations","Loop","Base"),d.execution,(2.8fr,.65fr,.65fr))
#note[Each source is an actual Wikipedia table paired with a changed table and a distinct human-agreed answer. The question, filename, schema and opening IDs stay identical. A correct completed answer must follow a read. A model can correctly interpret a multirow result without producing scalar SQL. Conversely a correct SQL scalar alone earns no delivered-answer credit. Each query is replayed on both original database files in the companion execution table.]

The two-sided exact sign-flip probability across the 24 independent question pairs was #d.p under the registered whole-question score. The separately masked, source-visible whole-question reading gave #d.semantic_p. Counting both versions as independent trials would overstate the amount of evidence. The 220 earlier requests measure retention. The first presentations of these specific 48 changed-table requests supply the new transfer test.

#pagebreak()
= Which capabilities improved?
#tab(("Request family","N","Loop correct","Base correct","Loop ended","Base ended"),d.family,(1.72fr,.45fr,.85fr,.85fr,.8fr,.8fr))
#note[Grounded scoring uses the registered criterion of each task. An ended turn must produce a complete visible answer. The open cohort has no binary automatic correctness score. A successful tool call can still lead to a wrong answer.]

#tab(("Human reasoning, mean token F1","Loop","Base"),d.f1,(2.5fr,.65fr,.65fr))
#tab(("Physical quantity, same full stream","Loop","Base"),d.cost,(2.5fr,1fr,1fr))
#note[Opening throughput divides actually processed prompt tokens, after within-conversation prefix reuse, by summed generation time to the first token of each turn. Decode throughput counts emitted tokens after the first per turn. Maximum allocation is measured by MLX. Cost distributions for individual requests and the generation/callback breakdown are published as source CSVs. Fixed generation limits allow both models equal opportunity to finish. Actual output lengths are model outcomes.]

Of the #d.prior_n previously identified successes, the candidate missed #d.prior_lost. On #d.joint.both_correct objective requests both models were correct. On #d.joint.neither_correct both missed. The candidate gained #d.joint.individual_gains individual correct answers, including #d.joint.additional_correct_and_all_three_costs_lower with simultaneous reductions in request time, time to first token and native peak allocation.

On the 208 objective requests shared with the preceding integrated-assistance model, the new loop solved #d.previous.scores.at(0) and the preceding loop solved #d.previous.scores.at(1). The new model gained #d.previous.new individual successes and lost #d.previous.lost. Untouched Base reproduced its complete answer, token, tool and attention-history trajectory on #d.previous.identical of the 220 repeated requests. These are comparisons across two separate runs.

#pagebreak()
= How much do source-level effects vary?
#image("table-policy-source-effects.svg",width:100%)
#note[Figure 2. Each dot is the paired percentage-point difference in grounded success. Whiskers are percentile intervals from 10,000 resamples of whole registered source groups within each family. Results condition on the particular trained model and enrolled material. A degenerate interval can arise when no group contributes a different outcome.]

#tab(("Request family","Sources","Gains","Losses","Net pp","95% interval, pp","Holm p"),d.intervals,(1.55fr,.63fr,.53fr,.53fr,.65fr,1.15fr,.65fr))
#note[Family probabilities use a two-sided exact sign flip of group-level differences followed by Holm adjustment over the displayed family comparisons. Sources sharing the same ecosystem may still be dependent. No composite score compensates for a regression. The preregistered whole-table endpoint is separate from these descriptive family results.]

#pagebreak()
= What did the source-visible reviewer credit?
#tab(("Request family","Arm","Fully usable","Partial","Wrong","Absent","Unsupported","Preferred"),d.masked,(1.4fr,.55fr,.79fr,.58fr,.59fr,.58fr,.89fr,.76fr))
#note[A project reviewer graded the full visible replies for all 48 changed tables, 24 database requests and 12 original open questions. Reply order was independently hashed per request and identical replies collapsed. The reviewer committed every grade, material-unsupported flag, comparative preference and request-specific reason before revealing identities or inspecting aggregate scores. Wording and characteristic model errors can still suggest identity. This is one reviewer. The source sheets and commitments support replication by independent readers.]

The loop lost #d.lost_semantic individual changed-table answers that the reviewer assessed as fully usable for Base. Every presentation was graded against its own actual source. The twelve open questions include original volunteer replies as context for the reviewer. Those references can contain errors or unverifiable claims. A documented source error can change what a correct response should say.

= Registered release decision
#tab(("Gate evaluated before promotion","Status"),d.gates,(2.5fr,1fr))
#note[Required conditions include whole-question transfer, earlier individual successes, all three human reasoning margins, per-family completion, grounding, preference and five physical measurements. Passing a training check demonstrates a gradient and a changed model. The direct pair supplies separate evidence about delivered capability.]

#pagebreak()
= Cases from the original requests
#note[One case per source type was selected by fixed hash before aggregate results were read. The complete request and masked-rating sheets contain the full replies. The excerpts below show up to 210 characters per response.]
#for e in d.examples [
  #block(above:8pt)[*#e.id*]
  #text(size:9.5pt)[#e.question]
  #tab(("Arm (masked slot)","Visible response"),(([Loop (#e.slots.at(0))],showreply(e.replies.at(0))),([Base (#e.slots.at(1))],showreply(e.replies.at(1)))),(1fr,3fr))
  #text(size:9.5pt)[*Masked rationale.* #e.reason]
]
= Scope, reproducibility and prior work
The current policy’s generalization evidence comes from one direct pair on one machine. Attributing a product advantage to extra depth requires the registered independent seed and an equally exposed ordinary-depth learner. Verification of a different earlier release on 131,084-token inputs does not establish this checkpoint’s long-context answer quality. A qualifying policy must use the same released weights on complete changed-source inputs beyond 131,072 tokens.

The policy digest is #raw(d.policy.slice(0,32))#linebreak()#raw(d.policy.slice(32)). Recheck all actual callbacks and scores with #raw(".venv/bin/python -m scripts.research_archive table-policy-gradient-01 --verify"). Rebuild the masked score, CSVs, figures and this edition with #raw(".venv/bin/python -m scripts.paper_table_policy"). The archived result inventory has SHA-256 #raw(d.archive.slice(0,32))#linebreak()#raw(d.archive.slice(32)).

The appended studies give the full methods and citations. Source tables and denotations follow Pasupat and Liang (ACL-IJCNLP 2015, ACL 2016). The optimizer uses the leave-one-out estimator studied by Ahmadian et al. (ACL 2024). CoSQL follows Yu et al. (EMNLP 2019). The original Qwen Base and instruction sibling have distinct upstream weight identities. #link("https://aclanthology.org/P16-1003/")[Changed tables] · #link("https://aclanthology.org/D19-1204/")[CoSQL] · #link("https://aclanthology.org/2024.acl-long.662/")[RLOO] · #link("https://huggingface.co/Qwen/Qwen3.5-0.8B-Base")[Original model]

#pagebreak()
= What can these experiments establish about recurrence?
Geiping et al. trained a language model with a recurrent block so inference can vary the amount of latent computation. Their scale and training objective differ substantially from a retrofit of a fixed small hybrid model. Our four-layer experiment freezes one count of native applications before evaluation. Its result describes a particular deployed decoder and budget. It does not measure a curve relating answer quality to increasing latent depth. #link("https://arxiv.org/abs/2502.05171")[Geiping et al., NeurIPS 2025]

Bae et al. study conversion of existing language models to recursive models, using layer-wise low-rank adaptation to relax otherwise tied weights. Their results distinguish comparisons at a similar physical parameter count from recovery of the original larger model. Here the same unmodified four physical native layers process two different states. The added residuals act on second-visit activations. The two models in our direct pair differ in quantization and training exposure as well as recurrence. A fitted ordinary-depth control and independent seed are required to distinguish those effects. #link("https://arxiv.org/abs/2410.20672")[Bae et al., Relaxed Recursive Transformers]

= Why source changes and open questions both matter
Pasupat and Liang use changed tables to identify programs that obtain a correct denotation on an original table for accidental reasons. We apply their source intervention to an assistant that must issue its own real tool calls and deliver the answer. A whole-question success requires both versions. Prose reading credits meaningful equivalent answers and records unsupported extra claims. The earlier studies in this edition showed that a gain in exact matches can coexist with worse source-visible answers. #link("https://aclanthology.org/P16-1003/")[Pasupat and Liang, ACL 2016]

The training archive exposes a gap between a valid database read and a useful computation. Independently annotated SQL from SQUALL provides a subsequent way to supervise column choice, predicates and sort direction. An independent remount found 149 human programs returning the target on both training tables and 176 executable alternatives returning an incorrect same-type value on both. FeTaQA provides complete human-written multifact answers grounded in full Wikipedia tables. The separate source audit identified one conflicting year and four qualified references among 24 fresh pages. These sources belong to a proposed successor. They supplied no evaluated answers or training updates to the measured model in this paper. #link("https://aclanthology.org/2020.findings-emnlp.167/")[Shi et al., Findings of EMNLP 2020] · #link("https://aclanthology.org/2022.tacl-1.3/")[Nan et al., TACL 2022]

= Scope of the long-context evidence
An earlier released policy answered two opposite cases with 131,084 real input tokens. On a separate pair of whole upstream lockfiles with 148,741 and 137,017 tokens, both earlier systems missed the changed version and byte count. Successful allocation and physically retained attention history alone cannot establish that a user receives the right fact. The fresh qualification uses complete English and Italian CLDR locale documents from two releases. The supplied inputs range from 152,185 to 164,271 tokens and the governing pattern occurs beyond the first 131,072 tokens. The relevant value must be interpreted and delivered. This four-layer primary contains no long-context observation on its own policy digest. The source inputs and qualification criteria are separately frozen.

The consolidated edition includes the prior task and context measurements in chronological research appendices. Results in those appendices concern different trained weights unless the policy digest is explicitly the same.
