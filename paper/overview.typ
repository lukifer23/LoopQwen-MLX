#set document(title:"Where shared depth helps a small language model",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.78in,y:.72in),footer:context [#text(size:8pt,fill:rgb("#596561"))[LoopQwen-MLX · Experimental overview · November 2026] #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:10.8pt)
#set par(leading:.47em,spacing:.46em)
#show heading.where(level:1): it => block(above:8pt,below:3pt)[#text(size:17pt,fill:rgb("#195f54"))[#it]]
#let d=json("overview-data.json")
#let t=d.scope.totals
#let tableview(head,rows)=table(columns:(1.52fr,1.08fr,.34fr,.49fr,.52fr,.68fr,.68fr),align:(left,left,right,right,right,right,right),inset:(x:3.5pt,y:3.5pt),stroke:none,fill:(x,y)=>if y==0 {rgb("#dcece6")} else if calc.odd(y) {rgb("#f0f4f2")},..head.map(y=>strong(y)),..rows.flatten().map(y=>[#y]))
#let note(body)=text(size:9pt,fill:rgb("#4b5855"),body)

#text(size:27pt,weight:"bold")[Where shared depth helps a small language model]
#linebreak()
#text(size:15pt)[Native Apple Silicon measurements of reasoning, executable tools and long-context inference]
#linebreak()
LoopQwen-MLX project · Research overview

= Abstract
Repeating one learned Qwen operation produced a large, reproducible gain on a controlled inference problem. A three-visit reader solved 62 of 64 held stories, compared with 33 for an equally trained one-visit reader. On 64 different stories the scores were 62 and 25. On executable tools, two visits completed 17 of 22 public tasks, against zero for untouched Base and 18 for a one-visit model with matched training. In the latest conversational study, the source-copy loop completed 130 of 144 presentations against 105 for Base in 110 versus 661 seconds. Across reading and finance, fully correct, supported delivery averaged 24.3% for the loop and 45.8% for Base. On two original Deno releases above 131,072 input tokens, a separate fit improved source-location loss by 4.78 nats. On 24 held field names, the trained two-read model and its parent each preferred the changed answer token in both releases on 18 names. One trained read did so on 20. A single model has yet to demonstrate the full combination of capabilities.

#image("overview-depth.svg",width:100%)
#note[Figure 1. Matched training at three visits versus one for the four induction rows and two visits versus one for public tools. The right panel uses measured elapsed time on each original pair. “New seed” refers to a new fit, while “new stories” evaluates the original fits on disjoint stories. They answer different reproducibility questions.]

= What changed in the model?
The common starting architecture is a 24-layer hybrid decoder with gated linear recurrence and full attention. A loop reuses a learned block at multiple depths. Each physical visit retains its own state across tokens. The induction reader shares its native Qwen operation across three latent updates. The tool assistant reuses layer 23 twice with a damped second residual and an eight-bit frozen prefix. Its instruction-trained sibling initialization adds a further difference from the untouched Base. All recorded model inference, gradients and optimizer updates use MLX on the M3 Pro with 18 GiB of unified memory.

#pagebreak()
= Accuracy beside elapsed time
#note[Table 1. Each line is a separate experiment. The denominator is the number of original items or requests in that row. Both time columns are sums of actual seconds on the complete ordered workload.]
#tableview(("Workload","Comparator","N","Loop ✓","Other ✓","Loop s","Other s"),if "conditional" in d {d.rows.slice(0,-1)} else {d.rows})

#image("overview-delivery.svg",width:100%)
#note[Figure 2. At each deadline, the numerator includes only correct final answers already delivered. Every enrolled request stays in the denominator, including unfinished generations and wrong answers. Each panel uses its own model and source cohort.]

In the induction task the model infers an attribute using other members of the same class. A follow-up changes class membership while preserving competing evidence. The loop answered both versions correctly on 32 of 32 pairs, compared with one complete pair for the one-visit control. In the public tool study an execution counts only when the model invokes the declared function and delivers the required final value. A correct number without the required observation can fail. The 17-to-18 depth comparison prevents assigning the entire public-tool gain to extra depth.

In the mixed source studies we re-execute local callbacks against supplied files, published package records or schedules. We also preserve unfinished generations and wrong observations. A model can finish a workload faster by emitting fewer tokens, including by abandoning hard questions. For this reason the detailed reports give completion, first-token wait, physical opening throughput, post-first decode rate and native peak alongside time and correctness. The CSV identifies the source archive and SHA-256 for every line in the table.

#pagebreak()
= What does 128K mean in practice?
A prompt of 131,080 actual tokens with a changed distant fact provides a direct test of full-history retrieval. The packed long-context decoder recorded all attention offsets, answered two of two cases and completed them in 171.88 seconds. Base answered neither in 391.67 seconds. The deployed tool assistant faces a harder demand. It must find a distribution version in original project data, execute a file measurement and communicate both values. Two consecutive upstream lockfiles supplied 148,741 and 137,017 input tokens. Their first occurrences of the requested version-bearing wheel entry were beyond position 131,072. Both models retained the histories, and neither returned a final answer. On the newer file Base generated the correct version inside its unfinished reasoning. On each file the loop repeated its plan.

The packed retrieval decoder and the instruction-derived tool decoder have different weights and execution routes. Within the tool decoder, speed also depends on request length. On the lockfile pair it decoded at 27.44 tokens/s against 26.88 for Base, but processed uncached input at 636 versus 741 tokens/s. Since most of the waiting occurred before generation, its complete request time was 56.51 seconds longer. The book conversation similarly exposed the difference between retaining a passage and answering a human reader. Neither arm completed its narrative questions. Two metadata requests succeeded for the loop versus three for Base.

= What transfers between trained adapters?
On 36 byte-identical retained requests at the same original 512-token budget, the valid-return adapter solved 13, the subsequent learner-history adapter solved 11 and untouched Base solved three. Between the two trained models there were eight individual gains and ten losses. Software decisions improved while several prior schedule and lookup answers disappeared. The full trajectories and corresponding unmodified Base generations make these regressions inspectable one at a time. Distinct fresh questions in the learner-history study produced only one source-grounded answer in 48 presentations.

A protected optimizer subsequently applied 735 of 800 proposed updates. Despite enforcing a loss tolerance on sampled older demonstrations, it retained none of the previous adapter’s eleven correct user requests and gained three different ones. Base answered eleven current-wheel questions correctly after checking the actual release, while the trained model answered twenty. A frozen exact-constraint scorer missed two correct Base answers, leading to a less conservative comparison. The detailed study reports both readings.

A readout trained on the actual second-visit change subsequently solved 38/112 requests versus 22/112 for Base. On two questions per whole software project it won 7/10 versus 4/10, and on four fixed-version swaps it won 3/4 versus 0/4. Human exact matches favored the loop 13/48 to 6/48. Source-masked review credited the loop on 8/24 human answers and Base on 16/24. The loop repeated its answer on nine of 24 backgrounds.

#if "assistance" in d [The consolidated two-visit assistant used one set of learned parameters on 220 requests. On the 208 objective requests it gained #d.assistance.gains individual answers and lost #d.assistance.losses Base successes. Its generated SQL grounded #d.assistance.sql.at(0) answers on 24 new questions versus #d.assistance.sql.at(1) for Base. An identity-masked project reviewer preferred the loop #d.assistance.open_win.at(0) times and Base #d.assistance.open_win.at(1) times on 12 open requests. The objective row in Table 1 includes only its 208 scored requests and their actual times.]

#if "four_layer" in d [The four-layer policy learned from its own executed outcomes and solved #d.four_layer.totals.at(0)/256 objective requests versus #d.four_layer.totals.at(1)/256. It gained #d.four_layer.wins individual answers and lost #d.four_layer.losses. The exact whole-question test on 24 changed tables credited #d.four_layer.whole.at(0) pairs to the loop and #d.four_layer.whole.at(1) to Base.]

These cohorts differ in author, difficulty and scoring unit. Repeated prompts establish retention across versions. New upstream lock revisions examine recent evidence that could not have been in the pinned checkpoint. Human-written datasets may already have contributed to the donor’s pretraining. Small paired cohorts give limited power to detect modest changes. One machine and an ordered run capture operational costs on that installation. Hardware variation, concurrent service demand and different user populations require new observations.

#pagebreak()
= Methodology and external evidence
Freeze the checkpoint identity, input token IDs, ordered requests, real tool interfaces, decoding budget, scorer and promotion thresholds before the direct comparison. Save both systems’ generated tokens and terminal statuses. Test source-level contrasts together when multiple questions share one source. For a mechanistic claim, match the training data, labels and optimization on an ordinary-depth model. For a deployment claim, compare costs on the complete request. A model receives long-context credit for the actual weights and cache path that processed the full prompt.

Published results provide useful comparisons at larger training scales. Geiping and colleagues trained a 3.5B recurrent-depth model on 800B tokens. Zhu and colleagues used large-scale shared-depth pretraining and a learned allocation of compute. The local adapters update a small fraction of a pretrained decoder on much smaller corpora. #link("https://arxiv.org/abs/2502.05171")[Geiping et al.] · #link("https://arxiv.org/abs/2510.25741")[Zhu et al.]

Marchenko and colleagues report controlled dense-to-loop conversion using learned input mixing and a smoothed exit loss. Sukthanker and Cameron compare small Qwen models with an equally trained dense twin and cheaper decoding alternatives. Their short full fine-tune of Qwen3.5-0.8B improved macro mathematics accuracy across four seeds, while each generated token cost more on their hardware. Both studies motivate the separate matched-depth and cost measurements here. #link("https://arxiv.org/abs/2610.00673")[Marchenko et al.] · #link("https://arxiv.org/html/2610.42030v1")[Sukthanker and Cameron, Tables 2, 3 and 10.]

ToolACE uses verified generated dialogues across a large API collection. Our public calls, distributions, metadata headers and current file returns provide a different check on whether a small assistant can act on a user’s actual resource. Authentic conversation questions and changed evidence help distinguish selecting a valid call from using the evidence to answer correctly. #link("https://arxiv.org/abs/2409.00920")[Liu et al.] · #link("https://aclanthology.org/D18-1241/")[Choi et al., QuAC.]

= Verification and artifacts
The complete registry contains #d.scope.registrations registrations. Its primary archives include #t.at("primary/passed") explicit passed decisions, #t.at("primary/missed") missed decisions and #t.at("primary/archived without boolean gate") without a Boolean release decision. Four recovered executions have separate provenance. We recomputed SHA-256 over #t.payloads archived payloads and compared their inventories to the registry. The separate closed-window ledgers contain #d.scope.closed_worker_windows.jobs sequential jobs across #d.scope.closed_worker_windows.windows windows, totaling #d.scope.closed_worker_windows.worker_hours worker-hours. This covers recorded windows only and includes host work inside each reservation. The archive count includes systems tests, source qualification and experiments with different criteria.

Read evidence-overview.pdf with overview-comparisons.csv for selected measurements and research-scope.csv for all original decisions, registrations and archive directories. The detailed paper includes methods, figures, ablations, archived failures and hardware measurements. The offline kits restore the model, runtime and sources and replay selected real requests outside the checkout. From the repository root, run #raw(".venv/bin/python -m scripts.paper_overview") to regenerate this overview and inventory.

#if "conditional" in d [
#pagebreak()
= The broader instruction-initialized model

#let c=d.conditional
The next four-layer model starts from Qwen's public instruction sibling, shares the last four hybrid layers and learns to combine the physical states. On the same 256 objective questions as the preceding policy it scored #c.objective.at(0) against #c.objective.at(1) for untouched BF16 Base, with #c.gains_losses.at(0) individual gains and #c.gains_losses.at(1) losses. It solved #c.tools.at(0) / #c.tools.at(1) of 154 tool-enabled requests and #c.ordinary.at(0) / #c.ordinary.at(1) of 102 ordinary objective requests. All 340 paired requests took #calc.round(c.costs.candidate.wall_s,digits:2) / #calc.round(c.costs.control.wall_s,digits:2) seconds.

#image("conditional-primary-quality.svg",width:100%)
#text(size:9pt)[Left, changes in objective success with intervals resampling the registered sources. Right, fully usable responses on all six masked primary categories. SQLite-backed Wikipedia answers also require two attested values in a genuine observed result.]

Whole questions test whether an answer follows its source. Both natural-table presentations were fully usable for #c.whole.natural.scores.at(0) / #c.whole.natural.scores.at(1) out of 24 questions. Both changed numeric tables received the correct grounded final answer on #c.whole_tables.at(0) / #c.whole_tables.at(1) out of 24 questions. The original supervised fit had generated 19 correct paired programs on 24 *training* questions entered from a supplied real schema.

The broad automatic criterion #if c.automatic [passed] else [failed] and the masked source criterion #if c.masked [passed] else [failed]. Quantization, instruction initialization and training differ alongside recurrence. The independent same-foundation depth control is needed to identify the loop's contribution. Historical 128K results belong to their measured checkpoints. The paired study provides complete per-request outputs, measured costs and source-level decisions.
]

#if "repair_query" in d [
#pagebreak()
= Two learned reads of the same source

#let r=d.repair_query
A later controller spends additional computation on each predicted output while retaining one key and value per original token at the final attention layer. It learns from human evidence, real report calculations and database histories containing both failed calls and plausible results to a different question. Its 721,920 trainable parameters are shared across two reads.

#image("repair-query-depth-probes.svg",width:100%)
#note[Figure 4. Each point is a complete training response. Positive values are lower reference-token loss when the learned controller makes its second read. Diamonds are within-family medians.]

The fit completed #r.fit.decision.updates updates over #r.fit.decision.positions supervised token positions. The extra read improved the reference likelihood of #r.depth.positive of #r.depth.n fixed training probes across #r.depth.groups groups. Mean group-balanced improvement was #calc.round(r.fit.decision.mean_second_visit_gain,digits:3) nats per token. Independent restoration reproduced all recorded probe losses at both trained depths. This comparison holds the weights fixed and measures local prediction under teacher forcing.

The same weights then processed #r.context.long.prompt tokens from a complete project lockfile and retained every row at the six attention sites through generation. Full-precision disk staging took #calc.round(r.context.long.first_s,digits:1) seconds to produce the first output. The four emitted positions consumed #r.context.long.read_only_visits recurrent reads. This test establishes capacity. The registered user workload contains four different full documents, 48 human reasoning topics, 32 financial issuers and 96 controlled tool histories.

#if "reading" in r [
  *Held user outcomes.* The full paired study contains 180 requests. The table shows exact registered successes, deliveries judged fully usable with their source, and individual gains and losses in exact success.
  #table(columns:(1fr,1fr,1fr,1fr),inset:4pt,[*Family*],[*Exact loop / Base*],[*Usable loop / Base*],[*Gains / losses*],..r.primary.counts.pairs().map(p=>{
     let k=p.at(0); let v=p.at(1); let a=r.reading.families.at(k,default:none)
     (k,str(v.total.at(0))+" / "+str(v.total.at(1)),if a==none {"—"} else {str(a.arms.candidate.grounded_usable)+" / "+str(a.arms.control.grounded_usable)},str(v.gains)+" / "+str(v.losses))
  }).flatten())
  Total measured wall time was #calc.round(r.primary.costs.candidate.wall_s,digits:2) seconds for the loop and #calc.round(r.primary.costs.control.wall_s,digits:2) seconds for untouched Base.
] else [
  The initial in-memory pair stopped on the first 152,185-token document before the two arms could be compared. The full staged comparison uses the same weights, prompt IDs, ordered requests and output budgets. The research paper records the incomplete attempt and its cost.
]
#if "bounded" in r [*Bounded source read.* On 48 new requests the loop completed 38 versus 28, taking 68.24 versus 265.78 seconds. It answered 13/24 reasoning choices versus 15/24, with seven supported explanations each. It executed valid calls on 23/24 financial reports versus one for Base. Two loop programs returned the correct amount and one delivered it correctly. A separate full-document gradient check reduced native peak allocation from 8.90 to 1.43 GiB at 147,481 positions. Held quality at 128K remains unverified for this checkpoint.] else if "fresh" in r [*Fresh correction fit.* After training on its own unfinished deliberation, the loop completed 37/48 versus 20/48 and took 57.93 versus 271.00 seconds. Correct reasoning choices were 9/24 versus 6/24, with six gains and three losses. Neither earned a fully usable explanation. Each had one strict calculation, and only Base delivered a fully usable financial answer with source units.]

]
