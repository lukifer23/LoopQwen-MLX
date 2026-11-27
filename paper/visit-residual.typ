#set document(title:"Testing what a second transformer visit contributes",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.84in,y:.73in),footer:context [#text(size:8pt,fill:rgb("#51645f"))[LoopQwen-MLX · Second-visit experiment] #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:10.6pt)
#set par(leading:.44em,spacing:.44em)
#show heading.where(level:1): it=>block(above:7pt,below:3pt)[#text(size:16pt,fill:rgb("#165f56"))[#it]]
#let d=json("visit-residual-paper.json")
#let grid(h,rows,widths)=table(columns:widths,inset:(x:4pt,y:2.6pt),stroke:none,fill:(x,y)=>if y==0 {rgb("#deece8")} else if calc.odd(y) {rgb("#f2f5f4")},..h.map(x=>strong(x)),..rows.flatten().map(x=>[#x]))
#let cap(body)=text(size:9pt,fill:rgb("#40574f"),body)

#text(size:25pt,weight:"bold")[Testing what a second transformer visit contributes]
#linebreak()
#text(size:14pt)[Human causal questions, dependency decisions and a learned readout of recurrent state]
#linebreak()
LoopQwen-MLX project · November 2026

= Abstract
Previous training improved answers about Python packages but lost all eleven previously successful requests on a common retention panel. We froze the two physical visits of a recurrent Qwen decoder and trained three small matrices to read the change made by its second visit. The primary test compares this candidate to untouched ordinary-depth Qwen Base on 112 requests. Each causal question has a closely related original question about the same situation but a different answer. Each software question requires a real wheel-metadata read and a judgment about a published dependency. The trained candidate #d.decision under the joint accuracy, preservation and cost rule. Its signed difference in successful requests was #d.net_correct out of 112.

#image("visit-residual-performance.svg",width:100%)
#cap[Figure 1. Each arm has its own actual per-request elapsed times. Left, number of successful requests deliverable within a given latency. Right, all terminated requests, including errors and token limits. Timing includes the first model load and all actual callbacks.]

#grid(("Requests","N","Loop correct","Base correct","Loop done","Base done"),d.quality,(1.65fr,.34fr,.75fr,.75fr,.75fr,.75fr))

#pagebreak()
= A readout tied to the second visit
The original 0.8B instruction decoder uses an eight-bit prefix and a repeated final attention layer. Its two visits share weights and have independent physical key and value caches. Let $x$ be the hidden state after the first visit and $y$ the state after the second visit with its 0.35 residual coefficient. For each token the trainable readout returns
$ y + op("RMS")(y) U[op("silu")(V n(x)) ⊙ D n(y-x)] , $
where $n$ is root-mean-square normalization with a small stabilizer and the product is elementwise. Each of the two input maps projects from 1,024 to 256 features. The output map projects back to 1,024. When the two visits agree, the correction is exactly zero. Initializing the output map to zero reproduced every one of the donor's 248,320 logits on the registered training position. The donor itself differs from untouched Base in instruction training, quantization and depth.

Native MLX cached first- and second-visit states for 2,072 full training turns and optimized only 786,432 parameters. We used 1,600 AdamW updates, learning rate 0.0002, global gradient norm limit 1.0 and weight decay 0.01. Every batch included both original questions from one background, two package turns and three examples from different earlier task families. On each example we used at most 24 response positions, retaining boundaries and the first token that distinguished the correct visible answer from its paired alternative. Every drawn position and its parent input identity is archived.

The objective assigns weights of 0.45 to new-source cross entropy, 0.15 to a smooth gold-over-foil margin, 0.20 to older-task cross entropy and 0.20 to full-vocabulary forward KL from the frozen donor on rehearsal positions. The new-source cross entropy itself weights all positions at 0.35 and visible positions at 0.65. Across training there were 232,243 sampled response positions and 85,739 full-vocabulary reference projections.

#image("visit-residual-training.svg",width:100%)
#cap[Figure 2. Left, actual pre-update losses. Right, the same 96 training endpoints measured with the unmodified donor, the trained readout and a substitution of the other question’s change on the same background. These endpoints were selected before fitting.]

#grid(("Training intervention","Mean margin","Mean gold NLL","Gold > foil","Gold top 1"),d.probe,(1.65fr,.85fr,.9fr,.8fr,.8fr))
Holding the first and second hidden states fixed while replacing only the change supplied to the readout altered the margin. The actual change exceeded the swapped change at #d.learned_gt_swap endpoints. Its mean margin advantage was #d.mean_learned_swap logits. These training endpoints test whether the learned function uses question-specific differences. Held free generation measures whether the behavior transfers.

#pagebreak()
= Human situations and changing software evidence
The ROPES authors collected human-written situations and questions about the effects described by a background passage. The source train and development partitions share 42 normalized background paragraphs. We selected disjoint sets of 192 training and 24 evaluation backgrounds and checked situation, question and paragraph similarity across them. In each selected situation both questions are original and their answers differ. Normalized exact match is the registered binary endpoint. We also report overlap F1 and a source-informed reading of complete visible answers. Whole-background sign flips on the paired exact scores gave a two-sided probability of #d.human_p. #link("https://aclanthology.org/D19-5808/")[Lin et al., 2019.]

The new software material comes from twenty training and ten held package projects, selected from different upstream lockfiles. We inspected each original wheel’s Requires-Dist records and used genuine released candidate dependency versions. Four more pairs hold the requested Sphinx version fixed while changing which package or package revision the user asks about. Those four share one dependency ecosystem. Each successful answer requires an actual matching tool return, the exact current constraint and the correct compatibility judgment. Bold delimiters are removed before parsing the constraint under a rule fixed before evaluation. The literal reading is preserved in the released scores.

#grid(("Source unit fully solved","Loop","Base"),d.whole,(2.5fr,.7fr,.7fr))
For human situations, project decisions and fixed-version swaps, a complete unit requires both correct answers. The older five families use their archived source groupings. Reusing those cases measures retention. The older panel lost #d.prior_lost individual Base successes under the new checkpoint.

= Control, limits and registered decision
Each arm received identical opening IDs, tool definitions, item order, reasoning-enabled template, greedy decoding and three turns of up to 384 tokens. When the model requested a tool, its own chosen arguments produced the observation. Call errors, missing answers and output limits remained in the denominator. Candidate then Base ran serially on one Apple M3 Pro with an 18 GiB unified-memory configuration. The resource limits were 4,350 seconds, 7.25 GiB peak native allocation and 1.5 GiB available physical memory.

The registered quality thresholds include eight net additional human exact answers, two-sided paired significance at 0.05, at least six whole human situations, at least two extra whole situations and an F1 gain of 0.08. Software requires at least four whole independent projects and two whole fixed-version swaps, each exceeding Base. The remaining criteria cover earlier individual Base successes, earlier whole practical pairs, completion within each family, total time, first-token time, peak memory, opening rate and decode rate. The unmet gates were #d.gates.

#pagebreak()
= Reading complete answers and measuring cost
Before the comparison, we chose both questions from twelve held human situations and all eight fixed-version wheel questions. One project-agent reviewer read the original situation or actual dependency header and the visible answer. Model identity, automatic score and timing were hidden. The reviewer sealed the ratings before reading the primary outcomes. Identical visible answers to the same prompt received a common rating. Unsupported claims can coexist with partially correct answers.

#grid(("Source","Masked judgment","Loop","Base"),d.masked,(.8fr,1.8fr,.7fr,.7fr))
#grid(("Across all 112 requests","Trained loop","Untouched Base"),d.cost,(2.3fr,1fr,1fr))

A correct response must survive the full interaction. Lower total time can reflect shorter successes, early errors or shorter thinking. The output counts, completion counts and correct-by-deadline curves allow these explanations to be examined together. First-token sums include the model load on the first request. Uncached opening throughput counts only tokens the model had to process. Decode throughput excludes the first generated token of each turn.

#pagebreak()
= Deployment and independent verification
The installation experiment used the same release-05 weights in both arms and exercised four actual two-message workflows in alternating crossover order. The extended prefix cache preserves more of a completed tool turn for later decoding. Thirty-two cold commands delivered 64 requests. All token streams, tool arguments, observations, terminal states and recorded physical cache histories were identical across cache modes.

#image("extended-prefix.svg",width:100%)
#cap[Figure 3. Every point represents an observed pair of four-workflow rounds. Lower is better for time and allocated memory. Higher is better for decode rate.]

Native turn opening fell from 8.480 to 7.790 seconds. Total cold response time fell from 89.860 to 88.686 seconds. The optional cache reused 2,048 extra tokens across the schedule and wheel workloads. Peak native allocation increased by 3.50 MiB. The memory requirement closed default promotion, so the extension remains opt-in. Both modes emitted 5,824 tokens. These measurements are a separate equal-weight runtime comparison.

An additional user-facing SQLite provider supports inspecting relationships and issuing read-only queries. We built the runtime wheel twice from archived source, installed it outside the checkout with networking disabled and verified 246 original CoSQL expert queries across 76 published databases. Seventy queries read multiple tables. To qualify future model judgments we executed the 24 held expert queries against every distributed instance for the 12 held schemas. Across 437 instances and 874 executions, each question’s answer changed at least once. Every reference query passed the resulting suite, whereas a SELECT returning its correct original answer as a constant passed zero of the 24 suites. This is verification of the tools and prospective scoring method. #link("https://aclanthology.org/D19-1204/")[Yu et al., CoSQL.] #link("https://aclanthology.org/2020.emnlp-main.29/")[Zhong et al., distilled test suites.]

= Reproduction and scope
The archive at research/results/visit-residual-01 preserves raw prompt IDs, generations, callbacks, scores, learned weights, AdamW moments, training draws, software identities and the registered decision. The project’s archive verifier reconstructs the sampling process, checks all saved tensor shapes and replays responses against original resources. The paper’s CSV files contain all paired requests, all training updates, the 96 interchange measurements and all 32 installed crossover commands. Rebuild the tables and figures with scripts.paper_visit_residual.

The first evaluation attempt stopped at tokenizer construction before any held generation. After normalizing the checkpoint path, the runner resumed from the completed training state using the same weights, example order, scorer and remaining budget. The archive records both execution windows.

The 128K results elsewhere in this volume belong to their named checkpoints and tasks. A claim about the present readout requires additional measurements on these exact weights. The directly compared Base is the untouched pinned Qwen3.5-0.8B-Base. Repeated depth, separately trained components and quantization can each affect the measured difference. A causal depth attribution requires a matched one-visit training run.
