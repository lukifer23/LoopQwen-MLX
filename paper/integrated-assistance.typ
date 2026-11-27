#set document(title:"Recurrent depth for grounded tools and open-ended assistance",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.8in,y:.72in),footer:context [#text(size:8pt,fill:rgb("#52675e"))[LoopQwen-MLX · Assistance and grounded reasoning] #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:10.5pt)
#set par(leading:.42em,spacing:.43em)
#show heading.where(level:1): it=>block(above:6pt,below:3pt)[#text(size:16pt,fill:rgb("#175f51"))[#it]]
#let d=json("integrated-assistance-paper.json")
#let grid(h,rows,widths)=table(columns:widths,inset:(x:4pt,y:2.25pt),stroke:none,fill:(x,y)=>if y==0 {rgb("#deece8")} else if calc.odd(y){rgb("#f1f5f3")},..h.map(x=>strong(x)),..rows.flatten().map(x=>[#x]))
#let cap(body)=text(size:9pt,fill:rgb("#42584f"),body)
#let ref(url, title)=link(url)[#title]
#let result=if d.qualified {"met both initial gates"} else {"missed the joint qualification rule"}

#text(size:24pt,weight:"bold")[Recurrent depth for grounded tools and open-ended assistance]
#linebreak()
#text(size:14pt)[Does one shared set of weights retain learned skills across tasks?]
#linebreak()
LoopQwen-MLX project · November 2026

= Abstract
Training small recurrent decoders on successive tasks produced complementary successes on tool calls, but also erased earlier abilities. We trained a new expert on human assistance, database conversations, causal and conversational reasoning, and observed software packages. We combined its parameter changes with three earlier experts and used the resulting fixed weights on every token of 220 requests. The paired ordinary-depth control was the pinned, untouched Qwen3.5-0.8B-Base. The loop solved #d.correct.at(0) of 208 objectively scored requests versus #d.correct.at(1) for Base. It gained #d.gains individual answers and lost #d.losses Base successes. On 12 open-ended user requests, identity-masked reading preferred the loop #d.open_win.at(0) times and Base #d.open_win.at(1) times. The loop #result. These results compare deployed systems with different training histories. A matched-depth experiment elsewhere in this project isolates a recurrent benefit on controlled induction.

#image("assistance-outcomes.svg",width:100%)
#cap[Figure 1. Left, correct answers delivered at a given per-request time allowance. Right, the number of database questions reaching each observed outcome. Schema inspection is optional. A query counts here when it returned rows on the requested database. The suite evaluates the model’s actual statement against every published alternate database for that schema.]

= Why count the entire interaction?
Producing a well-formed call is one decision. Selecting the right resource, retrieving relevant evidence, interpreting that evidence and delivering a supported answer are further decisions. Two SQL queries can return the same value on one database by coincidence. A word-overlap score can also prefer a short wrong answer to a longer correct one. We used executed queries on alternate database contents, source-level paired comparisons and a masked reading of every new open and database reply to examine these failure modes.

#pagebreak()
= Model and training objective
The deployed transformer has 24 distinct decoder layers. Layers 0–22 use the existing 8-bit instruction-sibling prefix. Both visits to layer 23 share its attention and MLP projections, read different incoming states and maintain separate attention histories. After the first visit, the second contributes 0.35 of its computed residual. All questions use the same model. The two visits execute for all prompt and generated tokens. The comparison uses BF16 Base at ordinary depth. Instruction post-training, numerical precision and recurrence all differ between arms.

The new expert changes seven rank-32 adapter pairs with 802,816 trainable parameters. Every optimizer update uses seven full source turns. Three are the schema request, actual query and original natural answer for the same CoSQL human question. The other four cover open assistance, reasoning, practical software and historical rehearsal. Draws come from shuffled decks. We include the first response position, first visible nonblank position and final token, then sample up to 48 positions per turn. Full causal context precedes every target. Seven native backward passes contribute equally before a single gradient-norm clip at 1. AdamW uses learning rate 0.000075 and weight decay 0.01.

For a gold response token let $l_1$ and $l_2$ denote full-vocabulary negative log-likelihood at its first and second visits. Minimize $l_2 + 0.15 op("softplus")(0.1 - (op("stop")(l_1) - l_2))$. The first-visit value supplies the local reference with stopped gradient. The shared parameters can still affect the first visit through the input to the second. The 1,000 updates sampled #d.targets positions across 2,156 turns. Native optimizer time was #d.fit_seconds seconds.

= Consolidation of four learned updates
For each projection and source expert we formed its scaled low-rank matrix product in FP32, then subtracted the corresponding product from the common typed donor. We retained the largest fifth of coordinates by absolute value separately for each source and projection, including cutoff ties. The sign of the sum of retained displacements wins. We averaged nonzero displacements with that sign. Exact cancellation produces zero change. We added the result to the common donor weight, rounded once and deployed the same seven compiled matrices for every request. Of #d.coordinates physical coordinates, #d.opposed had opposing retained signs. This construction follows the sign-resolution approach in #ref("https://proceedings.neurips.cc/paper_files/paper/2023/hash/1644c9af28ab7916874f6fd6228a9bcf-Abstract-Conference.html","Yadav et al., TIES-Merging").

#image("assistance-learning.svg",width:98%)
#cap[Figure 2. Left, training-time NLL before and after the second visit averaged over the seven groups. Right, the percentage of coordinates retained in the merged weight change and the percentage that had conflicting expert signs before election. These are training measurements.]

#pagebreak()
= Data and independent units
CoSQL contains real dialogues between a requester and a SQL analyst. We required two consecutive answerable questions with different single-cell results, verifiable expert statements and human replies that mention the value. We used 110 training dialogues after removing one entire dialogue whose opening question appeared verbatim in the held data. The 12 held dialogues use 12 different databases outside the training database partition. The requester’s original words and prior conversation were retained. This selection underrepresents requests needing clarification or multiple returned rows. #ref("https://aclanthology.org/D19-1204/","Yu et al., CoSQL (2019).")

OpenAssistant volunteers supplied the general requests and reference answers. We used disjoint training and validation trees, accepted English messages annotated as nonsynthetic, rank-zero replies and a minimum annotated quality of 0.66 on both messages. We retained the full selected questions and replies. A fixed hash sampled 256 training and 12 validation trees from 329 and 14 eligible trees. The topics include practical assistance, explanations, mathematics and creative writing. One crystallography reference incorrectly calls the defect a cation vacancy. The physical literature identifies an electron at an anion vacancy. We documented this before reading model answers. #ref("https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.184108","Physical Review B, 102, 184108 (2020).") Selection for short, well-rated answers limits extrapolation. #ref("https://arxiv.org/abs/2304.07327","Köpf et al., OpenAssistant Conversations (2023).")

The other 184 requests were previously exposed in evaluation. They measure retention on source-backed human reading, dependency metadata and existing workflows. We distinguish them from the 36 new conversations and database questions. Across the 2,152 distinct training opening sequences, none equaled any of the 220 held opening token sequences. A separate wording audit found one repeated original question across schemas. “How many teachers are there?” has the answer 15 in training and 7 in the held database. The registered primary retains both questions on that held database. Removing the entire database leaves 22 requests on 11 sources, with a net loop gain of #d.sql_overlap_net and source-paired p = #d.sql_overlap_p. Pretraining exposure to public data is unknown.

#grid(("Request family","N","Loop correct","Base correct","Loop done","Base done"),d.family,(1.8fr,.28fr,.7fr,.7fr,.65fr,.65fr))
#cap[Table 1. Quality and completion by family. “Correct” includes the grounding requirement for tasks that require tool use. Open requests use the complete-reply grades on page 4. There is no general exact answer key for them.]

= Query verification on changed worlds
We reran each model’s last successful query on the requested database when its table read and original one-cell result matched the reference. The unchanged emitted statement then had to match the expert denotation on every published alternate instance of that schema. Duplicates count, and the checker respects explicit ordering in the reference. Across 437 database instances, the 24 expert queries pass and 24 queries returning the correct original value as a constant all fail. The 437 instances are multiple tests of meaning. They are dependent and contribute no extra independent source units. The paired inference uses sign flips at the 12-database level. #ref("https://aclanthology.org/2020.emnlp-main.29/","Zhong et al., distilled test suites (2020).")

#pagebreak()
= Reviewing what the user actually received
A project-agent reviewer read all 36 new questions, their references and the complete visible replies. The two replies were digest-ordered independently for each question. Timing, identity, automatic scores and generation status were hidden. Identical visible answers appeared only once and necessarily tied. Judgments were committed before the mapping was opened. One project-agent reviewer supplied these judgments. Familiarity with earlier responses may make some styles recognizable even with labels hidden. An independent human preference estimate remains open. The archived sheet supports reannotation.

#grid(("Source","Visible-reply judgment","Loop","Base"),d.grades,(1.55fr,1.55fr,.6fr,.6fr))
#cap[Table 2. “Unsupported” counts replies with material unsupported claims. “Win” and “tie” give the paired preference. A reply may be useful and still contain an unsupported claim.]

The model grounded #d.sql.at(0) database answers versus #d.sql.at(1) and solved both questions on #d.whole_sql.at(0) databases versus #d.whole_sql.at(1). The two-sided whole-database sign-flip test of the per-source count difference gave p = #d.database_p. The 42 earlier successes set aside for retention came from the union of three adapters on 36 common requests and the earlier 20 successful Python compatibility checks. This candidate lost #d.prior_lost of #d.prior_total.

#grid(("Human reasoning","Loop mean token F1","Base mean token F1"),d.f1,(1.4fr,1fr,1fr))
#grid(("Cost over all 220 requests","Loop","Base"),d.costs,(2.1fr,.85fr,.85fr))
#cap[Table 3. Real single-machine times include the first actor load and tool callbacks. Opening throughput counts physically processed prompt tokens. Decode excludes each first generated token. Peak is the maximum MLX allocation observed per request. All open and failed requests contribute to cost.]

#pagebreak()
= Controls, related work and claim limits
The paired comparison fixed native thinking-enabled opening IDs, order, tools, unrestricted greedy decoding, three model turns per request and at most 384 generated tokens per turn. Both used the same Apple M3 Pro with 18 GiB of unified memory. The candidate ran first. Temperature, prompt, scorer and cap could not be selected after viewing outputs. The sequential ordering leaves a possible machine-state effect. The archive preserves actual output lengths and times, which distinguish finishing early from faster decoding.

Controlled induction in this project compared a recurrent and equally trained ordinary-depth decoder. The loop scored 62/64 versus 33/64, 62/64 versus 25/64 on fresh stories and 53/64 versus 25/64 after an independent seed. On 22 actual public tool requests a different recurrent system achieved 17/22 versus 0/22 for untouched Base and 17/22 versus 18/22 for its matched one-visit control. These studies motivate distinct claims for recurrence and for the complete deployed system. Conversion of pretrained transformers by sharing layers has also been studied at much larger uptraining budgets. #ref("https://deepmind.google/research/publications/122290/","Bae et al., Relaxed Recursive Transformers.")

On 36 common requests in four historical comparisons, same-cap Base replays reproduced every prompt and generated token. When the limit changed from 512 to 384, 20 of the 36 Base trajectories stayed identical in full. In the two 384-token studies, the median number of reasoning-channel words per request rose from 6 to 17 in the loop, while both loops still solved three requests. Enabling thinking and counting words measure different things from delivering a correct answer. In the current training material, 818 of 2,156 turns contain reasoning words, including 614 of the 640 reasoning examples. Across the 7,000 observed training draws, #d.reasoning_draws included a selected non-whitespace reasoning token. Such tokens accounted for #d.reasoning_targets of #d.targets gradient targets. All 660 database turns and 256 open human replies supervised calls or visible replies without an explicit reasoning target.

The qualified release 05 processed 131,084 actual input tokens and answered both changed-source requests correctly, versus neither for Base. On different full upstream lockfile revisions of 148,741 and 137,017 input tokens, both systems produced zero grounded final answers. Those timed trials used different weights from the current merged policy. The present primary provides evidence on tools, reasoning and cost at its observed lengths.

#grid(("Registered requirement","Outcome"),d.gates,(2fr,.6fr))
#cap[Table 4. Full promotion additionally requires independent-seed replication, equivalent single-visit training and physically measured 131,072-token changed-source requests on the same installed checkpoint. The primary has a 4,350-second and 7.25-GiB native ceiling.]

#pagebreak()
= Source-level variation and individual cost
#grid(("Family","Groups","Gains","Losses","Net pp","95% interval (pp)","Holm p"),d.intervals,(1.65fr,.5fr,.43fr,.48fr,.55fr,1.15fr,.55fr))
#cap[Table 5. Each gain is an individual request the loop solved and Base missed. Losses reverse the roles. The interval resamples complete registered source groups 10,000 times. The sign-flip p-values are two-sided and Holm-adjusted across the 12 displayed families. Both procedures assume exchangeability at the source level. Results for closely related dependency projects can share an ecosystem. Small panels and degenerate intervals limit what can be inferred.]

#grid(("Request time components (seconds)","Loop","Base"),d.phase_costs,(2fr,1fr,1fr))
#cap[The three components sum to measured wall time. Opening includes the first generated token on each tool or answer turn. The last row includes model construction on the first request, tokenization, callbacks and other work outside the generation timer. A shorter answer can reduce wall time at either decoding rate.]

Both models delivered complete answers on #d.both_finished.requests objective requests. Within these, the loop solved #d.both_finished.loop_correct versus #d.both_finished.base_correct for Base. This subgroup distinguishes improvements in completed answers from differences in how often each decoder finishes. Its membership depends on both systems. Across the 208 objective requests both models solved #d.joint.both_correct and both missed #d.joint.neither_correct. Among the #d.joint.individual_gains individual successes unique to the loop, #d.joint.additional_correct_and_all_three_costs_lower also improved all three per-request costs, including total time, first-token time and peak allocation. On the separate 12 open requests the exact two-sided sign test of non-tied preferences gave p = #d.preference_p. This describes one reviewer’s paired judgments.

#grid(("Cost unit","Loop lower / 220","Loop p50","Base p50","Loop p95","Base p95"),d.percentiles,(1.2fr,1fr,.75fr,.75fr,.75fr,.75fr))
#cap[Table 6. Elapsed and first-token times use seconds, allocation uses MiB. Percentiles refer to complete individual requests. The CSV also gives medians and tails of the within-request differences. A single sequential machine run establishes the measured costs on this stream. It supplies no estimate of hardware-to-hardware variation.]

The research program selected hypotheses using earlier experiments. Confidence intervals summarize variation across the enrolled source groups for fixed weights. A general deployment claim requires replication on new source material and a new training seed. The separate natural-language, software and physical results prevent a composite average from hiding a regression.

#pagebreak()
= What happened on two original user requests?
We selected these cases before seeing either model’s responses. One repeats a training question with different database contents. The other has an incorrect volunteer reference. Together they show how execution and independent source checks affect interpretation.

*Database question.* #d.examples.teachers.question The original training database returned 15 teachers. The held database contains seven. Grounding requires the last successful statement to read the table, match every alternate test instance and support the delivered answer.
#grid(("Observed evidence","Loop","Base"),(
 ("Executed SQL",..d.examples.teachers.queries),
 ("Visible reply",..d.examples.teachers.answers),
 ("Grounded",..d.examples.teachers.earned)
),(.68fr,1.16fr,1.16fr))

*Science question.* #d.examples.crystal.question
#grid(("Loop reply","Base reply"),(d.examples.crystal.answers,),(1fr,1fr))
#cap[The independent source check requires an electron at an anion vacancy, a formation mechanism and the effect of selective absorption on colour. The sealed review recorded the following rationale.]
#d.examples.crystal.reason

#image("assistance-completion-history.svg",width:95%)
#cap[Figure 3. Across five separate paired studies, split the total net gain into cases where both models delivered a visible answer and cases where only one did. Heights divide each net by the number of objective requests. Labels give the underlying net counts. The cohorts overlap and answer excerpts above preserve up to 380 characters. The downloadable files retain complete text.]

#pagebreak()
= Reproduction and data access
The source checkout contains the preregistration, preparation code, exact drawn training positions, held prompt IDs, judgments, raw model outputs, timestamps and complete tool events. The immutable study archive is #raw("research/results/integrated-assistance-01"). Its inventory has SHA-256 #raw(d.archive). The seven serving matrices have SHA-256 #raw(d.merged). Model inference, gradients and optimization ran through Apple Silicon MLX.

Install the environment from the pinned lockfile. Rebuild the original database dialogue and open-assistance selections with the source scripts below. The CoSQL database bytes and distilled instances are public downloads. Source paths and original licenses accompany their manifests. The result inventory records trained parameters and optimizer moments. Replay and scoring use the exact local databases. The paper build reads the sealed source and can be repeated without regenerating text.

#raw(".venv/bin/python -m scripts.dialogue_database_sources --acquire\n.venv/bin/python -m scripts.human_assistance_sources --acquire\n.venv/bin/python -m scripts.dialogue_database_semantics\n.venv/bin/python -m scripts.assistance_deliberation\n.venv/bin/python -m scripts.paper_integrated_assistance",block:true)

#grid(("File in paper/","Contents"),(
 ("assistance-requests.csv","Every prompt identifier, source, arm outcome, answer, cost and completion."),
 ("assistance-sql.csv","Actual statements and original expert SQL, alternate denotation match counts and final replies."),
 ("assistance-optimizer.csv","Every one of the 7,000 training views drawn, loss component, gradient norm and sampled-position count."),
 ("assistance-merge.csv","Seven physical projection sizes, retained source coordinates, sign conflicts and contributor counts."),
 ("assistance-visits.csv","Target, top prediction and NLL at preselected positions under one and two visits after consolidation."),
 ("common-reasoning-turns.csv","Observed reasoning-channel length on every turn of the four earlier 36-request comparisons."),
 ("research-scope.csv","Registered studies, outcome status and immutable file inventories.")
),(1.2fr,2.2fr))

= What the final visit learned during fitting
After compilation, both physical visits were probed on four predetermined original training turns from each of seven groups. At the #d.positions available opening, first-visible and terminal response positions, first-visit top-1 was #d.first_hits and second-visit top-1 was #d.second_hits. Their mean gold-token NLLs were #d.first_nll and #d.second_nll. These probes use teacher-forced training continuations. They diagnose whether a second native visit changes the predictive distribution under the deployed weights. Generalization is measured by the separate held generation above.

The complete volume follows this study with the earlier trials, mechanism controls, field comparisons, runtime tests, full-context trials and their limitations. Whole-source and per-request results remain attached to the checkpoint that produced them.
