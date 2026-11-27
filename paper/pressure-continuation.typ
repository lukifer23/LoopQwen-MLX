#set document(title:"Training recurrence under memory pressure and testing human tool requests",author:"LoopQwen-MLX")
#set page(paper:"us-letter",margin:(x:.73in,y:.68in),numbering:"1")
#set text(font:"Libertinus Serif",size:10.4pt)
#set par(leading:.44em,spacing:.54em)
#let d=json("pressure-continuation.json")
#let fmt(x,digits:2)=str(calc.round(x,digits:digits))
#let tbl(cols,head,rows)=table(columns:cols,inset:4.5pt,stroke:.4pt+rgb("#b9c4c5"),..head.map(x=>[*#x*]),..rows.flatten())
#text(size:23pt,weight:"bold")[Training recurrence under memory pressure and testing human tool requests]
#v(.35em)
*Qwen3.5-0.8B on an 18 GiB Apple M3 Pro* Research report · 27 November 2026

*Abstract.* Can one recurrent model outperform the ordinary Base on tools, reasoning, cost and long-context use? The last completed direct study compared 340 identical requests. The trained loop gained 49 objective answers and lost 25. It answered 17 of 48 masked reasoning requests fully versus 28 for Base. On four held human questions its generated SQL remained correct on 22 additional independent table versions. This supplement follows the next candidate from its original optimizer state and examines how to measure whether an assistant can recognize an incorrect result from a functioning tool.

*Product comparison.* The prior loop used the instruction sibling of the Base checkpoint and four-bit frozen weights. Its repeated four-layer suffix has a separate causal history per visit. The product measurements compare that complete system with untouched ordinary Base under identical thinking-enabled input tokens and greedy budgets. Quantization and instruction initialization each change the comparison. An equally trained ordinary-depth control and independent training seed are required to establish a broad causal contribution from recurrence.

#tbl((2.1fr,.85fr,.85fr,2.5fr),("Earlier direct measure","Loop","Base","Unit and population"),(
("Objective accuracy","66 / 256","42 / 256","Completed grounded answers under the registered scorer"),
("Tool-enabled accuracy","42 / 154","24 / 154","Subset of the objective item population"),
("Masked reasoning","17 / 48","28 / 48","Fully usable source-based answers"),
("Whole changed tables","4 / 24","0 / 24","Both original presentations correct, exact paired p = 0.125"),
("Wall time","354.30","1,727.84","Seconds over the complete 340-request stream"),
("Time to first token","60.12","52.82","Sum of first-token delays in seconds"),
("Opening throughput","2,361.85","2,776.64","Opening tokens per second"),
("Decode throughput","143.50","69.42","Post-first generated tokens per second"),
("Peak allocation","843.30","1,848.56","Native MLX peak in decimal MB"),
))
*Context.* An earlier installed checkpoint processed 131,084-token requests. A different two-request lockfile study processed full 137K- and 149K-token inputs but neither model delivered the required final result. The source-conditioned candidate must process four complete 152K–164K-token files and answer from evidence beyond token 131,072. No shorter example can establish that it passed this requirement.

*Reading guide.* The next page reports only closed training attempts. The third page reports source validation before any generation on its 96 controlled tool histories. The accompanying volume contains architecture diagrams, original comparisons, masked judgments, uncertainty and failure analyses. The repository includes every recorded model output and callback for the historical comparisons.

#pagebreak()
#text(size:19pt,weight:"bold")[Completing an identical optimizer schedule]
#v(.28em)
*Model and loss.* The learner takes the actual hidden-state differences from an extra visit to four shared Qwen layers. A trainable residual uses those differences at the output. The frozen training features contain all five physical states. The objective combines next-token cross entropy, forward KL from the teacher and a softplus preference term over labeled differences. The lower-memory implementation divides full-vocabulary computations into 48-position tiles. For a preference spanning tiles it computes the whole preference margin before applying its derivative to each participating position. It retains every changed position.

#tbl((1.6fr,.85fr,.85fr,.9fr,1.05fr,1fr),("Attempt","Updates to date","Native s","Peak GiB","Labels to date","Preferences"),d.stages.map(r=>(r.name.replace("-world-fit-01",""),str(r.updates),fmt(r.seconds),fmt(r.peak),str(r.targets),str(r.preferences))))
*Accounting.* Each row starts from the preceding certified weights, AdamW moments and ordered update history. Add the elapsed seconds across rows. Updates, labels and preferences are cumulative. The full schedule requires 3,392 updates, #d.target positive labels and 28,322 preference presentations. Comparing the first two runs measures different parts of the schedule. Their peak allocations do not estimate a paired speedup. At the last archived stage, #d.latest.updates updates had durable optimizer state. #if "status" in d.latest.decision [Its recorded stop reason was “#d.latest.decision.reason.”]
#image("pressure-continuation-optimization.svg",width:100%)
#text(size:9pt)[Fig. 1. Measured means in consecutive blocks of up to 64 completed updates. The denominators for response NLL and KL are positive target tokens. The third panel divides by preference presentations. Each inherited update is counted once.]

*Verification.* On three fixed real batches of 32, 49 and 83 positions, the tiled and original full-vocabulary calculations used the same weights and labels. Across archived preflights the largest relative difference in the loss components was #fmt(d.objective_relative,digits:9), the largest relative gradient difference was #fmt(100*d.gradient_relative,digits:5)% and the lowest gradient cosine was #fmt(d.cosine,digits:7). These checks establish agreement at the sampled points. Small rounding differences can accumulate in training.

*Decision.* Incorrectly predicted tokens never count as useful tool calls. Training qualification also requires unassisted programs on at least eight of 24 complete training questions with at least six distinct successful statements. Three of 12 complete paired qualitative cases and three of 12 service pairs must succeed. The registered direct comparison uses 148 matching requests per arm, actual elapsed times, all four full documents and masked review. Only a qualifying checkpoint can enter it.
#if "free_whole" in d.latest.decision [
*Observed free generations.* SQL #d.latest.decision.free_whole.operator / 24, qualitative #d.latest.decision.free_whole.qualitative / 12 and service #d.latest.decision.free_whole.service_reply / 12. The fit produced #d.latest.decision.free_distinct_sql distinct successful SQL statements. The joint fit decision was #if d.latest.decision.passed [pass.] else [miss.]
]

#pagebreak()
#text(size:19pt,weight:"bold")[Can the assistant tell whether a successful tool call answered the question?]
#v(.2em)
*Why this matters.* In the earlier changed-table evaluation the loop received a plausible wrong scalar on 27 requests and repeated it in 19 final answers. It recovered to a grounded final answer on zero. On 20 requests with an actual tool error and remaining turns, none of its subsequent calls succeeded. Base recovered a successful call on 19 of 41 such requests. Their error opportunities differed. On eight common opportunities the corresponding recovery counts were zero and five.

*Training material.* We reconstructed actual database errors and source-confusable successes using original CoSQL and WikiTableQuestions data. CoSQL contributes #d.training.at(0).turns supervised turns after crossed-database errors. Exact CSV keys and wrong column roles contribute #d.training.at(1).turns and #d.training.at(2).turns. A further #d.training.at(3).turns use paired human questions where precisely the same SQL and returned scalar require opposite behavior. An independent replay verified 3,360 callbacks in those later histories. Earlier fitting data contained 8,083 histories with prior tool returns and zero with a structured provider error. The new histories exercise a measured gap. Their construction provides no estimate of model improvement.

*Independent evaluation material.* We reserved two independently authored questions per schema, each with a nonempty expert scalar that appears in a human reply. Four conditions per question present an unassisted request, a correct completed query, a successful query that answers the partner’s question, or a real failed query from a different database. Paired successful histories are byte-identical after the user message. The 10 official development questions span five databases. Another 14 questions come from seven databases in the upstream training split excluded from our earlier CoSQL cohorts. The pretrained checkpoints’ exposure is unknown. The 96 requests contain only 24 human questions and 12 distinct databases.

#image("provider-field-denotations.svg",width:85%)
#text(size:9pt)[Fig. 2. Executing the human SQL on #d.fields.dev.alternate_instances alternate database instances changed the expected return for every development question at least once. In #d.fields.dev.distinct_partner_results instances the two human programs disagreed. The seven separately withheld training-split schemas have original databases only.]

*Annotation limitations.* A source audit flagged #d.flags of the 24 questions before model testing. The published response to a customer-count question is 28, the number of invoices. Counting distinct customers returns four. A request naming exhibitions has a reference answer of 13 attendance rows representing five exhibitions. A reference counts all students in response to a question about graduates. These examples require separate reporting of expert-SQL reproduction and reasonable answers to the human request. All enrolled cases and alternative SQL remain available for inspection.

*Reproduction and related work.* The repository includes original byte hashes, frozen prompt IDs, expert SQL, the separately interpreted queries and per-database return values. Rebuild the tool sources with #raw("python -m scripts.provider_field_denotations") and #raw("python -m scripts.provider_field_wording"). Distilled test suites check executable meaning over alternate database contents. We use the authors’ published instances on schemas available in their release. #link("https://aclanthology.org/2020.emnlp-main.29/")[Zhong, Yu and Klein (2020)] · #link("https://aclanthology.org/D19-1204/")[CoSQL, Yu et al. (2019)].
