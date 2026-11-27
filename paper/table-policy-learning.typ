#set document(title:"Learning from native sampled SQL and changed human denotations",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.76in,y:.75in),footer:context [#text(size:8pt,fill:rgb("#52675e"))[LoopQwen-MLX · Execution-conditioned training] #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:10.8pt)
#set par(leading:.45em,spacing:.45em)
#show heading.where(level:1): it=>block(above:7pt,below:4pt)[#text(size:16pt,fill:rgb("#175f51"))[#it]]
#let d=json("table-policy-learning-paper.json")
#let c=d.counts
#let tbl(head,rows,widths)=table(columns:widths,inset:(x:4pt,y:3pt),stroke:none,fill:(x,y)=>if y==0 {rgb("#deece8")} else if calc.odd(y){rgb("#f1f5f3")},..head.map(x=>strong(x)),..rows.flatten().map(x=>[#x]))
#let qual=if d.passed {"passed"} else {"missed"}
#let note(body)=text(size:9pt,fill:rgb("#45584f"),body)
#text(size:24pt,weight:"bold")[Can the loop learn from the consequences of its own tool calls?]
#linebreak()
#text(size:14pt)[Sampled actions and answers on original and changed Wikipedia tables]
#linebreak()
LoopQwen-MLX project · November 2026

= Results
The trained four-layer loop generated #d.draws trajectories covering #d.tokens tokens over #d.steps optimizer updates. Four independent samples received different returns on #d.varied questions. A nonzero policy gradient reached the shared decoder’s trainable parameters on #d.positive_pg questions. At least one generated action or answer obtained a human-labeled denotation on #d.correct_groups questions. The run #qual the prespecified training prerequisite. The native worker used #d.seconds seconds and reached #d.peak_gib GiB peak allocation. #if d.interrupted [The fixed budget stopped execution. Archived findings below cover the durable updates. Recorded reason: #d.reason.]
#image("table-policy-learning.svg",width:100%)
#note[Figure 1. Sixteen-update bins, with the two training conditions shown separately. Each question contains four jointly scored draws. Question difficulty and the trained policy both vary over time. The second panel counts gradients from the sampled return alone, before adding the fixed supervised anchor.]

#tbl(("Sample condition","Draws","Generated tokens","Natural stop","Closed thought","Visible text"),d.categories,(1.35fr,.65fr,1fr,.85fr,.9fr,.85fr))
#note[Table 1. A natural stop is an actual terminal token. A visible reply requires both a natural stop and a closing reasoning marker. Empty or length-limited replies still contribute to sampled-token cost.]

= Do actual SQL statements track the changed data?
Of #d.query_count sampled SQL trajectories, #c.valid_sql produced a valid, untruncated read of the named table in both worlds. #c.any_gold_sql returned a correct scalar in at least one world, and #c.both_sql returned the human value in both. Each statement ran unmodified on two distinct mounted databases with the same schema and filename. A query that returns an entire table can earn the small read reward. Correct scalar denotations receive separate credit.

#pagebreak()
= Do answers follow the observations?
For the #d.pair_count answer draws in each world, the loop returned #c.original_correct correct answers on original tables and #c.altered_correct on altered tables. It copied the other world’s gold answer #c.original_opposite and #c.altered_opposite times respectively. Both answers were correct for #c.both_answers paired draws. In #c.identical_nonempty pairs the loop produced the same nonempty visible text for the two different observations.

= Sampling, training and verification
Every third update began after a genuine schema callback and let the model write a query. Other updates began after a genuine read of all rows in each version. The source question, schema and database name remained identical. The source annotators supplied different correct values. The independent sample keys, all model-emitted token IDs, per-token behavior log probabilities, database outputs and native optimizer moments are preserved. This use of known training contexts makes a response to the observed data measurable. A field trial additionally requires the assistant to initiate each call and manage its own conversation.

Sampling used the original 248,320-token vocabulary, temperature 0.8, four trajectories, and a maximum of 128 tokens per turn. For a sampled SQL action the reward adds 0.1 for an untruncated read of the relevant table in both worlds, 0.2 for each correct single-world denotation and 0.5 when both are correct. For an answer, each world contributes 1 for a correct visible answer and −0.25 for the other world’s answer. The paired return averages the two and adds 0.5 when both are correct. For each of four samples the leave-one-out advantage subtracts the mean return of the other three. The summed token loss is divided by the fixed token cap. A separate supervised earlier-tool turn contributes weight 0.35 on every update. These distinct gradient sources are recorded individually before clipping the combined gradient. [1–3]

The median across contributing updates of their mean absolute per-token log-probability discrepancy between the sampled cached path and the full-sequence training path was #d.drift. Native BF16 intermediates and kernel selection can differ across these paths. Independently regenerating the prespecified draws at the original pre-update weights and random seeds reproduced #d.repeat_matched of #d.repeat_tokens archived tokens. An independent full-vocabulary normalization gave maximum log-probability discrepancy #d.max_err. The native replay scope and per-trajectory fingerprints are recorded separately.

#if d.conditions.len()>0 [#tbl(("Registered requirement","Observed decision"),d.conditions,(2fr,1fr))]
#note[The training prerequisite establishes whether an authentic and sufficiently frequent execution-derived learning signal was present. Gains on user requests, attribution to depth, and support for 131,072-token changed-source tasks each require their own comparison.]

#pagebreak()
= Worked executions
The examples below use the first query that read both tables for each observed number of correct worlds. They come from training. The complete draw table contains unsuccessful and repeated attempts.
#if d.examples.len()==0 [No sampled statement read both tables.]
#for e in d.examples [
  #block(above:8pt,below:5pt)[*#e.source · #e.correct_worlds of two denotations correct* #linebreak() #e.question]
  #raw(e.sql,block:true)
  #tbl(("World","Human gold","Result from the actual query"),(("Original",e.gold.at(0),e.obtained.at(0)),("Altered",e.gold.at(1),e.obtained.at(1))),(0.7fr,1.15fr,2.4fr))
]
#if d.paired_example!=none [
  #let e=d.paired_example
  *A changed answer after a full read* #linebreak()
  The first paired draw to answer both versions correctly occurred at update #e.step, draw #e.draw (zero indexed). #e.question
  #tbl(("World","Source answer","Model’s visible answer"),(("Original",e.gold.at(0),e.visible.at(0)),("Altered",e.gold.at(1),e.visible.at(1))),(0.7fr,1.15fr,2.4fr))
]
= Sources and reproducibility
1. Pasupat and Liang. _Inferring Logical Forms From Denotations._ ACL 2016. The source archive preserves the original WikiTableQuestions tables, changed tables, original questions and independent annotator responses. #link("https://aclanthology.org/P16-1003/")[Publication and source context]
2. Ahmadian et al. _Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs._ ACL 2024. A leave-one-out baseline reuses the other independently generated samples for the same prompt. #link("https://aclanthology.org/2024.acl-long.662/")[Method and experiments]
3. Pasupat and Liang. _Compositional Semantic Parsing on Semi-Structured Tables._ ACL-IJCNLP 2015. #link("https://aclanthology.org/P15-1142/")[Original question source]

The experiment identifier is #raw("table-execution-learning-01"). From the project root, run #raw(".venv/bin/python -m scripts.research_archive table-execution-learning-01 --verify") to retokenize training contexts, rerun the actual SQL, reproduce the rewards and advantages, and audit the native optimizer states. Run #raw(".venv/bin/python -m scripts.replay_table_policy_draws") to repeat the independent native sampling check. Run #raw(".venv/bin/python -m scripts.paper_table_learning") to rebuild this report and its three source tables. All calculations use the sealed training archive, SHA-256 #raw(d.archive.slice(0,32))#linebreak()#raw(d.archive.slice(32)).
