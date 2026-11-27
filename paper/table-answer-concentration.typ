#set document(title:"When different tables produce the same answer",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.8in,y:.78in))
#set text(font:"Libertinus Serif",size:11pt)
#set par(leading:.48em,spacing:.55em)
#let d=json("table-answer-concentration.json")
#let c=d.paired.candidate
#text(size:23pt,weight:"bold")[When different tables produce the same answer]
#linebreak()
#text(size:14pt)[A source and transcript audit of the four-layer loop]
#linebreak()
LoopQwen-MLX project · November 2026

The recurrent assistant made 46 successful complete table reads on 48 held requests. Its 46 SQL statements were identical. It returned the literal #raw("4") on 45 requests. Only one of the human-agreed held answers was four. The candidate received every row of *both* different tables on #c.both_read of 24 source pairs. On #c.unchanged_after_two_reads of those pairs it delivered identical nonempty text despite distinct observations. On #c.both_four_after_two_reads it returned four in both worlds.

#image("table-answer-concentration.svg",width:100%)
#text(size:9.5pt)[Figure 1. All sampled or requested answers enter the denominator, including empty results. “Human labels, sampled contexts” repeats each gold label once per actual sample. Sampling used temperature 0.8. The field comparison used ordinary greedy decoding.]

#table(columns:(2.55fr,.85fr,.8fr,1.15fr),stroke:none,inset:4pt,fill:(x,y)=>if y==0 {rgb("#dcece6")} else if calc.odd(y){rgb("#f1f5f3")},[*Population*],[*Instances*],[*Exactly 4*],[*Distinct strings*],..d.populations.flatten().map(x=>[#x]))

The shared query #raw("SELECT * FROM records ORDER BY rowid") returned the entire table. The evaluator recomputed SHA-256 identities over the returned column names and rows and compared these with the two original source files. Whole-table retrieval therefore exposed the changed evidence on the measured pairs. The frozen exact scorer credited one individual answer and zero complete pairs. The masked source-visible reviewer also credited one individually useful answer.

This audit is descriptive. Training labels, policy samples and held requests differ in both population and decoding conditions. Comparing their concentrations cannot isolate the effect of policy optimization from the preceding supervised stage. An improved policy must choose a program or generate an answer that responds to the changed source.

Rebuild the figure and every per-source result with #raw(".venv/bin/python -m scripts.table_policy_answer_concentration"). All literal counts and observed source identities are included in the accompanying CSVs. The seven-page direct comparison and the consolidated research paper include the 220 other requests and the full methods.
