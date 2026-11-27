#set document(title:"Do the four generated SQL programs survive further table changes?",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.74in,y:.67in),numbering:"1")
#set text(font:"Libertinus Serif",size:10.5pt)
#let d=json("conditional-unseen-tables.json")
#let a=d.arms.candidate
#let b=d.arms.control
#text(size:22pt,weight:"bold")[Do the four generated SQL programs survive further table changes?]
#v(.45em)
*Retrospective execution of fixed model outputs on independently annotated tables* \
LoopQwen-MLX · 27 November 2026

The loop generated SQL that returned the correct scalar on both enrolled worlds for four of 24 human questions. Those four statements also returned the right result on every additional eligible table. The audit adds 22 answers for the four successful programs. Across all versions, these statements returned five distinct Waterford counts and five opponent names. The same replay found zero programs from untouched Base that solved both enrolled worlds.

#table(columns:(2.75fr,.65fr,.65fr,1.18fr),inset:5pt,stroke:.45pt+gray,
[*Original user question*],[*All tables*],[*Extra tables*],[*Values returned*],
..d.validated_queries.map(r=>([ #r.question ],[#r.all],[#r.extra],[#r.distinct_answers.join(", ")])).flatten())

*Coverage.* We retained all 24 enrolled questions, including those on which neither model made a database query. Across those questions, #d.sources_with_extra had at least one additional eligible version. All #d.labeled_versions tables had at least three agreeing human ratings per answer. The earlier evaluation had exposed two tables per question, leaving #d.extra_versions additional files for replay. Four rated files failed the original table-shape or column-name requirements.

*Method.* We verified the three upstream archives by SHA-256 and reconstructed each SQLite database from its original table cells. We extracted every database-query statement actually issued on either enrolled version, including failures. Within each source we replayed each distinct statement unchanged on every available version. This produced #a.reexecutions loop and #b.reexecutions Base executions from #a.distinct_programs_within_source and #b.distinct_programs_within_source distinct programs. The check required a real read of the table, one row and one column, an untruncated observation and equality to the agreed human denotation. All source and opposite-world outcomes matched the original execution ledger.

*Interpretation.* Every successful source had seven or eight distinct eligible table versions. The tested answers changed across those versions, so returning a constant could not pass. These observations support program-level transfer to different contents of four tables. The model originally saw 24 questions, and this audit adds no new question or freshly generated answer. The release decision, masked reading and model-level sample size stay tied to the original comparison.

*Reproduction.* Run #raw("python -m scripts.conditional_unseen_tables") with the pinned upstream archives. The adjacent execution CSV contains every statement, file digest, target, callback output and error. The source CSV includes both models on all 24 questions. Original table questions and annotations are from #link("https://nlp.stanford.edu/software/sempre/wikitable/")[Pasupat and Liang’s WikiTableQuestions project].
