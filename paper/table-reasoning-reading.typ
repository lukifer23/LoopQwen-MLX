#set document(title:"Does the higher reasoning score survive a source reading?",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.8in,y:.72in))
#set text(font:"Libertinus Serif",size:11pt)
#set par(leading:.44em,spacing:.55em)
#let d=json("../docs/measurements/table-reasoning-source-reading.json")
#let fam=(("effects","Causal comparison",24),("human_first","Opening question",12),("human_followup","Follow-up question",12))
#let metric(g,k)=g.at(k,default:0)
#let pair(f,k)={let c=d.counts.candidate.at(f);let b=d.counts.control.at(f);str(metric(c,k))+" / "+str(metric(b,k))}
#let f1(f,n)={let c=d.counts.candidate.at(f);let b=d.counts.control.at(f);str(calc.round(c.f1_sum/n,digits:3))+" / "+str(calc.round(b.f1_sum/n,digits:3))}
#text(size:22pt,weight:"bold")[Does the higher reasoning score survive a source reading?]
#linebreak()
#text(size:13pt)[A masked sensitivity analysis of 48 previously generated answers per model]

The trained four-layer loop produced 22 fully usable answers and untouched Base produced 29. On causal comparisons the loop won the automatic exact-match count 7 to 3, yet the source reading favored Base 14 to 12. Both models sometimes answered correctly in words outside the benchmark reference. Twelve sampled conversations also exposed a weakness in keeping track of the question across turns.

#table(columns:(1.6fr,.48fr,1fr,1.2fr,1fr),stroke:none,inset:5pt,fill:(x,y)=>if y==0 {rgb("#dcece6")} else if calc.odd(y){rgb("#f1f5f3")},[*Question*],[*Count*],[*Exact*],[*Mean F1*],[*Fully usable*],..fam.map(r=>([ #r.at(1) ],[#r.at(2)],[#pair(r.at(0),"automatic_exact")],[#f1(r.at(0),r.at(2))],[#pair(r.at(0),"fully usable")])).flatten())
#text(size:9.5pt)[Table 1. Paired entries are loop / Base. F1 measures overlap with dataset answers. Fully usable means the visible response answers the current question with support from its source. Equivalent wording receives credit. A material factual addition prevents full credit.]

*Source selection and masking.* After inspecting the original aggregate scores, we fixed a hash seed and chose 12 of the 24 QuAC passages and 12 of the 24 ROPES backgrounds. Both questions from every chosen source entered. Selection used source identity alone. One project reviewer (the coding assistant) read the full passage, current question, prior human turn where applicable, source annotations and both hash-shuffled visible replies. The reviewer knew the earlier aggregates. Identical replies shared a grade. We committed the 48 reasons and grades before revealing identities. No model was run again.

*Where the scores differ.* Both models answered a Pixies question with the album the passage explicitly called the band's debut. All five reference spans instead described an earlier demo tape. On the Mars rover, one response correctly reported the rear-wheel failure even though annotators chose the front wheel. Another got both wheels right but incorrectly attributed the spectrometer slowdown to environmental exposure. These examples show how the reading can credit a different valid answer and penalize a plausible extra claim.

*Dependence within sources.* The loop solved both questions on 3 of 12 causal backgrounds versus 5 for Base (two-sided paired sign-flip _p_ = 0.6875). On 12 conversations it solved 2 whole pairs versus 5 (_p_ = 0.25). At the individual-question level it gained 3 causal answers and lost 5, then gained 4 conversational answers and lost 9. The test flips the sign of each source's whole-pair difference and enumerates all assignments, including ties. These small descriptive samples cannot establish a population-wide degradation. They give no corroboration of a broad reasoning advantage for these weights.

*Reproduction.* The comparison archive holds the original outputs. The selection plan, masked passages, committed judgments and computed totals are versioned. Regenerate the source-level counts with #raw(".venv/bin/python -m scripts.table_reasoning_reading report"). The primary preregistered decision and its original scores remain the basis for qualification.
