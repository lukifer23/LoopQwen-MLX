#set page(paper:"us-letter",margin:(x:0.88in,y:0.75in),numbering:"1")
#set text(size:10pt)
#let d=json("middle-depth-data.json")
#let shown(v)=if v==none { [—] } else {str(v)}
#text(size:23pt,weight:"bold")[Can four repeated Qwen layers turn evidence into a decision?]
#linebreak()
#text(size:12pt)[Native matched training and unrestricted generation on an 18 GiB M3 Pro]
= Donor, control and intervention
A frozen eight-question donor screen credited three answers from Qwen3.5-0.8B Base and zero from its instruction sibling at 256 generated tokens. Both fitted models therefore inherit the same Base text weights. The two-pass candidate repeats layers 12–15, including one global attention layer. Eight pretrained layers read out the result. Rank-16 adapters on six MLP output projections give each model 442,368 trainable coefficients. Only the recurrent model also fits the input-mixing coefficient. Both retain the ordinary vocabulary projection. Its one-pass initialization matched the donor logits exactly.
 
Huang et al. (arXiv:2610.23036) studied loop placement and input injection. Marchenko et al. (arXiv:2610.00673) motivated supervising multiple exits. Here the first 32 updates supervise one and two physical passes, followed by equal weighting of all three exits. The ordinary-depth sibling receives its own optimization on the same source order, final answers, rejected decisions and suffix adapters. The public effect and the causal depth effect use different comparators. Four families receive equal weight only when all four have measurements for the same checkpoint.

= Development generations
#image("middle-depth-generation.svg",width:100%)
Profiles and likelihood diagnostics never earn a supported answer. A required action must execute successfully. The figure uses the 128-token cap and the three short families. Each panel uses original, minimally changed and irrelevant-change versions of three independent sources per family. The source table and individual trajectory table retain the arm, input variant, score, wall time and error. Follow-ups are reported separately under actual and common histories.

#table(columns:(1.25fr,1fr,1fr,1fr,1fr),inset:3pt,[*Stage*],[*Arm*],[*Family*],[*Correct / items*],[*Seconds*],..d.summaries.filter(r=>r.stage=="early" and r.cap==128).map(r=>(r.stage,r.arm,r.family,str(r.correct)+"/"+str(r.total),str(r.seconds))).flatten())

#pagebreak()
= Frozen evidence and native cost
The first wave uses 288 turns, with 72 in each family. The mixture includes natural writing, qualitative and multi-fact reasoning, calculator and database work, complete tables and schedule records. Each turn has a complete supported target. Fitting on these short contexts leaves full-document transfer an empirical question. The new long development openings contain 145,891 tokens for both editions of _Anne of Green Gables_ and 167,014 or 167,017 tokens for _McTeague_. Each request uses a complete book, a documented changed world, and question-specific evidence at distant positions. The matched time decision uses 600 seconds on books and eight seconds for short work.

The two update-32 models made no live calls on 16 financial training roll-ins. Source review found #d.history.unique_failed_prefixes failed trajectories. The correction pool includes #d.history.shared_executions shared calculator executions, including a successful calculation submitting the wrong report's operands. The actual-history and simple conditions have identical #d.history.positions.y supervised target tokens. Both include ordinary language examples. A model-generated action and a shared executed history are counted independently.

#table(columns:(1.5fr,1fr),inset:4pt,[*Worker charge*],[*Seconds*],..d.spend.pairs().map(p=>(p.at(0),str(p.at(1)))).flatten())
The cost inventory includes stopped native jobs, optimizer work, generation and prefix building. Long-prefix warm timings exclude the one-time construction, which is recorded alongside each warm request. The source and cost CSVs identify the archived inputs and run receipts. Two independent cold restarts reproduced the prepared next optimizer update bit for bit. Recomputing the gradient on Metal varied by up to 0.0000355 in the next weights and optimizer state. At update 32 the relocatable two-model archive avoids #d.archive duplicate donor payload bytes, measured from hash-verified members.

= Boundary of the claim
#if d.decision == none [Development is in progress. These observations establish neither a product gain on all four families nor a replicated benefit of tied depth. A failed full-length answer is retained even when every physical attention history verifies.] else [#d.decision.reading]
The historical financial receipt fit stopped before its first optimizer update. Its available crossed receipts distinguish an executable result from a result supported by the question. It supplied no estimate of a trained recovery effect.
