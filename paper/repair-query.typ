#set document(title:"Can a recurrent read of source memory repair a plausible wrong answer?",author:"LoopQwen-MLX")
#set page(paper:"us-letter",margin:(x:.73in,y:.67in),numbering:"1")
#set text(font:"Libertinus Serif",size:10.5pt)
#set par(leading:.42em,spacing:.56em)
#let d=json("repair-query.json")
#let fmt(x,n:2)=str(calc.round(x,digits:n))
#let tbl(cols,h,rows)=table(columns:cols,inset:4.5pt,stroke:.45pt+rgb("#bdc7c6"),..h.map(x=>[*#x*]),..rows.flatten())
#text(size:23pt,weight:"bold")[Can a recurrent read of source memory repair a plausible wrong answer?]
#v(.35em)
*Source-grounded reasoning and tool use in Qwen3.5-0.8B on Apple Silicon* LoopQwen-MLX research study · November 2026

*Research question.* Does learning to revisit source evidence improve one model’s ability to reason and to correct a tool call that succeeded but answered a different question? In our earlier direct comparison, the loop repeated a plausible wrong scalar in 19 of 27 final answers after receiving one. The preceding training intervention mastered 14 of 24 SQL questions on all 87 relevant table versions but failed its registered qualitative-pair gate. Ten of those 14 generated programs reproduced the expert string exactly. Four differed from every positive SQL target in either training corpus. Those 24 questions had all been enrolled in fitting.

*Architecture.* The new model reads each prompt token through the original 24 Qwen layers once. The last attention layer retains its ordinary causal key/value memory. When predicting a token, the model performs two further attention reads against the same stored memory. A tied nonlinear controller updates the state after observing each result; the second query depends on the first update. Neither extra read appends a copy of the prefix. At deployment we count the actual output positions and recurrent reads. At fit time the model differentiates through the repeated attention computation. Temporarily removing one read at the learned weights supplies a within-model depth intervention.

*Loss and controls.* Alongside response-token likelihood and full-vocabulary teacher KL, training assigns probability to passages independently cited for reasoning steps or financial calculations. A before/after comparison on fixed training examples measures learning. A two-read/single-read comparison measures the contribution of the extra tied update at fixed weights. A held loop/Base product comparison and an equally trained ordinary-depth model answer separate scientific questions. The Base arm retains its published weights and ordinary depth. The loop uses the four-bit instruction sibling and 16-bit floating-point compute. User-facing quality and a causal effect of looping require separate evidence.

*Training population.* The curated collection has #d.examples distinct completed turns, #d.opening prompt positions, #d.positions supervised target positions and #d.citations human-cited queries per epoch. Deduplication removed 474 repeated sequences. Two identical financial-report questions with contradictory expert targets were removed with both associated replies. Each tool history contains actual source-provider observations. The crossed histories are controlled interventions on original human requests; the original users did not produce the crossed calls.

#image("repair-query-exposure.svg",width:100%)
#text(size:9pt)[Fig. 1. Opening and supervised positions are substantially different quantities. All sources use complete retained text and tool returns.]

#pagebreak()
#text(size:19pt,weight:"bold")[Measurement, verification and adoption criteria]
#v(.3em)
#tbl((2.3fr,1fr,1fr,1fr,1fr),("Archived phase","Decision","Native s","Peak GiB","Cap s"),d.stages.map(r=>(r.stage.replace("repair-query-","").replace("-01",""),if r.decision.passed {"pass"} else {"miss"},fmt(r.seconds),fmt(r.peak),str(r.budget))))
*Numerical control.* The BF16 donor alone exceeded a 0.55 absolute logit tolerance when the same prompt was divided into two prefill segments. Changing only the original BF16 floating-point arrays to FP16 preserved the packed integer matrix codes. On 12 hash-selected training prompts the ordinary and recurrent FP16 routes preserved every original top token. Across full and split partitions the maximum error was 0.127 for ordinary Qwen and 0.122 for recurrence. On the same partition the two architectures differed by at most 0.015625. These observations justify the registered extraction; they carry no task-accuracy claim.

#let f=d.stages.first()
#if "examples" in f.decision [
  *Native source memory.* The feature phase archived #f.decision.examples separate examples, #f.decision.labels labels and #fmt(f.decision.file_bytes/1073741824) GiB of native arrays. The recorded evidence includes token identities, full final-layer keys and values, ordinary output states and pre-attention states. The three original parity cases check all six attention offsets, every temporal memory owner and three physical visits to the output block.
] else [
  *Feature extraction stopped.* The recorded reason was #f.decision.reason. Completed files are counted only when the archive contains their verified identities.
]
#if "cold" in f [
  An independent process verified #f.cold.checked_files complete file digests and reconstructed #f.cold.independent_regenerations.len() examples from their original tokens.
]
*Primary population.* The frozen study enrolls 180 matched requests. There are 48 multi-step questions from different StrategyQA topics, 32 report questions from different FinQA issuers, 96 presentations of 24 CoSQL questions on 12 previously unused databases, and four complete CLDR documents of 152,185–164,271 tokens. Ten CoSQL questions come from the released development split and 14 from separately withheld training-split schemas. The model’s public pretraining exposure cannot be exhaustively determined. An independent search of all 6,394 fitting turns, including targets and tool observations, found none of the 104 distinct held short questions in full. Nine short human decomposition steps recurred. Four financial evidence fields shared at least twelve consecutive training words. One reasoning topic and three reports meet the frozen wording threshold; all remain enrolled. The supplement in the preceding volume documents source ambiguity and duplicate supervision.

*Actual tool behavior.* Each database question appears independently and with three possible histories. A correct result can be used directly. A real provider error or a plausible result belonging to the other question requires a new, correct call. The original source and every alternate database test the program. We recorded 168 published alternate instances for the development schemas and 56 separately generated, foreign-key-checked instances for the additional schemas. Constant and partner controls both failed on all 24 questions. Independently regenerating the 180 prompt sequences also re-executed the 144 tool callbacks appearing in the supplied histories.

*Decision and uncertainty.* Both arms use the same native thinking-enabled prompts, greedy decoding, output limits and evaluation order. The maximum is 384 tokens per generation, three turns per short request and one per long request. We record actual first-token and overall time, opening and decode throughput, peak MLX allocation and completed answers. Each short family requires six net gains, zero lost successes and a two-sided paired randomization probability at most 0.05. SQL sign flips are clustered by schema; the 96 related presentations are not independent users. All four tool-history conditions must show positive net gains. The four long answers must all be correct and strictly exceed Base. We mask every short response for source review and report seven preflagged wording ambiguities separately. Replication on a second seed, matched donor/precision/training at ordinary depth, broad retained tasks and offline installation are required before a release claim.

#if d.stages_done<3 [*Observed scope.* The archive covers #d.stages_done registered phase(s). The current paper contains no completed held product comparison for the two-read weights.]

#if d.stages_done>=2 [
 #pagebreak()
 #text(size:19pt,weight:"bold")[Native learning and physical depth intervention]
 #let r=d.stages.at(1).decision
 #if "groups" in r [
   The optimizer completed #r.updates updates over #r.positions supervised positions and #r.citations cited source queries. Mean citation NLL changed by #fmt(r.mean_citation_nll_gain,n:5) nats in the favorable direction. The mean group-balanced reduction in response NLL from the second learned read was #fmt(r.mean_second_visit_gain,n:5) nats.
   #image("repair-query-optimization.svg",width:100%)
   #text(size:9pt)[Fig. 2. Groups of sixteen real updates. Citation loss includes only explicitly annotated expert queries.]
   #tbl((1.4fr,.65fr,1.4fr,.65fr),("Training group","NLL gain","Training group","NLL gain"),{
     let a=r.groups.pairs().sorted(key:x=>x.at(0))
     range(0,a.len(),step:2).map(i=>(a.at(i).at(0).replace("_"," "),fmt(a.at(i).at(1),n:4),if i+1<a.len() {a.at(i+1).at(0).replace("_"," ")} else {""}, if i+1<a.len() {fmt(a.at(i+1).at(1),n:4)} else {""}))
   })
   A positive gain means lower target-token loss after training on the fixed probes. The acceptance rule also protects four earlier source groups, checks five kinds of tool repair, and requires gains in source reasoning and financial actions.
 ] else [
   The fit stopped with the recorded reason “#r.reason.” The archive preserves its actual resource expenditure and any durable optimizer state.
 ]
]
#if "reading" in d [
 #pagebreak()
 #text(size:19pt,weight:"bold")[Held product comparison]
 #let r=d.stages.at(2).decision
 #let t=r.costs
 #tbl((1fr,1fr,1fr,1fr,1fr),("Family","Requests","Loop / Base","Gains / losses","Paired p"),r.counts.pairs().map(pair=>{
   let val=pair.at(1)
   (pair.at(0),str(val.n),str(val.total.at(0))+" / "+str(val.total.at(1)),str(val.gains)+" / "+str(val.losses),if val.p==none {"—"} else {fmt(val.p,n:4)})
 }))
 #tbl((1.8fr,1fr,1fr),("Measured cost","Loop","Untouched Base"),(("Wall time, s",fmt(t.candidate.wall_s),fmt(t.control.wall_s)),("Summed first-token wait, s",fmt(t.candidate.first_s),fmt(t.control.first_s)),("Peak MLX GiB",fmt(t.candidate.peak/1073741824),fmt(t.control.peak/1073741824)),("Opening tokens/s",fmt(t.candidate.opening_tps),fmt(t.control.opening_tps)),("Post-first tokens/s",fmt(t.candidate.decode_tps),fmt(t.control.decode_tps))))
 #tbl((1.2fr,.8fr,.8fr,1fr,.8fr),("Masked short family","Loop usable","Base usable","Gains / losses","Paired p"),d.reading.families.pairs().map(pair=>{
   let val=pair.at(1)
   (pair.at(0),str(val.arms.candidate.grounded_usable),str(val.arms.control.grounded_usable),str(val.gains)+" / "+str(val.losses),fmt(val.p,n:4))
 }))
 The public request ledger contains both delivered responses, costs and decisions. Raw archives preserve prompt IDs, emitted tokens, causal history offsets, actual read-only visits and every provider observation. A project assistant performed the masked adjudication. These findings apply to the enrolled sources and conditions.
]
