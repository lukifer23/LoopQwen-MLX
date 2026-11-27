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

*Architecture.* The new model reads each prompt token through the original 24 Qwen layers once. The last attention layer retains its ordinary causal key/value memory. When predicting a token, the model performs two further attention reads against the same stored memory. A tied nonlinear controller updates the state after observing each result. The second query depends on the first update. Neither extra read appends a copy of the prefix. At deployment we count the actual output positions and recurrent reads. At fit time the model differentiates through the repeated attention computation. Temporarily removing one read at the learned weights supplies a within-model depth intervention.

*Loss and controls.* Alongside response-token likelihood and full-vocabulary teacher KL, training assigns probability to passages independently cited for reasoning steps or financial calculations. A before/after comparison on fixed training examples measures learning. A two-read/single-read comparison measures the contribution of the extra tied update at fixed weights. A held loop/Base product comparison and an equally trained ordinary-depth model answer separate scientific questions. The Base arm retains its published weights and ordinary depth. The loop uses the four-bit instruction sibling and 16-bit floating-point compute. User-facing quality and a causal effect of looping require separate evidence.

*Training population.* The curated collection has #d.examples distinct completed turns, #d.opening prompt positions, #d.positions supervised target positions and #d.citations human-cited queries per epoch. Deduplication removed 474 repeated sequences. Two identical financial-report questions with contradictory expert targets were removed with both associated replies. Each tool history contains actual source-provider observations. The crossed histories are controlled interventions on original human requests. The original users did not produce the crossed calls.

#image("repair-query-exposure.svg",width:100%)
#text(size:9pt)[Fig. 1. Opening and supervised positions are substantially different quantities. All sources use complete retained text and tool returns.]

#pagebreak()
#text(size:19pt,weight:"bold")[Measurement, verification and adoption criteria]
#v(.3em)
#tbl((2.3fr,1fr,1fr,1fr,1fr),("Archived phase","Decision","Native s","Peak GiB","Cap s"),d.stages.map(r=>(r.stage.replace("repair-query-","").replace("-01",""),if r.decision.passed {"pass"} else {"miss"},fmt(r.seconds),fmt(r.peak),str(r.budget))))
*Numerical control.* The BF16 donor alone exceeded a 0.55 absolute logit tolerance when the same prompt was divided into two prefill segments. Changing only the original BF16 floating-point arrays to FP16 preserved the packed integer matrix codes. On 12 hash-selected training prompts the ordinary and recurrent FP16 routes preserved every original top token. Across full and split partitions the maximum error was 0.127 for ordinary Qwen and 0.122 for recurrence. On the same partition the two architectures differed by at most 0.015625. These observations justify the registered extraction.

#let f=d.stages.first()
#if "examples" in f.decision [
  *Native source memory.* The feature phase archived #f.decision.examples separate examples, #f.decision.labels labels and #fmt(f.decision.file_bytes/1073741824) GiB of native arrays. The recorded evidence includes token identities, full final-layer keys and values, ordinary output states and pre-attention states. The three original parity cases check all six attention offsets, every temporal memory owner and three physical visits to the output block.
] else [
  *Feature extraction stopped.* The recorded reason was #f.decision.reason. Completed files are counted only when the archive contains their verified identities.
]

#if "cold" in f [
  An independent process verified #f.cold.checked_files complete file digests and reconstructed #f.cold.independent_regenerations.len() examples from their original tokens.
]

*Primary population.* The frozen study enrolls 180 matched requests. There are 48 multi-step questions from different StrategyQA topics, 32 report questions from different FinQA issuers, 96 presentations of 24 CoSQL questions on 12 previously unused databases, and four complete CLDR documents of 152,185–164,271 tokens. Ten CoSQL questions come from the released development split and 14 from separately withheld training-split schemas. The model’s public pretraining exposure cannot be exhaustively determined. An independent search of all 6,394 fitting turns, including targets and tool observations, found none of the 104 distinct held short questions in full. Nine short human decomposition steps recurred. Four financial evidence fields shared at least twelve consecutive training words. One reasoning topic and three reports meet the frozen wording threshold. All remain enrolled. The supplement in the preceding volume documents source ambiguity and duplicate supervision.

*Actual tool behavior.* Each database question appears independently and with three possible histories. A correct result can be used directly. A real provider error or a plausible result belonging to the other question requires a new, correct call. The original source and every alternate database test the program. We recorded 168 published alternate instances for the development schemas and 56 separately generated, foreign-key-checked instances for the additional schemas. Constant and partner controls both failed on all 24 questions. Independently regenerating the 180 prompt sequences also re-executed the 144 tool callbacks appearing in the supplied histories.

*Decision and uncertainty.* Both arms use the same native thinking-enabled prompts, greedy decoding, output limits and evaluation order. The maximum is 384 tokens per generation, three turns per short request and one per long request. We record actual first-token and overall time, opening and decode throughput, peak MLX allocation and completed answers. Each short family requires six net gains, zero lost successes and a two-sided paired randomization probability at most 0.05. SQL sign flips are clustered by schema to account for dependence across repeated presentations. All four tool-history conditions must show positive net gains. The four long answers must all be correct and strictly exceed Base. We mask every short response for source review and report seven preflagged wording ambiguities separately. Replication on a second seed, matched donor/precision/training at ordinary depth, broad retained tasks and offline installation are required before a release claim.

#if d.stages_done < 3 [*Observed scope.* The archive covers #d.stages_done of three registered phases. The current paper contains no completed held product comparison for the two-read weights.]

#pagebreak()
#text(size:19pt,weight:"bold")[Mechanism, optimization and relationship to prior work]

*One write and two further reads.* Let $h_0$ be the ordinary final-block output. At visit $j$, the normalized state $h_j$ supplies a learned query correction. The frozen Qwen attention, gate, output projection and feed-forward layer use that query to produce a proposal $p_j$ against the original causal keys and values. Let $d_j = p_j - h_j$, and let $s_j$ be the root mean square of $h_j$ with a numerical stabilizer. With learned gate $G$, content projection $C$ and bias $b$, the next state is

$ h_(j+1) = h_j + tanh(b + G(h_j / s_j)) ⊙ d_j + s_j C(d_j / s_j). $

The two visits share all parameters. Query and content corrections have rank 160, the gate correction has rank 32 and the Qwen hidden width is 1,024. Together the three projections and the bias contain 721,920 trainable FP32 scalars. Opening tokens that only populate future attention memory skip the final output projection, neural feed-forward step and recurrent reads. Each emitted position incurs both reads. Longer prompts therefore increase the evidence each generated token can attend to while preserving a single causal memory at the final attention layer.

*Optimization.* For each supervised position the loss includes target-token negative log likelihood and 0.24 times the full-vocabulary KL divergence from the frozen ordinary instruction donor to the candidate. At annotated positions the objective adds eight times the negative log of the probability mass assigned to the union of human-cited source spans, averaged across query heads before taking the log. The sum uses the number of supervised target positions as its denominator. Native MLX differentiates through causal attention in segments of at most 32 target positions. Complete examples accumulate until at least 512 target positions, with the final partial update retained in each epoch. AdamW uses a learning rate of 0.00022, weight decay of 0.01 and gradient-norm clipping at 1. The deterministic shuffle seed is 864371. Source spans are aligned to the original tokens before optimization.

*Recurrent computation.* Universal Transformers repeatedly apply a shared transformation across depth. Geiping and colleagues trained a language model with a recurrent core and measured the effects of changing latent depth on reasoning. Here a small controller revisits the last attention memory of an existing checkpoint at fixed depth. The strongest local attribution is a comparison with an equally trained one-visit model. #link("https://arxiv.org/abs/1807.03819")[Dehghani et al., Universal Transformers] · #link("https://arxiv.org/abs/2502.05171")[Geiping et al., Scaling up Test-Time Compute with Latent Reasoning].

*Evidence and executable tools.* StrategyQA provides reasoning decompositions and evidence. FinQA includes questions and financial calculations supplied by experts. CoSQL connects human requests to executable SQL across database domains. Zhong and colleagues evaluated program meaning by executing queries on multiple database instances. We use these resources to test whether a model can select a conclusion, compute a number from the correct operands and recognize when a successful query answered a different question. #link("https://doi.org/10.1162/tacl_a_00370")[Geva et al., StrategyQA] · #link("https://aclanthology.org/2021.emnlp-main.300/")[Chen et al., FinQA] · #link("https://aclanthology.org/D19-1204/")[Yu et al., CoSQL] · #link("https://aclanthology.org/2020.emnlp-main.29/")[Zhong et al., Semantic Evaluation for Text-to-SQL].

*Reproduction.* The feature archive records the pinned source revisions, input hashes, native library versions, numerical parity cases, all feature digests, timing and memory. The public script #raw("scripts.verify_shared_query_native") compares newly generated native tensors to these digests. The fit and primary registrations specify the frozen stage order and advancement criteria. #raw("python -m scripts.paper_repair_query") rebuilds the figures, tables and this document from archived records.

#if d.stages_done >= 2 [
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[Native learning and physical depth intervention])
 #let r=d.stages.at(1).decision
 #if "groups" in r [
   The optimizer completed #r.updates updates over #r.positions supervised positions and #r.citations cited source queries. Mean citation NLL changed by #fmt(r.mean_citation_nll_gain,n:5) nats in the favorable direction. The mean group-balanced reduction in response NLL from the second learned read was #fmt(r.mean_second_visit_gain,n:5) nats.
   #image("repair-query-optimization.svg",width:100%)
   #text(size:9pt)[Fig. 2. Groups of sixteen real updates. Citation loss includes only explicitly annotated expert queries.]
   #tbl((1.4fr,.65fr,1.4fr,.65fr),("Training group","NLL gain","Training group","NLL gain"),{
     let a=r.groups.pairs().sorted(key:x=>x.at(0))
     range(0,a.len(),step:2).map(i=>(a.at(i).at(0).replace("_"," "),fmt(a.at(i).at(1),n:4),if i + 1 < a.len() {a.at(i+1).at(0).replace("_"," ")} else {""}, if i + 1 < a.len() {fmt(a.at(i+1).at(1),n:4)} else {""}))
   })
   A positive gain means lower target-token loss after training on the fixed probes. The acceptance rule also protects four earlier source groups, checks five kinds of tool repair, and requires gains in source reasoning and financial actions.
 ] else [
   The fit stopped with the recorded reason “#r.reason.” The archive preserves its actual resource expenditure and any durable optimizer state.
 ]
]
#if "depth_effects" in d [
 #pagebreak()
 #text(size:19pt,weight:"bold")[What changes when the second read is removed?]

 At the final learned weights we supplied the same source memory, complete prompt, target tokens and controller to both paths. We removed the final tied attention read and its state update on one path. Both paths used teacher forcing. Each dot is one complete training probe, with equal weight for every target token within that probe.

 #image("repair-query-depth-probes.svg",width:100%)
 #text(size:9pt)[Fig. 3. Values to the right of zero favor the second physical read. Diamonds show the median within each training family. Vertical displacement separates overlapping observations.]

 #let v=d.depth_effects
 Across #v.n probes and #v.positions response positions, adding the second read increased target likelihood for #v.positive probes, reduced it for #v.negative, and left #v.tied unchanged. The mean gain was positive in #v.groups_positive of #v.groups training groups. Individual probes were chosen deterministically before training. These measurements identify local effects on reference-token predictions. Free generation and the controlled tool histories test whether those effects transfer to complete answers.

 #if "cold" in d.stages.at(1) [
  #let c=d.stages.at(1).cold
  An independent process rebuilt the input features, loaded the archived weights and repeated the one-read and two-read paths, along with the original initialization. It reproduced all eight training decisions. The largest absolute difference across the restored response losses, teacher KL and citation losses was #fmt(calc.max(..c.max_errors_by_state.values().map(x=>calc.max(..x.values()))),n:8). It also independently counted all 6,394 distinct complete examples in each epoch.
 ]
]
#if d.stages_done >= 3 and d.stages.at(2).decision.at("status", default: "") == "interrupted" [
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[Running the complete documents on 18 GiB of unified memory])

 The first paired attempt completed five short loop requests. During the sixth request, a 152,185-token opening, macOS reported system memory pressure level 2 with 3.65 GiB available. The native worker stopped after #fmt(d.stages.at(2).seconds) seconds at a peak MLX allocation of #fmt(d.stages.at(2).peak) GiB. The untouched Base arm had yet to run. The archive preserves the unfinished comparison and its cost. It yields no paired task score.

 *Storage change.* A separate implementation processes consecutive layers with one full attention memory resident per group. It retains every activation using its original 16-bit representation. An immutable copy of each completed layer’s keys and values moves to a request-local file. On each generated token the decoder restores that prefix, appends the new keys and values and executes the ordinary Qwen block. At the final layer the loop executes both learned reads against the complete evidence. Earlier long-context experiments with another model had measured a memory saving and a latency cost from staging. We record both quantities again with the trained two-read model.

 #if "staging_proof" in d [
  #let e=d.staging_proof
  *Independent qualification.* The engineering run lasted #fmt(e.native_seconds) seconds with a peak of #fmt(e.peak_bytes/1073741824) GiB. It recorded #e.short.len() ordinary or recurrent prompt comparisons, each with four successive full-vocabulary outputs under both storage paths.

  #if e.passed [
   Every greedy token decision matched. The maximum absolute logit difference was #fmt(calc.max(..e.short.map(x=>calc.max(..x.errors))),n:6). The original #(e.long.prompt)-token lockfile then produced four output positions with #e.long.read_only_visits physical recurrent reads. All six attention histories retained every preceding token. Time to first output was #fmt(e.long.first_s) seconds. Temporary files occupied #fmt(e.long.scratch_bytes/1073741824) GiB at peak. This trial measures full-context execution and numerical fidelity. The source-answer test uses the four original full-length documents.
  ] else [
   The registered engineering prerequisite stopped with “#e.reason.”
  ]
 ]
 #if d.stages_done == 4 and d.stages.last().decision.at("status",default:"") == "interrupted" [
  The second attempt also stopped. Its recorded reason was “#d.stages.last().decision.reason.” The archive records the completed prefixes for each arm without extrapolating to the remaining questions.
 ]
 #if "trace_correction" in d [
  *Trace correction and cost.* The first staged 152,185-token generation reached scoring. The validator expected the count of resident token-block advances, whereas the staged route advances layers in groups. The original attempt stopped before returning a score. The amended validator checks the actual prefix and decoded tail lengths at all six attention sites and requires zero resident advances. The answer comparison still requires the exact source value. We preserved the five preceding requests, discarded the unscored generation, and charged the full #fmt(d.trace_correction)-second interval and peak to its retry. The two code revisions and frozen receipt are part of the comparison evidence.
 ]
]
#if "reading" in d [
 #pagebreak()
 #text(size:19pt,weight:"bold")[Held product comparison]
 #let r=d.stages.last().decision
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

#if "distribution" in d [
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[Accuracy at user deadlines and variation across sources])

 Every request remains in the denominator. Each point counts correct final answers delivered by that deadline, with measured model loading, opening, tool execution and generation included. Short answers that arrive quickly still need a correct result to contribute.

 #image("staged-query-delivery.svg",width:100%)
 #text(size:9pt)[Fig. 4. Each panel uses the original workload and scoring. Deadlines are fixed in advance and plotted on a log time axis.]

 #tbl((1.3fr,.7fr,.7fr,.9fr,1.2fr),("Source family","Clusters","Requests","Loop − Base","95% interval"),d.distribution.intervals.filter(x=>x.metric=="correct").map(x=>(
 x.family,str(x.source_clusters),str(x.n),fmt(100*x.loop_minus_base,n:1)+" pp",
 fmt(100*x.lower,n:1)+" to "+fmt(100*x.upper,n:1)+" pp")))

 The interval resamples entire database schemas, financial issuers or reasoning topics within each family. Both responses to an item travel together on each of 10,000 draws. Intervals describe variability across the enrolled sources. The two locale clusters are reported individually in the request ledger. The original paired sign-flip tests and masked reading govern the decision. The archived timings reflect the order and thermal state of one Apple Silicon machine.

 #let l=d.distribution.families.all
 Of #l.both_correct requests answered correctly by both models, the loop finished first on #l.faster_on_both_correct. The loop gained correctness and also finished faster on #l.loop_gained_and_faster requests. It lost correctness and finished slower on #l.loop_lost_and_slower.
]
