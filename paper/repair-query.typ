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

#if "bounded_primary" in d.at("content_read", default: (:)) [
 #let a=d.content_read.bounded_primary
 #parbreak()
 *Abstract.* We tested whether two learned reads of the last attention layer allow a small language model to answer source-dependent questions and use tools more effectively than ordinary Qwen3.5-0.8B-Base. Both products receive identical initial prompt tokens, reasoning enabled, unrestricted greedy decoding and real calculator or database responses. We measure delivered answers, physical recurrent visits, elapsed wall time and native Apple Silicon memory. The recurrent product uses a quantized instruction-initialized backbone. The product comparison includes both this donor difference and the learned recurrence.
 #parbreak()
 In a 48-request held comparison on HotpotQA and TAT-QA, the bounded reader completed 38 requests against 28 for Base, in #fmt(a.costs.candidate.wall_s) versus #fmt(a.costs.control.wall_s) seconds. It made valid calculator calls on 23 of 24 reports versus one for Base. The loop returned the right final quantity twice and delivered it correctly once. It answered 13 versus 15 of the 24 reasoning choices correctly. Seven explanations per system were fully supported. A separate 180-request comparison obtained ten versus zero strictly grounded database answers but zero full-document answers for either model. A successful tool call, a lower loss during training and a faster response measured different properties in these studies.
 #parbreak()
 #if "source_copy" in d and "fit" in d.source_copy [A later gated source-copy model trained on 900 complete turns, including original human conversations. It passed all ten training checks and retained the two physical reads. #if "pair" in d.source_copy and "counts" in d.source_copy.pair [
     #let cp=d.source_copy.pair
     Its 48-source conversational test required a correct first answer, correct follow-up after the model’s own first reply, and correct follow-up after a shared first answer. All-three reading successes were #cp.counts.reading.all.total.map(str).join(" / ") and financial successes #cp.counts.report.all.total.map(str).join(" / ") (loop / Base). The 144 requests per arm took #fmt(cp.costs.candidate.wall_s) and #fmt(cp.costs.control.wall_s) seconds.
   ] ]
 #parbreak()
 On two separate 160K–170K-token source releases, training raised correct paired token preferences from one to three out of four held field names. Removing the trained second read returned the score to one. Average evidence-location loss nevertheless worsened. The 128K result uses supplied answer prefixes and one locale. Each result is reported under its own model identity. General superiority would require one policy to improve the reasoning and tool endpoints, retain earlier task successes, and deliver answers on independently authored long documents.
 #parbreak()
 *Study materials.* Each chapter gives the training or deployment measurement and its decision rule. The archive includes frozen input hashes, both complete token trajectories for the direct comparisons, real callback payloads, one source-masked project review, and per-request costs. The local study is followed by the preceding model and systems studies in the comprehensive PDF.
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[Source-repair design and first direct comparison])
]

#if "reading" in d [
 #let r=d.stages.last().decision
 #let pair(k)=r.counts.at(k).total.map(str).join(" / ")
 *Held result.* The loop and untouched Base respectively delivered #pair("field") grounded database answers out of 96, #pair("strategy") correct reasoning conclusions out of 48, #pair("report") grounded financial answers out of 32 and #pair("locale") exact full-document answers out of four. The combined objective and masked source-review gate #if r.passed and d.reading.passed [passed.] else [failed.] Both arms covered every enrolled request.
]

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

*Scope of inference.* Earlier project failures informed the architecture and the choice of tool histories. The team also knows the task families and can read their public source material. A per-study significance test addresses its frozen comparison. Project-wide claims must also account for these adaptive design choices. The 120 training probes measure the effect of removing a learned step under teacher forcing. Establishing a general reasoning or tool advantage requires fresh source-level transfers, complete answers under free generation and a matched ordinary-depth fit. The independent seed measures training stability. An external reviewer would also help quantify uncertainty from the single project adjudicator. #link("https://doi.org/10.1126/science.aaa9375")[Dwork et al., The Reusable Holdout].

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
 #if "content_depth" in d [
  #let c=d.content_depth
  *Which tokens changed?* We independently recomputed both reads at all #c.positions labeled positions, retaining the full vocabulary and the actual greedy choice. Original response offsets identify reasoning text, visible answers and the values supplied to tools.
  #tbl((1.4fr,.7fr,.8fr,.8fr,1fr),("Token class","Positions","1 read top-1","2 reads top-1","Mean NLL gain"), c.groups.pairs().sorted(key:p=>p.at(0)).map(p=>{
    let v=p.at(1)
    (p.at(0),str(v.n),str(v.one_correct),str(v.two_correct),fmt(v.mean_nll_gain,n:4))
  }))
  Top-1 counts compare the model’s most likely next token to the original target. Each prediction conditions on the preceding reference tokens. The largest discrepancy from the archived complete-response mean loss was #fmt(c.max_mean_error,n:7) nats.
 ]
]
#if d.stages_done >= 3 and d.stages.at(2).decision.at("status", default: "") == "interrupted" [
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[Running the complete documents on 18 GiB of unified memory])

 The first paired attempt completed five short loop requests. During the sixth request, a 152,185-token opening, macOS reported system memory pressure level 2 with 3.65 GiB available. The native worker stopped after #fmt(d.stages.at(2).seconds) seconds at a peak MLX allocation of #fmt(d.stages.at(2).peak) GiB. The untouched Base arm had yet to run. The archive preserves the unfinished comparison and its cost. It yields no paired task score.

 *Storage change.* A separate implementation processes consecutive layers with one full attention memory resident per group. It retains every activation using its original 16-bit representation. An immutable copy of each completed layer’s keys and values moves to a request-local file. On each generated token the decoder restores that prefix, appends the new keys and values and executes the ordinary Qwen block. At the final layer the loop executes both learned reads against the complete evidence. Earlier long-context experiments with another model had measured a memory saving and a latency cost from staging. We record both quantities again with the trained two-read model.

 *Storage cost.* Each of six attention layers keeps two key/value heads of width 256. At two bytes per scalar the keys and values occupy 12,288 bytes per input token. A working file adds 2,048 bytes per token for the 1,024-wide hidden state. The resulting 14,336 bytes per input token predict the scratch payload before file headers. The hidden file is removed before decoding. Repeatedly restoring each prefix during generation makes longer replies more expensive. FlexGen studies scheduling model inference across GPU, CPU and disk. PagedAttention addresses allocation and sharing of KV blocks across serving requests. Our measurement covers one user request at a time and exact 16-bit histories on unified memory. #link("https://proceedings.mlr.press/v202/sheng23a.html")[Sheng et al., FlexGen] · #link("https://doi.org/10.1145/3600006.3613165")[Kwon et al., PagedAttention].

 *Interpreting elapsed time.* The two deployed decoders perform different amounts of work at the last layer during a staged opening. Ordinary Base runs its stock final block on every prompt row. The loop writes keys and values for every row, then evaluates the final attention output, feed-forward block and learned revisits at the last prompt position. Both models retain all six complete attention histories. Opening time therefore includes the choice of output computation, backbone precision and storage I/O. An equal-precision, equal-output-schedule comparison would be needed to isolate the latency effect of recurrence.

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
 #block(text(size:19pt,weight:"bold")[Held product comparison])
 #let r=d.stages.last().decision
 #let t=r.costs
 #let misses=r.gates.pairs().filter(z=>not z.at(1)).map(z=>z.at(0).replace("_"," "))
 #let review_misses=d.reading.families.pairs().filter(z=>not z.at(1).passed).map(z=>z.at(0))
 #if misses.len() > 0 [The registered objective or cost criteria that missed their thresholds were #misses.join(", ").] else [All registered objective and cost criteria passed.]
 #if review_misses.len() > 0 [Masked source review missed its joint criterion for #review_misses.join(", ").] else [All three masked source-review criteria passed.]
 #tbl((1fr,1fr,1fr,1fr,1fr),("Family","Requests","Loop / Base","Gains / losses","Paired p"),r.counts.pairs().map(pair=>{
   let val=pair.at(1)
   (pair.at(0),str(val.n),str(val.total.at(0))+" / "+str(val.total.at(1)),str(val.gains)+" / "+str(val.losses),if val.p==none {"—"} else {fmt(val.p,n:4)})
 }))
 #tbl((1.8fr,1fr,1fr),("Measured cost","Loop","Untouched Base"),(("Wall time, s",fmt(t.candidate.wall_s),fmt(t.control.wall_s)),("Summed first-token wait, s",fmt(t.candidate.first_s),fmt(t.control.first_s)),("Peak MLX GiB",fmt(t.candidate.peak/1073741824),fmt(t.control.peak/1073741824)),("Opening tokens/s",fmt(t.candidate.opening_tps),fmt(t.control.opening_tps)),("Post-first tokens/s",fmt(t.candidate.decode_tps),fmt(t.control.decode_tps))))
 #tbl((1.2fr,.8fr,.8fr,1fr,.8fr),("Masked short family","Loop usable","Base usable","Gains / losses","Paired p"),d.reading.families.pairs().map(pair=>{
   let val=pair.at(1)
   (pair.at(0),str(val.arms.candidate.grounded_usable),str(val.arms.control.grounded_usable),str(val.gains)+" / "+str(val.losses),fmt(val.p,n:4))
 }))

 *Reasoning and tool use.* The loop improved the strict database count by ten and preserved every Base success under that scorer. In the masked review, its ten net gains after a correct supplied result were accompanied by three losses and zero gains following a provider error, and three losses and zero gains following a successful but irrelevant result. On financial questions the loop once selected a program that tracked changes to the actual operands but gave the user the wrong percentage. Among the reasoning requests, one loop response spent its entire 384-token budget repeatedly asking for the main idea of the passage.

 #if "completion" in d [
  *Where the output budget went.* On capped reasoning turns, the loop produced a repeated four-token sequence on #fmt(100*d.completion.strategy.candidate.repeated_fourgrams/d.completion.strategy.candidate.total_fourgrams,n:1)% of positions with four tokens available, versus #fmt(100*d.completion.strategy.control.repeated_fourgrams/d.completion.strategy.control.total_fourgrams,n:1)% for Base. The table counts last turns reaching 384 tokens before or after the model actually emitted its thought-closing token. Both kinds of unfinished response receive the registered score.
  #tbl((1.1fr,1.2fr,1.2fr,1.2fr,1.2fr),("Family","Loop in thought","Base in thought","Loop after close","Base after close"), d.completion.pairs().map(pair=>{
   let m=pair.at(1)
   (pair.at(0),str(m.candidate.at_limit_inside_thought),str(m.control.at_limit_inside_thought),str(m.candidate.at_limit_after_close),str(m.control.at_limit_after_close))
  }))
 ]

 *Long-context evidence.* In the two source-edited locale pairs, the recurrent model repeated the same visible answer on both versions. Base exhausted the output budget on all four documents. Both models processed the full input and neither recovered a correct changed value. This checkpoint therefore qualifies for neither the matched-depth confirmation nor a release.

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

#if "tool_consistency" in d [
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[Can one assistant handle the same question in four histories?])

 The 96 database presentations share 24 human questions and 12 schemas. For the same question we tested an empty history, a correct earlier answer, a successful query that answered the other question, and a real provider error. Each individual success requires the correct final scalar and a query that agrees with the expert on every enrolled alternate database. An earlier result can supply the evidence when it already answers the current question.

 #let meanings=(all_four:"All four presentations",both_repairs:"Both kinds of repair",selective_reuse:"Reuse right evidence and repair wrong evidence",crossed:"Two questions with identical supplied evidence")
 #let t=d.tool_consistency
 #tbl((2.1fr,.55fr,.55fr,.82fr,.67fr),("24 questions or contrasts","Loop","Base","Gains / losses","Schema p"), t.endpoints.pairs().map(p=>{
  let e=p.at(1)
  (meanings.at(p.at(0)),str(e.candidate),str(e.control),str(e.gains)+" / "+str(e.losses),fmt(e.schema_sign_flip_p,n:4))
 }))

 *Both kinds of repair* requires a fresh, successful correction following an erroneous lookup and following a successful but irrelevant lookup. *Selective reuse* additionally tests whether the assistant can leave a genuinely useful result alone. It must deliver the right result directly in that condition and make a successful new query when the previous result is misleading.

 For the crossed test, the previous assistant call and provider response are byte-identical for two different original user questions. The same evidence is relevant for one and misleading for the other. Both answers must succeed. There are two such contrasts per database. The sign-flip probability treats the schema as the independent unit. The 96 requests consumed #fmt(t.field_wall_s.candidate) seconds for the loop and #fmt(t.field_wall_s.control) for Base. The supplement records every question, ambiguity flag, constituent result and elapsed time.
]

#if "calculation" in d and "wording" in d [
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[Do calculations track the source operands?])

 For each financial question we separately replayed the final successful calculator call after changing one source number at a time. The expert program defines the expected change. An unrelated computation followed by a literal copy of the original answer can pass an original-value check. The changed-operand test detects that construction on all 32 expert controls. It tests a self-contained last call and exact quantity equivalence. Rounding, unit conversions and dependencies on earlier calls can reduce coverage.

 #tbl((1.4fr,1fr,1fr,1.2fr),("Financial reports","Originally grounded","Program follows changes","Both conditions"),
   d.calculation.results.pairs().map(p=>{
     let x=p.at(1)
     (p.at(0),str(x.at("registered_grounded",default:0)),str(x.at("program_matches",default:0)),str(x.at("grounded_and_matches",default:0)))
   }))

 #block(text(size:16pt,weight:"bold")[Performance by wording overlap])
 Before generation we marked sources containing a complete repeated field of eight or more normalized words, or any twelve consecutive repeated words. Whole associated sources stay together. The table uses the entire frozen panel, including overlapping material. The four locale documents were outside this audit.

 #tbl((.78fr,1.7fr,.66fr,.85fr,.85fr),("Family","Wording stratum","Requests","Loop / Base","Gains / losses"),d.wording.cells.map(x=>(x.family,x.stratum,str(x.items),str(x.candidate_correct)+" / "+str(x.base_correct),str(x.paired_gains)+" / "+str(x.paired_losses))))
]

#if "learner_prefix" in d [
 #pagebreak()
 #let a=d.learner_prefix
 #block(text(size:19pt,weight:"bold")[Learning to recover from unfinished deliberation])

 *Observed failure and hypothesis.* In the 180-request comparison the recurrent model often reached the output limit before closing its reasoning channel. The previous training conditioned every next-token target on an expert prefix. Its reasoning traces enumerated intermediate questions, and its financial traces often copied complete evidence rows. We tested whether supervising a correct continuation after the model's own unfinished text could teach it to reach a supported answer or execute the appropriate calculation. Ross, Gordon and Bagnell studied the effect of imitation learning on states visited by a learned policy. This study uses a single collection from a frozen parent and includes a separate distribution-preserving loss. #link("https://proceedings.mlr.press/v15/ross11a.html")[Ross et al., AISTATS 2011].

 *Training data.* The parent generated #a.native_tokens tokens on #a.training.rows earlier training sources, using the original thinking-enabled interface and two physical reads for each prediction. The collection used #fmt(a.training.seconds) seconds. We enrolled #a.regions.clean clean turns, #a.regions.learner unfinished learner continuations and #a.regions.retain ordinary rehearsal turns. The learner continuations came from reasoning and financial actions. In the other enrolled groups the parent had already emitted the end of its thought before reaching its assigned prefix length. No continuation is spliced after an already submitted tool call.

 *Optimization.* The model starts from the preceding trained controller. It learns the original human supporting facts on reasoning examples and the verified sequence of operations on calculator actions. All #a.positions target positions across #a.epochs epochs contribute to the fit. Closing a thought and ending the turn each receive three times the ordinary token weight. A full-vocabulary penalty with coefficient #a.kl anchors the two-read prediction to its fixed trained parent. Clean and learner-prefix examples use the same physical attention and the same deployed output head. Neither model receives an injected prefix during evaluation.

 *Prospective test.* The new panel has #a.panel.reasoning_topics previously unused StrategyQA topics and #a.panel.financial_tables TAT-QA development tables. The tokenized openings were reconstructed from the published contexts and every financial derivation was executed independently. The new financial contexts have no shared 16-word sequence with the previous financial material checked here. Their issuer identities are unavailable. The public donor's pretraining exposure is unknown. #link("https://nextplusplus.github.io/TAT-QA/")[TAT-QA data and paper] · #link("https://arxiv.org/abs/2101.02235")[Geva et al., StrategyQA].

 #if "fit" in a and "second_read_gain" in a.fit [
  *Training results.* The fit completed #a.fit.updates optimizer updates on #a.fit.positions supervised positions. Across #a.probes training probes, removing the second physical read increased mean target loss by #fmt(a.fit.second_read_gain,n:3) nats/token. Mean protected-group gain was #fmt(a.fit.protected_gain,n:3) nats/token.
  #if "fit_cost" in a [The complete feature and fitting window used #fmt(a.fit_cost.seconds) native seconds with a #fmt(a.fit_cost.peak/1073741824) GiB MLX peak.]
  #if "teacher_fix" in a [This includes #fmt(a.teacher_fix.native_seconds) seconds for the initial attempt. MLX traced a frozen teacher attention kernel without a supported backward pass. We preserved the original features and evaluated the same fixed teacher distribution before differentiation.]
  #image("learner-prefix-fit.svg",width:95%)
  #text(size:9pt)[Fig. 5. Per-example probe improvement at left and position-weighted optimization measurements at right. The two read counts share the fitted weights.]
  #if a.fit.passed [The fit passed its registered training criteria.] else [The missed training criteria were #a.fit.gates.pairs().filter(x=>not x.at(1)).map(x=>x.at(0)).join(", "). The registered direct test closed at this prerequisite.]
  #if "replay" in a [A fresh process checked #a.replay.feature_hashes feature files and reproduced all three probe paths with maximum absolute error #fmt(a.replay.max_abs_nll_error,n:6).]
 ] else if "fit" in a [
  The native fit stopped after #a.fit.updates updates. The recorded reason was #a.fit.reason.
 ] else [
  The comparison requires improved training-source likelihood for clean and learner-prefix reasoning and calculator actions, protected ordinary language, and a measured benefit from the second physical read.
 ]
 #if "primary" in a [
  #pagebreak()
  #block(text(size:19pt,weight:"bold")[Fresh source reasoning and financial actions])
  #image("learner-prefix-transfer.svg",width:85%)
  #text(size:9pt)[Fig. 6. Matched wall time for every held request on logarithmic axes. Points below the diagonal are faster with the loop. Green identifies a new strict success, red a lost success and gray an unchanged outcome.]
  #tbl((1.15fr,.65fr,.85fr,.85fr,1fr),("Family","Requests","Loop / Base correct","Gains / losses","Paired p"),a.primary.counts.pairs().map(x=>{
   let c=x.at(1);(x.at(0),str(c.n),str(c.total.at(0))+" / "+str(c.total.at(1)),str(c.gains)+" / "+str(c.losses),fmt(c.p,n:4))}))
  #tbl((1.15fr,1fr,1fr),("Resource","Loop","Untouched Base"),("wall_s","first_s","peak","opening_tps","decode_tps").map(k=>{
   let label=(wall_s:"Wall time (s)",first_s:"Summed first-token time (s)",peak:"Peak MLX GiB",opening_tps:"Opening tokens/s",decode_tps:"Decode tokens/s")
   let div=if k=="peak"{1073741824}else{1};(label.at(k),fmt(a.primary.costs.candidate.at(k)/div),fmt(a.primary.costs.control.at(k)/div))}))
  #tbl((1.15fr,.9fr,.9fr,1fr),("Masked source review","Usable loop","Usable Base","Gains / losses"),a.reading.families.pairs().map(x=>{
   let c=x.at(1);(x.at(0),str(c.arms.candidate.grounded_usable),str(c.arms.control.grounded_usable),str(c.gains)+" / "+str(c.losses))}))
  *Completion and transfer.* The loop completed 17 reasoning replies and 20 financial replies, compared with 9 and 11 for Base. Strict reasoning gains came with three individual losses. The opening-rate difference was 2.04 tokens/s in favor of Base. Both source-quality criteria failed. Confirmation, matched training and promotion closed for this checkpoint.

  *Where the calculations failed.* We replayed every call after unmasking. The loop attempted the provider on 21 reports and got a valid calculation on 20. Eight last successful programs contained every required operand magnitude and just one returned the requested value. Base attempted three and returned one correct value. The loop subtracted the right accrued-expense cells and delivered −181 without the thousands scale. On payroll it used a ratio for a requested arithmetic mean; Base executed the mean and delivered 111.4 in the source units. Matching digits alone cannot establish a correct row, operation or direction.

  *Uncertainty.* Resampling the 24 paired topics or reports 10,000 times gave a 95% interval of −12.5 to +37.5 percentage points for the reasoning difference and −12.5 to +12.5 points for financial correctness. Mean elapsed time per request fell by 4.03 s for reasoning (95% interval 3.54–4.47 s) and by 4.85 s for finance (4.25–5.54 s). The loop emitted 6,977 tokens overall and Base emitted 17,860. The elapsed difference reflects both the rate of decoding and the number of tokens produced. Fixed arm order limits hardware timing inference.

  One project reviewer saw the original sources, complete replies and executed calculations before model identity was revealed. Each of the 48 paired judgments was committed before unmasking.
 ]
]

#if "content_read" in d [
 #let a=d.content_read
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[Learning from complete documents])
 A long opening can finish successfully even when the answer is wrong. To train retrieval at the actual operating length, we made each original 16-bit temporal state reusable across questions. The immutable prefix contains all six native attention histories and the linear-state slots for the other layers. Each question has its own writable tail. Shared disk files store complete keys and values. We verify the opening token sequence and the unchanged source bytes at the deciding position.

 *Numerical verification.* On three original training requests we compared full-vocabulary predictions from independent extraction and from branches after a shared prefix. The highest absolute logit difference was #fmt(calc.max(..a.parity.rows.map(r=>r.logits_max)),n:4). All 18 greedy tokens agreed. After an intervening different continuation, re-running each first branch reproduced its features exactly.

 *Full-length measurements.* We processed #a.full.prefix_tokens common tokens of an original Deno Cargo.lock, then twelve different requests for versions of packages. The latest deciding version lies at positions 146,741–146,745 (zero-based). The earliest starts at 5,173, more than 142,000 tokens before its answer prompt. One shared final-layer KV file used #fmt(a.full.document_bytes/1048576) MiB. Staging the complete source took #fmt(a.full.staging_s) seconds. All extraction and replay took #fmt(a.full.seconds) seconds with a peak of #fmt(a.full.peak_bytes/1073741824) GiB. We repeated the first branch after the other eleven and reproduced it byte-for-byte. This experiment measures training-state construction. Answer accuracy requires independent free generation.
 #image("full-document-branches.svg",width:90%)
 #text(size:9pt)[Fig. 7. Each point is a different original package question given the whole lockfile. Times cover independent continuation and serialization after shared staging. The first branch also saved and hash-checked the common final-layer file.]

 Liu et al. found that models' ability to use an input can vary with the position of its deciding information. We use naturally occurring positions and preserve complete versioned sources so that changed values can be tested directly. Wang et al. trained a memory reader alongside a frozen backbone. Our content reader can attend to every original final-layer value at each recurrent visit. It requires neither an external corpus nor a retrieved excerpt. #link("https://arxiv.org/abs/2307.03172")[Liu et al., Lost in the Middle] · #link("https://papers.nips.cc/paper_files/paper/2023/hash/ebd82705f44793b6f9ade5a669d0f0bf-Abstract-Conference.html")[Wang et al., LongMem].
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[Choosing evidence for the current question])
 *Architecture.* The preceding primary identified frequent executable programs that used a different row, year or operation. We added a shared learned query into the original contextual value vectors after each of the two recurrent visits. Its query depends on the updated state. A learned projection writes the weighted result back into the same state. We initialize this projection at zero and freeze the donor and all previously fitted recurrent parameters. Each visit uses the same two new matrices. Only causal entries are visible. The ordinary vocabulary projection and public tool execution remain in the serving path.

 *Sources.* Each of 160 different TAT-QA training reports contributes two human questions with different verified answers and different sets of uniquely located numerical inputs. We execute both programs on the same provider that the models receive. HotpotQA supplies human comparison questions and supporting sentence indices. We keep all ten original passages in their published order. The 24 reserved validation questions have 240 distinct article titles, all absent from the 320 reasoning training examples. The 24 additional held TAT-QA reports each supply their original source and authored question. No held context shares a normalized 16-word span with the new training openings checked here. Public pretraining exposure and issuer independence are unknown. #link("https://hotpotqa.github.io/")[Yang et al., HotpotQA] · #link("https://nextplusplus.github.io/TAT-QA/")[Zhu et al., TAT-QA].

 *Tool histories.* A separate 324 authentic human database requests come from 60 earlier training schemas. Each member of a pair has the same type of answer but a different scalar. An executed lookup can answer one user and mislead the other. The model learns both to deliver a supported human answer and to make the correct fresh call. Schema declarations, expert statements and returns come from the original local database files.

 *Protocol.* The 1,800 training turns contain #a.source.training_opening_tokens opening tokens and 1,352 source-location labels per epoch. Three epochs provide 253,950 supervised output positions. We measure the probability placed on the correct operand or sentence at its output decision, full-vocabulary target loss, protected language and the effect of removing the second read. A passing fit and independent replay are prerequisites for a matched free-generation comparison. Source selection and thresholds were committed in advance.
 #if "fit" in a and "gates" in a.fit [
  *Training outcome.* The native optimizer completed #a.fit.updates updates. Source-location loss improved by #fmt(a.fit.citation_gain,n:3) nats per marked decision. The two-visit target-loss gain relative to a single visit was #fmt(a.fit.second_gain,n:3) nats/token. #if a.fit.passed [The fit passed all registered checks.] else [The missed checks were #a.fit.gates.pairs().filter(v=>not v.at(1)).map(v=>v.at(0)).join(", "). The direct test for this fit closed.]
 ] else if "fit" in a [*Training outcome.* The native worker stopped after #a.fit.updates updates. The archived reason was #a.fit.reason.]
]

#if "content_read" in d and "replay" in d.content_read [
 #let a = d.content_read
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[When a more accurate read worsens the answer])
 The first content reader improved the likelihood of its cited evidence by #fmt(a.fit.citation_gain,n:3) nats per marked decision. Its mean protected target loss rose by #fmt(-a.fit.protected_gain,n:3) nats per token. The native optimizer processed all #a.fit.positions output positions. After the first update, the next batch's divergence from the frozen parent reached #fmt(a.at("content-read_early").at(1)) nats/token. The first batch's divergence was below 0.001. Different batches contain different examples. The change identifies an early instability. A controlled intervention is needed to identify its cause. The unchanged trained parent tensors and all #a.replay.feature_hashes feature files passed independent verification. Four newly generated feature files were identical. Three physical paths reproduced their probe losses with maximum absolute error #fmt(a.replay.max_error,n:6).
 #image("content-write-optimization.svg",width:95%)
 #text(size:9pt)[Fig. 8. The left panel follows the first 16 optimizer batches. The right panel compares the untrained and trained checkpoint on the same preregistered group probes. Positive values mean a better trained prediction.]
 #parbreak()
 *Bounded write.* A separate fit starts again at the same learner-prefix checkpoint. It normalizes the read vector and bounds each FP32 output coordinate by one sixteenth of the current state's root mean square before casting back to the model's precision. The original seed, data order, loss and all eleven fit criteria are fixed. An extra gate limits divergence from the parent to 0.5 nats/token on each of the first 16 batches. The 48 held requests remained unopened at registration.
 #if "bounded" in a [
  #parbreak()
  *Result.* This fit completed #a.bounded.updates optimizer updates. #if "gates" in a.bounded [Peak early divergence fell to #fmt(calc.max(..a.at("bounded-read_early")),n:3) nats/token. Its reasoning, calculator-action and quantity gains were #fmt(a.bounded.gains.reasoning,n:3), #fmt(a.bounded.gains.calculate,n:3) and #fmt(a.bounded.gains.quantity,n:3) nats/token. Its source-location gain was #fmt(a.bounded.citation_gain,n:3) nats per decision. The second physical visit contributed #fmt(a.bounded.second_gain,n:3) nats/token. The protected mean change was #fmt(a.bounded.protected_gain,n:3), including a #fmt(-a.bounded.gains.full_rows,n:3) nats/token loss on the full-row group. Using the existing immutable features, this fit consumed #fmt(a.bounded_jobs.first().seconds) native seconds and peaked at #fmt(a.bounded_jobs.first().peak_bytes/1073741824) GiB. The earlier feature-construction and unbounded-fit window consumed #fmt(a.jobs.first().seconds) seconds together. #if a.bounded.passed [It qualified for the prospective generation comparison.] else [Its failed fit checks were #a.bounded.gates.pairs().filter(x=>not x.at(1)).map(x=>x.at(0)).join(", ").]] else [The stopping reason was #a.bounded.reason.]
 ]
 #if "full_document_adjoint" in a [
  #parbreak()
  *Native full-context differentiation.* The first differentiated training example processed 147,432 common document tokens and all seven supervised suffix positions. The complete gradient norm was #fmt(a.full_document_adjoint.history.first().gradient). MLX peak allocation was #fmt(a.full_document_adjoint.peak_bytes/1073741824) GiB, above the 3 GiB limit. The fit stopped after this update. None of the four held packages were scored. This closed the Foundry and Agave release comparison for these weights.
 ]
 #if "full_document_kernel" in a [
  #parbreak()
  *Reducing native peak allocation.* A separate kernel computed per-partition softmax moments over the same complete keys and values. It needs the softmax-weighted key, incoming value gradient, and their product to recover the query derivative. On the original 147,481-position example its gradient norm was #fmt(a.full_document_kernel.gradient,n:4) versus #fmt(a.full_document_kernel.reference.gradient,n:4), and its peak allocation was #fmt(a.full_document_kernel.peak_bytes/1073741824) GiB. The target NLL differed by #fmt(calc.abs(a.full_document_kernel.nll - a.full_document_kernel.reference.nll),n:7) nats/token. This single-example equivalence check used no held questions.
 ]
 #if "document_generation_parity" in a [
  #parbreak()
  *User-facing prefix verification.* On a real financial report, the independently forked public serving path reproduced all 48 greedy output tokens of fresh prefill across both real models, including repeated questions after a different branch. All six physical attention histories had the expected lengths. The engineering test used #fmt(a.document_generation_parity.seconds) seconds.
 ]
]

#if "content_read" in d and "delivery" in d.content_read [
 #let a=d.content_read
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[What one checkpoint delivers on one request])
 A user experiences cold load, opening, output generation and answer quality together. We matched each request by source identity and counted how often one recurrent checkpoint beat Base simultaneously on all five recorded costs. Faster total delivery can accompany a longer first wait, a slower opening or a wrong answer. The five costs are elapsed wall time, first-output latency, peak native allocation, throughput on newly processed opening tokens, and throughput after the first generated token. Each row uses only the paired requests belonging to its own weight digest.
 #let pretty=(staged-query-transfer-01:"Source repair",learner-prefix-transfer-01:"Learner prefix",bounded-read-transfer-01:"Bounded write")
 #tbl((1.1fr,.65fr,1fr,1fr,1fr),("Checkpoint","Requests","All five costs better","All five better and correct","No individual regression"),a.delivery.pairs().map(x=>{
    let r=x.at(1).all; (pretty.at(x.at(0)),str(r.n),str(r.wins_every_cost),str(r.all_costs_and_delivered),str(r.no_item_regression))}))
 #parbreak()
 *Interpretation.* “No individual regression” allows ties and requires all five delivery costs, objective correctness and completion to be at least as good as Base for that request. A lower wall time on an unanswered request contributes no credit to the “and correct” column. The original source-masked reading remains a separate requirement for usable explanations and financially meaningful units.
 #tbl((1.1fr,.8fr,.8fr,.8fr,.8fr),("Checkpoint","Slower first wait","Slower opening rate","Lost correct answer","Lost completion"),a.delivery.pairs().map(x=>{
    let r=x.at(1).all;(pretty.at(x.at(0)),str(r.first_s_losses),str(r.opening_tps_losses),str(r.success_losses),str(r.completed_losses))}))
 #parbreak()
 *Reproducibility.* The accompanying source-delivery-requests.csv contains both raw measurements, both outcome indicators and the exact weight identity for each matched request. source-delivery-summary.csv retains every family. Timings reflect a single fixed arm order on this host. The observed strict inequalities include small hardware-dependent differences. They are descriptive. The precommitted statistical decisions use paired outcome differences across independent source units.
 #parbreak()
 *Release conclusion.* Establishing deployable superiority still requires the same learned weights to clear fresh reasoning and tool-use comparisons, maintain useful answers under complete document changes beyond 128K, preserve older abilities, and replicate with a second training seed. A matched, equally trained ordinary-depth control is required to attribute an improvement to recurrence. Results from different weight identities do not supply that evidence.
]

#if "content_read" in d and "bounded_primary" in d.content_read [
 #let a=d.content_read
 #let r=a.bounded_primary
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[Free generation after bounding the read])
 The registered comparison used the same 48 original openings, official thinking template, order, three-turn limit and 384-token allowance per turn for both systems. Every actual response and executed calculator observation was preserved. All 48 anonymous paired judgments were recorded before model identity was exposed. The loop and Base came from their pinned, separately verified weights.
 #tbl((1.15fr,.8fr,1fr,1fr,1fr),("Family","Requests","Correct loop / Base","Gain / loss","Paired p"),r.counts.pairs().map(x=>{
   let c=x.at(1);(if x.at(0)=="strategy"{"Comparison reasoning"}else{"Financial actions"},str(c.n),c.total.map(str).join(" / "),str(c.gains)+" / "+str(c.losses),fmt(c.p,n:4))}))
 #tbl((1.15fr,1fr,1fr,1fr),("Masked assessment","Usable loop / Base","Gain / loss","Unsupported loop / Base"),a.bounded_reading.families.pairs().map(x=>{
   let c=x.at(1);(if x.at(0)=="strategy"{"Comparison reasoning"}else{"Financial actions"},str(c.arms.candidate.grounded_usable)+" / "+str(c.arms.control.grounded_usable),str(c.gains)+" / "+str(c.losses),str(c.arms.candidate.unsupported)+" / "+str(c.arms.control.unsupported))}))
 #image("bounded-read-transfer.svg",width:83%)
 #text(size:9pt)[Each point is a matched request. The axes show loop and Base wall seconds. Green marks a new strict success, red a lost strict success and gray an unchanged result.]
 #parbreak()
 #tbl((1.25fr,1fr,1fr),("Measured deployment cost","Loop","Untouched Base"),("wall_s","first_s","peak","opening_tps","decode_tps","output").map(k=>{
  let label=(wall_s:"Wall seconds",first_s:"Summed first-token seconds",peak:"Peak MLX GiB",opening_tps:"New opening tokens/s",decode_tps:"Post-first tokens/s",output:"Total emitted tokens")
  let div=if k=="peak"{1073741824}else{1};(label.at(k),fmt(r.costs.candidate.at(k)/div),fmt(r.costs.control.at(k)/div))}))
 *Decision.* #if r.passed and a.bounded_reading.passed [The registered product comparison and source-quality assessment both passed.] else [The complete product and masked quality requirement failed. Preregistered confirmation and release evaluation of this checkpoint closed.]
]


#if "content_read" in d and "bounded_primary" in d.content_read [
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[Where a valid tool call lost the answer])
 All 24 financial requests used complete reports. Both decoders could select their own expression, inspect the real calculator return and answer in the source units. We replayed every executed expression. Among final successful calls, the loop used all required operand magnitudes on five reports and Base on one. Two loop calls and zero Base calls returned the right amount. The project reviewer identified one correct delivered loop conclusion and two Base conclusions when a supporting call was optional. Requiring a correct executed call and its delivered quantity credited one loop answer and zero Base answers.
 #parbreak()
 *A successful delivery.* The loop subtracted 9,778 from 4,518 and reported the decrease of 5,260 awarded shares. On another report it chose the correct expression, 598 divided by 1,690 times 100. The calculator returned the exact fraction 460/13, approximately 35.38%. The model delivered 46.0%. On an appliances question it subtracted the two calendar years and returned one. The requested source values were 680 and 774 million. One failure occurred after a correct calculation, the other before execution.
 #parbreak()
 *What the reasoning score measures.* The registered choice scorer credited 13 loop and 15 Base answers. Reading the entire response for the actual conclusion credited 13 and 16. One Base answer opened affirmatively about one film and subsequently drew the correct comparison across both films. Requiring a complete supported explanation reduced each score to seven. The 24 original topics are paired, with five loop gains and seven losses under the frozen choice rule. A 10,000-draw topic bootstrap gives a 95% percentile interval from −37.5 to +20.8 percentage points for the loop’s accuracy difference. The corresponding financial interval spans 0 to +12.5 points. With one discordant financial success, the exact paired two-sided test gives p = 1.
 #parbreak()
 *Delivery.* Median reasoning wall time was 1.02 versus 4.85 seconds and median financial wall time was 0.89 versus 5.80 seconds. The loop finished 19 reasoning and 19 financial requests. Base finished 17 and 11 respectively. Across the 48 requests the loop emitted 7,101 tokens versus 16,920. Both the shorter emissions and the faster post-first token rate contributed to lower total time. All eight reasoning questions both systems answered correctly were faster on the loop. The source-level intervals and all 96 individual request records accompany this paper.
 #parbreak()
 *Limitations.* One project reviewer assessed the anonymous responses. Donor training and weight precision differ between the deployed candidates. A source-level accuracy difference at these sample sizes is imprecise, and a single paired runtime per request does not establish performance under concurrent use. The failed comparison closed the independent seed, equal-training depth control, broad retention and long-context confirmation for these weights.
]

#if "source_copy" in d [
 #let a=d.source_copy
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[Can a second read recover a source token?])
 We measured actual source probabilities at every annotated decision in 320 training questions and 320 training calculator calls. The query came from the same bounded weights after one or two physical visits. We normalized its scores over every causally available value, then summed the attention on all occurrences of each token ID in the original opening. The frequency baseline always chooses the most frequent input token. Under teacher forcing, the relevant evidence span can contain the right evidence without containing the next word of the answer.
 #image("source-address-probabilities.svg",width:100%)
 #text(size:9pt)[Two arms of a mechanism intervention on the same weights and training positions. Vocabulary accuracy is the ordinary next-token prediction from that recurrent state.]
 #parbreak()
 On the 678 financial decisions the second read placed its maximum in the cited span 384 times. Its source-only next token was correct on 149 decisions, against 23 for the frequency baseline and 380 for the native vocabulary. Its probability on the target ID exceeded uniform source sampling on 530 decisions. On the 674 reasoning decisions it placed the maximum inside the cited span 391 times, but copied the next token correctly only twice. The learned ordinary vocabulary was correct on 425 decisions. An input word can also coincide with the target by chance or appear in several unrelated locations. The audit motivated a learned gate between normalized source and vocabulary probabilities. See et al. applied this type of mixture to grounded summarization. Here the source distribution reads the recurrent decoder’s complete causal history. #link("https://aclanthology.org/P17-1099/")[See, Liu and Manning.] #link("https://arxiv.org/abs/1609.07843")[Merity et al., Pointer Sentinel Mixture Models.]
 #parbreak()
 *Consequences for training.* A larger, frozen model was given 24 financial training follow-ups, each with an original report, prior conversation and actual calculator results. It received no current gold program. In #fmt(a.teacher.seconds) native seconds, #a.teacher.accepted calls returned an appropriate number through a substantive operation involving a previous operand. Source review retained #a.accepted for student training. One numerically accepted trace mixed FIFO and LIFO inventory balances. Another treated the starting investment in May as the end-of-year index value. The complete trace and reason for each exclusion accompany the study. These are training-source results.

 #pagebreak()
 #block(text(size:19pt,weight:"bold")[From matched prompts to conversations that can go wrong])
 The prospective cohort contains 24 original two-question reading conversations and 24 original two-question financial conversations. Each model answers the first question, then receives the second question after its own visible answer and executed tool history. On a separate presentation, both models receive the identical published first answer and identical follow-up token IDs. This measures how errors propagate in a live dialogue while retaining a matched-input control on the same source. Each report issuer and reading source contributes one episode. Twenty-two financial issuers appeared on different reports in earlier evaluations. No enrolled report page ID occurs in earlier project data or source inventories.
 #parbreak()
 *Material.* CoQA passages and questions were collected in conversation between two people. We enrolled two consecutive open answers on literature and Wikipedia passages, with two independently recorded matching judgments per selected development answer. ConvFinQA supplies human-written financial follow-ups and published arithmetic programs. All 48 enrolled program results replay on the real calculator. Before enrollment, source checks rejected examples with an incorrect fiscal year, a different divisor in the annotation, or a question about time applied to two contemporaneous stock categories. The eligibility restrictions and the possibility of donor pretraining on public datasets limit generalization. #link("https://stanfordnlp.github.io/coqa/")[Reddy et al., CoQA.] · #link("https://aclanthology.org/2022.emnlp-main.421/")[Chen et al., ConvFinQA.]
 #parbreak()
 *Model.* The same learned loop runs twice at every output position. A learned gate mixes the corrected vocabulary distribution with the distribution over causal source token IDs. Repeated IDs accumulate probability at every matching position. Target likelihood includes the ordinary vocabulary when the token never occurs in the prefix. The model sees original source values and previously consumed tokens during generation. Its training includes #a.material.rows turns and #a.material.tokens target positions per epoch. All previous recurrent weights stay frozen. A bound limits the additional language residual. Each decoder receives the official thinking template, up to 384 tokens per turn, and the same public calculator.
 #parbreak()
 *Decision rule.* Each source must be correct at the first turn, on the model's own follow-up and on the matched follow-up. Promotion requires at least five net new source successes in each family, no lost source successes, a two-sided exact paired p at most 0.05, source-supported outputs, non-inferior completion and better costs on all five measures. The source review assesses units and meaning in addition to the frozen automatic scorer.
 #if "fit" in a [#parbreak() *Fit.* The native stage completed #a.fit.updates optimizer updates. #if "gates" in a.fit [The vocabulary-plus-source read gained #fmt(a.fit.copy_gain,n:3) nats/token over the language path alone across its conversation-reading and calculator probes. A second visit contributed #fmt(a.fit.second_gain,n:3) nats/token. #if a.fit.passed [It advanced to the independent generation stage.] else [The failed checks were #a.fit.gates.pairs().filter(x=>not x.at(1)).map(x=>x.at(0)).join(", "). Its held comparison closed.]]]
 #if "locale" in a [#parbreak() *Whole-document learning.* A separately initialized two-read model trained on the natural differences between two original Spanish locale files, with 160,210 and 169,951 common input tokens. It completed #a.locale.updates native updates in #fmt(a.locale.seconds) seconds, peaking at #fmt(a.locale.peak_bytes/1073741824) GiB. #if "training" in a.locale [Mean training answer and source-location NLL improved by #fmt(a.locale.training.answer_gain,n:3) and #fmt(a.locale.training.address_gain,n:3) nats. It preferred the correct first changed token under both source versions on #a.locale.training.both of six training fields.] #if "held" in a.locale [The trained model preferred the respective correct token on both versions for #a.locale.held.both of four untouched field names. Held address gain was #fmt(a.locale.held.address_gain,n:3) and the second-visit answer gain was #fmt(a.locale.held.second_gain,n:3) nats.] else [The training gate closed the held fields.] This study measures teacher-forced preferences on two versions of one file.]
]


#if "source_copy" in d and "locale" in d.source_copy and "held" in d.source_copy.locale [
 #let a=d.source_copy.locale
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[A field can change the answer without improving its source address])
 The inputs were two complete, unmodified releases of the Spanish locale file. Each question named the same XML selector in both files, with different correct values. We sealed six training and four held field names before fitting. We evaluated every visible value token and the turn terminator. At the first token where the two actual versions differed, the contrastive objective required the token from the active version to outrank the token from the other version. The source loss measured the probability assigned to the actual XML value. Each update used both real recurrent reads and the full native key/value history.
 #image("changed-locale-held.svg",width:100%)
 #text(size:9pt)[Eight held presentations. Teal and orange identify source versions. A positive left-hand margin favors the correct token after the supplied answer prefix. Lower right-hand values indicate more probability on the correct source span. The parent series uses the initial weights with two reads.]
 #parbreak()
 *Results.* The initial model favored both correct alternatives on #a.dual.initial of four held fields. With trained weights one visit succeeded on #a.dual.one and two visits on #a.dual.two. The two-read model favored both versions of the full weekday date patterns ending in yMMMMEd and yMMMEd, reversing two decisions missed by initialization. The era date format was already correct under both releases. For the Acre time zone, the two trained predictions still missed the old capitalized form. The two-read held source loss worsened by #fmt(-a.held.address_gain,n:3) nats on average. On the older era format its value rose from 10.39 to 31.27 nats. All eight trained two-visit answer losses were below their initial values, and the second visit reduced held answer loss by #fmt(a.held.second_gain,n:3) nats/token on average.
 #parbreak()
 *Interpretation.* This is evidence that the trained loop can change its next-token preference across real 160K–170K sources. The four field names are drawn from one locale and several share date-pattern structure. The experiment supplied the correct prefix up to the first difference. A delivered answer in free generation, a direct result against untouched Base and generalization to unrelated documents were outside this result. The preregistered evidence-location criterion failed. A training-only follow-up masked the exact value in both reads or an equal-length nearby span. The correct-span mask reduced the preference margin more on eight of 12 presentations, averaging 0.163 logits. The preregistered rule required nine of 12 and a mean above 0.1.
]


#if "source_copy" in d and "reading" in d.source_copy [
 #let a=d.source_copy
 #let run=a.pair
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[Does the conversational gain survive a model’s own history?])
 Both systems answered 48 original first questions, followed by 48 requests containing their own prior conversation and 48 matched requests with the same first answer supplied. The figure counts correct delivered conclusions in the source-masked review and total wall time. The table compares strict automatic success with fully usable answers.
 #align(center,image("source-copy-conversation-results.svg",width:82%))
 #tbl((1.2fr,1fr,1fr,1fr,1fr),("Family","All three strict (loop / Base)","Gained / lost","Exact paired p","All three usable (loop / Base)"),run.counts.pairs().map(x=>{
   let k=x.at(0); let r=x.at(1).all;let g=a.reading.families.at(k);(k,r.total.map(str).join(" / "),str(r.gains)+" / "+str(r.losses),fmt(r.p,n:4),str(g.usable.candidate)+" / "+str(g.usable.control))}))
 #tbl((1.5fr,1fr,1fr),("All 144 user requests per system","Loop","Untouched Base"),("wall_s","first_s","peak","opening_tps","decode_tps","output").map(k=>{
  let label=(wall_s:"Total wall seconds",first_s:"Summed first-output seconds",peak:"Peak MLX GiB",opening_tps:"Opening tokens/s",decode_tps:"Post-first tokens/s",output:"Total emitted tokens"); let n=if k=="peak"{1073741824}else{1};(label.at(k),fmt(run.costs.candidate.at(k)/n),fmt(run.costs.control.at(k)/n))
 }))
 *Decision.* The loop won all five costs on #a.delivery.all_costs of #a.delivery.requests paired presentations. It won all five and strictly answered the question on #a.delivery.both_faster_and_correct. #if a.reading.joint [The joint automatic and masked release gate passed.] else [The preregistered joint deployment and masked source-quality requirement failed. These weights do not establish a general release improvement.]
 #parbreak()
 The loop reported 39 percent growth for Entergy Texas after its own calculator returned 390/4033, about 9.67 percent. It also recovered from an invalid cross-call reference on an acquisition report and correctly delivered 18,658 thousand. The same follow-up with the published first answer supplied yielded 14,567 thousand, omitting customer-related assets. Masked review credited correct financial conclusions on 3 of 72 loop presentations and 28 of 72 Base presentations. The loop emitted 8,063 tokens against 40,688 for Base. Both length and decode rate affect total time.
 #parbreak()
 The source-copy-requests.csv file includes every timing and terminal status. The conversation file retains joint outcomes by source and history. The interval file resamples whole passages and issuers, keeping models and presentations paired. The direct comparison retains full prompt IDs even when the previous answer differs.
]


#if "source_copy" in d and "operative" in d.source_copy [
 #let a=d.source_copy.operative
 #pagebreak()
 #block(text(size:19pt,weight:"bold")[Where a correct financial computation is lost])
 We replayed the financial tool calls after completing the registered comparison and source-masked review. The 24 reports contain two original questions each. Their first, live and fixed-history presentations yield 72 requests per system. Each model saw the original report and its real calculator returns during generation. The perturbations below were applied afterward to the archived programs. They did not add observations or alter either model’s output.
 #image("source-copy-program-audit.svg",width:100%)
 #text(size:9pt)[A successful callback can return the wrong amount. The final two rows require the program to follow the annotated operative inputs when those inputs change. The final row also requires strict correct delivery on the original request.]
 #parbreak()
 *Method.* For each nonconstant number in the published derivation, shift that input independently by +11/7 and −5/13. Replay the expert and model expressions with exact arithmetic. Require agreement on the original and every change that affects the expert's result. For percentages, a common factor of 100 may distinguish ratio and percentage-point programs. Of the 48 original questions, #a.controls had at least one testable operand, producing #a.perturbations effective expert interventions. The remainder had only operands labeled as constants. Every eligible expert program passed its own checks, and substituting a fixed expression that copied its original result passed none.
 #parbreak()
 *Results.* The loop received at least one valid return on 52 of 72 presentations. The last successful program produced the correct original amount on eight. Seven tracked the operative inputs under perturbation. Strict delivery succeeded on two of these seven. Base produced one operative program and left its answer unfinished. Five of the seven loop presentations with an operative program were lost between execution and delivery. Selecting and combining the right operands remains the larger absolute failure.
 #parbreak()
 *A limit of single-call attribution.* In the acquisition example, after an invalid reference the loop successfully added 1,031 to the previous total of 17,627 and delivered 18,658 thousand. Changing the original goodwill amount in the final call alone leaves its pasted subtotal fixed. This genuine multi-turn success passes the registered delivery criterion and fails the single-call perturbation. Equivalent programs using differently scaled inputs can also fail this retrospective supplement. It is reported alongside the registered results.
 #parbreak()
 *Subsequent training intervention.* The new training corpus contains 320 original calculations on 160 financial reports. We executed both annotated programs on each report and combined each original question with its own receipt and the other question’s receipt. Training targets cover making the initial call, delivering a legitimate result, repairing a misleading observation and delivering after repair. The 16-report development set is withheld from the new intervention, though earlier project training included its source material. The registered decision also measures reasoning, reading and tool-history retention. A new product result requires independent free generation and comparison against Base.
 #parbreak()
 *Opening cost.* The first turn and common-history follow-up have identical input tokens across arms. Across these 96 openings (90,885 tokens), the loop processed 2,924 tokens/s against 2,907 for Base. The loop-minus-Base difference of 17 tokens/s had a 95 percent interval of −10 to +49 when resampling whole sources. The loop also opened 72 real callback continuations at 2,475 newly processed tokens/s compared with two for Base at 2,541. The registered all-turn rates remain 2,889 and 2,906. Timing varied across sources on a single host.
 #parbreak()
 *Related work.* Zhu et al. collected the financial questions and worked derivations used for this intervention. Qiao et al. studied selective tool use through execution feedback. Dong, Zhao and Chen used gold derivations and program execution to filter teacher supervision for a 7B student. Our within-report swap asks whether the decoder can identify an observation that is numerically valid for a different human question. The final output must still express the correct quantity and scale. #link("https://aclanthology.org/2021.acl-long.254/")[TAT-QA.] · #link("https://arxiv.org/abs/2305.13068")[TRICE.] · #link("https://arxiv.org/abs/2607.14709")[Gold-guided programmatic distillation.]

]
