#set document(title:"Does a trained recurrent Qwen improve assistance across tasks?",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.76in,y:.70in),numbering:"1")
#set text(font:"Libertinus Serif",size:10.7pt)
#set par(leading:.43em,spacing:.55em)
#let d=json("conditional-primary.json")
#let tint=(x,y)=>if y==0 {rgb("#dcece6")} else if calc.odd(y){rgb("#f0f3f2")}
#let r(x,n)=str(calc.round(x,digits:n))
#let pair(v)=str(v.at(0))+" / "+str(v.at(1))
#let cs=d.costs
#let a=cs.candidate
#let b=cs.control
#let tbl(cols,head,rows)=table(columns:cols,stroke:none,fill:tint,inset:4pt,..head.map(v=>[*#v*]),..rows.flatten().map(x=>[#x]))
#let label=(changed_table:"Changed tables",database:"Database questions",instruction:"New instructions",open:"Open assistance",natural_inline:"Wikipedia inline",natural_mounted:"Wikipedia via SQLite")
#text(size:23pt,weight:"bold")[Does a trained recurrent Qwen improve assistance across tasks?]
#linebreak()
#text(size:13pt)[A paired study of native tool use, source-grounded reasoning and response cost]
#linebreak()
LoopQwen-MLX project · November 2026

*Abstract.* An instruction-initialized recurrent Qwen with genuine re-entry into four shared layers solved #d.objective.at(0) of 256 objective requests. Untouched BF16 Qwen Base solved #d.objective.at(1). The new model gained #d.gains_losses.at(0) correct answers and lost #d.gains_losses.at(1) Base successes. On the full 340 requests the loop took #r(a.wall_s,2) seconds and Base took #r(b.wall_s,2). For novel Wikipedia questions presented both inline and via a real database, the source-based review credited #pair(d.whole.natural.scores) whole questions out of 24. #if d.eligible [The registered combined product gate passed. Independent replication, matched depth and full-context qualification are required for release.] else [The registered combined product gate failed. Independent replications, the matched depth study and the long-context release test are closed for these weights.]

#image("conditional-primary-quality.svg",width:100%)
#text(size:9pt)[Figure 1. Left, percentage-point changes in objective success, with descriptive 95% intervals obtained from 10,000 resamples of intact source groups within each family. Right, masked assessment of full responses. The mounted Wikipedia measure additionally requires an actual complete query result containing two source-attested values. Each panel uses its own preregistered denominators.]

*Design.* We compared the same ordered, token-identical openings under deterministic greedy generation, with thinking enabled on each decoder, three turns of up to 384 new tokens, the same tools and scorers, and one #d.environment.chip with #r(d.environment.physical_memory_bytes/1073741824,0) GiB unified memory. The runtime used MLX #d.environment.libraries.mlx, MLX-LM #d.environment.libraries.at("mlx-lm") and Python #d.environment.python. Of the 340 requests, 268 retain earlier prompts in order. Seventy-two are new, with 24 independent Wikipedia pages shown twice and 24 previously unevaluated original instructions from eight categories. The candidate was fitted before evaluation on the 72 new requests. Earlier results on the 268 retained requests informed model development, which limits claims of independent replication on that subset. The full packet preserves every visible answer, status, generated ID, executed call and provider observation.

*Comparison and attribution.* The delivered candidate uses four-bit weights of a public instruction sibling and 1,377,280 learned readout parameters. Its 28 execution sites revisit the last four shared layers after ordinary layer 24. Its control is the pinned original BF16 Base at ordinary depth. Initialization, quantization, optimization and recurrence differ together. A product-level advantage on a measure cannot identify the independent effect of a second visit. The fit supplement reports the same-weight intervention at all five readout inputs.

#pagebreak()
#text(size:20pt,weight:"bold")[Objective accuracy and reasoning in the source]
#tbl((2.0fr,.7fr,.7fr,.7fr),("Source family","Requests","Loop","Base"),d.families)
Across the 154 objective requests with tools available, scores were #pair(d.tools). On 102 objective requests without tools they were #pair(d.ordinary). Among the #d.both_finished.requests requests both models finished, the scores were #d.both_finished.loop_correct / #d.both_finished.base_correct. Across all objective requests, both models solved #d.joint.both_correct and neither solved #d.joint.neither_correct. #d.joint.additional_correct_and_all_three_costs_lower of the #d.gains_losses.at(0) newly correct responses also had lower wall time, first-token wait and peak allocation.

*Masked reading of reasoning questions.* We fixed the same 12 ROPES backgrounds and 12 QuAC passages used in the preceding source review before inspecting current results, and included both source-linked questions in each case. The reviewer read the actual source, dialogue history, annotations and hash-shuffled answers. The reviewer knew the earlier model comparison and could potentially recognize a familiar Base answer. We recorded a reason for each item before revealing arms.

#let reason=d.reasoning.counts
#let rf=(("effects","Causal comparison",24),("human_first","First question",12),("human_followup","Follow-up",12))
#let gr(f,k)=pair((reason.candidate.at(f).at(k,default:0),reason.control.at(f).at(k,default:0)))
#tbl((1.5fr,.5fr,1fr,1fr,1fr),("Reasoning task","Read","Exact","Fully usable","Unsupported"),rf.map(v=>(v.at(1),v.at(2),gr(v.at(0),"automatic_exact"),gr(v.at(0),"fully usable"),gr(v.at(0),"unsupported"))))
#text(size:9pt)[Table 2. Each pair is loop / Base. These are 48 paired readings. Automatic full-family F1, on all 24 first questions, 24 follow-ups and 48 causal questions, was #r(d.f1.candidate.human_first,3) / #r(d.f1.control.human_first,3), #r(d.f1.candidate.human_followup,3) / #r(d.f1.control.human_followup,3), and #r(d.f1.candidate.effects,3) / #r(d.f1.control.effects,3), respectively.]

Both questions from the same background were fully usable on #pair(d.reasoning.sources.causal.whole) of 12 causal sources (_p_ = #r(d.reasoning.sources.causal.whole_exact_p,4)) and #pair(d.reasoning.sources.conversation.whole) of 12 conversations (_p_ = #r(d.reasoning.sources.conversation.whole_exact_p,4)). The tests independently flip the signs of whole-source differences. Small families, shared tasks, a single reviewer and one checkpoint limit generalization. The request CSV and source bootstrap tables preserve all individual gains and losses, family-wise adjusted _p_ values and conditional-on-completion counts.

#pagebreak()
#text(size:20pt,weight:"bold")[Did the answer follow the tool result?]

For the 156 primary readings, the reviewer saw the original request, the complete source table or human reference where applicable, source-specific issues documented before generation, and the distinct visible answers in hash-shuffled order. Identical answers shared a grade. A complete answer had to address the user’s question, use the relevant evidence and avoid material unsupported additions. The reviewer committed grades, preferences and request-specific reasons before unmasking arm identity. This project-internal reading has no independent second reviewer.

#tbl((1.7fr,.55fr,1fr,1fr,1fr),("Source family","Items","Fully usable","Preferred","Unsupported"),d.reading.map(v=>(label.at(v.family),v.items,pair((v.candidate_grounded_usable,v.control_grounded_usable)),pair((v.candidate_preferred,v.control_preferred)),pair((v.candidate_unsupported,v.control_unsupported)))))
#text(size:9pt)[Table 3. Entries are loop / Base. Preferences exclude ties. A correct answer without a supporting query is ineligible on mounted Wikipedia questions. The CSV records the complete grade rationale for both arms and distinguishes a plausible guess from a grounded answer.]

*Wikipedia pages.* Both answers on the same page had to be fully usable. The mounted answer also required at least two source-attested values in a real untruncated read. The paired count was #pair(d.whole.natural.scores) of 24, with two-sided exact source _p_ = #r(d.whole.natural.p,4). The loop read #d.fresh.candidate.natural_mounted.read of 24 mounted tables and Base read #d.fresh.control.natural_mounted.read. Of 91 possible source-attested values, mounted-model outputs mentioned #d.fresh.candidate.natural_mounted.attested_mentions / #d.fresh.control.natural_mounted.attested_mentions and inline outputs mentioned #d.fresh.candidate.natural_inline.attested_mentions / #d.fresh.control.natural_inline.attested_mentions. Mention counts give a diagnostic of literal coverage. A reference may omit a necessary qualification or admit several valid answers.

*Changed numeric tables.* The 24 questions had two different independently human-annotated versions. Both models had to discover and use the actual SQLite database. The registered automatic whole-question counts were #pair(d.whole_tables). On the separate masked textual check the whole counts were #pair(d.whole.changed.scores), with paired _p_ = #r(d.whole.changed.p,4). A correct scalar on a training program entered from a supplied schema is an easier task. These held inputs start at the original user request and require an independently chosen sequence of calls and a usable final response.

*Open instructions.* The 24 new instructions span brainstorming, classification, closed questions, creative writing, general questions, extraction, open questions and summarization. Review used the user's requested form and the supplied source, with explicit allowance for stale or incorrect reference answers. The complete output file discloses all items including refusals, truncations, ties and unsupported claims.

*Related work.* Marchenko and colleagues continued a fully shared Qwen3-1.7B on 60 billion curated tokens and 84 billion web tokens. Against identically continued dense Qwen3-1.7B, Tables 3 and 23 show gains on nine of eleven curated tasks and eight of eight web tasks. BoolQ and CRUXEval-O on the curated mix illustrate why individual outcomes matter. Their full-stack recurrence, embedding injection and trainable backbone differ from the four-layer hybrid revisit studied here. LoopUS decomposes a pretrained model into a recurrent reasoning block with input-dependent gating and adaptive exits. Our fixed-depth comparison measures the cost of every visit. #link("https://arxiv.org/html/2610.00673v2")[Marchenko et al., Sections 4–5 and Appendix H] · #link("https://arxiv.org/abs/2605.11011")[Park et al., LoopUS]

#pagebreak()
#text(size:20pt,weight:"bold")[Native performance, decision and reproducibility]
#image("conditional-primary-cost.svg",width:99%)
#text(size:9pt)[Figure 2. Empirical distributions of request first-token wait and complete wall time for each physical decoder on the same 340 prompts, including incomplete answers and tool round trips.]

#let costs=(("Wall time (s)","wall_s",1),("Summed first-token waits (s)","first_s",1),("Opening throughput (tokens/s)","opening_tps",1),("Post-first decoding (tokens/s)","decode_tps",1),("Peak native allocation (GiB)","peak",1/1073741824),("Generated tokens","output",1))
#tbl((2.2fr,1fr,1fr),("Measured quantity","Loop","Base"),costs.map(v=>(v.at(0),r(a.at(v.at(1))*v.at(2),2),r(b.at(v.at(1))*v.at(2),2))))
Throughputs are work divided by measured time across all turns. Opening throughput counts prompt tokens actually processed after cache reuse. First-token wait in the table sums the first wait for each request. The timing CSV preserves each individual request and total decode work. One MLX worker ran candidate followed by Base on the same machine. Source preparation, audit and reporting also used the host CPU during the comparison. The stream order and background work limit precision for small timing differences. Of the 268 retained prompts, #d.prior_base_identical reproduced every Base output token from the preceding physical experiment.

*Length and decoding rate.* The difference in post-first decoding time (Base minus loop) was #r(d.decode_savings.total,2) seconds. Splitting the interaction equally attributes #r(d.decode_savings.length,2) seconds to the difference in emitted token counts and #r(d.decode_savings.rate,2) seconds to the difference in measured seconds per token. Negative values denote a cost to the loop. This is an exact arithmetic decomposition. Shorter generations can also end before providing a usable reply. Completion and masked usability retain those failures.

The first request includes model construction. Across the other 339 requests wall times were #r(d.warm_wall.at(0),2) / #r(d.warm_wall.at(1),2) seconds, and first-token waits summed to #r(d.warm_first.at(0),2) / #r(d.warm_first.at(1),2) seconds. The registered full-stream totals include the cold first request.

#pagebreak()
#text(size:20pt,weight:"bold")[Where tool use succeeds or fails]
#let tools=d.tool_diagnostics.summary
#let tr=(("Requested a tool","any_tool"),("Received a complete database read","valid_read"),("A query returned the requested scalar","correct_scalar"),("The same query works on both tables","correct_statement_both"),("Final grounded answer passed","correct_final"))
#tbl((2.8fr,.6fr,.6fr),("On 48 changed-table requests","Loop","Base"),tr.map(v=>(v.at(0),tools.candidate.at(v.at(1)),tools.control.at(v.at(1)))))
#text(size:9pt)[Table 5. Every original query was executed unchanged on both independently annotated table versions. Counts refer to requests with at least one qualifying action. A model can derive a correct final answer from a multirow read, so the last row has a different criterion.]

Across 24 separate questions, the loop produced a program that returned the correct singleton on both tables for #tools.candidate.whole_sources_with_general_query questions, versus #tools.control.whole_sources_with_general_query for Base. This allows a program emitted in either presentation to be executed on both underlying sources. Among questions on which both tables were successfully read, the model repeated the same nonempty final answer #tools.candidate.identical_after_two_reads / #tools.control.identical_after_two_reads times. The 48 outputs contained #tools.candidate.distinct_nonempty_replies / #tools.control.distinct_nonempty_replies distinct normalized nonempty answers. These diagnostics expose models that read successfully and then ignore the evidence.

*Sensitivity to available output.* Deterministic greedy generation allows exact reconstruction of stricter per-turn caps for the same scored responses. If any turn exceeded the smaller cap, that request would have stopped before its original completion. Otherwise the prompt, emitted tokens, tool use and return are identical. We count fully usable replies only when the complete originally graded response survives. The smaller caps are descriptive and carry no new timing measurements.

#let caps=d.tool_diagnostics.caps
#let caprow(n)={let rr=caps.filter(v=>v.per_turn_cap==n);(str(n),pair(rr.map(v=>v.completion)),pair(rr.map(v=>v.objective)),pair(rr.map(v=>v.masked_grounded)))}
#tbl((.85fr,1fr,1.2fr,1.15fr),("Tokens per turn","Completed / 340","Objective / 256","Usable / 156"),(32,64,128,256,384).map(caprow))
#text(size:9pt)[Table 6. Entries are loop / Base. The registered cap was 384 tokens per turn, and up to three turns were allowed. These counts cannot identify what either model would have produced above 384 tokens.]

*Audit.* The companion CSVs preserve each emitted SQL statement, its real observation and independently replayed answer on the other world, both final answers for every source change, and the lower-cap counts. A correct result generated by chance on a single table is insufficient to earn whole-query credit.

#pagebreak()
#text(size:20pt,weight:"bold")[Decision, independent replay and limitations]

*Decision.* The automatic gate #if d.automatic [passed] else [failed] and the masked gate #if d.masked [passed] else [failed]. Of #d.gates.len() named product, retention and masked tests, #d.gates.values().filter(x=>x).len() passed. The failed tests were #if d.gates.values().all(x=>x) [none] else [#d.gates.pairs().filter(p=>not p.at(1)).map(p=>p.at(0).replace("_"," ")).join(", ")]. The registration required the full conjunction and prohibited continuing these weights after a failed prerequisite. Earlier 128K runs apply to the weights that were actually tested. No long-context qualification is inferred from this short-context comparison.

*What training established.* The model freely generated a correct SQL query for both known table versions in 19 of 24 preselected training questions after receiving a real schema. On 56 teacher-forced training probes the real recurrent states improved mean loss by 0.104 nats against a frozen-weight ordinary-state intervention. Inputs and supervision were identical on the two sides of that intervention. The held end-to-end evaluation separately requires choosing to call a tool, selecting an operation from the user’s original request, observing its real result and correctly answering the question. A fitted probability or a correct memorized program cannot substitute for completing this sequence on a new source.

*Information exposure.* We searched #d.overlap.fields original instruction, question, context and answer fields from the 48 fresh sources across #d.overlap.occurrences words in the complete 7,591-turn fit. None of those fields occurred in full within a training field. The longest shared consecutive phrase had #d.overlap.maximum_run words. The audit cannot inspect the public model’s original pretraining, nor does it prove paraphrases are absent. The 268 retained cases have been used to make research decisions. Several fresh tasks still share a format or a dataset with training cases.

*Statistical scope.* Correctness and elapsed time share the same denominator wherever they are combined. Error bars resample complete independent units as registered within each family. They condition on one chosen model and one run on one computer. The masked reading has one project reviewer. Automatic source-token counts and exact matches can reward an incomplete answer or penalize a genuinely correct paraphrase. Each released row preserves the evidence needed for an independent reassessment.

*Reproduction.* Run #raw(".venv/bin/python -m scripts.research_archive conditional-program-primary-01 --verify") to check sources, tokenizer openings, tool observations, all scores and resource receipts. Run #raw(".venv/bin/python -m scripts.paper_conditional_primary") to reproduce these tables and figures from the committed blind ratings and primary archive. The full volume includes training methodology, primary-source citations, causal checks, executed SQL and the previous studies. Code, registrations, sources, weights and machine-readable results are available in the accompanying repository.

*Primary sources.* Qwen describes its Base release as a pretrained checkpoint intended for further training and research. The candidate imports the separately released instruction weights. Both model cards and the pinned local digests identify the starting points. #link("https://huggingface.co/Qwen/Qwen3.5-0.8B-Base")[Qwen3.5-0.8B-Base] · #link("https://huggingface.co/Qwen/Qwen3.5-0.8B")[Qwen3.5-0.8B]

The two reasoning sources use different kinds of context. QuAC contains contextual question-and-answer dialogues. ROPES asks what happens in a new situation given a relationship in a background passage. CoSQL supplies conversational database tasks. #link("https://aclanthology.org/D18-1241/")[Choi et al., QuAC] · #link("https://aclanthology.org/D19-5808/")[Lin et al., ROPES] · #link("https://aclanthology.org/D19-1204/")[Yu et al., CoSQL]

FeTaQA pairs Wikipedia tables with free-form questions, answers and supporting cells, allowing the same information need to be evaluated as a reading task and as an actual database task. Dolly supplies open-ended instructions authored by Databricks employees. The source manifests identify the exact enrolled records, filtering and train/evaluation separation. #link("https://aclanthology.org/2022.tacl-1.3/")[Nan et al., FeTaQA] · #link("https://huggingface.co/datasets/databricks/databricks-dolly-15k")[Databricks, Dolly 15k]
