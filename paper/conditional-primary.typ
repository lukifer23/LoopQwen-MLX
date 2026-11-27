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
Throughputs are work divided by measured time across all turns. Opening throughput counts prompt tokens actually processed after cache reuse. First-token wait in the table sums the first wait for each request. The timing CSV preserves each individual request and total decode work. The single-machine execution order was candidate followed by Base. Of the 268 retained prompts, #d.prior_base_identical reproduced every Base output token from the preceding physical experiment.

*Decision.* The automatic gate #if d.automatic [passed] else [failed] and the masked gate #if d.masked [passed] else [failed]. Of #d.gates.len() named product, retention and masked tests, #d.gates.values().filter(x=>x).len() passed. The failed tests were #if d.gates.values().all(x=>x) [none] else [#d.gates.pairs().filter(p=>not p.at(1)).map(p=>p.at(0).replace("_"," ")).join(", ")]. The registration required the full conjunction and prohibited continuing these weights after a failed prerequisite. Earlier 128K runs apply to the weights that were actually tested. No long-context qualification is inferred from this short-context comparison.

*Reproduction.* Run #raw(".venv/bin/python -m scripts.research_archive conditional-program-primary-01 --verify") to check sources, tokenizer openings, tool observations, all scores and resource receipts. Run #raw(".venv/bin/python -m scripts.paper_conditional_primary") to reproduce these tables and figures from the committed blind ratings and primary archive. The full volume includes training methodology, primary-source citations, causal checks, executed SQL and the previous studies. Code, registrations, sources, weights and machine-readable results are available in the accompanying repository.
