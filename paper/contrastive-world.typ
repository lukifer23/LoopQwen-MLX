#set document(title:"Can a recurrent language model change its answer when the evidence changes?",author:"LoopQwen-MLX")
#set page(paper:"us-letter",margin:(x:.72in,y:.65in),numbering:"1")
#set text(font:"Libertinus Serif",size:10.4pt)
#set par(leading:.43em,spacing:.55em)
#let d=json("contrastive-world.json")
#let tbl(cols,head,rows)=table(columns:cols,inset:4pt,stroke:.45pt+rgb("#b8c5c6"),..head.map(x=>[*#x*]),..rows.flatten())
#let rnd(x,digits:2)=str(calc.round(x,digits:digits))
#let paired(x)=str(x.at(0))+" / "+str(x.at(1))
#text(size:23pt,weight:"bold")[Can a recurrent language model change its answer when the evidence changes?]
#v(.45em)
*Source-conditioned training of a four-layer Qwen loop on Apple Silicon* \
LoopQwen-MLX research report · November 2026

*Objective.* Produce a single recurrent assistant that can use source evidence, perform a tool action and deliver a correct response when a table, question or document changes. The preceding instruction-initialized model solved 66/256 scored requests against 42/256 for ordinary Base, took longer to produce a first token and scored 17/48 against 28/48 in masked source-based reasoning. Two trained loops shared 23 gains over Base and 17 losses. These outcomes motivate training on cases for which a particular answer becomes wrong after the source changes.

*Observed status.* The archive contains #d.stage_count of the four registered stages and records #rnd(d.total_native_s) seconds of native worker time. The last closed stage was #d.latest.name and its registered decision was #if d.latest.passed [*pass*.] else [*miss*.] #if d.stage_count < 4 [A held product comparison has no archived result for these weights.] else if "service" in d [The product comparison below includes both masked source readings.] else [The product comparison stopped before both full streams were available.]

#tbl((2.1fr,.72fr,.85fr,.9fr,.85fr),("Closed stage","Gate","Worker s","Cap s","Peak GiB"),d.stages.map(r=>(r.name,if r.passed {"pass"} else {"miss"},rnd(r.elapsed),str(r.max_seconds),rnd(r.peak_gib))))
*Experiment.* The published instruction-sibling text weights were packed to four bits and frozen. The model executes 24 ordinary layers, then reuses its final four with separate causal histories. A learned residual correction uses the four physical increments and a gate. Replacing all recurrent states by the ordinary exit makes the correction identically zero. Thus the path can use information produced by a second visit. A separate equally trained ordinary-depth control is necessary to attribute any product gain specifically to looping.

*Training sources.* The added corpus contains #d.enrollment.verified_positive_turns complete turns and #d.enrollment.complete_incompatible_pairs bidirectional comparisons. It adds to 7,591 earlier turns. The new examples include 942 human SQL questions with a program verified across three to nine original or counterfactual table versions with human-annotated answers, 1,201 opposing QuaRTz question pairs, and 900 pairs of grounded service replies. A second answer becomes a negative only under the other source. Training retains original tool call arguments and observations. The study preregistered #d.fit_target positive label exposures and #d.preference_target pairwise presentations across its full fit.

*Data reuse.* We searched all 19,233 fitting turns, including rejected targets, serialized tools and joined natural dialogues. None contained a whole held table question, qualitative question, customer conversation or human service reply. Exact wording of one background occurred in training. Six final service utterances also recurred. The frozen lexical sensitivity groups retain all source partners. They flag six of 48 qualitative and ten of 48 service presentations. The pretrained model’s original exposure is unknown.

*Related work.* Geiping et al. trained a depth-recurrent model whose computation can vary at inference. Research on retrofitting recurrence also compares fine-tuned recurrent and ordinary architectures at matched training compute. Our product pair asks what a user receives from the complete trained system. Its differing weight precision and instruction initialization motivate a separate same-initialization depth control. #link("https://openreview.net/pdf/14c70cb9442ba3067a5abf112e08a003028eec21.pdf")[Geiping et al.] · #link("https://openreview.net/forum?id=eC85h3y4pG")[Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence].

*Hardware and budget.* All model forwards, differentiation and optimization run on an Apple M3 Pro with 18 GiB of unified memory using native MLX. Each stage has a #(d.limits.wall_seconds)-second ceiling, an #d.limits.peak_gib GiB MLX allocation limit and a #d.limits.available_gib GiB available-memory floor. A failed prerequisite closes its dependent stage. The ledger records actual expenditure on closed stages.

#pagebreak()
#text(size:19pt,weight:"bold")[Physical training evidence and verification]
#v(.25em)
#tbl((1.6fr,.75fr,1fr,1fr,1fr),("Source group","Turns","Opening tokens","Label positions","Changed negative positions"),d.exposures.map(r=>(r.group.replace("_"," "),str(r.turns),str(r.opening_tokens),str(r.target_positions),str(r.negative_positions))))
#if d.exposures.len()>0 [#image("contrastive-world-exposure.svg",width:96%)]
*Figure.* Counts come from indexed physical outputs in closed stages. A negative prediction that differs at just one position can reuse the positive states at the identical causal prefix. Longer changes require separately evaluated prefixes. The source index preserves which tensors each example used.

#let cold=d.stages.filter(r=>"checked_files" in r)
#if cold.len()>0 [
  *Cold verification.* #for r in cold [For #r.name, an independent process checked every one of #r.checked_files serialized feature files (#r.checked_bytes bytes), regenerated #r.rebuilt complete files byte for byte, and compared individual positive and negative physical states on #r.native_cases source turns. ]
]
*Optimization.* We minimize full-vocabulary next-token cross-entropy, teacher-to-student KL with coefficient 0.18, and a softplus loss on the length-normalized chosen-versus-rejected log odds over differing positions. The teacher is the ordinary exit of the same instruction backbone. The prospective fit checks all positions, finite nonzero updates, protected source groups and freely generated actions. Written and spoken answers are scored after generation. A fitting diagnostic uses its own enrolled training examples.

*Held protocol.* The direct comparison enrolls 48 changed-table presentations, 48 qualitative questions, 48 customer requests and four complete CLDR inputs of 152,185 to 164,271 tokens. Graphical or abbreviated representations of the documents are never sent to the models. The evidence occurs after token 131,072. For service requests the published dialogue simulator supplies the original recorded rows. Each model has to issue its own read-only SQLite call to access them. The benchmark charges the entire prompt and uses the same greedy, thinking-enabled generation, tool schema and 384-token per-turn cap for both arms.

*Decision.* Require at least six completely correct source pairs and an advantage of five pairs for tables, qualitative reasoning and masked services, with paired source-level p ≤ 0.05 and zero individual losses on those endpoints. Require both complete long-document pairs, improved individual long answers, preserved completion and improvement in all five recorded cost or throughput metrics. Replication and equally trained depth attribution follow joint automatic and masked qualification.

*Source methodology.* Pasupat and Liang used human judgments on altered tables to identify queries whose correct original answer was coincidental. QuaRTz supplies paired situated science questions, and Schema-Guided Dialogue supplies the service conversations and call records. We test whether a freely decoding assistant can use these differing sources and deliver a supported reply. #link("https://aclanthology.org/P16-1003/")[Pasupat and Liang] · #link("https://aclanthology.org/D19-1608/")[Tafjord et al.] · #link("https://research.google/pubs/towards-scalable-multi-domain-conversational-agents-the-schema-guided-dialogue-dataset/")[Rastogi et al.].

*Reproduction.* The project stores registrations under #raw("research/experiments"), compressed raw records and native environment inventories under #raw("research/results"), and independent verification under #raw("docs/measurements"). Run #raw("python -m scripts.paper_contrastive_world") from the repository root. The adjacent JSON records the exact archive and closed-session SHA-256 for each line of the resource table.

#if d.stage_count >= 3 [
  #pagebreak()
  #text(size:19pt,weight:"bold")[Fitting outcomes and free generation]
  #let fit=d.stages.at(2).decision
  #if "free_whole" in fit [
    The optimizer completed #fit.updates steps, #fit.supervised_positions supervised positions and #fit.preferences preference comparisons. On the training diagnostics, the loop generated SQL that returned a correct scalar in all versions of #fit.free_whole.operator of 24 source tables, using #fit.free_distinct_sql distinct successful statements. It freely answered both contexts correctly on #fit.free_whole.qualitative/12 qualitative pairs and #fit.free_whole.service_reply/12 service pairs.
    #if "specificity" in d [
      #let measured=d.specificity.at("freely generated loop")
      On the 24 individual service probes, #measured.service_reply.current_only replies mentioned two compatible fields specific to the actual source and #measured.service_reply.both matched either source. Both members had source-specific details on #measured.service_reply.both_members_current_only/12 pairs. Human replies met the same source-specific condition on 11/12 service pairs. On qualitative questions, #measured.qualitative.current_only of 24 generated answers selected the actual source’s choice.
    ]
    #image("contrastive-world-optimization.svg",width:100%)
    *Figure.* Each point summarizes up to 64 consecutive optimizer updates. Token losses are weighted by positive target counts. Preference loss is weighted by the number of presentations. These are training diagnostics.
    #tbl((2fr,1fr),("Training probe group","Mean NLL reduction, nats"),fit.mean_nll_gain.pairs().map(v=>(v.at(0).replace("_"," "),rnd(v.at(1),digits:4))))
    #tbl((2fr,1fr),("Preregistered fit gate","Observed"),fit.gates.pairs().map(v=>(v.at(0).replace("_"," "),if v.at(1) {"pass"} else {"miss"})))
  ] else [
    The archived fit was interrupted. The registered reason was #fit.reason. The decision includes its retained evidence. The training measurements on the previous pages remain available.
  ]
]
#if "service" in d [
  #pagebreak()
  #text(size:19pt,weight:"bold")[Held product comparison]
  #let primary=d.stages.at(3).decision
  #tbl((1.2fr,1fr,1fr,.85fr,.7fr),("Source family","Individual loop / Base","Whole pairs loop / Base","Paired p","Losses"),primary.counts.pairs().map(x=>(x.at(0),paired(x.at(1).individual),paired(x.at(1).whole),rnd(x.at(1).p,digits:4),str(x.at(1).losses))))
  The masked service review credited #paired(d.service.individual) individual grounded answers and #paired(d.service.whole) complete source pairs (loop / Base). The separate qualitative reading credited #paired(d.qualitative.source_pairs.individual) individual answers and #paired(d.qualitative.source_pairs.whole) complete sources. Both reviews used a source-specific rationale sealed before model attribution. The reviewer was the project assistant.
  #let rec=d.tool_recovery.arms
  On the 96 tool-enabled requests, provider errors occurred on #paired((rec.candidate.all.requests_with_error,rec.control.all.requests_with_error)) requests. Of the #paired((rec.candidate.all.could_generate_after_error,rec.control.all.could_generate_after_error)) with a subsequent turn available, #paired((rec.candidate.all.successful_later_callback,rec.control.all.successful_later_callback)) obtained a successful later callback. Grounded finals after an error totaled #paired((rec.candidate.all.grounded_after_error,rec.control.all.grounded_after_error)).
  #let ch=d.generation_channels.streams
  Actual generated tokens before the first thinking-channel close totaled #paired((ch.candidate.all.before_close_tokens,ch.control.all.before_close_tokens)) across the held requests. After that close the totals were #paired((ch.candidate.all.after_close_tokens,ch.control.all.after_close_tokens)). On the 48 qualitative questions, #paired((ch.candidate.qualitative.requests_with_words,ch.control.qualitative.requests_with_words)) had a nonempty deliberation. Scientific correctness and these length measurements are reported separately.
  #let c=primary.costs.candidate
  #let b=primary.costs.control
  #tbl((2.6fr,1fr,1fr),("Measured product metric","Loop","Base"),(("All requests, seconds",rnd(c.wall_s),rnd(b.wall_s)),("First token, summed seconds",rnd(c.first_s),rnd(b.first_s)),("Peak MLX allocation, GiB",rnd(c.peak/1073741824),rnd(b.peak/1073741824)),("Opening tokens per second",rnd(c.opening_tps),rnd(b.opening_tps)),("Post-first tokens per second",rnd(c.decode_tps),rnd(b.decode_tps))))
  #tbl((1.0fr,2.2fr,.7fr,1fr),("Family","Wording stratum","Requests","Loop / Base correct"),d.sensitivity.map(r=>(r.family,r.stratum,str(r.items),paired((r.candidate_correct,r.base_correct)))))
]
