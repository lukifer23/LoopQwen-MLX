#set document(title:"Learning a second visit to a fixed hybrid Base block",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.80in,y:.74in),footer:context [#text(size:8pt,fill:rgb("#52675e"))[LoopQwen-MLX · Base-rooted shared block] #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:10.8pt)
#set par(leading:.44em,spacing:.46em)
#show heading.where(level:1): it=>block(above:7pt,below:4pt)[#text(size:17pt,fill:rgb("#175f51"))[#it]]
#let d=json("reentry-fit-paper.json")
#let tablex(head,rows,widths)=table(columns:widths,inset:(x:4pt,y:3pt),stroke:none,fill:(x,y)=>if y==0 {rgb("#deece8")} else if calc.odd(y){rgb("#f1f5f3")},..head.map(x=>strong(x)),..rows.flatten().map(x=>[#x]))
#let note(body)=text(size:9pt,fill:rgb("#44574f"),body)
#let conclusion=if d.passed {"passed"} else {"missed"}
#text(size:25pt,weight:"bold")[Learning a second visit to a fixed hybrid Base block]
#linebreak()
#text(size:14pt)[Native MLX optimization and independent replay of a four-layer recurrent decoder]
#linebreak()
LoopQwen-MLX project · November 2026

= Abstract
A two-visit system that consolidated separately learned updates substantially improved grounded tool execution and lost accuracy on questions without tools. We tested a different mechanism on the published Qwen3.5-0.8B-Base. It reapplies four actual terminal decoder layers to their first-visit states and learns only four residual MLPs and one channel gate. All original layer tensors in the repeated block remain byte-identical. On 32 predetermined training turns the mean gold-token negative log-likelihood difference between the ordinary and learned second visits was #d.mean_gain in favour of the second. It reduced loss on #d.probe_wins of 32 turns and #conclusion the prespecified training check. Reloading the sealed parameters and directly reconstructing the full-vocabulary likelihoods gave maximum absolute error #d.max_replay_error. All observations on this page come from fitting and native replay. The held comparison is a separate registered stage.

#image("reentry-fit.svg",width:100%)
#note[Figure 1. Left, source-balanced training on 1,024 updates, grouped in consecutive blocks of 64. Right, eight preselected training turns per source. Each small segment joins ordinary first-visit loss at its left endpoint to second-visit loss at its right. Bold segments are unweighted means over the eight turns.]

#tablex(("Training source","Distinct turns drawn","Selected positions","First NLL","Second NLL","Improved / 8"),d.groups,(1.5fr,.85fr,.95fr,.65fr,.8fr,.75fr))
#note[Table 1. Training samples and probes use the complete original preceding text. The 32 probes each contribute equal weight to the qualification statistic. Repeated positions in optimization count each time a gradient was taken.]

#pagebreak()
= Data changes that make a memorized answer wrong
Pasupat and Liang collected questions about real Wikipedia tables and obtained independent denotations for altered versions. We selected 256 training questions and 24 development questions on different source tables, one question per table. Every selected original and alternative has at least three agreeing annotators and the answer changes. Each pair keeps the user request, schema, file name and opening token IDs fixed. The actual SQLite rows change. No development request participates in supervised fitting or policy updates. All 560 databases reproduced their published column names, row order and values. The filtered set contains 8,095 body rows and 1,696 participant annotations. It favours small rectangular tables and unambiguous single answers. #link("https://aclanthology.org/P15-1142/")[WikiTableQuestions (2015)] · #link("https://aclanthology.org/P16-1003/")[Changed-table annotations (2016)]

The 48 development versions have no byte-identical training table, and none of the 24 questions exactly repeats a normalized training question. Fourteen versions share an ordered column schema with training. The highest multiset overlap between training and development cell values was 0.661. These similarities remain in the enrolled material. On eight manually written training-only queries, seven returned the correct value on both versions, and a constant query returned both answers on zero. The miss involved a first-name lookup with two initial missing-name markers in an altered table.

= Native decoder and optimizer
The original prefix and tied embedding use native four-bit group-64 weights. The first visit then traverses the final three Gated DeltaNet layers and one full-attention layer. The second visit reuses these same four original layer tensors with four separate histories. Each repeated layer is followed by a rank-128 residual MLP that operates on the RMS-normalized current state and rescales its update to the original RMS. Write $h$ for the original first-visit state and $z$ for the completed second visit. The output before the inherited final normalization is $h + tanh(g) dot (z - h)$, with elementwise multiplication and one learned gate value per hidden channel. The gate and the MLP output projections start at zero. At that point this exactly reproduces the ordinary quantized backbone while both visits still execute.

The 1,049,600 trainable parameters are the four pairs of 1,024-by-128 projections and the 1,024-value gate. AdamW uses learning rate 0.00045 with a 12-fold multiplier on gate updates, weight decay 0.01 and global gradient-norm clipping at 1. Each update averages four sequential native gradients. One full user turn comes from each of the changed-table, database-dialogue, language and earlier-tool decks. Within each deck a seeded shuffle draws without replacement before reshuffling. Each eligible complete turn has at most 1,536 tokens. At most 24 response positions per draw receive full-vocabulary loss, including the first, first visible and terminal positions. No vocabulary restriction applies. The run used #d.targets selected target positions and #d.tokens causal input positions across repeated training accesses.

Consistent training drops the unused prefix gradient. It computes the ordinary first visit once on each complete sequence and stores its exact BF16 states. Gradients still traverse the complete second block. The #d.views cached arrays contain #d.cache_tokens hidden positions, occupy #d.cache_bytes bytes and took #d.cache_seconds seconds between the first and last file completions. This time includes disk writes and scheduling. The full recorded native-worker attempts consumed #d.window_native_s seconds and reached #d.mlx_peak_gib GiB peak MLX allocation on the Apple M3 Pro running with 18 GiB unified memory.

#pagebreak()
= Does the learned computation actually run?
The portable pack contains 543,234,410 bytes. Comparing all 320 original tensors with 632 stored tensor entries identified 156 quantized matrices and 164 byte-identical tensors. All 53 tensors in the repeated four-layer block, containing 83,018,720 parameters, match the original Base exactly. The root-mean-square difference across the 668,958,720 quantized values, divided by their original RMS, was 9.34%. Numerical change in a weight tensor does not itself predict a change in model accuracy.

On the registered training opening, the initial recurrent and ordinary quantized outputs were identical over all #d.cold_exact vocabulary logits. Loading the portable pack in another decoder repeated that full-vocabulary equality. Both visits stored distinct convolution and delta states in each of the three repeated Gated DeltaNet layers. Both instances of full attention reached the entire opening length. The first-visit state fingerprint is checked before and after optimization. Splitting a prefill can change low-order values in native floating-point arithmetic. We record the maximum logit discrepancy and the top token in both executions.

#tablex(("Model","Full top ID","Incremental top ID","Max abs. logit difference","Attention offset","Distinct site caches"),d.parity,(1fr,.72fr,1.0fr,1.18fr,.83fr,.9fr))

Every trainable tensor and each AdamW first and second moment is accounted for by name, dtype and shape. Of #d.steps actual optimizer steps, #d.positive_gradient_steps had positive gradient norm and #d.clipped_steps exceeded the clipping threshold. The median pre-clipping norm was #d.grad_median. The learned channel gate had RMS #d.gate_rms after applying tanh. The four residual output norms were #d.up_norms.join(", "). An initial cache-audit error treated MLX’s enclosing state list as a tensor. The first attempt stopped before any parameter update. The preserved restart record and additional regression assertion identify the correction, and both attempts count in elapsed time.

#tablex(("Registered first-stage decision","Outcome"),d.qualified_gates,(2fr,1fr))
#note[The independent native replay reconstructs each selected target from the source conversation, reloads the sealed tensors and directly evaluates log-sum-exp minus the gold logit. It covers #d.probe_targets positions over 32 original training turns.]

= What this experiment establishes
The ordinary reference in the training probe shares the quantized prefix, frozen layers and token history with the second visit. A held comparison against untouched BF16 Base asks about the deployed system. A separately trained ordinary-depth control with the same exposure is required for an inference about the benefit of recurrence under matched training. The next registered stage samples the model’s own SQL or natural answers on original and changed training tables and derives rewards from actual executions and human annotations. Only a qualifying policy update opens the 268-request held comparison. An observed response beyond 131,072 input tokens applies to the particular measured weights and cache path.

The training draws, optimizer state, first-visit feature hashes, per-tensor comparison, paired probe losses and replay script are included in the checkout. From the project root use #raw(".venv/bin/python -m scripts.research_archive table-reentry-fit-01 --verify"), then #raw(".venv/bin/python -m scripts.verify_table_fit_native") and #raw(".venv/bin/python -m scripts.paper_reentry_fit").
