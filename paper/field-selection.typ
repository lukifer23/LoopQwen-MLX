#set document(title:"Can a recurrent language model choose and interpret the right field?",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.83in,y:.73in),footer:context [#set text(size:8pt,fill:rgb("#54645f"))
LoopQwen-MLX · Human evidence fields and executable tools #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:11pt)
#set par(leading:.53em,spacing:.55em)
#show heading.where(level:1): h => block(above:8pt,below:3pt)[#set text(size:16pt,fill:rgb("#185c52")) #h]
#let m=json("field-selection-paper.json")
#let grid(headers,data,widths) = table(columns:widths,align:(x,y)=>if x==0 {left} else {right},inset:(x:5pt,y:3pt),stroke:none,fill:(x,y)=>if y==0 {rgb("#dcece7")} else if calc.odd(y) {rgb("#f3f6f5")},..headers.map(v=>text(weight:"bold",v)),..data.flatten().map(v=>[#v]))
#let caption(body)=text(size:9pt,fill:rgb("#43524d"),body)

#text(size:25pt,weight:"bold")[Can a recurrent language model choose and interpret the right field?]
#linebreak()
#text(size:14pt)[Cross-domain human judgments, package compatibility and two physical visits to one Qwen layer]
#linebreak()
November 2026 · LoopQwen-MLX project

= Abstract
The proof-supervised loop in the preceding experiment grounded 22 of 72 human judgments versus two for Base. It also reversed the search and return columns on all 16 practical requests. We tested whether training on opposite-direction calls and three competing judgments about the same table improves evidence selection and interpretation. We trained a fresh recurrent candidate and compared it directly with the pinned ordinary-depth Base model. #m.abstract

#image("field-selection-accuracy.svg",width:100%)
#caption[Figure 1. Each count is a completed correct answer. File and package success also requires execution of the expected lookup and receipt of the selected value. The same eight retention prompts appeared in earlier studies.]

= Motivation and scope
A user asking whether an environment can run a package needs two things from an assistant. It must read the constraint on the intended release and apply that constraint to the requested interpreter. Producing valid JSON is only one intermediate step. We use a separate human reasoning task where an annotated field supplies relevant evidence, then score the decisions and actual calls from the same model.

#pagebreak()
= Architecture and learning
The control is the untouched BF16 Qwen3.5-0.8B-Base text checkpoint. The candidate begins with the instruction sibling from release 05, an eight-bit prefix, and the shared rank-32 adapters on seven attention and feed-forward projections at layer 23. The last layer runs twice for every prompt and generated token. Its second visit uses residual mixing coefficient 0.35 and an independent physical attention history. The ordinary vocabulary remains available at every step. The direct pair measures the combined effects of the instruction starting weights, quantization, tuning and depth. Only a matched ordinary-depth fit can assign a measured difference to the loop itself.

At each update we select an entire table and three original human claims with distinct answers. All three share a reading mode. For the file mode training executes the correct lookup first. For inline training the input contains the full table. The short training explanation identifies the human-selected field and quotes its value. It supplies evidence for the verdict, with no authored explanation of the logical step. We supervise the human verdict in the visible answer.

For the three claims, assemble a $3 times 3$ matrix of logits at the first semantic verdict position. Its rows index the claims, its columns the distinct leading tokens for true, false and uncertain. Each diagonal cell names the correct claim and verdict. Mean row-wise and column-wise cross-entropy gives a loss that a constant class preference cannot minimize. This term receives 30% of the triplet loss. Evidence-text token loss receives 25%, first-verdict token loss 25% and all-token loss 20%. The unrestricted logits also train the ability to emit a complete assistant turn.

The triplet receives 60% of the update. Another 20% trains an evidence opening, a WikiSQL lookup direction, or recovery from an actually failed swapped-column request. For the latter the native tool returns its real error before the model sees the corrected call. Package openings and verdicts receive 12%, with 8% for previously used table responses. The engine computes and accumulates gradients on MLX on one 18 GiB M3 Pro. Every hundred updates it checkpoints the weights, optimizer moments and cumulative exposure. #m.fit

= Previous work and design choices
Geiping and colleagues trained a dedicated depth-recurrent transformer at much larger scale. Our smaller retrofit poses a different empirical question about how much capability can be recovered from an existing model. InfoTabS and its subsequent human evidence annotations provide reference claims and a way to identify which part of a real table is relevant. ToolSandbox motivates inspecting the actual state after calls, and ToolACE studies verification of tool-training trajectories. Our runtime executes local lookups; evidence here is specific to the tool and task families measured. [1–5]

#pagebreak()
= Data and checks against shortcuts
The original InfoTabS train partition supplies 128 tables. Each contributes one entailed, contradicted and neutral hypothesis. The primary uses 12 cross-domain α3 test tables with no shared entity ID, normalized title or identical claim across our train and test sets. Each enrolled item requires one relevant infobox field, no outside-evidence marker and at least three identical evidence-selection ballots. At least four independent validators agreed with the original author’s verdict on every held claim. These rules select comparatively well-agreed, single-field instances. None of the 36 held claims exactly matched a training claim after masking full titles, numbers, punctuation and case. The largest five-word shingle Jaccard similarity across 13,824 pairs was 0.20. Claims, factual values and original labels are preserved. Seven of the 12 selected tables concern medical topics. This is an evaluation of judgments against supplied tables, with no clinical decision task.

An inline prompt contains every field. In the file condition the tool sees a genuine wide CSV whose rows represent the intended entity and two distractors. We select other records from the same source partition, preferring shared field names, and preserve the original field texts. All three requests for each entity share the same catalog. File scoring requires the originally selected evidence field. A plausible alternative field can provide useful information and still miss the registered score. The analysis records the first executable call, any later successful retrieval, and the final visible verdict separately.

WikiSQL supplies 96 further training tables with two columns that have unique values in both directions. Every pair comprises two lookups with opposite search and return fields, different input values and actual observations. For each selected row a deliberately reversed call with the original key really fails. Ten disjoint training distributions supply package examples. Eight new PyPI releases supply the held compatibility questions. Each release is tested with real Python patch versions on opposite sides of its inclusive lower bound. Both questions require the same observation and opposite final answers. This panel tests simple minimum versions. Compound ranges, exclusions, pre-releases and dependency resolution remain separate capabilities.

#m.packages

#m.integrity

= Controls and decision rule
Both models receive the same ordered initial token IDs and tools, with thinking enabled and greedy full-vocabulary output. The budget is 512 tokens per turn and at most three turns per user request. Trajectories can diverge after an actual tool call. The primary resamples the table as the unit of dependence and enumerates all 4,096 paired sign flips. Promotion requires ≥30/72, a net gain of ≥12 at two-sided $p≤0.05$, at least two whole tables, gains in both modes, four complete package pairs, retention, completion, and the registered efficiency measures. The protocol fixes a 4,100-second native budget and 7.25 GiB peak limit before training.

#pagebreak()
= What the model did with the tool
#image("field-selection-arguments.svg",width:100%)
#caption[Figure 2. First executable call per request. A swapped call reverses both columns, preserves the key text, and uses the correct document. “None” includes trajectories that never executed a call.]

#grid(("Input","System","Read field","Correct after read","Later repair"),m.lookups,(1fr,1.5fr,1fr,1.3fr,1fr))
#caption[Table 1. The third column counts all trajectories that at any point obtained the requested observation. The fourth requires a correct final answer on those trajectories. The last counts correct retrieval after an earlier unsuccessful execution.]

#image("field-selection-confusion.svg",width:100%)
#caption[Figure 3. The 24 instances of each label span both reading modes. No verdict includes capped replies and text from which the registered parser could not identify a final judgment. A correct class alone does not imply a successful file read.]

#m.scope

#m.formatting

#pagebreak()
= Accuracy, time and inference cost
#grid(("Measurement","Trained loop","Untouched Base"),m.summary,(2.2fr,1fr,1fr))
#caption[Table 2. Each of 96 requests starts a fresh conversation, retaining the model in memory across requests within an arm. Synchronized wall time includes construction on the first request, input processing, calls, and output generation. Peak is the maximum of the per-request MLX peaks.]

#image("field-selection-timing.svg",width:100%)
#caption[Figure 4. Empirical time distributions over all 96 requests, including failures. Left shows completion or termination. Right shows first emitted token. Log horizontal axes make short and capped generations visible together.]

The opening rate divides all initial prompt tokens by the summed first-token latency. The subsequent rate divides tokens after the first in each turn by the summed remaining turn time. The loop always ran before Base under the registered stream. Relative temperature and operating-system activity can affect small timing differences. A shorter total stream can reflect fewer generated tokens, including premature completion. Reported peak measures MLX allocations on this device. It does not measure wall-outlet energy or total system RAM.

#m.per_request

= Blinded reading
#grid(("Explanation rating","Trained loop","Untouched Base"),m.reading,(2fr,1fr,1fr))
#caption[Table 3. Twenty prespecified requests from each arm. One project-agent reader judged evidence, reasoning and the visible answer, with arm names and aggregate scores withheld. “Supported” requires the inferential link to the correct verdict. A source fact without that link can receive partial credit. An absent visible verdict receives absent. Style can reveal a model’s identity. Independent human raters would strengthen this measurement.]

#pagebreak()
= Decision, context and reproducibility
#m.decision

The primary comparison asks about a complete deployed system. A statistically significant difference on the 12 selected entities does not by itself establish a broadly superior assistant or a causal benefit of extra depth. Pretraining data for either Qwen model may include public text overlapping these sources. A fresh panel protects against selecting evaluation rows based on this project’s outputs. It does not establish pretraining exclusivity. Eight repeat requests measure retention against the same control.

The archived release 05 separately exercised two actual 131,084-token requests on the same Mac. The previous matched induction experiment measured a three-pass system at 124/128 and a parameter- and training-matched one-pass system at 58/128 across two panels. Those results concern different weights and tasks. This paper assigns the new field-selection weights only the measurements they underwent. Later matched and long-history work is conditional on the registered direct-comparison decision. The remainder of this edition reproduces the previous technical report and accumulated positive and negative experiments.

The source tree includes the registration, original held prompt token IDs, training examples, original per-field and verdict vote counts, PyPI release and wheel hashes, candidate and Base token traces, tool observations, per-turn cache histories, adapter weights and optimizer receipts. The request-level CSV includes every item and both replies. Running the archive verifier reconstructs the tool calls and scores from the saved token streams. The figures read those same records. Source downloads, native execution and paper commands are documented with the experiment. SHA-256 identities tie each release to the report.

= Sources
[1] V. Gupta, M. Mehta, P. Nokhiz and V. Srikumar. _InfoTabS: Inference on Tables as Semi-structured Data._ ACL, 2020. #link("https://aclanthology.org/2020.acl-main.210/")[Publisher paper.]

[2] V. Gupta, S. Zhang, A. Vempala, Y. He, T. Choji and V. Srikumar. _Right for the Right Reason: Evidence Extraction for Trustworthy Tabular Reasoning._ Zenodo, 2022. #link("https://zenodo.org/records/6578593")[Human evidence and annotation protocol.]

[3] J. Geiping and colleagues. _Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach._ NeurIPS, 2025. #link("https://arxiv.org/abs/2502.05171")[Original paper.]

[4] J. Lu and colleagues. _ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities._ NAACL Findings, 2025. #link("https://machinelearning.apple.com/research/toolsandbox-stateful-conversational-llm-benchmark")[Research project.]

[5] W. Liu and colleagues. _ToolACE: Winning the Points of LLM Function Calling._ ICLR, 2025. #link("https://arxiv.org/abs/2409.00920")[Original paper.]

[6] V. Zhong, C. Xiong and R. Socher. _Seq2SQL: Generating Structured Queries from Natural Language Using Reinforcement Learning._ 2017. #link("https://github.com/salesforce/WikiSQL")[WikiSQL release.] T.-p. Chung. _PEP 658: Serve Distribution Metadata in the Simple Repository API._ #link("https://peps.python.org/pep-0658/")[Metadata specification.]
