#set document(title: "Where shared decoder depth earns its cost", author: "LoopQwen-MLX project")
#set page(paper: "us-letter", margin: (x: 0.86in, top: 0.7in, bottom: 0.76in), footer: context [#set text(size: 8pt, fill: rgb("#51605e"))
LoopQwen-MLX · Research report #h(1fr) #counter(page).display()])
#set text(font: "Libertinus Serif", size: 10.7pt, lang: "en")
#set par(justify: false, leading: 0.62em, spacing: 0.65em)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => block(above: 10pt, below: 7pt)[#set text(size: 16pt, fill: rgb("#14574e"))
#it]
#show heading.where(level: 2): it => block(above: 8pt, below: 3pt)[#set text(size: 11pt)
#it]
#let repo = "https://github.com/lukifer23/LoopQwen-MLX"
#let tab(..body) = table(columns: (1.65fr,1.05fr,1.05fr), align: (left,center,center), stroke: none, inset: 5pt, fill: (x,y) => if y == 0 {rgb("#deede9")} else if calc.odd(y) {rgb("#f5f7f6")}, ..body)
#let note(body) = block(inset: 8pt, fill: rgb("#edf4f1"), width: 100%, body)
#text(size: 26pt, weight: "bold")[Where shared decoder depth earns its cost]
#linebreak()
#text(size: 14pt)[Reasoning, executable tool use, and full-history inference on an 18 GiB Apple Silicon machine]
#v(4pt)
LoopQwen-MLX project · 26 November 2026 · Technical report
= Abstract
We study whether pretrained Qwen3.5-0.8B-Base can answer more questions correctly by applying a decoder layer more than once. We compare the deployed system with untouched Base and separately compare depth under matched supervision. The clearest effect occurs when a repeated native attention layer updates a question from fixed sentence representations. Across two disjoint induction panels it answered 124 of 128 questions correctly, against 58 with a separately trained one-pass layer. The paired difference survived a second training seed, a same-weight bypass, a longer one-pass fit and a class-swap intervention. In larger matched arithmetic and entailment studies both depths tied on final-answer accuracy. A trained kinship model performed better with one pass.

We extend the comparison to calling external functions, following their observations, comparing returned quantities and asking for missing arguments. We record the generated calls, actual program responses and final answers. A tool-mode model initialized from the instruction-tuned Qwen sibling and fitted on unrelated demonstrations then executed 17/22 publisher-authored numerical requests correctly, versus 0/22 for untouched Base. Base answered eight numerically without the required call. On previously unseen competing function schemas the same tool weights generated 18/24 correct calls versus 8/24 for Base. An equally trained one-pass tool model completed 18/22 numerical executions, one more than the two-pass model. A subsequent checkpoint trained on paired changes in tool observations completed 112 of 132 mixed tool tasks versus 15 for Base, including 26 of 30 episodes on four new catalog domains. Its equally trained one-pass control solved 108/132, leaving the incremental depth effect uncertain. On separate two-user-turn requests about actual Python modules the qualified tool model scored 12/16 against zero for Base. The original language release retrieved both answers from 131,083-token register prompts while taking longer than Base. A subsequent causal page-prefill kernel retrieved both changed answers on a fresh 131,073-token input in 176.55 s total versus zero in 391.72 s for Base. It used 14.4% more peak MLX memory. With direct 8-bit KV products on a different 131,080-token document, the recurrent model scored 2/2 versus 0/2, took 171.88 versus 391.67 s, and reduced peak allocation by 26.0%. The two interventions measure a specific ability to retrieve a changed record. On a separate complete 162K-token novel, the release and Base generated identical outputs and both missed the human-authored question in both versions. The shipped combination reproduced all 200 previously qualified language generations and tied Base on 9,328 raw next-token targets by aggregate loss. On a cold-start stream of 50 field requests it scored 21 versus 12 in 83.97 versus 234.20 s. Its arithmetic accuracy fell by one, first-token time rose and peak memory increased by 5.4 MB. A separate Base-initialized raw decoder improved loss, top-one accuracy, time and peak allocation on each of three suites totaling 36,126 targets. Its per-domain safeguards retained the original ordinary route.

= Measured system comparisons
#table(columns: (1.45fr,0.95fr,0.95fr,1.18fr,0.98fr,0.46fr),align:(left,left,center,center,center,center),inset:3.5pt,stroke:none,fill:(x,y)=>if y==0 {rgb("#deede9")} else if calc.odd(y){rgb("#f5f7f6")},
[*Candidate*],[*Endpoint*],[*Correct\ of N*],[*Total seconds*],[*Peak GiB*],[*Gate*],
..csv("system-comparisons.csv",row-type:dictionary).map(r=>([#r.system],[#r.endpoint],[#r.candidate / #r.base (#r.n)],[#r.seconds_candidate / #r.seconds_base],[#r.peak_candidate / #r.peak_base],[#r.gate])).flatten(),
)
Every pair lists the candidate first and untouched Base second. The denominator in parentheses counts requests. Each row uses its named checkpoint or inference policy. The last column applies the complete predetermined decision rule, including per-family floors where specified. Grounded success requires the recorded callback and final answer to agree with the reference. Distinct resource and timing protocols are detailed by study below. The shipped assistant is evaluated separately.

= Research questions
*System benefit.* Under identical prompts, thinking policy, scorer and output allowance, does a user receive a correct completed answer sooner? A sparse answer correction, adapter training or a change in response termination can improve this endpoint. Each requires its own attribution test.

*Depth benefit.* With the same data, parameterization and optimizer exposure, does recurrence increase accuracy enough to justify its time and memory? A model run at one pass after training at three also tests dependence on its learned trajectory. The separately trained one-pass control addresses the stronger allocation question.

*Generalization.* Can the learned behavior follow changed evidence, work on new schemas and preserve independent capabilities? We test reworded requests, counterfactual facts, distinct human language datasets, raw prose likelihood and real long-input generation. The tool section distinguishes the authors' controlled environment from independently authored function signatures.

= Model and implementation
The pinned checkpoint is Qwen/Qwen3.5-0.8B-Base at revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Its text backbone has 752,393,024 parameters, a hidden width of 1,024, 24 decoder layers and a 248,320-entry vocabulary. Every fourth layer uses full attention. Other layers carry gated delta and convolutional state. All model inference, automatic differentiation and optimizer updates in this report use native MLX on an M3 Pro with 18 GiB unified memory. The new tool experiments ran under macOS 27.3 with Python 3.12.12 and Transformers 5.24.0. The project pins MLX 0.32.4 and mlx-lm 0.32.2. Timing comparisons are machine-specific. [1]

== Shared computation and state
The original recurrent decoder embeds its input, executes layers 0–9, repeats the shared 10–15 span, then executes 16–23 and the output projection. For a residual-gated repeated block $F$, later passes compute $h_(k+1) = h_k + alpha (F(h_k) - h_k)$. Repeated sites share weights. Each attention site has its own key/value cache and each linear-attention site has its own recurrent state. Sharing these histories across passes changes the autoregressive model. The implementation checks cache ownership, span, gate and window policy before reuse.

The strongest induction result uses a different placement of recurrence. Layers 0–14 encode the story. Four pooled regions per sentence supply features for a learned compression into one premise token. Starting from the compressed question, the model reapplies native layer 15 to the fixed premises and an evolving question. Low-rank updates modify Q/K/V/O, and a deterministic step encoding identifies the pass. During training, three head pairs receive losses on the publisher's supporting-fact indices. In the matched one-pass model the same head pairs receive those labels on its single call. A learned vocabulary-logit correction incorporates the final state into ordinary full-prompt generation. Support indices are unavailable during evaluation.

The two-read deduction component instead learns retrieval steps and runs the pretrained 10–15 span on sentence states. The register component scans overlapping windows of token-embedding features, scores windows against the query, and biases native attention toward the selected passage at generation. Its full-history decoder still consumes every token. The deployed language policy selects a component from the request, including an ordinary Base path. For induction, both a story gate and a 4,098-parameter question-intent classifier must accept the input.

== Training and comparison rules
The induction fit uses 1,536 training stories, 3,000 Adam updates, batches of 32 and 552,810 trainable parameters per arm. The question and nine shuffled facts are identical for paired models. We count completed answers, also score four candidate words at a shared prefix, and measure full inference time. We retain all enrolled ambiguous items. The different training seeds change initialization and minibatch order together. The system and depth comparisons answer different questions and have different controls.

The repository freezes ordered material and a threshold before each primary comparison. Timing includes prefill and greedy generation. A short response can win on endpoint latency while using more compute per token. We give seconds, output counts or matched-length comparisons where possible. Two-sided exact paired tests condition on discordant items. Swapped pairs and entailment triplets use their shared underlying unit of analysis. Reported tests are study-specific. Repeated experiments selected promising architectures, so the full collection does not have a single family-wise significance guarantee.

= Reasoning results under matched training
#figure(image("depth.svg",width: 99%), caption: [Completed answers with a separately trained ordinary-depth control. Counts at right give repeated and one-pass scores. Arithmetic and entailment use the same scoring convention for both arms within their respective depth studies.])
The three-pass induction model scored 62/64 on both the primary and confirmation panels. The matched single-pass model scored 33/64 and 25/64. The primary had 30 wins and one loss; confirmation had 37 wins and zero losses. At 13 generated tokens per item the paired models took about 16 seconds per panel. Training took 114.63 seconds of optimization at three passes and 42.28 at one. A one-pass answer-only control received 15,000 updates and 163.29 optimizer seconds and scored 28/64. With independently initialized and trained weights, the respective scores were 53/64 and 25/64. The second fit shows both replication of direction and sensitivity to seed. [2]

A same-weights intervention reduced the confirmation score from 62 to 19 when the second and third applications were bypassed. On 32 counterfactual pairs, exchanging two class tokens switched the relevant classmate. The loop answered both versions in 32 pairs and the matched control in one. A separate explicit-rule test gave both models the induction rule in their system prompt. Three passes answered 32/32 and ordinary untouched Base answered 5/32 at 768 output tokens.

The deduction reader answered 112/112 across two panels versus 68/112 with one read. Retaining both learned reads while bypassing the second native span retained 56/56 on confirmation. The causal evidence here concerns information retrieval across reads. Kinship supplies a contrary depth result on human-written CLUTRR stories. Three passes scored 34/64 versus 1/64 for untouched Base, but a separately trained one-pass model scored 44/64. The depth gate closed the reserved official test. [3]

#note[*Interpretation.* The induction controls establish that this trained native layer and its successive states contribute to the measured answers. The evidence concerns short generated stories, a small answer vocabulary and annotated support facts. It places no numerical bound on performance in unrestricted natural language.]

= Language retention, cost and 128K context
#figure(image("cost.svg",width: 96%),caption: [Measured batch-one elapsed time. The vertical line marks equal time. Controls are labelled separately. The matched induction and deduction accuracy comparisons use short fixed-length outputs; the 768-token Base comparisons shown here allow both arms longer responses.])
On pooled human inference prompts, blind scoring credited 85/120 to the trained model and 66/120 to Base. Premise-clustered paired randomization gave $p = 0.0070$. On the 80-item pooled arithmetic panel at 1,024 output tokens, reviewed answer content scored 58/80 versus 52/80. An earlier 32-question development comparison gave two passes 13 completed correct answers to the matched ordinary model’s 10, in 64.64 versus 53.56 s with 10,083 identical supervised target tokens. In the separate 80-question equal-training depth study, the strict arithmetic endpoint was 52/80 for both depths. The same distinction matters for entailment, where the trained dense model tied the recurrent model at 85/120. A combined mixed panel yielded 55/80 versus 41/80 under review and 49/80 versus 33/80 under the strict scorer. Its full runs took 419.12 and 536.26 seconds. [4]

There are regressions. In science composition, the trained candidate completed more correct responses at the registered cap (23 vs 4 of 40), but scored 32 vs 37 on the matched answer probe. On an ordinary prose reading test the context candidate found all 20 supporting passages yet achieved exact match on 9 versus 10 for Base and F1 of 0.392 versus 0.517. Localization and successful answer generation are distinct objectives.

The standalone language release reproduced all 64 induction, 56 deduction and 80 mixed output streams from its qualified predecessors. All 192 negative-intent checks retained their previous route and nine complete greedy replays matched. The ordinary route reproduced NLL 2.982658 on 9,328 targets in 120 passages. This exact tie is a preservation result. On the two 131,083-token register prompts both full-history outputs contained the correct value after swapping equal-width key names. The release took 525.84 s in total with 4.33 GiB peak MLX allocation. Base scored 0/2 in 393.20 s. The pair measures sensitivity to a distant binding within one synthetic document. Broad 128K prose reasoning, long tool sessions, and a population accuracy estimate require separate evidence.

= Executable tool use
#include "tool-results.typ"
= Benchmark endpoints and external transfer

#tab([*Endpoint*],[*Recurrent*],[*Control*],
..csv("benchmarks.csv", row-type: dictionary).map(r => ([#r.endpoint],[#r.recurrent / #r.n],[#r.control / #r.n #r.comparator])).flatten(),
)
#v(4pt)
The public function test contains the first 24 consecutive questions from BFCL v3 simple, with publisher-authored typed, optional and array arguments. Both arms see the original question and function descriptions, the same instruction to use the function and a 384-token thinking allowance. We normalize the publisher's parameter type vocabulary to JSON Schema and check output names, required types and argument alternatives. This first-turn evaluation does not execute a callback. The catalog model completed all 24 turns in 3.78 seconds and supplied zero valid calls. Base completed 12, with eight valid and correct requests in 95.07 seconds. The catalog model frequently asked for a missing quantity even when it was explicit in the question. This failure motivated a separate preregistered training recipe with heterogeneous typed arguments and an external execution endpoint.

== Training and inference costs
The final-layer tool adapter occupies 3.06 MiB as FP32 factors over the frozen BF16 decoder. Its two fits had identical input batches and 19,364 scored response tokens. Prefix feature preparation took 25.13 seconds for two passes and 25.11 seconds for one. The optimizer consumed 39.47 and 25.11 seconds respectively. The matched run timed full interactive sessions, including multiple fresh prefills, greedy decoding, argument validation and Python callback execution. These are warm-model measurements. Startup, model download and construction of the combined export are outside the session timer. The shared-backbone tool view requires no second copy of the ordinary weights.

= Grounded reasoning over tool observations
The second Base-initialized fit repeats layers 20–23 twice. Each active pass includes three gated delta layers and one global attention layer. The repeated suffix has 1,803,776 trainable LoRA parameters. All preceding decoder features are cached during fitting. Written numeric comparisons and short intermediate references supervise the model's generated reasoning before the chat boundary. We also include 89 unrelated read-only demonstrations from the Hermes/Glaive corpus across 31 called function names. The resulting 818 teaching turns receive 760 AdamW updates, minibatches of four, and 64,707 response targets. Native Gated DeltaNet differentiation and activation checkpointing support training on the 18 GiB device. Feature preparation took 54.66 seconds and optimization took 647.85 seconds.

Ten episodes per task family span railway, tournament, warehouse and archive catalogs. Half the requests use a new wording construction. On this fresh panel the trained model completed all 50 episodes and succeeded on 38 episodes, versus 7/50 for Base. The two arms used 1,486 and 8,040 output tokens and took 35.43 and 119.05 seconds. The trained model gained 31 items with zero losses; the episode-level exact paired test gives $p < 10^(-8)$. Its end-to-end successes were 9/10 lookups, 6/10 chains, 4/10 comparisons, 10/10 direct answers and 9/10 clarifications. The four domains had net gains of 9, 7, 8 and 7 in the order given above.

The gate required at least 7/10 chains and 6/10 comparisons. The fit failed both floors. In six comparisons the model named a site without calling either data source. In another case it used an entity name as the key for a location measurement and repeated the error after the lookup failed. A finish marker and a plausible answer do not establish grounding. Counting final text alone would credit 42/50. The failed gate closed this checkpoint's deployment and its matched-depth follow-up.

#figure(image("trajectory-outcomes.svg",width:95%),caption: [Mutually exclusive terminal outcomes. The width is the share of enrolled episodes in that arm. A valid final answer supported by an incorrect or missing call falls in the finished, incorrect category.])

= External execution with typed arguments
The first catalog adapter learned to ask for a missing entity on a multi-argument request where every quantity was given. We addressed the input distribution and the starting model in a new trial. The typed model starts from Qwen3.5-0.8B at revision `2fc06364715b967f1860aea9cf38778875588b17`, an instruction-tuned sibling of the pinned Base. Frozen embeddings and linear weights outside layer 23 use affine 8-bit quantization in groups of 64. Layer 23 remains BF16 with shared rank-32 adapters and runs twice. Both study arms use the pinned Base tokenizer, identical chat messages, EOS and turn-end IDs, and free greedy decoding.

The new curriculum contains 384 two-turn interactions with eight local utilities covering dates, shopping, calibration, strings and lists, 89 public demonstrations and 128 controlled teaching turns. We generated a real callback result for each local training request. In total 1,074 supervised turns supplied 53,271 sampled response targets over 600 AdamW updates, batch four. The learning rate was 0.00012. Native feature preparation took 92.17 seconds, followed by 76.45 optimizer seconds. The trial changes training, parent checkpoint, recurrent depth and precision relative to Base.

The first 22 consecutive BFCL v3 executable-simple questions specify 11 numerical functions with two questions apiece. The original user request and typed signature are exposed, alongside the same instruction to call and report the returned number. A local callback executes with the generated arguments. Both models receive the actual returned value rounded to seven significant digits. The evaluator separately reads the publisher's literal name and arguments, executes the reference function and requires a final number within 0.5% relative error. For the reported success, the model must also have executed exactly one correctly typed publisher call. The three-turn limit and 384-token allowance apply separately to both arms. These results describe a local BFCL subset and score.

#tab([*On 22 public requests*],[*Typed recurrent*],[*Untouched Base*],
[Correct executed task],[17 / 22],[0 / 22],
[Correct final number],[17 / 22],[8 / 22],
[Completed final answer],[19 / 22],[12 / 22],
[Generated tokens],[1,084],[5,634],
[Full session time],[15.15 s],[82.82 s],
[Peak MLX allocation, warm model],[1.08 GiB],[1.66 GiB],
)
The recurrent system gained 17 episodes with zero losses. The exact episode-level paired test gives $p = 0.0000153$; a sign-flip sensitivity test by function gives $p = 0.00391$. It supplied 5 and 4 where the two compound-interest calls required 0.05 and 0.04. The resulting callback outputs were faithfully reported and numerically wrong for the requested question. The two long-list means and one velocity call failed argument parsing or validation. This decomposition distinguishes parameter grounding from fidelity to the observation.

= Transfer and matched controls for the frozen tool model
#include "transfer-results.typ"

= Counterfactual training across tool families
#include "joint-results.typ"

= Grounding live database and package requests
#include "field-results.typ"

= Offline deployment qualification
The tool release contains two text backbones, a shared tokenizer and the local runtime in 2.207 GiB of files. The previous language-only directory occupied 1.443 GiB. An explicit tool list selects the recurrent instruction-initialized tool model. For ordinary requests the wrapper releases the tool backbone and loads the qualified language system. The caller supplies each callback. The export contains no training data lookup or network service used for answering these requests.

The packaged tool branch reproduced all 22 public trajectories, including every input ID, emitted ID, call and observation. We also asked for the installed versions of MLX, NumPy, psutil, safetensors and PyYAML using live Python distribution metadata. Four of five answers contained the correct queried version. For psutil the model returned an incomplete statement without a version. An isolated interpreter with only the bundle and installed dependencies reproduced the first field trace in offline mode.

#figure(image("tool-latency.svg", width: 85%), caption: [Each point is one human-authored numerical request with its actual callbacks. Points below the dashed line finished earlier with the tool model. The symbol records whether the recurrent system completed the correct execution.])

Across the 22 numerical sessions the median time to a terminal outcome fell from 4.197 s to 0.614 s. The nearest-rank 90th percentiles were 5.602 and 1.016 s. Median first-turn time to the first token was slightly higher for the recurrent system (92.81 ms versus 91.46 ms). Aggregate generated tokens per session-second were 71.57 versus 68.02, including repeated prefill and callback time. The endpoint speed advantage comes primarily from shorter conversations.

A balanced crossover on these exact typed weights ran every numerical request four times. The two full-prefill measurements per request totaled 30.14 s across the panel; prefix reuse totaled 27.85 s. All 88 trajectories retained identical generated tokens, function arguments and actual observations. Reuse saved session time on 19 of 22 requests. The largest per-request slowdown was 1.7%. The checkpoint-specific policy reuses 6,656 token positions across the measured sessions.

After switching from tool use, 64 induction, 56 deduction and 80 mixed language generations matched the previous qualified export in full. All 120 raw prose requests used the ordinary path and all five domain losses matched, including pooled NLL 2.982658. Both complete 131,083-token requests reproduced the correct answer, exact token stream and located evidence span. The pair took 525.82 seconds at a peak of 4.33 GiB. This end-to-end test ties context capacity, ordinary language retention and executable tools to the same export inventory.

== Reusing the tokenizer across live mode switches

Four fresh processes each served the same 50 requests, with policies in ABBA order. Each process held one active backbone at a time. The reuse policy checks tokenizer-defining files before reusing the BPE object. It rebuilt two tokenizers over its two streams, compared with 32 over the two reload streams. All 200 generated trajectories, actual callbacks and terminal outcomes reproduced the frozen field reference. Across 30 switches per policy, first-token time fell 44.3%, from 29.43 to 16.39 s. Summed first-token time over all requests was 26.39 versus 39.45 s; wall time was 152.03 versus 164.20 s. Peak MLX bytes matched at 1,810,571,734. Whole-process maximum RSS was 1,916,698,624 versus 2,701,754,368 bytes. The two policies ran on the same packed release. The installed runtime enables reuse for the exact qualified inventory. The original self-contained archive remains identifiable by its published hash.

#figure(image("tokenizer-streams.svg",width:94%),caption:[Each point measures one entire 50-request stream in a fresh process. Switch latency includes all work from the incoming request to its first emitted token.])

== Folding adjacent native layers into one mixed-use backbone

The released assistant changes backbones when a caller switches between tools and ordinary language. Bae et al. [38] initialized recursive models from unshared transformers and used depth-specific adapters to recover specialization. We test a local conversion of the qualified instruction sibling. We discard gated-delta layer 21 and run the same physical layer 20 twice, maintaining two convolution and delta states. A rank-192 bottleneck sees the second call’s input and residual. It learns to reconstruct the removed block’s output in units of the input RMS. The 23 remaining physical layers supply 24 layer calls for ordinary text and 25 for tools, which also use the existing final-layer tool adapter. Both modes share the same instruction arrays. The fitting targets are native hidden vectors from 512 prose excerpts and 512 distinct tool conversations, with 24 sampled positions per document. Another 64 documents measure error against skipping the removed transform or repeating its predecessor without repair.

The public comparison uses 20 fresh human word problems, 20 fresh human BoolQ questions, all 22 retention calls from the public numerical suite and the 30 live file, database and package requests of the field benchmark. We preserve each initial prompt ID, repeat the same shuffled 92-request stream on untouched ordinary Base and include the initial load, subsequent switches, actual tool observations and terminal outcomes. Any lost previously correct public action within these subsets closes deployment. A matched one-application attribution study and independent long-context qualification require a passing direct result first.

== Lossless distribution of the complete assistant
The qualified bundle contains separate language and tool backbones. We compared ordinary level-six DEFLATE of its complete file inventory with a reversible representation that uses the original Base tensors and the sibling's actual scale and bias to predict quantized weight bytes. The archive stores the residual bytes; the decoder reconstructs the exact original tensor headers and payloads. The encoding is applied before DEFLATE and leaves the installed model precision intact.

The installed 2,370,262,907-byte bundle became a 1,996,837,806-byte ZIP or a 1,544,018,439-byte paired archive. The paired representation saved 34.86% versus the installed bundle and 22.68% versus ordinary ZIP. Its size was 2.60% greater than the 1,504,827,865 bytes of the Base text weights alone. Ordinary ZIP creation took 100.69 s, the custom encoder took 143.71 s, and verified restoration took 7.85 s. Every restored file matched its original SHA-256. A clean offline interpreter using only the restored runtime reproduced the live package-version call, observation and output tokens. Installed size and the cost of switching backbones remain separate deployment measurements.

#figure(image("transport.svg",width:93%),caption:[Measured bytes and times for the identical qualified payload. The Base text-weight file is one component of the installed assistant. The complete archive is usable for an offline restoration.])

The qualified packed-03 release replicated the transport result on updated runtime files. Its 63-file payload totals 2,370,301,274 bytes. Ordinary ZIP used 1,996,851,217 bytes and the paired encoding used 1,544,032,264 bytes, 22.68% below ZIP and 34.86% below the installed files. The two encoders took 100.91 and 143.19 s. Restoration and file verification took 7.07 s. An isolated interpreter using the restored runtime then reproduced the reference package lookup and exact native tokens. The release-specific CSV records the two separately qualified archive hashes.

= Human-edited reading questions
#include "reading-results.typ"

= The cost of full-history attention
The released decoder repeats layers 10–15, including two global attention sites. Two separately registered experiments tested whether changing where it spends that work preserves the ability to use an earlier record. Each paired run includes both an original ledger and a counterfactual with the same numbers at the same token positions but two record names swapped. Timing includes locating the evidence. Each length contributes one pair per study.

#figure(image("register-efficiency.svg", width: 96%), caption: [Four actual paired requests per graph. Each circle or square is a request; lines join the means at nominal lengths 16K and 32K. The right panel counts completed correct answers. The two experiments use different fixed seeds.])

Replacing global layer 7 with a second application of gated-delta layer 6 retained all four correct answers at five full-history attention sites. Base answered zero. Aggregate time to first token was 54.30 versus 54.73 s, below the required 5% improvement. A separate 25-site model repeated global layer 15, retained full histories at its second application and layers 19 and 23, and used 1,024-token local windows at four other sites. It located all four true spans and reduced aggregate first-token time to 49.36 versus 54.61 s. It answered zero questions correctly. The remaining global history was large enough to hold the answer, but the new attention schedule changed the decoder's use of it. Both preregistered gates closed further long-context promotion of those graphs.



A third study kept the released eight-global-site graph intact and changed only the allocation of full-history keys and values. All four outputs, selected spans and consumed offsets matched. Reserving the full prompt and decoding allowance used 2.29 GiB at peak versus 2.37 GiB with incremental allocation. Summed first-token time was 74.45 versus 76.12 s, a 2.20% saving. Total time was 77.35 versus 79.01 s. This fell short of the registered 5% threshold, closing the additional physical 128K trial for this allocation policy.

== Causal page selection during prefill
The retriever keeps every K/V row at all eight global sites. For an interior 128-token query block it uses the first query, which is causally visible throughout that block, to rank mean keys of completed pages. It attends the first page, sixteen selected historical pages, fifteen recent pages and all causally available tokens in the current page. The last 512 input tokens and every output token receive dense full-history attention. Earlier hidden states can differ, so the comparison requires exact output IDs and evidence spans. MInference and FlexPrefill motivate sparse computation during prefill. KIVI studies the separate question of reducing stored KV precision, with different schemes for keys and values [25]. Their hardware, page-selection policies and measured speedups differ from this native MLX design. [22, 23]

#figure(image("page-prefill.svg",width: 95%),caption: [Directly measured time to first token and the proportion of causal query–key pairs attended. Every global site physically retained every consumed position. The two pilot arms use identical recurrent weights.])

On two 16,400-token inputs and two 32,783-token inputs all four outputs matched stock token for token. Summed first-token time fell from 76.09 to 58.48 s (23.13%). Total time fell from 78.96 to 61.34 s. Maximum MLX allocation fell from 2,545,878,064 to 2,535,916,474 bytes. The mechanism attended 31.20% of the available causal query–key pairs across the four requests. Each smaller request retained 16,429 physical positions at every site and each longer request retained 32,813.

On a fresh 131,073-token document the queried name moved between earlier and later entries while every digit stayed at its original position. The recurrent decoder returned 409799 and 947623 in 30 output tokens per request. Base reached the 96-token cap on both. Aggregate first-token time was 173.99 versus 384.79 s and full request time was 176.55 versus 391.72 s. Each recurrent site processed 987 sparse blocks, kept all 131,102 consumed positions and attended 7.05% of its possible causal pairs. The six Base sites each kept all 131,168 consumed positions. The peak allocation of 4,643,439,100 bytes exceeded Base’s 4,059,680,512 by 14.38%. This measures one document with two counterfactual placements. It establishes a working 131K retrieval intervention on this machine. It does not estimate a population accuracy or a latency distribution. Its preregistered peak-memory condition closed promotion.

A storage-only comparison packed every key and value into affine groups of 64 using eight bits per channel. All four generated outputs and located evidence windows exactly replayed the BF16 page control. On a 32K case the live codes, scales and biases at eight sites occupied 285,604,352 bytes, compared with 537,608,192 bytes for BF16 values at the same positions. Peak allocation fell by just 1.28%, from 2,535,883,706 to 2,503,356,928 bytes. Native attention still restored full BF16 arrays for dense queries. First-token time totaled 58.03 versus 58.24 s and end-to-end time 62.05 versus 61.09 s. The 5% peak threshold closed this storage-only implementation. The next implementation reserved packed buffers once, gathered the selected pages before restoring them during interior prefill, and computed dense attention directly against packed keys and values in tiles of up to eight queries. It replayed four of four original token sequences. Peak allocation fell 15.11%, from 2,535,900,090 to 2,152,767,034 bytes. The summed first-token and end-to-end times increased from 58.32 to 62.71 s and from 61.21 to 66.00 s. This satisfied the engineering gate of 5% lower peak and at most 15% longer execution, opening a separate full-length comparison against untouched Base.

On an independent 131,080-token document with another pair of names, the direct packed decoder returned 270761 for the earlier placement and 145163 after the switch. Each generation used 31 tokens. Base reached the 96-token limit twice. Aggregated over both requests, time to first token was 168.35 versus 384.65 s and total time 171.88 versus 391.67 s. Peak MLX allocation fell from 4,059,680,512 to 3,004,581,396 bytes, a 25.99% reduction. Each of the eight recurrent global sites kept 131,110 positions, visited 987 sparse blocks and computed 678 dense queries directly against packed arrays in 111 tiles. It attended 606,503,653 of 8,594,981,605 available causal pairs per site. All six ordinary Base sites retained 131,175 positions. Prompt tokens per first-token second increased from 682 to 1,557. After the first token, the recurrent decoder emitted 60 further tokens in 3.53 s (16.99 tokens/s), versus 190 in 7.01 s (27.09 tokens/s) for Base. Its shorter completed answers and faster prefill account for the end-to-end win despite slower individual output steps. The deployed-resource condition, the exact answers and the physical-history audits all passed. This qualifies the mechanism on the measured register task. The resulting self-contained export passed a same-weight retention comparison against the earlier field release. The context classifier and every model weight were byte-identical. The upgrade applies only to resident register requests of at least 16,384 tokens using 128-token prefill blocks. Each run began with a fresh cache and interpreter and included short text, a register, natural prose, more short text, and a live package lookup. Order was candidate–parent in the first block and parent–candidate in the second.

On two fresh 24,593-token documents both packages returned both correct answers with identical tokens. The new release used 32.44 versus 37.52 s in total and 30.82 versus 36.12 s to first token. Peak allocation fell from 2,417,359,042 to 2,085,227,658 bytes (13.74%). On *A Christmas Carol* (41,196 tokens) and *Strange Case of Dr Jekyll and Mr Hyde* (34,682), both packages used the ordinary raw-text route. Each physically consumed the preceding history and scored the final 256 natural tokens. Every loss and argmax matched exactly. Their per-book NLLs were 3.00960 and 3.41013 with 102 and 90 correct predictions respectively. Both short suites and the two live version lookups retained all output tokens and actions. Across the two books the upgraded export took 48.55 versus 48.72 s and used 16,420 more peak bytes. Total cold initialization across both blocks was 1.37 versus 1.26 s. The long-register speed and memory gains apply to the selected register path.

#figure(image("prefill-resource-comparisons.svg", width: 95%),caption: [Measured resource differences within each registered comparison. Values below zero favor the candidate. Each 131K total includes both placements of the queried record. The scale reports relative change from the control named on each row.])

*Single-token attention on physical retained rows.* At 131,080 tokens, the direct packed model's original output rate was 16.99 tokens/s against 27.09 for ordinary Base. A two-stage Metal operator computes per-partition maxima, normalizers and weighted values directly from the packed affine bytes, and merges the partitions with one shared softmax normalization. The comparison captures actual queries, keys and values after the first three Gated DeltaNet layers and the projections of Qwen's first full-attention layer. For each length and mask we timed four counterbalanced repeats of eight distinct queries, with exact-shape compilation outside each timer.

#figure(image("packed-decode.svg",width:94%),caption:[Single-query attention latency on physical Qwen cache rows. Small points show the four-repeat median of each actual query. The two panels compare the ordinary causal history and an additive learned register bias on the same tensors.])

Each Metal partition maintains a running maximum, exponential sum and weighted-value sum in FP32. A second native reduction combines the partitions before rounding the output to BF16. The online normalizer [35] and work partitioning in FlashAttention-2 [36] give the mathematical and systems context. Our measurements concern batch-one Apple GPU decoding over native affine 8-bit cache rows.

At 4,112 tokens the existing operator was faster in both conditions. At 32,784 tokens the fused operator was 1.22× and 1.24× as fast for the plain and reader masks. At 131,080 tokens it took 2.42 and 2.48 ms against 4.67 and 4.69 ms, respectively. Maximum scaled absolute difference was 0.00758 and minimum cosine similarity was 0.999993. Peak additional workspace increased from 2.25 MB to 8.83 MB in the longest reader condition. The registered operator threshold passed. The full-model crossover measures both complete counterfactual requests twice per policy, including every prefill block and generated token. The single-query integration begins at 32,768 retained positions and preserves the existing multi-query prefill path.

#figure(image("full-decoder.svg",width:95%),caption:[Every 131,080-token request, in execution order. Both implementations used the same model and emitted the same 31 tokens for each paired input. Two counterbalanced repetitions expose variation in prefill time.])

All eight complete requests reproduced the historical correct tokens and selected source span. The four fused requests retained every key and value and made 960 single-query fused site calls (30 further output tokens across eight sites on four requests). Pooled post-first-token speed was 26.10 versus 17.09 tokens/s, a 52.7% gain. The 120 further tokens took 4.60 versus 7.02 s. Summed first-token time was 336.18 versus 335.50 s, and full requests took 340.78 versus 342.53 s. Maximum MLX allocation differed by only 16,384 bytes (3,004,426,260 versus 3,004,442,644). Thus a substantial change in output rate saved 1.75 seconds over these four long reads. The direct Base comparison uses a fresh 131,084-token counterfactual with every numeric value fixed in place.


== An entire natural novel
We tested the offline assistant on Frank Norris's _McTeague_ using a human-written NarrativeQA question about who reports McTeague's missing dental license to City Hall [19]. Both publisher annotators named Marcus. In the second version we changed all 440 occurrences of Marcus, 24 of Mark and 65 of Schouler to Felix, Ned and Radford respectively. We kept the question and every chapter. The two prompts contained 162,714 and 162,711 tokens.

The release selected its ordinary-depth route on both books. On the original, both systems exhausted all 384 output tokens without supplying a visible answer. On the renamed text both completed a 195-token generation identifying the sheriff. Neither supplied the correct name on either version. All output tokens matched between systems, and all six attention sites retained 163,097 and 162,905 consumed positions respectively. Aggregate first-token times were 571.28 and 571.31 s, full request times 596.38 and 596.42 s, and peaks 4,658,671,388 and 4,653,289,216 bytes. This cleanly separates demonstrated full-history capacity from the ability to answer a human question about it. It closes the claim of a narrative-comprehension gain for the shipped router on this intervention.

= Causal raw-language prediction
#include "raw-results.typ"

= Relationship to prior work
API-Bank [37] used manually annotated conversations and executable APIs to study calling, retrieval and planning. Our package and file evaluations resolve properties from a live local installation, and our arithmetic evaluations also score a correct final answer when the model avoids a call. The distinction changes which model wins on the human mathematics subset.

Universal Transformers share a transformation across depth and study adaptive computation [5]. Geiping and colleagues pretrained a 3.5B recurrent-depth architecture on 800B tokens and scaled latent iterations at inference [6]. Here an already pretrained hybrid decoder is adapted at small scale. The project measures a benefit from a particular supervised recurrent computation, under a matched one-pass budget. The earlier three-seed direct-span study averaged 23.7% versus 32.6% for dense adaptation on out-of-distribution traced problems. Simply repeating the pretrained span was insufficient in that study.

Weston and colleagues' bAbI tasks isolate reasoning operations [2]. Their explicit support labels make direct attention supervision feasible here. CLUTRR varies kinship relation compositions [3]. The narrower induction advantage and the adverse kinship depth comparison illustrate the cost of extrapolating across relation systems. RULER benchmarks multiple effective-context skills across lengths [7]. Our register intervention isolates key identity at fixed positions, alongside separate prose evaluations.

The Berkeley Function-Calling Leaderboard distinguishes argument structure, executable behavior and multi-turn state [8]. τ-bench evaluates conversational agents interacting with stateful environments under domain policies [9]. We borrow the distinction between emitting a plausible call and reaching a verified outcome. Our hidden worlds are deterministic, callbacks are local and read-only, and tasks have short horizons. The separate BFCL slice examines argument transfer. BoolQ supplies naturally occurring human yes/no questions; the Gardner et al. contrast set pairs an original question and a human edit with the same passage [10, 11]. Neither benchmark's aggregate leaderboard score can be inferred from this experiment.


Program-aided language models move arithmetic execution into a real interpreter [12]. Our calculator study separately scores a correct visible number and the actual expression accepted by a restricted rational-arithmetic tool. ASDiv supplies independently authored arithmetic questions [13]. WikiSQL supplies human questions paired with executable table queries [14]. SQLova factors this task into supervised choices of column, aggregate, conditions, operators and values [24]. Our schema-conditioned parser also predicts these choices, then sends its call to the ordinary runtime and uses the recurrent tool decoder to answer from the returned rows. Replaying a model-generated query on a second row population checks whether a fortuitous aggregate value survives an intervention. HotpotQA provides natural bridge questions with labeled supporting articles [15]. Its annotations supervise the recurrent document reader during training. Continuous neural caches use similarities to earlier hidden states to redistribute probability to observed tokens [16]. We test whether keys from an extra shared decoder pass can pay for that work in held-out likelihood and real prediction time.

PICARD used incremental parsing during decoding to improve text-to-SQL results on Spider and CoSQL [17]. Our crossover measures the effect of restricting a call’s JSON representation on the downstream result of executing it. MultiWOZ 2.2 provides human dialogue and corrected state annotations [18]. NarrativeQA supplies questions about entire stories [19]. For the whole-book intervention we retain every chapter and change the target character's names consistently throughout the text.

= Limitations and scope
Across the public numerical function requests, competing-schema questions, SQLite questions, calculator word problems, natural bridge questions and venue conversations, no enrolled human request exactly matched a local fit input after case and punctuation normalization. The supplement reports the maximum five-word-shingle Jaccard similarity in each panel with and without masking numbers. The structured table parser excludes training questions with a normalized five-gram Jaccard similarity of at least 0.60 to an enrolled table question. The reading passage exceptions and their sensitivity analyses appear above. This screen detects lexical reuse and cannot establish semantic novelty. Public benchmark questions could have appeared in backbone pretraining. Exclusions and disjoint fine-tuning splits establish experiment-level separation. Synthetic worlds have a shared task generator, overlapping entity vocabularies and only a few linguistic constructions. Distinct function names test some schema transfer; the statistical unit is an episode conditional on a small number of domains. Correlation within a domain can make an item-level test optimistic. Single-device timing includes runtime and thermal variability. Most tool comparisons run one complete arm and then the other on the same item order. The page-prefill studies alternate the measured arm order by request. Their reported first-token and full-generation timers start after tokenization and model loading, immediately before learned route selection. They include evidence selection, every prefill block and greedy decoding. Cold startup is timed separately in the deployment studies. The resource measurements cover elapsed time and native allocations. They do not measure device energy. Peak MLX bytes omit some process and system memory. Adapter training consumes additional compute even when inference finishes sooner. The project adapts subsequent experiments to earlier results and repeatedly evaluates public tool retention. Reported paired probabilities apply to individual comparisons without a project-wide multiplicity correction. Exact replay on reused items measures retention.

The model is a conditional system. Its different routes have distinct training histories, including previously adapted language components. The catalog fits start from untouched Base. The independently registered typed-tool experiment uses the official instruction-tuned sibling and records that different initialization. Deployment-level improvements do not establish a positive derivative with respect to recurrent depth. A broad claim would require fresh user-selected tools with richer data types, independently authored multi-turn requests, longer policies, replication across seeds and hardware, and equal-training comparisons on those populations.

= Reproducibility and evidence
The supplement indexes registered studies and their archive SHA-256 inventories, including negative results and recovered sessions. Each tool record contains the initial IDs, subsequent prompt IDs, sampled IDs, timings, calls, observations and terminal status. The verifier reconstructs typed calls and hidden database values from the frozen worlds, reconstructs subsequent prompts and recomputes the endpoint. The installed `ask --compare-base` command also accepts an arbitrary local UTF-8 prompt file, optional read-only tools and follow-ups. It times the cold request from before loading each backbone and preserves each model’s own subsequent conversation. In an illustrative calendar exchange outside the registered scoring panels, the two-pass intent adapter returned the correct ISO weekday (7) and ordinal day (60) for February 29, 2032 through separate provider calls in 1.29 and 0.54 s. Ordinary Base exhausted both caps in 6.35 and 5.51 s. The initial prompt IDs matched. Every follow-up inherited the corresponding prior output and observations. These two questions constitute an interface check. Figure sources are recorded in `figure-sources.json`. Build and replay commands accompany the manuscript.

= Sources and artifact guide
#include "compute-results.typ"
[1] Qwen Team. Qwen3.5-0.8B-Base. #link("https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")[Pinned checkpoint and text configuration].

[2] Weston et al. (2015). Towards AI-complete question answering. #link("https://arxiv.org/abs/1502.05698")[Paper]. Study IDs `native-induction-anchored-01`, `native-induction-anchored-replication-01`, `native-induction-class-swap-01` and their registered controls.

[3] Sinha et al. (2019). CLUTRR, a diagnostic benchmark for inductive reasoning from text. #link("https://aclanthology.org/D19-1458/")[EMNLP-IJCNLP]. See `native-kinship-01` and `native-kinship-matched-01`.

[4] Cobbe et al. (2021). Training verifiers to solve math word problems. #link("https://arxiv.org/abs/2110.14168")[Paper]. Williams et al. (2018). A broad-coverage challenge corpus for sentence understanding through inference. #link("https://aclanthology.org/N18-1101/")[NAACL]. Camburu et al. (2018). e-SNLI. #link("https://arxiv.org/abs/1812.01193")[Paper]. Patel et al. (2021). Are NLP models really able to solve simple math word problems? #link("https://aclanthology.org/2021.naacl-main.168/")[NAACL]. Study families `learned-exit`, `entailment-triplets` and `conditional-language`.

[5] Dehghani et al. (2019). Universal Transformers. #link("https://arxiv.org/abs/1807.03819")[ICLR].

[6] Geiping et al. (2025). Scaling up test-time compute with latent reasoning. #link("https://arxiv.org/abs/2502.05171")[Paper and model].

[7] Hsieh et al. (2024). RULER. #link("https://arxiv.org/abs/2404.06654")[Paper]. Full-history study `conditional-language-128k`, standalone replay `native-language-release-02`.

[8] Patil et al. (2025). The Berkeley Function Calling Leaderboard. #link("https://proceedings.mlr.press/v267/patil25a.html")[ICML]. The BFCL data are Apache-2.0. Our retained subset uses original questions, signatures and alternative argument values from revision `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda`.

[9] Yao et al. (2024). τ-bench. #link("https://arxiv.org/abs/2406.12045")[Paper].

[10] Clark et al. (2019). BoolQ, exploring the surprising difficulty of natural yes/no questions. #link("https://aclanthology.org/N19-1300/")[NAACL].

[11] Gardner et al. (2020). Evaluating models’ local decision boundaries via contrast sets. #link("https://aclanthology.org/2020.findings-emnlp.117/")[Findings of EMNLP].

[12] Gao et al. (2023). PAL, program-aided language models. #link("https://proceedings.mlr.press/v202/gao23f.html")[ICML].

[13] Miao, Liang and Su (2020). A diverse corpus for evaluating and developing English math word problem solvers. #link("https://aclanthology.org/2020.acl-main.92/")[ACL].

[14] Zhong, Xiong and Socher (2017). Seq2SQL and the WikiSQL dataset. #link("https://github.com/salesforce/WikiSQL")[Original dataset and scripts].

[15] Yang et al. (2018). HotpotQA, a dataset for diverse, explainable multi-hop question answering. #link("https://aclanthology.org/D18-1259/")[EMNLP].

[16] Grave, Joulin and Usunier (2017). Improving neural language models with a continuous cache. #link("https://openreview.net/pdf?id=B184E5qee")[ICLR].

[17] Scholak, Schucher and Bahdanau (2021). PICARD, parsing incrementally for constrained auto-regressive decoding from language models. #link("https://aclanthology.org/2021.emnlp-main.779/")[EMNLP]. #link("https://github.com/guidance-ai/llguidance/blob/main/docs/json_schema.md")[llguidance JSON Schema semantics].

[18] Zang et al. (2020). MultiWOZ 2.2. #link("https://aclanthology.org/2020.nlp4convai-1.13/")[NLP4ConvAI]. #link("https://github.com/budzianowski/multiwoz")[Original dialogues and database].

[19] Kočiský et al. (2018). The NarrativeQA reading comprehension challenge. #link("https://aclanthology.org/Q18-1023/")[TACL]. #link("https://www.gutenberg.org/ebooks/165")[Full text of McTeague].

[20] Amini et al. (2019). MathQA, towards interpretable math word problem solving with operation-based formalisms. #link("https://aclanthology.org/N19-1245/")[NAACL].

[21] Koncel-Kedziorski et al. (2016). MAWPS, a math word problem repository. #link("https://aclanthology.org/N16-1136/")[NAACL].

[22] Jiang et al. (2024). MInference 1.0, accelerating pre-filling for long-context LLMs via dynamic sparse attention. #link("https://proceedings.neurips.cc/paper_files/paper/2024/hash/5dfbe6f5671e82c76841ba687a8a9ecb-Abstract-Conference.html")[NeurIPS].

[23] Lai et al. (2025). FlexPrefill, a context-aware sparse attention mechanism for efficient long-sequence inference. #link("https://proceedings.iclr.cc/paper_files/paper/2025/hash/03645743ea35690f30d795d6bac149a5-Abstract-Conference.html")[ICLR].
[24] Hwang et al. (2019). A comprehensive exploration on WikiSQL with table-aware word contextualization. #link("https://github.com/naver/sqlova")[SQLova paper and implementation].

[25] Liu et al. (2024). KIVI, a tuning-free asymmetric 2-bit quantization for KV cache. #link("https://proceedings.mlr.press/v235/liu24bz.html")[ICML].

[26] Xiong et al. (2021). Answering complex open-domain questions with multi-hop dense retrieval. #link("https://openreview.net/forum?id=EMHoBG0avc1")[ICLR].

[27] Wang et al. (2021). Selective knowledge distillation for neural machine translation. #link("https://aclanthology.org/2021.acl-long.504/")[ACL-IJCNLP].

[28] See, Liu and Manning (2017). Get to the point, summarization with pointer-generator networks. #link("https://aclanthology.org/P17-1099/")[ACL].

[29] Rajpurkar et al. (2016). SQuAD, 100,000+ questions for machine comprehension of text. #link("https://aclanthology.org/D16-1264/")[EMNLP].

[30] Choi et al. (2018). QuAC, question answering in context. #link("https://aclanthology.org/D18-1241/")[EMNLP].

[31] Zhang, Baldridge and He (2019). PAWS, paraphrase adversaries from word scrambling. #link("https://aclanthology.org/N19-1131/")[NAACL].

[32] Lei et al. (2016). Semi-supervised question retrieval with gated convolutions. #link("https://arxiv.org/abs/1512.05726")[Paper] and #link("https://github.com/taolei87/askubuntu")[human relevance judgments].

[33] Chen et al. (2017). Enhanced LSTM for natural language inference. #link("https://aclanthology.org/P17-1152/")[ACL].

[34] Yao et al. (2023). ReAct, synergizing reasoning and acting in language models. #link("https://arxiv.org/abs/2210.03629")[ICLR].

[35] Milakov and Gimelshein (2018). Online normalizer calculation for softmax. #link("https://arxiv.org/abs/1805.02867")[Paper].

[36] Dao (2024). FlashAttention-2, faster attention with better parallelism and work partitioning. #link("https://arxiv.org/abs/2307.08691")[ICLR].

[37] Li et al. (2023). API-Bank, a comprehensive benchmark for tool-augmented LLMs. #link("https://aclanthology.org/2023.emnlp-main.187/")[EMNLP].

[38] Bae et al. (2025). Relaxed recursive transformers, effective parameter sharing with layer-wise LoRA. #link("https://proceedings.iclr.cc/paper_files/paper/2025/hash/54d6a55225cebbdc16fbb0e45c5bdf2b-Abstract-Conference.html")[ICLR].

*Earlier stages.* The pilot and Phases 2–4 studied procedural questions, residual gates and training seeds. Phases 5–7 studied state supervision and interventions. Randomized training achieved 3,067/3,072 correct third-hop decisions over shuffled presentations of 384 permutation maps. Sixth-hop complete paths fell from 190/192 on permutations to 6/192 on arbitrary maps. Adaptive replay won at equal updates and lost at matched time. The phase reports under `docs/` contain the configs, negative runs and measurements. The study-index CSV covers the subsequent registered research program. #link(repo)[Project and phase reports].
