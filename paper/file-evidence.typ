#set document(title:"When only the file changes",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:0.8in,y:0.75in),footer:context [#set text(size:8pt,fill:rgb("#48625e"))
LoopQwen-MLX · Native file evidence #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:11pt)
#set par(leading:0.55em,spacing:0.60em)
#show heading.where(level:1): h => block(above:8pt,below:3pt)[#set text(size:16pt,fill:rgb("#125b51"))
#h]
#text(size:29pt,weight:"bold")[Where shared decoder depth earns its cost]
#linebreak()
#text(size:15pt)[Research report · tools, language reasoning and full-history inference on Apple Silicon]
#v(4pt)
November 2026 · LoopQwen-MLX project

= The project decision
The deployment goal is one offline recurrent model that improves useful answers over the original Qwen3.5-0.8B-Base, preserves existing behavior and can act on at least 131,072 input tokens. We measure the time to the first token, total request time, decoding rate and peak allocation alongside answer quality. A claim about recurrence itself additionally needs an equally trained one-pass control. The current evidence supports different conclusions for different fixed checkpoints.

#include "file-evidence-synthesis.typ"

= How to read the findings
The first study combines reasoning and function use in the same request. A correct response requires the model to retrieve the caller's own passage and interpret it. The second study holds the claim constant and changes the inline source. Two following studies test trained and interpolated instruction-derived loops on real table lookups, package inventories and human referent questions. The original technical paper then provides architecture, matched depth controls, native kernels, language retention and the offline 128K release. Each stage keeps its original decisions and timings.

The looped induction model's advantage replicated across two matched panels. The long-context document reader followed a genuine Git file-size change. On the two 131,084-token register requests the mixed-precision recurrent system answered both in 167.26 s versus zero in 390.86 s for Base. It processed 1,586 versus 683 input tokens/s, decoded 29.27 versus 27.42 tokens/s and used 3.32 versus 3.78 GiB peak MLX allocation. The qualified default subsequently reproduced both changed values from an empty offline installation. These observations apply to the identified weights and workloads. No study in this edition establishes simultaneous superiority across general language, tools, full-context reasoning and every cost measure for a single deployment.

#pagebreak()
#text(size:26pt,weight:"bold")[When only the file changes]
#linebreak()
#text(size:14pt)[Can a shared recurrent decoder read a real source and revise its answer?]
#v(4pt)
November 2026 · LoopQwen-MLX project

= Abstract
#include "file-evidence-abstract.typ"

#figure(image("file-evidence.svg",width:100%),caption:[Individual correct judgments, actual grounded file replies and independent article pairs. Each pair requires opposite correct conclusions.])

= A test the user can repeat
Place two versions of a clipping in separate folders under the same filename. Each file contains an article key and a passage, together with two genuine distractor rows. Ask the same question twice in independent conversations. The filenames, question, column names, function declarations, row order and all opening model tokens match. Exactly one CSV cell changes. For example, if a reported count crosses a threshold, both versions of the response can only be right if the assistant uses the observation it just received. The direct condition supplies original passages in the prompt to different articles. The two conditions use the same decoder. The following sections preserve every enrolled request, including failed tool calls and responses.

#pagebreak()
= Architecture and fitting
The starting model is the untouched Qwen3.5-0.8B-Base text decoder from the certified local release. Layers 0–21 use an eight-bit frozen prefix. The final Gated DeltaNet layer and the final full-attention layer form a shared block. Both execute twice on every input and every emitted token. The repeated block mixes its second output into its first output with a residual coefficient of 0.35. Attention and recurrent states are separately allocated at each execution site. During deployment the low-rank updates are fused into the respective BF16 maps. Both the training and the complete free-generation comparisons ran on the project's 18 GiB M3 Pro under macOS 27.3 (26C5047e), Python 3.12.12, MLX 0.32.4 and mlx-lm 0.32.2.

The training source comprises 384 distinct articles with original opposite human judgments from the VitaminC training split. For each article the model receives both an inline reading and readings after a genuine CSV lookup. Training contains only one opening file call for each identical pair of openings. Both file-return turns are present. An additional 128 messages from original WikiSQL training tables rehearse other lookup questions and returned values. The 28 held-out articles and all 24 from the preceding revision study are excluded by title. Exact evaluation claims and passage texts are checked against the selected training material. After replacing numbers with a shared symbol, both passages from one film have an exact verbal template match to training. Both passages from a second film have five-word Jaccard similarity of 0.73. None of the 28 claims reaches 0.60. The primary retains these articles. We also report scores excluding both.

#include "file-evidence-fit.typ"

= Why training and deployment differ
At fitting time the correct opening call demonstrates how to request the source. At inference the model has to write its own complete call in Qwen's native protocol. The runtime validates arguments, opens the actual document, checks the key is unique, then adds the actual returned content or error to the conversation. The same model produces the visible answer. Both systems receive identical tools, a 640-token allowance per turn and up to three turns with thinking enabled. There is no forced tool choice or restricted answer vocabulary. A guessed correct judgment is retained in the individual answer count. A file grounding success requires the correct actual callback and the correct completed judgment.

#pagebreak()
= Field decisions and serving costs
#include "file-evidence-costs.typ"

#include "file-evidence-reading.typ"

Some historical edits are corrections or contain inaccurate assertions. Each judgment concerns what the supplied passage supports. The independent source check retrieved all 28 historical Wikipedia edit IDs and their immediate parents. The actual archived changed lines have at least 50% three-word-sequence overlap with both supplied excerpts on opposite sides of 27 edits. The project’s AI coding agent read all 28 source pairs before model evaluation. For the secondary answer reading it received the claim, passage and visible reply for each noncanonical completion, with the arm, publisher label, timing and score withheld. This single-reader judgment has no inter-rater agreement estimate. Every pair remains in the registered primary. For example, the stated building heights of 925 and 927 feet give opposite answers to a threshold of 926 feet. A description of a city as the most populous also supports calling it one of the most populous, so that publisher contrast warrants qualification.

= Relation to earlier results
Schuster, Fisch and Barzilay introduced contrastive evidence built from Wikipedia edits. API-Bank evaluates the execution of model-selected APIs. Toolformer learns to place and use tool calls in language generation. Here we intervene on the caller's file while holding the entire opening prompt fixed, then grade the actual returned evidence and its subsequent use. The earlier chapters examine the complementary settings of semantic reasoning with an inline source, genuine WikiSQL lookups, full-length project documents, and equally trained one-pass controls for induction.

#link("https://aclanthology.org/2021.naacl-main.52/")[VitaminC, NAACL 2021] · #link("https://aclanthology.org/2023.emnlp-main.187/")[API-Bank, EMNLP 2023] · #link("https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html")[Toolformer, NeurIPS 2023].

#pagebreak()
= What one article contributes
Each article supplies two related user requests. A file pair is successful only when both requests obtain the intended cell and answer correctly. An inline pair requires two correct answers. The two systems see the same articles. The table counts articles where both systems succeeded, where exactly one succeeded and where neither succeeded.

#include "file-evidence-uncertainty.typ"

The interval estimates the chance of a loop win conditional on the two systems disagreeing on an article. It inverts the two exact binomial tails at 2.5% each. Articles on which the systems tie appear in the table and do not enter that interval. The primary test uses the same discordant articles for a two-sided exact McNemar test. With 16 file pairs, six wins and zero losses give p = 0.03125. Five wins and zero losses give p = 0.0625. An improvement of five pairs alone therefore cannot pass. The inline articles and independent readings expose regressions and interpretation differences. The small panels and shared publisher limit extrapolation to other texts. Each p-value is conditional on its registered comparison. The project has tested many model designs. Selecting the most favorable historical p-value across them would overstate evidence for a general gain. We retain the failures and identify the induction result that actually replicated under a matched depth control.

= Measuring a completed task
Actual wall time includes first construction, prompt processing, reasoning tokens, generated function calls, callback returns and the final reply. Both arms have the same per-turn token ceiling. They can use different amounts of it. Summed first-token time measures the user's initial wait on each independent request. The native opening rate on the preceding page measures only the inference engine's first turn. A shorter answer can reduce wall time even if the cost per emitted token rises. Each arm ran once in the frozen order on the same Mac, candidate first. The comparison cannot establish the distribution of timing differences across fresh processes, temperatures or machines.

#pagebreak()
= What latency bought on each request
#figure(image("file-evidence-latency.svg",width:100%),caption:[Both axes show actual whole-request seconds on logarithmic scales. Each point is one user request. A point below the diagonal finished sooner on the recurrent model. File success includes a correct real callback and a correct visible answer.])

A quick incorrect response costs the user a failed task. A slower correct response can be preferable. The symbols distinguish which system succeeded on each request. The first point in execution order for each model includes model construction, and the other points use its resident weights. The downloadable request table identifies every point by article and supplies both timings, both visible replies and the executed calls.

= How this compares with other looped models
Huginn trained a 3.5B depth-recurrent model on 800 billion tokens. Relaxed Recursive Transformers converted pretrained models by sharing layers and learning depth-specific corrections. Training-Free Looped Transformers measured the effect of reapplying frozen middle layers. In their Qwen3-4B-Instruct timing study, looping on every generated token increased wall time by 21.6% on 200 GSM8K questions. Their main multiple-choice results used likelihood scoring. Our changed-file experiment starts with a smaller hybrid Base, shares the same two trained layers on every pass, and scores whole free-running conversations after tool execution. These differences limit direct numerical comparison.

#link("https://arxiv.org/abs/2502.05171")[Geiping et al., 2025] · #link("https://arxiv.org/abs/2410.20672")[Bae et al., 2025] · #link("https://arxiv.org/html/2605.23872v1")[Training-Free Looped Transformers, Appendix I].

= What would justify a release?
A positive first pair would license independent evidence and a matched one-pass control trained with the same data and optimizer. The single candidate would then need to retain its tool, question-answering and ordinary-language behavior and complete full 128K requests that change the evidence. Distinct models in earlier chapters each establish narrower results. The locked score and cost comparison here remains specific to the listed weights, prompt policy and user tasks.

#pagebreak()
= Which file questions changed the answer?
#figure(image("file-evidence-articles.svg",width:100%),caption:[Each row is one article. Across the two versions of the same file, the human-annotated conclusion switches from yes to no. Both requests for an article within an arm shares the same prompt, system, tool declarations and input token IDs.])

The colors distinguish retrieving the right excerpt from reaching a correct conclusion about it. A response can be correct with no matching lookup, or it can return the right cell and still give the wrong answer. Both possibilities matter for a person deciding whether to rely on a claimed use of their file.

This study has one keyed-file function, one filename and two column names for the source judgments. All requests explicitly identify their target article. The observed file scores characterize that interface. The independent inline questions contain no tool declaration. All selected human claims predate these experiments, although similar source material could have appeared in Qwen's original pretraining.

#pagebreak()
= A full pair, from the file to the answer
#include "file-evidence-example.typ"

= Reproducing the evidence
The source builder and study registration record the publisher split hashes, training texts and both sets of model inputs. The field audit opens both CSV files and verifies that the selected cell is the only difference. Each measurement archive includes output token IDs for every turn, tool arguments, actual returns, completion status, visible text, elapsed time and full-attention lengths. The CSV accompanying this paper includes the article, claim, two source URLs, first selected function and arguments, visible answer and costs. Each row also links to its article's real historical diff.

Training compares a newly fitted two-pass system with ordinary untouched Base. The comparison combines recurrent computation, additional training and an eight-bit prefix. It does not isolate depth as a cause. The archived matched-depth experiments in later chapters answer that separate question for their original workloads. Long-context evidence is reported for the actual models that processed those inputs. The two 131,084-token installed replays used the qualified release 05, in a clean environment with network access blocked.
