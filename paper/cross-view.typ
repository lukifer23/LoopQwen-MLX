#set document(title:"Can a recurrent assistant read its own tool returns?",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:0.82in,y:0.73in),footer:context [#set text(size:8pt,fill:rgb("#48625e"))
LoopQwen-MLX · Cross-view reading #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:11pt)
#set par(leading:0.55em,spacing:0.59em)
#show heading.where(level:1): h => block(above:8pt,below:3pt)[#set text(size:16pt,fill:rgb("#125b51"))
#h]
#let m = json("cross-view-paper.json")
#text(size:26pt,weight:"bold")[Can a recurrent assistant read its own tool returns?]
#linebreak()
#text(size:14pt)[A direct test on human questions with the same passage delivered two ways]
#linebreak()
November 2026 · LoopQwen-MLX project

= Abstract
The preceding trial exposed a gap between obtaining a passage and interpreting it. Here the same model answers each original human question twice in independent sessions. One user provides the passage inline. The other supplies a file with the same passage and two distractors. Both users request the same short answer. Every article has a second question with a different human answer. A complete article requires four successful replies. #m.summary

#figure(image("cross-view-channels.svg",width:100%),caption:[Each bar describes the same 40 human questions. A successful file reply requires the correct executed lookup, actual returned passage and correct visible answer. The file and inline halves remain paired within each question.])

= Project decision
#m.decision The earlier 131K system, induction model and edited-passage readers each retain their own measured qualification. The following 83 pages preserve their results, matched controls, dated registrations, failures and independent installation evidence.

#pagebreak()
= Training one decoder for both settings
The starting point is the released recurrent instruction-sibling action adapter. Its final native attention layer and feed-forward network execute twice on every token. The two physical visits share the same weights, retain separate attention histories and mix the repeated result with residual coefficient 0.35. Training updates seven rank-32 projection adapters. Serving uses the same resident decoder for opening calls, inline reading and post-observation answers. The only decoding rule is unrestricted greedy generation with the registered reasoning template. Both arms receive the same maximum of 640 tokens per turn and three turns per request.

A frozen four-billion-parameter MLX Community Qwen3.5 teacher reads SQuAD training passages and the human question. We extract its distribution at each token of the reference answer under teacher forcing. We store its 32 largest probabilities and their token IDs at temperature two. The remaining vocabulary is a single probability mass during distillation. The candidate learns that distribution in both the inline and real CSV-return settings. The primary reference remains the human answer. The teacher sees neither Quoref items nor the scored table requests. Its 3.03 GB checkpoint is required to regenerate the teacher targets. The archived distributions also permit offline repetition of student training. Deployed inference uses the smaller recurrent student alone.

#m.training

A logical update includes two different questions on the same training article, each in both views, their article’s native opening action for one question and one genuine WikiSQL rehearsal. Sixty percent of the objective weights the four answer examples equally. Their losses use answer-token cross entropy, 0.2 times whole-response cross entropy and 0.45 times squared-temperature grouped teacher cross entropy. Opening-action and rehearsal cross entropies receive 30% and 10%. This prevents long function syntax from determining the answer loss simply by contributing more tokens. The earlier file experiment used a token-weighted objective and separate margins. Its 13.2% judgment-token exposure motivated this change.

= Source and evaluation independence
The teaching articles are selected from SQuAD v1.1 train. The external evaluation uses the Quoref development release. Both datasets contain questions written by people about passages. Quoref was designed to test coreferential reasoning. We enrolled 20 different article titles and two distinct single-span questions per passage. Answers contain at most 12 words, each has an exact source offset and neither normalized answer contains the other. All 40 questions omit their own answer string. The highest five-word Jaccard overlap between any enrolled evaluation passage and any selected SQuAD passage is 0.0027. The use of publicly available Wikipedia material leaves possible upstream pretraining exposure. Wu et al. found exploitable biases in Quoref. Correct answers here demonstrate success on the enrolled reading questions. Establishing which referential steps were necessary would take an intervention on those steps.

#link("https://aclanthology.org/D19-1606/")[Dasigi et al., Quoref] · #link("https://aclanthology.org/D16-1264/")[Rajpurkar et al., SQuAD] · #link("https://aclanthology.org/2021.acl-long.448/")[Wu et al., coreference artifacts] · #link("https://arxiv.org/abs/1503.02531")[Hinton et al., temperature distillation] · #link("https://aclanthology.org/2023.findings-acl.507/")[Hsieh et al., rationale supervision] · #link("https://huggingface.co/mlx-community/Qwen3.5-4B-4bit")[Frozen training teacher].

#pagebreak()
= Human answers and actual cost
#include "cross-view-metrics.typ"

#m.sensitivity

Each case earns content credit if the completed visible answer contains its human reference as a bounded phrase and omits the other question’s different reference. Exact scoring uses SQuAD case, article, punctuation and whitespace normalization. Token F1 gives graded credit for partial overlap. A real file outcome also requires the matching executed call and exact returned passage. The eight real WikiSQL table requests measure retention against Base. Full regression qualification also requires replay of the previously successful release tasks on the same candidate. The archived primary fixes these choices for both models. A separate review examines complete nonexact replies with the model identity and publisher answer hidden.

The whole-stream timings include model construction on the first request, generation, real callbacks and final answers. The reported first-token sum includes user-visible setup. Opening throughput uses the two arms’ identical opening-token total divided by that sum, so the two measures carry the same directional information. Decode throughput pools tokens after the first output over their actual decode seconds. Each arm ran once in a fixed order on the same 18 GiB Apple M3 Pro. Timing variability across hosts and repeated runs is outside this comparison.

#figure(image("cross-view-service.svg",width:100%),caption:[Left, grounded or correct answers delivered over one cold stream in frozen request order. Right, elapsed time for all 88 individual requests, including failures. Both systems received the same generation cap.])

#pagebreak()
= Where both modalities succeeded
#figure(image("cross-view-articles.svg",width:100%),caption:[Each row is a distinct Quoref article. The four columns for each model show its two inline and two file answers. A complete article has four correct outcomes with both file reads grounded.])

#m.example

The accompanying CSV includes every request, source, caller-visible reply, first call, exact and content scores, F1, peak memory and actual latency. The registration fixes thresholds and the maximum budget in advance. A direct comparison with untouched Base combines instruction post-training, recurrence, distillation and precision. An equally trained single-pass control is needed to attribute a quality gain to the loop itself.

The historical system-comparison CSV had reported counts of correct first answers as seconds for the 16 module follow-up dialogues. The archived timers across their 32 user requests sum to 2.768 s for the loop and 2.528 s for Base. The corrected CSV uses these values. The original whole-dialogue outcome was 12/16 versus 0/16 in 13.97 versus 115.77 s.

#pagebreak()
= How much varies between articles?
Each article contributes two questions in two settings to both models. Counting those eight replies as independent would understate uncertainty. For this descriptive analysis we resampled 20 whole articles with replacement 20,000 times, preserving all questions, settings and arms within an article. The table gives the difference on the original sample and percentile bounds. The eight table-retention requests have a different sampling unit and are excluded. The primary requires at least five net additional complete articles and two-sided exact paired p ≤ 0.05. With zero discordant losses, six wins are needed to meet both conditions. A sparse 20-article sample limits the precision of secondary intervals.

#include "cross-view-intervals.typ"

= Can the answer change with the question?
The two human questions on each passage have different reference answers. We compared freely generated replies after case, punctuation, article and whitespace normalization. #m.repetitions Different replies can still both be wrong. This count describes answer behavior among completed pairs.

= Where the generation budget went
Both systems saw the official thinking-enabled template. The student's supervised answer turns contain the short human answer and an empty reasoning field. #m.generation The token count measures generated work and leaves its validity to the answer and execution checks.

= Physical verification after a tool return
#m.intervention The probe uses a separate project file and issues no scored answers. It establishes that the repeated layer influences a real post-observation prediction. The all-article comparison measures whether the complete system answers users successfully.

= Reproducing the measurements
The update supplies the panel, original source archives, per-token teacher distributions, fitted adapters, compiled layer, per-request traces and an installable runtime. Kit 01 supplies the original smaller checkpoints and locked native dependencies. The reconstruction script verifies tokenization, human answer offsets, article disjointness, teacher ranks, scheduled optimizer exposure, model hashes, physical attention histories, every real callback and the article-level decision. The published CSV and SVG plots are regenerated from both complete traces. The field guide includes the offline commands for an arbitrary caller-supplied CSV.
