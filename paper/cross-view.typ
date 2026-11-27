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

A frozen four-billion-parameter MLX Community Qwen3.5 teacher reads SQuAD training passages and the human question. We extract its distribution at each token of the reference answer under teacher forcing. We store its 32 largest probabilities and their token IDs at temperature two. The remaining vocabulary is a single probability mass during distillation. The candidate learns that distribution in both the inline and real CSV-return settings. The primary reference remains the human answer. The teacher sees neither Quoref items nor the scored table requests. Its 3.03 GB checkpoint is required for repeating training. Deployed inference uses the smaller recurrent student alone.

#m.training

A logical update includes two different questions on the same training article, each in both views, their article’s native opening action for one question and one genuine WikiSQL rehearsal. Sixty percent of the objective weights the four answer examples equally. Their losses use answer-token cross entropy, 0.2 times whole-response cross entropy and 0.45 times squared-temperature grouped teacher cross entropy. Opening-action and rehearsal cross entropies receive 30% and 10%. This prevents long function syntax from determining the answer loss simply by contributing more tokens. The earlier file experiment used a token-weighted objective and separate margins. Its 13.2% judgment-token exposure motivated this change.

= Source and evaluation independence
The teaching articles are selected from SQuAD v1.1 train. The external evaluation uses the Quoref development release. Both datasets contain questions written by people about passages. Quoref was designed to test coreferential reasoning. We enrolled 20 different article titles and two distinct single-span questions per passage. Answers contain at most 12 words, each has an exact source offset and neither normalized answer contains the other. All 40 questions omit their own answer string. The highest five-word Jaccard overlap between any enrolled evaluation passage and any selected SQuAD passage is 0.0027. The use of publicly available Wikipedia material leaves possible upstream pretraining exposure.

#link("https://aclanthology.org/D19-1606/")[Dasigi et al., Quoref] · #link("https://aclanthology.org/D16-1264/")[Rajpurkar et al., SQuAD] · #link("https://aclanthology.org/2023.findings-acl.507/")[Hsieh et al., reasoning distillation] · #link("https://huggingface.co/mlx-community/Qwen3.5-4B-4bit")[Frozen training teacher].

#pagebreak()
= Human answers and actual cost
#include "cross-view-metrics.typ"

#m.sensitivity

Each case earns content credit if the completed visible answer contains its human reference as a bounded phrase and omits the other question’s different reference. Exact scoring uses SQuAD case, article, punctuation and whitespace normalization. Token F1 gives graded credit for partial overlap. A real file outcome also requires the matching executed call and exact returned passage. The archived primary fixes these choices for both models. A separate review examines complete nonexact replies with the model identity and publisher answer hidden.

The whole-stream timings include model construction on the first request, generation, real callbacks and final answers. The reported first-token sum includes user-visible setup. Decode throughput pools tokens after the first output over their actual decode seconds. Each arm ran once in a fixed order on the same 18 GiB Apple M3 Pro. Timing variability across hosts and repeated runs is outside this comparison.

#pagebreak()
= Where both modalities succeeded
#figure(image("cross-view-articles.svg",width:100%),caption:[Each row is a distinct Quoref article. The four columns for each model show its two inline and two file answers. A complete article has four correct outcomes with both file reads grounded.])

#m.example

The accompanying CSV includes every request, source, caller-visible reply, first call, exact and content scores, F1, peak memory and actual latency. The registration fixes thresholds and the maximum budget in advance. A direct comparison with untouched Base combines instruction post-training, recurrence, distillation and precision. An equally trained single-pass control is needed to attribute a quality gain to the loop itself.
