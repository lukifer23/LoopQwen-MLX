#set document(title: "When the referent changes", author: "LoopQwen-MLX project")
#set page(paper: "us-letter",margin:(x:0.82in,y:0.72in),footer:context [#set text(size:8pt,fill:rgb("#48625e"))
LoopQwen-MLX · Field study supplement #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:11pt)
#set par(leading:0.58em,spacing:0.66em)
#show heading.where(level:1): h => block(above:10pt,below:5pt)[#set text(size:16pt,fill:rgb("#125b51"))
#h]
#let costs=csv("instruction-interpolation-costs.csv")
#let quality=csv("instruction-interpolation-accuracy.csv")
#let tbl(cells, cols) = table(columns:cols,align:(left,center,center),stroke:none,inset:4pt,fill:(x,y)=>if y==0 {rgb("#dfede9")} else if calc.odd(y) {rgb("#f4f6f5")},..cells.flatten())
#text(size:26pt,weight:"bold")[When the referent changes]
#linebreak()
#text(size:14pt)[A continuously recurrent Qwen decoder on human pronoun pairs and live file lookups]
#v(4pt)
November 2026 · Supplement to _Where shared decoder depth earns its cost_

= Abstract
#include "instruction-interpolation-abstract.typ"

#figure(image("instruction-interpolation.svg",width:100%),caption:[The denominators distinguish individual requests from pairs. File success requires an actual observation and a correct final answer. A difference also requires a successful calculation.])

= Why pair these requests?
Choosing the wrong entity in a sentence and putting a filename into the wrong function argument can both produce a fluent but incorrect answer. The test asks the model to distinguish two plausible entities and use evidence obtained from a file. An individual Winograd sentence can be answered correctly by choosing a name independent of context. Scoring both members credits a model only if its decision follows the change in meaning. Completing both file requests requires carrying a previously observed value into a new calculation.

#pagebreak()
= Model and comparison
The recurrent system uses the instruction-tuned Qwen3.5-0.8B sibling in release 05. Its first 23 layers use the qualified eight-bit prefix. Each of the seven attention and feed-forward projections in the final layer receives half of its previously trained rank-32 update. We compute the low-rank product in FP32, add it to the original projection and round once to the donor dtype. On every token the system evaluates the same resulting layer twice. The second application contributes 0.175 of its proposed residual change. Each attention execution maintains its own key/value state. No parameters were fitted to this panel.

Weight interpolation has been studied as a way to preserve capabilities of an original model during adaptation. Our intervention also changes the strength of the recurrent residual. The comparator is the certified ordinary-depth BF16 Qwen3.5-0.8B-Base from the same local export. Instruction pretraining, quantization, the learned update and depth all differ across arms. This comparison answers which system solved the requests. A causal claim about depth requires a matched trained one-pass model. #link("https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html")[Wortsman et al., CVPR 2022.]

= Materials and scoring
We used the original Winograd collection, grouped adjacent sentences with identical choices, source and pronoun, opposite publisher labels, and at least 0.80 sequence similarity. A fixed hash selected 20 disjoint pairs. Option order was hashed within each pair, and all 40 questions ran as independent conversations in a mixed order. Source quotes and answer options stayed intact. The deterministic scorer considers a direct option, an explicit conclusion, or a single unambiguous option in prose. The archive preserves the full visible answer. Public availability means pretraining exposure cannot be excluded. #link("https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html")[Collection of Winograd Schemas.]

Eight separately selected package distributions had no overlap with the preceding package panel. For each we materialized every wheel entry from the pinned project lockfile in an actual CSV with filename, size and digest columns. Two hash-selected wheels with distinct sizes supplied the requests. Both models received the same exact-key lookup and arithmetic tool. We reopened the file for each call, verified source identity and required the lookup key to identify one row. The final numerical answer, the exact key/column binding, the value returned by the tool and the calculator result are independently visible. The derivative check changes each input operand in the reference and emitted expression to test whether they depend on the same values.

Both systems used the publisher's thinking-enabled chat template, greedy decoding, up to 640 tokens per model turn and four turns per user request. The first prompts matched token for token. Follow-ups included each model's own prior messages. We measured one cold stream per arm on an 18 GiB M3 Pro using MLX Metal. First-token time includes setup for the first request. Wall time includes callbacks and subsequent decoding. Reported peak covers MLX allocations. Opening throughput divides opening prompt tokens by summed first-token waits. Decode throughput excludes each turn's first token and its latency. The recurrent stream ran first. Host load, thermal and order effects limit performance generalization.

#pagebreak()
= Frozen decision and observed costs
The protocol required at least three net extra completely solved meaning pairs with a two-sided exact paired test at 0.05, more solved inventory pairs and no loss on operand-sensitive differences. It required no family loss on correct answers or completions, a significant pooled gain, strictly faster time to first token and end-to-end time, higher opening and decode rates and lower peak allocation. Registration set a 1,500-second native ceiling and 7 GiB peak limit before generation. Inference used 28 paired units: one per meaning pair or physical inventory.
#tbl(costs,(2.8fr,1fr,1fr))

#include "instruction-interpolation-findings.typ"

= Replication and evidence
The accompanying CSV contains all 56 paired user requests with original questions, labels, both visible responses, exact actions and observations, independent statuses, opening and output token counts, timings and peak allocations. The two compressed trajectories also retain every prompt and generated token and physical attention history, including failed calls. The source manifest identifies the original XML and lockfile byte for byte. The public dataset and underlying pretrained models preclude a clean-room novelty claim.

The add-on installs on the original field kit. The two example commands use the first frozen meaning and file sessions in panel order and compare both systems locally with network access disabled during verification. The original 64-page report covers the separate matched depth studies, retained 131K generation, complete document retrieval, language and service regressions. Each result belongs to its measured model and context length.
