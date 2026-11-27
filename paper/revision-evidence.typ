#set document(title:"Did the answer change with the article?",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:0.82in,y:0.75in),footer:context [#set text(size:8pt,fill:rgb("#48625e"))
LoopQwen-MLX · Edited evidence #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:11pt)
#set par(leading:0.56em,spacing:0.58em)
#show heading.where(level:1): h => block(above:9pt,below:4pt)[#set text(size:16pt,fill:rgb("#125b51"))
#h]
#text(size:25pt,weight:"bold")[Did the answer change with the article?]
#linebreak()
#text(size:14pt)[A compact recurrent reader on actual Wikipedia revisions]
#v(4pt)
November 2026 · LoopQwen-MLX project

= Abstract
#include "revision-evidence-abstract.typ"

#figure(image("revision-evidence.svg",width:100%),caption:[The article pair is the primary unit. Each arm reads the two excerpts in separate, independently reset conversations. The direct-source subset was recorded before inspecting model answers.])

= What changed for the user
A reader checks a claim against a particular passage. On one historical sports article the source names one Super Bowl MVP in a revision and another in the adjacent version. On an album page, the certification for the same event changes from gold to platinum. A fluent unchanging reply can be correct once on every pair. Counting both answers together measures whether the reader actually follows the changed source.

Our preceding tool study provided a reason to ask this question. Its trained last-layer loop returned more correct individual answers and better lookup arguments, yet gave exactly the same visible response on 22 of 24 opposite-meaning sentence pairs. Here the starting point is a different, already trained six-layer semantic model with two earlier successful human entailment comparisons. We evaluate its transfer to changed evidence without fitting on VitaminC, using its original Base backbone.

#pagebreak()
= Model, comparison and costs
Every input and output passes through the same 24 native decoder layers. Layers 10–15 run a second time, giving 30 execution sites. Both applications share learned projections and the six recurrent sites keep independent histories. We merged the existing entailment adapter with the pinned Base text weights and applied native MLX 8-bit affine quantization with groups of 64 to every linear projection and embedding. The comparator uses the untouched BF16 Base at ordinary depth. These systems differ in training, recurrence and numerical precision. The previously measured, equally trained dense control is preserved in the repository.

#include "revision-evidence-conversion.typ"

#include "revision-evidence-costs.typ"

Each arm read the same ordered 48 requests with the same Qwen thinking template and exact input IDs. The greedy decoder remained unrestricted until the first turn boundary or a 768-token allowance. Each half began with a fresh model load. The candidate ran first in one half and Base first in the other. We sum measured wall time and first-token delays including the two loads for each arm. Native rates use total token counts divided by their respective measured phases. Archive writes are outside the user request timer. Allocation is the largest MLX-tracked allocation observed during the two serving streams. The archive also preserves all generated IDs and physical full-attention lengths.

#pagebreak()
= What the edited passages establish
The original VitaminC test file is pinned by revision and SHA-256. The sample contains one real edit per article with a constant claim and opposite human labels. Articles were selected by a frozen seeded hash, with no model output available. Each prompt supplies exactly one source excerpt. Selection excludes literal Unicode replacement characters. Original punctuation, occasional mojibake and awkward wording survive. A published benchmark can have appeared in pretraining. This is a held-out project transfer study and does not establish freedom from original pretraining exposure.

The Wikimedia API returned every registered revision ID with its immediate parent. We saved the actual changed lines and timestamps. For 23 of 24 articles, at least half of the distinct three-word sequences in each excerpt appeared in opposite changed lines from that edit. This mechanical text-overlap check tolerates neither every template expansion nor every citation markup. It measures provenance separately from the truth of an annotation.

A source-only reviewer examined all 24 contrasts before viewing model answers. Fifteen edits directly distinguish the labels under the article subject. Nine need qualification. For example, a typical interval fails to establish an absolute maximum, a percentage can be mistaken for a count, and replacing one example of an operatic role with another leaves the first role possible. The primary retains all 24 as registered.

#include "revision-evidence-interpretation.typ"

= Reproduction and limits
The frozen registration, 48 underlying question records, both generation archives, source-only readings and all 24 upstream diffs accompany this edition. The request CSV includes both answers, completion, actual time, exact historical diff links and the scope judgment for each article. The short-passage study does not qualify this exact model for tool use or 128K. The following field reports document actual tools and the installed full-context replays. An effect of depth requires an independently matched one-pass comparison.

#link("https://aclanthology.org/2021.naacl-main.52/")[Schuster, Fisch and Barzilay (NAACL 2021), Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence.] The authors' released examples incorporate Wikipedia material. The repository includes the publisher's license, article-level revision links and source selection.
