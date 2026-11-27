#set document(title:"Qualifying explanation supervision before training a recurrent decoder",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.85in,y:.74in),footer:context [#set text(size:8pt,fill:rgb("#54645f"))
LoopQwen-MLX · Explanation qualification #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:11pt)
#set par(leading:.55em,spacing:.58em)
#show heading.where(level:1): h => block(above:7pt,below:3pt)[#set text(size:16pt,fill:rgb("#185c52"))
#h]
#let m=json("reasoned-evidence-paper.json")
#text(size:25pt,weight:"bold")[Can a small teacher supply the missing reasoning?]
#linebreak()
#text(size:14pt)[A preregistered qualification on human-written claims]
#linebreak()
November 2026 · LoopQwen-MLX project

= Abstract
The preceding two-pass reader returned an empty reasoning field on all 139 generated turns. We investigated whether an independent four-billion-parameter model could supply concise explanations grounded in human premises. Each input contained one original training claim and its numbered premises. The teacher received no gold verdict. #m.headline The accepted set contained #m.counts. The study ended before student optimization or held evaluation.

#figure(image("reasoned-evidence-qualification.svg",width:100%),caption:[All 192 training claims. The upper dashed line marks the 16-example minimum per class on the horizontal scale. Each reply belongs to one disposition. The lower graph preserves truncated generations at the 176-token ceiling.])

= Where qualification failed
The teacher generated #m.resource. #m.terminal Presenting accuracy only among replies with terminal verdicts would omit nearly half of the attempts. An accepted explanation also needed 5–100 words, a correct terminal verdict and one or more in-range premise references. These requirements check formatting and the answer label. They cannot establish that a premise actually supports each inference.

#pagebreak()
= Source, selection and measurement
We sampled 192 distinct stories from the FOLIO training release, one claim per story and 64 per published verdict. The selection excluded overlapping normalized premises and test claims. The training plan also reserved 384 author-written explanations from e-SNLI, balanced across the same three labels. The native four-bit MLX Qwen3.5-4B teacher generated greedily in batches of four. Each response had up to 176 tokens. We archived every input token, output token, decoded string and parsed result. The 58 accepted rows remain research observations. No student consumed these rows.

The table below shows mutually exclusive counts. Truncation has priority, followed by a missing final verdict, explanation length, citation validity and verdict correctness. This ordering means that a truncated reply can also contain an incomplete explanation. The 95 truncations were concentrated among claims labeled false. Under the registered criteria uncertainty was the least well-covered category.

#table(columns:(2.1fr,1fr,1fr,1fr,1fr),align:(left,right,right,right,right),stroke:.4pt,
  [*Teacher disposition*],[*True*],[*False*],[*Uncertain*],[*All*],
  [Accepted],[30],[17],[11],[58],
  [Reached token limit],[26],[43],[26],[95],
  [Outside explanation length],[6],[3],[16],[25],
  [Wrong verdict after other checks],[1],[1],[6],[8],
  [Missing or invalid citation],[1],[0],[5],[6],
  [*Enrolled*],[*64*],[*64*],[*64*],[*192*])

= What the held study would measure
The frozen comparison contains 12 held human stories with three different verdicts each. Each claim appears inline and through an executed CSV lookup. Sixteen practical requests ask whether specific Python versions satisfy authentic package metadata. Eight requests test earlier WikiSQL capabilities. Both arms would receive identical registered token IDs, decoding and time measurement. The teacher gate closed all 96 requests for this study. An unrun comparison contributes no accuracy or efficiency estimate.

An independent satisfiability check applied each published first-order premise set to the claim and its negation. #m.oracle English sentences can differ from their formal annotations in quantifier scope, word sense and polarity. The primary panel retains all original cases, with the seven unflagged stories identified in advance for a secondary analysis.

= Replication and sources
The per-generation CSV supplies source evidence, claim, human label, teacher output, token count and disposition. The companion audit reconstitutes the original teacher prompt and decodes the archived tokens before checking the acceptance rules. Reproduce from the repository with #raw("python -m scripts.reasoned_evidence_teacher_audit") and #raw("python -m scripts.paper_reasoned_evidence"). Links to original publications and source licenses accompany the machine-readable study. #link("https://aclanthology.org/2024.emnlp-main.1229/")[Han et al., FOLIO] · #link("https://github.com/OanaMariaCamburu/e-SNLI")[Camburu et al., e-SNLI] · #link("https://huggingface.co/Qwen/Qwen3.5-4B")[Qwen teacher].
