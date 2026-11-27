#set document(title:"Does training the second visit improve reasoning after a tool call?",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.84in,y:.72in),footer:context [#set text(size:8pt,fill:rgb("#54645f"))
LoopQwen-MLX · Proof curriculum #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:11pt)
#set par(leading:.55em,spacing:.55em)
#show heading.where(level:1): h => block(above:8pt,below:3pt)[#set text(size:16pt,fill:rgb("#185c52"))
#h]
#let m=json("proof-curriculum-paper.json")
#let matrix(data) = table(columns:(2.2fr,1fr,1fr),align:(left,right,right),inset:(x:6pt,y:3pt),stroke:none,fill:(x,y)=>if y==0 {rgb("#dcece7")} else if calc.odd(y) {rgb("#f3f6f5")},[*Measurement*],[*Trained loop*],[*Untouched Base*],..data.flatten().map(v=>[#v]))
#text(size:25pt,weight:"bold")[Does training the second visit improve reasoning after a tool call?]
#linebreak()
#text(size:14pt)[Shared recurrent depth on human inference and real dependency metadata]
#linebreak()
November 2026 · LoopQwen-MLX project

= Abstract
A local assistant must retrieve the right data and interpret it. A previous recurrent model often executed the correct file call while losing accuracy when the same text was supplied inline. We independently trained a two-pass model on author-written explanations, machine-verifiable proofs and real package constraints. Each reasoning update explicitly rewards the second visit for raising the correct verdict’s odds over the first. #m.abstract

#figure(image("proof-curriculum-accuracy.svg",width:100%),caption:[Each grounded success requires the correct visible answer. File and compatibility items also require the model to execute the correct callback and receive the exact value. Both models see the same item order and initial token IDs.])

= Decision
#m.decision Every enrolled request contributes to the reported denominators. The source trial, previous independent models and full technical methods follow this supplement.

#pagebreak()
= Architecture and training
The pinned comparison is ordinary-depth Qwen3.5-0.8B-Base, revision #raw("dc7cdfe2"). The candidate starts from release 05’s trained action adapter over the post-trained Qwen3.5-0.8B instruction sibling, revision #raw("2fc06364"). Prefix layers use the existing eight-bit resident backbone. The model visits its last full attention and feed-forward layer twice on every input and generated token. Both visits share seven projection updates with rank 32 and retain separate physical attention histories. The second output is mixed with the first using residual coefficient 0.35. This direct comparison measures the entire system, including initialization, quantization, supervision and recurrence.

Each update assembles two labeled examples. The first is an original human e-SNLI explanation. The second is drawn from independently verified ProofWriter worlds with probability 0.75 and authentic wheel metadata with probability 0.25. Labels are balanced within the chosen source. Both items have an inline view and a separately executed file-return view. A correct opening tool call and a historical table response complete the six-example update.

For each reasoning view we calculate the verdict log odds of the correct leading token against the other two verdicts. The first-visit odds form a stop-gradient reference. The second visit incurs a hinge penalty when it improves these odds by less than 0.25 nats. This penalty accounts for 15% of the reasoning objective. The remaining 85% combines cross-entropy on explanation tokens, verdict tokens and the full reply in proportions 0.50, 0.35 and 0.15. The four reasoning views receive 65% of the update, the opening call 20% and table rehearsal 15%. Deployment generates over the full vocabulary. The solver and version evaluator operate only on training material or research labels.

#m.exposure

= Independent data qualification
We drew 192 distinct open-world depth-three training theories, balancing three verdicts and both query polarities. A signed Horn-rule interpreter had to reconstruct the published label. For decisive cases it emits derivations naming actual source premises. For unresolved cases it checks both polarities and identifies missing conditions when a related rule exists. A second implementation using Z3 Datalog reproduced all 192 verdicts. Of 128 decisive proofs, 118 require at least two hops. The corpus uses generated English. The human evaluation stories come from a different publisher and reasoning formalism.

Other training sources comprise 384 original human e-SNLI explanations and 70 version checks from ten actual wheels. The latter cover inclusive lower bounds and seven released interpreter versions. Their package names are disjoint from all 44 held inventory packages. Ten packages offer limited evidence about other forms of Python constraints. Each training and evaluation file lookup runs the same keyed CSV provider available to users.

#pagebreak()
= Dependence, uncertainty and evidence
Each of 12 held FOLIO stories contains three human conclusions with different verdicts. The assistant receives each conclusion once with inline evidence and once through a file. A whole-story success requires six correct judgments. The primary test applies all 4,096 possible sign flips to paired story totals. The gate requires at least 12 additional correct judgments, a two-sided probability at most 0.05, more whole stories and gains in each view, with no loss by answer category. #m.uncertainty

#figure(image("proof-curriculum-stories.svg",width:100%),caption:[Each point shows all six judgments for one original story. Asterisks identify English/formula differences recorded before scoring. The seven remaining stories form the preselected sensitivity set.])

#m.reasoning We independently enumerated every inclusion-minimal premise set for the 24 decisive claims with 1,288 solver queries before model scoring. A model can express a valid inference without numbering premises. Conversely, a list of sufficient premises can accompany a flawed derivation. We therefore also rated the substance of explanations.

#m.blind One item per original logical story and one per practical package pair were predetermined. The reader used the original English or exact package constraint and the model’s emitted final reasoning and visible answer. The full response-level annotations include the reason for each rating.

#matrix(m.ratings)

#pagebreak()
= Practical performance
#m.evidence #m.example

#figure(image("proof-curriculum-costs.svg",width:100%),caption:[Left, actual elapsed time accumulated in deployment order. Right, first-token wait for each of the 96 user requests, including cold initialization on the first item. All tool calls, returned observations and subsequent generation count toward request wall time.])

#matrix(m.cost)

The two arms share the reasoning-enabled chat template, greedy generation, 512-token allowance per model turn, three-turn maximum, actual files and answer scorer. Each operates in its own resident model state. Opening rate divides total initial tokens by total time to the first generated token. Subsequent decode rate excludes the first generated token of each turn and its measured startup time. Peak allocation is the largest measured native MLX allocation over the requests. The order was fixed, with the candidate run first and Base second. One timed stream per arm cannot determine variability across machines or cold launches.

= Scope of the finding
#m.context The replicated, matched-depth induction result also belongs to a separate K3 model. The requirements for a replacement remain a repeatable improvement on human reasoning, usable functions, resource costs and retained long-context behavior in the same released weights. The public evaluation material may also have appeared in the original pretraining corpora, whose complete contents we cannot inspect. Generated explanations provide evidence about behavior. They alone cannot identify the hidden computations responsible for a verdict.

#pagebreak()
= Relation to other work
Hsieh and colleagues use large-model explanations as additional supervision for small models. Our first 4B teacher missed the minimum number of usable uncertain explanations. The present fit uses original human or independently verified formal supervision. Tafjord, Dalvi and Clark study proof generation over controlled rule systems. Here their open-world training theories supply checkable derivations, while held human FOLIO stories and package requests test transfer. #link("https://aclanthology.org/2023.findings-acl.507/")[Distilling Step-by-Step] · #link("https://aclanthology.org/2021.findings-acl.317/")[ProofWriter] · #link("https://aclanthology.org/2024.emnlp-main.1229/")[FOLIO].

Geiping and colleagues pretrain a recurrent-depth model at 3.5 billion parameters on 800 billion tokens. Relaxed Recursive Transformers introduce loop-specific low-rank corrections. Our model begins from a much smaller pretrained instruction sibling, shares the projection updates across its two visits, and explicitly measures the cost of independent full attention histories. Its single late repeated layer and modest training budget limit transfer to deeper or differently pretrained systems. #link("https://papers.nips.cc/paper_files/paper/2025/hash/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Abstract-Conference.html")[Recurrent-depth pretraining] · #link("https://deepmind.google/research/publications/122290/")[Relaxed Recursive Transformers].

= Replication record
The study archive contains the registration, code revision and environment, qualified sources, training views, all saved optimizer checksums, final optimizer moments, adapter, fused layer, both inference streams and the decision. The file #raw("paper/proof-curriculum-requests.csv") contains each actual request, original observation, model answer, free-generated reasoning, executed calls and latency.

The registration preceded optimization and held scoring. A startup wiring error occurred before the first optimization or evaluation. Its original log and source manifest are included. The corrected startup separately built and checked its 2,066 training views. Independent verification reconstructs source derivations and training exposure, reads original weights, and rebuilds every callback, input and scored model turn. The companion reading protocol and English/formula review were frozen before evaluation.

From the repository root with the locked environment and pinned source files:
#raw("python -m scripts.proof_curriculum_material\npython scripts/research_archive.py proof-curriculum-01 --verify\npython -m scripts.file_evidence_checkpoints --proof-curriculum\npython -m scripts.proof_curriculum_exposure\npython -m scripts.proof_curriculum_reading report\npython -m scripts.paper_proof_curriculum",block:true)

The training archive records the full SHA-256 of the adapter. Its leading sixteen hexadecimal characters are #raw(m.training.slice(0,16)). The comparison uses the sealed Base and release checkpoint on Apple Silicon with native MLX inference, automatic differentiation and optimization. The earlier teacher study and previous technical report begin on the following pages.
