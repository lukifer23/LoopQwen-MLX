#set document(title:"What must a recurrent decoder materialize between prompt blocks?",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.85in,y:.74in),footer:context [#set text(size:8pt,fill:rgb("#506863"))
LoopQwen-MLX · Native runtime measurements #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:10.6pt)
#set par(leading:.46em,spacing:.45em)
#show heading.where(level:1): h => block(above:8pt,below:2pt)[#set text(size:15pt,fill:rgb("#205953"))
#h]
#let d=json("cache-state-runtime.json")
#let grid(head,rows,cols)=table(columns:cols,align:(x,y)=>if x==0 {left}else{right},inset:(x:5pt,y:3pt),stroke:none,fill:(x,y)=>if y==0 {rgb("#dcece6")}else if calc.odd(y){rgb("#f1f5f4")},..head.map(v=>text(weight:"bold",v)),..rows.flatten().map(v=>[#v]))
#text(size:24pt,weight:"bold")[What must a recurrent decoder materialize between prompt blocks?]
#linebreak()
#text(size:13pt)[Two same-weight experiments from native Metal inference to an installed tool conversation]
#linebreak()
November 2026 · LoopQwen-MLX project

= Result
The same two-visit Qwen checkpoint answered eight prompts with identical generated tokens, full-vocabulary next-token logits and physical cache states after intermediate prompt blocks. Evaluating the histories needed by future tokens lowered first-token time by 3.03%. A second experiment repeated real tool conversations through a wheel installed outside the checkout. Across four balanced crossover pairs, native turn opening fell by 2.79%. First-token time from process launch fell by 1.76%, total process duration by 0.75%, and peak native allocation by 15.72 MiB.

#image("cache-state-conversations.svg",width:100%)
#text(size:8.7pt)[Figure 1. Each point compares two consecutive blocks using the same questions and weights. Execution alternated AB, BA, AB, BA. All eight paired comparisons favored native cache-state evaluation. Repeated timings measure variation on one machine.]

= The computation
The frozen 0.8B instruction-sibling backbone applies its last native attention layer twice to every token, with distinct physical histories for the two visits. On each intermediate 128-token block, ordinary prefill asks MLX to evaluate the final hidden activation. The alternative runs the same hidden function, then materializes every physical history. A final hidden activation on an intermediate block is consumed only insofar as a later state depends on it. Both settings retain the complete final prompt block and take the unrestricted greedy argmax on the same vocabulary. There is no change to weights, input length, block boundaries, attention span or number of recurrent visits.

The first experiment used four file and four inline human-source prompts, 24 generated tokens per request, two settings, and four crossovers. An untimed replay compared every bit of the 248,320 final-prompt logits and all 25 site states on every prompt, including convolution, delta, full keys, values and offsets. The initial serializer failed on nested native arrays. The study retained every original timing and reran only the untimed numerical checks. A retained probe tensor inflated the final stock block’s allocator peak. The median block peak differed by approximately 16 MiB. [1]

#pagebreak()
= What a user measured through the installed command
The wheel was built twice from source revision #raw(d.wheel.source.slice(0,12)) and installed in a temporary location. All LoopQwen imports resolved there. The model cache was empty and network access was blocked. Four two-message conversations were repeated in eight process blocks, with two function formats and identical 256-token and three-turn limits. The first token was observed via the native progress callback. Cold times include process launch and model loading. Each setting represents 16 complete command invocations and 32 user messages.

#grid(("Measured quantity","Stock","State"),d.metrics,(2.7fr,1fr,1fr))

#grid(("Workload","Stock total (s)","Stock opening (s)","State total (s)","State opening (s)"),d.by_workload,(1.8fr,1fr,1.25fr,1fr,1.25fr))

The state implementation executed 120 intermediate blocks across the four workload types. Each setting generated 5,824 tokens and reused 28,672 input tokens. Every pair matched on prompt tokens, outputs, actual tool observations, response statuses, per-site attention history and number of prefix tokens reused. The installed setting improved all four total command times. The decode-rate difference of approximately 0.015 tokens/s is very small relative to the cost of loading and generation. [2]

= What these conversations actually achieved
Both settings gave the file’s correct byte size and line count. The schedule conversation used a real date lookup and calendar operation to give the correct weekday, then exhausted its turns on the follow-up. In the package conversation the model read NumPy 2.5.5, subsequently retrieved MLX 0.32.4, and repeated the previous answer of 2.5.5. Both compatibility questions on the release manifest exhausted their turn budgets. A faster response preserves those errors when all generated IDs are identical.

The experiments compare two execution settings of release 05 on one 18 GiB Apple M3 Pro. They measure a localized speed effect and its survival through process startup and multi-turn callback handling. The earlier 131,084-token qualification uses the released long-context path. The conversation crossover here used shorter prompts. Performance on the separately trained preference policy and the effect of recurrence on task quality require their own controls.

= Artifacts
[1] Fixed-prompt crossover, numerical fingerprints and timing blocks in #raw("research/results/cache-state-prefill-01"). \
[2] Installed crossover, complete conversations and build identity in #raw("research/results/cache-state-conversations-01"). The same paths are present in the research checkout. The standalone CSV gives every measured process, opening, duration, allocation and output length. The installed wheel SHA-256 is #raw(d.wheel.sha256).
