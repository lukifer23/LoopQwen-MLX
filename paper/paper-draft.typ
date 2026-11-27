#let d = json("data.json")
#let ink = rgb("#192b36")
#let teal = rgb("#08665c")
#let muted = rgb("#4c5c65")
#let pale = rgb("#eaf2ef")
#let gray = rgb("#d2ddda")
#set document(date:datetime(year:2026,month:11,day:27),title:d.title+" Controlled experiments with Qwen3.5-0.8B on Apple Silicon",author:d.authors,keywords:("recurrent language models","shared weights","MLX","tool use","long context"))
#set page(paper:"us-letter",margin:(top:0.84in,bottom:0.78in,x:0.91in),
  header:context if counter(page).get().first()>1 {set text(font:"Helvetica Neue",size:8pt,fill:muted); [LoopQwen-MLX #h(1fr) Shared depth on Apple Silicon #linebreak() #line(length:100%,stroke:0.45pt+gray)]},
  footer:context {set text(font:"Helvetica Neue",size:8pt,fill:muted); [Research draft · 27 November 2026 #h(1fr) #counter(page).display("1")]})
#set text(font:"Libertinus Serif",size:11.7pt,fill:ink,lang:"en")
#set par(justify:true,leading:0.68em,spacing:0.7em)
#set heading(numbering:"1.")
#show heading.where(level:1): it => block(above:9pt,below:7pt)[#text(font:"Helvetica Neue",size:19pt,weight:"bold",fill:ink)[#counter(heading).display() #it.body]]
#show heading.where(level:2): it => block(above:6pt,below:3pt)[#text(font:"Helvetica Neue",size:12pt,weight:"bold",fill:teal)[#it.body]]
#let small(body) = text(font:"Helvetica Neue",size:8.8pt,fill:muted,body)
#let cap(n,body)=block(above:3pt,below:5pt)[#small([#strong(n) #body])]
#let tab(cols,head,..cells)=table(columns:cols,align:left,inset:(x:5.5pt,y:5pt),stroke:none,
  fill:(x,y)=>if y==0 {pale} else if calc.even(y) {rgb("#f6f8f7")},..head.map(h=>text(font:"Helvetica Neue",size:9pt,weight:"bold",h)),..cells.pos().map(c=>text(size:10pt,c)))
#let label(body)=text(font:"Helvetica Neue",size:9.2pt,weight:"bold",fill:teal,body)
#let repo="https://github.com/lukifer23/LoopQwen-MLX"
#let chip(body)=box(fill:pale,inset:(x:8pt,y:6pt),radius:2pt,body)
#label[RESEARCH ARTICLE · WORKING DRAFT]
#v(5pt)
#block[#set par(justify:false)
#text(font:"Helvetica Neue",size:32pt,weight:"bold",tracking:-0.45pt)[When Does Shared Depth\ Earn Its Cost?]]
#text(size:17pt,fill:muted)[Controlled experiments with Qwen3.5-0.8B on Apple Silicon]
#v(6pt)
#text(font:"Helvetica Neue",size:12pt)[#link("https://github.com/lukifer23")[lukifer23] #h(12pt) GPT 6.2 Astra]
#linebreak()
#small[LoopQwen-MLX project · Evidence cut at the first matched development checkpoint]
#v(9pt)
#label[Abstract]
A language model can retrieve the correct evidence and still make the wrong decision. We test whether repeating a pretrained block helps a small model use that evidence, while controlling for training and measuring actual time. Earlier controlled rule-induction experiments provide a positive case. A three-visit reader answered 62 of 64 held stories correctly, versus 33 for a trained one-pass reader at nearly equal elapsed time. The advantage persisted on fresh stories. On public executable tools, a recurrent model completed 17 of 22 tasks and its trained one-pass control completed 18. We use this difference to motivate a joint test of chat, grounded reasoning, live tools and complete documents. Both new models start from the same Base checkpoint and adapt the same six MLP projections. At update 32, each gave 5 supported answers on 27 short requests at a 128-token output cap. The loop gained one changed-evidence answer and lost another on the same source. Both initial models processed 167,014 input tokens on a complete novel and exhausted their short output budget before delivering an answer. The general-assistant and replicated depth claims remain open.

= Introduction
A comparison with an untouched model measures the benefit a user might receive from the entire training recipe. It leaves open whether the benefit came from another application of the same weights. We therefore train an ordinary-depth sibling alongside the loop. Both see the same questions, source edits, targets and tool histories. We credit a model only when its visible answer is correct, follows the user's request and supports its claims. When a user requests an action, we also require a corresponding successful execution.

The distinction matters on a small machine. Another pass adds four physical layer visits at each generated token. A long document adds attention history for each repeated attention layer. A design that solves more counterfactual questions at the same token cap can still solve fewer by the time its user needs an answer. We report the two outcomes separately.

#pagebreak()
= Where shared depth has already helped
The project began with synthetic rule induction, where a small change to an observed relation can change the correct conclusion. A matched one-pass reader received the same stories, number of updates and trainable parameter budget. On the original and fresh cohorts together, the three-pass model answered 124/128 correctly and the one-pass model answered 58/128. Both produced 13 tokens per answer and took about 32 seconds for each complete 128-question workload. A second fit seed yielded 53/64 versus 25/64. These experiments supply a narrow case in which additional tied computation changed the final decision.

#image("history.svg",width:100%)
#cap([Figure 1.],[Historical comparisons with independently trained one-pass readers. Each row uses its own registered cohort. The four story rows test different aspects of one task. The public-tools row uses a separate model and workload. Lines connect the two arms within each experiment.])

The public-tool comparison changes the interpretation. A previous recurrent assistant completed 17/22 tasks against zero for untouched Base. An independently trained ordinary-depth model completed 18/22. In another experiment, a four-layer loop fetched both complete tables on 23 evidence pairs. On 22 of those pairs it gave the answer “four” in both worlds. Only one of the 48 reference answers was four. Access to the relevant table left the arithmetic or selection decision unresolved.

A later source-copy system finished 130/144 presentations against 105/144 for Base. Across its two observed domains, source review credited fully supported answers on 24.3% of presentations for the loop and 45.8% for Base. This measure includes the quality of the final response. Completion alone would have favored the opposite system. The three experiments used different checkpoints, workloads and scoring rules. We retain their individual results instead of pooling them into a model ranking.

== Relation to recent looped models
Huang et al. [1] separately varied recurrence placement, input injection and recurrent state design during large-scale training. Marchenko et al. [2] studied intermediate exits and conversion of a pretrained dense language model. Our experiment asks a narrower deployment question. With a small adaptation budget, can repeated transformations inside an existing hybrid decoder pay for themselves in freely generated, supported answers?

#pagebreak()
= Model and matched training
We begin with the pinned Qwen3.5-0.8B Base text decoder [3]. Before selecting a donor, both untouched checkpoints answered eight frozen, thinking-enabled prompts at a 256-token cap. A masked review credited Base on 3/8 and the instruction sibling on 0/8. The registered rule selected Base for both trained arms. Eight capped prompts are weak evidence about the donors' overall assistant quality. The untouched instruction sibling remains a comparator at longer output allowances.

#align(center)[#chip[Pretrained prefix · 12 layers] #h(3pt) → #h(3pt) #chip[Tied span · 4 layers] #h(3pt) → #h(3pt) #chip[Suffix · 8 layers]]
#cap([Figure 2.],[The shared span occupies zero-based layers 12–15, including the global-attention layer at 15. Each physical visit owns a separate temporal state. All exits use the ordinary suffix, final normalization and vocabulary projection.])

Let $a$ be the prefix output and $B$ the four-layer middle block. The first pass computes $h_1 = B(a)$. A subsequent pass computes
$ h_(k+1) = B((1-alpha) h_k + alpha a). $
We initialize $alpha$ to 0.8 and fit it jointly with the loop adapters. The ordinary-depth model executes the original 24 layers once. The two-pass serving model executes 28 layer visits, and the three-pass mechanism probe executes 32. Each model adapts rank-16 output projections at layers 12, 13, 14, 15, 19 and 23. The ordinary model has 442,368 trainable coefficients. The loop adds one scalar. Its initialized one-pass exit reproduced the donor logits exactly.

== Supervision and exposure
The frozen short curriculum has 288 turns and 17,133 target positions per epoch. At batch size two, four epochs correspond to 576 optimizer updates per arm. AdamW warms up over 24 updates to a learning rate of 0.0003. The loop supervises its first two exits for 32 updates, followed by all three. Each exit predicts the complete supported answer. Where a verified foil exists, a margin term penalizes the first divergent wrong token. The one-pass model receives the same row order, targets, foil comparisons and suffix adaptation.

#tab((1fr,3.9fr,.55fr),([Family],[Sources and supervision],[Turns]),
  [Reasoning],[48 qualitative polarity cases and 24 grounded comparisons. Source edits can reverse the correct decision.],[72],
  [Tools],[Opening calls, delivery, repair, recovery, reuse and redirection. Includes financial arithmetic and database actions.],[72],
  [Chat],[32 conversation turns, 32 rewrites and eight edited replies. Ordinary unrestricted answers are part of the objective.],[72],
  [Documents],[40 complete-table turns and 32 schedule turns. Whole-book training has a separate native gradient gate.],[72])
#cap([Table 1.],[The source-balanced short curriculum is shared by both arms. No teacher generated the targets. One optimizer update costs more for the loop because it supervises multiple physical exits.])

#pagebreak()
= Evaluation and early results
Each short family contains three development sources with original, relevant-change and irrelevant-change presentations. The prompt IDs, tools, source order, decoding policy and cap are frozen across arms. Model identity is masked during review of completed responses. Three real tool turns are permitted. An unfinished answer earns zero. Own-history follow-ups retain the model's generated actions and actual observations. A separate shared-history condition starts both models from the same prior exchange.

#tab((1.3fr,1.15fr,1.15fr,1.6fr),([Family],[Two-pass loop],[Ordinary depth],[Seconds, loop / ordinary]),
  ..("reasoning","tools","chat").map(f=>{
    let l=d.early.find(r=>r.family==f and r.arm=="loop")
    let o=d.early.find(r=>r.family==f and r.arm=="ordinary")
    (f,str(l.success)+"/"+str(l.episodes),str(o.success)+"/"+str(o.episodes),str(calc.round(l.seconds,digits:2))+" / "+str(calc.round(o.seconds,digits:2)))
  }).flatten())
#cap([Table 2.],[Update 32, 128 generated tokens per turn. Each arm answered 5/27 correctly. Total request time was #d.early_times.loop seconds for the loop and #d.early_times.ordinary seconds for ordinary depth.])

#image("time.svg",width:100%)
#cap([Figure 3.],[Left, total time to serve all nine requests in each family, including failed requests. Right, number of correct final answers completed by each elapsed deadline. The time axis starts when the request is received. This interim panel has nine independent sources and three variants per source.])

== One source explains the observed gain and loss
The bridge guide closes a crossing when water covers a blue marker. In one version, the observer reports that both markers are covered. The loop inferred closure. Its ordinary sibling asked for the markers' relative heights, which were unnecessary. An irrelevant change elsewhere left the observation as “water covers the red marker.” The loop then said the observed marker was unspecified. The ordinary model answered correctly. Each model retained the same total reasoning score. The one gain and one loss were two presentations of the same source.

On 16 training-only financial roll-ins from the early models, neither model initiated a callback. One response nevertheless gave the correct numerical change. Review retained 15 failed trajectories for a registered correction and physically executed 16 common correct or crossed calculations. Actual-history and canonical-prefix fits receive the same 3,198 supervised target tokens. A separate comparison will measure whether training on those failures restores successful tool use.

#pagebreak()
= What the long-context run establishes
The measurement unit is the actual tokenized prompt, including the complete document, question, chat template and tool declarations. The two versions of _Anne of Green Gables_ each occupy 145,891 input tokens. The published and edited _McTeague_ prompts occupy 167,014 and 167,017. The Anne task requires combining two age statements and expressing their difference. The McTeague task compares a bird's ability to escape with its owner's predicament. An edited ending changes the required final decision.

#tab((1.5fr,.7fr,.8fr,.85fr,.8fr),([Model],[KV sites],[First token, s],[Request, s],[MLX GiB]),
  ..d.books.map(r=>(r.name,str(r.sites),str(r.first),str(r.total),str(r.peak))).flatten())
#cap([Table 3.],[Initial, unadapted models on the same 167,014-token published book. Both generated all 128 allowed tokens and produced no final answer. Base used a restartable staged prefix and the initial loop used direct staged opening. Times include opening the document, and the two runtime paths differ.])

The ordinary decoder retained six global attention histories and the loop retained seven. Each global site's retained length equaled its physical offset through decoding. Linear-attention sites retained their convolution state and delta state separately at each visit. A completed group can be persisted with a digest and reopened without aliasing a later pass. A causal branch test changed future tokens while holding the earlier prefix fixed and observed identical earlier logits.

== Exact gradients and cache cost
To train on complete documents, the proposed full-length stage freezes the trunk and updates the final pointwise MLP adapter. An initial sparse implementation reproduced the serving logits, yet its adapter gradient differed from the full native training graph. On Metal, the attention forward used while differentiating can differ numerically from the serving forward. Capturing the residual from the training forward made the complete-reference and sparse gradients match exactly on a 64-position native chunk with 20 supervised positions. The real long-document optimizer effect remains to be measured.

We also tested whether a branch can load each exact KV prefix once into its own resident buffer. On a 162,688-token prefix, 15 further decode steps took 3.64 seconds with staged reads and 0.68 seconds with resident state. Peak MLX allocation rose from 2.46 to 3.77 GiB. The two modes produced equal full-vocabulary logits at all 16 checked positions and replayed the same public calculator trajectory. This comparison measures runtime on one prefix. The original short-cap book run still counts as a failed answer.

Input length, output headroom and elapsed deadline answer different questions. The 128-token condition tests short output. The terminal protocol also measures 512 tokens, and the registered headroom diagnostic allows 4,096 per assistant turn with normal end-of-turn stopping. The book deadline is 600 seconds. A longer cap can reveal a correct late answer that misses that deadline.

#pagebreak()
= Interpretation, limitations and release criteria
The historical induction studies establish that shared depth can improve a decision when the task has a narrow, learnable structure. In the current development comparison, the trained models tied at the short cap and the loop took longer. The bridge pair identifies a specific computation worth studying, but its two oppositely signed effects come from one source. Neither an accurate pointer nor a completed callback earns credit for a wrong final answer. The current results do not establish an improved general assistant.

The main uncertainties are source coverage, output truncation, single-reviewer judgment and training maturity. There are only three short development sources per family and the early read uses one seed. The long opening result covers one book and one cap. Broader document-family qualification, full-length training, longer-output scoring and independent confirmation are outstanding at this evidence cut. The 18 GiB machine also imposes a real stopping constraint. The older receipt-repair fit stopped on OS memory pressure before its first optimizer update, so it provides no estimate of training effectiveness.

The registered strong product target is a ten-point macro gain over untouched Base, a positive gain in each family and a better by-deadline supported-answer rate. The minimum is five points with a positive paired aggregate 95% lower bound, nonnegative family estimates and simultaneous one-sided family bounds above −5 points. Reading qualification targets at least 50% supported in each actual-input band. A causal depth claim additionally requires improvement over the separately trained ordinary model at matched time, followed by fresh-source confirmation and a positive second seed. All thresholds precede the associated scored generations.

== Reproducibility and cost
One exclusive worker runs on an 18 GiB Apple M3 Pro with a #str(d.hardware.gpu_cores)‑core GPU (macOS #d.hardware.macos). Inference, differentiation and optimization use MLX 0.32.4 with MLX-LM 0.32.2. Closed charges through the resident-cache verification total #calc.round(d.costs.profile.seconds+d.costs.fit.seconds+d.costs.short.seconds+d.costs.long.seconds,digits:1) worker seconds, including stopped attempts. Profiling and regression checks used #d.costs.profile.seconds seconds, fitting and training roll-ins #d.costs.fit.seconds, short evaluation #d.costs.short.seconds, and full-document development #d.costs.long.seconds. The initial development ceiling is 43,200 seconds. The ledger preserves later work separately.

Cold optimizer replay reproduced every saved-gradient update tensor on both models. Recomputing the backward pass from scratch changed the next weights or moments by up to $3.55 times 10^(-5)$. The two policy adapters share 1,527,751,054 verified donor payload bytes. All 36 selected native cache, source-copy, session and release CLI regression checks passed. The repository contains the source cohorts, masked reviews, per-request trajectories, stage registrations and checksummed snapshot used to build this draft. #link(repo)[Project repository.]

== References and contributions
#small[[1] Huang, B. et al. (2026). _Towards Looped Models Done Right, Part I_. #link("https://arxiv.org/abs/2610.23036")[arXiv:2610.23036]. #linebreak()
[2] Marchenko, A. et al. (2026). _Closing the Loop: Practical Training Recipes for Looped Language Models_. #link("https://arxiv.org/abs/2610.00673")[arXiv:2610.00673v2]. #linebreak()
[3] Qwen Team. _Qwen3.5-0.8B-Base_ and _Qwen3.5-0.8B_ instruction checkpoints. #link("https://huggingface.co/Qwen/Qwen3.5-0.8B-Base")[Base model card]. #linebreak()
#strong[Contributions.] lukifer23 directed the research questions, hardware constraints and acceptance criteria. GPT 6.2 Astra implemented experiments, operated the native runs, reviewed trajectories and prepared this manuscript. The byline identifies the AI contributor explicitly.]
