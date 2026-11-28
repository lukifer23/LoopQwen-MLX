#let d = json("data.json")
#let ink = rgb("#192b36")
#let teal = rgb("#08665c")
#let muted = rgb("#4c5c65")
#let pale = rgb("#eaf2ef")
#let gray = rgb("#d2ddda")
#set document(date:datetime(year:2026,month:11,day:27,hour:12,minute:0,second:0),title:d.title+" Controlled experiments with Qwen3.5-0.8B on Apple Silicon",author:d.authors,keywords:("recurrent language models","shared weights","MLX","tool use","long context"))
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
#label[RESEARCH ARTICLE · WORKING DRAFT 5]
#v(5pt)
#block[#set par(justify:false)
#text(font:"Helvetica Neue",size:32pt,weight:"bold",tracking:-0.45pt)[When Does Shared Depth\ Earn Its Cost?]]
#text(size:17pt,fill:muted)[Controlled experiments with Qwen3.5-0.8B on Apple Silicon]
#v(6pt)
#text(font:"Helvetica Neue",size:12pt)[#link("https://github.com/lukifer23")[lukifer23] #h(12pt) GPT 6.2 Astra]
#linebreak()
#small[LoopQwen-MLX project · Matched development results through update 576]
#v(9pt)
#block(below:3pt)[#label[Abstract]]
A language model can retrieve the correct evidence and still make the wrong decision. We test whether repeating a pretrained block helps a small model use that evidence, while controlling for training and measuring actual time. Earlier controlled rule-induction experiments provide a positive case. A three-visit reader answered 62 of 64 held stories correctly, versus 33 for a trained one-pass reader at nearly equal elapsed time. The advantage persisted on fresh stories. On public executable tools, a recurrent model completed 17 of 22 tasks and its trained one-pass control completed 18. We use this difference to motivate a joint test of chat, grounded reasoning, live tools and complete documents. Both new models start from the same Base checkpoint and adapt the same six MLP projections. After four epochs the loop gave 5 supported answers on 27 short development requests, the trained one-pass model 4, untouched Base 2 and the instruction sibling 6, at 512 generated tokens per turn. The loop cost 3.78 times as much optimizer time as its trained control. All three requests on which those controls differed were edits that should leave the decision unchanged. A token audit revealed a training mismatch. Most supervised turns closed the thinking segment immediately, and both fitted models did so on every development opening. On two 145,891-token editions of a novel, the loop recovered both stated ages, including a changed fraction, and omitted the difference both times. Neither trained model supported its conclusion on either ending of a second novel above 167,000 input tokens. Across four equally weighted families, the loop gained 2.1 percentage points over Base at 512 output tokens (paired 95% interval −9.7 to +16.0). Neither standalone model completed either of two project jobs with live tools.

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
Universal Transformers [4] established a recurrent alternative to a stack of independent layers. Geiping et al. [5] studied scaling internal computation in a model pretrained for that purpose. Huang et al. [1] separately varied recurrence placement, input injection and recurrent state design during large-scale training. Marchenko et al. [2] studied intermediate exits and conversion of a pretrained dense language model. Thinking Machines Lab reports that Inkling 2 Mini repeats its middle 12 layers three times, with a learned message from the previous pass. Its variable-depth research model gains most on multihop tasks and shows little change on closed-book recall [7]. Our experiment asks a narrower deployment question. With a small adaptation budget, can repeated transformations inside an existing hybrid decoder pay for themselves in freely generated, supported answers?

#pagebreak()
= Model and matched training
A donor is the checkpoint whose pretrained weights initialize a fitted model. We begin with the pinned Qwen3.5-0.8B Base text decoder [3]. Before selecting a donor, both untouched checkpoints answered eight frozen, thinking-enabled prompts at a 256-token cap. A masked review credited Base on 3/8 and the instruction sibling on 0/8. The registered rule selected Base for both trained arms. Eight capped prompts are weak evidence about the donors' overall assistant quality. The untouched instruction sibling remains a comparator at longer output allowances.

#image("architecture.svg",width:100%)
#cap([Figure 2.],[The shared span occupies zero-based layers 12–15, including the global-attention layer at 15. Each physical visit owns a separate temporal state. All exits use the ordinary suffix, final normalization and vocabulary projection.])

Let $a$ be the prefix output and $B$ the four-layer middle block. The first pass computes $h_1 = B(a)$. A subsequent pass computes
$ h_(k+1) = B((1-alpha) h_k + alpha a). $
The prefix supplies a fixed copy of the input to the block, called the anchor. At initialization the second pass receives 80% anchor and 20% first-pass output. We fit the mixing coefficient jointly with the adapters. Three of the four repeated layers use linear attention with recurrent state. The fourth can attend directly to earlier tokens. Eight ordinary layers after the block can learn how to turn its output into a decision. The ordinary-depth model executes the original 24 layers once. The two-pass serving model executes 28 layer visits, and the three-pass mechanism probe executes 32. Both models keep the donor fixed and fit small rank-16 updates to MLP output projections at layers 12, 13, 14, 15, 19 and 23. The ordinary model has 442,368 trainable coefficients. The loop adds one scalar. Its initialized one-pass exit reproduced the donor logits exactly.

== Supervision and exposure
The frozen short curriculum has 288 turns and 17,133 target positions per epoch. At batch size two, four epochs correspond to 576 optimizer updates per arm. AdamW warms up over 24 updates to a learning rate of 0.0003. The loop supervises its first two exits for 32 updates, followed by all three. Each exit predicts the complete supported answer. Where a verified foil exists, a margin term penalizes the first divergent wrong token. The one-pass model receives the same row order, targets, foil comparisons and suffix adaptation.

One epoch presents 288 turns in 144 updates. Both arms processed 140,168 input positions including the preceding target tokens. The ordinary model spent 148.80 seconds in optimizer updates and the loop spent 470.40. Counting middle and suffix layers once per input position per forward execution gives 1,682,016 layer-token visits for the ordinary model and 5,931,384 for the loop. This count excludes prefix extraction and backpropagation. Training supervises several exits, so its cost ratio differs from the cost of serving a single exit.

#pagebreak()
= Training data and source changes
We fix the source material, order and targets before fitting either model. Each short epoch contains 123,323 prompt positions and 17,133 target positions. The objective rewards the entire reference response, including the explanation when requested. For rows with a verified wrong alternative, an additional term favors the correct token at the first point where the two responses diverge. Unrestricted chat responses use the ordinary vocabulary without an imposed answer format.

#tab((.86fr,3.4fr,.65fr),([Family],[Origin and construction],[Groups]),
[Reasoning],[48 QuaRTz examples vary the direction of a qualitative relationship. Another 24 use HotpotQA comparisons grounded in passages.],[48],
[Tools],[TAT-QA supplies financial operands and receipts. CoSQL supplies nonnumeric database requests. Six groups of 12 turns cover opening, delivery, repair, recovery, reuse and redirection.],[67],
[Chat],[32 OpenAssistant conversations, 32 Dolly summarization or rewriting turns and eight project-authored turns with edited instructions or facts.],[68],
[Documents],[40 complete FeTaQA table questions and 32 turns comparing two FOSDEM 2023 sessions. Official calendar and XML records corroborate the event times.],[56])
#cap([Table 1.],[Each family has 72 training turns. The last column counts distinct source groups within the family. An edited turn can share a source with its original. The inventory links source revisions, attribution, licenses and derived token files. #link(repo+"/tree/main/research/sources/middle-depth")[Source catalogue.]])
== Worked training pair
A printed note and a newer packing slip disagree about the units in a crate. In the first version, the current slip says 24 *pairs* per crate, so three crates contain $3 times 24 times 2 = 144$ bottles. In its partner the current slip says 24 *loose bottles* and the answer is 72. Both prompts contain the same two quantities and both units. The model must resolve which document has authority and which unit it assigns. Each version uses the other answer as its incorrect alternative. The stored prompts have 53 tokens apiece, with 39 and 30 supervised positions respectively.

== Separation from development
The audit found 288 distinct training openings, zero exact shared openings with short development and zero complete held questions in the training text. The 27 development presentations have 25 distinct opening hashes because some edits change what the live tool returns. The full-document curriculum separately enrolls four inputs from _The House of the Seven Gables_ and the FOSDEM 2025 schedule. In the schedule pair an edited end time changes the second session from 10 to 60 minutes and reverses which talk is longer. Neither development novel is a full-document fit source. These checks do not establish independence from the donor's pretraining corpus.

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

A source is the unit of comparison. A relevant edit should produce a different correct decision. An irrelevant edit should preserve the original correct decision. The three variants of a source share evidence and cannot be counted as three independent observations. Own-history follow-ups measure what happens after each model's mistakes. Shared-history follow-ups isolate what the model does given the same established facts. Both remain separate from the 27 opening questions in this table.

#pagebreak()
= What a longer answer budget changes
The 128-token condition limits what each assistant turn can generate. It has no bearing on how much input the model reads. A short cap can hide an answer when a reasoning-enabled model spends most of its tokens thinking. We registered a longer diagnostic before evaluating the 512-token terminal conditions. Four models received the same three opening requests with up to 4,096 generated tokens per assistant turn. Two additional presentations followed up the same invoice using either the model's own conversation or a common verified history.

#image("headroom.svg",width:100%)
#cap([Figure 4.],[A held invoice asks for 11 cartons of 18 tiles less 14 chipped tiles, using a calculator. Elapsed time includes all turns and real provider calls. The vertical line marks the registered eight-second short-request deadline. Triangles mark exhausted turns.])
== A correct number and a successful action
Untouched Base eventually wrote 184 after two calculator errors. It had supplied equations with equality signs that the real calculator rejected. The visible number was correct, but the requested action never succeeded. The instruction sibling submitted two executable expressions, received 198 and 184, and delivered the supported answer in 9.34 seconds. This earns full-budget credit and misses the eight-second deadline. The trained ordinary model exhausted 4,096 tokens in 69.78 seconds. The trained loop exhausted them in 76.18 seconds.

When nine more tiles were removed, the instruction sibling delivered 175 in 3.77 seconds on its own history and 4.12 seconds on the shared history. The trained models exhausted their allowance on their own histories. Given the shared starting value of 184, both returned nine without a callback. Providing the verified starting point removed one obstacle while leaving selection of the requested operation unresolved.

== How far the result travels
All four models failed the archive reasoning question and the welcoming rewrite. The instruction sibling therefore earned 1/3 opening requests across these three sources, against 0/3 for the others. The invoice follow-ups reuse its source and do not supply two more independent wins. The short donor screen selected Base under its registered cap. The longer diagnostic exposes one executable ability that the screen missed. The supplemental review used masked answer records, but the project reviewer had already seen some labeled output snippets. It lacks independent adjudication.

#pagebreak()
= What the long-context run establishes
Input counts include the complete text, question, chat template and tool declarations. Both editions of _Anne of Green Gables_ occupy 145,891 tokens. The question asks for the difference between Anne’s early and late statements of her age. A single edit changes the correct difference from 66 to 69 months. A separate question asks for a clock minute the text never gives. _McTeague_ occupies 167,014 tokens in the published edition and 167,017 after the edit. The original ending leaves the bird caged and McTeague handcuffed to the dead Marcus. In the edited ending the bird escapes while the man remains trapped.

#tab((1.5fr,.7fr,.8fr,.85fr,.8fr),([Model],[KV sites],[First token, s],[Request, s],[MLX GiB]),
  ..d.books.map(r=>(r.name,str(r.sites),str(r.first),str(r.total),str(r.peak))).flatten())
#cap([Table 3.],[Initial, unadapted models on the same 167,014-token published book. Both generated all 128 allowed tokens and produced no final answer. Base used a restartable staged prefix and the initial loop used direct staged opening. Times include opening the document, and the two runtime paths differ.])

Each global attention site retained its full physical offset. Every linear-attention visit kept separate convolution and delta states. Changing future tokens left earlier logits unchanged in a causal branch test.

#tab((1.5fr,.76fr,.87fr,.77fr,.8fr,.85fr),([Model],[Tokens],[Prefix, s],[Warm, s],[Cold, s],[Peak GiB]),
  ..d.anne_pair.map(r=>(r.model,str(r.generated),str(r.build),str(r.warm),str(r.seconds),str(r.peak))).flatten())
#cap([Published Anne opening at 128 output tokens.],[Cold time includes building the complete shared prefix from the book. Warm time includes the rest of the opening and generation. Each arm reproduced staged full-vocabulary logits at two positions before using its resident cache. Base reached its 128-token output limit. The loop finished in 27 tokens. Prefix files used #str(d.anne_pair.at(0).scratch) GiB for Base and #str(d.anne_pair.last().scratch) GiB for the loop. All three decoders scored 0/2 on the published and edited questions at this cap.])

== Exact gradients and cache cost
To train on complete documents, the proposed full-length stage freezes the trunk and updates the final pointwise MLP adapter. The initial sparse implementation reproduced the serving logits, but its relative gradient error was 6.58% on a cached prefix and 7.46% when differentiating through prefill. On Metal, the attention forward used while differentiating can differ numerically from the serving forward. Capturing the residual from the training forward made the complete-reference and sparse gradients match exactly on a 64-position native chunk with 20 supervised positions. The real long-document optimizer effect remains to be measured.

We also tested whether a branch can load each exact KV prefix once into its own resident buffer. On a 162,688-token prefix, 15 further decode steps took 3.64 seconds with staged reads and 0.68 seconds with resident state. Peak MLX allocation rose from 2.46 to 3.77 GiB. The two modes produced equal full-vocabulary logits at all 16 checked positions and replayed the same public calculator trajectory. This comparison measures runtime on one prefix. The original short-cap book run still counts as a failed answer.

Input length, output headroom and elapsed deadline answer different questions. The 128-token condition tests short output. The terminal protocol also measures 512 tokens, and the registered headroom diagnostic allows 4,096 per assistant turn with normal end-of-turn stopping. The book deadline is 600 seconds. A longer cap can reveal a correct late answer that misses that deadline.

#pagebreak()
= What each additional visit bought during training
At each update, we compared successive exits on the same training turn and weights. Each exit predicts the answer through the ordinary suffix. The objective averages loss over target tokens and includes the registered incorrect-answer penalty where available. We weighted sources equally within families.

#image("exits.svg",width:100%)
#cap[Figure 5.][Negative values favor the deeper exit. Second visits cover all 288 turns. Third visits cover the 224 turns after warmup.]

The second visit lowered the objective for 33 of 48 reasoning sources and 39 of 67 tool sources. It raised the mean objective on chat and document turns. A separate comparison with the trained ordinary sibling gave the second exit a mean advantage of 0.063 on reasoning and 0.065 on tools. The third exit increased the mean objective in each family on its eligible turns. Free generation at these same weights scored 4/27 for each model with either 128 or 512 tokens per turn. At 128 both lost two bridge answers and gained one correct studio message relative to update 32. On the changed bridge question the loop replaced its explanation of the closure with “yes.”

== Execution improved before tool use did
Both models invoked tools on all nine tool openings at each cap. At 512 the loop received a successful provider response on six and ordinary depth on three. Both scored 0/9. For the ink question both computed $16 times 35 - 750$ and reported −190 mL. The required remainder was 190 mL. On the original tile invoice, the loop called the calculator three times with the same program and ended without an answer. The ordinary model repeatedly sent expressions with equals signs that the provider rejected.

== The cost of reaching a fair comparison
A 1,024-position financial training example repeatedly triggered the Mac’s memory-pressure guard. Recomputing the backward pass one native layer at a time preserved the scalar loss in all #d.recovery.checks checks on two real examples. The largest relative gradient difference was #calc.round(100*d.recovery.relative,digits:2)%. Loading the frozen prefix weights on demand allowed the previously interrupted update to finish with a #calc.round(d.recovery.peak_gib,digits:2) GiB peak. Both arms use the same differentiation method from update 145. Replaying the loop from its saved first epoch preserves that comparison and adds a charge to the ledger.

Both models saved the 139,520-token prefix shared by two editions of _Anne of Green Gables_, with 28 distinct layer histories for the loop and 24 for ordinary depth. Deduplication saved 868,067,701 bytes. At that first-epoch checkpoint, only the ordinary original-edition response finished. The terminal paired reading on page 7 used the same full inputs.

#pagebreak()
= The completed fit and its precision sensitivity
Both models completed 576 optimizer updates on 1,152 presentations and 560,672 input positions per arm. The loop used 2,918.32 seconds of optimizer time and ordinary depth used 772.58. At both tested output caps, the loop scored 5/27 and ordinary depth 4/27. At 512 tokens their decisions differed on three irrelevant edits. The loop gained the archive shipment and the parcel pickup question, and lost the rehearsal message. There was no gain over ordinary depth on an original or answer-changing presentation. The registered changed/compositional score difference at the eight-second deadline was +3.7 points against a five-point target.

#let arms = (("terminal","base","Untouched Base"),("terminal","instruction","Instruction sibling"),("terminal","ordinary","Trained ordinary"),("terminal","loop","Trained loop"),("quantized","ordinary","Q4 ordinary"),("quantized","loop","Q4 loop"))
#tab((1.5fr,1fr,1fr,1fr,1fr),([Decoder],[Reasoning],[Tools],[Chat],[Total]),
  ..arms.map(t=>{
    let rows=d.terminal_results.at(t.at(0)).filter(r=>r.cap==512 and r.arm==t.at(1))
    let values=("reasoning","tools","chat").map(f=>rows.find(r=>r.family==f).success)
    (t.at(2),..values.map(x=>str(x)+"/9"),str(values.sum())+"/27")
  }).flatten())
#cap([Terminal development panel.],[Nine independent short sources, each shown in three worlds. Correct tool answers require actual supported execution. Four follow-ups per arm are archived separately. The four-bit rows reuse the same fitted adapters.])

== Where the fitted models first diverge
Every enrolled prompt began with reasoning enabled. Yet 264/288 target sequences closed the thinking segment without any intervening prose, including all 72 tool targets. Sixty tool rows already included a provider response. Only 12 trained the first action from a fresh user request. The terminal loop and ordinary sibling generated empty thinking segments on 27/27 openings each. Untouched Base did so on 0/27. On tools its median first thinking segment was 148 token positions. We measured this change in generated behavior. The contribution of supervision for immediate closure to final-answer quality requires its own controlled test.

The instruction sibling executed and answered all six arithmetic openings correctly. Neither trained arm obtained the requested value on any of nine tool openings. Replaying #d.callback_replays physical provider observations across the four BF16 arms and two Q4 arms reproduced every observation. These cases locate an error before the final prose grade, in deciding which tool and operands to send.

== Deployment on the M3 Pro
Group-64 affine quantization reduced the shared donor file from 1,504,827,865 to 424,021,044 bytes. The median time for 32 identical cached decode positions fell from 0.570 to 0.246 seconds for the loop and from 0.510 to 0.215 for ordinary depth. Both completed a disposable adapter update with finite gradients. The Q4 loop gained a rehearsal rewrite and lost its archive decision. Successful provider responses fell from six to zero tool openings for the loop and five to zero for ordinary depth. We retained BF16 as the reference. All attention histories kept their original precision.

#pagebreak()
= Failures, pivots and remaining tests
The historical induction studies establish that shared depth can improve a decision when the task has a narrow, learnable structure. The terminal loop gained one net short answer over its trained sibling at a substantially higher training cost. The bridge pair identifies a specific computation worth studying, but its two oppositely signed effects come from one source. Neither an accurate pointer nor a completed callback earns credit for a wrong final answer. The current results do not establish an improved general assistant.

== Why we changed the experiment
*Evidence retrieval.* In the earlier Deno study, 640 updates improved source-location likelihood by 4.78 nats. On held changed pairs the fitted reader and its parent each succeeded on 18/24, versus 20/24 for a separately trained single-read model. Together with the repeated “four” on changed tables, this motivated teaching the final decision at every exit and checking the visible answer early.

*Recovery after a bad tool result.* The prior streamed financial fit resumed from its archived pre-update boundary. It stopped on OS memory pressure after another 29.82 seconds, before any optimizer step. We closed its dependent pilot. This run establishes a resource limit and cannot estimate the benefit of receipt training. In the new experiment, neither early model called a tool on 16 training-only roll-ins. One response gave the correct number. We retained the 15 failed histories and executed 16 common correct or crossed calculations. Each 32-update continuation received 3,198 supervised target tokens. At 128 generated tokens, actual-history training yielded #str(d.correction.scores.corrected.values().sum())/27 supported loop answers, the canonical-prefix control #str(d.correction.scores.simple.values().sum())/27 and the original parent #str(d.correction.scores.terminal.values().sum())/27. All three scored 0/9 on tools. The registered rule retained the original weights.

*Complete documents.* On two complete editions of _Anne of Green Gables_, the fitted loop retrieved age 11 and the correct later age, 16½ in the published book and 16¾ after the edit. It stopped before giving the requested difference at both output caps. Trained ordinary depth used the unsupported ages 13 and 19 in both editions. With 512 tokens Base supported the published answer of 5½ years within 285.6 seconds cold. It gave two years on the edit. Thus Base earned #str(d.anne_scores.at("512").base.values().filter(x=>x).len())/2 and both trained models 0/2 on this one development source. The full-context fit tests whether supervising supported final answers can change this outcome. Its selection rule requires a book gain with no decline in either band or short-task macro accuracy.

*What happened when we put the evidence in reach.* We retained both relevant passages from each development novel, supplying 1.3–1.7K or 13.6–18.5K actual input tokens and 512 output tokens. Across the two books, two lengths and two editions, the early loop and ordinary model each scored 0/8. Base scored 1/8 on the original medium-length Anne excerpt. One response quoted both correct ages, subtracted them correctly, then asserted that Anne was eleven at birth. Another quoted both ages and dropped the fraction when subtracting. These errors distinguish grounding the operands from calculating and delivering a faithful final response.

== What would justify a stronger claim
The registered strong product target is a ten-point macro gain over untouched Base, a positive gain in each family and a better by-deadline supported-answer rate. The minimum is five points with a positive paired aggregate 95% lower bound, nonnegative family estimates and simultaneous one-sided family bounds above −5 points. Reading qualification targets at least 50% supported in each actual-input band. A causal depth claim additionally requires improvement over the separately trained ordinary model at matched time, followed by fresh-source confirmation and a positive second seed. All thresholds precede the associated scored generations.

#pagebreak()
= Does the loop help on a deployment task?
We evaluated the same BF16 weights on two jobs that arise when operating this project. The first asks which optimizer checkpoint can be resumed and how many updates a rollback would lose. The second asks how much extra space a model transfer needs and whether both copies contain identical backbone bytes. Each job includes a follow-up on the model’s own history. File search and read-only SQLite exposed the actual sources. A separate direct execution verified that both providers could return the necessary evidence. Source snapshots, generated text and tool histories stay local.

#let w=d.local_workflows.arms
#tab((2.8fr,1fr,1fr),([Local operator screen],[Base],[Trained loop]),
[Complete two-turn jobs],[#str(w.base.complete_local_jobs)/2],[#str(w.loop.complete_local_jobs)/2],
[Correct requests within 30 seconds],[#str(w.base.local_correct_by_30_seconds)/4],[#str(w.loop.local_correct_by_30_seconds)/4],
[Executed and replayed callbacks],[#w.base.replayed_callback_results],[#w.loop.replayed_callback_results],
[Total time across four requests],[#calc.round(w.base.local_elapsed,digits:1) s],[#calc.round(w.loop.local_elapsed,digits:1) s],
[Mean time to correct, charging failures 30 s],[#w.base.local_mean_capped_time_to_correct s],[#w.loop.local_mean_capped_time_to_correct s])
#cap([Table.],[Four fresh sequential processes alternated the starting model across two blocks. The two standalone models received identical opening token IDs, tools and output allowances. A complete job requires both answers and the requested execution.])

Base never queried a provider. The loop obtained the inventory schema once. Six searches or queries returned no records, and one calculator call returned an error. It interpreted empty search results as evidence that the restart pointers were absent. A query using a word present in both files retrieved both pointers. On four unmodified questions from the public GSM8K test split, the loop answered #str(w.loop.gsm8k_correct)/4 and Base #str(w.base.gsm8k_correct)/4. These four items permit individual answer checks, with little precision for a benchmark-wide comparison. #link("https://github.com/openai/grade-school-math")[GSM8K source.]

== Four equally weighted families
#let pc=d.parent_comparisons
#let pp(x)=str(calc.round(100*x,digits:1))
#tab((2fr,1.2fr,1.1fr,1.7fr),([Comparison],[Output cap],[Gain (pp)],[Paired 95% interval]),
..for (cap,arm,name) in (("128","product","Loop vs Base"),("512","product","Loop vs Base"),("512","matched_depth","Loop vs trained one-pass")) {
  let r=pc.at(cap).at(arm)
  ([#name],[#cap],[#pp(r.macro_delta)],[#pp(r.paired_interval.first()) to #pp(r.paired_interval.last())])
})
At 512 output tokens the loop gained one reasoning answer and two net chat answers over Base. Tools yielded zero correct answers on either model. Base alone solved the published Anne question. On _McTeague_ the loop gave the same negative answer after the ending was edited to release the bird. Neither trained model supported a correct answer in either length band. General quality, useful tool specialization and a causal gain from shared depth each require independent evidence.

#pagebreak()
= Reproducibility, costs and references
One exclusive worker runs on an 18 GiB Apple M3 Pro with a #str(d.hardware.gpu_cores)‑core GPU (macOS #d.hardware.macos). Inference, differentiation and optimization use MLX 0.32.4 with MLX-LM 0.32.2. Closed charges through both complete development novels and the local deployment screen, including the standalone models, matched correction and Q4 evaluation, total #calc.round(d.costs.profile.seconds+d.costs.fit.seconds+d.costs.short.seconds+d.costs.long.seconds,digits:1) worker seconds, including stopped attempts. Profiling and regression checks used #d.costs.profile.seconds seconds, fitting and training roll-ins #d.costs.fit.seconds, short evaluation #d.costs.short.seconds, and full-document development #d.costs.long.seconds. The initial development ceiling is 43,200 seconds. Attempts that stopped account for #calc.round((d.costs.profile.stopped_seconds+d.costs.fit.stopped_seconds+d.costs.short.stopped_seconds+d.costs.long.stopped_seconds)/60,digits:1) worker-minutes. This includes time spent producing checkpoints and prefix states that later runs reused. The one-epoch optimizer costs on page 3 are a separate measurement window. Adding them to the ledger would double-count some training.

Cold optimizer replay reproduced every saved-gradient update tensor on both models. Recomputing the backward pass from scratch changed the next weights or moments by up to $3.55 times 10^(-5)$. Storing one verified donor for both early adapters avoids a second 1,527,751,054-byte backbone copy. All 36 selected native cache, source-copy, session and release CLI regression checks passed. Two offline installations then reproduced all 20 previously graded trajectories. Base exited before the loop loaded. Cold process times were #calc.round(d.standalone.arms.base.process_seconds,digits:1) s for Base and #calc.round(d.standalone.arms.loop.process_seconds,digits:1) s for the loop. Each scored 0/10 on this fixed replay subset. #link(repo+"/blob/main/docs/STANDALONE_MIDDLE.md")[Weights and reproduction instructions.]

== How to reproduce the figures
The PDF build reads a frozen JSON snapshot. It generates the vector figures and typesets the paper from that snapshot. Run `.venv/bin/python -m scripts.paper_draft` at the repository root. The input manifest identifies the underlying charges, trajectories and grade records by digest. The version stored with each paper is the appropriate basis for reproducing it while training continues. #link(repo+"/tree/main/paper")[Manuscript and snapshot.] #link(repo+"/tree/main/docs/measurements/middle-depth")[Native evidence.]

== Limits of this draft
The development set has nine short sources, the headroom test reuses three, and the terminal book comparison covers only two independent novels, each in two editions at two output caps. One reviewer graded the generated answers. The matched quality comparisons use one seed at updates 32, 144 and 576. Independent held confirmation, a second fit and a comparison against an untied extra-computation control are prerequisites for attributing a general improvement to weight sharing.

== References and contributions
#small[[1] Huang, B. et al. (2026). _Towards Looped Models Done Right, Part I_. #link("https://arxiv.org/abs/2610.23036")[arXiv:2610.23036]. #linebreak()
[2] Marchenko, A. et al. (2026). _Closing the Loop: Practical Training Recipes for Looped Language Models_. #link("https://arxiv.org/abs/2610.00673")[arXiv:2610.00673v2]. #linebreak()
[3] Qwen Team. _Qwen3.5-0.8B-Base_ and _Qwen3.5-0.8B_ instruction checkpoints. #link("https://huggingface.co/Qwen/Qwen3.5-0.8B-Base")[Base] · #link("https://huggingface.co/Qwen/Qwen3.5-0.8B")[Instruction]. #linebreak()
[4] Dehghani, M. et al. (2018). _Universal Transformers_. #link("https://arxiv.org/abs/1807.03819")[arXiv:1807.03819]. #linebreak()
[5] Geiping, J. et al. (2025). _Scaling up Test-Time Compute with Latent Reasoning_. #link("https://arxiv.org/abs/2502.05171")[arXiv:2502.05171]. #linebreak()
[6] Liu, N. et al. (2023). _Lost in the Middle: How Language Models Use Long Contexts_. #link("https://arxiv.org/abs/2307.03172")[arXiv:2307.03172]. #linebreak()
[7] Thinking Machines Lab (2026). _Introducing Inkling 2 Mini_. #link("https://thinkingmachines.ai/news/inkling-2-mini/")[Architecture and variable-depth experiments]. #linebreak()
#strong[Contributions.] lukifer23 directed the research questions, hardware constraints and acceptance criteria. GPT 6.2 Astra implemented experiments, operated the native runs, reviewed trajectories and prepared this manuscript. The byline identifies the AI contributor explicitly.]
