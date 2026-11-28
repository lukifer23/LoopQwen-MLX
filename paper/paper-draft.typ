#let d = json("data.json")
#let ink = rgb("#192b36")
#let teal = rgb("#08665c")
#let muted = rgb("#4c5c65")
#let pale = rgb("#eaf2ef")
#let gray = rgb("#d2ddda")
#set document(date:datetime(year:2026,month:11,day:27,hour:12,minute:0,second:0),title:d.title+" on an 18 GiB Mac",author:d.authors,keywords:("recurrent language models","shared weights","MLX","tool use","long context"))
#set page(paper:"us-letter",margin:(top:0.84in,bottom:0.78in,x:0.91in),
  header:context if counter(page).get().first()>1 {set text(font:"Helvetica Neue",size:8pt,fill:muted); [LoopQwen-MLX #h(1fr) LoopQwen versus Qwen Base #linebreak() #line(length:100%,stroke:0.45pt+gray)]},
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
#let pp(x)=str(calc.round(x*100,digits:1))
#let sec(x)=str(calc.round(x,digits:1))
#let arms=("base","loop")
#let name(a)=if a=="loop" {"LoopQwen"} else {"Untouched Base"}
#let families=("reasoning","tools","chat","documents")
#let pretty(f)=(reasoning:"Grounded reasoning",tools:"Executed tools",chat:"Chat and instructions",documents:"Complete documents").at(f)
#label[RESEARCH ARTICLE · WORKING DRAFT 10]
#v(6pt)
#block[#set par(justify:false)
#text(font:"Helvetica Neue",size:32pt,weight:"bold",tracking:-0.45pt)[LoopQwen versus Qwen Base]]
#text(size:16pt,fill:muted)[Recall, executable tools and time to a correct answer on an 18 GiB Mac]
#v(6pt)
#text(font:"Helvetica Neue",size:12pt)[#link("https://github.com/lukifer23")[lukifer23] #h(12pt) GPT 6.2 Astra]
#linebreak()
#small[Native Apple Silicon MLX · Update 576 and full-document follow-on]
#v(9pt)
#label[Abstract]

We ask whether a small looped language model can retain the overall quality of untouched Qwen3.5-0.8B-Base and improve the work a local user can finish. LoopQwen repeats four pretrained layers and learns from complete answers, changed evidence and tool histories. At a 512-token output allowance it delivered 5 correct supported answers on 27 short requests, compared with 2 for Base. On four complete-book questions spanning 145,891 to 167,017 input tokens, LoopQwen delivered zero supported answers and Base delivered one. With four equally weighted task families the difference was +2.08 percentage points (paired 95% interval −9.72 to +15.97). The interval includes losses larger than the two-point deployment margin adopted after these parent results. The loop reproduced both requested ages in two versions of one novel, including an edited fraction, but omitted the requested subtraction. It made eight callbacks in two project-local jobs and finished neither. With weights fixed, one, two and three passes solved 10, 5 and 3 of the same 27 short requests. Additional visits corrected no short answers and lost seven. On the edited novel, the second pass recovered the correct present age where one pass introduced a contradiction. Both missed the difference. A separate full-document training update failed its selection rule. Recall on one novel and shorter replies therefore coexist with a measured cost to repeating the block on these short tasks.

= Introduction
On a personal computer the relevant question is how often an assistant delivers the correct result before its user gives up. We compare one complete trained LoopQwen checkpoint with the pinned, untouched Base checkpoint. A difference can arise from training, repeated computation or their interaction. This design estimates the result of choosing the deployed loop over Base. We keep the requested action, supporting evidence and final answer in the score, including when an incorrect response arrives quickly.

#pagebreak()
= Models, training and sources
Both models use the certified BF16 Base text backbone and tokenizer. Base executes the 24 original decoder layers. LoopQwen runs the original prefix through layer 11, repeats layers 12–15 and runs layers 16–23 once before the ordinary vocabulary head. The repeated block contains three delta layers and one full-attention layer. Eight later layers can convert the resulting state into the final answer. At two passes there are 28 physical layer visits per token. The one-pass initialization reproduced donor logits before training.

The recurrent update mixes the current state with the fixed input to the block. The learned mixing coefficient at update 576 was #calc.round(d.weights.last_mixing_coefficient,digits:3). We trained low-rank updates on four middle and two suffix MLP output projections and the mixing scalar. During warmup the suffix predicted the full target from both the first and second exits. The third exit joined after 32 updates. This gave #str(d.weights.supervised_exposures_by_depth.at("1")) supervised presentations at exit one and #str(d.weights.supervised_exposures_by_depth.at("3")) at exit three. The product comparison always uses two passes.

#tab((1.0fr,2.2fr,0.6fr),([Training family],[Material and supervision],[Turns]),
[Reasoning],[Human-grounded polarity decisions and comparisons. Changed premises can reverse the answer.],[72],
[Chat],[Conversations, unrestricted rewrites and edited replies, including human-authored Dolly examples.],[72],
[Tools],[Openings, delivery, error repair, reuse and redirection under real provider histories.],[72],
[Documents],[Complete table evidence and versioned conference schedules, with source-grounded targets.],[72])
#cap([Table 1.],[Training is balanced by family. Several rows can use the same source. The audit found 68 distinct chat source groups, 48 reasoning, 67 tools and 56 documents. No complete short-development question appeared verbatim in the fitted examples. This check cannot establish what the donor saw during pretraining.])

The AdamW fit completed 576 updates, four epochs over 288 examples in batches of two. It processed #str(d.fit.positions) input and target positions, with #sec(d.fit.optimizer_seconds) seconds in the optimizer measurements. Real generation was also checked early. Fifteen failed learner histories entered one bounded correction trial with verified calculations. Its loop produced 3/27 supported short answers at 128 output tokens; the selected parent produced 5/27. We retained the parent. Training loss, a located operand and a successful tool return each provide different evidence from a supported answer.

#pagebreak()
= Evaluation on the same Mac
Natural reasoning, verified tools, ordinary chat and full-document supported answers each receive one quarter of the primary score. Each short family has three source groups. Each group contributes an original opening, a meaningful change and an irrelevant change. We average within source first. The book family contains two novels, with original and edited editions of each. The 31 openings therefore have unequal individual weights. Own-history and shared-history follow-ups have separate records.

Each paired opening uses the same token IDs, live providers, greedy reasoning-enabled decoding and output allowance. We report 128 and 512 generated tokens per model turn and allow up to three model turns when callbacks are available. These limits exclude the input. Raters reviewed complete visible outputs against the source and the actual action, with model labels masked. An incomplete answer, a material unsupported assertion or failure to execute an explicitly requested action receives zero.

#image("quality.svg",width:100%)
#cap([Figure 1.],[Supported answers within each family. Each panel uses the same input cohorts. Giving Base more output headroom recovered two chat answers and one document answer. Neither model delivered a correct verified tool opening at either allowance.])

#tab((0.95fr,1fr,1.65fr),([Output allowance],[Macro difference],[Paired 95% interval, pp]),
..for c in ("128","512") {
 let x=d.caps.at(c)
 ([#c tokens],[#pp(x.macro_delta) pp],[#pp(x.paired_interval.first()) to #pp(x.paired_interval.last())])
})
#cap([Table 2.],[LoopQwen minus Base, resampling paired source groups within families. An independent enumeration of all #str(d.direct_audit.at("512").ordered_resamples) ordered source resamples reproduced both intervals. At 512 tokens, omitting one source at a time gave macro differences from #pp(calc.min(..d.direct_audit.at("512").macro_after_leaving_one_source_out.values())) to #pp(calc.max(..d.direct_audit.at("512").macro_after_leaving_one_source_out.values())) points. Both caps use the same 11 sources.])

#pagebreak()
= Which sources changed the outcome?
#tab((1.15fr,1.0fr,0.55fr,0.55fr,1.6fr),([Family],[Source],[Base],[Loop],[Loop gains / losses]),
..for r in d.caps.at("512").sources {
 let group=if r.source=="book" {"McTeague"} else if r.source=="anne" {"Anne"} else {r.source}
 let gains=if r.gained.len()>0 {"+"+r.gained.join(", +")} else {""}
 let losses=if r.lost.len()>0 {"−"+r.lost.join(", −")} else {""}
 ([#pretty(r.family)],[#group],[#str(r.base)/#str(r.n)],[#str(r.loop)/#str(r.n)],[#((gains,losses).join(" "))])
})
#cap([Table 3.],[All enrolled sources at 512 generated tokens. Gains and losses refer to the specific edition or prompt change that switched which model succeeded. An empty last column means the models agreed on every item in that source.])

The largest short contribution came from two studio-writing requests. On the parcel chat requests, the loop gained the changed response and lost the original. The archive reasoning gain occurred after an irrelevant edit. The loop lost the published Anne question that Base answered correctly. The estimate at 512 is therefore sensitive to a few individual decisions. Across the tool sources there were no correct explicit actions from either model. Without a positive effect in those cases, the chat gain offers little evidence that an operator should switch for tool work.

The original stronger-assistant bar calls for at least +5 points with a positive paired lower bound and protected families, with +10 points and positive effects in every family as the stronger target. The narrower deployment question allows an overall loss of at most two points and demands material recall, tool and efficiency benefits. At 512 tokens the current checkpoint clears neither decision. At 128 its macro difference reaches the numerical minimum on development, while both models still score zero on tools and documents.

#pagebreak()
= Reading complete books and using the evidence
The input to each request includes the complete text, the question, chat template and tool declarations. In _Anne of Green Gables_, the original and edited inputs each contain 145,891 tokens. The reader must compare Anne’s early statement that she was eleven last March with a statement near the end. The original later age is sixteen and a half; the edit changes it to sixteen and three quarters. Correct differences are 5½ and 5¾ years. The answers require two distant facts and an operation.

In _McTeague_, the published and edited requests contain 167,014 and 167,017 tokens. The bird begins in a cage. In the published ending it is half-dead and still caged. The edit lets it escape, while McTeague remains handcuffed to the dead Marcus. The question requests a decision and supporting facts from the opening and closing scenes. Naming the bird or guessing the correct polarity is insufficient.

#tab((1.35fr,1.2fr,0.9fr,0.9fr),([Book],[512-token result],[Base],[LoopQwen]),
..for (src,title) in (("anne","Anne"),("book","McTeague")){
 let rows=d.recall.filter(r=>r.cap==512 and r.source==src)
 let b=rows.find(r=>r.arm=="base")
 let l=rows.find(r=>r.arm=="loop")
 ([#title],[All required facts],[#str(b.fact_presentations)/2],[#str(l.fact_presentations)/2],
 [],[Complete answer],[#str(b.complete_presentations)/2],[#str(l.complete_presentations)/2])
})
#cap([Table 4.],[The evidence-only column criterion was reviewed retrospectively on the same frozen generations. The complete-answer criterion was registered before generation. Each novel is one independent source.])

LoopQwen supplied both ages in both Anne editions and never gave the difference. Base supplied the correct original answer in 285.6 seconds cold, then answered two years on the edit. On McTeague the loop said “No” in both editions. It omitted the requested evidence on the original and missed the reversal in the edit. Cold time for the upper-band originals was about 399 seconds for LoopQwen and 358 seconds for Base. Exact full attention and separate hybrid state at every physical visit establish that the model processed those tokens. The zero supported answers from the loop in both bands establish the present limit on useful context.

#if d.at("book_depth",default:none) != none [
#strong[Does the second pass supply the recall?] We repeated the two Anne readings at one pass with identical weights and opening tokens. On the original, one pass supplied both ages. On the edit, it called Anne thirteen and three quarters and described sixteen and three quarters as a future age. The source gives sixteen and three quarters as her present age. Two passes preserved that distinction and both ages. Cold time for the edited answer rose from #sec(d.book_depth.rows.K1.edited.cold_seconds) to #sec(d.book_depth.rows.K2.edited.cold_seconds) seconds. Each depth missed both differences. This intervention locates a benefit in reporting a changed fact on one novel. It adds zero supported final answers.
]
#if d.at("primary_books",default:none) != none [
#parbreak()
#strong[Neutral edits, abstention and full-history tools.] At 512 generated tokens Base answered the published and neutrally edited Anne openings correctly, giving it two of six openings across both novels. LoopQwen scored zero of six. Asked for the exact clock minute of Anne’s statement, LoopQwen correctly said the text did not specify it on both editions, at both allowances. Base reached both output limits on both editions. On four calculator follow-ups per model and cap, covering own and shared histories, neither model executed the requested action. Base gave the right 66 months on the two published-edition follow-ups at 512. Both failed the execution criterion. Repeated editions of the same novel supply one source of evidence.
]
#pagebreak()
= Time to a correct response and executable tools
#image("deadline.svg",width:100%)
#cap([Figure 2.],[Paired success at practical deadlines using the 512-token allowance. The short panel weights its three families equally. The long panel includes the cost of building the entire document prefix. Each plotted point uses recorded completion times.])

#tab((2.7fr,1fr,1fr),([Short workload, 27 openings],[Base],[LoopQwen]),
[Elapsed across all requests],[#sec(d.serving.base.seconds) s],[#sec(d.serving.loop.seconds) s],
[Output tokens across requests],[#str(d.serving.base.tokens)],[#str(d.serving.loop.tokens)],
[Layer visits across all processed token positions],[#str(d.direct_audit.at("512").costs.at("base/short").warm_layer_token_visits)],[#str(d.direct_audit.at("512").costs.at("loop/short").warm_layer_token_visits)],
[Supported by eight seconds],[#str(d.serving.base.by_deadline)/27],[#str(d.serving.loop.by_deadline)/27],
[Mean time to correct, failures charged eight seconds],[#sec(d.serving.base.mean_time_to_correct) s],[#sec(d.serving.loop.mean_time_to_correct) s],
[Peak MLX allocation],[#sec(d.serving.base.peak / 1073741824) GiB],[#sec(d.serving.loop.peak / 1073741824) GiB])
#cap([Table 5.],[Elapsed time includes unfinished or incorrect answers. The time-to-correct measure retains every failed request. Shorter unsuccessful text can reduce raw elapsed time substantially.])

The loop’s short generations reflect the training targets. Of 288 distinct targets, 264 immediately closed the thinking segment, including all 72 tool targets. Sixty tool turns began with a provider response already in the conversation. The loop immediately closed the segment on all 27 terminal short openings. Base did so on zero. We observed a change in generated behavior and a net improvement of three short answers. The shorter sequences reduced raw elapsed time by 82%. Charging failures the eight-second deadline reduced mean time to correct by only 12%.

#pagebreak()
= Local work, independent use and limitations
An operator asked which restart pointer was farther along and how much work a rollback would lose. A second job asked for the extra bytes needed to transfer one standalone directory and whether the backbone files matched. Each job required a real read and an own-history follow-up. We exposed the genuine files and a read-only inventory through the normal tool interface. A direct provider check returned every required source value. Four fresh processes alternated the starting model between blocks.

#let local=d.local.arms
#tab((2.4fr,0.9fr,0.9fr),([Local jobs],[Base],[LoopQwen]),
[Complete jobs],[#str(local.base.complete_local_jobs)/2],[#str(local.loop.complete_local_jobs)/2],
[Correct requests],[#str(local.base.correct_local_requests)/4],[#str(local.loop.correct_local_requests)/4],
[Executed callbacks],[#str(local.base.replayed_callback_results)],[#str(local.loop.replayed_callback_results)],
[Elapsed across four requests],[#sec(local.base.local_elapsed) s],[#sec(local.loop.local_elapsed) s],
[Mean time to correct, charging failures 30 s],[#sec(local.base.local_mean_capped_time_to_correct) s],[#sec(local.loop.local_mean_capped_time_to_correct) s])
#cap([Table 6.],[The loop obtained a real table schema once. Six searches or queries returned empty records and one calculator call failed. Base never invoked a provider. In an independent public four-question GSM8K diagnostic the loop answered one correctly and Base zero.])

We regard improved recall on the changed Anne source as a reason to test a narrower hypothesis. In a separate historical experiment, a different LoopQwen checkpoint answered 62/64 three-fact induction questions versus 2/64 for untouched Base at 768 generated tokens. That older result belongs to a specialized reader, synthetic rule structure and different checkpoint. It gives a concrete example of a job the project once improved. The current weights must earn any transfer claim independently.

#context if d.specialist != none [#strong[Additional development observations.] #d.specialist.paper_summary The full meeting inputs contained 21,737 and 5,251 tokens. In the operator job Base omitted the JSON import on two attempts. LoopQwen repeatedly tried to import the report as a Python module. The reference program ran successfully on the same files.]

The precision screen reduced the donor file from 1.50 GB BF16 to 424 MB at four bits. Greedy trajectories changed, including a lost reasoning answer and fewer productive tool calls. The published reference pair retains BF16. A useful deployment recommendation requires an overall paired lower bound above −2 points, material source-level advantages in recall and tool jobs, and a replicated time-to-correct benefit at a fixed deadline. No such recommendation follows from the current sample.

#if d.at("standard", default:none) != none [
#pagebreak()
= Five standard benchmarks
We sampled 50 labeled items without replacement from each test or validation split, froze their token identities, and used the zero-shot completion tasks in EleutherAI’s lm-evaluation-harness 0.4.13. We scored every answer choice against the full native vocabulary. WinoGrande scores the common suffix under two possible antecedents. The other four tasks use character-normalized conditional log likelihood. This measures which completion each checkpoint ranks highest. Our chat and tool studies measure what users actually receive.

#let names=(arc_challenge:"ARC-Challenge",hellaswag:"HellaSwag",winogrande:"WinoGrande",openbookqa:"OpenBookQA",sciq:"SciQ")
#tab((1.3fr,.7fr,.7fr,.75fr,1.35fr),([Benchmark],[Base /50],[Loop /50],[Gain/loss],[Difference, 95% interval in pp]),
..for (key,r) in d.standard.benchmarks.pairs() {
 ([#names.at(key)],[#r.base.hits],[#r.loop.hits],[#r.gains / #r.losses],[#pp(r.delta) (#pp(r.paired95.first()), #pp(r.paired95.last()))])
})
#cap([Standard benchmark results.],[Gain and loss count discordant paired items. Each benchmark has a distinct 50-item cohort. Intervals use paired bootstrap resampling within benchmarks. Per-arm Wilson intervals, exact McNemar tests, Holm correction and each item’s score are in the archived result.])
The equal-benchmark mean difference was #pp(d.standard.macro_delta) percentage points (paired 95% interval #pp(d.standard.paired95.first()) to #pp(d.standard.paired95.last())). Each answer is worth two points within its own benchmark. Independent reference Base scores on the larger evaluation were 40.1% on ARC-Challenge, 54.9% on HellaSwag, 60.2% on WinoGrande, 35.4% on OpenBookQA and 91.7% on SciQ. These published results contextualize our Mac measurements. #link("https://github.com/YuhuanStudio/Yunmo-Next-1B/blob/main/reports/evaluation.md")[Reference evaluation and settings.]

The loop gained 12 answers and lost 10. Its largest loss was four net answers on ARC-Challenge. No individual benchmark reached the registered significance threshold after Holm adjustment. SciQ had no switches in 50 items. Even then, the exact one-sided 95% upper bound on the population switch rate is 5.8%.

Base spent #sec(d.standard.serving.base.wall_seconds_across_items) seconds scoring all choices and LoopQwen spent #sec(d.standard.serving.loop.wall_seconds_across_items) seconds. Peak MLX allocation was #sec(d.standard.serving.base.peak_mlx/1073741824) and #sec(d.standard.serving.loop.peak_mlx/1073741824) GiB, respectively. The Base process exited and its 250 outcomes passed the upstream scorer before we loaded LoopQwen. Both models consumed the same token IDs. Each processed every candidate ending in full. A second calculation from raw per-token log probabilities checked every grade. We found all 50 sampled items per benchmark had distinct source keys, including WinoGrande author pairs and nonempty SciQ passages. Two generic OpenBookQA stems of one and two words occurred in adaptation text. Neither matching training row contained an answer choice.

#if d.at("standard_depth",default:none) != none [
#strong[Which changes require another pass?] Holding the trained weights fixed, one pass scored 153/250 in #sec(d.standard_depth.serving.at("loop-k1").wall_seconds_across_items) seconds. A second pass fixed four answers and lost seven (−1.2 points, paired 95% interval −4.0 to +1.2). Ten of the twelve two-pass wins over Base were already correct at one pass. The second visit lost four ARC-Challenge items, gained one net WinoGrande item and had zero net change on the other three tasks. The complete benchmark panel gives no reliable aggregate accuracy gain from the extra computation.]
]
#pagebreak()
= Can the loop learn to use the retrieved facts?
We continued from the trained loop with four complete training inputs. The published and edited versions of Hawthorne’s _The House of the Seven Gables_ asked whether an old deed still supported a claim. In the complete FOSDEM calendar, a change to one event’s end time reversed a duration comparison. The book inputs contained 158,281 and 158,283 tokens. Both calendars contained 232,749 tokens. We held the recurrent block fixed and updated the two low-rank factors in the final decoder MLP. This isolates what a small output-side adaptation can recover from an existing representation.

Each AdamW update included answers for all four inputs, with 33, 33, 37 and 45 supervised positions respectively. The training loss fell on all four inputs over 16 updates. On the held novels the updated loop rounded both of Anne’s later ages to sixteen, giving the same five-year difference in both editions. On both editions of _McTeague_ it quoted the publisher’s royalty notice. All four openings failed at each output allowance. At 512 generated tokens the short score fell from 5/27 to 4/27. One rehearsal message improved, one archive decision reversed incorrectly and one clarification lost the missing day. The registered selection rule retained update 576.

The arithmetic tools reveal a different loss. On six invoice variants from two sources, the retained loop issued six executable first calculations. On the ink invoice it calculated for four bottles when the order specified sixteen. Across all six it never delivered the correct final quantity. Base delivered all six quantities correctly, although its first calculator request was rejected five times and absent once. Neither model earned executed-task credit. All #d.training_actions.opening_rows initial tool targets came from TAT-QA financial tables spanning #d.training_actions.distinct_opening_clusters source groups. The five multiplications in those programs each scaled a previous result by 100. Neither invoice’s quantity-times-unit calculation appeared in that form. All 12 database targets followed a provider response. These exposure counts suggest a transfer problem in choosing the first operands and operation.

#if d.at("depth",default:none) != none [
#strong[Pass count on the same weights.] We varied only the number of actual executions of layers 12–15. The question, weights, providers, decoding and 512-token allowance were fixed. Each count produced 27 openings and four follow-ups. Capped time charges failures the full eight-second deadline.
#tab((1.3fr,1.0fr,1.0fr,1.2fr),([Execution],[Correct / 27],[By 8 s],[Mean capped time]),
..for (key,title) in (("base","Untouched Base"),("K1","Loop, one pass"),("K2","Loop, two passes"),("K3","Loop, three passes")){
  let s=d.depth.summary.at(key)
  ([#title],[#str(s.supported)],[#str(s.by_deadline)],[#sec(s.capped_mean) s])
})
The second visit lost the original archive decision, the original and irrelevant-change bridge answers, the irrelevant-change studio message and the changed volunteer instruction. The third visit also lost the irrelevant-change archive answer and the original studio message. Neither transition fixed an answer. No pass count solved both meaningful versions of a reasoning or tool source by the deadline. These are source-masked development judgments on nine sources. The one-pass result uses the same trained weights and establishes no benefit from repeating the block at inference on this panel.
]
#pagebreak()
= Reproducibility and costs
All qualifying runs use the 18 GiB M3 Pro with #str(d.hardware.gpu_cores) GPU cores and macOS #d.hardware.macos. One process owns the GPU lease. The packaged models each contain the tokenizer, runtime wheel and weights and can be moved independently. In the initial 20-request offline replay both models missed all ten supported answers. The expanded installed benchmark reproduced every prompt, output, callback, status and attention-history length across 116 requests. Four separate processes loaded the complete models outside the checkout with network access blocked. At 512 generated tokens, the installed loop solved 5/27 openings in 32.4 seconds of generation and Base solved 2/27 in 179.6 seconds. Both missed both own-history follow-ups. Mean deadline-capped time was 6.64 versus 7.58 seconds. The 56 observed callbacks across both caps are execution events, including provider errors.

#tab((2.2fr,1.1fr),([Development worker charges at this evidence cut],[Seconds]),
[Profiling and verification],[#sec(d.charges.profile)],
[Training and training roll-ins],[#sec(d.charges.fit)],
[Short development and tools],[#sec(d.charges.short)],
[Complete-document work],[#sec(d.charges.long)])
#cap([Table 7.],[Ledger totals include stopped invocations and archived work before the direct-pair scope was adopted. The sum includes earlier experimental costs and is not the price of running one checkpoint. Optimizer seconds on page 2 overlap the ledger. The original development envelope was 43,200 worker-seconds.])

The native fitter saves adapter weights, optimizer moments, random state and order. Exact replay of a stored update and recomputation from scratch are recorded separately. Reusable immutable prefix files have hash-checked manifests. Staged attention retains the full KV histories on disk. Resident execution is used only after qualification against staged outputs. Long inputs are measured in actual prompt positions. #link(repo+"/blob/main/docs/STANDALONE_MIDDLE.md")[Standalone pair and offline restoration.] #link(repo+"/blob/main/RESEARCH_GOAL.md")[Current handoff.] #link(repo+"/tree/main/docs/measurements/middle-depth")[Native evidence and cost receipts.]

== Data, prior work and contributions
#small[Qwen Team. #link("https://huggingface.co/Qwen/Qwen3.5-0.8B-Base")[Qwen3.5-0.8B-Base]. Huang et al. (2026), #link("https://arxiv.org/abs/2610.23036")[Towards Looped Models Done Right, Part I]. Marchenko et al. (2026), #link("https://arxiv.org/abs/2610.00673")[Closing the Loop]. Zhong et al. (2021), #link("https://aclanthology.org/2021.naacl-main.472/")[QMSum]. Cobbe et al. (2021), #link("https://github.com/openai/grade-school-math")[GSM8K]. Project Gutenberg, _Anne of Green Gables_ and _McTeague_; source editions, edits and rubrics are pinned in the #link(repo+"/tree/main/research/diagnostics/middle-depth")[source registrations].]

lukifer23 directed the research goals, hardware constraints and acceptance criteria. GPT 6.2 Astra implemented the experiments, operated the native runs, reviewed the generated trajectories and prepared the manuscript. The PDF is built by `python -m scripts.paper_draft` from a frozen evidence snapshot.
