#set document(title:"What happens when the wrong tool call succeeds?",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.83in,y:.72in),footer:context [#set text(size:8pt,fill:rgb("#54645f"))
LoopQwen-MLX · Preference learning from real observations #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:10.5pt)
#set par(leading:.48em,spacing:.43em)
#show heading.where(level:1): h => block(above:8pt,below:3pt)[#set text(size:16pt,fill:rgb("#185c52"))
#h]
#let m=json("valid-return-paper.json")
#let grid(headers,data,widths) = table(columns:widths,align:(x,y)=>if x==0 {left} else {right},inset:(x:5pt,y:3pt),stroke:none,fill:(x,y)=>if y==0 {rgb("#dcece7")} else if calc.odd(y) {rgb("#f3f6f5")},..headers.map(v=>text(weight:"bold",v)),..data.flatten().map(v=>[#v]))
#let cap(body)=text(size:9pt,fill:rgb("#43524d"),body)
#let example(e)=[#emph(e.question) #e.claim
#linebreak()Expected evidence: #e.expected. Executed by the loop: #e.actual.
#linebreak()Loop’s visible response: “#e.answer”. Reference judgment: #e.gold.]

#text(size:25pt,weight:"bold")[What happens when the wrong tool call succeeds?]
#linebreak()
#text(size:14pt)[Learning to select evidence across human judgments, software dependencies and conference schedules]
#linebreak()
November 2026 · LoopQwen-MLX project

= Abstract
An assistant can select the correct file, row and tool and still answer from the wrong fact. The preceding field-selection experiment produced exactly this failure on 12 of 16 software compatibility requests. We trained a new two-visit Qwen policy with preferences between successful calls that return different properties of the same source. We tested whether this transfers to fresh human judgments, eight newly selected Python projects and eight real conference events. #m.abstract

#image("valid-return-outcomes.svg",width:100%)
#cap[Figure 1. Unrestricted generation with identical initial prompt IDs and the same tool interfaces. Practical requests require the observed evidence and a correct final answer. Repeated lookups are reported separately.]

= The user’s task
Checking whether a particular interpreter can run a distribution requires a constraint from its metadata and an inference about a proposed Python version. Verifying a talk’s weekday requires retrieving a date and applying a calendar operation to it. Both problems admit a different call that succeeds on the same object. The study tests whether the model itself selects and interprets the relevant return. Public held labels never direct execution.

#pagebreak()
= Model, physical recurrence and preference objective
The candidate starts from the installed release 05 instruction sibling with an eight-bit prefix and a shared rank-32 adapter on seven attention and feed-forward projections in layer 23. The final layer executes twice for each input and generated token. Each visit receives its own physical attention history. The second visit uses residual coefficient 0.35. The control is the pinned untouched BF16 Qwen3.5-0.8B-Base at ordinary depth. Training and execution use native Apple Silicon MLX. An ordinary-depth matched fit would be necessary to isolate recurrence from instruction initialization, quantization and training.

For each preferred response and its foil, preserve every token of the user and tool history. Let $d$ be the preferred full-response log likelihood minus the rejected likelihood, summing through the terminal token. Compute $d_0$ on the starting recurrent model before any update. The preference loss is $log(1+exp(-0.25(d-d_0)))$. Add a positive likelihood objective, averaging mean cross-entropy over the whole preferred turn and over its semantic positions. For judgments these positions supply the verdict. For calls the focus begins at the first token where the arguments or operation differ. The mixed loss gives the positive term 55% and the preference term 45%. This is a task-supervised application of DPO with executable counterfactuals [1].

Each of 800 AdamW updates draws field selection or calendar operation (44% objective weight), a judgment (41%), a correction after a real irrelevant return (10%), and a retained table example (5%). The latter two use imitation. The optimizer changes 802,816 shared parameters at learning rate 0.00007. The rest of the network remains frozen. #m.fit

#image("valid-return-learning.svg",width:100%)
#cap[Figure 2. Cumulative distributions of the reference model’s full-response preference margins and the mixed objective on the actual sampled updates. The nine training points use different batches; they do not estimate held accuracy.]

#grid(("Reference preference","Distinct pairs","Median nats","Positive margin"),m.references,(1fr,1fr,1fr,1fr))

#pagebreak()
= Source population and independent oracles
The human cohort comprises 12 previously unused InfoTabS ordinary-domain test entities (α1). For each entity select one original true, false and uncertain claim and ask it both inline and using a multi-record file. Each claim has at least four consistent independent human verdict votes and three identical relevant-field selections. The previous study used a separate α3 cross-domain cohort. The preference fit uses 64 training tables, 96 human e-SNLI explanations and disjoint practical sources. Rehearsal and the eight retention requests come from earlier studies. #m.source

Published compatibility checks cover #m.projects. Read each complete original wheel, examine its embedded distribution metadata, check the separately published metadata digest, and apply the declared constraint to two actual interpreter versions. The held conference events came from the published FOSDEM 2025 program. FOSDEM 2023 and 2024 supplied the training events. Verify all 2,759 schedule entries across the three programs against the organizer’s XML and iCalendar exports, including the room and local start and end. Independently re-execute 68 registered CSV lookups with SQLite and eight weekday calls with Python’s calendar. Source documents, callback arguments, observations and raw trajectories are available for inspection [2–6].

= Registered comparison and inference
Both arms receive the identical initial token IDs, tool definitions and ordered requests. Each can use up to three turns and 512 unrestricted greedy tokens per turn with reasoning enabled. Each action is parsed from the generated wire format and its return is appended to the conversation. Only an answered request can receive credit. Weekday success additionally requires the scheduled date and its correct calendar result to appear in order. The entire recurrent stream runs before the control stream. Per-request wall and first-token time therefore include effects of arm order, machine state and the initial load.

The gates require at least 30 grounded human judgments, 12 net gains, exact paired significance at 0.05, and at least two complete tables with more than Base. Both presentation modes must improve, each verdict must retain its score, and the model must solve at least four complete opposing pairs in each practical domain with more than Base. Preserve every successful retained lookup and satisfy completion, meaningful reasoning and resource measures. The full registration specifies direction and thresholds. Native work stops on violation of its 4,100-second or 7.25-GiB budget. A failed primary closes dependent ablation and long-context runs on these weights.

#pagebreak()
= Judgments and evidence selection
#m.humans

#grid(("Family","Loop · correct / finished / total","Base · correct / finished / total"),m.families,(1.12fr,1.55fr,1.55fr))
#grid(("All relevant requests correct","Trained loop","Untouched Base"),m.complete,(1.8fr,1fr,1fr))
#grid(("Human reference (24 each)","Trained loop","Untouched Base"),m.bylabel,(1.8fr,1fr,1fr))

#image("valid-return-milestones.svg",width:100%)
#cap[Figure 3. The first bar counts successful but irrelevant first operations. The remaining bars show the required lookup, complete ordered evidence and a correct grounded answer. For a wheel the required lookup alone supplies the complete sequence. A weekday also requires a subsequent calendar result.]

#grid(("Family","Loop · valid wrong / recovered","Base · valid wrong / recovered"),m.grounding,(.85fr,1.6fr,1.6fr))
#cap[Recovery requires that the full expected sequence eventually appears after a successful irrelevant first return. A later correct final label can still be lost if the sequence is incomplete.]

#pagebreak()
= Real cases and independent reading
The following examples were selected by enrollment order within each family. The CSV retains the final-turn text, executed calls and returns for all 224 trajectories. The archive also retains the full token history from every earlier turn.

#example(m.wheel)

#example(m.agenda)

#grid(("Blind review, 28 per arm","Trained loop","Untouched Base"),m.reviews,(1.5fr,1fr,1fr))
#cap[One project-agent reading. Case IDs and the rubric were fixed before scoring. Raters see the source, question and response with model labels suppressed. Partial credit describes the explanation and does not alter the primary score. Distinctive writing styles can reveal model identity.]

#grid(("Human judgments, 72 per arm","Trained loop","Untouched Base"),m.sensitivity,(1.5fr,1fr,1fr))
#cap[The registered scorer accepts an explicit final verdict. The bare scorer accepts the precise label alone. The last scorer accepts short, unambiguous final sentences. Each retains the original completion and evidence requirements.]

= Limits of the comparison
Three human labels, two conference days, one file API and a small number of real projects restrict coverage. The eight deliberately opposed session pairs cannot establish general calendar competence. The identical human statement appears in two presentation modes, so inference uses the whole entity as its unit. The public source texts may have appeared in model pretraining. Source and phrase overlap checks only apply to the known study materials. The control uses a different instruction history. The earlier 131,084-token qualification and matched three-pass induction gains belong to other identified weights.

#pagebreak()
= Performance, decision and reproducibility
#grid(("All 112 requests","Trained loop","Untouched Base"),m.costs,(1.65fr,1fr,1fr))
#image("valid-return-cost.svg",width:97%)
#cap[Figure 4. Empirical cumulative fractions over every request, including failed and unfinished conversations. Each panel uses a logarithmic time axis. Short responses affect total time as well as utility.]

#m.trade
#grid(("Resource measure","Requests with a lower recurrent value / 112"),m.costwins,(1fr,2fr))
Registered gate failures: #m.gates

The archived registration binds the data, starting policy, decision thresholds and native limits. The archive stores the exact response IDs, real tool returns, starting reference log odds, trained parameters and final AdamW moments. The accompanying request CSV and figures rebuild from that archive. The following 103 pages retain the earlier investigations, matched controls, portable tooling and installed long-context trials.

= References
#text(size:9pt)[
1. Rafailov et al. (2023). #link("https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract.html")[Direct Preference Optimization]. NeurIPS. \
2. Gupta et al. (2020). #link("https://aclanthology.org/2020.acl-main.210/")[InfoTabS: Inference on Tables as Semi-structured Data]. ACL. \
3. #link("https://zenodo.org/records/6578593")[Published table evidence annotations]. Anonymous field-selection and justification records. \
4. Camburu et al. (2018). #link("https://arxiv.org/abs/1812.01193")[e-SNLI: Natural Language Inference with Natural Language Explanations]. NeurIPS. \
5. FOSDEM. #link("https://archive.fosdem.org/2025/schedule/")[Official conference schedule and machine-readable exports]. PyPA. #link("https://packaging.python.org/en/latest/specifications/core-metadata/")[Core metadata specifications]. \
6. Lu et al. (2025). #link("https://aclanthology.org/2025.findings-naacl.65/")[ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities]. Findings of NAACL.]
