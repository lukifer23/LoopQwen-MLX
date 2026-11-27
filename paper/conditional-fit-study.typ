#set document(title:"Did broad response training teach the loop to act on a table?",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.77in,y:.74in),numbering:"1")
#set text(font:"Libertinus Serif",size:10.5pt)
#set par(leading:.43em,spacing:.50em)
#let d=json("conditional-fit-study.json")
#let q=d.decision
#let r(x,n)=str(calc.round(x,digits:n))
#let all=d.sql_margins.last()
#let tint=(x,y)=>if y==0 {rgb("#dcece6")} else if calc.odd(y){rgb("#f1f5f3")}
#text(size:23pt,weight:"bold")[Did broad response training teach the loop to act on a table?]
#linebreak()
#text(size:13pt)[Measured training and free program generation on Apple Silicon]
#linebreak()
LoopQwen-MLX project · November 2026

The instruction-initialized four-layer loop completed #q.updates optimizer updates on #q.all_response_positions supervised positions. At the end of training, #q.training_sql_both_correct of 24 preselected questions produced a freely generated SQL statement with the correct result on both human-annotated tables. The model produced #q.distinct_correct_programs distinct SQL statements among these complete successes. #if d.primary_eligible [It passed the registered prerequisites for one held comparison against untouched Qwen Base.] else [It missed the joint training prerequisite. The registered 340-request product comparison is closed for these weights.]

*Exposure and cost.* Two shuffled epochs supplied #d.source_turns_seen complete response presentations from seven groups, with #q.preference_presentations paired presentations of an executable incorrect program. The native MLX optimizer updated #d.fit.trainable parameters. The frozen backbone was the same 424 MB Qwen instruction pack that produced the five causal hidden states for each position. The fit, predetermined likelihood probes and free program generation consumed #r(d.seconds,2) native seconds and peaked at #r(d.peak_gib,3) GiB of MLX allocation. The preceding feature stage consumed 1,180.34 seconds separately. The minimum and maximum pre-clipping gradient norms were #r(d.smallest_norm,3) and #r(d.largest_norm,3).

#image("conditional-fit-study.svg",width:100%)
#text(size:9.5pt)[Figure 1. Left, the optimizer objective grouped into nonoverlapping sets of up to 32 updates and weighted by actual supervised positions. The sum contains whole-response cross entropy and the SQL preference term. Right, native full-vocabulary loss before and after fitting on eight predetermined training questions per group. These 56 questions measure fitting and are ineligible as held-out quality evidence.]

#table(columns:(1.6fr,.78fr,.78fr,.78fr,.85fr,.84fr),stroke:none,inset:4pt,fill:tint,[*Training group*],[*Initial NLL*],[*Fitted NLL*],[*Gain, nats*],[*State gain*],[*Improved / 8*],..d.group_probes.map(g=>([#g.label],[#r(g.before,3)],[#r(g.after,3)],[#r(g.gain,3)],[#r(g.state_information_gain,3)],[#g.improved])).flatten())

“State gain” compares the fitted readout on its real states against the same fitted weights given the ordinary exit at all five inputs. The real states reduced loss on #d.ablated_improved of 56 probes, with a signed mean gain of #r(d.ablated_mean_gain,3) nats. The replacement inputs are out of distribution. This intervention measures sensitivity to the second visit. An independently fitted single-visit model is needed for attribution.

The seven equally weighted mean changes give a gain of #r(d.balanced_gain,3) nats. The initial reference is the ordinary four-bit instruction sibling, which exactly matched the recurrent model on its initialization probe. Each comparison in the table uses the same teacher-forced prefix, supervised token IDs and normalization. A cold reload independently reconstructed both distributions at #d.cold_positions target positions, with maximum absolute discrepancy of #r(d.cold_error,8) nats across the 56 turn means.

#pagebreak()
#text(size:20pt,weight:"bold")[What changed at the actual decision?]

The practical SQL test fixes the opening through a real schema observation and then hands generation to the model. All 24 questions were selected by hash before fitting and were included in its training set. Generation is greedy with a 192-token cap. The model must end normally and emit a query recognized by the deployed provider. SQLite executes the emitted statement unchanged on both versions of the table. Credit requires a read of the source table and one correctly valued cell with no truncation.

Of 24 attempts, #d.closed_responses ended normally, #d.submitted_queries submitted a recognized query, and #d.both_read read the source successfully in both worlds. The model obtained the right scalar on #d.original_correct originals and #d.altered_correct altered tables. Only #q.training_sql_both_correct were correct on both worlds. The associated CSV records each question, complete visible answer, generated statement, token count and both human values. The real schema step was provided in this diagnostic. Choosing the right tool from the user's opening and composing a final answer form separate end-to-end tasks.

*Correct versus incorrect SQL.* We independently recomputed each program at its own causal prefix on the cold-loaded physical model. For each of 176 verified incorrect alternatives, the margin is the mean log probability on changed correct tokens minus the mean on changed incorrect tokens. A positive value favors the verified program. A value of one nat is the location of the registered softplus margin. Each alternative enters the summary once regardless of its exposure count during training.

#table(columns:(1fr,.62fr,1.25fr,1.25fr,.9fr),stroke:none,inset:5pt,fill:tint,[*Edit*],[*Pairs*],[*Correct ranked higher, initial / fitted*],[*Margin ≥ 1, initial / fitted*],[*Median fitted margin*],..d.sql_margins.map(m=>([#m.edit],[#m.pairs],[#m.ordinary_ranked / #m.trained_ranked],[#m.ordinary_above_one / #m.trained_above_one],[#r(m.trained_median,2)])).flatten())

A higher fitted margin occurred for #all.improved of #all.pairs alternatives. This measures discrimination between known, localized alternatives. Open-ended generation permits the entire vocabulary at every step and can fail before reaching the edited span. Correct ranking and correct free execution therefore have separate denominators.

*Qualification.* The five registered prerequisites cover complete scheduled exposure, finite nonzero gradients, balanced seven-group likelihood improvement, four broad groups individually protected from loss regression, and correct free SQL on six whole questions with four different correct statements. #if d.primary_eligible [All five passed. The fitted weights are eligible for the separate locked held comparison.] else [The final conjunction failed. No held-out score, depth attribution or 128K result is assigned to this fit.] The ordinary four-bit instruction sibling diagnoses the fit. Only the fresh, timed product comparison against untouched BF16 Base can establish model-level superiority. Attribution to a second physical visit additionally requires the equally trained depth control.

*Reproduction.* Run #raw(".venv/bin/python -m scripts.verify_conditional_fit") to cold-load the archived readout and reproduce the probability measurements, then #raw(".venv/bin/python -m scripts.paper_conditional_fit") to rebuild the figures, native step log, SQL predictions, margin table and complete paper. The preceding architecture supplement identifies every underlying source, tokenizer constraint, and the distinct upstream weights in the two model families.
