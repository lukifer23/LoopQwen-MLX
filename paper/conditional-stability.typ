#set document(title:"Which improvements repeat across recurrent Qwen candidates?",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.7in,y:.55in),numbering:"1")
#set text(font:"Libertinus Serif",size:10.2pt)
#let d=json("conditional-stability.json")
#let a=d.all
#let g=d.repeat_review
#text(size:22pt,weight:"bold")[Which improvements repeat across recurrent Qwen candidates?]
#v(.4em)
*Item stability and repeat grading in two native MLX comparisons* \
LoopQwen-MLX · 27 November 2026

The earlier loop solved 64 of 256 objectively graded requests and the current loop solved 66. Untouched Base solved 42 in both runs. The increase of two correct answers conceals substantial turnover. Both loops solved #a.both_gain requests that Base missed. The earlier loop alone solved #a.earlier_only_gain further requests that Base missed, and the current loop alone solved #a.current_only_gain. Both loops missed #a.both_loss requests that Base solved.

#table(columns:(2.1fr,1fr,1fr,1fr,1fr),inset:5pt,stroke:.4pt+gray,
[*Base result*],[*Both loops correct*],[*Earlier only correct*],[*Current only correct*],[*Neither correct*],
[Base wrong],[#a.both_gain],[#a.earlier_only_gain],[#a.current_only_gain],[#a.neither_correct],
[Base correct],[#a.all_correct],[#a.current_only_loss],[#a.earlier_only_loss],[#a.both_loss])
#image("conditional-stability.svg",width:100%)
*Figure.* Left, strict accuracy differences between the two candidates and descriptive 95% intervals from 10,000 within-family draws of complete registered sources (seed 41857). Right, repeat ratings of identical nonempty Base answers. Each cell counts one source request.

*Methods.* We joined the two physically generated archives on 268 registered IDs, verified all four opening token streams against the source, and checked equality of every emitted Base token, executed call, observation and automatic score. We excluded 12 open prompts from objective accuracy. The candidates differed in initialization and learning. Repeated results on the same requests measure item stability. Independent training seeds and fresh panels are required to measure reproducibility.

*Repeat grading.* The same project assistant reviewed two shuffled sets against their source passages. It knew the preceding study and could recognize previously read answers. On identical Base replies, four-category agreement was #g.control_all.exact/#g.control_all.n (nominal Cohen’s κ #g.control_all.kappa). Restricting to nonempty replies gave #g.control_nonempty.exact/#g.control_nonempty.n (κ #g.control_nonempty.kappa). Binary fully usable status agreed on #g.control_nonempty.usable_equal/#g.control_nonempty.n and the material unsupported flag agreed on #g.control_nonempty.unsupported_equal/#g.control_nonempty.n. Masked alternatives could change between readings. These measurements describe within-reviewer repeatability on these sources.

*Artifacts.* The adjacent objective and ratings CSVs contain the item-level outcomes. The JSON preserves stream fingerprints, every family’s contingency counts, source intervals and both arms’ repeat-rating matrices. Candidate entries in the latter condition on identical visible text. Pooling them with Base would duplicate shared replies.
