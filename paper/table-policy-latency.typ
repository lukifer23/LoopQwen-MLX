#set document(title:"Why the four-layer loop finished sooner",author:"LoopQwen-MLX project")
#set page(paper:"us-letter",margin:(x:.75in,y:.73in))
#set text(font:"Libertinus Serif",size:11pt)
#set par(leading:.5em,spacing:.6em)
#let d=json("table-policy-latency.json")
#let round(x)=str(calc.round(x,digits:1))
#text(size:24pt,weight:"bold")[Why the four-layer loop finished sooner]
#linebreak()
#text(size:13pt)[Opening latency and output length across all 268 requests]

On identical first prompts containing 120,361 tokens in total, the loop spent 49.94 seconds in native generation up to the first token, versus 43.12 for Base. The largest group had 257–512 prompt tokens. Its median user-visible first-token times were 191 and 167 milliseconds. Only 1 of the 256 requests longer than 128 tokens had a faster first token on the loop. In the 12 shorter requests the loop was faster seven times.

#image("table-policy-latency.svg",width:100%)
#text(size:9.5pt)[Figure 1. Left, medians within five prompt-length groups. Right, additive components of the difference in total native worker time. Bars below zero reduce loop time. Timings come from two complete runs with first-request model construction included.]

#table(columns:(1.1fr,.75fr,1.25fr,1.25fr,1.1fr),stroke:none,inset:4pt,fill:(x,y)=>if y==0 {rgb("#dcece6")} else if calc.odd(y){rgb("#f1f5f3")},[*Initial tokens*],[*Requests*],[*Loop tokens/s*],[*Base tokens/s*],[*Loop first faster*],..d.bins.map(r=>([#r.length],[#r.requests],[#round(r.candidate_opening_tps)],[#round(r.control_opening_tps)],[#r.loop_faster_first])).flatten())

*Time accounting.* The loop generated 24,841 tokens after the first token of each of 495 turns. Base generated 87,068 across 383 turns. The corresponding measured decode totals were 217.78 and 1,239.78 seconds. To split their difference, we multiply the change in token count by the mean seconds per token, then multiply the change in seconds per token by the mean token count. The sum equals the exact decode-time difference. Of the 1,022.00 seconds saved in decoding, 715.80 seconds (70.0%) correspond to fewer generated tokens and 306.20 seconds (30.0%) to the lower observed time per token.

The loop spent an additional 6.82 seconds on identical first openings and another 11.15 seconds on later openings, whose tokens and number of tool calls differ between models. Time outside the measured generation calls fell by 0.31 seconds. Adding the five contributions accounts for the full 1,004.33 seconds saved.

*Interpretation.* Short replies can be appropriate, truncated, repetitive or wrong. The primary paper provides the paired quality outcomes. Different generated histories change attention and tool workloads, so the observed decode-rate contribution cannot be assigned entirely to a hardware kernel. First-opening throughput uses the same ordered tokens and is the cleaner measurement of the prefill cost on this machine. Each model processed the panel once.

Rebuild from both committed raw model traces using #raw(".venv/bin/python -m scripts.table_policy_latency"). The length-binned CSV and five-term decomposition accompany this paper.
