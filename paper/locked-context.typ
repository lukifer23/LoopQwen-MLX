#set page(paper:"us-letter",margin:(x:.85in,y:.75in),footer:context [#set text(size:8pt,fill:rgb("#516860")) LoopQwen-MLX · Full-context field study #h(1fr) #counter(page).display()])
#set text(font:"Libertinus Serif",size:11pt)
#set par(leading:.46em,spacing:.45em)
#show heading.where(level:1): it=>block(above:8pt,below:3pt)[#set text(size:16pt,fill:rgb("#185c52"))#it]
#let d=json("locked-context-paper.json")
#let grid(headers,rows,widths)=table(columns:widths,stroke:none,inset:(x:4pt,y:3.3pt),fill:(x,y)=>if y==0 {rgb("#dcece7")} else if calc.odd(y) {rgb("#f2f5f4")},..headers.map(x=>strong(x)),..rows.flatten().map(x=>[#x]))
#text(size:25pt,weight:"bold")[Read the version that changed]
#linebreak()
#text(size:14pt)[A paired full-context tool task on two adjacent commits in a real Python project]
#linebreak()
LoopQwen-MLX project · November 2026

= The user’s question
An engineer has a complete uv.lock file and wants the pinned version of the package “virtue” and the file’s actual byte count. The dependency updated eight minutes after the earlier lockfile commit. Both revisions were published on November 14, after publication of the two pinned Qwen model snapshots. We supplied every original file byte in the prompt and exposed the file tool on the same physical file. #d.summary

#image("locked-context.svg",width:100%)
#text(size:9pt)[Left, the full rendered input, its first version occurrence, and the shared 131,072-token reference point. Right, the time the user waited for each request, including any tool turns. Both revisions concern one package in one project.]

= Inputs and controls
The earlier revision contains 229,543 bytes. The 148,741-token request contains “2025.7.1” at offsets 138,342, 138,499 and 138,701. The later revision contains 211,210 bytes. Its 137,017-token request contains “2026.11.2” at offsets 136,058, 136,207 and 136,408. Both decimal byte counts are absent from their respective documents. We verified Git blob identities, parsed the unmodified TOML with the standard library, and checked every token prefix. Among the package names present in both snapshots, virtue is the only one whose resolved version set changed. #link("https://github.com/python-jsonschema/jsonschema/commit/4ba8468a0130d890328bb821786bb74bdca32298")[Upstream change.]

Both arms ran through the same offline installed CLI, Qwen thinking template, list of files, cache-state prefill, 128-token block size, retained conversation prefix, and greedy cap of three 256-token turns. Each pair runs loop then Base. The recurrent release reads through seven full-attention histories, including both visits to layer 23. Untouched BF16 Base uses six. Replaying every generated token, function argument and physical return also checks each site’s retained length against the actual consumed sequence. The original limits are 2,350 seconds and 7.5 GiB of native allocation.

#pagebreak()
= What the two models returned
#grid(("Revision","Model","Calls","Termination","User-visible answer"),d.responses,(.65fr,.55fr,.44fr,.95fr,2.7fr))
#text(size:9pt)[Long visible answers are abbreviated to 240 characters in this table. The request CSV preserves the final responses and callbacks. Raw traces retain all prompt and generated tokens, including unfinished model output.]

= Quality and measured cost
#grid(("Metric on two complete revisions","Loop","Untouched Base"),d.metrics,(2.4fr,.95fr,1.1fr))
The primary score requires a completed visible answer giving both exact values and a preceding real file_bytes call returning the correct length. Partial measures separate retrieval, actual measurement and final use. First-token wait includes the request and loading costs recorded by the installed comparison. Decode rate excludes the first token of every generation turn. The prompt rate uses the two original openings and their native first-token times. The resource totals use complete requests in a controlled order on one M3 Pro. Criteria that missed the registered threshold were #d.missing.

= Scope and reproduction
The test measures a full document and a known factual change at the end of that document. It supplies two controlled observations. It cannot estimate general long-context accuracy or cross-host latency. The fixed release here differs from the separately fitted open-answer adapter. The previous 131,084-token changed-register successes, misses on full published books, and repository-tree failures remain part of the project history.

The source, license, clocks and token offsets are in research/sources/locked-context. Regenerate both complete prompts with scripts.locked_context_material. The study archive research/results/locked-context-01 contains the direct comparisons in requests.json.gz and the SHA-256 inventory. Run scripts.paper_locked_context to re-execute the file callbacks, verify the histories and rebuild this report.
