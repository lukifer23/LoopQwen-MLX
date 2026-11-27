#let observations = json("file-evidence-example.json")
The first changed-file article in the frozen request order was *#observations.first().page*. Each fresh session received the same caller question and the same file and column names. The claim was:

#block(inset:7pt,fill:rgb("#e3f0eb"),width:100%)[#observations.first().claim]

#for (i,r) in observations.enumerate() [
  #text(weight:"bold")[Snapshot #str(i+1) · publisher judgment #r.gold]
  #linebreak()
  #r.evidence
  #linebreak()
  The loop finished with status #emph(r.candidate_status) and #if r.candidate_correct_lookup [read the intended excerpt] else [did not return the intended excerpt].
  Base finished with status #emph(r.control_status) and #if r.control_correct_lookup [read the intended excerpt] else [did not return the intended excerpt].
  #linebreak()
  #text(size:9pt)[Loop #r.candidate_excerpt]
  #linebreak()
  #text(size:9pt)[Base #r.control_excerpt]
  #v(3pt)
]
#link(observations.first().diff_url)[View the original Wikipedia edit and its parent.]
