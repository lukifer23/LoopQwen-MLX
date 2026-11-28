# LoopQwen versus untouched Qwen Base

The project evaluates one trained looped checkpoint against pinned, untouched Qwen3.5-0.8B-Base at ordinary depth. The current research question is whether it can stay within two percentage points of Base overall and deliver materially better recall, executed tool work and time to a correct answer. The general improvement targets remain +5 and +10 points. Results on the two models quantify the effect of the complete training and architecture recipe. [Standing scope](research/diagnostics/middle-depth/direct-pair.md) · [Specialist thresholds](research/diagnostics/middle-depth/specialist-transfer.md) · [Paper draft](paper/paper-draft.pdf)

## Current execution point

The update-576 direct comparison is complete on 27 short and four complete-book openings at each output allowance. At 512 generated tokens the four-family gain is +2.08 points (paired 95% interval −9.72 to +15.97). The loop returned the required facts on both Anne editions but supplied neither difference. Neither model recalled all the required facts on McTeague. The previous local tool jobs yielded zero complete jobs for either model. [Paired evidence](docs/measurements/middle-depth/direct/parent-512.json) · [Book recall and whole-answer grades](docs/measurements/middle-depth/parent-evidence-recall.json) · [Local operator jobs](docs/measurements/middle-depth/local-workflow-screen.json)

The specialist screen is complete. LoopQwen solved both versions of one of four changed arithmetic pairs and Base solved zero pairs. Both models scored 0/2 on public meeting transcripts and neither completed the local file handoff. [Source-level specialist results](docs/measurements/middle-depth/specialist-result.json)

Full-document fitting continues only for the loop from update 576. Its frozen training inputs contain 158,281–232,749 tokens. Use `.venv/bin/python -m loopqwen resources` before resuming an interrupted worker. The direct stages are `python -m scripts.middle_depth_direct fit`, then `children`. Lock complete source-masked short and book grades before `select` and `final-books`. The child is packaged before evaluation. After selection, run the 116-request installed standalone benchmark and the separate cold full-prefix check. [Run commands and score audit](docs/STANDALONE_MIDDLE.md#benchmarking-the-selected-standalone-pair) [Scope-change receipt](docs/measurements/middle-depth/ordinary-full-fit-closure.json) · [Source identities](docs/measurements/middle-depth/specialist-inputs.json)

## Earlier research

The [archived handoff at 2c01ce5](https://github.com/lukifer23/LoopQwen-MLX/blob/2c01ce585b6216f9cbd8a505b55010854f99c98d/RESEARCH_GOAL.md#historical-handoff) preserves earlier decisions, costs and restart commands as they stood when those studies ran.
