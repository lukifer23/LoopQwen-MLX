# Repository workflow

Keep the initial commit and future project updates on `main`. Use `https://github.com/lukifer23/LoopQwen-MLX` as `origin`. The owner authorizes local commits and requires authorization before a remote push.

Run model inference, differentiation and optimization through native Apple Silicon MLX. Preserve the pinned Base checkpoint and historical control archives. The active project, paper and new evaluations always compare the complete trained loop with untouched Qwen3.5-0.8B-Base at ordinary depth. Train the loop only. Report effects from this pair as whole-model effects.

Default new research iterations to one direct two-model comparison. For language reasoning, compare the trained recurrent candidate against the pinned untouched Qwen Base at ordinary depth, with reasoning enabled for both. Freeze identical prompt token IDs, generation policy, output budget, ordered items and scorer. Record accuracy and actual time together. Register an explicit decision threshold, maximum budget and stop rule before starting. A failed prerequisite closes downstream jobs. A larger study keeps the same two models. Inspect the same loop at different pass counts only as a within-checkpoint diagnostic. Finish and archive already-committed primary comparisons. See [staged studies](docs/TRAINING.md#stopping-a-staged-study).
