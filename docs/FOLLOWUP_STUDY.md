# Reproducing the human follow-up and wheel study

The same two-visit decoder must answer two questions about a passage, distinguish a new package from an earlier one, and preserve previously measured reading and tool skills. Both arms use the same thinking-enabled Qwen prompts, greedy decoding and three turns of up to 384 tokens. The comparison uses 108 requests on one Apple Silicon machine. The registration fixes the decision before generation. [Registered comparison](../research/experiments/followup-protected-01.json)

The registered software scorer found 10 complete projects for the loop and three for Base. Reviewing ambiguous formatting and clearly attributed comparisons raises Base to five. Human follow-up F1 was 0.164 versus 0.265. Among 36 repeated user requests the new fit gained three and lost all eleven solved by its immediate predecessor. No matched-depth or long-context claim follows for this checkpoint. [Results and original rules](../paper/followup.pdf) · [Per-project sensitivity analysis](../paper/followup-wheel-sensitivity.csv)

## Training and source checks

The human material preserves the first two questions asked in real QuAC conversations. The continuation includes the first human answer. The software material comes from unmodified wheels pinned in separate application lockfiles. Each model can ask the public wheel tool to read a specific metadata field. The tool opens a distribution as an archive and reads its METADATA member. The two verdicts for each held project use different released Python versions. A grounded success requires the correct observation before the final answer. [Human source selection](../research/sources/followup-reading/README.md) · [Package enrollment](../research/sources/wheel-archives/README.md)

Training changes 802,816 shared layer-23 adapter parameters. For each proposed AdamW displacement the implementation computes the gradient of two earlier demonstrations, projects a conflicting displacement and checks the actual resulting loss before accepting the update. The decision on the sampled demonstrations and the outcome on held older requests are measured separately. The first and second visits also record their score for the first visible token on which the positive demonstration and stale alternative differ. [Native training](../loopqwen/research/followup_fit.py) · [Masked reading registration](../research/diagnostics/followup-reading.json)

## Offline installation and replay

The follow-up add-on uses the model and dependency wheels from field kit 01, the plotting dependencies and source history from cross-view update 04, and the newer native runtime included under `updated-runtime/`. Keep all three archives locally. The replay driver verifies each parent hash, restores the pinned Git history, installs in an isolated environment and reconstructs release 05 from its stored checkpoint. Seven example pairs are selected by their position in the frozen panel. They cover a complete human conversation as two separate fixed-history requests, two decisions on the same actual wheel, two earlier schedule requests and a retained lookup.

From the source checkout with the recorded local inputs and dependencies, run

```sh
.venv/bin/python -m scripts.followup_oracle
.venv/bin/python -m scripts.research_archive followup-protected-01 --verify
.venv/bin/python -m scripts.paper_followup
.venv/bin/python scripts/proof_curriculum_field_update.py --followup
```

The last command recreates the add-on, restores it outside the source checkout, compares every installed output token, callback, return and retained attention history with the original selected records, and hashes the regenerated paper and figures. From an extracted installation with `release-05/`, `policy/` and the installed wheel, each JSON file under `examples/` is a complete argument vector for the public `loopqwen` executable. The examples name actual local files, so run them from that installation directory.
