# Training and reproduction

## Reproduce the gated refinement study

Use the locked arm64 environment and the official Base weights. The Phase 3 training cohort uses the V2 mixed answer/trace corpus. Seed 73, rank-eight block LoRA, four examples per optimizer update and a 256-position ceiling are fixed across paired arms. All arms cross each token budget at the first complete update. The schedule digest includes tokens, loss boundaries, problem IDs and batch boundaries.

```sh
source .venv/bin/activate
python -m loopqwen doctor
python -m loopqwen download
python -m loopqwen data --preset v2
python -m loopqwen data --preset v3
python -m loopqwen.data.retention3
scripts/run_phase3.sh main
scripts/run_phase3.sh promote
```

The main command includes the gate screen and the 25K selection before resuming three arms to 100K. Promotion continues the matched pair with their AdamW moments. The measured 250K run consumed 6,420 unique examples per arm. One user-wide GPU lease serializes workers. To inspect a running study from a second terminal, use `python -m loopqwen resources`. The scheduled arm and immutable checkpoint pointers are under `outputs/phase3/configs/` and `checkpoints/p3_*/targets-*.json`. The [experiment registry](EXPERIMENT_REGISTRY.md) includes the evaluation and report commands, and the [results](PHASE3_RESULTS.md) explain why confirmation stayed sealed.


Run commands from the repository root using the locked environment. YAML overrides inherit configs/defaults.yaml and an optional hardware preset in configs/presets.yaml. The trainer validates spans, precision, checkpoint identity, model, tokenizer, optimizer and dataset. A run name identifies a checkpoint directory. Reusing the name resumes its latest complete update. Use a new name for an independent run.

## Data and loss

The pilot generator calculates integer, decimal, fraction and combinatorial gold answers directly. The project distributes its original procedural examples under its MIT license. Eleven categories share globally unique instance keys across all splits. The dataset manifest pins the tokenizer and records per-split counts and SHA-256 digests. Data construction shuffles each split deterministically. Training performs a second seeded shuffle shared by both twins and cycles that order on subsequent epochs. Public data comes from pinned MIT-licensed GSM8K, with 500 short train questions and 32 examples from the official test split. The pilot uses the gold number and EOS as the response. No model generates labels. The fixed prose set is held back for retention measurement. Use `data --procedural-only` for a pilot with no external reasoning data. [GSM8K source](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c)

The standard batch independently right-pads complete documents. It skips overlength training examples. Response masking includes the first answer token predicted by the last prompt token and ends at EOS. Set response_only to false for full-document loss. A minimum training input width of 16 avoids the short-query attention vector kernel, which lacks the log-sum-exp saved for the VJP on this MLX release. Extra right padding cannot affect an earlier causal prediction. The optimizer sums per-token gradients through all microbatches, divides by the total scored tokens, clips the resulting global norm, then updates FP32 adapters with bias-corrected AdamW. Counts distinguish nonpadding input tokens and supervised labels, including EOS.

Optional `packing: true` concatenates documents and preserves explicit decoder reset offsets. Each segment runs its own hybrid decoder before the combined vocabulary projection. This avoids cross-document attention, delta state and targets. The padded path is the pilot default.

## Adapter and curriculum choices

Policies are block, nearby (two neighbors on each side), last8 and all. Each selects existing linear projections, including the attention and MLP projections. The optional projections list filters leaf names. The implementation uses MLX-LM LoRALinear with scale equal to alpha divided by rank. Frozen decoder layers downstream of the block remain on the derivative path. The integration tests differentiate each visit of a shared adapter independently and check their sum against its tied gradient.

Direct training fixes the configured K. Gradual training linearly increases the probability of the configured depth from zero to one over the first third. Mixed training samples weights for K=1,2,3. Sampling is deterministic from seed, step and microbatch. The optimizer warmup runs per update, with constant learning rate thereafter. Activation checkpointing recomputes each executed decoder layer, including shared calls, using MLX's module-aware transform.

## Resumption and memory

Checkpoint generations contain adapter/gate weights, optimizer tensors, resolved config, base and tokenizer revisions, data manifest, step, cursor, order digest, next step's RNG seed, hardware and measured history. Each update has its own deterministic MLX seed and the data order and curriculum are also reconstructible. Saving writes a new generation before atomically changing latest.json. A writer lock prevents conflicting trainers. Interrupt or terminate the process to finish its active update and checkpoint. Use `--until 6` to test a planned stop. Passing the original config resumes. Each generation can be supplied directly to generate or export.

Memory pressure reduces the sequence limit, then microbatch size, then enables layer checkpointing. If the minimum setting is still under pressure, training stops. Resuming reuses the saved cursor. The comparison checks manifest, effective order digest, token totals, updates, targets and configuration before scoring. Changing available memory may change batch selection. A resulting unmatched comparison fails explicitly. Peak MLX allocation excludes macOS and other apps.

For a different mixture, `data --gsm-traces` includes publisher-supplied solutions on short GSM8K training rows with a 128-input-token cap. Choose the balanced preset to use their full lengths. The public evaluation rows retain numeric gold answers. `data --retention-file documents.jsonl` accepts one document per line with text, source and license fields, records attribution and source digest, removes duplicate documents, and scores each retained text from its second token through EOS. These options change the manifest. The measured pilot used the default numeric-answer mixture.

## Phase 4 logical batches and exits

A logical optimizer update comprises `batch_size × accumulation` ordered examples. With `max_microbatch_positions` set, the trainer draws the logical depth once and streams smaller, independently right-padded forward/backward passes. One global denominator covers all physical passes. The trainer clips once, updates AdamW once and advances the cursor only after the complete update. The minimum physical width of 16 matches the locked Metal attention VJP. A cap below the maximum eligible document length is rejected.

Set `first_exit_weight: 0.25` on a fixed K=2 configuration to supervise both suffixes. Joint and raw per-exit CE, logical targets, head-target evaluations, physical shapes, decoder site-positions, pre-update alpha and optimizer timings are recorded in each checkpoint. Explicit compatible transfer via `fork --mode compatible --migration-reason <reason>` starts a new optimization trajectory and lists restored, omitted and newly initialized tensor names and shape mismatches.

```sh
source .venv/bin/activate
scripts/run_phase4.sh efficiency audit
scripts/run_phase4.sh replicate dual-exit evaluate --seed 109
python -m loopqwen compare --experiment phase4
```

The dense time control consumes the measured update seconds of the gated run. Its data-matched comparison is the immutable 250K milestone from that same dense trajectory, when reached. `scripts/run_phase4.sh verify` exercises native gradients and stock K=1 deployment. The [Phase 4 registration](PHASE4_PLAN.md) specifies when to run follow-up ablations.

## Standalone pointer training

Use the locked environment from the repository root and the pinned Qwen Base model. These stages run without Phase 4 training outputs.

```sh
scripts/run_phase5.sh audit data profile
scripts/run_phase5.sh smoke install --seed 109
scripts/run_phase5.sh evaluate diagnostics followup latency --seed 109
scripts/run_phase5.sh donors transfer --seed 109
```

The four arms share the ordered training maps and requested depths. The frozen probe and both dense comparators receive one terminal label per map. The direct recurrent arm receives one target per executed pass. Run `install` again to resume a compatible interrupted arm. Recovery generations contain all trainable weights, including the categorical head, AdamW moments and the update cursor. The 24-, 120-, 360- and 480-update pointers retain the evaluated generations. The machine-wide lease and checkout conductor serialize workers.

The follow-up command evaluates the prespecified installation rule. A miss triggers a matched low-rank bridge. A qualifying direct model triggers stronger neural controls, unseen depths and terminal-only continuation. [Phase 5 decision rules](PHASE5_PLAN.md)

The published Phase 5 seed-109 categorical checkpoints use identity schema 1. During closeout we found the generator and trainer shared the basename `pointer5.py` in the source digest map. Their recorded Git trees allow independent verification of both files for this study. New runs use separate, fully qualified module names. Legacy resume is restricted to the reviewed compatible trainer and original resolved settings. The archive stores both paths and their SHA-256 values. [Source inventory](measurements/phase5/source-files.json)

## Phase 6 matched installation

From the repository root with the locked native environment, run `scripts/run_phase6.sh data`, then `scripts/run_phase6.sh train`. These stages reuse the verified Phase 5 map identities. Each arm has separate atomic optimizer and milestone checkpoints under `checkpoints/phase6/109/`. Rerun the training command to resume from the latest complete logical update. Use `scripts/run_phase6.sh evaluate` for paired order evaluation and `scripts/run_phase6.sh followup` for the registered decision and eligible continuation.

For native numerical and timing checks run `scripts/run_phase6.sh benchmark --arms randomized`. After evaluating the matched arms and eligible follow-up, archive an independent reproduction with `.venv/bin/python scripts/phase6_archive.py --destination outputs/phase6/reproduction`. Run `python3 scripts/verify_phase6_archive.py outputs/phase6/reproduction` to check the weight hashes and independently score the stored distributions. Run `.venv/bin/python scripts/phase6_predict.py --archive outputs/phase6/reproduction --arm randomized --verify` to recompute 48 predictions in a fresh native process.

## Stopping a staged study

Before launching workers, register the scientific question, mandatory matched controls, minimum success thresholds, maximum training exposure and any update-time or wall-clock budget. Name the stages each threshold unlocks. Reserve enough time for paired scoring, artifact verification and publication even when promotion fails.

At the first registered milestone where a decisive score is available, record the checkpoint, cohort, numerator, denominator, threshold and decision. A failed prerequisite closes its dependent stages immediately. A missing score leaves them pending. An expensive worker starts only when every prerequisite passes. Apply the same check to a manual stage invocation and to an automatic queue. Phase 6's reusable decision function returns the failed and outstanding requirements. [Stage gate](../loopqwen/study_gates.py)

Finish the precommitted comparisons needed to interpret the primary question. Cancel or skip unstarted follow-up training, replication, sealed evaluation and transfer. Record remaining work, elapsed update time and an ETA for closeout. If evidence motivates a different budget, representation or threshold, register a new study before running it. Preserve the original decision and publish the negative result. In Phase 6 the randomized checkpoint cleared both shuffled targets but missed canonical third-hop accuracy by five maps. The matched dense evaluations still answered a distinct primary question.

### Default to a two-model iteration

For the next language-reasoning iteration use the trained recurrent checkpoint at its declared depth and the original, unadapted Qwen3.5-0.8B-Base at ordinary K=1. Enable reasoning for both. The pinned tokenizer template accepts `enable_thinking=True` with `add_generation_prompt=True` and opens the assistant response with `<think>`. Render once, verify the tokenized prompt, and send both models those exact IDs. The switch supplies a reasoning prompt. Actual emitted reasoning, answer completion, truncation, token use and latency need measurement.

Use one short, frozen cohort with the same requested task, examples, order, unrestricted vocabulary, generation policy, token cap and scorer. Count paired wins, losses, ties, complete answers and measured seconds. Pick the numerical promotion condition and wall-clock cap first. Stop after the pair unless the result justifies a specific follow-up. The comparison measures the deployed effect of the trained checkpoint at its selected depth. If attribution to recurrent execution becomes important, run a separately registered pair against the same checkpoint at K=1. Keep findings from categorical readouts in their own protocol.

The first inference-only rapid pair runs with `scripts/run_phase6.sh reasoning-pair` against the saved seed-109 randomized installation. It gives both models up to 384 unrestricted greedy tokens per question. A six-question floor check stops the benchmark if both models exhaust the cap on every question without producing a scorable answer. Otherwise it completes 20 paired questions. Each item is checkpointed and timed independently. [Pair registration](../experiments/phase6_reasoning_pair.json)

## Phase 7 exact exposure continuation

From the registered checkout and locked environment, `scripts/run_phase7.sh exposure` verifies the full local Phase 6 randomized generation against its published metadata, source, adapter and optimizer digests. It resumes at absolute cursor 480 and writes a separate child tree under `checkpoints/phase7/109/exposure`. Both AdamW moments are necessary for exact continuation. A compact inference release alone supports prediction. A newly reproduced Phase 6 run with different native reduction results needs its own lineage registration.

The worker saves complete logical updates on interruption and checks available memory, system pressure and MLX peak. After recurrent pressure warnings, `scripts/run_phase7.sh close-exposure` scores the saved endpoint and preserves the stop. `scripts/run_phase7.sh language` runs the bounded independent Base calibration. A usable protocol unlocks the registered pair. Each generation uses the archived token stream with a fresh cache and a fixed request depth.
