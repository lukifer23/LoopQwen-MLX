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
scripts/run_phase4.sh replicate dual-exit latent evaluate --seed 109
python -m loopqwen compare --experiment phase4
```

The dense time control consumes the measured update seconds of the gated run. Its data-matched comparison is the immutable 250K milestone from that same dense trajectory, when reached. `scripts/run_phase4.sh verify` exercises native gradients and stock K=1 deployment. The [Phase 4 registration](PHASE4_PLAN.md) specifies when to run follow-up ablations.
