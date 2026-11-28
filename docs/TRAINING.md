# Training and reproduction

## Resume the native middle-depth study

Both BF16 short fits reached update 576. The actual-history correction missed its registered threshold, so the complete-document development comparison uses those two parents and untouched Base. On the training Mac, check the lease and the remaining budget before starting a worker. The document command verifies existing outputs and continues in the registered order. The 128 and 512 settings limit generated tokens per turn. Each complete novel supplies at least 131,072 actual input tokens.

```sh
.venv/bin/python -m loopqwen resources
.venv/bin/python -m scripts.middle_depth_account
.venv/bin/python -m scripts.middle_depth_stages parent-books
```

The standalone guide includes the model directories, public restoration kit and isolated offline comparison. The long-stage registration gives the full-document fitting and selection criteria. [Standalone models](STANDALONE_MIDDLE.md) · [Long-stage registration](../research/diagnostics/middle-depth/long-stage-decision.md) · [Current handoff](../RESEARCH_GOAL.md)

### Native process and disk accounting

`loopqwen resources` lists the checkout conductor and the processes holding its GPU lease, including an isolated model running from a temporary installation. The standalone paired receipt records each model PID, exit and peak MLX allocation. On the verified pair, Base exited before the loop launched. [Standalone process and answer records](measurements/middle-depth/standalone/pair.json)

The storage audit counted 55.47 GB of logical export files and 15.03 GB of blocks private to those files. Shared blocks also occupy disk. Across four compaction passes, APFS clones shared 2,028,363,172 verified payload bytes while preserving independently writable files. The final small-shard pass covered 36 duplicate files and reduced their private allocation from 47,501,312 bytes to zero. Full SHA-256 checks verified each copy. The study limits rebuildable scratch to 16 GiB and stops below 24 GiB free. [Checkout inventory](measurements/middle-depth/retention/checkout-hygiene.json) · [Historical release archives](measurements/middle-depth/retention/transport-archive-clone.json) · [Indexed features](measurements/indexed-feature-storage.json) · [Large duplicate payloads](measurements/middle-depth/retention/legacy-identical-payloads.json) · [Small duplicate shards](measurements/middle-depth/retention/legacy-small-shards.json)

## Archived receipt-repair fit

The original two attempts and the resumed run used 134.83 worker-seconds in total. The resumed run reached the memory-pressure stop during feature extraction before its first optimizer update. The verifier recorded the missing fit and closed the downstream pilot. The original inputs, thresholds and charge history remain available for inspection. [Closure and charges](measurements/receipt-repair-fit-streamed/closure.json) · [Source-copy model guide](SOURCE_COPY_MODEL.md)

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

The transfer pilot has a separately registered, weights-only fork from qualified R624. Run `scripts/run_phase7.sh transfer` on the training host. Its optimizer starts at zero. The model receives randomized verbal prompts and the selected exit predicts one state and one unrestricted vocabulary token from the same hidden representation. The worker preserves the last complete update at its first resource stop. Reinvoking a closed pilot evaluates its saved weights.

From an archive-equipped clone, predict without any optimizer files:

```sh
.venv/bin/python scripts/phase7_predict.py --verify
.venv/bin/python scripts/phase7_predict.py --vocabulary --verify
.venv/bin/python scripts/phase7_predict.py --vocabulary --map 5 4 7 6 0 2 1 3 --start 0 --depth 3
python3 scripts/verify_phase7_archive.py
```

Use `scripts/run_phase7.sh long-latency` for independent K=4–6 requests. Archive an independent run to a new destination with `.venv/bin/python scripts/phase7_archive.py --destination outputs/phase7/reproduction`. Predictions from a new run can differ under native BF16 reductions.

## Bounded curriculum study

Run `scripts/run_rsi.sh pilot` for the two-cycle categorical comparison and `python3 scripts/rsi_archive.py` to verify the published seed-719 records and adapter hashes. The scorer traverses each map and recomputes predictions from all eight probabilities per exit. The learner's selections are training-only. A completed pilot is reused. An unclosed partial lineage remains available for inspection.

The separate commands `scripts/run_rsi.sh time-control`, `scripts/run_rsi.sh replicate` and `scripts/run_rsi.sh traces` name the registered follow-ups. The measured-time control requires the local seed-719 static optimizer generation. The other studies start from compact Phase 7 artifacts. Each worker uses native Metal and runs under a single user-wide GPU lease. [Study registration](RSI_PLAN.md)

## Complete verified responses

On an archive-equipped clone with the locked native environment, the CPU commands validate preserved files and recount recorded predictions. The native command reloads the released adapter and compares eight actual generated token streams.

```sh
python3 scripts/research_registry.py
python3 scripts/research_archive.py trace-depth-01 --verify
.venv/bin/python scripts/research_predict.py --archive research/results/trace-depth-01 --depth 1 --verify
.venv/bin/python scripts/research_predict.py --archive research/results/trace-depth-01 --depth 1 --map 5 4 7 6 0 2 1 3 --start 0
```

For a fresh explicitly authorized training session, open a bounded local window once. Use a separate checkout and output tree for independent reproduction. New optimization can yield different weights on Metal. The exact measured release and optimizer checksum remain in its archive. The training cohort, parent digests, objectives and stop rules are frozen in the [registrations](../research/experiments).

```sh
.venv/bin/python scripts/research_session.py
.venv/bin/python -m loopqwen.research.bootstrap trace-bootstrap-01
.venv/bin/python -m loopqwen.research.depth
.venv/bin/python -m loopqwen.research.bootstrap suffix-trace-01
.venv/bin/python -m loopqwen.research.bootstrap joint-trace-01
.venv/bin/python -m loopqwen.research.calibrate
.venv/bin/python -m loopqwen.research.external
.venv/bin/python -m loopqwen.research.functional
```

Each command runs serially under a user-wide lease. Training uses one clipped update per eight maps with a padded 168-position physical cap. The symbolic replay input contains only the map, start and readout cue. Its three labels enter cross-entropy after recurrent execution. Verbal SFT scores only the response and registered turn-ending token. Denominators are total state labels and total supervised vocabulary tokens in the logical update.

Run `.venv/bin/python -m loopqwen.research.selftrain` for the registered document-map comparison from the released joint parent. A partial run restores its selected map IDs, original generated token sequences, previous generation, exact update cursor and local AdamW state. Generation two samples from the first self-trained successor. Both arms deliberately initialize fresh AdamW at each generation. The fixed procedural verifier checks all three states and the response boundary. Model-dependent selection and sampled responses can both affect the comparison. [Registration](../research/experiments/verified-generations-01.json) · [STaR method](https://arxiv.org/abs/2203.14465)

## Functional replay and thinking-enabled word problems

The terminal-only matched study runs with `.venv/bin/python -m loopqwen.research.terminal`. For the lower-memory replay comparison, run `.venv/bin/python -m loopqwen.research.replay functional-replay-cap84`. It computes curriculum feedback during each real training forward. The preserved 384-position trial ended after six static updates under the system guard.

The GSM8K pilot uses the tracked [frozen source-derived cohort](../research/data/gsm-equations-01.json) and the standard research environment. Run `.venv/bin/python -m loopqwen.research.equations` in a fresh bounded reproduction. The recorded endpoint is update 16 of 32 planned. On the original training host the optimizer, weights and metadata digests are recorded in [the closure record](../research/results/gsm-equations-01/closure.json). A compatible stopped generation can be evaluated using `--assess-saved`. If both paired generations completed before a retention interruption, `--close-scored` seals that measured endpoint without additional neural execution.

```sh
python3 scripts/research_archive.py functional-terminal-01 --verify
python3 scripts/research_archive.py functional-replay-cap84 --verify
python3 scripts/research_archive.py gsm-equations-01 --verify
python3 scripts/audit_gsm_equations.py
```

Study time limits include prior charged invocations in this checkout. A closed research window rejects new native jobs. Preserve the local budget ledgers alongside optimizer generations for exact continuation. The study of mixed permutation and functional training saved the control after update one. The recovery check validates the adapter and AdamW file digests, moment shapes, metadata and step without loading the model. [Commands and remaining budget](NEXT_EXPERIMENTS.md)

The November 24 interleaving window closed early at the owner’s request. Its monitor has exited. A future authorized run uses the original optimizer generation, the frozen schedule, and the remaining study allowance. The closure record separates preflight observations from worker time. [Restart instructions](NEXT_EXPERIMENTS.md) · [Final accounting](../research/sessions/interleave-closure-1795489467/closure.json)

## Reproducing the learned attention reader

The released reader lives beside the arithmetic adapter in `research/results/attention-memory-01/model`. The optional `attention-reader.json` binds its 32,929-parameter scorer to the exact adapter digest. Native generation activates it at 1,024 prompt tokens. The ordinary CLI accepts user text, and `--thinking` applies the same Qwen template used in evaluation. The export bundles both the scorer and the full-history generator. [Trained checkpoint](../research/results/attention-memory-01/model)

```sh
.venv/bin/python scripts/reader_prompt.py --study attention-memory-01 --index 7 > /tmp/loopqwen-reader.txt
.venv/bin/python -m loopqwen generate --checkpoint research/results/attention-memory-01/model --thinking --prompt-file /tmp/loopqwen-reader.txt --max-tokens 96
.venv/bin/python -m loopqwen export --checkpoint research/results/attention-memory-01/model --output exports/my-attention-reader
```

Use `--study attention-memory-128k --index 0` or `--index 1` to reconstruct either 131,078-token request. Add `--offload-dir` with an existing scratch directory to run the same BF16 storage policy used for those documents. Both the CLI and the bundled generator return the selected passage, emitted IDs, time to first token and total seconds. Disk-backed results also give the temporary byte count.

For an independent training reproduction, use the pinned environment and a fresh output tree. The primary module regenerates and checks the published synthetic training and held-out data before fitting the scorer. Its optimizer checkpoints preserve all four trainable tensors, eight Adam moment tensors and the update cursor. The 128K study requires a verified qualifying archive, then checks a disk replay against the resident reference before starting the paired requests.

```sh
.venv/bin/python -m loopqwen.research.attention_memory
python3 scripts/research_archive.py attention-memory-01
.venv/bin/python -m loopqwen.research.counterfactual_memory
python3 scripts/research_archive.py attention-memory-128k
```


The standalone merged reader was verified in a fresh process. Its 2,054-token example produced all 29 original output IDs, and its merged short-probe relative logit RMS was 0.01651. The merged decoder and reader contain 752,425,954 unique parameters, including the decoder's learned recurrence scalar and the 32,929 passage parameters. [Deployment and independent validation record](../research/sessions/2026-11-25-followup/closure.json)

## Learned exit and natural inference

The learned exit reuses the ordinary Qwen text weights for 704 generated tokens. An open thought then replays its entire history at K2. The trained linear readout adds one scalar to the existing thinking-close token's logit. Greedy decoding still considers every vocabulary token. The standalone schema-5 export stores the dense array set once, the unmerged LoRA tensors and the learned readout. Export verification uses a fresh process. A separately recorded request crosses the depth boundary and reproduces all 731 original output tokens. [Export verification](measurements/learned-exit-export.json)

The entailment model trains on human-written e-SNLI and MultiNLI triplets. Each optimizer update contains two premises, three distinct relations per premise and one language anchor. The independent comparison uses another wording and excludes every primary training and evaluation premise. The strict scorer reads the visible answer in each completed first turn. A blinded second pass credits unambiguous free-text judgments, and the paired permutation test swaps the models' predictions for all three claims together. These commands regenerate the pooled counts from the archived answers and frozen review labels.

```sh
.venv/bin/python -m scripts.research_answer_audit learned-exit-01 \
  --annotations docs/measurements/answer-audits/learned-exit-01.json \
  --replication learned-exit-confirm-01 \
  --replication-annotations docs/measurements/answer-audits/learned-exit-confirm-01.json
.venv/bin/python -m scripts.research_relation_audit entailment-triplets-01 \
  --annotations docs/measurements/answer-audits/entailment-triplets-01.json \
  --replication entailment-triplets-confirm-01 \
  --replication-annotations docs/measurements/answer-audits/entailment-triplets-confirm-01.json
```

The shared-weight conditional checkpoint learns its input selector on 384 separate requests. The normal `generate` command runs the selector before decoding, and includes its work in the reported latency. It accepts the existing `--thinking`, `--max-tokens`, `--prompt-file` and `--offload-dir` options. The schema-4 standalone export includes the selector, deployed adapter banks and full-history attention reader. [First integrated 56-request comparison](../research/results/conditional-depth-01)

## Reproducing the three-pass native layer comparison

The pinned base revision is `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Each standalone induction reader stores its small set of trainable tensors plus normalization statistics. Loading the reader binds its low-rank tensors to the original Qwen layer 15. The primary and matched archives contain the three-pass and single-pass weights respectively. Their registration files specify the shared training seed, optimizer, batch size, examples and fully rendered test token IDs through the material manifest. [Three-pass weights](../research/results/native-induction-anchored-01) · [Matched one-pass weights](../research/results/native-induction-anchored-matched-01) · [Frozen study material](../research/data/native-induction-anchored-01.json)

To verify the saved comparisons, run these commands in the pinned project environment. They check the immutable file inventory, independent story splits, native parameter counts, frozen weights, emitted token boundaries, answer scores, common-prefix scores, timing and paired decisions.

```sh
.venv/bin/python scripts/research_registry.py
.venv/bin/python scripts/research_archive.py --verify native-induction-anchored-01
.venv/bin/python scripts/research_archive.py --verify native-induction-anchored-matched-01
.venv/bin/python scripts/research_archive.py --verify native-induction-anchored-replication-01
.venv/bin/python scripts/research_archive.py --verify native-induction-anchored-bypass-01
```

In a fresh checkout with the native MLX environment, the corresponding training and scoring entry point is `.venv/bin/python -m loopqwen.research.native_induction STUDY`. Replace `STUDY` with a registered study ID from the commands above. Open a local bounded execution window with `.venv/bin/python scripts/research_session.py` before the first measurement. The primary study fits the three-pass reader from pinned Base. The matched study fits the one-pass reader. Replication and bypass load their registered archived weights. Fresh local measurements go under `outputs/research/STUDY`. Running the same command with `--freeze` regenerates and verifies the input material without model inference.


The combined release stores the native layer's LoRA tensors, learned projections, prompt scope and question-intent scope alongside the prior adapter banks. The dense pretrained arrays are shared. The standalone export contains 757,888,967 distinct scalar parameters and normalization buffers. An independent process reads the bundled runtime, compares vocabulary logits across the ordinary and recurrent banks, and reproduces both short reader probes. The registered release then reloads the standalone weights for all generation and retention measurements. [Full native release record](../research/results/native-language-release-02)

To replay the actual second long-context intervention with those weights, generate its frozen user message and submit it through the same public CLI. The prompt helper checks that applying the thinking template reproduces every original input ID.

```sh
.venv/bin/python scripts/reader_prompt.py --study conditional-language-128k --index 1 > /tmp/loopqwen-131083.txt
.venv/bin/python -m loopqwen generate --checkpoint exports/native-language-release-02 --thinking --prompt-file /tmp/loopqwen-131083.txt --max-tokens 96
```

## Reproducing executable tool training and evaluation

The recurrent execution model has an 8-bit instruction-tuned text prefix and a BF16 native final attention layer. The final layer and its rank-32 adapters execute twice with an independent cache at each site. Both the untouched Base and trained one-pass comparisons use the Base tokenizer, a genuine open thinking turn, greedy full-vocabulary decoding and the same registered generation allowance.

```sh
.venv/bin/python scripts/research_session.py
.venv/bin/python scripts/tool_study.py tool-dialogue-typed-01
.venv/bin/python scripts/research_archive.py tool-dialogue-typed-01 --verify
.venv/bin/python scripts/tool_study.py tool-dialogue-typed-depth-01
.venv/bin/python scripts/tool_bfcl_study.py tool-bfcl-multiple-01
```

Each run writes fresh native outputs to `outputs/research/<study>/`. The archive verifier reconstructs the whole conversation from emitted token IDs, validates each call against the supplied JSON Schema, replays its callback and recomputes the final score. To verify an existing publication use `scripts/research_archive.py STUDY --verify`. To seal measurements for a new study run `scripts/research_archive.py STUDY`, then update the registry. The matched-depth check uses the same 1,074 examples, 600 updates, minibatch order, rank and learning rate. The public execution set and its acceptance criteria are frozen before either fit.

To rebuild the standalone assistant from the verified parent artifacts in a fresh checkout, first recreate the qualified language export. The second command assembles a local directory and writes a SHA-256 inventory of all bundled files. Change destination paths when an existing export is present.

```sh
.venv/bin/python -m loopqwen export --checkpoint research/results/native-language-release-02/model --output exports/language-parent
.venv/bin/python scripts/export_tool_assistant.py \
    --language exports/language-parent \
    --language-record research/results/native-language-release-02 \
    --trained research/results/tool-dialogue-typed-01 \
    --qualification research/results/tool-dialogue-typed-01 \
    --destination exports/my-tool-assistant
```

The [paper and supplements](../paper/README.md) include the structured outcomes, compute costs, controls and data attribution. The [local package example](../examples/local_packages.py) shows how a caller passes ordinary Python functions into an interactive tool session.

### Human reasoning and real tool workloads

`scripts/tool_study.py STUDY` fits or restores exactly the weights in the registration. The calculator study is `tool-dialogue-program-reasoning-01`, the unchanged-model SQLite and installed-package studies are `tool-dialogue-table-transfer-01` and `tool-dialogue-package-chain-01`, and the cross-service venue study is `tool-dialogue-venue-state-02`. The venue builder records the whole human conversation up to each enrolled turn, freezes two inventories and checks every training count with both SQLite and a separate row filter. `scripts/contrast_study.py contrast-boolq-transfer-01` runs the human-edited reading comparison. The natural document and raw language studies use `python -m loopqwen.research.bridge_reading bridge-reading-01` and `python -m loopqwen.research.cache_language cache-language-01`. Each command needs the environment and exclusive native lease described above. [Study registrations](../research/experiments) · [Data notices](../research/licenses)

Each recent tool study records its own checkpoint lineage. The jointly trained adapter and its one-pass control use the same 2,527 turns and target positions. Replay the paired decisions with the two commands below. The matched comparison needs the archived candidate and both transcripts. It does not need another optimizer run.

```sh
.venv/bin/python scripts/research_archive.py tool-dialogue-counterfactual-01 --verify
.venv/bin/python scripts/research_archive.py tool-dialogue-counterfactual-depth-01 --verify
```

The learned calculator planner runs with `python -m loopqwen.research.operand_study operand-planner-01`. Its first turn serializes the native layer's chosen operations and actual operand pointers. Subsequent turns use ordinary free generation over the tool response. The semantic endpoints score numerical content and verified tool grounding separately. The causal page-prefill comparison runs with `python -m loopqwen.research.register_allocation register-page-prefill-01` and records attended query–key pairs and physical KV offsets at each site. [Registered workloads and budgets](../research/experiments)

The installed Python API also accepts follow-up user messages in a persistent conversation. The session stores visible assistant messages and actual JSON tool observations. A new request rerenders that history under the same tool definitions. Earlier callbacks execute only when the model makes a new call. Clear the context by constructing a new conversation.

```python
from loopqwen.package_tools import tools
from loopqwen.tool_model import Assistant

chat = Assistant("exports/tool-language-01").conversation(tools())
first = chat.respond("What version of the Python distribution mlx is installed in this environment? Use the available tool.")
follow = chat.respond("Repeat the version from that observation.")
print(first["answer"], follow["answer"])
```

Multi-user-turn callers can use the installed CLI with repeated follow-up arguments, or retain a conversation from the Python assistant. The release workload replays actual package and interpreter callbacks. Its equal-training control restores the one- and two-pass adapters, checks identical supervision and reconstructs every prompt and observation from both users’ messages. [Module evaluation](../research/results/conversation-followup-modules-01) · [Equal-training registration](../research/experiments/conversation-followup-depth-01.json)

### Resuming from an external tool result

Use this entry point when your application owns the tool execution. Supply the original user request, the assistant call and one tool return per call in the original order. The model sees that history and can answer or make a fresh call through the supplied catalog. The returned trace contains the newly generated turns. The input history is preserved.

```python
from loopqwen.calculator_tool import tool
from loopqwen.conditional_reentry import ConditionalAssistant
from loopqwen.qwen_tools import assistant_message
from loopqwen.tool_session import compact

calculator = tool()
assistant = ConditionalAssistant(
    "exports/tool-language-packed-05",
    "research/results/conditional-program-fit-01",
)
action = {"name": calculator.name, "arguments": {"steps": ["37 * 14"]}}
result = calculator.callback(**action["arguments"])
history = [
    {"role": "user", "content": "What is 37 times 14? Use the calculator."},
    assistant_message([action]),
    {"role": "tool", "content": compact(result)},
]
reply = assistant.continue_after_tool(history, [calculator], max_turns=2)
print(reply["status"], reply["answer"])
```

The external-history continuation was also checked on the trained conditional model. After a real callback, resuming from its supplied history reproduced the uninterrupted model's 542-token opening and all 70 generated token IDs. [Uninterrupted and resumed traces](measurements/external-tool-native.json)

### Reproducing the source-reasoning and tool comparisons

The registered staged pair ran 180 identical user requests through each model. It includes complete answers, actual provider observations, waiting time, peak native memory and full-history traces. The reports below rebuild from the immutable local archives and the separately committed source judgments. They also recount every financial calculator intervention and all four database histories per question.

```sh
.venv/bin/python -m scripts.research_archive staged-query-transfer-01 --verify
.venv/bin/python -m scripts.shared_query_reading report --staged
.venv/bin/python -m scripts.staged_query_distribution report
.venv/bin/python -m scripts.staged_tool_consistency report
.venv/bin/python -m scripts.shared_query_calculation_replay --report --staged
.venv/bin/python -m scripts.staged_completion_audit
.venv/bin/python -m scripts.paper_repair_query
.venv/bin/python -m scripts.paper_overview
```

The corrective fit starts from the exact preceding two-read weights. It learns from 672 original training turns and the 270 continuations on which that parent remained inside its thought channel at the assigned prefix length. It also rehearses 192 complete language and tool turns. The corrected continuation is a training target. Evaluated assistants start with the original published user prompt and generate freely. The registered fresh study compares one model against untouched Base on 24 new StrategyQA topics and 24 separate TAT-QA tables. [Native training-prefix receipt](../research/sources/learner-prefix/qualification.json) · [Corrective fit](../research/experiments/learner-prefix-fit-01.json) · [Fresh study](../research/experiments/learner-prefix-transfer-01.json)

```sh
.venv/bin/python -m scripts.learner_prefix_panel
.venv/bin/python -m scripts.verify_learner_prefix_panel
.venv/bin/python -m scripts.long_branch_sources --download
```

The last command reconstructs the six separate long-context training sources from pinned upstream commits. Each question asks for a value that changed between two actual files. The source receipt records the deciding token offsets and the exact complete user prompt. [All changed source values and hashes](../research/sources/long-branch/qualification.json)

### Running the trained two-read model with a full context

The shared-query decoder uses the packed instruction backbone from the conditional study and the trained controller under `research/results/repair-query-fit-01`. On a long document the `--staged-context-dir` option moves completed attention prefixes into private temporary files. Each generated position still reads the complete prefix twice at the final Qwen layer. Use an existing scratch directory with several GiB free. A complete 148,364-token lockfile occupied 1.98 GiB of temporary space during the native qualification. [Exact logits and physical histories](measurements/staged-query-memory.json)

```sh
.venv/bin/python -m loopqwen ask \
  --checkpoint exports/tool-language-packed-05 \
  --session-policy research/results/repair-query-fit-01 \
  --tool-format qwen --staged-context-dir /tmp --trace \
  --prompt-file request.txt
```

Place the full document and question in `request.txt`. The trace records the complete opening, emitted tokens, temporary byte count and retained length at each attention layer. Requests that fit in memory can omit the staging option. Generation and all model arithmetic use native MLX in either storage mode.

The linked document reader runs with `python -m loopqwen.research.linked_reading linked-reading-01`. Its material builder excludes earlier human questions and support titles before enrolling the new fit and evaluation. The reader exposes its two document scores per item. The archive verifier reconstructs the selected indices, the full input IDs and the attention window, then regrades the complete unforced answer. [Training and evaluation registration](../research/experiments/linked-reading-01.json)

### Rebuild the field and packed-register releases

The field release requires the qualified language and typed-tool exports and the published typed prefix crossover. The packed-register release also requires the sealed physical 131K comparison. The following entry points verify the published artifacts without regenerating the models.

```sh
.venv/bin/python scripts/research_archive.py tool-language-release-02 --verify
.venv/bin/python scripts/research_archive.py register-packed-direct-full-01 --verify
.venv/bin/python scripts/research_archive.py packed-release-retention-01 --verify
```

To reconstruct the packed-register bundle from an existing verified field export, run the following from the repository root at the qualification revision. It verifies both published qualifications and every parent file before building the new directory. The same output path can be reused when its contents match exactly.

```sh
.venv/bin/python - <<'PYCODE'
from pathlib import Path
from scripts.export_packed_register import build

r = Path("research/results")
build(Path("exports/tool-language-field-02"), r / "tool-language-release-02",
      Path("exports/tool-language-packed-03"), r / "register-packed-direct-full-01")
PYCODE
```

The native qualification runner is `python -m loopqwen.research.packed_release packed-release-retention-01`. It alternates isolated offline processes and records actual model outputs, package callbacks, and all per-token target losses on the two books. [Frozen protocol](../research/experiments/packed-release-retention-01.json) · [Archival checks and output traces](../research/results/packed-release-retention-01)

To rebuild qualified release 05 after reconstructing release 04, run the following from the checkout. The builder verifies the parent inventory, copies its payloads and updates the tokenizer class and release note.

```sh
.venv/bin/python -c 'from pathlib import Path; from loopqwen.research.unicode_release import build; build(Path("exports/tool-language-packed-04"), Path("research/results/packed-release-retention-03"), Path("exports/tool-language-packed-05"))'
.venv/bin/python scripts/research_archive.py packed-release-retention-04 --verify
```

To reconstruct corrected release 04 from qualified release 03, use the following build. It pins the packed evidence-mask implementation and its operator instrumentation. The archival check verifies the four original outside-checkout workload streams.

```sh
.venv/bin/python - <<'PYCODE'
from pathlib import Path
from scripts.export_packed_register import build

r = Path("research/results")
build(Path("exports/tool-language-packed-03"), r / "packed-release-retention-01",
      Path("exports/tool-language-packed-04"), r / "register-mixed-weight-full-01",
      compact=True, packed_reader=True)
PYCODE
.venv/bin/python scripts/research_archive.py packed-release-retention-03 --verify
```

### Reproducing the newer reasoning and file studies

Each command below uses the saved registration and frozen material. Compute trials require the corresponding original weights, the locked MLX environment and an open local research session with its full budget available. Run a single worker at a time. To verify an existing result without fitting it again, use `python scripts/research_archive.py STUDY --verify`. Each archived decision includes the original study registration.

| Study | Native module after `python -m` | What the worker measures |
| --- | --- | --- |
| `meaning-alignment-01` | `loopqwen.research.meaning_alignment` | Free and fixed-prefix judgments on changed sentence pairs and full public support posts |
| `register-fused-base-full-01` | `loopqwen.research.register_allocation` | Physical 131K evidence, prefill and post-first throughput against untouched Base |
| `packed-decode-kernel-02` | `loopqwen.research.packed_decode` | Query and KV tensor identities, numerical outputs and synchronized times across two native operators |
| `draft-program-01` | `loopqwen.research.draft_program` | Ordinary-prefix drafts, recurrently generated programs, calculator observations and final human-problem answers |
| `folded-system-01` | `loopqwen.research.folded_system` | Teacher state reconstruction and user-visible outcomes from one physical recurrent backbone |
| `action-preference-01` | `loopqwen.research.action_preferences` | Frozen-reference preference fitting and exact executable results after user clarification |
| `register-mixed-weight-full-01` | `loopqwen.research.register_allocation` | Every changed map, charged conversion, physical history and throughput on both 131K counterfactuals |
| `packed-release-retention-02` | `loopqwen.research.packed_release` | Four outside-checkout workload streams, original 131K outputs, exact book probabilities and real calls |
| `report-calculation-01` | `loopqwen.research.report_calculation` | Issuer-disjoint financial reports and external school tables, actual intermediate operations, final answer and cost |
| `document-search-field-01` | `loopqwen.research.document_search` | Caller-supplied full files, native model queries and returned character spans on human questions |
| `document-evidence-01` | `loopqwen.research.document_search` | Native first-pass evidence supervision during actual document searches, followed by free recurrent answers on 40 files |
| `packed-release-retention-03` | `loopqwen.research.packed_release` | Corrected evidence-mask delivery against the shipped release at 131K and 24K, whole books, short reasoning and tools |
| `dialogue-state-fields-01` | `loopqwen.research.action_preferences` | First-pass field-presence supervision and unconstrained second-pass calls and clarifications |
| `selective-raw-return-01` | `loopqwen.research.selective_raw` | A zero-initialized learned return from the second native layer application, trained across 320 real works and tested on 24 additional works |
| `future-token-prose-01` | `loopqwen.research.selective_raw` | Shared Q/K/V/O at two prediction horizons, paired source-level inference and six complete cache replays |
| `session-tokenizer-02` | `loopqwen.research.session_tokenizer` | Four cold release-04 repetitions of the same 50 requests with 15 switches, measuring exact outputs and tokenizer identity |
| `repository-tree-field-01` | `loopqwen.research.narrative_book` | Two genuine project object listings, independently verified PDF sizes and complete native history above 169K |
| `shared-session-unified-01` | `loopqwen.research.shared_session_study` | A single recurrent adapter across reasoning and calls, with equal per-request loss and 72 fresh human questions |
| `shared-session-01` | `loopqwen.research.shared_session_study` | One resident instruction backbone, two recurrent views, physically shared weights and a cold 120-request comparison |

For example, `python -m loopqwen.research.action_preferences action-preference-01` fits from the archived two-pass typed tool weights. The preference builder retains the original source line for every chosen/rejected pair, selects read-oriented topics, normalizes schemas, converts single calls to the installed JSON protocol, and splits distinct questions before fitting. Rebuild the frozen inputs using `python scripts/preference_material.py /tmp/loopqwen-preference-source`. NVIDIA distributed the original When2Call data under CC BY 4.0. The file rows in the two ecological tables come from Kristen Gorman and the Palmer Station LTER, distributed in the Palmer Penguins package. [Original preference data](https://huggingface.co/datasets/nvidia/When2Call) · [Attribution license](https://creativecommons.org/licenses/by/4.0/) · [Original ecological measurements](https://allisonhorst.github.io/palmerpenguins/)


The future-token study uses raw paragraphs from Standard Ebooks editions by Carroll, Shelley, Stevenson, Doyle, Douglass, Franklin, Adams and Muir. Its eight manual sources come from the Git project at version 2.47.0. The encyclopedia paragraphs are by Wikipedia contributors as distributed in SQuAD 1.1 by Rajpurkar and colleagues under CC BY-SA 4.0. The material builder records the work, the source-file revision or hash, and the paragraph or manual range for each excerpt. It normalizes whitespace and retains up to 129 input tokens. Regenerate the inputs with `python scripts/future_token_material.py`. Use `python -m loopqwen generate --checkpoint research/results/future-token-prose-01 --prompt-file passage.txt --max-tokens 64` after the fit has been archived to generate a continuation from its actual weights. [Source inventory](../research/data/future-token-panel.json) · [Standard Ebooks](https://standardebooks.org/) · [Git documentation and license](https://github.com/git/git/tree/777489f9e09c8d0dd6b12f9d90de6376330577a2) · [SQuAD data and license](https://rajpurkar.github.io/SQuAD-explorer/)

Rebuild the shared-session material from its pinned public sources with `python scripts/shared_session_material.py`. The reasoning adapter learns from worked MetaMathQA solutions, BoolQ passages, and all ten supplied passages per HotpotQA question. The request gate trains separately on external-action and directly answerable requests. Run `python -m loopqwen.research.shared_session_study shared-session-01` inside an open research session to fit both components and execute the two-model trial. Each model starts cold, sees the same shuffled request stream, and can invoke the same real tools. The worker writes weights, every opening and generated token, observations, request timings, cache offsets and physical parameter-sharing checks. [Study registration](../research/experiments/shared-session-01.json) · [SmolTalk source](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) · [BoolQ](https://github.com/google-research-datasets/boolean-questions) · [HotpotQA](https://hotpotqa.github.io/)

After archiving the fit, run your own question with the same policy and ordinary Base. The trained policy binds to the exact packed-03 export inventory. The trace reports the selected view, the model’s probability that it needs external evidence when tools are present, and each actual callback.

```sh
.venv/bin/python -m loopqwen ask --checkpoint exports/tool-language-packed-03 \
  --session-policy research/results/shared-session-01 --calculator \
  --prompt "We packed four crates with 18 jars each. Five jars broke. How many are intact?" \
  --max-tokens 384 --max-turns 3 --no-reuse-prefix --compare-base --trace
```


For the trained document reader, regenerate the 40-file evaluation and all fitting searches with `python -m scripts.document_evidence_material`. The source builder opens the original SQuAD and QuAC material, actually calls the same SQLite-backed tool that users receive, and labels the returned character intervals. The archive verifier rebuilds the fitting dialogs and executes every evaluation search again. To fit and evaluate, run `python -m loopqwen.research.document_search document-evidence-01`. After archiving the checkpoint, pass `--tool-adapter research/results/document-evidence-01` alongside `--document your-file.txt` to `python -m loopqwen ask`. [Frozen registration](../research/experiments/document-evidence-01.json)

The repository-tree field check uses one practical question against the *complete*, unmodified output of `git ls-tree -r -l` at two real revisions. Both the stored PDF object and its byte count changed. To reproduce one of the actual user prompts and run it against the installed models, use the following commands. This reads 169,890 input tokens in each arm. The study also runs the second revision and alternates model order.

```sh
.venv/bin/python -m scripts.repository_tree_material \
  --show-request c6a08ff72dd895ed1578a0c9cc4940159dcc52c8 > /tmp/loopqwen-tree-question.txt
.venv/bin/python -m loopqwen ask --checkpoint exports/tool-language-packed-04 \
  --prompt-file /tmp/loopqwen-tree-question.txt --max-tokens 160 --compare-base --trace
```

Recreate the work-balanced token windows with `.venv/bin/python scripts/source_balanced_material.py`. The script uses pinned PG-19 book generations, the CPython documentation at a fixed commit, and the SQuAD article sources. Training and primary evaluation are split by complete work and screened for shared 24-token spans. The ordinary 8-bit first pass is frozen, and the 64-channel projection starts at zero. Run `.venv/bin/python -m loopqwen.research.selective_raw selective-raw-return-01` in an open research session. [Study and thresholds](../research/experiments/selective-raw-return-01.json) · [Source and license information](../research/licenses/SELECTIVE-PROSE-NOTICE.txt)

Rebuild the universal short-request panel and fitting messages with `.venv/bin/python scripts/unified_session_material.py`. Both available APIs and directly answerable prompts use the same two native applications. The separately archived `--session-policy research/results/shared-session-unified-01` attaches these weights to `exports/tool-language-packed-04`. All 24 held-out Dolly requests have distinct original contexts. [Registered primary and decision thresholds](../research/experiments/shared-session-unified-01.json) · [Publisher-authored instruction data](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

The original unified comparison stopped on unequal token IDs in a passage containing several scripts. Its fitted weights and 120 previously valid records are archived. The repaired installed loader selects Qwen3.5 explicitly. Reproduce the complete cold inference comparison with `.venv/bin/python -m loopqwen.research.shared_session_study shared-session-tokenizer-replay-01`. Both arms start over on the identical ordered panel. [Original input-ID receipt](../research/results/shared-session-unified-01/tokenizer-parity.json) · [Repaired comparison registration](../research/experiments/shared-session-tokenizer-replay-01.json)


For the official Qwen tool interface, rebuild the 1,244 message targets and the enrolled 58 conversations with `.venv/bin/python scripts/native_function_material.py`. The builder checks existing registered data and requires the recorded distributions for the live version targets. From an open native research window, run `.venv/bin/python -m loopqwen.research.native_function_study native-function-dialogue-01`. Recount the archive with `.venv/bin/python scripts/research_archive.py native-function-dialogue-01 --verify`. The verifier reconstructs subsequent native prompts from the recorded model tokens and independently re-executes actions against the frozen environment.


The checked-equality field study uses the original shipped tool weights and the same `--checked-calculator` for both systems. Rebuild the untouched 24-item panel with `.venv/bin/python scripts/checked_equations_material.py` and run the bounded study with `.venv/bin/python -m loopqwen.research.checked_equations checked-equations-field-01`. [Registered paired decision](../research/experiments/checked-equations-field-01.json)

The observation review study trains only after actual tool returns. Rebuild the material with `.venv/bin/python scripts/observation_review_material.py` and run `.venv/bin/python -m loopqwen.research.observation_review_study observation-review-01` inside an open research session. The 48 free arithmetic questions and 24 receipt probes are distinct. The replay reconstructs every official Qwen turn and checks the extra attention site only after a real callback. Run `.venv/bin/python scripts/observation_review_cli.py` to build the research wheel twice and exercise it outside the checkout.

The document-frequency reader uses the pinned CPython ZIP and source-located SQuAD training answers. Rebuild its complete documents and training references with `.venv/bin/python scripts/document_frequency_material.py`. Run the registered pair with `.venv/bin/python -m loopqwen.research.document_frequency_study document-frequency-reader-01`. The full original Git listings and lockfile can be reconstructed from their actual Git objects. The runtime ranks physical source intervals while the recurrent decoder processes the entire original token stream. It exposes the selected interval and each retained cache length.

For the question-contrast fit, run `.venv/bin/python scripts/question_contrast_material.py` to rebuild the 300 paired goals, replay examples and complete field prompts. The script verifies the original DROP distribution, SQuAD data and numerical program results. Run `.venv/bin/python -m loopqwen.research.question_contrast question-contrast-01` in an open session. The archive verifier checks all DROP answers against the original independent validations and reconstructs the exact sampled training-token exposure and both sets of free tool interactions.

The independent conversation-prefill crossover uses the released two-pass tool assistant to read the entire original project review and plotting source, then check size and line count through actual file callbacks. The opt-in private snapshot belongs to one caller's conversation. On each decode the runtime compares the token prefix and graph identity, restores the longest matching aligned state, and retains separate histories for both visits to the shared final layer. The preregistration requires exact output streams on all six user messages as well as grounded answers and measured time and memory. [Crossover registration](../research/experiments/conversation-document-prefill-01.json)

The crossover passed all seven registered gates, including six exact complete trajectories, with a 4.6% increase in peak MLX allocation. The licensed direct field comparison uses the same retained state for both decoders, Qwen's native template and complete public-domain text files, including original notices. Its opening prompts are 158,693 and 16,212 tokens. The first question about each work and both independent reference answers come from NarrativeQA. Each subsequent file operation is checked against the exact original bytes. [Frozen protocol](../research/experiments/book-conversation-field-01.json) · [Publisher dataset](https://github.com/google-deepmind/narrativeqa)

## Read a quantity or compute it from the same report

The report-action experiment initializes two shared applications of Qwen layer 23 from the pinned instruction sibling. Native MLX trains the same rank-32 attention adapters at both applications. The frozen ordinary BF16 Qwen Base receives identical opening IDs and the same actual calculator. The first pair includes human questions about 24 original reports, eight held-out package inventories and the original public and live-service retention requests. [Registered data, thresholds and resource ceiling](../research/experiments/report-action-field-01.json)

From a research checkout with release 05 and the pinned original model snapshots present, these commands reproduce the source selection and run the bounded pair. The published archives can also be verified independently of a new fit.

```sh
.venv/bin/python scripts/report_action_material.py
.venv/bin/python scripts/report_distribution_bytes.py --fetch
.venv/bin/python scripts/research_session.py
.venv/bin/python -m loopqwen.research.report_actions report-action-field-01
```

The first command verifies the publisher's original reports against pinned SHA-256 hashes and compares the regenerated examples with the frozen research data. The second reads the original wheel bytes and their embedded package metadata. The training output under `outputs/research/report-action-field-01` includes weights, counts, optimizer and preparation time, token-level trajectories, callback observations and the gate decision. To verify the published records, run `.venv/bin/python scripts/research_archive.py report-action-field-01 --verify`. The offline inference kit includes both sets of weights and runs the installed model on its bundled questions. [Offline kit instructions](FIELD_KIT.md)

## Argument binding in natural questions and file operations

The interpolation study ran the trained final instruction layer twice on every token, with half of the previous update and 0.175 of its second residual. The independent role-binding fit starts from the original instruction sibling and trains a fresh two-pass adapter. Its minibatches contrast the original targets of paired human questions and include executed file returns. Both comparisons use the pinned ordinary Base, identical native thinking prompts and fully recorded callbacks. Each registration specifies its domains, output allowance, resource ceiling and stop rule.

```sh
.venv/bin/python scripts/interpolation_material.py
.venv/bin/python scripts/binding_material.py
.venv/bin/python scripts/research_archive.py instruction-interpolation-01 --verify
.venv/bin/python scripts/research_archive.py role-binding-01 --verify
.venv/bin/python scripts/binding_source_check.py
.venv/bin/python scripts/report_distribution_bytes.py --keyed-inventories
.venv/bin/python scripts/interpolation_activation.py
.venv/bin/python scripts/role_binding_reading.py
.venv/bin/python scripts/role_binding_exposure.py
.venv/bin/python scripts/paper_instruction_interpolation.py
.venv/bin/python scripts/paper_role_binding.py
```

The first two commands reconstruct the training and test data from pinned sources. The archive verifiers replay each model’s history and execute every recorded tool call. The source checks compare the held-out labels with the publisher's original SQLite database and the recorded wheel hashes. Adding `--fetch` to the distribution check downloads and measures all 16 original files. The activation check uses native MLX and an independent prompt. The paper build puts both supplements and the historical 64-page publication in one bookmarked PDF.

### Compact recurrent reader on Wikipedia revisions

The qualified six-layer entailment adapter descends from the original Qwen Base. The compact reader fuses its 46 trained maps, packs all dense and embedding weights at eight bits, and reuses that same block on every token. The 48-question study freezes the original excerpt, claim, prompt IDs, output limit and direct source revision. The registered unit is a whole article with two opposite human judgments. Its [registration](../research/experiments/revision-evidence-01.json), [source-only reading](../research/diagnostics/revision-source-reading.json) and [upstream edit check](measurements/revision-wikipedia-sources.json) are public in the checkout.

Run these commands from a research checkout with the original field release and dependencies installed. The first command rebuilds the selection against the pinned publisher distribution. The second reproduces the independent source check from the saved actual Wikipedia diffs. Add `--fetch` to revisit Wikimedia directly. The third independently grades both saved token streams and verifies each full-attention cache length.

```sh
.venv/bin/python -m scripts.revision_material
.venv/bin/python -m scripts.revision_source_check
.venv/bin/python scripts/research_archive.py revision-evidence-01 --verify
```

The following command performs a two-model comparison on a caller-supplied passage using the separately saved compact model. The explicit Base path uses the certified original weights from kit 01. The first token and wall measurements for both arms include loading. The input file is UTF-8 text containing the excerpt and the question.

```sh
.venv/bin/loopqwen generate --checkpoint exports/compact-revision-reader-01 \
  --compare-base --base-checkpoint exports/tool-language-packed-05 \
  --thinking --trace --max-tokens 768 --prompt-file question.txt
```

To reproduce the compact model in a new directory, run the native conversion below. The standalone loader verifies each saved file against the export manifest. The project paper build includes an item-level CSV and an article-paired figure. The production wheel can be built and exercised outside the checkout with `scripts/verify_compact_reader.py`. The script creates an empty Python environment, installs locked local wheels, blocks network access, and compares both versions of the first registered article with the archived output IDs.

```sh
.venv/bin/python -m loopqwen.compact_entailment exports/tool-language-packed-05 \
  research/results/entailment-triplets-01/model /tmp/my-compact-reader
```

### Independent requests about the same full document

A loaded fixed Qwen assistant can stage the shared tokens of several complete requests once. Each request still supplies the complete document to the tokenizer. The runtime checks the exact prefix, gives each request private recurrent and attention tails, and preserves the original disk-backed keys and values. The same API works with the ordinary-depth control and a trained shared-query assistant.

```python
from tempfile import TemporaryDirectory
from loopqwen.shared_query_assistant import SharedQueryAssistant

assistant = SharedQueryAssistant(
    "exports/tool-language-packed-05", "research/results/learner-prefix-fit-01"
)
with open("Cargo.lock", encoding="utf-8") as source:
    lock = source.read()
requests = [
    f"Here is the complete Cargo.lock file.\n\n{lock}\n\nWhich version of {name} is locked?"
    for name in ("serde", "tokio")
]
with TemporaryDirectory() as scratch:
    document = assistant.prepare_document(requests, scratch)
    replies = [
        assistant.respond(q, max_tokens=384, document_prefix=document)
        for q in requests
    ]
```

The first staging time belongs in end-to-end measurements. A branch reports both the logical prompt length and the newly processed and reused token counts. Report the cost of a cold request separately when users might never ask a second question. [Full document source and branch measurements](measurements/full-document-features.json)

## Portable checkpoints from the matched middle-span study

The source-supervised fit completed 576 updates for each arm on 1,152 ordered training turns. Both policies keep the certified BF16 Qwen donor frozen. The loop normally makes two visits to layers 12–15 and the ordinary sibling makes one. The terminal comparison also runs untouched Base and the instruction sibling. Each evaluation block is a separate process under the single native worker lease. [Fit identities and optimizer integrity](measurements/middle-depth/terminal-fit-integrity.json) · [Training costs](measurements/middle-depth/fit-work-0576.json)

The local directory `exports/middle-depth/` contains the complete model data, tokenizer and runtime wheel. Its `donor/`, `models/` and `wheels/` directories use relative links to `objects/`. Copy the whole directory with links preserved when moving it to another Mac. The `models/source-supervised/loop/` directory on its own contains the adapter and configuration. Install the project’s native dependencies on Apple Silicon. From the checkout, use the following entry points.

```sh
.venv/bin/python -m scripts.middle_depth_archive verify --directory exports/middle-depth --stage source-supervised
.venv/bin/loopqwen ask --checkpoint exports/middle-depth/donor \
  --session-policy exports/middle-depth/models/source-supervised/loop \
  --tool-format qwen --calculator --prompt "Use the calculator to subtract 45 from 218." \
  --max-turns 3 --max-tokens 512 --trace
```

For an independent installation, install the wheel from `wheels/source-supervised/` into the prepared environment and supply the relocated donor and policy paths to the same CLI. The `early`, `epoch-reading` and `source-supervised` manifests identify the 32-, 144- and 576-update models. Each manifest covers both arms. The optimizer checkpoints and RNG/order cursor remain separate from the inference package. The installation receipts identify which exact weights, contexts and callbacks have undergone cold replay. A completed fit or verified file inventory earns no answer-quality credit. [576-update archive](measurements/middle-depth/archive-source-supervised.json) · [Installed replays](measurements/middle-depth/installed) · [Standalone source-copy release and offline replay](SOURCE_COPY_MODEL.md)

Research records include the frozen source identities and rubrics, raw responses and provider observations, masked grades, and native job charges including interrupted work. The ledger distinguishes time in a stopped invocation from the optimizer work it saved. [Study registration](../research/diagnostics/middle-depth/registration.md) · [Source-linked trajectories](measurements/middle-depth/short) · [Accounting and retained updates](measurements/middle-depth/accounting)
