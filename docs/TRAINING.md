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
| `future-token-prose-01` | `loopqwen.research.selective_raw` | Shared Q/K/V/O at two prediction horizons, paired source-level inference and six complete cache replays |
| `session-tokenizer-02` | `loopqwen.research.session_tokenizer` | Four cold release-04 repetitions of the same 50 requests with 15 switches, measuring exact outputs and tokenizer identity |
| `repository-tree-field-01` | `loopqwen.research.narrative_book` | Two genuine project object listings, independently verified PDF sizes and complete native history above 169K |
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
