# LoopQwen-MLX

A local MLX research implementation of shared recurrent depth in the official Qwen3.5-0.8B-Base text decoder. Layers 10–15 run twice at K=2. The 30 execution sites reuse 752,393,024 pretrained text parameters. Training adds 1,336,320 shared LoRA parameters to the six-layer span. Each execution site owns its own attention or Gated DeltaNet state.

## Learned full-history retrieval

The released passage reader answered **8/8** held-out lookup questions across 2K and 16K contexts. Untouched Base answered **0/8** on the same thinking prompts with the same 96-token output allowance. Total generation time was **52.2 s versus 45.1 s**. The reader adds 32,929 learned parameters to the frozen arithmetic checkpoint. All 32 archived short math answers reproduced token for token. [Reader checkpoint, prompts and outputs](research/results/attention-memory-01) · [Measured accuracy and cost](docs/CAPABILITY_SCOREBOARD.md#frozen-arithmetic-decoder-with-a-learned-passage-reader)

At **131,078 input tokens**, the same reader answered **2/2** full-history requests in **643.7 s** on an 18 GiB M3 Pro. Base answered **0/2 in 499.7 s**. The two documents contain the same question and every number in the same position. Swapping two record names moved the right answer from token 13,124 to token 98,297. The reader produced both different values. These are two versions of one synthetic document. [Full-context outputs and timing](research/results/attention-memory-128k)

On **12 natural Wikipedia reading questions** with roughly 2K tokens of same-topic paragraphs, the reader scored **2/12 exact** in 27.5 s and Base scored **4/12** in 32.6 s. The learned span overlapped the annotated source paragraph on only three requests. The released reader is a register-lookup specialist. Base remains the general reading reference. [Natural prose transfer results](research/results/prose-reading-01)

![Measured register accuracy and per-request latency at 2K, 16K and 128K](docs/measurements/reader-context.png)

Reproduce the first 2K request on Apple Silicon. The expected visible answer is `191242`.

```sh
.venv/bin/python scripts/reader_prompt.py > /tmp/loopqwen-register.txt
.venv/bin/python -m loopqwen generate --checkpoint research/results/attention-memory-01/model --thinking --prompt-file /tmp/loopqwen-register.txt --max-tokens 96
```

## Replicated full-budget arithmetic improvement

The same learned-exit checkpoint answered **52/80** word problems correctly against **46/80** for untouched Qwen Base at a common **1,024 generated tokens**. With blind credit for completed answers in prose, the totals were **58/80 versus 52/80**. It gained answers on both reserved GSM8K and SVAMP in each panel. The six wins rescued unfinished Base turns, with zero paired losses. All six shared prefixes had already calculated the correct number before the depth switch. Elapsed generation time, including recurrent replay, was **542.44 versus 592.24 seconds** on the M3 Pro. The nominal two-sided paired test on these two panels gives **p = 0.03125**. The project explored multiple earlier designs. Raw language NLL was 2.99538 versus 2.98266. [Primary pair and checkpoint](research/results/learned-exit-01) · [Independent replication](research/results/learned-exit-confirm-01) · [Blinded pooled scores](docs/measurements/answer-audits/learned-exit-pooled.json)

The model uses ordinary depth for the first 704 generated tokens. If the thought remains open, it reprocesses the entire sequence at recurrent depth and chooses subsequent tokens from the full vocabulary. To run this checkpoint, pass `--checkpoint research/results/learned-exit-01/model --thinking --max-tokens 1024` to the native `generate` command. The self-contained native export reproduces the measured token stream through a real depth switch.

## Thinking-enabled word problems

The stabilized K=2 checkpoint trained on 128 publisher-written GSM8K explanations answered **13/32** development questions correctly in **64.6 seconds**. Untouched Qwen Base K=1 answered **1/32** in **118.1 seconds** on identical prompts with a 256-token limit. The candidate produced 32 scorable answers and Base produced one. The candidate fixed 13 Base errors and lost one correct answer. Its prose-retention NLL was 2.9873 against Base at 2.9827. These 32 questions had been used in earlier development studies. [Checkpoint and per-question outputs](research/sessions/2026-11-25/gsm-rationale-01) · [End-to-end review](docs/PROJECT_REVIEW.md)

On **48 fresh public word problems**, the same adapter answered **23/48** correctly within 256 tokens in **82.6 s**, compared with **9/48 in 172.9 s** for untouched Base. With 1,024 tokens, Base reached **31/48 in 383.1 s** and the adapter reached **23/48 in 114.1 s**. All 23 of the adapter’s correct turns finished by token 167. The longer-budget promotion gate failed. [Frozen prompts and paired results](research/results/rationale-transfer-01)

Two follow-up training methods also beat Base at 256 tokens on separate 32-question cohorts. Inverse-checked supervision scored **12 vs 3** and sampled-solution preference training scored **16 vs 2**. Both trailed Base at 1,024 tokens. Giving the original adapter and Base multiple attempts within one shared 1,024-token allowance produced **20/32 vs 25/32** in **283.2 vs 468.6 seconds**. [Follow-up scores and costs](docs/CAPABILITY_SCOREBOARD.md#follow-up-evidence-on-language-retrieval-and-prefill)

A different trained checkpoint preserves the first 512 greedy Base tokens when thinking is enabled. If the thought is still open, it reprocesses the entire prefix at K=2 and finishes within the original allowance. On 48 fresh problems at 1,024 tokens it scored **27/48 vs 25/48** for Base in **353.05 vs 380.68 seconds**, including the second prefill. The gain came from two additional SVAMP answers. GSM8K tied. A blind review that credited correct visible answers written in prose gave **31/48 vs 30/48**. The result fell below the registered promotion margin. The native CLI loads this research checkpoint with `--checkpoint research/results/recurrent-deliberation-01/model --thinking --max-tokens 1024`. [Frozen paired outputs and checkpoint](research/results/recurrent-deliberation-01) · [Answer review](docs/measurements/answer-audits/recurrent-deliberation-01.json)

![Correct completed turns versus output allowance and total generation time on 48 fresh public word problems](docs/measurements/rationale-budget.png)

Run the measured adapter on a native Apple Silicon Mac with the locked environment and pinned Base checkpoint.

```sh
.venv/bin/python -m loopqwen generate --checkpoint research/sessions/2026-11-25/gsm-rationale-01/model --thinking --max-tokens 256 --prompt 'Rocky boxed 190 fights in his career.  50 percent of his fights were knockouts, and 20 percent of the knockouts were in the first round.  How many knockouts did Rocky have in the first round?
Reason briefly, then give a single number on the last line as Final answer: <number>.'
```

On the same development set with identical training exposure, the ordinary-depth adapter answered **10/32 in 53.6 s**. The recurrent adapter gained six answers and lost three against that control. Both used 32 updates and 10,083 supervised response tokens. [Matched dense checkpoint and scores](research/results/rationale-dense-01)

The arithmetic adapter alone processed **131,072 input tokens** and completed its response on the 18 GiB M3 Pro. Time to first token was **261.3 s**. It returned 13494 for an early ledger entry whose value was 965265. This earlier run established full-context capacity and recorded a failed lookup. [128K deployment record](research/results/context-rationale-128k)

For long prompts, `--prompt-file path/to/prompt.txt` reads the entire UTF-8 file. The optional `--offload-dir /path/to/scratch` stores full-precision KV prefixes on disk while keeping every position accessible to attention. [Context protocol and reproduction](docs/PROJECT_REVIEW.md)

## What recurrence contributes

On **128 unseen arbitrary document-link maps**, both models received the same explicit three-hop question and generated only the requested final number. The ordinary-depth model answered **61/128** correctly at **107 ms** median latency. The three-pass model answered **44/128** at **150 ms**. Both completed every request after identical training exposure. A one-link lookup answered 51/128 correctly. [Scoreboard](docs/CAPABILITY_SCOREBOARD.md)

Both arms of the six-hop retention study completed their 16 updates. Mixing two old-task examples per batch preserved 57/96 old-task answers versus 39/96 for the functional-only control. Both reached 21/128 on the new endpoint and zero complete paths on 78 maps with moving final steps. The parent retained 96/96, so the mixed model missed its retention requirement. [Paired comparison](research/results/functional-interleave-01)

## Can it generate a complete verified answer?

Training the shared block on 256 gold room trajectories raised complete three-hop answers from **0/64 to 64/64** on fresh maps. Its categorical third-hop score fell from 95/96 to 81/96. A frozen-block suffix adapter preserved 95/96 and generated one correct path. On public development anchors, the previous verbal transfer checkpoint at K=3 scored **1/32 on GSM8K** and **18/32 on ARC-Easy**, compared with 2/32 and 18/32 for untouched Base. GSM8K measures generated numeric answers. ARC uses normalized choice likelihood. [Capability scoreboard](docs/CAPABILITY_SCOREBOARD.md) · [Experiment ledger](docs/RESEARCH_LOG.md) · [Active research state](research/state.json)

On **96 further unseen maps**, a single-pass model trained on those same examples solved **96/96** complete trajectories (198 ms median). The recurrently trained checkpoint solved **95/96 at K=3** (447 ms) and **32/96 at K=1**. A subsequent joint state-and-response objective restored categorical retention while generating **64/64** verified room traces. On a separate zero-shot structural test, the six-hop symbolic model solved **190/192** complete permutation paths and **6/192** arbitrary successor paths.

![State accuracy across six applications and complete generated trace accuracy versus latency](docs/measurements/research-generalization.png)

## Can the model choose better training examples?

In the new document-reference comparison, the qualified parent produced **47/64** verified paths. Two generations using verified model responses reached **50/64 and 57/64**. Gold supervision reached **61/64 and 60/64** with the same update budgets. All 32 selected model responses exactly matched the gold tokens on their respective maps, and model sampling added 47.7 seconds. [Paired results and costs](docs/CAPABILITY_SCOREBOARD.md#verified-self-training-on-document-references)

On a fresh six-hop pointer cohort, two generations of learner-selected training reached **248/256** correct terminal states versus **236/256** for static selection at equal updates. When static training received the same time allowance, it reached **255/256**. An independent curriculum seed stopped after one matched round under the registered retention rule. A separate self-training trial produced **zero fully verified trajectories in 32 sampled responses**, including 17 correct final numbers. Static training remains the stronger measured six-hop operating point. [Capability scoreboard and compute accounting](docs/RSI_RESULTS.md) · [Frozen study](docs/RSI_PLAN.md)

## What another 1,152 map presentations bought

With its original optimizer restored, the seed-109 recurrent checkpoint improved from **360 to 384 correct canonical third hops out of 384**. Across eight shuffles of those same unseen maps it improved from **93.91% to 99.84%**. The paired map-bootstrap interval for the gain was **+4.10 to +7.94 percentage points**. The architecture, map set and adapter capacity stayed fixed. System memory pressure ended training at update 624 of a planned 960.

A separate 60-update pilot taught the original Qwen vocabulary head to emit the third-hop digit on **187/192** held-out verbal prompts, up from **39/192** before transfer. The categorical mechanism retained 99.35% shuffled third-hop accuracy. Prose NLL was 3.033 at K=1 and 3.488 at K=3, against untouched Base at 2.983. Free-form reasoning remains unresolved. The better of two thinking-enabled Base prompts gave five valid answers out of eight and missed the six-answer gate. [Vocabulary pilot and retention](docs/PHASE7_RESULTS.md#full-vocabulary-transfer)

The improvement carried past the trained depth. On the fixed random presentation, sixth-hop accuracy rose from **106/384 to 284/384**. Across four display-order conditions, transplanting a carried hidden vector caused the recipient to advance from the donor state on **61–62/64** cases per condition. The unmodified and matched-noise controls scored **0/64** each. This evidence applies to an eight-state categorical task on one development seed. [Phase 7 findings and gate decisions](docs/PHASE7_RESULTS.md) · [Training and reproduction](docs/TRAINING.md#phase-7-exact-exposure-continuation)

![Third-hop accuracy across eight shuffles and extrapolation through six recurrent applications](docs/measurements/phase7-exposure.png)

```sh
.venv/bin/python scripts/phase7_predict.py --map 5 4 7 6 0 2 1 3 --start 0 --depth 3
python3 scripts/verify_phase7_archive.py
```

## Generalization to shuffled pointer maps

Changing only the training display order raised third-hop accuracy from **46.9% to 93.9%** and complete three-hop paths from **43.7% to 93.5%** across eight presentations of each of 384 unseen development maps. The same recurrent checkpoint achieved **360/384 (93.75%)** on canonical third hops, five maps short of the registered 95% floor for promotion. This is a categorical prediction from Qwen's shared middle layers. The Phase 6 language-transfer and confirmation stages remain gated. [Phase 6 results](docs/PHASE6_RESULTS.md) · [Registered study](docs/PHASE6_PLAN.md) · [Reproduction](docs/TRAINING.md#phase-6-matched-installation)

The strongest dense control achieved **19.1%** on shuffled third hops. The separately scored thinking-enabled deployment pair hit its prespecified floor. On six identical questions, both the base and trained loop used all 384 tokens before emitting a final answer. Base took 34.6 seconds and the loop took 56.8 seconds. Ordinary answer quality remains unresolved. [Two-model scores](docs/measurements/phase6/reasoning-pair/scores.json) · [Independent review handoff](docs/PHASE6_REVIEW.md)

![Phase 6 measured accuracy, order sensitivity, learning curves and cost](docs/measurements/phase6-orders.png)

```sh
.venv/bin/python scripts/phase6_predict.py --arm randomized --map 5 4 7 6 0 2 1 3 --start 0 --depth 3
python3 scripts/verify_phase6_archive.py
```

[Released adapter and readout](docs/measurements/phase6/randomized) · [Full measurement archive](docs/measurements/phase6)

## Learning an internal pointer transition

With an explicit state target after each application, the same six Qwen layers learned to follow an eight-node permutation. At seed 109 the shared recurrent model predicted all three successive states on **384/384 unseen development maps**. At the same map exposure the dense middle, dense full and frozen controls completed **1, 10 and 0** paths. A shared eight-way head reads the state. This experiment used no generated tokens between passes.

Changing the order of the displayed edges reduced third-hop accuracy to **36/64**. Transplanting a hidden vector from a different map made the next prediction follow the recipient map from the donor state in **64/64** registered cases, compared with **0/64** without the transplant. A second transplant check with 64 distinct donor maps also scored 64/64. The categorical model reached 384/384 at the untrained fourth hop. In a verbal rendering, ordinary greedy three-hop digit generation scored 42/96 at K=3, 1/96 with the same weights at K=1 and 15/96 for the untouched base. Prose retention worsened at K=3. These are exploratory results on one task and one training seed. [Phase 5 results](docs/PHASE5_RESULTS.md) · [Architecture](docs/PHASE5_ARCHITECTURE.md) · [Registration](docs/PHASE5_PLAN.md)

![Pointer state and trajectory accuracy at matched exposure, with dense and recurrent controls](docs/measurements/phase5-pointer.png)

Run the compact categorical checkpoint from the repository root with the locked native environment and pinned base model.

```sh
.venv/bin/python scripts/phase5_predict.py --arm recurrent --map 5 4 7 6 0 2 1 3 --start 0 --depth 3
python3 scripts/verify_phase5_archive.py
```

[Installed adapter and state head](docs/measurements/phase5/models/recurrent) · [Terminal-only continuation](docs/measurements/phase5/models/outcome) · [Study commands](docs/EXPERIMENT_REGISTRY.md#phase-5-pointer-installation)

## Learning useful recurrent computation

Phase 4 tests two deployment questions. Can training with an extra block pass improve a checkpoint that runs at ordinary K=1 inference cost? Can explicitly supervising both exits make the second pass solve problems that its own first pass misses? The language experiments use real Qwen weights, both matched-data and matched-training-time dense controls, and the same greedy development prompts. Phase 5 now trains the shared block to predict successive states on an eight-node pointer task. Its primary comparison and registered persistence checks are complete. [Phase 5 registration](docs/PHASE5_PLAN.md) · [Categorical architecture](docs/PHASE5_ARCHITECTURE.md). [Study registration](docs/PHASE4_PLAN.md) · [Architecture and objectives](docs/PHASE4_ARCHITECTURE.md) · [Reproduction](docs/EXPERIMENT_REGISTRY.md#phase-4-experiments)

On the 18 GiB M3 Pro, the integrated 360-position physical microbatch cap reduced peak MLX allocation from 6.85 to 4.63 GiB on the matched profile. Both arms kept four examples per optimizer update. At seed 109, dense and gated training consumed the same 250,211 supervised targets. The dense control reached 334,012 targets when given the gated model’s optimizer time. [Training guide](docs/TRAINING.md)

Seed 109's 250K model solved **114/300 traced development problems at K=1**, versus **108/300** for both dense controls. Its second pass reduced that total to **112/300**. Neither answer fixed by the extra pass had a verified derivation. The two-point single-pass gain fell below the registered promotion threshold. Jointly supervising both exits at 100K also favored running once (98/300 versus 95/300). [Phase 4 results](docs/PHASE4_RESULTS.md)

![Traced answer accuracy for both 250K controls and both 100K joint exits on the same ID and OOD development items.](docs/measurements/phase4/development.png)

A reviewer can verify the published files and recount all nine traced comparisons from a fresh clone with `python3 scripts/verify_phase4_archive.py`. This uses the Python standard library and requires no model download. [Per-item records](docs/measurements/phase4/eval/primary) · [Paired results](docs/measurements/phase4/primary-pairs.json)

## Phase 3 development result

The development study compares dense adaptation, direct recurrence and a near-identity refinement at the same 100,068 supervised tokens. A single learned sigmoid controls the extra pass. Every recurrent checkpoint also runs at true K=1 so its adapters can be evaluated without the extra computation. The primary outcome is greedy final-answer accuracy on traced problems. Exact procedural certificates, independent language retention and measured latency provide additional evidence. [Registration and reproduction](docs/EXPERIMENT_REGISTRY.md) · [Architecture](docs/PHASE3_ARCHITECTURE.md)

At 250,031 matched supervised tokens, the learned-gate checkpoint solved **61/150 fresh ID and 47/150 fresh OOD development problems**. Bypassing its second pass produced the same two totals. On each split the extra pass fixed one answer and broke one. The matched dense checkpoint scored 58/150 and 41/150. This seed suggests a possible training-time benefit. It gave us no net final-answer gain from running the extra six layers at inference. The registered promotion rule kept the three-seed confirmatory test sealed. [Dense adapter](adapters/phase3/p3_dense_250k) · [Learned refinement adapter](adapters/phase3/p3_learn1_250k) · [Development results](docs/PHASE3_RESULTS.md)

![Counts of correct answers on fresh ID and OOD problems and paired differences against the dense and bypass controls.](docs/measurements/phase3/recurrent-contribution.png)

## Run the seed-109 model

On a native arm64 Mac, install the locked environment and resolve the pinned Base checkpoint. The 250K gated adapter's best observed traced accuracy on this cohort came at K=1.

```sh
./scripts/bootstrap.sh
source .venv/bin/activate
python -m loopqwen download
python -m loopqwen generate --checkpoint adapters/phase4/gated_109_250k --passes 1 --prompt 'Evaluate (((4 - 13) - 14) * 3).
Show your steps. End with Final answer: <number>.
Solution:' --max-tokens 192
```

The K=1 adapter generated the following verified solution for that prompt during the batched development evaluation.

```text
4 - 13 = -9
-9 - 14 = -23
-23 * 3 = -69
Final answer: -69
```

Use `--passes 2` to run the extra pass. The matched dense model is in `adapters/phase4/dense_109_250k`. Earlier published adapters and their measurements remain available in the [experiment registry](docs/EXPERIMENT_REGISTRY.md).

## Reproduce the study

The [experiment registry](docs/EXPERIMENT_REGISTRY.md) lists the ordered commands from data preparation through training, evaluation and archival. The [training guide](docs/TRAINING.md) explains checkpoints and update budgets. The [performance report](docs/PHASE3_OPTIMIZATIONS.md) records historical deployment measurements on the 18 GiB M3 Pro. The [Phase 4 trainer profile](docs/PHASE4_OPTIMIZATIONS.md) reports target throughput, memory and checkpoint cost.

Measured figures include the [development learning curves](docs/measurements/phase3/learning-curves.png) and [memory over 1,605 gated updates](docs/measurements/phase3/training-memory.png).

## Shared block and independent caches

```mermaid
flowchart LR
    P["Embedding and layers 0–9"] --> A["F: layers 10–15, cache set A"]
    A --> B["Same F and LoRA, cache set B"]
    A --> M["h₁ + α (candidate − h₁)"]
    B --> M
    M --> S["Layers 16–23, norm and tied head"]
    A -. "K=1 bypass" .-> S
```

## Historical experiments

The 1,000-update loop answered 59/110 held-out procedural questions correctly. The same checkpoint at K=1 answered 55/110, and the matched dense twin reached 65/110. The experiment compares the untouched base, the untrained loop and two matched trained twins. Its primary outcomes are held-out answer loss and greedy numeric accuracy. Prose retention, variable depth and a longer-output dense control are also measured. [Experimental results](docs/RESULTS.md)

## Try the trained loop

The repository includes small inference adapters for both training pairs. From a clone, run

```sh
./scripts/bootstrap.sh
source .venv/bin/activate
python -m loopqwen download
python -m loopqwen generate --checkpoint adapters/extended_loop --prompt 'Question: 17 × 23
Answer:' --max-tokens 16
```

## Reproduce the pilot

Use a native arm64 macOS Python environment and run these commands from the repository root. Bootstrap uses uv when available and otherwise creates a Python 3.12 venv with constraints from the lockfile. The default dataset builder downloads small pinned public math splits. The model download uses the official Base checkpoint.

```sh
./scripts/bootstrap.sh
source .venv/bin/activate
python -m loopqwen download
python -m loopqwen baseline
python -m loopqwen probe --start-layer 10 --num-layers 6 --passes 2
python -m loopqwen cache-check
python -m loopqwen gradient-check
python -m loopqwen gdn-check
./scripts/run_pilot.sh
python -m loopqwen benchmark --config configs/benchmark.yaml
python -m pytest -q
python -m loopqwen export --checkpoint checkpoints/pilot_loop --output exports/loopqwen-0.8b
python -m loopqwen generate --checkpoint exports/loopqwen-0.8b --prompt 'What is 17 times 23?' --max-tokens 32
```

The pilot script builds data, trains both 12-update smoke runs, trains both 200-update pilots and runs evaluation. A completed run name reuses its latest checkpoint. A clean checkout reproduces training from the official weights. The full integration suite expects the completed pilot and its manifest.

## Individual commands

```sh
python -m loopqwen doctor
python -m loopqwen data --preset pilot
python -m loopqwen train --config configs/smoke.yaml --until 6
python -m loopqwen train --config configs/smoke.yaml
python -m loopqwen train --config configs/pilot_dense.yaml
python -m loopqwen train --config configs/pilot_loop.yaml
python -m loopqwen evaluate --experiment pilot
python -m loopqwen compare --experiment pilot
python -m loopqwen report --experiment pilot
python -m loopqwen generate --checkpoint checkpoints/pilot_loop --prompt 'Question: 17 × 23
Answer:' --max-tokens 16
```

The base model uses raw text. The adaptation examples use a question followed by a newline and `Answer:`. Generation is greedy, capped and EOS-aware. Adapter checkpoints require the pinned base snapshot. The BF16 merged export includes its own tokenizer and recurrent runtime, with a fresh-process check on every logit of the export verification prompt.

## Reproduce the extended comparison

The 1,000-update pair starts from the same pretrained base, using four documents per update and the existing pilot splits. In the activated environment, run the following commands after building the data.

```sh
python -m loopqwen train --config configs/extended_dense.yaml
python -m loopqwen train --config configs/extended_loop.yaml
python -m loopqwen evaluate --experiment extended
python -m loopqwen compare --experiment extended
python -m loopqwen export --checkpoint checkpoints/extended_loop --output exports/loopqwen-extended-0.8b
```

## Project files

| Path | Contents |
|---|---|
| `configs/` | Hardware presets, matched training runs, benchmark and evaluation settings |
| `loopqwen/looped_model.py` | Shared execution schedule, interpolation and tied output head |
| `loopqwen/recurrent_cache.py` | Independent execution-site caches |
| `loopqwen/kernels/gdn_backend.py` | Validated native training path and stock fallback |
| `loopqwen/training/` | Adapter placement, token-weighted loss, optimizer and trainer |
| `loopqwen/checkpoint.py`, `loopqwen/export.py` | Atomic resumable generations and merged export |
| `data/pilot/` | Local tokenized examples, split digests and source manifest |
| `checkpoints/<run>/latest.json` | Pointer to the most recent complete generation |
| `outputs/<experiment>/` | Each generated answer, decoding settings, scores and Markdown comparison |
| `docs/` | Measured results, implementation details and reproduction notes |

The code license is MIT. Official Qwen weights retain their own license, copied into merged exports. GSM8K and MATH source licenses are recorded beside the downloaded data. Full backbone weights, merged exports, local optimizer checkpoints and raw datasets stay outside Git. The four compact adapter bundles are included. [Architecture](docs/ARCHITECTURE.md) · [Training](docs/TRAINING.md) · [Metal benchmarks](docs/METAL_OPTIMIZATIONS.md) · [Troubleshooting](docs/TROUBLESHOOTING.md) The natural reading panel adapts SQuAD v1.1 by Rajpurkar and colleagues and the contributing Wikipedia authors. It reorders complete paragraphs from each named article and distributes those transformed data under CC BY-SA 4.0. [Publisher and source download](https://rajpurkar.github.io/SQuAD-explorer/) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

## Phase 2 recurrent-depth study

V2 uses 24,000 distinct procedural problems with verified derivations, a separate answer-only prompt, held-out difficulty regimes and independent prose retention data. The historical pilot and extended runs retain their original scoring protocols. The [registered design](docs/PHASE2_PLAN.md), [architecture audit](docs/PHASE2_ARCHITECTURE.md), [Metal investigation](docs/PHASE2_OPTIMIZATIONS.md) and [first V2 development comparison](docs/measurements/phase2/phase2_direct/comparison.md) describe the new study.

From the project root on native Apple Silicon:

```sh
uv sync --locked --group figures
.venv/bin/python -m loopqwen download
.venv/bin/python -m loopqwen data --preset v2
.venv/bin/python -m loopqwen train --config configs/phase2/p2_a_trace_dense.yaml
.venv/bin/python -m loopqwen train --config configs/phase2/p2_a_trace_loop.yaml
.venv/bin/python -m loopqwen compare --experiment phase2_direct
```

`scripts/run_phase2.sh` runs all registered controls, development selection, three confirmatory seeds and diagnostics sequentially. Use a separate checkout for an independent reproduction. In the original workspace, identical configurations resume using their saved optimizer and deterministic cursor. The `figures` dependency group supplies Matplotlib. Group-level evaluation requires every registered checkpoint to be complete. The comparison command can also read the archived table in a fresh clone.

For fast checks independent of model weights or Metal, use `python -m pytest -q tests/test_phase2.py -k 'numeric or gold_and_fingerprints'`. The remaining Phase 2 checks require the locked environment and built V2 corpus. The historical integration suite also requires the original pilot dataset and completed pilot checkpoints. The expensive three-seed experiment runs through the study script separately from pytest.

`python -m loopqwen resources` lists project processes, their parent PIDs, the current exclusive GPU lease, OS memory pressure and swap occupancy. A standalone model worker acquires the lease before loading weights. The study launcher runs one coordinator per checkout and forwards cancellation to its process group, allowing an active trainer to save at an update boundary. A lock file may remain on disk after exit. The resources command checks the kernel lock to determine whether a worker owns it.

The [completed three-seed Phase 2 report](docs/PHASE2_RESULTS.md) includes ID and OOD accuracy, uncertainty, measured compute controls, prose retention, export parity and memory measurements. The [seed-43 recurrent adapter](adapters/p2_confirm_43_loop) and its [matched dense twin](adapters/p2_confirm_43_dense) load directly with the generation CLI. Subsequent study reproductions write an independent archive under `outputs/phase2/reproduction/` once the original results have a checksum index.
