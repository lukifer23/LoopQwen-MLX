# LoopQwen project review, November 25

The deployment question is whether a person gets more correct, completed answers from the trained decoder than from the pinned Qwen3.5-0.8B-Base, at a measured cost, while retaining ordinary language and useful long context. Establishing that repeating layers deserves credit requires a second comparison with a trained ordinary-depth model. These questions have produced different answers in this repository. [Capability scoreboard](CAPABILITY_SCOREBOARD.md)

## What each phase established

| Evidence | Finding | Consequence for deployment |
|---|---|---|
| Pilot and extended training | With 4,000 matched training examples, the recurrent adapter answered 59/110 procedural questions correctly and its dense twin answered 65. The extra pass fixed four net answers over the same adapter at ordinary depth. | A same-checkpoint depth gain and the best use of the training budget can point to different models. [Extended study](EXTENDED.md) |
| Three-seed Phase 2 | Direct recurrence averaged 23.7% on out-of-distribution traced problems versus 32.6% for matched dense adaptation. The paired interval for the difference was −16.3 to −2.2 percentage points. | Blindly repeating a pretrained span is a poor general default. [Phase 2 results](PHASE2_RESULTS.md) |
| Gated arithmetic, Phases 3 and 4 | Two development seeds supported a small training-time effect. At seed 109, ordinary-depth inference from the recurrently trained checkpoint scored 114/300, its second pass scored 112 and both dense controls scored 108. The second pass cost additional time. | The small development gains missed the registered promotion threshold. A stabilized arithmetic parent is a plausible substrate for a language experiment. [Phase 3](PHASE3_RESULTS.md) · [Phase 4](PHASE4_RESULTS.md) |
| Explicit state supervision, Phases 5–7 | The shared middle layers learned a permutation transition. Randomized presentation and continued training brought third-hop performance to 3,067/3,072 shuffled presentations of 384 maps. Hidden-state transplant controls show a causal role for the carried state. | The operation is real on the covered distribution. Sixth-hop complete paths fell from 190/192 on permutations to 6/192 on arbitrary successor maps. [Mechanism results](PHASE7_RESULTS.md) · [Structural transfer](CAPABILITY_SCOREBOARD.md#zero-shot-structural-transfer) |
| Generated paths and replay | On 96 fresh maps, trained K1 solved 96 complete traces at 198 ms median. Trained K3 solved 95 at 447 ms. Adaptive six-hop data selection won at equal updates and lost to static training at matched time. | Publish generated answers, exact paths and synchronized cost together. Correct intermediate states alone cannot establish a faster or better language model. [Generated paths](CAPABILITY_SCOREBOARD.md#same-requested-trace-fresh-96-map-cohort) · [Adaptive selection](RSI_RESULTS.md) |
| Public language anchors | The earlier verbal checkpoint scored 1/32 versus 2/32 on GSM8K and tied Base on ARC-Easy. Equation-only training later produced 26 completed numeric answers against zero for thinking-enabled Base, with one correct answer. | Training the answer boundary can help completion. Accuracy and independent prose retention determine whether that helps the user. [Public comparison](../research/results/external-anchor-01) · [Equation outputs](../research/results/gsm-equations-01) |

## Why test full history and language adaptation together

The checkpoint supports 262,144 positions in its text configuration. At 131,072 tokens and batch one, its six full-attention sites need 1.50 GiB for BF16 keys and values alone. Repeating layers 10 through 15 once adds two full-attention sites and raises that payload to 2.00 GiB. Both figures exclude model weights, temporary buffers, cache growth and convolutional/DeltaNet state. Changing a context-limit constant would buy no memory. [Pinned Qwen configuration](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/blob/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/config.json)

RULER evaluates retrieval, multi-hop tracing and aggregation across lengths. Its results motivate measuring full prompt ingestion, retrieval of an early fact, and reasoning over distant evidence independently. The registered ledger probe covers the first two on one item. Its full token stream is archived so the two storage paths and two model depths can be compared exactly. [RULER paper](https://arxiv.org/abs/2404.06654) · [Benchmark implementation](https://github.com/NVIDIA/RULER)

Geiping and colleagues trained a 3.5B recurrent-depth model on 800B tokens and found gains from additional inference computation. LoopQwen adapts six existing layers in a much smaller hybrid decoder. The pretraining result motivates the mechanism, while this repository’s dense twins and same-weight bypasses measure whether the retrofit earns its extra passes. [Recurrent-depth paper](https://arxiv.org/abs/2502.05171) · [Training and inference code](https://github.com/seal-rg/recurrent-pretraining)

GSM8K supplies human-written multistep solutions. The language adaptation uses complete publisher explanations during training, including units and the relationship between quantities, then scores one freely generated first turn. Hsieh and colleagues studied rationales as supervision for smaller models. Their data and multitask method differ from this response-only experiment. [Publisher dataset](https://github.com/openai/grade-school-math) · [Rationale supervision](https://arxiv.org/abs/2305.02301)

## What counts as progress

The primary language pair fixes ordered questions, exact prompt token IDs, the opening thinking marker, greedy generation, output cap and scorer. The newly trained checkpoint is compared with untouched ordinary-depth Base. Accuracy, paired gains and losses, actual generation seconds, produced tokens and retention on 9,328 independent language targets are measured together. The 32 development questions were already used in earlier studies. Improvement there needs a fresh sample for confirmation. Unknown Base pretraining overlap limits any claim of unseen public questions. [Frozen language protocol](../research/experiments/gsm-rationale-01.json) · [Question and training identities](../research/data/gsm-rationale-01.json)

The scored 128K pair uses identical input IDs for both depths with full-history caches at every attention site. The disk-backed implementation writes intermediate activations as BF16 bits, stores each completed site's prefix in its original dtype, and reads the prefix during decode. It trades storage I/O for lower resident cache memory. All model inference and training use native MLX on Metal. The 259-token preflight compares every vocabulary logit across four positions before committing to long prefill. [Full-history protocol](../research/experiments/context-128k-01.json) · [Disk-backed protocol](../research/experiments/context-128k-disk.json)

FlexGen schedules tensor movement among GPU, host and disk for high-throughput batched inference. Here the host and GPU share the same memory, so moving a cache from Metal to a resident CPU array would still consume unified memory. The implementation stores completed KV prefixes on disk and measures the cost of reading them on every decode step at batch one. [FlexGen paper](https://arxiv.org/abs/2303.06865) · [MLX-LM project](https://github.com/ml-explore/mlx-lm)

The fresh panel uses 24 of Scale AI’s 50 publicly released GSM1K examples and 24 SVAMP examples, six per arithmetic operation. The same adapter and scorer face a 1,024-token ceiling. This directly tests whether additional time for Base to finish changes the deployment conclusion. The shorter 256-token comparison is read from the same greedy streams, with its elapsed time measured during generation. The publisher datasets are public, so prior exposure during Qwen pretraining is unknown. [Registered transfer comparison](../research/experiments/rationale-transfer-01.json) · [GSM1K public release](https://github.com/scaleapi/gsm1k_eval) · [SVAMP study](https://aclanthology.org/2021.naacl-main.168/)

## What fits on the 18 GiB M3 Pro

Both resident-cache runs processed the same 131,072 input IDs and generated all 96 allowed output tokens. The target number appeared about 13,100 tokens into the ledger. Both generations were still in the thinking span at the limit and neither contained the target. MLX peak allocations are process measurements. The operating system and filesystem cache also use unified memory. [Prompt, emitted IDs and timings](../research/sessions/2026-11-25/context-128k-01)

| Resident checkpoint | Time to first token | Total time | Prefill tokens/s | Decode tokens/s |
|---|---:|---:|---:|---:|
| Untouched Base K1 | 192.44 s | 195.97 s | 681 | 26.9 |
| Phase 4 arithmetic parent K2 | 261.75 s | 266.53 s | 501 | 19.9 |
| Deployed 32-update rationale K2 | 261.27 s | 262.37 s | 502 | 19.1 |

The actual deployed rationale adapter also consumed every input token. It completed a 22-token answer and returned 13494. The early ledger entry was 965265. This directly establishes capacity on the trained weights and records an incorrect retrieval even when the answer finished within the allowance. [Trained checkpoint at 128K](../research/results/context-rationale-128k)

The original run stopped on a system memory warning after processing 112,768 Base input tokens. After Chrome closed, available memory reached 9.48 GiB. The completed test used the original registration, prompt and allocation guard. The working context limit therefore depends on other applications on this machine. [Initial interruption](../research/results/context-128k-01) · [Recovery record](../research/sessions/2026-11-25/resumption.json)

### Disk-backed memory and speed

| Checkpoint and storage | Peak MLX allocation | Temporary disk | First token | Total for 96 generated tokens | Decode tokens/s |
|---|---:|---:|---:|---:|---:|
| Base K1 resident | 3.78 GiB | 0 | 192.44 s | 195.97 s | 26.9 |
| Base K1 disk-backed | 2.22 GiB | 1.75 GiB | 232.13 s | 248.79 s | 5.7 |
| Phase 4 K2 resident | 4.32 GiB | 0 | 261.75 s | 266.53 s | 19.9 |
| Phase 4 K2 disk-backed | 2.59 GiB | 2.25 GiB | 315.45 s | 338.21 s | 4.2 |

Both measured storage modes produced identical sequences of 96 output IDs at each depth. The disk-backed preflight also produced bit-identical full-vocabulary logits at four positions on a 259-token prompt at both depths. The disk-backed path saves 40% of peak MLX allocation on K2 and takes 27% longer for this complete request. During decode the disk path rereads all stored prefixes for every generated token. The table reports the MLX peak since the start of each study worker, which includes earlier arms and parity checks. Temporary directories are removed when the request exits. [Parity, per-arm timings and outputs](../research/sessions/2026-11-25/context-128k-disk)

### Recount the archived deployments

From the checkout root, these commands use Python's standard library to verify checkpoint identities, exact prompts, emitted first turns, timing aggregates and the registered decisions. Native inference additionally needs the pinned checkpoint and locked Apple Silicon environment.

```sh
python3 scripts/research_registry.py
python3 scripts/research_archive.py gsm-rationale-01 --verify --recovered-session 2026-11-25
python3 scripts/research_archive.py context-128k-01 --verify --recovered-session 2026-11-25
python3 scripts/research_archive.py context-128k-disk --verify --recovered-session 2026-11-25
python3 scripts/research_archive.py context-rationale-128k --verify
python3 scripts/research_archive.py rationale-transfer-01 --verify
python3 scripts/research_archive.py rationale-dense-01 --verify
```

## Which model to use for these word problems

The trained recurrent adapter improved correct first turns under 256 output tokens on all three measured groups. It scored 13/32 on reused GSM8K development questions, 7/24 on fresh public GSM1K and 16/24 on fresh SVAMP. Base scored 1, 1 and 8 respectively. Accuracy at 1,024 tokens favored Base on both fresh sources. The candidate finished all 23 of its correct fresh answers within 167 tokens, and Base first matched that count at token 482. This crossover describes these archived items. Both sources remain frozen against further tuning. [Budget curve and per-source scores](CAPABILITY_SCOREBOARD.md#fresh-word-problems-at-two-output-budgets)

Most of the short-budget development gain is also available at ordinary depth. The dense twin trained on the identical 128 publisher explanations delivered 10/32 correct turns in 53.56 s, versus 13/32 in 64.64 s for K2. K2 gained six paired answers and lost three, meeting the prespecified two-answer and 1.25× time criteria. The exact two-sided paired p-value is 0.508 on nine discordant items. That modest difference combines training and deployment depth and comes from a single lineage. The V3 retention NLL differed by less than 0.00004 between trained models. [Matched dense evidence](../research/results/rationale-dense-01)
