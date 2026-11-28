# Running the terminal BF16 model pair

The research checkpoint repeats native Qwen layers 12–15 twice at serving time. The comparison model is the untouched 24-layer Qwen3.5-0.8B-Base text decoder. Each directory contains every model file, the tokenizer and the native runtime wheel. The loop directory includes its four-epoch trained weights. Both use the ordinary vocabulary head. You can move either directory by itself and run it offline with the pinned Apple Silicon Python environment.

From a checkout with the project dependencies installed, fetch the original Qwen snapshot once. The public restoration kit supplies the trained coefficients and layout information. Restoration checks the official shard SHA-256 and the complete hash of each final file. Each destination must be a new directory.

```sh
.venv/bin/hf download Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --local-dir data/standalone-qwen-base
.venv/bin/python -m scripts.middle_depth_portable restore --snapshot data/standalone-qwen-base --arm base --out exports/standalone-middle-576/base
.venv/bin/python -m scripts.middle_depth_portable restore --snapshot data/standalone-qwen-base --arm loop --out exports/standalone-middle-576/loop
```

In the project environment, run either model directly. These are separate processes. Each command exits and releases its model before the next one begins.

```sh
.venv/bin/python -m loopqwen ask --checkpoint exports/standalone-middle-576/base --tool-format qwen --calculator --prompt 'Use the calculator to subtract 18 from 35.' --max-tokens 512 --trace
.venv/bin/python -m loopqwen ask --checkpoint exports/standalone-middle-576/loop --tool-format qwen --calculator --prompt 'Use the calculator to subtract 18 from 35.' --max-tokens 512 --trace
```

The native reproducibility check installs the bundled wheel in isolation outside the checkout, denies network access and loads each model once for all ten requests. It records tokens, executed calls, observations, exit status, elapsed time, peak allocation and the operating-system PID. It compares each response against the previously graded trace. This is a repeatability check on development data. Run it on an idle worker with the study ledger and frozen grades available.

```sh
.venv/bin/python -m scripts.middle_depth_standalone run
```

The cold replay matched all 20 reference trajectories. Base exited 1.58 seconds before the loop process launched. Each model loaded once. The table includes model loading and process teardown in the cold time. Both models missed every supported answer on these four sources and their follow-ups. One earlier Base installation spent 70.02 worker-seconds and stopped because the audit treated a Python namespace package as a regular file. The successful run verified both package types. [Paired timings and trace hashes](measurements/middle-depth/standalone/pair.json) · [First attempt](measurements/middle-depth/charges/dac82bfec4cf43b2b55bd565434bd5fc.json)

| Offline model | Matched responses | Supported answers | Load time | Cold process time | Peak MLX |
|---|---:|---:|---:|---:|---:|
| Untouched Base | 10/10 | 0/10 | 1.72 s | 71.81 s | 1.66 GiB |
| Trained loop | 10/10 | 0/10 | 1.77 s | 18.32 s | 1.66 GiB |

The four-epoch short panel gave 5/27 supported answers for the loop and 2/27 for untouched Base at 512 generated tokens. Weight-only Q4 made the loop faster and smaller but changed actual decisions. BF16 remains the direct-pair reference. [Weight-only quantization](measurements/middle-depth/quantization/short-decision.json) · [Restoration kit and identities](../research/results/middle-depth-576)

On the first published-edition Anne opening, the trained loop returned a 27-token response to 145,891 actual input tokens in 301.0 seconds cold (27.1 seconds with the prefix ready), peaking at 3.73 GiB MLX. Untouched Base reached its 128-token output limit in 275.3 seconds cold (28.5 seconds warm), peaking at 3.43 GiB. Their opening token IDs were identical. Both resident caches reproduced the exact staged vocabulary logits at two consecutive positions. Source-masked review of both complete editions credited 0/2 for each model at 128 tokens. At 512, Base supported the published answer and missed the edit (1/2). LoopQwen scored 0/2. The loop retrieved both ages in both editions and stopped before reporting their difference. [Source-level grades](measurements/middle-depth/document-review/parent/books/anne-512.json). [Registered document panel and trace hashes](measurements/middle-depth/document-panels/parent.json) · [Paper draft, complete books](../paper/paper-draft.pdf)

## Local jobs and time to a correct result

We compared the two restored directories in four fresh processes on the M3 Pro. One job required reading optimizer pointers and answering a rollback follow-up. The other required querying the actual file inventory and comparing the backbone hashes. The providers could supply all four answers. The loop executed eight callbacks. One returned the inventory schema, six returned no records and one returned a calculation error. Base executed zero. Neither completed a job or answered a request correctly. Across four requests the loop used 12.27 seconds and Base used 24.27. Charging unsuccessful requests the registered 30-second deadline yielded 30 seconds per request for both. On a separate four-question GSM8K sample, the loop answered one correctly and Base zero. Raw operational data and replies remain local. [Prospective protocol](../research/diagnostics/middle-depth/local-workflows.md) · [Aggregate result](measurements/middle-depth/local-workflow-screen.json)

The subsequent screen compared these same two directories on two full QMSum meetings, four changed GSM8K questions and an operator handoff that required real Python execution and a CSV follow-up. LoopQwen answered both versions of one math problem. Base answered both versions of zero. Neither model delivered a supported meeting summary or completed the operator job, including when genuine stdout was supplied. [Aggregate scores and native timings](measurements/middle-depth/specialist-result.json)

## Benchmarking the selected standalone pair

The full-document loop gets its own self-contained directory named by its trained-weight hash before the short and book development suites run. After source-masked scoring selects a checkpoint, the following command benchmarks the selected directory and the untouched Base directory outside the checkout. It installs each directory's bundled wheel offline and waits for each model process to exit. Both generation allowances use the same nine short sources, their changed and irrelevant versions, and two real follow-ups. The complete-book primary measurements and specialist screens have separate receipts. [Frozen benchmark and stop rules](../research/diagnostics/middle-depth/standalone-benchmark.md)

```sh
.venv/bin/python -m scripts.middle_depth_standalone benchmark
.venv/bin/python -m scripts.middle_depth_pair_audit chosen
```

The update-576 audit reconstructed both four-family scores from the locked grades and hash-checked trajectories. All 78,732 source resamples give the same 512-token percentile interval as the registered Monte Carlo estimate, −9.72 to +15.97 percentage points. Omitting one source moved the estimated effect from −4.17 to +8.33 points. The 27 short openings used 333,676 layer visits across processed token positions for LoopQwen and 652,200 for Base. The loop generated 1,504 tokens to Base’s 11,063. Mean time to a correct short answer, charging failures eight seconds, was 6.65 versus 7.58 seconds. On the complete books it was 600 versus 521 seconds at the 600-second deadline. [Audited accuracy, confidence intervals, costs and source sensitivity](measurements/middle-depth/direct/parent-audit.json)

Once the selected 512-token Anne opening and own-history references exist, run the full-prefix cold check separately for each installed model. It reads the complete published book, serves the calculator follow-up through the public CLI and records every actual callback and the OS-observed time to first stdout.

```sh
.venv/bin/python -m scripts.middle_depth_cold --standalone --arm base --source anne --cap 512
.venv/bin/python -m scripts.middle_depth_cold --standalone --arm loop --source anne --cap 512
```

## Five public benchmarks

The standard suite uses the official zero-shot completion definitions and metrics in lm-evaluation-harness 0.4.13. The standalone native MLX process receives label-free context/continuation pairs. An independent process gives the resulting likelihoods back to the harness for grading. The source files are small pinned Parquet downloads. The isolated harness environment keeps its dependencies out of the native training environment. [Frozen tasks, external Base scores, statistics and budget](../research/diagnostics/middle-depth/standard-five.md) · [Enrolled split indices](measurements/middle-depth/standard-five/enrollment.json)

```sh
uv venv --python .venv/bin/python data/standard-five/env
uv pip install --python data/standard-five/env/bin/python 'lm-eval==0.4.13'
data/standard-five/env/bin/python -m scripts.middle_depth_bench_sources freeze
.venv/bin/python -m scripts.middle_depth_bench pair
```

Run the pair on an idle native worker. It checks the complete model manifests, opens the Base folder with its installed wheel, finishes and regrades all 250 rows, then opens the loop folder in a new process. Native scores are saved after each item. The two grade files and the paired report are under `docs/measurements/middle-depth/standard-five`.

## Restoring the loop trained on complete documents

The four-document fit has a standalone 1,530,361,024-byte checkpoint. It has completed the 16 registered updates. The repository stores the change from the certified update-576 directory in 1,639,404 additional compressed bytes. Restore to a new directory using the update-576 loop created above. Every restored file must match the new hash manifest. The command was verified in a separate temporary directory. [Full-document fit](measurements/middle-depth/long-fit-loop.json) · [Manifest](measurements/middle-depth/standalone/long-loop-files.json) · [Restoration receipt](../research/results/middle-depth-full-document/changes.json)

```sh
.venv/bin/python -m scripts.middle_depth_portable restore-full-fit --parent exports/standalone-middle-576/loop --out exports/restored-full-document-loop
```
