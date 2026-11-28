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

The four-epoch short panel gave 5/27 supported answers for the loop, 4/27 for the trained ordinary sibling and 2/27 for untouched Base at 512 tokens. The untouched instruction sibling scored 6/27. In the weight-only Q4 trial both fitted models decoded the same fixed positions more than twice as fast. The Q4 loop lost its one supported archive decision. Across nine explicit tool openings, successful provider returns fell from six to zero for the loop and five to zero for ordinary depth. BF16 is the reproducible research reference. The full-document quality thresholds remain open. [Frozen scoring](measurements/middle-depth/review/terminal/summary.json) · [Precision decision](measurements/middle-depth/quantization/short-decision.json) · [Restoration kit and identities](../research/results/middle-depth-576)
