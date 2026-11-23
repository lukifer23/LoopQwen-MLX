# LoopQwen-MLX

A local MLX research implementation of shared recurrent depth in the official Qwen3.5-0.8B-Base text decoder. Layers 10–15 run twice at K=2. The 30 execution sites reuse 752,393,024 pretrained text parameters. Training adds 1,336,320 shared LoRA parameters to the six-layer span. Each execution site owns its own attention or Gated DeltaNet state.

The experiment compares the untouched base, the untrained loop and two matched trained twins. Its primary outcomes are held-out answer loss and greedy numeric accuracy. Prose retention, variable depth and a longer-output dense control are also measured. [Experimental results](docs/RESULTS.md)

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

The code license is MIT. Official Qwen weights retain their own license, copied into merged exports. GSM8K and MATH source licenses are recorded beside the downloaded data. Model weights, checkpoints and raw datasets stay outside Git. [Architecture](docs/ARCHITECTURE.md) · [Training](docs/TRAINING.md) · [Metal benchmarks](docs/METAL_OPTIMIZATIONS.md) · [Troubleshooting](docs/TROUBLESHOOTING.md)
