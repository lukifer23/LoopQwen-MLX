# Run the two-read source-copy research model

The measured conversational weights are tracked under [source-copy-fit-01](../research/results/source-copy-fit-01). The 401,800,172-byte ZIP is a local build under `exports/`. A network-blocked clean installation reproduced all 167 generated tokens across three requests, the real calculator callback, and the attention histories. This verifies reproducible execution. See the [archive manifest](measurements/source-copy-installation/archive.json) and [offline replay](measurements/source-copy-installation/offline.json).

The archive contains the source-copy checkpoint, its quantized instruction backbone, the certified Qwen tokenizer, the matching runtime wheel, and two original training-source examples. It is a research checkpoint. The paper reports its training results and its independently registered comparison with untouched Qwen Base.

On an Apple Silicon Mac with Metal and the dependencies in the included uv lockfile, extract the archive and run these commands from its root.

```sh
uv sync --frozen --no-install-project --project runtime
uv pip install --no-deps --python runtime/.venv/bin/python runtime/*.whl

runtime/.venv/bin/loopqwen ask --checkpoint checkpoint --session-policy policy \
  --tool-format qwen --calculator --prompt-file examples/report.txt \
  --max-turns 3 --max-tokens 192 --trace

runtime/.venv/bin/loopqwen ask --checkpoint checkpoint --session-policy policy \
  --tool-format qwen --prompt-file examples/passage.txt \
  --follow-up "$(cat examples/follow-up.txt)" --max-tokens 192 --trace
```

Both responses use the same weights. The calculator executes a generated expression and returns its actual result to the model. The retained trace records each original prompt, generated token ID, tool call, return and terminal status. Benchmark scores use the separate enrolled conversations and their longer registered output budgets.

The optional untouched Base is distributed separately in `loopqwen-reentry-reference-01.zip`. Extract it into the same directory to enable `--compare-base`. The research archive records the exact Base and donor identities. The dependency environment requires about 400 MB for the quantized backbone, in addition to the tokenizer, policy and native runtime. [Model and comparisons](https://github.com/lukifer23/LoopQwen-MLX) · [Qwen donor](https://huggingface.co/Qwen/Qwen3.5-0.8B)
