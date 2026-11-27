# Run the four-layer recurrent research model

This distribution contains the actual Base-derived four-bit backbone, the four learned second-visit residuals, the tokenizer, a native MLX runtime wheel, the training paper and a database example. The checkpoint passed its supervised training check on 32 predetermined turns. A comparison of general assistance quality requires the separately trained and evaluated policy. The included original CoSQL prompt comes from training and exercises the installation.

Use an Apple Silicon Python environment with the versions in the enclosed project lockfile. Extract the recurrent archive into an empty directory and run the following commands there.

```sh
shasum -a 256 -c SHA256SUMS
uv pip install --offline --no-deps runtime/loopqwen-0.1.0-py3-none-any.whl
loopqwen ask --checkpoint checkpoint --session-policy policy --tool-format qwen \
  --relational-db examples/customers_card_transactions.sqlite \
  --prompt-file examples/database-question.txt --max-turns 3 --max-tokens 128 --trace
```

For a new request, pass your own text or SQLite file. With an empty tool catalog the same learned decoder answers ordinary questions. The cache-state option advances the full attention histories at intermediate prompt blocks.

```sh
loopqwen ask --checkpoint checkpoint --session-policy policy --tool-format qwen \
  --prompt 'Explain how to compare two fractions with different denominators.' \
  --max-tokens 384 --max-turns 1 --cache-state-prefill
```

The separate reference archive contains the original 320 untouched BF16 text tensors and their certified inventory. Extract it in the same directory, verify `reference/SHA256SUMS` from the root, then add `--compare-base --trace` to either command. This runs the two real local decoders at the same initial token IDs and output allowance. Recurrent generation alone uses only the smaller recurrent archive. The original weight and code licenses are included. The training database is from CoSQL and Spider. [Qwen Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) · [CoSQL](https://yale-lily.github.io/cosql) · [Spider](https://yale-lily.github.io/spider)
