# LoopQwen offline field kit

The kit contains the qualified release 05 weights, the untouched ordinary Base text weights, the installed Python runtime, five field-study archives and the technical paper. Use an Apple Silicon machine with the project’s Python 3.12 MLX environment. The model and tokenizer are local. The kit includes the original software licenses with the restored release.

From the extracted directory, verify the files, install the wheel into that environment and restore the model.

```sh
shasum -a 256 -c SHA256SUMS
uv pip install --offline --no-deps runtime/*.whl
python -m loopqwen.weight_transport unpack model/tool-language-packed-05.lqzip release-05
```

## Read a complete repository listing

The supplied user request contains an entire real Git listing. Its chat input is 173,129 tokens. This command runs the opt-in trained reader, then the ordinary Base from the same release. It prints both generated texts, timings and physical attention lengths. On the M3 Pro the original comparison took about seven minutes for the pair.

```sh
python -m loopqwen generate --checkpoint release-05 \
  --document-reader studies/document-frequency-reader-01 \
  --prompt-file examples/complete-tree.txt --thinking --max-tokens 160 --compare-base --trace
```

The verified reference is 7,039 bytes. The loop returned it on the archived run. Base exhausted its output budget. The second revision and a complete lockfile are documented in the paper.

## Ask the model about an actual file

```sh
python -m loopqwen ask --checkpoint release-05 --tool-format qwen \
  --tool-adapter studies/question-contrast-01 --file paper/README.md \
  --prompt "Use the file tool to check the size of paper/README.md in bytes." \
  --follow-up "Now check how many lines that same file has." \
  --retain-conversation-prefix --max-tokens 256 --max-turns 3 --compare-base --trace
```

The file provider reads the chosen file on disk. Supply a different text file to check your own data. The loop and Base receive the same functions and opening tokens. The trained question-contrast and observation-review weights are experimental. They lost to ordinary Base on fresh reasoning endpoints. The paper gives the registered service scores alongside a recount that handles numbers inside filenames.

Read `paper/loopqwen.pdf` for methods, figures, results and limits. The CSV supplements accompany the paper. Each included study contains its registration, full model outputs, decision and checksums. The source checkout contains the training and verification commands.
