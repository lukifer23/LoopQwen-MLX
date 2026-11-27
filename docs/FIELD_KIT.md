# LoopQwen offline field kit

The kit contains the qualified release 05 weights, the untouched ordinary Base text weights, the installed Python runtime, six field-study archives and the technical paper. Use an Apple Silicon Mac with Python 3.12 and uv. The kit supplies the locked native MLX runtime and its Python dependencies. The model and tokenizer are local. The kit includes the original software licenses with the restored release.

From the extracted directory, verify the files, create an empty environment, install the supplied wheels and restore the model.

```sh
shasum -a 256 -c SHA256SUMS
uv venv --offline --python python3.12 .venv
source .venv/bin/activate
uv pip install --offline --no-index --find-links dependencies runtime/*.whl
python -m loopqwen.weight_transport unpack model/tool-language-packed-05.lqzip release-05
```

## Read a complete repository listing

The supplied user request contains an entire real Git listing. Its chat input is 173,129 tokens. This command runs the opt-in trained reader, then the ordinary Base from the same release. It prints both generated texts, timings and physical attention lengths. On the M3 Pro the original comparison took about seven minutes for the pair.

```sh
python -m loopqwen generate --checkpoint release-05 \
  --document-reader studies/document-frequency-reader-01 \
  --prompt-file examples/complete-tree.txt --thinking --max-tokens 160 --compare-base --trace
```

The verified reference is 7,039 bytes. The loop returned it on the archived run. Base exhausted its output budget. Repeat the command with `examples/complete-tree-next-revision.txt` for the changed file or `examples/complete-lockfile.txt` for the third case. Both complete original books, including their notices, are under `data/research_external/narrativeqa/field-books/`.

## Ask both models a human question about a complete financial report

The two included inputs are the first arithmetic question in the frozen order for each source. One contains a human question, its original table and its paragraphs. The other asks for a difference in sizes from a real package listing. To use the latter, substitute `examples/distribution-question.txt` in the same command.

```sh
python -m loopqwen ask --checkpoint release-05 --tool-format qwen \
  --tool-adapter studies/report-action-field-01 --calculator \
  --prompt-file examples/report-question.txt --reuse-prefix \
  --max-tokens 320 --max-turns 3 --compare-base --trace
```

## Ask the model about an actual file

```sh
python -m loopqwen ask --checkpoint release-05 --tool-format qwen \
  --tool-adapter studies/question-contrast-01 --file paper/README.md \
  --prompt "Use the file tool to check the size of paper/README.md in bytes." \
  --follow-up "Now check how many lines that same file has." \
  --retain-conversation-prefix --max-tokens 256 --max-turns 3 --compare-base --trace
```

The file provider reads the chosen file on disk. Supply a different text file to check your own data. The loop and Base receive the same functions and opening tokens. The trained question-contrast and observation-review weights are experimental. They lost to ordinary Base on fresh reasoning endpoints. The paper gives the registered service scores alongside a recount that handles numbers inside filenames.

Read `paper/loopqwen.pdf` for methods, figures, results and limits. The CSV supplements accompany the paper. Each included study contains its registration, full model outputs, decision and checksums. Frozen training and evaluation manifests are in `research/data/`. The source snapshot with the training and verification programs, tests and dependency lock is in `sources/loopqwen-source.tar.gz`. The registrations identify additional upstream corpora and the adapter parents.

The paired financial questions keep the original tables, paragraphs and questions from Fengbin Zhu and colleagues’ TAT-QA. The kit supplies their unmodified training and development JSON, original notices, and the complete independent software package lock used for the transfer test. The training examples translate verified published arithmetic derivations into native Qwen calculator calls. [TAT-QA paper](https://aclanthology.org/2021.acl-long.254/) · [Dataset license, CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
