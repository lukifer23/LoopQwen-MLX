# Run the four-layer model trained on its executed table actions

This experimental policy uses all 24 original Base layers and repeats the final four layers with separate histories. It adds 192 updates on its own sampled database actions and answers to the supervised four-layer checkpoint. Both stages use native MLX. The compact parent provides the shared frozen weights, tokenizer, dependency lockfile and runtime. The update contains the exact online-trained weights, source documents, two genuine versions of one held table, and the measured comparison. [Warm model installation](REENTRY_MODEL.md) · [Execution-learning methods](../paper/table-policy-learning.pdf)

Extract the compact model archive and the table-policy update into the same new directory. Verify the inventories and install the wheel in an Apple Silicon Python environment with the locked packages.

```sh
shasum -a 256 -c SHA256SUMS
shasum -a 256 -c table-policy/SHA256SUMS
uv pip install --offline --no-deps runtime/loopqwen-0.1.0-py3-none-any.whl
loopqwen ask --checkpoint checkpoint --session-policy table-policy/policy \
  --tool-format qwen --relational-db table-policy/examples/original/table.sqlite \
  --prompt-file table-policy/examples/question.txt --max-turns 3 --max-tokens 384 --trace
```

To run the changed version, use `table-policy/examples/altered/table.sqlite` with the same question and flags. The database name, schema and initial token IDs are identical. Its rows and the human-agreed answer differ. The assistant chooses its own queries, observes the real database and generates a visible answer. The same model handles ordinary text through `--prompt-file table-policy/examples/open.txt` without a database. For an original-Base comparison, extract the separate reference archive and add `--compare-base --trace`. In the repository, the complete study, paper and per-request evidence are linked from the README.
