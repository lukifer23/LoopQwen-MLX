# Run the changed-file assistant offline

This update supplies the newly trained two-layer adapter, the compact six-layer reader's reconstruction inputs, the source sentences and real edit records, the field measurements and the consolidated paper. Both experimental models use the original Qwen Base weights. Kit 01 supplies those weights and the locked native dependencies. The ordinary comparator uses its certified BF16 text tensors. Use an Apple Silicon Mac with Python 3.12 and uv.

Extract kit 01 and update 03 into separate directories. From the update directory run the commands below, replacing the kit path with its extracted location.

```sh
kit=/absolute/path/to/field-kit-01
shasum -a 256 -c SHA256SUMS
uv venv --no-project --offline --python 3.12 .venv
uv --no-cache pip install --offline --no-index --python .venv/bin/python \
  --find-links "$kit/dependencies" runtime/*.whl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -m loopqwen.weight_transport unpack "$kit/model/tool-language-packed-05.lqzip" release-05
.venv/bin/python -c 'import json,subprocess; subprocess.run([".venv/bin/loopqwen", *json.load(open("examples/file-0.json"))],check=True)'
.venv/bin/python -c 'import json,subprocess; subprocess.run([".venv/bin/loopqwen", *json.load(open("examples/file-1.json"))],check=True)'
```

These two commands pass exactly the same question to independent sessions on the recurrent model and ordinary Base. The single changed cell is inside the two supplied copies of `clippings.csv`. Each CLI response displays the actual prompt tokens, emitted tokens, arguments, returns, final answer and costs. For a different CSV, identify a unique key and target field in the question. The runtime exposes exact header names and opens the caller's file at execution.

```sh
.venv/bin/loopqwen ask --checkpoint release-05 --session-policy studies/file-evidence-01 \
  --tool-format qwen --keyed-csv your-file.csv --prompt-file your-question.txt \
  --max-tokens 640 --max-turns 3 --trace --compare-base
```

For the standalone six-layer reader, reconstruct the eight-bit weights once. The command uses the actual trained adapter and the certified original checkpoint. A free comparison can then use any UTF-8 input question.

```sh
.venv/bin/python -m loopqwen.compact_entailment release-05 compact-source compact-reader
.venv/bin/loopqwen generate --checkpoint compact-reader --compare-base --base-checkpoint release-05 \
  --thinking --trace --prompt-file your-question.txt --max-tokens 768
```

The default release has separate long-context qualification. Its full registered examples are `examples/register-0.txt` and `examples/register-1.txt`. Omit the experimental session policy for these requests. Each is 131,084 actual input tokens after native chat rendering.

```sh
.venv/bin/loopqwen ask --checkpoint release-05 --prompt-file examples/register-0.txt --max-tokens 96 --trace
```

To rebuild the paper without a GPU, copy the original 64-page PDF from kit 01, install its two local typesetting wheels, and verify both assembly stages. Each stage independently typesets the supplements and checks byte identity of the PDFs.

```sh
cp "$kit/paper/loopqwen.pdf" paper/loopqwen.pdf
uv --no-cache pip install --offline --no-index --python .venv/bin/python "$kit"/paper-tools/*.whl
.venv/bin/python scripts/assemble_field_paper.py paper --verify
.venv/bin/python scripts/assemble_evidence_paper.py paper --verify
```

The full project source, measurements and commit sequence are available offline through the original Git bundle and this update's incremental bundle. The current update applies directly to kit 01.

```sh
git clone --bare "$kit/sources/loopqwen-research.bundle" loopqwen-source.git
git -C loopqwen-source.git fetch ../sources/field-update.bundle refs/heads/main:refs/heads/main
git clone loopqwen-source.git research-checkout
```

The two original VitaminC distributions in `sources/vitaminc` include the project training split and untouched publisher test split. Their original licenses and the sampled Wikipedia edit provenance accompany them. The study uses different articles and claims for training and testing. The earlier field update retains its original instructions and receipts.
