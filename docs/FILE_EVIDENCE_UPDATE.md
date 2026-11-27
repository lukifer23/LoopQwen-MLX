# Run the changed-file assistant offline

This update supplies the newly trained two-layer adapter, the compact six-layer reader's reconstruction inputs, the source sentences and real edit records, the field measurements and the consolidated paper. Both experimental models use the original Qwen Base weights. The update includes the first attempt’s memory-limit decision, the second fit’s native gradient comparison and the exact inputs used for its successful optimizer steps. The held-out test was reserved across both registrations. Kit 01 supplies those weights and the locked native dependencies. The ordinary comparator uses its certified BF16 text tensors. Use an Apple Silicon Mac with Python 3.12 and uv.

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
.venv/bin/loopqwen ask --checkpoint release-05 --session-policy studies/file-evidence-accumulated-01 \
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

To verify training exposure from the restored checkout, place the supplied publisher files under `data/research_external/vitaminc/` and unpack release 05 under `exports/tool-language-packed-05/`. Run `.venv/bin/python -m scripts.file_evidence_exposure` using the installed native environment. This independently replays the registered sample order and teacher tokenization and checks the published total.

The two original VitaminC distributions in `sources/vitaminc` include the project training split and untouched publisher test split. Their original licenses and the sampled Wikipedia edit provenance accompany them. The study uses different articles and claims for training and testing. The earlier field update retains its original instructions and receipts.

## Repeat the native fit and the direct comparison

Use a fresh clone of the restored Git history so its checkout and local compute ledger belong to this run. Set `py` to the absolute path of the Python interpreter from the offline installation above. From the cloned repository root, place release 05 in `exports/tool-language-packed-05/` and copy the supplied original VitaminC train and test files into `data/research_external/vitaminc/`. The registered training and test material, controls, original human labels and study decisions are tracked in Git. The commands below rebuild the material, open a bounded session, fit new weights, run both models and check the new run’s optimizer and exposure. They can take tens of minutes on an M3 Pro.

```sh
py=/absolute/path/to/the/offline/.venv/bin/python
study=file-evidence-accumulated-01
"$py" -m scripts.file_evidence_material
"$py" -m scripts.research_session
"$py" -u -m loopqwen.research.file_evidence "$study"
"$py" -m scripts.file_evidence_checkpoints --record
"$py" -m scripts.file_evidence_exposure --study-dir "outputs/research/$study"     --output "outputs/research/$study/exposure.json"
"$py" -c 'from pathlib import Path; from loopqwen.io import read; from loopqwen.research.file_evidence import verify; p=Path("outputs/research/file-evidence-accumulated-01"); verify(p,read(p/"decision.json"))'
```

The newly measured weights, trajectories, decision and wall times are under `outputs/research/file-evidence-accumulated-01/`. The published run stays under `research/results/`. Its `environment.json` names the original execution revision and its precise source hashes. The 720 optimizer updates used a fixed sample order on the training articles. Comparisons across installations should check raw scores and receipts separately from timing, which varies with the device and host load.
