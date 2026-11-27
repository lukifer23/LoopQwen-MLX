# Run the proof-supervised two-pass model

This research policy lets you ask the same trained model to reason over text or to use a real file. It uses the field kit 01 checkpoint and field update 04 runtime. The policy directory contains the learned adapter, its compiled shared layer, optimizer state and direct-comparison evidence. The paper reports which capabilities qualified.

Extract this update into a new directory. Set `kit` to the extracted field kit 01 and `previous` to the extracted update 04. On an Apple Silicon Mac with Python 3.12, run the following inside the new directory.

```sh
shasum -a 256 -c SHA256SUMS
uv venv --offline --python 3.12 .venv
uv --no-cache pip install --offline --no-index --python .venv/bin/python \
  --find-links "$kit/dependencies" "$previous"/runtime/*.whl
.venv/bin/python -m loopqwen.weight_transport unpack \
  "$kit/model/tool-language-packed-05.lqzip" release-05
```

Each JSON file in `examples/` is the exact argument array for an enrolled request. Five examples cover an identical human claim inline and in an attached file, opposite package-compatibility outcomes at the same interpreter version, and a retained table lookup. To replay all five with both the loop and certified original Base:

```sh
.venv/bin/python - <<'PY'
import json
import subprocess
from pathlib import Path
for case in sorted(Path("examples").glob("*.json")):
    args = json.loads(case.read_bytes())
    with (case.parent / (case.stem + ".trace")).open("w") as out:
        subprocess.run([".venv/bin/loopqwen", *args], stdout=out, check=True)
PY
```

For a personal CSV with a header and unique keys, give the assistant the filename, lookup key and desired field in your request. The runtime exposes the file you provide. Use ordinary plain text for an inline question.

```sh
.venv/bin/loopqwen ask --checkpoint release-05 --session-policy policy \
  --tool-format qwen --keyed-csv my-records.csv --prompt-file my-question.txt \
  --trace --compare-base --max-tokens 512 --max-turns 3
```

## Restore the measured project and rebuild the paper

From the extracted update directory, restore the source bundles in chronological order. Each fetch updates the local bare repository. Use an absolute path for each bundle.

```sh
update="$(pwd)"
git clone --bare "$kit/sources/loopqwen-research.bundle" research.git
git -C research.git fetch "$previous/sources/update.bundle" refs/heads/main:refs/heads/main
git -C research.git fetch "$update/sources/proof-study.bundle" refs/heads/main:refs/heads/main
git clone research.git research
mkdir -p research/exports
ln -s "$update/release-05" research/exports/tool-language-packed-05
uv --no-cache pip install --offline --no-index --python .venv/bin/python \
  --find-links "$kit/dependencies" "$previous"/figure-tools/*.whl "$kit"/paper-tools/*.whl
cd research
"$update/.venv/bin/python" -m scripts.proof_curriculum_exposure
"$update/.venv/bin/python" -m scripts.proof_curriculum_reading report
"$update/.venv/bin/python" -m scripts.paper_proof_curriculum
```

The per-request CSV includes actual tool arguments, observations, reasoning, final answers, timings and success under the registered scorer. The regenerated graphs and paper use the measured model streams. The whole source-checkout bundle preserves the original study data, frozen interpretation protocol and blind ratings. Full replay of source selection additionally needs the original publisher files and wheel archives.

## Reconstruct the sources and repeat training

Set up the project’s locked native environment in the restored checkout. Place the two FOLIO v0.0 JSONL files in `data/research_external/folio/` and the original e-SNLI `esnli_train_1.csv` under `data/research_external/`. Place the original ProofWriter ZIP at `data/research_external/proofwriter-V2020.12.3.zip`. Their URLs and SHA-256 values are in the two experiment registrations. The original kit contains the 44 held distribution wheels under `dependencies/` and the paper wheels under `paper-tools/`. The previous update contains the figure wheels additional to that kit. The training selector ignores packages present among the 44 held wheels. To restore the required inventories without downloading any packages, create these local links from the restored checkout.

```sh
ln -s "$kit/dependencies" exports/field-dependencies-01
ln -s "$kit/paper-tools" exports/field-paper-dependencies-01
ln -s "$previous/figure-tools" exports/field-figure-dependencies-04
```
 See the source license and provenance record. [FOLIO data](https://github.com/Yale-LILY/FOLIO/tree/5d7bb84c7edab3fb358e057d2807f19cf5cf5e2d/data/v0.0) · [Human e-SNLI training data](https://github.com/OanaMariaCamburu/e-SNLI/tree/7b585a3f077fdea899780eb0473940522ae44a2e/dataset) · [Original ProofWriter release](https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip)

Within a fresh output directory run:

```sh
python -m scripts.reasoned_evidence_material
python -m scripts.proof_curriculum_material
python scripts/research_session.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u -m loopqwen.research.proof_curriculum
python -m scripts.file_evidence_checkpoints --record --proof-curriculum
python -m scripts.proof_curriculum_exposure --study-dir outputs/research/proof-curriculum-01
```

To verify the published primary result from the restored source and dependencies, run `python scripts/research_archive.py proof-curriculum-01 --verify`. The independent proof and minimal-support solvers use the isolated environment `uv run --isolated --no-project --python 3.12 --with z3-solver==4.15.4.0 python` followed by `scripts/proof_curriculum_oracle.py` or `scripts/proof_curriculum_support.py`. Neither solver runs inside model generation.

The restored source includes byte-preserved original records for all 192 selected ProofWriter worlds, the complete annotation fields for 384 e-SNLI examples and the unmodified FOLIO development file under `research/sources/proof-curriculum/`. To repeat the independent signed proof solver from that compact copy, pass `--compact` to `scripts/proof_curriculum_oracle.py`. Reconstruction of the original sampling from the entire population still uses the pinned full upstream archives.
