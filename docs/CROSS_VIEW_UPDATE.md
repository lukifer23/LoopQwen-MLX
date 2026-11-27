# Run a trained cross-view reader on your own file

This update contains one small recurrent student’s compiled reading layer, trainable adapters, the original research outputs, exact teacher targets and a complete paper. The same student generates actions, interprets returned passages and answers ordinary text. Its two passes share the final layer. Kit 01 supplies its frozen instruction backbone, the ordinary Qwen Base comparator and the locked Apple Silicon wheels.

Extract kit 01 and update 04 in separate directories. Start from the update directory and use the local kit path.

```sh
kit=/absolute/path/to/field-kit-01
shasum -a 256 -c SHA256SUMS
uv venv --no-project --offline --python 3.12 .venv
uv --no-cache pip install --offline --no-index --python .venv/bin/python \
  --find-links "$kit/dependencies" runtime/*.whl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -m loopqwen.weight_transport unpack "$kit/model/tool-language-packed-05.lqzip" release-05
.venv/bin/python -c 'import glob,json,subprocess; p=sorted(glob.glob("examples/case-*-file.json"))[0]; subprocess.run([".venv/bin/loopqwen", *json.load(open(p))], check=True)'
```

For another source, supply a CSV with a header and unique record key. The assistant receives the declared field names, generates a real call, and receives the selected value. Its answer, actual call arguments and executed returns appear in the trace. The ordinary comparator receives the same opening token IDs and tool.

```sh
.venv/bin/loopqwen ask --checkpoint release-05 --session-policy policy --tool-format qwen \
  --keyed-csv your-notes.csv --prompt-file your-question.txt --trace \
  --compare-base --max-tokens 640 --max-turns 3
```

For an inline reading, omit `--keyed-csv` and include the passage in the question file. The five packaged commands cover both forms of two original human questions and a real table lookup. The most recent long-context qualification applies to the release 05 default. The paper identifies the checkpoint used in each study.

## Research reproduction

The two source bundles reconstruct the exact project tree in the field receipt. Supply the absolute path of the update bundle to the fetch command.

```sh
git clone --bare "$kit/sources/loopqwen-research.bundle" research.git
git -C research.git fetch ../sources/update.bundle refs/heads/main:refs/heads/main
git clone research.git research-source
```

Set `py` below to the interpreter installed above. From the restored source root, put release 05 under `exports/tool-language-packed-05/`. Copy the supplied `sources/squad-train-v1.1.json` and `sources/quoref/train-dev.zip` under `data/research_external/`, preserving their relative paths. Set `update` to the extracted update directory. To repeat the optimizer using the genuine sealed teacher distributions, run the following in a fresh checkout.

```sh
study=cross-view-reading-01
mkdir -p "outputs/research/$study"
cp "research/experiments/$study.json" "outputs/research/$study/registration.json"
cp "$update/policy/teacher.json.gz" "$update/policy/teacher-summary.json" "outputs/research/$study/"
"$py" -m scripts.cross_view_material
"$py" -m scripts.cross_view_source_check
"$py" -m scripts.cross_view_exposure --study-dir "outputs/research/$study" --output "outputs/research/$study/teacher-audit.json"
"$py" -m scripts.research_session
"$py" -u -m loopqwen.research.cross_view_study
"$py" -m scripts.cross_view_exposure --study-dir "outputs/research/$study" --output "outputs/research/$study/exposure.json"
"$py" -c 'from pathlib import Path; from loopqwen.io import read; from loopqwen.research.cross_view_study import verify; p=Path("outputs/research/cross-view-reading-01"); verify(p,read(p/"decision.json"))'
```

To independently repeat teacher inference, start with a fresh output directory, run `"$py" -m scripts.cross_view_sources` with network access once, then execute the study offline. That command fetches the pinned 3.03 GB four-billion-parameter checkpoint. Only SQuAD training answers enter the teacher. The student and the installed Base comparison work without the teacher checkpoint.

## Rebuild the figures and paper from the original answers

The restored source contains the frozen predictions, human reference spans, blind review labels and the earlier 83-page report. With the two source archives and release 05 in the locations above, install the update’s plotting wheels and the original kit’s typesetting wheels. The paper build reads both model traces, recalculates the article-level intervals and writes the tables, SVG figures, request-level CSV, separate manuscript and consolidated PDF under `paper/`.

```sh
uv --no-cache pip install --offline --no-index --python "$py" \
  --find-links "$kit/dependencies" "$update"/figure-tools/*.whl "$kit"/paper-tools/*.whl
"$py" -m scripts.cross_view_material
"$py" -m scripts.cross_view_source_check
"$py" -m scripts.cross_view_exposure
"$py" -m scripts.cross_view_reading
"$py" -m scripts.paper_cross_view
shasum -a 256 paper/loopqwen-study.pdf
```
