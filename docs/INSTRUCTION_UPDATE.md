# Recurrent field update 02

This add-on contains two experimental policies, the complete research report and the evidence for both comparisons. Both policies run the trained final attention layer twice on every input and generated token. The two executions share all seven fused projections and maintain separate attention histories. The add-on uses the original field kit’s weights and dependency wheels. The first study in the report covers the newly fitted model. The second covers the interpolated model.

## Install and replay

On an Apple Silicon Mac with Python 3.12 and uv, unpack the original field kit and this update in separate directories. Set the paths below to their extracted roots. The example assumes a new environment and a writable update directory.

```sh
field=/absolute/path/to/extracted-field-kit
update=/absolute/path/to/extracted-update
cd "$update"
shasum -a 256 -c SHA256SUMS
uv venv --no-project --offline --python 3.12 .venv
uv --no-cache pip install --offline --no-index --python .venv/bin/python --find-links "$field/dependencies" runtime/*.whl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -m loopqwen.weight_transport unpack "$field/model/tool-language-packed-05.lqzip" release-05
.venv/bin/python -c 'import json, subprocess; subprocess.run([".venv/bin/loopqwen", *json.load(open("examples/reference.json"))], check=True)'
.venv/bin/python -c 'import json, subprocess; subprocess.run([".venv/bin/loopqwen", *json.load(open("examples/inventory.json"))], check=True)'
```

Each example runs both the recurrent system and the certified ordinary Base on identical opening tokens. One asks a human-authored pronoun question. The other has two user messages, exposes a real CSV and calculator, and retains each model's own conversation history. Both examples were selected by frozen panel order. Use `--keyed-csv PATH` to expose a different UTF-8 document. The caller supplies the file. An exact, unique key identifies the return field. Transcript files inside `studies/instruction-interpolation-01/` include emitted tokens, observed errors and successful returns for every request.

The add-on includes the panel, original material, runtime and incremental Git bundle based on the first kit's source history, the residual intervention, and the paper tables, source and figure. The package inventories come from the project's pinned uv.lock. Human sentences, candidate referents and labels come from the Collection of Winograd Schemas assembled by Ernest Davis, Leora Morgenstern and Charles Ortiz. Individual examples retain their source attribution in the panel. [Collection and original sources](https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html)

## Trained argument-binding model

The update also includes a second, independently trained rank-32 adapter. It runs the full learned update on the same recurrent layer for both ordinary conversation and file requests. The current report in `paper/loopqwen-complete.pdf` opens with this experiment and includes both field supplements and the original technical paper. Two additional installed examples reproduce the first fresh meaning question and first held-out table question.

```sh
.venv/bin/python -c 'import json, subprocess; subprocess.run([".venv/bin/loopqwen", *json.load(open("examples/meaning.json"))], check=True)'
.venv/bin/python -c 'import json, subprocess; subprocess.run([".venv/bin/loopqwen", *json.load(open("examples/table.json"))], check=True)'
```

The file in `sources/winogrande-1.1.zip` is AI2's WinoGrande 1.1 distribution. Credit Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula and Yejin Choi. The publisher licenses the dataset under CC BY 2.0. The selected examples retain their original identifiers and sentences. Our prompts present the two provided choices in a fixed order per pair. [WinoGrande project](https://github.com/allenai/winogrande) · [Dataset license](https://creativecommons.org/licenses/by/2.0/)

The original WikiSQL questions, programs and tables come from Victor Zhong, Caiming Xiong and Richard Socher at Salesforce Research. We transcribed complete selected source tables to CSV and used their original SQL to verify the requested cells. The archive includes the upstream notice and original data package. [WikiSQL project](https://github.com/salesforce/WikiSQL)

## Full-context release regression

The update also carries two complete historical register prompts. Running the original release 05 default through the new runtime replays its long-input behavior. Each prompt has 131,084 input tokens before decoding. The expected output IDs, all global attention histories, and native packed-attention counters accompany the documents. The short experimental policies have their own decision rules and context qualifications.

```sh
.venv/bin/loopqwen ask --checkpoint release-05 --prompt-file examples/register-0.txt --max-tokens 96 --trace
.venv/bin/loopqwen ask --checkpoint release-05 --prompt-file examples/register-1.txt --max-tokens 96 --trace
```

## Restore the source and evidence

The update bundle is incremental against the Git history in the original field kit. From the extracted update directory, the following reconstructs the source revision, including code, study registrations, fitted adapter, scored trajectories and paper.

```sh
git clone "$field/sources/loopqwen-research.bundle" restored-project
git -C restored-project fetch "$update/sources/field-update.bundle" refs/heads/main
git -C restored-project merge --ff-only FETCH_HEAD
```
