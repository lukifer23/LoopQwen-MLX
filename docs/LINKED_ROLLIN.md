# Run the recurrent policy trained on its own tool histories

The two-visit research policy answers questions about a real table and its linked articles through the ordinary keyed-CSV interface. Both original training and scoring use the complete row or article returned by the file tool. The primary compares one fixed recurrent policy with untouched ordinary-depth Base. The archive contains every generated token and actual observation from both arms. [Registered decision and limits](../research/experiments/linked-rollin-02.json) · [Frozen source review](../research/diagnostics/linked-rollin-source-review.json)

## Run a new question

From this checkout, give the assistant a UTF-8 question and the relevant CSV files. For the bundled HybridQA material, records.csv holds table cells and links. Articles.csv maps the page ID to its full text. For an independent request, supply your own CSVs and reference their actual filenames in your question.

```sh
.venv/bin/loopqwen ask --checkpoint exports/tool-language-packed-05 \
  --session-policy research/results/linked-rollin-02 --tool-format qwen \
  --keyed-csv records.csv --keyed-csv articles.csv --prompt-file question.txt \
  --max-turns 3 --max-tokens 512 --trace --compare-base
```

The installed command loads both models in order, prints the user-visible response from each and records the prompt IDs, output IDs, callbacks, return values, timing and attention histories. To use the loop alone, omit `--compare-base`.

## Reconstruct the source panel and the native paper

The complete selected tables, linked article collections and original annotations are stored in research/sources/linked-tables. This reconstruction uses those immutable bytes, including the replacement training bundle. It checks both directions for aliases and overlapping answer passages, reopens the actual public file providers, and regenerates both presentations with the release tokenizer.

```sh
.venv/bin/python -m scripts.verify_linked_sources
.venv/bin/python -m scripts.file_evidence_checkpoints --linked-rollin-text
.venv/bin/python -m scripts.linked_rollin_exposure
.venv/bin/python -m scripts.linked_rollin_analysis report
.venv/bin/python -m scripts.paper_linked_rollin
```

The public source review fixes spelling variants, source sufficiency and one masked presentation per question. The masked judgments supplement the gold-answer analysis. The registered primary always uses the original answer. [Source attribution and licenses](../research/sources/linked-tables/README.md) · [Method and paper](../paper/linked-rollin.pdf)

## Repeat training

Use a separate checkout containing the same frozen source records, pinned dependencies and qualified release 05. From a fresh native research window, run the following commands. The final immutable registration and the training budget are carried into the local output before optimization. The checkpoints contain eight AdamW states. An interrupted local fit resumes with its stored optimizer, registered RNG draws and actual training rollouts.

```sh
.venv/bin/python -m scripts.research_session
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -u -m loopqwen.research.linked_rollin --text-disjoint
.venv/bin/python -m scripts.file_evidence_checkpoints --linked-rollin-text --record
.venv/bin/python -m scripts.linked_rollin_exposure --root outputs/research/linked-rollin-02
.venv/bin/python -m scripts.research_archive linked-rollin-02
.venv/bin/python -m scripts.research_registry --update
.venv/bin/python -m scripts.research_session --close
```

## Use the offline kit

Extract the base field kit, runtime update 04, and this research-policy add-on into separate directories. From the add-on directory, use the original runtime and restore the certified model. The base kit supplies pinned dependencies. For paper reproduction, install its paper-tools wheels and update 04’s figure-tools wheels into the same environment.

```sh
base_kit=/path/to/field-kit-01
runtime_kit=/path/to/field-update-04
uv venv --offline --no-project --python 3.12 .venv
uv --no-cache pip install --offline --no-index --python .venv/bin/python \
  --find-links "$base_kit/dependencies" "$runtime_kit"/runtime/*.whl
.venv/bin/python -m loopqwen.weight_transport unpack \
  "$base_kit/model/tool-language-packed-05.lqzip" release-05
.venv/bin/python -c 'import json, subprocess; subprocess.run([".venv/bin/loopqwen", *json.load(open("examples/1-linked_inline.json"))], check=True)'
```

The two material presentations, two software requests, two schedule requests and one lookup example follow enrollment order. The sources directory contains an incremental Git bundle. Fetch base field kit history, update 04 history and then the linked-rollin bundle into a local bare repository before cloning its main branch. The restored checkout contains the original CSV generator, public token traces, native fitting code and closed paper. To rebuild the add-on from the checkout, run `.venv/bin/python scripts/proof_curriculum_field_update.py --linked-rollin`. The routine installs a fresh offline environment, regenerates its paper from raw evidence, and runs all seven example pairs.
