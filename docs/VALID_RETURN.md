# Reproduce the successful-return preference comparison

The study measures whether the repeated final layer can choose between authentic facts about the same source, recover from an irrelevant return and deliver a grounded judgment. Its fresh human cohort comes from the ordinary-domain InfoTabS test split. Separate practical checks use the original metadata of eight Python distributions and the published FOSDEM 2025 program. The two-visit candidate and ordinary-depth Base answer the same 112 tokenized prompts, with real callbacks and identical generation limits. [Training and decision protocol](../research/experiments/valid-return-01.json) · [Fixed blinded reading](../research/diagnostics/valid-return-reading.json)

## Use the policy on a local agenda

Extract field kit 01, update 04 and the preference-policy add-on into separate directories. From the unpacked policy directory, install the runtime and unpack the model using the original kits.

```sh
base_kit=/path/to/field-kit-01
runtime_kit=/path/to/field-update-04
uv venv --offline --no-project --python 3.12 .venv
uv --no-cache pip install --offline --no-index --python .venv/bin/python \
  --find-links "$base_kit/dependencies" "$runtime_kit"/runtime/*.whl
.venv/bin/python -m loopqwen.weight_transport unpack \
  "$base_kit/model/tool-language-packed-05.lqzip" release-05
.venv/bin/loopqwen ask --checkpoint release-05 --session-policy policy \
  --tool-format qwen --keyed-csv agenda.csv --resource calendar \
  --prompt-file question.txt --max-turns 3 --max-tokens 512 --trace --compare-base
```

Supply a CSV with a header and a question about one of its entries. The trace includes both models’ generated requests, actual tool results, token IDs and physical histories. The seven bundled examples use two human-question formats, two software constraints, one retained lookup and two questions about the same conference session. The selected examples follow fixed enrollment order.

## Verify the sources

From the research checkout, use the pinned Python environment. Acquisition of the original InfoTabS and annotation archives is documented in the previous study. The builders verify source hashes before use. [Original public archive commands](FIELD_SELECTION.md#restore-the-sources)

```sh
.venv/bin/python -m scripts.agenda_sources --acquire
.venv/bin/python -m scripts.valid_return_material
.venv/bin/python -m scripts.field_selection_provenance --valid-return
.venv/bin/python -m scripts.valid_return_oracle --acquire
```

For offline verification, restore `sources/update.bundle` from field update 04 and then `sources/valid-return.bundle` onto the base kit's bare source repository. Clone its explicit `main` branch. Set `exports/tool-language-packed-05` in the checkout to the unpacked release. Place `schedule-sources/*` in `data/research_external/fosdem/` and `source-wheels/*` in `data/research_external/field_selection/wheels/`. The source replay reads all enrolled human ballots and the complete published conference XML, calendars and Python wheels. The full InfoTabS sampling archives are only needed to repeat selection from the entire population.

```sh
.venv/bin/python -m scripts.field_selection_provenance --valid-return --compact
.venv/bin/python -m scripts.agenda_sources
.venv/bin/python -m scripts.valid_return_oracle
.venv/bin/python -m scripts.verify_field_selection_frozen --root research/results/valid-return-01
```

## Reproduce the native study and audit the report

Use a clean output and optimizer location in a separate checkout to repeat the original training. The runner captures its source revision and checks the native time and memory budgets throughout preparation, fitting and evaluation. The archived run retains the starting odds on every preference pair, trained weights and final optimizer moments.

```sh
.venv/bin/python scripts/research_session.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -u -m loopqwen.research.valid_return
.venv/bin/python -m scripts.file_evidence_checkpoints --record --valid-return
.venv/bin/python -m scripts.valid_return_exposure --root outputs/research/valid-return-01
.venv/bin/python scripts/research_archive.py valid-return-01
.venv/bin/python scripts/research_registry.py --update
.venv/bin/python scripts/research_session.py --close
```

The independent reader and publication routines also operate on the archived study.

```sh
.venv/bin/python -m scripts.file_evidence_checkpoints --valid-return
.venv/bin/python -m scripts.valid_return_exposure
.venv/bin/python -m scripts.valid_return_analysis blind
.venv/bin/python -m scripts.valid_return_analysis report
.venv/bin/python -m scripts.valid_return_ballots
.venv/bin/python -m scripts.paper_valid_return
```

The report uses the original blinded ratings. The ordered-events audit retains every wrong but successful return. A weekday call made before the correct date lookup cannot satisfy the registered requirement. The additional message-level sensitivity also excludes calendar calls issued in the same assistant message as the original lookup. Timing and peak allocation are original measurements on one M3 Pro. The separate portable kit verifier starts from an empty model cache and rebuilds the paper and seven installed pairs with network access blocked.
