# Reproduce the grounded field-selection study

The experiment starts from the qualified release 05 backbone. It uses a shared rank-32 update on the final layer and executes that layer twice per token. The held data include cross-domain human table judgments, real distribution requirements and retained file lookups. The decision requires gains in answer quality and speed on the same trained weights. [Frozen protocol](../research/experiments/field-selection-01.json) · [Analysis protocol](../research/diagnostics/field-selection-analysis.json)

## Restore the sources

Use the pinned environment from the repository root. The full InfoTabS and evidence archives are stored outside Git in the research cache. The source builder checks their SHA-256 digests before sampling. The human agreement counts, selected original values, package sidecars and final frozen prompt IDs are already present in the checkout.

```sh
mkdir -p data/research_external/infotabs data/research_external/wikisql
curl --fail --location https://codeload.github.com/infotabs/infotabs/zip/34e7dc0900a0d861ad62be3d7371d8eb0aace4fe -o data/research_external/infotabs/source.zip
curl --fail --location https://zenodo.org/api/records/6578593/files/annotation.zip/content -o data/research_external/infotabs/annotation.zip
curl --fail --location https://raw.githubusercontent.com/salesforce/WikiSQL/cffb423077756d04c1bac5bcd45167c86903fbcb/data.tar.bz2 -o data/research_external/wikisql/source.tar.bz2
.venv/bin/python -m scripts.field_selection_packages
.venv/bin/python -m scripts.field_selection_material
.venv/bin/python -m scripts.field_selection_oracle --acquire
```

The independent oracle opens each original wheel as a ZIP and checks the name, version, declared Python constraint and the complete embedded distribution metadata. It also executes the registered queries in an in-memory SQLite database. Subsequent runs of the oracle can use the local wheel cache.

## Run the native pair

Restore `exports/tool-language-packed-05` from the project's base field kit. Use an independent checkout and empty study output and optimizer directories for a new measurement. The runner preserves the actual stream order, registers a native resource lease, and refuses to rerun a closed comparison into its original output directory.

```sh
.venv/bin/python scripts/research_session.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -u -m loopqwen.research.field_selection
.venv/bin/python -m scripts.file_evidence_checkpoints --record --field-selection
.venv/bin/python -m scripts.field_selection_exposure --root outputs/research/field-selection-01
.venv/bin/python scripts/research_archive.py field-selection-01
.venv/bin/python scripts/research_session.py --close
```

## Audit the measurements and figures

The original first-token timing includes model startup on the first request. Running on a different Mac yields different timings. Every output token, real observation, generated argument and optimizer exposure can be inspected independently.

```sh
.venv/bin/python scripts/research_archive.py field-selection-01 --verify
.venv/bin/python -m scripts.file_evidence_checkpoints --field-selection
.venv/bin/python -m scripts.field_selection_exposure
.venv/bin/python -m scripts.field_selection_analysis blind
.venv/bin/python -m scripts.field_selection_analysis report
.venv/bin/python -m scripts.paper_field_selection
```

The blinded responses and reading protocol support independent raters. The report uses the archived original ratings to reproduce the published analysis. The direct pair cannot isolate the causal contribution of recurrent depth. The separate 128K release and matched-depth induction studies retain their own registrations and weights.
