# Where shared decoder depth earns its cost

The manuscript covers the original shared-span experiments, the matched reasoning controls, the standalone 128K language model, and the addition of executable tool sessions. The paper's numerical results come from native Apple Silicon MLX runs.

- [Read the paper](loopqwen.pdf)
- [Manuscript source](loopqwen.typ)
- [Machine-readable paired scores and timings for the frozen tool model](typed-transfer.csv)
- [Training lineage, optimizer exposure and checkpoint hashes](model-lineage.csv)
- [Median and tail tool latency and first-token time](tool-latency.csv)
- [Trajectory-level costs, calls and failure types](tool-trajectories.csv)
- [Study-level evidence index](study-index.csv)
- [Every registration, threshold, budget and publication status](registry-index.csv)
- [SHA-256 inventories behind the figures](figure-sources.json)

## Rebuild and verify

From the repository root, the following commands rebuild the vector figures, evidence index and PDF. The verification option checks stored predictions and per-file hashes against the independent archive recounts. Some older verifiers require the pinned tokenizer in the local Hugging Face cache.

```sh
uv sync --group figures --group paper
.venv/bin/python scripts/build_paper.py --verify
```

Each row in `study-index.csv` points to a checked-in measurement directory. Its `registration.json` contains the question, budget and stop rule. Its `checksums.json` identifies all recorded files. Some earlier phases use the separate `docs/measurements/` hierarchy and are described in [the phase review](../docs/PROJECT_REVIEW.md).

## Reproduce native tool comparisons

Clone the repository on Apple Silicon and run `.venv/bin/python -m loopqwen download` for the pinned Base. The typed, quantized study also requires the revision-bound official Qwen instruction sibling recorded in its registration. Begin a new bounded research window with `.venv/bin/python scripts/research_session.py`. Run one study at a time. Each root-level entry below references an immutable registration. The runner reads its settings and frozen data. Fresh local outputs go under `outputs/research/`; existing completed outputs are protected against overwrite.

```sh
.venv/bin/python scripts/tool_study.py tool-dialogue-loop-01
.venv/bin/python scripts/tool_study.py tool-dialogue-boundary-01
.venv/bin/python scripts/tool_study.py tool-dialogue-depth-01
.venv/bin/python scripts/tool_bfcl_study.py tool-bfcl-transfer-01
.venv/bin/python scripts/tool_study.py tool-dialogue-grounded-01
.venv/bin/python scripts/tool_study.py tool-dialogue-typed-01
```

The archived comparisons use their checked-in parents. For a read-only check, use `.venv/bin/python scripts/research_archive.py STUDY --verify`. Run `.venv/bin/python scripts/research_registry.py` to verify data, registrations and archive inventory identities. The data builders for the public tasks and training traces pin the Hugging Face revisions. [Data attribution and licenses](../research/licenses/TOOL-DATA-NOTICE.txt)

## Additional registered comparisons

The list below gives the normal native entry point for each family. Pass the desired immutable study ID as its argument. Before re-running a relative-path example, change to the repository root.

| Runner | Registered workload |
| --- | --- |
| `scripts/contrast_study.py` | Completed free answers, both members of human question edits and an identical-prefix readout |
| `scripts/tool_prefix_study.py` | Balanced repeated measurements with original or cached prefill |
| `scripts/tool_bfcl_study.py` | Independent single- and multi-schema publisher questions |
| `scripts/tool_study.py` | Single-pass tool attribution, instruction-parent control and caller-executed tools |
| `scripts/tool_release.py` | Combined offline build, field callbacks, exact language streams and physical 131K swaps |
| `-m loopqwen.research.balanced_register` | Equal-site-count register graph versus full-history Base |

Use `scripts/research_archive.py STUDY` to seal a completed local run in a new checkout. To verify existing committed evidence use `--verify`. After all measured jobs in a window are archived, run `scripts/research_registry.py --update` and `scripts/research_session.py --close`. All script paths in this table are arguments to the repository's Python interpreter.

Underlying human questions and verified partitions are in `research/data/`. The reproducible material builders pin publisher revisions and SHA-256 hashes. Rebuilding the BoolQ subsets needs PyArrow; numerical inference and stored-result verification do not depend on PyArrow. [BoolQ attribution](../research/licenses/BOOLQ-NOTICE.txt) · [Tool data attribution](../research/licenses/TOOL-DATA-NOTICE.txt)
