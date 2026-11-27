# Linked tables and original human questions

**Source qualification stopped before held evaluation.** One complete held article also appears as a training distractor under another Wikipedia page name. The URL-level check missed the Milwaukee redirect. The project retained all 64 original training rollouts, closed the study before optimizer updates and preserved the 24 held questions. [Normalized text audit](../../../docs/measurements/linked-source-overlap.json) · [Interruption record](../../diagnostics/linked-rollin-source-interruption.json)

This source cohort contains 128 publisher training questions and 24 publisher development questions from HybridQA. Each question retains its original free-text answer, complete Wikipedia table and every linked passage in the source repository. The source cohort itself has no model score. [Chen et al., Findings of EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.91/) · [Question release](https://github.com/wenhuchen/HybridQA/tree/db22fda8c5951438fade3c69d75b350335ba93b3) · [Table and passage release](https://github.com/wenhuchen/WikiTables-WithLinks/tree/dc066e1a6d5281511d8b73a6107d5ad2824cc2b2)

The original question and answer were collected from people. The published answer-node candidates came from the authors’ string matcher. Enrollment independently requires a single matching article, an actual hyperlink to it from its stated table cell, and a different cell in the same row whose value occurs in the question and identifies a unique row in its column. The target linked title is absent from the question. The answer is absent from the question and all table cells. The cue requirement selects relatively explicit joins. It does not establish that the question requires multiple steps for a model with prior world knowledge.

The split is fixed by hashing original identities. Each case uses a distinct source table, question and target article. No target article from either split occurs anywhere in the other split, even as a distractor. Forty-three unrelated article titles appear in both sets. The source files preserve the authors’ tokenization, punctuation and spelling. Each sample includes the original Wikipedia source URL and linked page identifiers. Attribution for the encyclopedic source material belongs to the respective Wikipedia contributors. The HybridQA repository includes the authors’ MIT notice. [Original notice](https://github.com/wenhuchen/HybridQA/blob/db22fda8c5951438fade3c69d75b350335ba93b3/LICENSE) · [Wikimedia reuse terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)

From the checkout, `.venv/bin/python -m scripts.linked_table_archive` replays every eligibility check on the preserved full source. To repeat selection from the original population, run `.venv/bin/python -m scripts.linked_table_sources --acquire`, then `.venv/bin/python -m scripts.linked_table_archive --freeze`. The downloader pins both repositories and verifies the Git blob identity of each table and passage bundle and the SHA-256 of both full question releases. [Audit receipt](../../../docs/measurements/linked-table-source-audit.json)

## Exercise the corpus with actual files

The renderer provides a full record table and an article table in CSV. Inline mode displays the full original table, including every hyperlink. File mode makes that table available through the same native lookup tool as the article file. The lookup oracle calls the unmodified tool on disk, follows the actual link and checks the complete article return. Across all 304 presentations, it executed 456 file operations. The model will have to choose its own arguments. [Prompt and callback hashes](../../../docs/measurements/linked-table-callbacks.json)

```sh
.venv/bin/python -m scripts.linked_table_material
.venv/bin/python -m scripts.linked_table_material --example 3c2db0e43d519ca0 --directory /tmp/linked-table-example
.venv/bin/loopqwen ask --checkpoint exports/tool-language-packed-05 \
  --tool-format qwen --keyed-csv /tmp/linked-table-example/records.csv \
  --keyed-csv /tmp/linked-table-example/articles.csv \
  --prompt-file /tmp/linked-table-example/file-question.txt \
  --max-tokens 512 --max-turns 3 --trace --compare-base
```

For the inline question, pass only `articles.csv` and use `inline-question.txt`. Both exported prompts preserve the exact text used to compute the frozen token hashes.
