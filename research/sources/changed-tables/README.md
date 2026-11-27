# Answering the same human question after a table changes

Pasupat and Liang collected original user questions about Wikipedia tables and later asked annotators to answer the same questions on fictitious versions. The source includes people’s answers to the changed tables. A memorized original answer can therefore be checked against an independently labeled alternative. The altered tables are research interventions. [WikiTableQuestions](https://aclanthology.org/P15-1142/) · [Counterfactual annotation paper](https://aclanthology.org/P16-1003/) · [Original tables and CC BY-SA 4.0 license](https://github.com/ppasupat/WikiTableQuestions/tree/7d455a5a707b96341ef72aff9428749d443d8aa9) · [Authors’ changed tables and answers](https://downloads.cs.stanford.edu/nlp/software/sempre/wikitable/dpd/)

We retain pairs with at least three agreeing participant responses on each version and different single answers across versions. The original answer must also agree with the earlier WikiTableQuestions gold. Eligibility uses the authors’ first table-disjoint random split, published table sizes and matching original column names. A fixed hash selected one question per original table, giving 256 training pairs and 24 development pairs on 280 different tables. The two inputs in a pair use the same user question, database name, schema and initial prompt tokens. The rows behind the mounted table change. Across 560 complete tables, the enrollment preserves 8,095 data rows and 1,696 source ratings. Worker and assignment identifiers are omitted. The selection favors small rectangular tables, single-valued questions and clear annotator agreement.

Each enrolled table is mounted as an actual SQLite table named `records`. Column names, row order and literal cell text are preserved. The ordinary read-only tool returns the full source exactly on 560/560 calls. Seven of eight manually written training-only queries recovered both annotated answers when executed independently on the two physical files. Zero of eight constant queries reading a real table recovered both. All eight results, including the miss, appear in [the reward check](reward-verification.json). These queries are absent from student training and serving. [Enrollment](originals.json) · [Source and prompt qualification](qualification.json)

Run these commands from the checkout to reconstruct the receipts.

```sh
.venv/bin/python -m scripts.changed_table_sources --acquire
.venv/bin/python -m scripts.table_policy_material
.venv/bin/python -m scripts.table_policy_rewards
```

The missed query asked for the first name listed and returned the first row. In the altered table the first two Name fields contain a dash. The first actual name appears in the third row. All three annotators chose that name. The diagnostic preserves this miss, which illustrates why even a plausible executable query needs to be tested on the changed source.

The Base-rooted fit packs the fixed 4-bit prefix and the four original BF16 suffix layers once. A separate cold decoder must reproduce all vocabulary logits on the registered real training opening from the packed weights. The learned policy records these bytes’ digest together with the original Base and config digests. The CLI can load the verified pack beside the policy or from the research export. With only the original Base release it reconstructs the quantized graph from the registered full-precision tensors. Both evaluation arms include actual cold construction in their timing.

An independent text audit compared each of the 48 held versions with all 512 training versions. No original held question repeated a normalized training question, and no complete table hash repeated. The highest question four-word-set Jaccard similarity was 0.143. Fourteen held versions had exactly the same ordered column names as a training table, and the maximum case-normalized, multiplicity-weighted cell overlap was 0.661. These similarities remain in the evaluation. [Per-table results](../../../paper/changed-table-source-overlap.csv) · [Rebuild the overlap audit](../../../scripts/changed_table_overlap.py)
