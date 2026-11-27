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
