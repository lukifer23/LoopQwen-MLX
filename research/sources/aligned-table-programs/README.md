# Human SQL on independently altered training tables

Shi, Zhao, Boyd-Graber, Daumé and Lee enriched WikiTableQuestions with 11,276 manually written SQL programs and lexical alignments. We joined their annotations to the 256 questions in our changed-table **training** partition using original question and table identifiers. We ran the same translated statement through the deployed read-only SQLite provider on each complete original and altered table. The evaluation partition contributed no queries, executions or answers to this audit. The SQUALL source and these derived annotations carry CC BY-SA 4.0. [Peer-reviewed paper](https://aclanthology.org/2020.findings-emnlp.167/) · [Pinned data and attribution](https://github.com/tzshi/squall/tree/ee6dda91eacafd5ee800eb3fdc5d3a6575dc47e7) · [License](LICENSE) · [Changed-table source](../changed-tables/README.md)

Of 234 matching annotations, 165 could be translated into the raw-file schema with the implemented type conversions. Both human gold denotations matched for 149 questions and 148 distinct SQL strings. Four matched only the original, four only the altered, and eight neither. Of the accepted programs, 111 use a WHERE clause, 50 a count, 43 ordering and eight grouping. These counts overlap. The 149 verified pairs supply reference programs for a possible training intervention. No learned model produced these results.

The compiler retains the annotated operators, literals and order of operations. It resolves each source column against a unique normalized original header, preserves original text results and uses case-insensitive text comparison. It represents the source row identifier with an unshadowed SQLite row number. A projected numeric column must contain valid numbers or recognized missing values in both versions. Calculations use the original raw rows. Ambiguous headers, specialized date or list projections, and nonnumeric cells are excluded. Two correct denotations provide stronger evidence than one and still allow accidental agreement.

For the question about athletes outside Cyprus, the unchanged count returned 13 on the original and 12 on the altered table. For the question about the first listed name, the original annotated first-row query produced Janelle Atkinson on the original table and a missing-name marker in the modified table. The source annotators selected Julia Stowers on the modified table. The latter program did not qualify. [All attempted programs, observed rows and labels](execution.json) · [Counts and exclusions](qualification.json)

From the project root, run the following command to rebuild the exact source join and both-world SQL executions. The script verifies the pinned upstream annotation bytes before it processes any training row.

```sh
.venv/bin/python -m scripts.squall_table_programs
```
