# Original questions, tool results and human replies from CoSQL

The CoSQL authors collected conversations between a person requesting information and a SQL expert. Their public archive includes the original utterances, the expert’s queries, the returned rows, natural replies and SQLite databases. This enrollment retains genuine consecutive turns from 111 training dialogues on 64 databases and 12 held dialogues on 12 other databases. It uses the publisher’s train and development database partitions. [Yu et al., EMNLP 2019](https://aclanthology.org/D19-1204/) · [Official dataset and CC BY-SA 4.0 distribution](https://yale-lily.github.io/cosql)

Both selected questions must have an original INFORM_SQL label and an immediate recorded query followed by a CONFIRM_SQL reply. Each query must return a single nonempty value quoted in the human reply. The two values must differ. Only conversations marked Ready are eligible. The sampler orders databases and dialogues by fixed hashes, takes at most two distinct training dialogues per database and one held dialogue per database, and checks every original result by rerunning its query against the exact released SQLite bytes. It preserves the original preceding conversation. This selection favors concise answerable questions, so it cannot characterize arbitrary SQL requests or clarification behavior.

The export includes human questions and answers, the queries, returned values, source-dialogue IDs, original turn indices and database hashes. Worker and assignment identifiers are omitted. The untouched source is a 105,319,480-byte ZIP with SHA-256 `0d6fc424ea914fcbb6728055b847685acc854e16c034b7831c4937bfe18d53bf`. Run `.venv/bin/python -m scripts.dialogue_database_sources --acquire` to download and reconstruct the selection. Running the command again with the local ZIP verifies it offline. This is source qualification. No model score has been obtained on this material.

The command below exposes the tables, column types, primary keys and foreign keys in a caller-selected database. The assistant can issue a read-only SELECT and see the column labels, row values, truncation flag and tables SQLite actually read. Pass the flag again to expose a second database. Each query addresses one supplied database by filename.

```sh
.venv/bin/python -m loopqwen ask --checkpoint exports/tool-language-packed-05 \
  --relational-db /path/to/library.sqlite --prompt 'Which author has the most books in the catalog?' --trace
```

The original experts’ 246 SELECTs reproduced their recorded values through these callbacks on 76 databases. Seventy queries read multiple tables. The measured outcome establishes that the interface can execute the reference work. The model still has to select the tools, write a query and interpret its result. [Per-database verification](../../../docs/measurements/database-dialogue-tools.json)
