# Evidence selection on original requests

The short comparison uses 24 HotpotQA validation comparisons and 24 TAT-QA development reports. Every HotpotQA request includes the ten published passages and the original human question. Every finance request includes the original table, paragraphs and question. Both models see the same prompt token IDs and calculator declaration. The 24 reasoning cases have 240 distinct article titles, disjoint from all 3,007 titles in the new reasoning training subset. We checked all 1,800 new training openings for shared 16-word spans with the reserved requests. The tests do not establish independence from public pretraining or establish issuer identity. [Original HotpotQA project and CC BY-SA 4.0 attribution](https://hotpotqa.github.io/) · [TAT-QA project and paper](https://nextplusplus.github.io/TAT-QA/) · [Exact source revisions and checksums](inventory.json)

Training uses 320 original HotpotQA questions and the authors' marked supporting sentences. Each of 160 financial reports provides two different questions with different answers and source cells. Arithmetic targets preserve the original derivations, and return targets include the published scale. The 324 CoSQL questions come from 60 training schemas. For each, an original expert SQL query ran against the original file. Paired users can receive identical results that are relevant to only one of their requests. [Original CoSQL authors and data](https://yale-lily.github.io/cosql) · [Training queries, source members and verified returns](../native-tool-histories/training.json)

With the repository's locked data extra and the earlier pinned source caches installed, these commands retrieve the two public QA datasets, reproduce their selection, rebuild the SQL histories and check the actual token arrays.

```sh
.venv/bin/python -m scripts.content_read_material --download
.venv/bin/python -m scripts.native_history_sources
.venv/bin/python -m scripts.register_content_read
.venv/bin/python -m scripts.verify_content_material
```

The source check reconstructs all 688 distinct QA requests from the publisher records, checks the original support sentences and runs every financial derivation. The two sets of SQL targets and 192 protected language tasks account for the other 840 training turns. [Source reconstruction](../../../docs/measurements/content-read-sources.json) · [Paired comparison and stop rules](../../diagnostics/content-read-direct.json)
