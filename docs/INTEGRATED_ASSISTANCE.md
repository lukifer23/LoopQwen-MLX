# Run the consolidated recurrent assistant

This policy repeats the last native transformer layer of the 0.8B instruction sibling on every token. Four independently learned parameter updates contribute to the shared projections. The same compiled tensors serve database queries, open conversation and ordinary text. The pinned BF16 Qwen3.5-0.8B-Base is the ordinary-depth control. The measured training exposure contains 258,469 selected target positions. [Study registration](../research/experiments/integrated-assistance-01.json) · [Training exposure](measurements/integrated-assistance-exposure.json)

The direct comparison delivered 56 grounded or source-scored answers on 208 objective requests versus 42 for ordinary Base. Across all 220 requests the models took 421.66 and 1,089.86 seconds. The loop gained 41 individual answers and lost 27 Base successes. On the new database questions it grounded 3/24 answers versus 1/24 and completed both questions on 0/12 databases in either arm. A project reviewer, masked to model identity, preferred the loop three times and Base once over 12 open questions, with eight ties. The loop produced four materially unsupported open replies versus two for Base. Its scores fell on all three human reasoning measures. These weights remain experimental. [Complete decision](../research/results/integrated-assistance-01/decision.json) · [Sealed review](measurements/integrated-assistance-blinded-ratings.json)

Use the pinned Apple Silicon environment and the qualified release 05 files. In the extracted add-on, make that checkpoint available as `release-05` and install the enclosed runtime wheel. Each example includes the complete original request and all its input files. The commands below run outside the source checkout.

```sh
uv pip install --offline --no-deps runtime/loopqwen-0.1.0-py3-none-any.whl
python -c 'import json, subprocess; subprocess.run(["loopqwen", *json.load(open("examples/database.json"))], check=True)'
python -c 'import json, subprocess; subprocess.run(["loopqwen", *json.load(open("examples/open.json"))], check=True)'
```

To ask a new question, supply your own SQLite file. The tool opens it read-only and exposes schema inspection and SELECT. The installed policy chooses the calls and arguments. The optional `--trace` records actual tokens, tool returns and attention histories.

```sh
loopqwen ask --checkpoint release-05 --session-policy policy --tool-format qwen \
  --relational-db sales.sqlite --prompt 'Which product has the highest total sales?' \
  --max-tokens 384 --max-turns 3 --compare-base --trace
```

In the checkout use `exports/tool-language-packed-05` for the checkpoint and `research/results/integrated-assistance-01` for the policy. The sealed evidence records every prompt ID, trained adapter, compiled matrix, native optimizer moment, held generation and callback. The compiled model requires release 05’s original quantized backbone. Full-context qualifications belong to their individually tested checkpoint digests.

```sh
.venv/bin/python -m scripts.research_archive integrated-assistance-01 --verify
.venv/bin/python -m scripts.paper_integrated_assistance
.venv/bin/python scripts/integrated_distribution.py
```

[Release 05 and offline dependencies](FIELD_KIT.md) · [Model comparison and graphs](../paper/integrated-assistance.pdf) · [Source and scorer verification](../research/sources/database-dialogues/README.md)
