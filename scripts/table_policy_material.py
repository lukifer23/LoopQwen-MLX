"""Freeze disjoint warm-up turns, paired environments and genuine evaluation requests."""

from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen import qwen_tools
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.relational_tools import tools
from loopqwen.research.cross_view_study import tokenizer
from loopqwen.research.native_function_data import token_pair
from loopqwen.research.table_worlds import (
    FILENAME,
    SOURCE,
    locations,
    prepare_tables,
    present,
    prompt,
    state,
)
from loopqwen.tool_session import compact

NAME = "table-policy-gradient-01"
PANEL = Path("research/data/table-policy-panel.json")
TRAIN = Path("research/data/table-policy-training.json")
AUDIT = Path("research/sources/changed-tables/qualification.json")


def persist(path, obj):
    if path.exists():
        assert read(path) == obj
    else:
        save(path, obj)


def build():
    sources = read(SOURCE)["cohorts"]
    integrated = read("research/data/integrated-assistance-training.json")["examples"]
    tok = tokenizer({"checkpoint": "exports/tool-language-packed-05"})
    supervised = []
    table_rows = []
    read_count = 0
    identical_headers = 0
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for split, cohort in sources.items():
            prepare_tables(folder, cohort)
            for w in cohort:
                msgs, defs = state(w, folder)
                identical_headers += 1
                for j, (version, path) in enumerate(
                    zip(w["versions"], locations(folder, w), strict=True)
                ):
                    _schema, query = tools([path])
                    args = {
                        "database_name": FILENAME,
                        "statement": "SELECT * FROM records ORDER BY rowid",
                    }
                    obs = query.callback(**args)
                    assert (
                        not obs["truncated"]
                        and obs["columns"] == version["rows"][0]
                        and obs["rows"] == version["rows"][1:]
                    )
                    read_count += 1
                    assert present(
                        version["answer"], version["answer"]
                    ) and not present(w["versions"][1 - j]["answer"], version["answer"])
                    key = f"{w['id']}-{version['world']}"
                    if split == "dev":
                        text = prompt(w)
                        opening = tok.encode(
                            qwen_tools.render(
                                tok, [{"role": "user", "content": text}], defs
                            ),
                            add_special_tokens=False,
                        )
                        table_rows.append(
                            {
                                "id": key,
                                "family": "changed_table",
                                "cluster": w["id"],
                                "source": w["context"],
                                "world": version["world"],
                                "row_id": w["id"],
                                "questions": [text],
                                "tools": defs,
                                "initial_ids": opening,
                                "gold": version["answer"],
                            }
                        )
                    else:
                        if j == 0:
                            supervised.append(
                                {
                                    "id": key + "-schema",
                                    "group": "table_schema",
                                    "messages": msgs[:1],
                                    "tools": defs,
                                    "target": msgs[1],
                                }
                            )
                            supervised.append(
                                {
                                    "id": key + "-read",
                                    "group": "table_read",
                                    "messages": msgs,
                                    "tools": defs,
                                    "target": qwen_tools.assistant_message(
                                        [{"name": query.name, "arguments": args}]
                                    ),
                                }
                            )
                        after = [
                            *msgs,
                            qwen_tools.assistant_message(
                                [{"name": query.name, "arguments": args}]
                            ),
                            {"role": "tool", "content": compact(obs)},
                        ]
                        supervised.append(
                            {
                                "id": key + "-reply",
                                "group": "table_reply",
                                "tools": defs,
                                "messages": after,
                                "target": {
                                    "role": "assistant",
                                    "content": version["answer"],
                                },
                            }
                        )
    # Keep full, broad source supervision in every pass through the decks.
    for w in integrated:
        group = w["kind"] if w["group"] == "database" else w["group"]
        supervised.append(
            {k: w[k] for k in ("id", "messages", "tools", "target")} | {"group": group}
        )
    old = read("research/data/integrated-assistance-panel.json")
    table_rows.extend(old["rows"])
    table_rows.sort(key=lambda r: fingerprint([456318, r["id"]]))
    held = {tuple(w["initial_ids"]) for w in table_rows}
    held_questions = {q for w in table_rows for q in w["questions"]}
    train_ids = set()
    counts = Counter()
    sequence_lengths = []
    for w in supervised:
        p, y = token_pair(tok, w)
        assert tuple(p) not in held
        train_ids.add(fingerprint(p))
        counts[w["group"]] += 1
        sequence_lengths.append(len(p) + len(y))
    assert not any(prompt(w) in held_questions for w in sources["train"])
    by_source = defaultdict(list)
    for w in table_rows:
        if w["family"] == "changed_table":
            by_source[w["cluster"]].append(w)
    assert len(by_source) == 24 and all(
        len(p) == 2
        and p[0]["initial_ids"] == p[1]["initial_ids"]
        and p[0]["gold"] != p[1]["gold"]
        for p in by_source.values()
    )
    receipt = {
        "source": digest(SOURCE),
        "identical_schema_pairs": identical_headers,
        "verified_complete_reads": read_count,
        "frozen_views": len(supervised),
        "unique_training_openings": len(train_ids),
        "groups": dict(sorted(counts.items())),
        "max_training_tokens": max(sequence_lengths),
        "paired_test_questions": len(by_source),
        "requests": len(table_rows),
        "overlapping_training_openings": 0,
    }
    persist(AUDIT, receipt)
    persist(
        TRAIN,
        {
            "examples": supervised,
            "source": digest(SOURCE),
            "qualification": digest(AUDIT),
        },
    )
    persist(
        PANEL,
        {
            "rows": table_rows,
            "documents": old["documents"],
            "qualification": digest(AUDIT),
        },
    )
    print(receipt, flush=True)


if __name__ == "__main__":
    build()
