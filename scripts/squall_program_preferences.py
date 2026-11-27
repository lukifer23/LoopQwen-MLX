"""Construct execution-certified, same-type incorrect alternatives to real human SQL programs."""

import re
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from loopqwen.research.table_worlds import (
    SOURCE,
    canonical,
    prepare_tables,
    query_results,
)
from scripts.squall_table_programs import DEST, LOCAL, NUMBER, SHA, translate

SEED = 194736
SUBSTITUTIONS = {
    "=": "!=",
    "!=": "=",
    "<": ">",
    ">": "<",
    "<=": ">=",
    ">=": "<=",
    "asc": "desc",
    "desc": "asc",
    "min": "max",
    "max": "min",
}


def proposals(record):
    tokens = record["sql"]
    for i, (kind, text, _) in enumerate(tokens):
        options = []
        if kind == "Keyword" and text in SUBSTITUTIONS:
            options.append(
                (
                    "direction"
                    if text in ("asc", "desc", "min", "max")
                    else "predicate",
                    SUBSTITUTIONS[text],
                )
            )
        if kind == "Column" and re.fullmatch(r"c[1-9][0-9]*(?:_number)?", text):
            tail = "_number" if text.endswith("_number") else ""
            options.extend(
                ("column", f"c{n}{tail}")
                for n in range(1, len(record["columns"]) + 1)
                if f"c{n}{tail}" != text
            )
        for op, alt in options:
            new = list(tokens)
            new[i] = [kind, alt, []]
            yield op, {**record, "sql": new}
        if (
            kind == "Keyword"
            and text == "limit"
            and any(t[1] == "order" for t in tokens[:i])
            and tokens[i - 1][1] not in ("asc", "desc")
        ):
            yield (
                "direction",
                {**record, "sql": [*tokens[:i], ["Keyword", "desc", []], *tokens[i:]]},
            )


def numeric(x):
    return bool(NUMBER.fullmatch(canonical(x)))


def run():
    source = read(DEST / "execution.json")
    assert digest(LOCAL) == SHA and source["source_dataset_sha256"] == SHA
    data = {r["nt"]: r for r in read(LOCAL)}
    train = {w["id"]: w for w in read(SOURCE)["cohorts"]["train"]}
    kept = []
    summary = Counter()
    with TemporaryDirectory() as tmp:
        parent = Path(tmp)
        prepare_tables(parent, list(train.values()))
        for positive in source["executed"]:
            if positive["status"] != "correct both":
                continue
            w = train[positive["id"]]
            record = data[w["id"]]
            positive_values = [v["answer"] for v in w["versions"]]
            found = []
            seen = set()
            for edit, other in proposals(record):
                try:
                    sql = translate(other, w)
                except ValueError:
                    continue
                if sql in seen or sql == positive["statement"]:
                    continue
                seen.add(sql)
                summary["tested"] += 1
                outcomes = query_results(sql, w, parent)
                if any(not r["used"] or r["correct"] for r in outcomes):
                    continue
                tables = [r["observation"]["rows"] for r in outcomes]
                if any(
                    len(rs) != 1 or len(rs[0]) != 1 or rs[0][0] is None for rs in tables
                ):
                    continue
                values = [rs[0][0] for rs in tables]
                if any(
                    numeric(x) != numeric(t)
                    for x, t in zip(values, positive_values, strict=True)
                ):
                    continue
                if any(not canonical(x) for x in values) or canonical(
                    values[0]
                ) == canonical(values[1]):
                    continue
                found.append((edit, sql, values))
            summary["verified_alternatives"] += len(found)
            if found:
                # Choose one alternative independently of any model. Preserve operator types when a source has several.
                choices = []
                for category in sorted({e for e, _, _ in found}):
                    e, sql, v = min(
                        (s for s in found if s[0] == category),
                        key=lambda s: fingerprint([SEED, w["id"], s[1]]),
                    )
                    choices.append({"edit": e, "sql": sql, "results": v})
                kept.append(
                    {
                        "id": w["id"],
                        "question": w["question"],
                        "original": positive["statement"],
                        "answers": positive_values,
                        "rejected": choices,
                    }
                )
    output = DEST / "preferences.json"
    save(output, kept)
    report = {
        "supervision": "Every positive is a published human SQL program, translated to the source schema, and returns the human scalar on both tables. Each rejected program differs by one operator, predicate or source column, executes on both, returns two distinct, nonempty single scalars of matching numeric/text type, and is wrong on both. One hash-selected alternative per available edit class per question is retained.",
        "source": digest(DEST / "execution.json"),
        "seed": SEED,
        "positive_questions": sum(
            r["status"] == "correct both" for r in source["executed"]
        ),
        "paired_questions": len(kept),
        **summary,
        "retained": sum(len(r["rejected"]) for r in kept),
        "categories": dict(Counter(v["edit"] for r in kept for v in r["rejected"])),
        "output": digest(output),
    }
    save(DEST / "preference-qualification.json", report)
    print(report)


if __name__ == "__main__":
    run()
