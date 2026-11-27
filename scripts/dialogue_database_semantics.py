"""Compare an actually emitted SELECT with the original expert query on alternate databases."""

from collections import Counter
import re
import sqlite3

from loopqwen.io import digest, read, save
from loopqwen.relational_tools import select
from scripts.dialogue_database_sources import DEST
from scripts.dialogue_database_suite import FILES, MANIFEST


def compare(statement, gold_query, schema, turn, instances):
    matched = []
    # Each enrolled expert query selects one column. Ignore row order unless the
    # reference explicitly orders its rows. All multiplicities count.
    ordered = re.search(r"\border\s+by\b", gold_query, re.I) is not None
    for entry in instances:
        expected = entry["results"][turn]
        path = FILES / schema / entry["member"].rsplit("/", 1)[1]
        try:
            obs = select(path, statement, max_rows=max(1, len(expected)))
            values = obs["rows"]
            if obs["truncated"] or len(obs["columns"]) != 1:
                equal = False
            elif ordered:
                equal = expected == values
            else:
                equal = Counter(x[0] for x in expected) == Counter(x[0] for x in values)
        except (sqlite3.Error, TypeError, ValueError):
            equal = False
        matched.append(equal)
    return {"passed": all(matched), "matches": sum(matched), "instances": len(matched)}


def verify():
    manifest = read(MANIFEST)
    cases = []
    for db in read(DEST / "originals.json")["cohorts"]["dev"]:
        name = db["database"]
        inst = manifest["by_schema"][name]["databases"]
        for i, q in enumerate(db["dialogues"][0]["sequence"]):
            gold = compare(q["query"], q["query"], name, i, inst)
            # Generate the strongest possible constant-answer lookup, including correct quoting.
            with sqlite3.connect(":memory:") as conn:
                const = (
                    "SELECT "
                    + conn.execute("SELECT quote(?)", (q["returned"],)).fetchone()[0]
                )
            constant = compare(const, q["query"], name, i, inst)
            assert gold["passed"] and not constant["passed"]
            cases.append(
                {"database": name, "turn": i, "expert": gold, "constant": constant}
            )
    receipt = {
        "source": digest(MANIFEST),
        "expert_pass": sum(c["expert"]["passed"] for c in cases),
        "constant_pass": sum(c["constant"]["passed"] for c in cases),
        "cases": cases,
    }
    dest = DEST / "semantic-verification.json"
    if dest.is_file():
        assert read(dest) == receipt
    else:
        save(dest, receipt)
    print(
        receipt["expert_pass"],
        "expert and",
        receipt["constant_pass"],
        "constant queries pass",
        flush=True,
    )


if __name__ == "__main__":
    verify()
