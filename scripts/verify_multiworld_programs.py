"""Execute every enrolled human program independently and audit trivial denotations."""

import sqlite3
from collections import Counter
from contextlib import closing

from loopqwen.evaluation.mechanism import unpack
from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save
from scripts.changed_table_sources import norm
from scripts.multiworld_program_material import OUT


def scalar(rows, statement):
    with closing(sqlite3.connect(":memory:")) as db:
        headers = ", ".join('"' + h.replace('"', '""') + '" TEXT' for h in rows[0])
        db.execute("CREATE TABLE records (" + headers + ")")
        db.executemany(
            "INSERT INTO records VALUES (" + ",".join("?" for _ in rows[0]) + ")",
            rows[1:],
        )
        db.execute("PRAGMA query_only=ON")
        cells = db.execute(statement).fetchall()
        assert len(cells) == 1 and len(cells[0]) == 1
        return cells[0][0]


def run():
    source = OUT / "training.json.gz"
    receipt = read(OUT / "qualification.json")
    assert digest(source) == receipt["training_sha256"]
    rows = unpack(source)
    constant_four = row_count = whole_count = constant_ceiling = 0
    source_counts = Counter()
    world_counts = Counter()
    records = []
    for w in rows:
        answers = Counter()
        count_hits = 0
        for v in w["versions"]:
            val = scalar(v["rows"], w["statement"])
            assert val == v["observed_scalar"] and norm(str(val)) == norm(
                str(v["answer"])
            )
            assert len(v["votes"]) >= 3 and all(
                norm(x) == norm(v["answer"]) for x in v["votes"]
            )
            ans = norm(v["answer"])
            answers[ans] += 1
            constant_four += ans == "4"
            count_hits += ans == str(len(v["rows"]) - 1)
        assert len(answers) >= 3
        constant_ceiling += max(answers.values())
        whole_count += count_hits == len(w["versions"])
        row_count += count_hits
        source_counts[len(answers)] += 1
        world_counts[len(w["versions"])] += 1
        records.append(
            [
                w["id"],
                len(w["versions"]),
                len(answers),
                count_hits,
                max(answers.values()),
            ]
        )
    assert sum(k * v for k, v in world_counts.items()) == receipt["annotated_worlds"]
    assert len(rows) == receipt["questions"]
    result = {
        "training": digest(source),
        "questions": len(rows),
        "worlds": receipt["annotated_worlds"],
        "distinct_answers_per_question": dict(sorted(source_counts.items())),
        "constant_four_correct_worlds": constant_four,
        "constant_four_correct_whole_questions": 0,
        "oracle_question_specific_constant_correct_worlds": constant_ceiling,
        "oracle_question_specific_constant_correct_whole_questions": 0,
        "table_row_count_correct_worlds": row_count,
        "table_row_count_correct_whole_questions": whole_count,
        "all_queries_all_denotations_verified": True,
        "item_columns": [
            "id",
            "worlds",
            "distinct_gold",
            "row_count_hits",
            "upper_bound_using_one_answer",
        ],
        "item_fingerprint": fingerprint(records),
        "items": records,
    }
    save(OUT / "independent-replay.json", result)
    print({k: v for k, v in result.items() if k != "items"})


if __name__ == "__main__":
    run()
