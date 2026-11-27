"""Prove that query rewards depend on the observed changed contents on original training questions."""

from pathlib import Path
from tempfile import TemporaryDirectory

from loopqwen.io import digest, read, save
from loopqwen.research.table_worlds import SOURCE, prepare_tables, query_results

# These explanatory checks use training tables only. The statements never enter policy fitting.
EXAMPLES = {
    "nt-5572": 'SELECT "Name" FROM records ORDER BY rowid LIMIT 1',
    "nt-5453": 'SELECT "Area" FROM records ORDER BY rowid LIMIT 1',
    "nt-11884": 'SELECT "Act" FROM records ORDER BY CAST("# Albums released under Bad Boy" AS INTEGER) DESC, rowid LIMIT 1',
    "nt-9419": 'SELECT "Name" FROM records ORDER BY CAST("Years until mandatory retirement" AS INTEGER) DESC, rowid LIMIT 1',
    "nt-6758": 'SELECT "Nation" FROM records WHERE "Gold" = \'6\'',
    "nt-484": 'SELECT "Name" FROM records WHERE lower("Position") = \'farmer\'',
    "nt-9466": 'SELECT "Club" FROM records WHERE CAST("Wins" AS INTEGER) = 4',
    "nt-12470": 'SELECT "Nation" FROM records ORDER BY CAST("Gold" AS INTEGER) DESC, rowid LIMIT 1',
}


def audit():
    examples = [w for w in read(SOURCE)["cohorts"]["train"] if w["id"] in EXAMPLES]
    assert len(examples) == len(EXAMPLES)
    records = []
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        prepare_tables(folder, examples)
        for w in examples:
            sql = EXAMPLES[w["id"]]
            actual = query_results(sql, w, folder)
            untouched = w["versions"][0]["answer"].replace("'", "''")
            constant = query_results(f"SELECT '{untouched}'", w, folder)
            with_table = query_results(
                f"SELECT '{untouched}' FROM records LIMIT 1", w, folder
            )
            records.append(
                {
                    "id": w["id"],
                    "sql": sql,
                    "versions": [v["answer"] for v in w["versions"]],
                    "returned": [r.get("observation", {}).get("rows") for r in actual],
                    "correct": [r["correct"] for r in actual],
                    "constant": [r["correct"] for r in constant],
                    "constant_with_read": [r["correct"] for r in with_table],
                }
            )
            assert (
                not any(r["correct"] for r in constant)
                and sum(r["correct"] for r in with_table) <= 1
            )
    report = {
        "materials": digest(SOURCE),
        "examples": records,
        "both": sum(all(r["correct"]) for r in records),
        "both_constant": sum(all(r["constant_with_read"]) for r in records),
    }
    assert report["both"] >= 6
    target = SOURCE.with_name("reward-verification.json")
    if target.exists():
        assert read(target) == report
    else:
        save(target, report)
    print(
        report["both"],
        "paired decisions of",
        len(records),
        "from unmodified executions",
    )


if __name__ == "__main__":
    audit()
