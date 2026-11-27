"""Replay every enrolled CoSQL query through the actual caller-facing SQLite providers."""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from loopqwen.io import digest, read, save
from loopqwen.relational_tools import tools
from scripts.dialogue_database_sources import CACHE, DEST, SHA


def audit():
    assert digest(CACHE) == SHA
    cohort = read(DEST / "originals.json")
    all_rows = []
    with ZipFile(CACHE) as z, TemporaryDirectory() as tmp:
        work = Path(tmp)
        for split, dbs in cohort["cohorts"].items():
            for db in dbs:
                path = work / (db["database"] + ".sqlite")
                path.write_bytes(z.read(db["member"]))
                assert digest(path) == db["sqlite_sha256"]
                schema, query = tools([path])
                tables = schema.callback(database_name=path.name)["tables"]
                n = joins = 0
                for dialogue in db["dialogues"]:
                    for turn in dialogue["sequence"]:
                        obs = query.callback(
                            database_name=path.name, statement=turn["query"]
                        )
                        assert (
                            obs["rows"] == [[turn["returned"]]] and not obs["truncated"]
                        )
                        assert (
                            obs["referenced_tables"]
                            and set(obs["referenced_tables"]) <= tables.keys()
                        )
                        assert len(obs["columns"]) == 1
                        n += 1
                        joins += len(obs["referenced_tables"]) > 1
                all_rows.append(
                    {
                        "split": split,
                        "database": db["database"],
                        "tables": len(tables),
                        "queries": n,
                        "multitable": joins,
                    }
                )
                path.unlink()
    result = {
        "source": digest(DEST / "originals.json"),
        "databases": len(all_rows),
        "queries": sum(r["queries"] for r in all_rows),
        "referencing_multiple_tables": sum(r["multitable"] for r in all_rows),
        "per_database": all_rows,
    }
    dest = Path("docs/measurements/database-dialogue-tools.json")
    if dest.exists():
        assert read(dest) == result
    else:
        save(dest, result)
    print(
        result["queries"],
        "original human turns;",
        result["referencing_multiple_tables"],
        "multi-table queries",
    )


if __name__ == "__main__":
    audit()
