"""Create seeded, referentially valid subsets of original local-only test databases and execute human SQL."""

import argparse
import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read
from scripts.dialogue_database_sources import CACHE, SHA
from scripts.provider_recovery_field import PANEL
from scripts.provider_field_denotations import outcomes
from scripts.table_policy_material import persist
from loopqwen.evaluation.mechanism import unpack

LOCAL = PANEL.with_name("withheld-training-split-requests.json.gz")
PLAN = PANEL.with_name("field-subset-plan.json")
OUTPUT = PANEL.with_name("field-subset-denotations.json")
SEED = 472811
PERCENT = (97, 93, 87, 81, 74, 67, 60, 53)


def quoted(name):
    return '"' + name.replace('"', '""') + '"'


def variant(path, db, i):
    with closing(sqlite3.connect(path)) as con, con:
        # A final foreign_key_check verifies the completed subset. Cascading during
        # initial thinning would make the result depend on table iteration order.
        con.execute("PRAGMA foreign_keys=OFF")
        tables = sorted(
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        )
        failures = list(con.execute("PRAGMA foreign_key_check"))
        before = {
            t: con.execute(f"SELECT COUNT(*) FROM {quoted(t)}").fetchone()[0]
            for t in tables
        }
        threshold = (1 << 64) * PERCENT[i] // 100
        for t in tables:
            key = hashlib.sha256(f"{SEED}/{db}/{i}/{t}/".encode())

            def keep(idx, seed=key):
                h = seed.copy()
                h.update(str(idx).encode())
                return int.from_bytes(h.digest()[:8], "big") < threshold

            con.create_function("keep_row", 1, keep, deterministic=True)
            con.execute(f"DELETE FROM {quoted(t)} WHERE NOT keep_row(rowid)")
        while True:
            bad = list(con.execute("PRAGMA foreign_key_check"))
            if not bad:
                break
            removed = 0
            for t, idx in sorted({(t, idx) for t, idx, _, _ in bad}):
                assert idx is not None
                removed += con.execute(
                    f"DELETE FROM {quoted(t)} WHERE rowid=?", (idx,)
                ).rowcount
            assert removed
        assert con.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        after = {
            t: con.execute(f"SELECT COUNT(*) FROM {quoted(t)}").fetchone()[0]
            for t in tables
        }
    return {"before": before, "after": after, "initial_fk_violations": len(failures)}


def freeze():
    persist(
        PLAN,
        {
            "local": digest(LOCAL),
            "original_archive": SHA,
            "seed": SEED,
            "keep_percent": list(PERCENT),
            "method": "Across every local-only held schema generate all eight fixed SHA-256 rowid subsets from independent original copies. Apply a deterministic threshold per table and row. Recursively remove child rows that violate declared foreign keys and require an empty foreign-key audit and SQLite integrity check. Retain empty answers, ties and all model-independent variants. Record every original violation. These are project-generated alternate databases, separate from published distilled databases; expert results are computed from both human SQL releases. They do not supply independent human labels.",
        },
    )


def build():
    assert read(PLAN)["local"] == digest(LOCAL) and read(PLAN)["keep_percent"] == list(
        PERCENT
    )
    assert digest(CACHE) == SHA
    frozen = unpack(LOCAL)
    oracle = read(PANEL.with_name("field-denotations.json"))["panels"]["train"][
        "by_database"
    ]
    records = {}
    with ZipFile(CACHE) as z, TemporaryDirectory() as tmp:
        for db, d in sorted(oracle.items()):
            pair = [
                r
                for r in frozen["rows"]
                if r["source"] == db and r["condition"] == "unassisted"
            ]
            assert len(pair) == 2
            records[db] = []
            for i in range(len(PERCENT)):
                path = Path(tmp) / f"{db}-{i}.sqlite"
                path.write_bytes(z.read(d["original"]["member"]))
                assert digest(path) == d["original"]["sha256"]
                stats = variant(path, db, i)
                answers = outcomes(path, pair, d["official_sql"])
                records[db].append(
                    {
                        "index": i,
                        "sha256": digest(path),
                        "counts": stats,
                        "denotations": answers,
                    }
                )
    rows = [r for r in frozen["rows"] if r["condition"] == "unassisted"]
    out = {
        "plan": digest(PLAN),
        "sqlite": sqlite3.sqlite_version,
        "databases": records,
        "instances": sum(map(len, records.values())),
        "changed_questions": sum(
            any(
                x["denotations"][r["id"]] != [[r["gold"]]] for x in records[r["source"]]
            )
            for r in rows
        ),
        "different_partner_results": sum(
            len({fingerprint(v) for v in x["denotations"].values()}) == 2
            for xs in records.values()
            for x in xs
        ),
    }
    persist(OUTPUT, out)
    print({k: v for k, v in out.items() if k != "databases"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "build"))
    if parser.parse_args().mode == "freeze":
        freeze()
    else:
        build()
