"""Validate original human database conversations against the released SQLite files."""

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import urllib.request
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

from loopqwen.identity import fingerprint
from loopqwen.io import digest, read, save

SOURCE = "https://drive.usercontent.google.com/download?id=1Y3ydpFiQQ3FC0bzdfy3groV95O_f1nXF&export=download&confirm=t"
SHA = "0d6fc424ea914fcbb6728055b847685acc854e16c034b7831c4937bfe18d53bf"
CACHE = Path("data/research_external/cosql/cosql.zip")
DEST = Path("research/sources/database-dialogues")


def collect(fetch=False):
    if not CACHE.is_file() and fetch:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".partial")
        with (
            urllib.request.urlopen(SOURCE, timeout=90) as response,
            tmp.open("wb") as f,
        ):
            shutil.copyfileobj(response, f)
        assert digest(tmp) == SHA
        tmp.replace(CACHE)
    assert digest(CACHE) == SHA
    with ZipFile(CACHE) as z:
        base = "cosql_dataset/"

        def data(path):
            return json.loads(z.read(base + path + ".json"))

        split = {
            side: {r["database_id"] for r in data("sql_state_tracking/cosql_" + side)}
            for side in ("train", "dev")
        }
        assert not split["train"].intersection(split["dev"])
        raw = data("cosql_all_info_dialogs")
        by_db = defaultdict(list)
        for key, d in raw.items():
            if d.get("disabled") or d.get("reviewstate") != "Ready":
                continue
            ts = d["turns"]
            for i in range(len(ts) - 5):
                seq = []
                for q, e, r in (ts[i : i + 3], ts[i + 3 : i + 6]):
                    if not (
                        q.get("isUser")
                        and q.get("label") == ["INFORM_SQL"]
                        and e.get("isSql")
                        and r.get("label") == ["CONFIRM_SQL"]
                    ):
                        break
                    rs = e.get("sql_result")
                    sql = e.get("rawSql", "").strip()
                    if not (
                        isinstance(rs, list)
                        and len(rs) == 2
                        and len(rs[1]) == 1
                        and rs[1][0] not in (None, "")
                        and sql.lower().startswith("select")
                    ):
                        break
                    value = rs[1][0]
                    if not re.search(
                        r"(?<!\w)" + re.escape(str(value)) + r"(?!\w)", r["text"], re.I
                    ):
                        break
                    seq.append(
                        {
                            "question": q["text"],
                            "query": sql,
                            "returned": value,
                            "column": rs[0][0],
                            "human_answer": r["text"],
                        }
                    )
                if len(seq) != 2 or seq[0]["returned"] == seq[1]["returned"]:
                    continue
                db = d["db_id"]
                prior = [
                    {
                        "role": "user" if t.get("isUser") else "assistant",
                        "content": t["text"],
                    }
                    for t in ts[:i]
                    if not t.get("isSql") and t.get("text")
                ]
                by_db[db].append(
                    {"source": key, "turn": i, "prior": prior, "sequence": seq}
                )
        enrolled = {}
        for part, n in (("train", 64), ("dev", 12)):
            result = []
            for db in sorted(split[part], key=lambda x: fingerprint([598102, part, x])):
                members = by_db.get(db, [])
                if not members:
                    continue
                filename = base + "database/" + db + "/" + db + ".sqlite"
                contents = z.read(filename)
                with closing(sqlite3.connect(":memory:")) as connection:
                    connection.deserialize(contents)
                    connection.execute("PRAGMA query_only=ON")
                    selected = []
                    for pair in sorted(
                        members,
                        key=lambda w: fingerprint([598102, w["source"], w["turn"]]),
                    ):
                        if any(x["source"] == pair["source"] for x in selected):
                            continue
                        try:
                            actual = [
                                connection.execute(y["query"]).fetchall()
                                for y in pair["sequence"]
                            ]
                        except sqlite3.DatabaseError:
                            continue
                        if any(
                            val != [(y["returned"],)]
                            for val, y in zip(actual, pair["sequence"], strict=True)
                        ):
                            continue
                        selected.append(pair)
                        if len(selected) == (2 if part == "train" else 1):
                            break
                if selected:
                    result.append(
                        {
                            "database": db,
                            "member": filename,
                            "sqlite_sha256": hashlib.sha256(contents).hexdigest(),
                            "dialogues": selected,
                        }
                    )
                if len(result) == n:
                    break
            enrolled[part] = result
        assert len(enrolled["train"]) == 64 and len(enrolled["dev"]) >= 10
        assert not {w["database"] for w in enrolled["train"]}.intersection(
            w["database"] for w in enrolled["dev"]
        )
    DEST.mkdir(parents=True, exist_ok=True)
    payload = {"archive_sha256": SHA, "official_source": SOURCE, "cohorts": enrolled}
    path = DEST / "originals.json"
    if path.exists():
        assert read(path) == payload
    else:
        save(path, payload)
    print(
        {
            k: {
                "databases": len(rs),
                "original_pairs": sum(len(w["dialogues"]) for w in rs),
            }
            for k, rs in enrolled.items()
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acquire", action="store_true")
    collect(p.parse_args().acquire)
