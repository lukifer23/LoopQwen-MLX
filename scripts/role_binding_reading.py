"""Render distinct table answers without arm labels and recount independently read conclusions."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "research/data/role-binding-panel.json"
SOURCE = ROOT / "docs/measurements/role-binding-reading-decisions.json"
OUTPUT = ROOT / "paper/role-binding-reading.csv"


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split()).strip(
        ' *"“”.!'
    )


def records(folder):
    panel = json.loads(PANEL.read_bytes())["rows"]
    arms = {
        s: json.loads(gzip.decompress((folder / (s + ".json.gz")).read_bytes()))
        for s in ("candidate", "control")
    }
    all_rows = []
    for i, w in enumerate(panel):
        if w["family"] != "table":
            continue
        for s, arm in arms.items():
            assert arm[i]["id"] == w["id"]
            q = arm[i]["requests"][0]
            tr = q["trajectory"]
            answer = tr["answer"] or ""
            key = hashlib.sha256(
                json.dumps(
                    [w["id"], answer], ensure_ascii=False, separators=(",", ":")
                ).encode()
            ).hexdigest()
            automatic = (
                False
                if tr["status"] != "answered" or not answer
                else (True if normalize(answer) == normalize(w["gold"]) else None)
            )
            all_rows.append(
                {
                    "id": w["id"],
                    "table": w["cluster"],
                    "question": w["questions"][0],
                    "gold": w["gold"],
                    "visible": answer,
                    "key": key,
                    "automatic": automatic,
                    "side": s,
                    "grounding": q["score"]["observed"],
                    "primary": q["score"]["success"],
                }
            )
    return all_rows


def main(blind):
    folder = (
        ROOT / ("outputs/research" if blind else "research/results") / "role-binding-01"
    )
    rows = records(folder)
    pending = {
        r["key"]: {k: r[k] for k in ("key", "question", "gold", "visible")}
        for r in rows
        if r["automatic"] is None
    }
    if blind:
        for key, r in sorted(pending.items()):
            print(json.dumps(r, ensure_ascii=False))
        return
    votes = json.loads(SOURCE.read_bytes())["reviewed"]
    assert set(votes) == set(pending)
    stats = {}
    for r in rows:
        final = r.pop("automatic")
        r["method"] = "full reply" if final is None else "exact reply or unfinished"
        r["reading"] = (
            votes[r["key"]] if final is None else ("correct" if final else "incorrect")
        )
        assert r["reading"] in {"correct", "incorrect", "ambiguous"}
        r["grounded_reading"] = r["grounding"] and r["reading"] == "correct"
    for side in ("candidate", "control"):
        here = [r for r in rows if r["side"] == side]
        tables = {r["table"] for r in here}
        stats[side] = {
            "correct_visible": sum(r["reading"] == "correct" for r in here),
            "grounded": sum(r["grounded_reading"] for r in here),
            "complete_tables": sum(
                all(r["grounded_reading"] for r in here if r["table"] == t)
                for t in tables
            ),
        }
    for row in rows:
        row["visible_json"] = json.dumps(row.pop("visible"), ensure_ascii=False)
    f = io.StringIO(newline="")
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    OUTPUT.write_bytes(f.getvalue().encode())
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blind", action="store_true")
    main(p.parse_args().blind)
